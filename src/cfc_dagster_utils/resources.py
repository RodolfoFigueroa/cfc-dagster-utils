import hashlib
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import sqlalchemy
from sqlalchemy.schema import DropTable

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.types import (
    PostgresIndex,
    PostgresIndexMethod,
    PostgresRelation,
    PostgresTableSpec,
    PostgresWriteMode,
)

POSTGRES_IDENTIFIER_MAX_LENGTH = 63

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )

try:
    import geoalchemy2  # noqa: F401
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="geoalchemy2",
        dependency_name="geoalchemy2",
        extra="postgres",
    )

try:
    import psycopg  # noqa: F401
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="psycopg",
        dependency_name="psycopg",
        extra="postgres",
    )

from pydantic import (  # noqa: E402 - optional dependency checks must run first.
    PrivateAttr,
)


class PostgresResource(dg.ConfigurableResource):
    """Manage PostgreSQL connections and transactional table publication."""

    host: str
    port: int
    user: str
    password: str
    database: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        """Create the SQLAlchemy engine used during Dagster execution."""
        url = sqlalchemy.URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        self._engine = sqlalchemy.create_engine(url)

    def teardown_after_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        """Dispose pooled database connections after Dagster execution."""
        self._engine.dispose()

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """Yield a connection without automatically committing a transaction."""
        with self._engine.connect() as conn:
            yield conn

    @contextmanager
    def begin(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """Yield a connection in a commit-or-rollback transaction."""
        with self._engine.begin() as conn:
            yield conn

    @staticmethod
    def relation_exists(
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
    ) -> bool:
        """Return whether a table or view exists for ``relation``."""
        return sqlalchemy.inspect(conn).has_table(
            relation.name,
            schema=relation.schema,
        )

    @staticmethod
    def _table(relation: PostgresRelation) -> sqlalchemy.Table:
        return sqlalchemy.Table(
            relation.name,
            sqlalchemy.MetaData(),
            schema=relation.schema,
        )

    @staticmethod
    def _qualified_name(
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
    ) -> str:
        preparer = conn.dialect.identifier_preparer
        return (
            f"{preparer.quote_schema(relation.schema)}.{preparer.quote(relation.name)}"
        )

    @staticmethod
    def _quote_columns(
        conn: sqlalchemy.Connection,
        columns: Sequence[str],
    ) -> str:
        preparer = conn.dialect.identifier_preparer
        return ", ".join(preparer.quote(column) for column in columns)

    @staticmethod
    def ensure_schema(
        conn: sqlalchemy.Connection,
        schema: str,
    ) -> None:
        """Create ``schema`` when it does not already exist."""
        quoted_schema = conn.dialect.identifier_preparer.quote_schema(schema)
        conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))

    @contextmanager
    def stage_dataframe(
        self,
        conn: sqlalchemy.Connection,
        frame: pd.DataFrame,
        target: PostgresRelation,
    ) -> Iterator[PostgresRelation]:
        """Stage a DataFrame in a run-unique regular table.

        The staging table is created in the target schema and removed on context exit
        unless publication has renamed it.
        """
        stage = PostgresRelation(
            schema=target.schema,
            name=f"_cfc_stage_{uuid4().hex}",
        )
        self.ensure_schema(conn, stage.schema)
        if isinstance(frame, gpd.GeoDataFrame):
            geometry_column = frame.geometry.name
            if not isinstance(geometry_column, str):
                msg = "GeoDataFrame geometry column names must be strings"
                raise TypeError(msg)
            frame.to_postgis(
                stage.name,
                conn,
                schema=stage.schema,
                if_exists="fail",
                index=False,
            )
            self._drop_geometry_indexes(conn, stage, geometry_column)
        elif isinstance(frame, pd.DataFrame):
            frame.to_sql(
                stage.name,
                conn,
                schema=stage.schema,
                if_exists="fail",
                index=False,
            )
        else:
            msg = f"Unsupported staged frame type: {type(frame).__name__}"
            raise TypeError(msg)

        completed = False
        try:
            yield stage
            completed = True
        finally:
            # The surrounding transaction owns rollback after an error. Avoid issuing
            # cleanup SQL against a potentially aborted transaction.
            if completed and self.relation_exists(conn, stage):
                conn.execute(DropTable(self._table(stage), if_exists=True))

    @contextmanager
    def stage_query(
        self,
        conn: sqlalchemy.Connection,
        query: str,
        target: PostgresRelation,
    ) -> Iterator[PostgresRelation]:
        """Stage a row-producing query in a run-unique regular table."""
        query = query.strip()
        if not query:
            msg = "A staged PostgreSQL query must not be empty"
            raise ValueError(msg)

        stage = PostgresRelation(
            schema=target.schema,
            name=f"_cfc_stage_{uuid4().hex}",
        )
        self.ensure_schema(conn, stage.schema)
        conn.execute(
            sqlalchemy.text(
                f"CREATE TABLE {self._qualified_name(conn, stage)} AS {query}",
            ),
        )

        completed = False
        try:
            yield stage
            completed = True
        finally:
            # Match DataFrame staging: rollback owns cleanup after an error, while a
            # successful unpublished stage is removed explicitly.
            if completed and self.relation_exists(conn, stage):
                conn.execute(DropTable(self._table(stage), if_exists=True))

    def materialize_query(
        self,
        query: str,
        table_spec: PostgresTableSpec,
    ) -> None:
        """Stage and publish a query result in one database transaction."""
        with (
            self.begin() as conn,
            self.stage_query(conn, query, table_spec.relation) as stage,
        ):
            self.publish_relation(conn, stage, table_spec)

    def publish_relation(
        self,
        conn: sqlalchemy.Connection,
        source: PostgresRelation,
        spec: PostgresTableSpec,
    ) -> None:
        """Publish a staged relation according to a table specification."""
        target = spec.relation
        if source.schema != target.schema:
            msg = "Source and target relations must use the same schema"
            raise ValueError(msg)
        if not self.relation_exists(conn, source):
            msg = f"Staged relation {source.display_name} does not exist"
            raise ValueError(msg)

        target_exists = self.relation_exists(conn, target)
        if spec.write_mode is PostgresWriteMode.CREATE:
            if target_exists:
                msg = f"Target relation {target.display_name} already exists"
                raise ValueError(msg)
            self._rename_relation(conn, source, target)
            self._apply_table_spec(conn, spec)
            return

        if spec.write_mode is PostgresWriteMode.REPLACE:
            if target_exists:
                conn.execute(DropTable(self._table(target)))
            self._rename_relation(conn, source, target)
            self._apply_table_spec(conn, spec)
            return

        if not target_exists:
            msg = (
                f"Write mode {spec.write_mode.value!r} requires existing target "
                f"{target.display_name}"
            )
            raise ValueError(msg)

        columns = self._compatible_columns(conn, source, target)
        if spec.write_mode is PostgresWriteMode.TRUNCATE_INSERT:
            conn.execute(
                sqlalchemy.text(
                    f"TRUNCATE TABLE {self._qualified_name(conn, target)}",
                ),
            )

        quoted_columns = self._quote_columns(conn, columns)
        conn.execute(
            sqlalchemy.text(
                f"INSERT INTO {self._qualified_name(conn, target)} "  # noqa: S608 - identifiers are dialect-quoted.
                f"({quoted_columns}) SELECT {quoted_columns} "
                f"FROM {self._qualified_name(conn, source)}",
            ),
        )
        self.ensure_geometry_index(conn, spec)

    def _rename_relation(
        self,
        conn: sqlalchemy.Connection,
        source: PostgresRelation,
        target: PostgresRelation,
    ) -> None:
        target_name = conn.dialect.identifier_preparer.quote(target.name)
        conn.execute(
            sqlalchemy.text(
                f"ALTER TABLE {self._qualified_name(conn, source)} "
                f"RENAME TO {target_name}",
            ),
        )

    @staticmethod
    def _compatible_columns(
        conn: sqlalchemy.Connection,
        source: PostgresRelation,
        target: PostgresRelation,
    ) -> tuple[str, ...]:
        inspector = sqlalchemy.inspect(conn)
        source_columns = tuple(
            column["name"]
            for column in inspector.get_columns(source.name, schema=source.schema)
        )
        target_columns = tuple(
            column["name"]
            for column in inspector.get_columns(target.name, schema=target.schema)
        )
        if set(source_columns) != set(target_columns):
            msg = (
                "Staged and target relations must have identical columns for "
                "append or truncate_insert"
            )
            raise ValueError(msg)
        return target_columns

    def _apply_table_spec(
        self,
        conn: sqlalchemy.Connection,
        spec: PostgresTableSpec,
    ) -> None:
        relation_name = self._qualified_name(conn, spec.relation)
        if spec.primary_key:
            columns = self._quote_columns(conn, spec.primary_key)
            conn.execute(
                sqlalchemy.text(
                    f"ALTER TABLE {relation_name} ADD PRIMARY KEY ({columns})",
                ),
            )

        for foreign_key in spec.foreign_keys:
            columns = self._quote_columns(conn, foreign_key.columns)
            referenced_columns = self._quote_columns(
                conn,
                foreign_key.referenced_columns,
            )
            referenced_relation = self._qualified_name(
                conn,
                foreign_key.referenced_relation,
            )
            conn.execute(
                sqlalchemy.text(
                    f"ALTER TABLE {relation_name} ADD FOREIGN KEY ({columns}) "
                    f"REFERENCES {referenced_relation} ({referenced_columns})",
                ),
            )

        self.ensure_geometry_index(conn, spec)

        for index in spec.indexes:
            self._create_index(conn, spec.relation, index)

    def ensure_geometry_index(
        self,
        conn: sqlalchemy.Connection,
        spec: PostgresTableSpec,
    ) -> None:
        """Ensure a spatial table has exactly one managed geometry GiST index."""
        if spec.geometry_column is None:
            return
        if self._geometry_indexes(conn, spec.relation, spec.geometry_column):
            return
        self._create_index(
            conn,
            spec.relation,
            PostgresIndex(
                columns=(spec.geometry_column,),
                method=PostgresIndexMethod.GIST,
            ),
        )

    def _drop_geometry_indexes(
        self,
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        geometry_column: str,
    ) -> None:
        preparer = conn.dialect.identifier_preparer
        for index_name in self._geometry_indexes(conn, relation, geometry_column):
            qualified_index = (
                f"{preparer.quote_schema(relation.schema)}.{preparer.quote(index_name)}"
            )
            conn.execute(
                sqlalchemy.text(
                    f"DROP INDEX {qualified_index}",
                ),
            )

    @staticmethod
    def _geometry_indexes(
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        geometry_column: str,
    ) -> tuple[str, ...]:
        indexes = sqlalchemy.inspect(conn).get_indexes(
            relation.name,
            schema=relation.schema,
        )
        return tuple(
            index_name
            for index in indexes
            if index.get("column_names") == [geometry_column]
            and isinstance(index.get("dialect_options"), Mapping)
            and index["dialect_options"].get("postgresql_using") == "gist"
            and isinstance(index_name := index.get("name"), str)
        )

    def _create_index(
        self,
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        index: PostgresIndex,
    ) -> None:
        index_name = index.name or self._generated_index_name(relation, index)
        quoted_index = conn.dialect.identifier_preparer.quote(index_name)
        unique = "UNIQUE " if index.unique else ""
        columns = self._quote_columns(conn, index.columns)
        conn.execute(
            sqlalchemy.text(
                f"CREATE {unique}INDEX {quoted_index} "
                f"ON {self._qualified_name(conn, relation)} "
                f"USING {index.method.value} ({columns})",
            ),
        )

    @staticmethod
    def _generated_index_name(
        relation: PostgresRelation,
        index: PostgresIndex,
    ) -> str:
        base = f"{relation.name}_{'_'.join(index.columns)}_{index.method.value}_idx"
        if len(base) <= POSTGRES_IDENTIFIER_MAX_LENGTH:
            return base
        digest = hashlib.sha1(base.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"{base[:54]}_{digest}"

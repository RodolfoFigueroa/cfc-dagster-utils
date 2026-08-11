from typing import Annotated, TypeAlias, cast

import sqlalchemy

from cfc_dagster_utils._optional import raise_optional_dependency_error

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
    import sqlparse
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="sqlparse",
        dependency_name="sqlparse",
        extra="postgres",
    )

from dagster.components.lib.sql_component.sql_client import (
    SQLClient,
)
from pydantic import BaseModel, Field

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource
from cfc_dagster_utils.types import (
    PostgresForeignKey,
    PostgresIndex,
    PostgresIndexMethod,
    PostgresRelation,
    PostgresTableSpec,
    PostgresWriteMode,
)


class PostgresRelationArgs(dg.Model, dg.Resolvable):
    """YAML-facing arguments for a PostgreSQL relation."""

    name: str
    schema_name: str = Field(default="public", alias="schema")


class PostgresForeignKeyArgs(dg.Model, dg.Resolvable):
    """YAML-facing arguments for a PostgreSQL foreign key."""

    columns: list[str]
    referenced_relation: PostgresRelationArgs
    referenced_columns: list[str]


class PostgresIndexArgs(dg.Model, dg.Resolvable):
    """YAML-facing arguments for a PostgreSQL index."""

    columns: list[str]
    method: PostgresIndexMethod = PostgresIndexMethod.BTREE
    unique: bool = False
    name: str | None = None


class PostgresTableSpecArgs(dg.Model, dg.Resolvable):
    """YAML-facing arguments for :class:`PostgresTableSpec`."""

    relation: PostgresRelationArgs
    write_mode: PostgresWriteMode = PostgresWriteMode.CREATE
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[PostgresForeignKeyArgs] = Field(default_factory=list)
    indexes: list[PostgresIndexArgs] = Field(default_factory=list)
    geometry_column: str | None = None


def _resolve_table_spec(
    context: dg.ResolutionContext,
    model: BaseModel,
) -> PostgresTableSpec:
    args = PostgresTableSpecArgs.resolve_from_model(context, model)
    return PostgresTableSpec(
        relation=PostgresRelation(
            name=args.relation.name,
            schema=args.relation.schema_name,
        ),
        write_mode=args.write_mode,
        primary_key=tuple(args.primary_key),
        foreign_keys=tuple(
            PostgresForeignKey(
                columns=tuple(foreign_key.columns),
                referenced_relation=PostgresRelation(
                    name=foreign_key.referenced_relation.name,
                    schema=foreign_key.referenced_relation.schema_name,
                ),
                referenced_columns=tuple(foreign_key.referenced_columns),
            )
            for foreign_key in args.foreign_keys
        ),
        indexes=tuple(
            PostgresIndex(
                columns=tuple(index.columns),
                method=index.method,
                unique=index.unique,
                name=index.name,
            )
            for index in args.indexes
        ),
        geometry_column=args.geometry_column,
    )


ResolvedPostgresTableSpec: TypeAlias = Annotated[
    PostgresTableSpec,
    dg.Resolver(
        _resolve_table_spec,
        model_field_type=PostgresTableSpecArgs.model(),
    ),
]


class PostgresConnectionComponent(
    dg.Component,
    dg.Resolvable,
    dg.Model,
    SQLClient,
):
    """Configure PostgreSQL resources and serve as Dagster's SQL client."""

    host: str
    port: int
    user: str
    password: str
    database: str
    resource_key: str = "postgres_resource"
    io_manager_key: str = "postgres_manager"

    def model_post_init(self, context: object, /) -> None:  # noqa: ARG002
        if self.resource_key == self.io_manager_key:
            msg = "resource_key and io_manager_key must be different"
            raise ValueError(msg)

    def create_resource(self) -> PostgresResource:
        """Create the resource represented by this connection configuration."""
        return PostgresResource(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:  # noqa: ARG002
        resource = self.create_resource()
        return dg.Definitions(
            resources={
                self.resource_key: resource,
                self.io_manager_key: PostgresIOManager(postgres_resource=resource),
            },
        )

    def connect_and_execute(self, sql: str) -> None:
        """Execute generic SQL transactionally for Dagster's base SQL component."""
        resource = self.create_resource()
        with dg.build_init_resource_context() as init_context:
            resource.setup_for_execution(init_context)
            try:
                with resource.begin() as conn:
                    conn.execute(sqlalchemy.text(sql))
            finally:
                resource.teardown_after_execution(init_context)


ResolvedPostgresConnection: TypeAlias = Annotated[
    PostgresConnectionComponent,
    dg.Resolver(lambda _context, value: value, model_field_type=str),
]


class PostgresTemplatedSqlComponent(dg.TemplatedSqlComponent):
    """Materialize one SQL query according to a PostgreSQL table contract."""

    connection: ResolvedPostgresConnection
    table_spec: ResolvedPostgresTableSpec

    def model_post_init(self, context: object, /) -> None:  # noqa: ARG002
        if len(self.assets or []) != 1:
            msg = "PostgresTemplatedSqlComponent requires exactly one asset"
            raise ValueError(msg)

    @property
    def resource_keys(self) -> set[str]:
        return {self.connection.resource_key}

    def _configured_asset(self) -> dg.AssetSpec:
        asset = cast("list[dg.AssetSpec]", self.assets)[0]
        kinds = {"sql", "postgres"}
        if self.table_spec.geometry_column is not None:
            kinds.add("postgis")
        return asset.merge_attributes(
            metadata=self.table_spec.to_dagster_metadata(),
            kinds=kinds,
        ).with_io_manager_key(self.connection.io_manager_key)

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        configured = self.model_copy(update={"assets": [self._configured_asset()]})
        return dg.TemplatedSqlComponent.build_defs(configured, context)

    @staticmethod
    def _row_query(sql: str) -> str:
        statements = [
            statement for statement in sqlparse.parse(sql) if str(statement).strip()
        ]
        is_select = len(statements) == 1 and (
            statements[0].get_type() == "SELECT"
            or any(
                token.ttype is sqlparse.tokens.DML and token.normalized == "SELECT"
                for token in statements[0].tokens
            )
        )
        if not is_select:
            msg = "SQL assets require exactly one row-producing SELECT or WITH query"
            raise ValueError(msg)
        return str(statements[0]).strip().removesuffix(";").rstrip()

    def execute(
        self,
        context: dg.AssetExecutionContext,
        component_load_context: dg.ComponentLoadContext,
    ) -> None:
        query = self._row_query(self.get_sql_content(context, component_load_context))
        resource = cast(
            "PostgresResource",
            getattr(context.resources, self.connection.resource_key),
        )
        resource.materialize_query(query, self.table_spec)

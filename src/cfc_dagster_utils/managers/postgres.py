from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import geopandas as gpd
import pandas as pd
import sqlalchemy

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.resources import PostgresResource
from cfc_dagster_utils.types import PostgresRelation, PostgresTableSpec

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )


class _FrameHandler(Protocol):
    def load(
        self,
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        columns: tuple[str, ...] | None,
        spec: PostgresTableSpec,
    ) -> pd.DataFrame: ...


def _qualified_name(
    conn: sqlalchemy.Connection,
    relation: PostgresRelation,
) -> str:
    preparer = conn.dialect.identifier_preparer
    return f"{preparer.quote_schema(relation.schema)}.{preparer.quote(relation.name)}"


def _select_sql(
    conn: sqlalchemy.Connection,
    relation: PostgresRelation,
    columns: tuple[str, ...] | None,
) -> sqlalchemy.TextClause:
    if columns is None:
        selected = "*"
    else:
        preparer = conn.dialect.identifier_preparer
        selected = ", ".join(preparer.quote(column) for column in columns)
    return sqlalchemy.text(
        f"SELECT {selected} FROM {_qualified_name(conn, relation)}"  # noqa: S608 - identifiers are dialect-quoted.
    )


class _PandasHandler:
    def load(
        self,
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        columns: tuple[str, ...] | None,
        spec: PostgresTableSpec,  # noqa: ARG002
    ) -> pd.DataFrame:
        return pd.read_sql(_select_sql(conn, relation, columns), conn)


class _GeoPandasHandler:
    def load(
        self,
        conn: sqlalchemy.Connection,
        relation: PostgresRelation,
        columns: tuple[str, ...] | None,
        spec: PostgresTableSpec,
    ) -> gpd.GeoDataFrame:
        geometry_column = spec.geometry_column or "geometry"
        if columns is not None and geometry_column not in columns:
            msg = (
                f"GeoDataFrame input columns must include geometry column "
                f"{geometry_column!r}"
            )
            raise ValueError(msg)
        return gpd.read_postgis(
            _select_sql(conn, relation, columns),
            conn,
            geom_col=geometry_column,
        )


class PostgresIOManager(dg.ConfigurableIOManager):
    """Persist pandas, GeoPandas, or server-native PostgreSQL relation assets."""

    postgres_resource: dg.ResourceDependency[PostgresResource]

    @staticmethod
    def _spec(metadata: Mapping[str, Any]) -> PostgresTableSpec:
        return PostgresTableSpec.from_dagster_metadata(metadata)

    @staticmethod
    def _columns(metadata: Mapping[str, Any]) -> tuple[str, ...] | None:
        value = metadata.get("columns")
        if value is None:
            return None
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str)
            or not all(isinstance(column, str) and column for column in value)
        ):
            msg = "Input metadata 'columns' must be a sequence of non-empty strings"
            raise ValueError(msg)
        return tuple(value)

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: pd.DataFrame | PostgresRelation,
    ) -> None:
        """Persist a frame or validate an already-published relation."""
        spec = self._spec(context.definition_metadata)
        if isinstance(obj, PostgresRelation):
            self._handle_relation_output(spec, obj)
            context.add_output_metadata(self._output_metadata(spec, None))
            return

        if not isinstance(obj, pd.DataFrame):
            msg = f"Unsupported PostgreSQL output type: {type(obj).__name__}"
            raise TypeError(msg)

        if isinstance(obj, gpd.GeoDataFrame):
            geometry_column = spec.geometry_column or "geometry"
            if obj.geometry.name != geometry_column:
                msg = (
                    f"GeoDataFrame geometry column {obj.geometry.name!r} does not "
                    f"match table specification {geometry_column!r}"
                )
                raise ValueError(msg)

        with (
            self.postgres_resource.begin() as conn,
            self.postgres_resource.stage_dataframe(
                conn,
                obj,
                spec.relation,
            ) as stage,
        ):
            self.postgres_resource.publish_relation(conn, stage, spec)

        context.add_output_metadata(self._output_metadata(spec, len(obj)))

    def _handle_relation_output(
        self,
        spec: PostgresTableSpec,
        relation: PostgresRelation,
    ) -> None:
        if relation != spec.relation:
            msg = (
                f"Returned relation {relation.display_name} does not match declared "
                f"relation {spec.relation.display_name}"
            )
            raise ValueError(msg)
        with self.postgres_resource.connect() as conn:
            if not self.postgres_resource.relation_exists(conn, relation):
                msg = f"Published relation {relation.display_name} does not exist"
                raise ValueError(msg)

    def load_input(
        self,
        context: dg.InputContext,
    ) -> pd.DataFrame | PostgresRelation:
        """Return a zero-copy relation or load the requested frame type."""
        upstream_output = context.upstream_output
        if upstream_output is None:
            msg = "PostgreSQL inputs require an upstream output"
            raise ValueError(msg)

        spec = self._spec(upstream_output.definition_metadata)
        requested_type = context.dagster_type.typing_type
        if requested_type is PostgresRelation:
            return spec.relation

        handler = self._handler_for_type(requested_type)
        columns = self._columns(context.definition_metadata)
        with self.postgres_resource.connect() as conn:
            return handler.load(conn, spec.relation, columns, spec)

    @staticmethod
    def _handler_for_type(requested_type: object) -> _FrameHandler:
        if isinstance(requested_type, type) and issubclass(
            requested_type,
            gpd.GeoDataFrame,
        ):
            return _GeoPandasHandler()
        if isinstance(requested_type, type) and issubclass(
            requested_type,
            pd.DataFrame,
        ):
            return _PandasHandler()
        msg = (
            "PostgresIOManager inputs must be annotated as PostgresRelation, "
            "pandas.DataFrame, or geopandas.GeoDataFrame"
        )
        raise TypeError(msg)

    @staticmethod
    def _output_metadata(
        spec: PostgresTableSpec,
        row_count: int | None,
    ) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "schema": spec.relation.schema,
            "table": spec.relation.name,
            "relation": spec.relation.display_name,
            "write_mode": spec.write_mode.value,
        }
        if row_count is not None:
            metadata["row_count"] = row_count
        if spec.geometry_column is not None:
            metadata["geometry_column"] = spec.geometry_column
        return metadata

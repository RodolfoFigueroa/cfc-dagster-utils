from collections.abc import Mapping
from hashlib import sha256
from typing import Any

import geopandas as gpd
import pandas as pd
import sqlalchemy

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.managers._postgres import converted_rows, frame_columns
from cfc_dagster_utils.resources import PostgresResource

try:
    from psycopg import sql
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error, import_name="psycopg", dependency_name="psycopg", extra="postgres"
    )

try:
    import dagster as dg
    from pydantic import field_validator
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )


class PostgresIOManager(dg.ConfigurableIOManager):
    """Write pandas and GeoPandas outputs to PostgreSQL tables."""

    postgres_resource: dg.ResourceDependency[PostgresResource]
    batch_size: int = 1000

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value: object) -> int:
        if type(value) is not int or value <= 0:
            msg = "batch_size must be a positive integer"
            raise ValueError(msg)
        return value

    @staticmethod
    def _destination(metadata: Mapping[str, Any]) -> tuple[str, str]:
        schema = metadata.get("schema")
        table = metadata.get("table")
        if not isinstance(schema, str) or not schema.strip():
            msg = "PostgreSQL outputs require non-empty string metadata 'schema'"
            raise ValueError(msg)
        if not isinstance(table, str) or not table.strip():
            msg = "PostgreSQL outputs require non-empty string metadata 'table'"
            raise ValueError(msg)
        return schema, table

    @staticmethod
    def _ensure_schema(conn: sqlalchemy.Connection, schema: str) -> None:
        # Stable across worker processes; held only through schema setup's commit.
        digest = sha256(f"cfc_dagster_utils:schema:{schema}".encode()).digest()
        lock_id = int.from_bytes(digest[:8], byteorder="big", signed=True)
        conn.execute(
            sqlalchemy.text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": lock_id},
        )
        quoted_schema = conn.dialect.identifier_preparer.quote_schema(schema)
        conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: pd.DataFrame,
    ) -> None:
        """Replace the configured PostgreSQL table with a frame."""
        if not isinstance(obj, pd.DataFrame):
            msg = f"Unsupported PostgreSQL output type: {type(obj).__name__}"
            raise TypeError(msg)

        schema, table = self._destination(context.definition_metadata)
        geometry_column: str | None = None
        if isinstance(obj, gpd.GeoDataFrame):
            geometry_name = obj.geometry.name
            if not isinstance(geometry_name, str):
                msg = "GeoDataFrame geometry column names must be strings"
                raise TypeError(msg)
            geometry_column = geometry_name

        columns = frame_columns(obj)
        destination = sqlalchemy.Table(
            table,
            sqlalchemy.MetaData(),
            *(sqlalchemy.Column(column.name, column.sql_type) for column in columns),
            schema=schema,
        )
        copy_statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
            sql.Identifier(schema, table),
            sql.SQL(", ").join(sql.Identifier(column.name) for column in columns),
        )
        with self.postgres_resource.begin() as conn:
            self._ensure_schema(conn, schema)
        with self.postgres_resource.begin() as conn:
            destination.drop(conn, checkfirst=True)
            destination.create(conn)
            driver = conn.connection.driver_connection
            if driver is None:
                msg = "PostgreSQL connection has no driver connection"
                raise RuntimeError(msg)
            with driver.cursor() as cursor, cursor.copy(copy_statement) as copy:
                for start in range(0, len(obj), self.batch_size):
                    batch = obj.iloc[start : start + self.batch_size]
                    for row in converted_rows(batch, columns):
                        copy.write_row(row)
                    del batch

        metadata: dict[str, str | int] = {
            "schema": schema,
            "table": table,
            "relation": f"{schema}.{table}",
            "row_count": len(obj),
        }
        if geometry_column is not None:
            metadata["geometry_column"] = geometry_column
        context.add_output_metadata(metadata)

    def load_input(self, context: dg.InputContext) -> None:
        """Reject input loading because this manager is output-only."""
        msg = "PostgresIOManager is output-only and cannot load inputs"
        raise NotImplementedError(msg)

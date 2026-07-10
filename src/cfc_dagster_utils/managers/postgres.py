from typing import Any, Generic, Literal

import dagster as dg
import geopandas as gpd
import pandas as pd
import sqlalchemy
from dagster._config.pythonic_config.resource import TResValue

from cfc_dagster_utils.resources import PostgresResource
from cfc_dagster_utils.types import DFType


class _DataFrameBasePostgresManager(
    dg.ConfigurableIOManager,
    Generic[DFType, TResValue],
):
    if_exists: Literal["fail", "append", "replace", "cascade_replace"]
    """Base class for Dagster IO managers that read and write DataFrames to/from
    PostgreSQL.

    Subclasses must implement ``write_table`` and ``load_table`` to define how data is
    serialized and deserialized for a specific DataFrame type.

    Args:
        postgres_resource: A Dagster resource dependency providing a PostgreSQL
            connection.
        if_exists: Behavior when the target table already exists. Options are:
            - ``"fail"``: Raise an error (default).
            - ``"append"``: Append data to the existing table.
            - ``"replace"``: Drop the existing table and create a new one.
            - ``"cascade_replace"``: Drop the existing table and any dependent objects,
              then create a new table.

    Attributes:
        postgres_resource: A Dagster resource dependency providing a PostgreSQL
            connection.
    """

    postgres_resource: dg.ResourceDependency[PostgresResource]

    def _prepare_for_write(
        self, conn: sqlalchemy.Connection, table_name: str
    ) -> Literal["fail", "append", "replace"]:
        if self.if_exists != "cascade_replace":
            return self.if_exists

        inspector = sqlalchemy.inspect(conn)
        if not inspector.has_table(table_name):
            return "replace"

        conn.execute(sqlalchemy.text(f"DROP TABLE {table_name} CASCADE;"))
        return "replace"

    def write_table(
        self,
        df: DFType,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a DataFrame to a PostgreSQL table.

        Must be implemented by subclasses.

        Args:
            df: The DataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        msg = "write_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> DFType:
        """Load data from a PostgreSQL table into a DataFrame.

        Must be implemented by subclasses.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"``
                for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded DataFrame.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        msg = "load_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: DFType,
    ) -> None:
        """Persist a DataFrame as a PostgreSQL table.

        Writes the DataFrame to the table specified in
        ``context.definition_metadata["table_name"]``. Optionally adds a primary key
        constraint and foreign key constraints based on additional metadata keys.

        Args:
            context: The Dagster output context. Relevant metadata keys:
                - ``table_name`` (str, required): Target table name.
                - ``primary_key`` (str, optional): Column to set as the primary key.
                - ``foreign_keys`` (list[dict], optional): List of foreign key mappings,
                  each with keys ``column``, ``ref_table``, and ``ref_column``.
            obj: The DataFrame to write.

        Raises:
            ValueError: If the specified primary key or a foreign key column is not
                present in the DataFrame's columns.
        """
        table = context.definition_metadata["table_name"]

        with self.postgres_resource.begin() as conn:
            self.write_table(obj, table, conn)

            if "primary_key" in context.definition_metadata:
                primary_key = context.definition_metadata["primary_key"]

                if primary_key not in obj.columns:
                    err = f"Primary key {primary_key} not found in DataFrame columns"
                    raise ValueError(err)

                conn.execute(
                    sqlalchemy.text(
                        f'ALTER TABLE {table} ADD PRIMARY KEY ("{primary_key}");',
                    ),
                )

            if "foreign_keys" in context.definition_metadata:
                foreign_keys = context.definition_metadata["foreign_keys"]

                for fk_map in foreign_keys:
                    fk_col = fk_map["column"]
                    ref_table = fk_map["ref_table"]
                    ref_col = fk_map["ref_column"]

                    if fk_col not in obj.columns:
                        err = (
                            f"Foreign key column {fk_col} not found in DataFrame "
                            "columns."
                        )
                        raise ValueError(err)

                    conn.execute(
                        sqlalchemy.text(
                            f"""
                            ALTER TABLE {table}
                            ADD FOREIGN KEY ("{fk_col}")
                            REFERENCES {ref_table}("{ref_col}")
                            """,
                        ),
                    )

    def load_input(self, context: dg.InputContext) -> DFType:
        """Load a DataFrame from the PostgreSQL table written by the upstream output.

        Reads from the table specified in the upstream output's ``table_name`` metadata.
        If the input metadata contains a ``columns`` key, only those columns are
        selected.

        Args:
            context: The Dagster input context. Relevant metadata keys:
                - ``columns`` (list[str], optional): Specific columns to load. Defaults
                        to all columns.

        Returns:
            The loaded DataFrame.

        Raises:
            ValueError: If no upstream output is found.
        """
        upstream_output = context.upstream_output
        if upstream_output is None:
            err = "No upstream output found."
            raise ValueError(err)

        table = upstream_output.definition_metadata["table_name"]

        in_metadata = context.definition_metadata
        if "columns" in in_metadata:
            wanted_cols = in_metadata["columns"]
            cols_str = ", ".join(wanted_cols)
        else:
            cols_str = "*"

        with self.postgres_resource.connect() as conn:
            return self.load_table(table, cols_str, conn)


class DataFramePostgresManager(_DataFrameBasePostgresManager[pd.DataFrame, Any]):
    """Dagster IO manager for reading and writing pandas DataFrames to/from PostgreSQL.

    Uses pandas ``to_sql`` and ``read_sql`` for serialization and deserialization.
    Inherits output and input handling (including primary and foreign key constraints)
    from ``_DataFrameBasePostgresManager``.

    Args:
        postgres_resource: A Dagster resource dependency providing a PostgreSQL
            connection.
    """

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a pandas DataFrame to a PostgreSQL table using ``to_sql``.

        Replaces the table if it already exists.

        Args:
            df: The pandas DataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.
        """
        if_exists = self._prepare_for_write(conn, table_name)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> pd.DataFrame:
        """Load a pandas DataFrame from a PostgreSQL table using ``read_sql``.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"``
                for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded DataFrame.
        """
        return pd.read_sql(f"SELECT {cols_str} FROM {table_name}", conn)  # noqa: S608


class GeoDataFramePostGISManager(_DataFrameBasePostgresManager[gpd.GeoDataFrame, Any]):
    """Dagster IO manager for reading and writing GeoDataFrames to/from PostGIS.

    Uses geopandas ``to_postgis`` and ``read_postgis`` for serialization and
    deserialization. Assumes a ``geometry`` column is present for spatial data.
    Inherits output and input handling from ``_DataFrameBasePostgresManager``.

    Args:
        postgres_resource: A Dagster resource dependency providing a PostgreSQL
            connection.
    """

    def write_table(
        self,
        df: gpd.GeoDataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a GeoDataFrame to a PostGIS table using ``to_postgis``.

        Replaces the table if it already exists.

        Args:
            df: The GeoDataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.
        """
        if_exists = self._prepare_for_write(conn, table_name)
        df.to_postgis(table_name, conn, if_exists=if_exists)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> gpd.GeoDataFrame:
        """Load a GeoDataFrame from a PostGIS table using ``read_postgis``.

        Assumes the geometry column is named ``geometry``.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"``
                for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded GeoDataFrame.
        """
        return gpd.read_postgis(
            f"SELECT {cols_str} FROM {table_name}",  # noqa: S608
            conn,
            geom_col="geometry",
        )

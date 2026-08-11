# ruff: noqa: SLF001

from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import dagster as dg
import pytest
import sqlalchemy

from cfc_dagster_utils.components.postgres import (
    PostgresConnectionComponent,
    PostgresTemplatedSqlComponent,
)
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource
from cfc_dagster_utils.types import (
    PostgresRelation,
    PostgresTableSpec,
    PostgresWriteMode,
)

TEST_PASSWORD = "secret"  # noqa: S105 - inert test credential.


def _connection() -> PostgresConnectionComponent:
    return PostgresConnectionComponent(
        host="db.example",
        port=5432,
        user="user",
        password=TEST_PASSWORD,
        database="warehouse",
        resource_key="warehouse_resource",
        io_manager_key="warehouse_manager",
    )


def _component(*, geometry: bool = True) -> PostgresTemplatedSqlComponent:
    return PostgresTemplatedSqlComponent(
        assets=[dg.AssetSpec("analytics/result", kinds={"census"})],
        connection=_connection(),
        sql_template="SELECT 1 AS id",
        table_spec=PostgresTableSpec(
            relation=PostgresRelation(name="result", schema="analytics"),
            write_mode=PostgresWriteMode.REPLACE,
            geometry_column="geometry" if geometry else None,
        ),
    )


def test_connection_component_wires_resource_and_io_manager() -> None:
    component = _connection()

    definitions = component.build_defs(MagicMock(spec=dg.ComponentLoadContext))

    resources = definitions.resources
    assert resources is not None
    assert set(resources) == {"warehouse_resource", "warehouse_manager"}
    assert isinstance(resources["warehouse_resource"], PostgresResource)
    assert isinstance(resources["warehouse_manager"], PostgresIOManager)


def test_connection_component_executes_generic_sql_transactionally() -> None:
    component = _connection()
    resource = MagicMock(spec=PostgresResource)
    conn = MagicMock(spec=sqlalchemy.Connection)
    transaction = MagicMock(spec=AbstractContextManager)
    transaction.__enter__.return_value = conn
    resource.begin.return_value = transaction

    with patch.object(
        PostgresConnectionComponent,
        "create_resource",
        return_value=resource,
    ):
        component.connect_and_execute("CREATE TEMP TABLE example (id integer)")

    resource.setup_for_execution.assert_called_once()
    transaction.__enter__.assert_called_once_with()
    statement = conn.execute.call_args.args[0]
    assert str(statement) == "CREATE TEMP TABLE example (id integer)"
    resource.teardown_after_execution.assert_called_once()


def test_sql_component_adds_contract_kinds_and_io_manager_metadata() -> None:
    component = _component()

    spec = component._configured_asset()

    assert spec.kinds == {"census", "sql", "postgres", "postgis"}
    assert spec.metadata["dagster/io_manager_key"] == "warehouse_manager"
    assert (
        PostgresTableSpec.from_dagster_metadata(spec.metadata) == component.table_spec
    )


def test_non_spatial_sql_component_does_not_add_postgis_kind() -> None:
    assert _component(geometry=False)._configured_asset().kinds == {
        "census",
        "sql",
        "postgres",
    }


@pytest.mark.parametrize(
    "assets", [None, [], [dg.AssetSpec("one"), dg.AssetSpec("two")]]
)
def test_sql_component_requires_exactly_one_asset(
    assets: list[dg.AssetSpec] | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one asset"):
        PostgresTemplatedSqlComponent(
            assets=assets,
            connection=_connection(),
            sql_template="SELECT 1",
            table_spec=PostgresTableSpec(PostgresRelation("result")),
        )


@pytest.mark.parametrize(
    "sql",
    ["", "DELETE FROM source", "SELECT 1; SELECT 2", "CREATE TABLE x (id int)"],
)
def test_sql_component_rejects_non_row_queries(sql: str) -> None:
    with pytest.raises(ValueError, match="one row-producing"):
        PostgresTemplatedSqlComponent._row_query(sql)


def test_sql_component_accepts_select_and_with_queries() -> None:
    assert PostgresTemplatedSqlComponent._row_query("SELECT 1;") == "SELECT 1"
    assert PostgresTemplatedSqlComponent._row_query(
        "WITH source AS (SELECT 1 AS id) SELECT id FROM source",
    ).startswith("WITH source")


def test_sql_component_materializes_rendered_query_with_runtime_resource() -> None:
    component = _component()
    resource = MagicMock(spec=PostgresResource)
    context = MagicMock(spec=dg.AssetExecutionContext)
    context.resources.warehouse_resource = resource
    load_context = MagicMock(spec=dg.ComponentLoadContext)

    with patch.object(
        PostgresTemplatedSqlComponent,
        "get_sql_content",
        return_value=" SELECT 1 AS id; ",
    ):
        component.execute(context, load_context)

    resource.materialize_query.assert_called_once_with(
        "SELECT 1 AS id",
        component.table_spec,
    )

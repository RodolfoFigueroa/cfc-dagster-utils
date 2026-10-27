from collections.abc import Iterator
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import dagster as dg
import geopandas as gpd
import pandas as pd
import pytest
import sqlalchemy
from shapely.geometry import Point
from sqlalchemy.dialects import postgresql

from cfc_dagster_utils.managers._postgres import converted_rows
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

TEST_PASSWORD = uuid4().hex
SPECIAL_TEST_PASSWORD = f"p@{uuid4().hex}/:rd"


def _resource() -> PostgresResource:
    return PostgresResource(
        host="db.example",
        port=5432,
        user="user",
        password=TEST_PASSWORD,
        database="warehouse",
    )


def _output_context(metadata: dict[str, object] | None = None) -> MagicMock:
    context = MagicMock(spec=dg.OutputContext)
    context.definition_metadata = (
        metadata if metadata is not None else {"schema": "analytics", "table": "target"}
    )
    return context


def _writer(
    batch_size: int = 1000,
) -> tuple[PostgresIOManager, MagicMock, MagicMock, MagicMock]:
    manager = PostgresIOManager(postgres_resource=_resource(), batch_size=batch_size)
    conn = MagicMock(spec=sqlalchemy.Connection)
    conn.dialect = postgresql.dialect()
    transaction = MagicMock(spec=AbstractContextManager)
    transaction.__enter__.return_value = conn
    setup_conn = MagicMock(spec=sqlalchemy.Connection)
    setup_conn.dialect = postgresql.dialect()
    setup = MagicMock(spec=AbstractContextManager)
    setup.__enter__.return_value = setup_conn
    return manager, conn, transaction, setup


def _copy(conn: MagicMock) -> MagicMock:
    cursor = (
        conn.connection.driver_connection.cursor.return_value.__enter__.return_value
    )
    return cursor.copy.return_value.__enter__.return_value


@pytest.fixture(autouse=True)
def mock_ddl() -> Iterator[None]:
    with (
        patch.object(sqlalchemy.Table, "drop"),
        patch.object(sqlalchemy.Table, "create"),
    ):
        yield


def test_resource_builds_psycopg_url_and_disposes_engine() -> None:
    resource = PostgresResource(
        host="db.example",
        port=5432,
        user="user@tenant",
        password=SPECIAL_TEST_PASSWORD,
        database="warehouse",
    )
    engine = MagicMock(spec=sqlalchemy.Engine)
    with patch("sqlalchemy.create_engine", return_value=engine) as create_engine:
        resource.setup_for_execution(MagicMock())
    url = create_engine.call_args.args[0]
    assert isinstance(url, sqlalchemy.URL)
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "user@tenant"
    assert url.password == SPECIAL_TEST_PASSWORD
    resource.teardown_after_execution(MagicMock())
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"schema": "analytics"},
        {"table": "target"},
        {"schema": "", "table": "target"},
        {"schema": "   ", "table": "target"},
        {"schema": 1, "table": "target"},
        {"schema": "analytics", "table": ""},
        {"schema": "analytics", "table": "   "},
        {"schema": "analytics", "table": 1},
    ],
)
def test_invalid_destination_does_not_connect(metadata: dict[str, object]) -> None:
    manager, _, _, _ = _writer()
    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(ValueError, match="metadata"),
    ):
        manager.handle_output(_output_context(metadata), pd.DataFrame({"id": [1]}))
    begin.assert_not_called()


@pytest.mark.parametrize("size", [1, 2, 1000])
def test_bounded_copy_and_metadata(size: int) -> None:
    manager, conn, transaction, setup = _writer(size)
    context = _output_context({"schema": 'custom " schema', "table": 'target " table'})
    frame = pd.DataFrame(
        {"id": range(5), 'quoted " column': [None, "", "a\nb", "á", "\\"]}
    )
    original = frame.copy(deep=True)
    with (
        patch.object(PostgresResource, "begin", side_effect=[setup, transaction]),
        patch(
            "cfc_dagster_utils.managers.postgres.converted_rows", wraps=converted_rows
        ) as convert,
        patch.object(pd.DataFrame, "to_sql") as to_sql,
        patch.object(gpd.GeoDataFrame, "to_postgis") as to_postgis,
    ):
        manager.handle_output(context, frame)
    assert [len(call.args[0]) for call in convert.call_args_list] == [
        min(size, len(frame) - start) for start in range(0, len(frame), size)
    ]
    assert [call.args[0] for call in _copy(conn).write_row.call_args_list] == [
        (0, None),
        (1, ""),
        (2, "a\nb"),
        (3, "á"),
        (4, "\\"),
    ]
    cursor = (
        conn.connection.driver_connection.cursor.return_value.__enter__.return_value
    )
    assert cursor.copy.call_args.args[0].as_string() == (
        'COPY "custom "" schema"."target "" table" '
        '("id", "quoted "" column") FROM STDIN'
    )
    to_sql.assert_not_called()
    to_postgis.assert_not_called()
    pd.testing.assert_frame_equal(frame, original)
    context.add_output_metadata.assert_called_once_with(
        {
            "schema": 'custom " schema',
            "table": 'target " table',
            "relation": 'custom " schema.target " table',
            "row_count": 5,
        }
    )
    transaction.__exit__.assert_called_once_with(None, None, None)


def test_geometry_metadata_and_empty_output() -> None:
    manager, conn, transaction, setup = _writer()
    frame = gpd.GeoDataFrame(
        {"id": pd.Series(dtype="Int64")}, geometry=gpd.GeoSeries([], crs=4326)
    )
    frame.rename_geometry("geom", inplace=True)
    context = _output_context()
    with patch.object(PostgresResource, "begin", side_effect=[setup, transaction]):
        manager.handle_output(context, frame)
    _copy(conn).write_row.assert_not_called()
    assert context.add_output_metadata.call_args.args[0]["geometry_column"] == "geom"
    assert context.add_output_metadata.call_args.args[0]["row_count"] == 0


def test_geometry_column_requires_string_name() -> None:
    manager, _, _, _ = _writer()
    frame = gpd.GeoDataFrame({0: [Point(0, 0)]}, geometry=0, crs=4326)
    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(TypeError, match="geometry column"),
    ):
        manager.handle_output(_output_context(), frame)
    begin.assert_not_called()


def test_failure_rolls_back_without_success_metadata() -> None:
    manager, conn, transaction, setup = _writer(1)
    context = _output_context()
    _copy(conn).write_row.side_effect = [None, RuntimeError("write failed")]
    with (
        patch.object(PostgresResource, "begin", side_effect=[setup, transaction]),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        manager.handle_output(context, pd.DataFrame({"id": [1, 2]}))
    assert transaction.__exit__.call_args.args[0] is RuntimeError
    context.add_output_metadata.assert_not_called()


def test_commit_failure_has_no_success_metadata() -> None:
    manager, _, transaction, setup = _writer()
    context = _output_context()
    transaction.__exit__.side_effect = RuntimeError("commit failed")
    with (
        patch.object(PostgresResource, "begin", side_effect=[setup, transaction]),
        pytest.raises(RuntimeError, match="commit failed"),
    ):
        manager.handle_output(context, pd.DataFrame({"id": [1]}))
    context.add_output_metadata.assert_not_called()


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "1000"])
def test_batch_size_is_strictly_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PostgresIOManager.model_validate(
            {"postgres_resource": _resource(), "batch_size": value}
        )


def test_rejects_unknown_outputs_and_inputs() -> None:
    manager, _, _, _ = _writer()
    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(TypeError, match="Unsupported"),
    ):
        manager.handle_output(_output_context(), MagicMock(spec=dict))
    begin.assert_not_called()
    with pytest.raises(NotImplementedError, match="output-only"):
        manager.load_input(MagicMock(spec=dg.InputContext))


@pytest.mark.parametrize("failure", ["lock", "create", "commit"])
def test_schema_setup_failure_prevents_table_write(failure: str) -> None:
    manager, conn, transaction, setup = _writer()
    context = _output_context()
    error = RuntimeError("setup failed")
    setup_conn = setup.__enter__.return_value
    if failure == "commit":
        setup.__exit__.side_effect = error
    else:
        setup_conn.execute.side_effect = [error] if failure == "lock" else [None, error]
    with (
        patch.object(
            PostgresResource, "begin", side_effect=[setup, transaction]
        ) as begin,
        pytest.raises(RuntimeError, match="setup failed"),
    ):
        manager.handle_output(context, pd.DataFrame({"id": [1]}))
    assert begin.call_count == 1
    transaction.__enter__.assert_not_called()
    conn.connection.driver_connection.cursor.assert_not_called()
    context.add_output_metadata.assert_not_called()


def test_schema_setup_commits_before_table_write() -> None:
    manager, conn, transaction, setup = _writer()
    setup_conn = setup.__enter__.return_value

    def enter_write() -> MagicMock:
        setup.__exit__.assert_called_once_with(None, None, None)
        return conn

    transaction.__enter__.side_effect = enter_write
    with patch.object(PostgresResource, "begin", side_effect=[setup, transaction]):
        manager.handle_output(
            _output_context({"schema": 'custom " schema', "table": "target"}),
            pd.DataFrame({"id": [1]}),
        )
    lock, create = setup_conn.execute.call_args_list
    assert str(lock.args[0]) == "SELECT pg_advisory_xact_lock(:lock_id)"
    assert lock.args[1] == {"lock_id": 2621279102301430195}
    assert str(create.args[0]) == 'CREATE SCHEMA IF NOT EXISTS "custom "" schema"'
    conn.execute.assert_not_called()

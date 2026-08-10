from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import dagster as dg
import geopandas as gpd
import pandas as pd
import pytest
import sqlalchemy
from shapely.geometry import Point
from sqlalchemy.dialects import postgresql

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import (
    POSTGRES_IDENTIFIER_MAX_LENGTH,
    PostgresResource,
)
from cfc_dagster_utils.types import (
    PostgresForeignKey,
    PostgresIndex,
    PostgresIndexMethod,
    PostgresRelation,
    PostgresTableSpec,
    PostgresWriteMode,
)

TEST_PASSWORD = "secret"  # noqa: S105 - inert test credential.
SPECIAL_TEST_PASSWORD = "p@ss/w:rd"  # noqa: S105 - URL-encoding fixture.


def _resource() -> PostgresResource:
    return PostgresResource(
        host="db.example",
        port=5432,
        user="user",
        password=TEST_PASSWORD,
        database="warehouse",
    )


def _connection() -> MagicMock:
    conn = MagicMock(spec=sqlalchemy.Connection)
    conn.dialect = postgresql.dialect()
    return conn


def _dagster_type(python_type: type) -> dg.DagsterType:
    return dg.DagsterType(
        type_check_fn=lambda _context, value: isinstance(value, python_type),
        name=f"Test{python_type.__name__}",
        typing_type=python_type,
    )


def _statement_strings(conn: MagicMock) -> list[str]:
    return [
        str(call.args[0].compile(dialect=conn.dialect)).strip()
        for call in conn.execute.call_args_list
    ]


def _spec(
    *,
    write_mode: PostgresWriteMode = PostgresWriteMode.CREATE,
) -> PostgresTableSpec:
    return PostgresTableSpec(
        relation=PostgresRelation(name="target", schema="analytics"),
        write_mode=write_mode,
        primary_key=("state", "id"),
        foreign_keys=(
            PostgresForeignKey(
                columns=("state", "parent_id"),
                referenced_relation=PostgresRelation(
                    name="parent",
                    schema="reference",
                ),
                referenced_columns=("state", "id"),
            ),
        ),
        indexes=(
            PostgresIndex(
                columns=("geometry",),
                method=PostgresIndexMethod.GIST,
            ),
        ),
        geometry_column="geometry",
    )


def test_table_spec_round_trip() -> None:
    spec = _spec(write_mode=PostgresWriteMode.REPLACE)

    assert PostgresTableSpec.from_dagster_metadata(spec.to_dagster_metadata()) == spec
    assert spec.relation.display_name == "analytics.target"


def test_table_spec_validates_composite_foreign_key() -> None:
    with pytest.raises(ValueError, match="equal length"):
        PostgresForeignKey(
            columns=("one", "two"),
            referenced_relation=PostgresRelation(name="parent"),
            referenced_columns=("one",),
        )


def test_table_spec_requires_dagster_metadata() -> None:
    with pytest.raises(TypeError, match="to_dagster_metadata"):
        PostgresTableSpec.from_dagster_metadata({})


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


def test_stage_dataframe_uses_geodataframe_handler_before_pandas() -> None:
    resource = _resource()
    conn = _connection()
    frame = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    target = PostgresRelation(name="a" * 100, schema="analytics")

    with (
        patch.object(gpd.GeoDataFrame, "to_postgis") as to_postgis,
        patch.object(pd.DataFrame, "to_sql") as to_sql,
        patch.object(PostgresResource, "relation_exists", return_value=False),
        resource.stage_dataframe(conn, frame, target) as stage,
    ):
        assert stage.schema == target.schema
        assert len(stage.name) <= POSTGRES_IDENTIFIER_MAX_LENGTH

    to_postgis.assert_called_once()
    to_sql.assert_not_called()


def test_stage_dataframe_cleans_up_unpublished_table() -> None:
    resource = _resource()
    conn = _connection()
    frame = pd.DataFrame({"id": [1]})

    with (
        patch.object(pd.DataFrame, "to_sql") as to_sql,
        patch.object(PostgresResource, "relation_exists", return_value=True),
        resource.stage_dataframe(
            conn,
            frame,
            PostgresRelation(name="target"),
        ),
    ):
        pass

    to_sql.assert_called_once()
    statements = _statement_strings(conn)
    assert len(statements) == 1
    assert statements[0].startswith("DROP TABLE IF EXISTS public._cfc_stage_")


def test_stage_dataframe_does_not_mask_transaction_failure() -> None:
    resource = _resource()
    conn = _connection()
    frame = pd.DataFrame({"id": [1]})

    def fail_server_side_transformation() -> None:
        with resource.stage_dataframe(
            conn,
            frame,
            PostgresRelation(name="target"),
        ):
            msg = "server-side transformation failed"
            raise RuntimeError(msg)

    with (
        patch.object(pd.DataFrame, "to_sql"),
        patch.object(PostgresResource, "relation_exists") as relation_exists,
        pytest.raises(RuntimeError, match="server-side transformation"),
    ):
        fail_server_side_transformation()

    relation_exists.assert_not_called()
    conn.execute.assert_not_called()


def test_publish_create_quotes_identifiers_and_applies_contract() -> None:
    resource = _resource()
    conn = _connection()
    source = PostgresRelation(name="stage table", schema="analytics")
    inspector = MagicMock()
    inspector.has_table.side_effect = [True, False]

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.publish_relation(conn, source, _spec())

    statements = _statement_strings(conn)
    assert statements[0] == ('ALTER TABLE analytics."stage table" RENAME TO target')
    assert "ADD PRIMARY KEY (state, id)" in statements[1]
    assert "REFERENCES reference.parent (state, id)" in statements[2]
    assert "USING gist (geometry)" in statements[3]


def test_publish_replace_drops_without_cascade() -> None:
    resource = _resource()
    conn = _connection()
    source = PostgresRelation(name="stage", schema="analytics")
    inspector = MagicMock()
    inspector.has_table.side_effect = [True, True]

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.publish_relation(
            conn,
            source,
            PostgresTableSpec(
                relation=PostgresRelation(name="target", schema="analytics"),
                write_mode=PostgresWriteMode.REPLACE,
            ),
        )

    statements = _statement_strings(conn)
    assert statements[0] == "DROP TABLE analytics.target"
    assert "CASCADE" not in statements[0]
    assert statements[1] == "ALTER TABLE analytics.stage RENAME TO target"


@pytest.mark.parametrize(
    ("write_mode", "expects_truncate"),
    [
        (PostgresWriteMode.APPEND, False),
        (PostgresWriteMode.TRUNCATE_INSERT, True),
    ],
)
def test_publish_into_existing_table(
    write_mode: PostgresWriteMode,
    *,
    expects_truncate: bool,
) -> None:
    resource = _resource()
    conn = _connection()
    source = PostgresRelation(name="stage", schema="analytics")
    inspector = MagicMock()
    inspector.has_table.side_effect = [True, True]
    inspector.get_columns.side_effect = [
        [{"name": "id"}, {"name": "select"}],
        [{"name": "select"}, {"name": "id"}],
    ]
    spec = PostgresTableSpec(
        relation=PostgresRelation(name="target", schema="analytics"),
        write_mode=write_mode,
    )

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.publish_relation(conn, source, spec)

    statements = _statement_strings(conn)
    assert any(statement.startswith("TRUNCATE TABLE") for statement in statements) is (
        expects_truncate
    )
    assert statements[-1] == (
        'INSERT INTO analytics.target ("select", id) '
        'SELECT "select", id FROM analytics.stage'
    )


def test_publish_rejects_incompatible_columns() -> None:
    resource = _resource()
    conn = _connection()
    inspector = MagicMock()
    inspector.has_table.side_effect = [True, True]
    inspector.get_columns.side_effect = [
        [{"name": "id"}],
        [{"name": "different"}],
    ]

    with (
        patch("sqlalchemy.inspect", return_value=inspector),
        pytest.raises(ValueError, match="identical columns"),
    ):
        resource.publish_relation(
            conn,
            PostgresRelation(name="stage", schema="analytics"),
            PostgresTableSpec(
                relation=PostgresRelation(name="target", schema="analytics"),
                write_mode=PostgresWriteMode.APPEND,
            ),
        )


def test_manager_returns_relation_without_connecting() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = _spec()
    upstream = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    context = dg.build_input_context(
        upstream_output=upstream,
        dagster_type=_dagster_type(PostgresRelation),
    )

    with patch.object(PostgresResource, "connect") as connect:
        result = manager.load_input(context)

    assert result == spec.relation
    connect.assert_not_called()


def test_manager_loads_dataframe_with_quoted_selected_columns() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = _spec()
    upstream = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    context = dg.build_input_context(
        upstream_output=upstream,
        definition_metadata={"columns": ["select", "id"]},
        dagster_type=_dagster_type(pd.DataFrame),
    )
    conn = _connection()
    connect = MagicMock(spec=AbstractContextManager)
    connect.__enter__.return_value = conn
    expected = pd.DataFrame({"select": [1], "id": [2]})

    with (
        patch.object(PostgresResource, "connect", return_value=connect),
        patch("pandas.read_sql", return_value=expected) as read_sql,
    ):
        result = manager.load_input(context)

    assert result is expected
    assert str(read_sql.call_args.args[0]) == (
        'SELECT "select", id FROM analytics.target'
    )


def test_manager_loads_geodataframe_with_configured_geometry() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = _spec()
    upstream = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    context = dg.build_input_context(
        upstream_output=upstream,
        definition_metadata={"columns": ["id", "geometry"]},
        dagster_type=_dagster_type(gpd.GeoDataFrame),
    )
    conn = _connection()
    connect = MagicMock(spec=AbstractContextManager)
    connect.__enter__.return_value = conn
    expected = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    with (
        patch.object(PostgresResource, "connect", return_value=connect),
        patch("geopandas.read_postgis", return_value=expected) as read_postgis,
    ):
        result = manager.load_input(context)

    assert result is expected
    assert read_postgis.call_args.kwargs["geom_col"] == "geometry"


def test_manager_rejects_unknown_input_type() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    upstream = dg.build_output_context(
        definition_metadata=_spec().to_dagster_metadata()
    )
    context = dg.build_input_context(
        upstream_output=upstream,
        dagster_type=_dagster_type(dict),
    )

    with pytest.raises(TypeError, match="must be annotated"):
        manager.load_input(context)


def test_manager_validates_already_published_relation() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = _spec()
    context = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    conn = _connection()
    connect = MagicMock(spec=AbstractContextManager)
    connect.__enter__.return_value = conn

    with (
        patch.object(PostgresResource, "connect", return_value=connect),
        patch.object(PostgresResource, "relation_exists", return_value=True),
    ):
        manager.handle_output(context, spec.relation)


def test_manager_rejects_mismatched_relation_output() -> None:
    manager = PostgresIOManager(postgres_resource=_resource())
    context = dg.build_output_context(definition_metadata=_spec().to_dagster_metadata())

    with pytest.raises(ValueError, match="does not match"):
        manager.handle_output(context, PostgresRelation(name="other"))


def test_manager_stages_and_publishes_frame() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = _spec()
    context = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    conn = _connection()
    stage = PostgresRelation(name="stage", schema="analytics")
    begin = MagicMock(spec=AbstractContextManager)
    begin.__enter__.return_value = conn
    staging = MagicMock(spec=AbstractContextManager)
    staging.__enter__.return_value = stage
    frame = pd.DataFrame({"id": [1]})

    with (
        patch.object(PostgresResource, "begin", return_value=begin),
        patch.object(
            PostgresResource,
            "stage_dataframe",
            return_value=staging,
        ) as stage_dataframe,
        patch.object(PostgresResource, "publish_relation") as publish_relation,
    ):
        manager.handle_output(context, frame)

    stage_dataframe.assert_called_once_with(conn, frame, spec.relation)
    publish_relation.assert_called_once_with(conn, stage, spec)

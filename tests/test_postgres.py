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
STAGING_STATEMENT_COUNT = 2


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
                columns=("parent_id",),
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


def test_table_spec_rejects_explicit_managed_geometry_index() -> None:
    with pytest.raises(ValueError, match="managed automatically"):
        PostgresTableSpec(
            relation=PostgresRelation(name="spatial"),
            indexes=(
                PostgresIndex(
                    columns=("geom",),
                    method=PostgresIndexMethod.GIST,
                ),
            ),
            geometry_column="geom",
        )


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
    inspector = MagicMock()
    inspector.get_indexes.return_value = [
        {
            "name": "idx_stage_geometry",
            "column_names": ["geometry"],
            "dialect_options": {"postgresql_using": "gist"},
        }
    ]

    with (
        patch.object(gpd.GeoDataFrame, "to_postgis") as to_postgis,
        patch.object(pd.DataFrame, "to_sql") as to_sql,
        patch("sqlalchemy.inspect", return_value=inspector),
        patch.object(PostgresResource, "relation_exists", return_value=False),
        resource.stage_dataframe(conn, frame, target) as stage,
    ):
        assert stage.schema == target.schema
        assert len(stage.name) <= POSTGRES_IDENTIFIER_MAX_LENGTH

    to_postgis.assert_called_once()
    to_sql.assert_not_called()
    assert _statement_strings(conn) == [
        "CREATE SCHEMA IF NOT EXISTS analytics",
        "DROP INDEX analytics.idx_stage_geometry",
    ]


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
    assert len(statements) == STAGING_STATEMENT_COUNT
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS public"
    assert statements[1].startswith("DROP TABLE IF EXISTS public._cfc_stage_")


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
    assert _statement_strings(conn) == ["CREATE SCHEMA IF NOT EXISTS public"]


def test_stage_query_creates_schema_table_and_cleans_up() -> None:
    resource = _resource()
    conn = _connection()

    with (
        patch.object(PostgresResource, "relation_exists", return_value=True),
        resource.stage_query(
            conn,
            "SELECT id FROM source",
            PostgresRelation(name="target", schema="analytics"),
        ) as stage,
    ):
        assert stage.schema == "analytics"

    statements = _statement_strings(conn)
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS analytics"
    assert statements[1].startswith(
        "CREATE TABLE analytics._cfc_stage_",
    )
    assert statements[1].endswith(" AS SELECT id FROM source")
    assert statements[2].startswith(
        "DROP TABLE IF EXISTS analytics._cfc_stage_",
    )


def test_stage_query_rejects_empty_query_without_creating_schema() -> None:
    resource = _resource()
    conn = _connection()

    with (
        pytest.raises(ValueError, match="must not be empty"),
        resource.stage_query(
            conn,
            "  \n ",
            PostgresRelation(name="target"),
        ),
    ):
        pass

    conn.execute.assert_not_called()


def test_materialize_query_uses_one_transaction_and_propagates_failure() -> None:
    resource = _resource()
    conn = _connection()
    stage = PostgresRelation(name="stage", schema="analytics")
    begin = MagicMock(spec=AbstractContextManager)
    begin.__enter__.return_value = conn
    staging = MagicMock(spec=AbstractContextManager)
    staging.__enter__.return_value = stage
    spec = _spec(write_mode=PostgresWriteMode.REPLACE)

    with (
        patch.object(PostgresResource, "begin", return_value=begin),
        patch.object(PostgresResource, "stage_query", return_value=staging),
        patch.object(
            PostgresResource,
            "publish_relation",
            side_effect=RuntimeError("publication failed"),
        ) as publish,
        pytest.raises(RuntimeError, match="publication failed"),
    ):
        resource.materialize_query("SELECT 1", spec)

    publish.assert_called_once_with(conn, stage, spec)
    assert begin.__exit__.call_args.args[0] is RuntimeError


def test_publish_create_quotes_identifiers_and_applies_contract() -> None:
    resource = _resource()
    conn = _connection()
    source = PostgresRelation(name="stage table", schema="analytics")
    inspector = MagicMock()
    inspector.has_table.side_effect = [True, False]
    inspector.get_indexes.return_value = []

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.publish_relation(conn, source, _spec())

    statements = _statement_strings(conn)
    assert statements[0] == ('ALTER TABLE analytics."stage table" RENAME TO target')
    assert "ADD PRIMARY KEY (state, id)" in statements[1]
    assert "REFERENCES reference.parent (state, id)" in statements[2]
    assert "USING gist (geometry)" in statements[3]
    assert "USING btree (parent_id)" in statements[4]


def test_ensure_geometry_index_preserves_matching_existing_index() -> None:
    resource = _resource()
    conn = _connection()
    inspector = MagicMock()
    inspector.get_indexes.return_value = [
        {
            "name": "existing_spatial_index",
            "column_names": ["geometry"],
            "dialect_options": {"postgresql_using": "gist"},
        }
    ]

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.ensure_geometry_index(conn, _spec())

    conn.execute.assert_not_called()


def test_ensure_geometry_index_creates_deterministic_index_when_missing() -> None:
    resource = _resource()
    conn = _connection()
    inspector = MagicMock()
    inspector.get_indexes.return_value = []

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.ensure_geometry_index(conn, _spec())

    assert _statement_strings(conn) == [
        (
            "CREATE INDEX target_geometry_gist_idx "
            "ON analytics.target USING gist (geometry)"
        )
    ]


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
        [{"name": "id"}, {"name": "select"}, {"name": "geometry"}],
        [{"name": "select"}, {"name": "geometry"}, {"name": "id"}],
    ]
    inspector.get_indexes.return_value = []
    spec = PostgresTableSpec(
        relation=PostgresRelation(name="target", schema="analytics"),
        write_mode=write_mode,
        geometry_column="geometry",
    )

    with patch("sqlalchemy.inspect", return_value=inspector):
        resource.publish_relation(conn, source, spec)

    statements = _statement_strings(conn)
    assert any(statement.startswith("TRUNCATE TABLE") for statement in statements) is (
        expects_truncate
    )
    assert statements[-2] == (
        'INSERT INTO analytics.target ("select", geometry, id) '
        'SELECT "select", geometry, id FROM analytics.stage'
    )
    assert statements[-1] == (
        "CREATE INDEX target_geometry_gist_idx "
        "ON analytics.target USING gist (geometry)"
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
    begin = MagicMock(spec=AbstractContextManager)
    begin.__enter__.return_value = conn

    with (
        patch.object(PostgresResource, "begin", return_value=begin),
        patch.object(PostgresResource, "relation_exists", return_value=True),
        patch.object(PostgresResource, "ensure_geometry_index") as ensure_index,
    ):
        manager.handle_output(context, spec.relation)

    ensure_index.assert_called_once_with(conn, spec)


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


def test_manager_infers_conventional_geometry_and_publishes_managed_spec() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    spec = PostgresTableSpec(
        relation=PostgresRelation(name="target", schema="analytics"),
    )
    context = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    frame = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    conn = _connection()
    stage = PostgresRelation(name="stage", schema="analytics")
    begin = MagicMock(spec=AbstractContextManager)
    begin.__enter__.return_value = conn
    staging = MagicMock(spec=AbstractContextManager)
    staging.__enter__.return_value = stage

    with (
        patch.object(PostgresResource, "begin", return_value=begin),
        patch.object(PostgresResource, "stage_dataframe", return_value=staging),
        patch.object(PostgresResource, "publish_relation") as publish_relation,
    ):
        manager.handle_output(context, frame)

    published_spec = publish_relation.call_args.args[2]
    assert published_spec.geometry_column == "geometry"


def test_manager_rejects_inferred_duplicate_geometry_index() -> None:
    manager = PostgresIOManager(postgres_resource=_resource())
    spec = PostgresTableSpec(
        relation=PostgresRelation(name="target"),
        indexes=(
            PostgresIndex(
                columns=("geometry",),
                method=PostgresIndexMethod.GIST,
            ),
        ),
    )
    context = dg.build_output_context(definition_metadata=spec.to_dagster_metadata())
    frame = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="managed automatically"):
        manager.handle_output(context, frame)

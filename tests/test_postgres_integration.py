"""Run with CFC_TEST_POSTGRES_URL pointing to a disposable PostGIS database."""

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import dagster as dg
import geopandas as gpd
import pandas as pd
import psycopg
import pytest
import sqlalchemy as sa
from shapely import LinearRing, Point, Polygon

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

pytestmark = pytest.mark.postgres
WGS84 = 4326
WEB_MERCATOR = 3857


@pytest.fixture
def database() -> Iterator[tuple[PostgresResource, str]]:
    url_text = os.environ.get("CFC_TEST_POSTGRES_URL")
    if not url_text:
        if os.environ.get("CFC_REQUIRE_POSTGRES_TESTS") == "1":
            pytest.fail("CFC_TEST_POSTGRES_URL is required")
        pytest.skip("Set CFC_TEST_POSTGRES_URL to run PostGIS integration tests")
    url = sa.make_url(url_text)
    resource = PostgresResource(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username or "postgres",
        password=url.password or "",
        database=url.database or "postgres",
    )
    schema = f'test " {uuid4().hex}'
    with dg.build_init_resource_context() as context:
        resource.setup_for_execution(context)
        try:
            with resource.begin() as conn:
                conn.execute(sa.text("SELECT PostGIS_Version()"))
            yield resource, schema
        finally:
            with resource.begin() as conn:
                conn.execute(sa.schema.DropSchema(schema, if_exists=True, cascade=True))
            resource.teardown_after_execution(context)


def _write(
    resource: PostgresResource, schema: str, frame: pd.DataFrame, table: str = "target"
) -> None:
    manager = PostgresIOManager(postgres_resource=resource, batch_size=2)
    with dg.build_output_context(
        definition_metadata={"schema": schema, "table": table}
    ) as context:
        manager.handle_output(context, frame)


def _relation(conn: sa.Connection, schema: str, table: str = "target") -> sa.Table:
    return sa.Table(table, sa.MetaData(), schema=schema, autoload_with=conn)


def test_scalar_round_trip_and_schema(database: tuple[PostgresResource, str]) -> None:
    resource, schema = database
    frame = pd.DataFrame(
        {
            "id": range(5),
            "integer": pd.Series([None, None, 2**63 - 1, -(2**63), 0], dtype="Int64"),
            "unsigned": pd.Series([None, None, 2**64 - 1, 0, 1], dtype="UInt64"),
            "amount": pd.Series(
                [None, None, 1.5, float("inf"), -float("inf")], dtype="Float64"
            ),
            "flag": pd.Series([None, None, True, False, True], dtype="boolean"),
            'text " column': pd.Series(
                [None, "", "001", "comma, tab\t newline\n", "á\\'"], dtype="string"
            ),
            "category": pd.Series([None, None, True, False, True], dtype="category"),
            "object": pd.Series([None, None, Decimal("1.23"), 42, "a"], dtype="object"),
            "timestamp": pd.to_datetime(
                [None, None, "2020-01-01 00:00:00.000001999", None, None]
            ),
            "zoned": pd.to_datetime(
                [None, None, "2020-01-01 00:00:00+02:00", None, None], utc=True
            ),
            "duration": pd.to_timedelta([None, None, 123, -123, 0], unit="ns"),
        }
    )
    _write(resource, schema, frame, 'target " table')
    with resource.connect() as conn:
        relation = _relation(conn, schema, 'target " table')
        rows = conn.execute(sa.select(relation).order_by(relation.c.id)).all()
        types = {
            col["name"]: str(col["type"])
            for col in sa.inspect(conn).get_columns('target " table', schema=schema)
        }
    assert rows[0] == (0, None, None, None, None, None, None, None, None, None, None)
    assert rows[1][5] == ""
    assert rows[2] == (
        2,
        2**63 - 1,
        Decimal(2**64 - 1),
        1.5,
        True,
        "001",
        "True",
        "1.23",
        datetime(2020, 1, 1, microsecond=1, tzinfo=UTC).replace(tzinfo=None),
        datetime(2019, 12, 31, 22, tzinfo=UTC),
        123,
    )
    assert rows[3][1] == -(2**63)
    assert rows[3][3] == float("inf")
    assert rows[3][5] == "comma, tab\t newline\n"
    assert rows[4][5] == "á\\'"
    assert types["integer"] == "BIGINT"
    assert types["unsigned"] == "NUMERIC(20, 0)"
    assert types["object"] == "TEXT"
    assert types["category"] == "TEXT"


def test_geometry_round_trip(database: tuple[PostgresResource, str]) -> None:
    resource, schema = database
    frame = gpd.GeoDataFrame(
        {"id": range(6)},
        geometry=[
            None,
            None,
            Point(1, 2, 3),
            Polygon(),
            LinearRing([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(0, 0), (1, 0), (0, 1)]),
        ],
        crs=WGS84,
    )
    frame["other"] = gpd.GeoSeries([Point(5, 6)] * len(frame), crs=WEB_MERCATOR)
    original = frame.copy(deep=True)
    _write(resource, schema, frame)
    with resource.connect() as conn:
        relation = _relation(conn, schema)
        rows = conn.execute(
            sa.select(
                sa.func.ST_AsEWKT(relation.c.geometry),
                sa.func.ST_SRID(relation.c.geometry),
                sa.func.ST_NDims(relation.c.geometry),
                sa.func.ST_SRID(relation.c.other),
            ).order_by(relation.c.id)
        ).all()
    assert rows[0] == (None, None, None, WEB_MERCATOR)
    assert rows[2] == ("SRID=4326;POINT(1 2 3)", WGS84, 3, WEB_MERCATOR)
    assert rows[3][0] == "SRID=4326;POLYGON EMPTY"
    assert rows[4][0] == "SRID=4326;LINESTRING(0 0,1 0,0 1,0 0)"
    assert rows[5][0] == "SRID=4326;POLYGON((0 0,1 0,0 1,0 0))"
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize("failure", ["conversion", "database"])
def test_later_batch_failure_rolls_back(
    database: tuple[PostgresResource, str], *, present: bool, failure: str
) -> None:
    resource, schema = database
    if present:
        _write(resource, schema, pd.DataFrame({"original": [42]}))
    invalid = [1] if failure == "conversion" else "contains\x00nul"
    frame = pd.DataFrame({"value": pd.Series(["a", "b", invalid], dtype="object")})
    with pytest.raises((TypeError, psycopg.Error)):
        _write(resource, schema, frame)
    with resource.connect() as conn:
        assert sa.inspect(conn).has_table("target", schema=schema) is present
        if present:
            relation = _relation(conn, schema)
            assert conn.execute(sa.select(relation)).all() == [(42,)]


@pytest.mark.parametrize("geometry", [True, False])
def test_empty_replacement(
    database: tuple[PostgresResource, str], *, geometry: bool
) -> None:
    resource, schema = database
    _write(resource, schema, pd.DataFrame({"old": [1]}))
    frame = pd.DataFrame({"id": pd.Series(dtype="Int64")})
    if geometry:
        frame = gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries([], crs=WGS84))
    _write(resource, schema, frame)
    with resource.connect() as conn:
        relation = _relation(conn, schema)
        assert (
            conn.execute(sa.select(sa.func.count()).select_from(relation)).scalar_one()
            == 0
        )
        assert sa.inspect(conn).get_columns("target", schema=schema)[0]["name"] == "id"


def test_all_null_geometry(database: tuple[PostgresResource, str]) -> None:
    resource, schema = database
    _write(resource, schema, gpd.GeoDataFrame(geometry=[None, None, None], crs=WGS84))
    with resource.connect() as conn:
        relation = _relation(conn, schema)
        assert conn.execute(sa.select(relation.c.geometry)).all() == [(None,)] * 3


def test_concurrent_writes(database: tuple[PostgresResource, str]) -> None:
    resource, schema = database
    # Schema creation races are distinct from independent table writes.
    with resource.begin() as conn:
        conn.execute(sa.schema.CreateSchema(schema))

    manager = PostgresIOManager(postgres_resource=resource, batch_size=2)
    frame = pd.DataFrame({"id": range(100)})
    with ExitStack() as stack:
        contexts = [
            stack.enter_context(
                dg.build_output_context(
                    definition_metadata={"schema": schema, "table": table}
                )
            )
            for table in ("one", "two")
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(manager.handle_output, context, frame)
                for context in contexts
            ]
            for future in futures:
                future.result()
    with resource.connect() as conn:
        for table in ("one", "two"):
            relation = _relation(conn, schema, table)
            assert conn.execute(
                sa.select(relation.c.id).order_by(relation.c.id)
            ).scalars().all() == list(range(100))

from datetime import UTC, datetime
from decimal import Decimal

import geopandas as gpd
import pandas as pd
import pytest
from shapely import LinearRing, Point, Polygon, from_wkb, get_srid, has_z
from sqlalchemy.dialects import postgresql

from cfc_dagster_utils.managers._postgres import converted_rows, frame_columns

WGS84 = 4326
WEB_MERCATOR = 3857


@pytest.mark.parametrize(
    ("dtype", "values", "sql_type", "expected"),
    [
        ("boolean", [True, None], "BOOLEAN", [(True,), (None,)]),
        ("bool", [True, False], "BOOLEAN", [(True,), (False,)]),
        ("Int32", [2**31 - 1, None], "BIGINT", [(2**31 - 1,), (None,)]),
        ("Int64", [2**63 - 1, None], "BIGINT", [(2**63 - 1,), (None,)]),
        ("uint32", [2**32 - 1], "BIGINT", [(2**32 - 1,)]),
        ("UInt64", [2**64 - 1, None], "NUMERIC(20, 0)", [(2**64 - 1,), (None,)]),
        (
            "Float64",
            [1.5, None, float("inf")],
            "DOUBLE PRECISION",
            [(1.5,), (None,), (float("inf"),)],
        ),
        ("float32", [1.5, float("nan")], "DOUBLE PRECISION", [(1.5,), (None,)]),
        ("string", ["001", "", None], "TEXT", [("001",), ("",), (None,)]),
        ("object", [Decimal("1.23"), 42, None], "TEXT", [("1.23",), ("42",), (None,)]),
        ("category", ["a", "b", None], "TEXT", [("a",), ("b",), (None,)]),
        ("category", [True, False, None], "TEXT", [("True",), ("False",), (None,)]),
        ("category", [1, 2, None], "TEXT", [("1",), ("2",), (None,)]),
        (
            "datetime64[ns]",
            [pd.Timestamp("2020-01-01 00:00:00.000001999"), None],
            "TIMESTAMP WITHOUT TIME ZONE",
            [
                (datetime(2020, 1, 1, microsecond=1, tzinfo=UTC).replace(tzinfo=None),),
                (None,),
            ],
        ),
        (
            "datetime64[ns, UTC]",
            [pd.Timestamp("2020-01-01", tz="UTC"), None],
            "TIMESTAMP WITH TIME ZONE",
            [(datetime(2020, 1, 1, tzinfo=UTC),), (None,)],
        ),
        (
            "timedelta64[ns]",
            [pd.Timedelta(123, unit="ns"), None],
            "BIGINT",
            [(123,), (None,)],
        ),
    ],
)
def test_dtype_mapping_and_conversion(
    dtype: str, values: list[object], sql_type: str, expected: list[tuple[object, ...]]
) -> None:
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    columns = frame_columns(frame)
    assert str(columns[0].sql_type.compile(dialect=postgresql.dialect())) == sql_type
    assert list(converted_rows(frame, columns)) == expected


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean", "string", "object"])
def test_schema_does_not_depend_on_first_batch(dtype: str) -> None:
    frame = pd.DataFrame({"value": pd.Series([None, None], dtype=dtype)})
    assert (
        frame_columns(frame)[0].sql_type.__class__
        is frame_columns(frame.iloc[:0])[0].sql_type.__class__
    )
    assert list(converted_rows(frame, frame_columns(frame))) == [(None,), (None,)]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2]], columns=["a", "a"]),
        pd.DataFrame({"": [1]}),
        pd.DataFrame({1: [1]}),
    ],
)
def test_invalid_columns(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="column"):
        frame_columns(frame)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1j]),
        pd.Series(pd.period_range("2020", periods=2, freq="M")),
    ],
)
def test_unsupported_dtypes(series: pd.Series) -> None:
    with pytest.raises(TypeError, match=r"Column 'bad'.*unsupported dtype"):
        frame_columns(pd.DataFrame({"bad": series}))


@pytest.mark.parametrize("value", [[1], {"a": 1}, (1, 2), {1}, 1j])
def test_nested_and_complex_values_fail_with_column_name(value: object) -> None:
    frame = pd.DataFrame({"bad": pd.Series([value], dtype="object")})
    with pytest.raises(TypeError, match=r"Column 'bad'.*unsupported value"):
        list(converted_rows(frame, frame_columns(frame)))


def test_geometry_types_dimensions_nulls_and_crs() -> None:
    frame = gpd.GeoDataFrame(
        {"id": range(6)},
        geometry=[
            Point(1, 2),
            Point(1, 2, 3),
            None,
            Polygon(),
            LinearRing([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(0, 0), (1, 0), (0, 1)]),
        ],
        crs=4326,
    )
    frame["other"] = gpd.GeoSeries([Point(5, 6)] * len(frame), crs=3857)
    original = frame.copy(deep=True)
    columns = frame_columns(frame)
    rows = list(converted_rows(frame, columns))
    geometries = [from_wkb(row[1]) if row[1] is not None else None for row in rows]
    assert geometries[0] is not None
    assert geometries[3] is not None
    assert geometries[4] is not None
    assert geometries[5] is not None
    assert geometries[0].equals(Point(1, 2))
    assert has_z(geometries[1])
    assert geometries[2] is None
    assert geometries[3].is_empty
    assert geometries[4].geom_type == "LineString"
    assert geometries[5].geom_type == "Polygon"
    assert get_srid(geometries[0]) == WGS84
    assert get_srid(from_wkb(rows[0][2])) == WEB_MERCATOR
    pd.testing.assert_frame_equal(frame, original)
    assert str(columns[1].sql_type.compile(dialect=postgresql.dialect())) == "geometry"


def test_geometry_without_crs_warns_and_uses_zero() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(1, 2)])
    with pytest.warns(UserWarning, match="SRID 0"):
        columns = frame_columns(frame)
    assert get_srid(from_wkb(next(converted_rows(frame, columns))[0])) == 0


def test_empty_and_all_null_geometry() -> None:
    for geometries in ([], [None, None]):
        frame = gpd.GeoDataFrame(geometry=geometries, crs=4326)
        assert list(converted_rows(frame, frame_columns(frame))) == [(None,)] * len(
            geometries
        )


def test_esri_authority_identifier() -> None:
    esri_srid = 102003
    frame = gpd.GeoDataFrame(geometry=[Point(1, 2)], crs=f"ESRI:{esri_srid}")
    columns = frame_columns(frame)
    assert columns[0].srid == esri_srid
    assert get_srid(from_wkb(next(converted_rows(frame, columns))[0])) == esri_srid


def test_unresolvable_crs_warns() -> None:
    frame = gpd.GeoDataFrame(
        geometry=[Point(1, 2)], crs="+proj=aeqd +lat_0=12.34 +lon_0=56.78 +datum=WGS84"
    )
    with pytest.warns(UserWarning, match="SRID 0"):
        columns = frame_columns(frame)
    assert columns[0].srid == 0

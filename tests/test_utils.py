import math
from typing import Literal, assert_type, cast

import geopandas as gpd
import geopandas.testing as gpd_testing
import pandas as pd
import pytest

from cfc_dagster_utils.utils import cast_all_columns_to_numeric


def _typecheck_cast_all_columns_to_numeric(
    dataframe: pd.DataFrame,
    geodataframe: gpd.GeoDataFrame,
) -> None:
    assert_type(
        dataframe.pipe(cast_all_columns_to_numeric),
        pd.DataFrame,
    )
    assert_type(
        dataframe.pipe(
            cast_all_columns_to_numeric,
            ignore=["identifier"],
            errors="coerce",
            prefer_integer=True,
        ),
        pd.DataFrame,
    )
    assert_type(
        cast_all_columns_to_numeric(geodataframe),
        gpd.GeoDataFrame,
    )


def test_cast_all_columns_to_numeric_returns_converted_copy() -> None:
    dataframe = pd.DataFrame(
        {
            "identifier": ["001", "002"],
            "integer": ["1", "2"],
            "decimal": ["1.5", "2.5"],
        },
    )
    original = dataframe.copy()

    result = cast_all_columns_to_numeric(
        dataframe,
        ignore=["identifier"],
        prefer_integer=True,
    )

    expected = pd.DataFrame(
        {
            "identifier": ["001", "002"],
            "integer": pd.Series([1, 2], dtype="Int64"),
            "decimal": pd.Series([1.5, 2.5], dtype="Float64"),
        },
    )
    pd.testing.assert_frame_equal(result, expected)
    assert result is not dataframe
    pd.testing.assert_frame_equal(dataframe, original)


def test_cast_all_columns_to_numeric_coerces_invalid_values() -> None:
    dataframe = pd.DataFrame({"value": ["1", "invalid"]})

    result = dataframe.pipe(cast_all_columns_to_numeric, errors="coerce")

    expected = pd.DataFrame({"value": pd.Series([1, None], dtype="Int64")})
    pd.testing.assert_frame_equal(result, expected)


def test_cast_all_columns_to_numeric_raises_for_invalid_values() -> None:
    dataframe = pd.DataFrame({"value": ["1", "invalid"]})

    with pytest.raises(ValueError, match="Unable to parse string"):
        cast_all_columns_to_numeric(dataframe)


def test_cast_all_columns_to_numeric_preserves_geodataframe() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"value": ["1", "2"]},
        geometry=gpd.points_from_xy([0, 1], [2, 3]),
        crs="EPSG:4326",
    )

    result = cast_all_columns_to_numeric(
        geodataframe,
        ignore=["geometry"],
        prefer_integer=True,
    )

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == geodataframe.crs
    pd.testing.assert_series_equal(
        result["value"], pd.Series([1, 2], name="value", dtype="Int64")
    )
    gpd_testing.assert_geoseries_equal(result.geometry, geodataframe.geometry)


@pytest.mark.parametrize("prefer_integer", [True, False])
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["9007199254740993", None], None),
        ([str(-(2**63)), str(2**63 - 1), None], None),
        ([-(2**63), 2**63 - 1, None], "Int64"),
        ([1, None], "Int32"),
    ],
)
def test_integer_precision_and_missing_values(
    values: list[str | int | None], dtype: str | None, *, prefer_integer: bool
) -> None:
    dataframe = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    expected = pd.Series(values, name="value", dtype="Int64")

    result = cast_all_columns_to_numeric(dataframe, prefer_integer=prefer_integer)

    pd.testing.assert_series_equal(result["value"], expected)


@pytest.mark.parametrize("errors", ["raise", "coerce"])
@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_integer_overflow(value: int, errors: Literal["raise", "coerce"]) -> None:
    dataframe = pd.DataFrame({"count": [str(value), None]})
    original = dataframe.copy()

    with pytest.raises(OverflowError, match=r"Column 'count'.*Int64"):
        cast_all_columns_to_numeric(dataframe, errors=errors)

    pd.testing.assert_frame_equal(dataframe, original)


@pytest.mark.parametrize("prefer_integer", [True, False])
@pytest.mark.parametrize(
    ("values", "can_be_integer"),
    [
        ([1.0, None], True),
        ([float(-(2**63)), None], True),
        ([math.nextafter(float(2**63), -math.inf), None], True),
        ([float(2**63), None], False),
        ([math.nextafter(float(-(2**63)), -math.inf), None], False),
        ([1e20, None], False),
        ([1.5, None], False),
        ([math.nextafter(1.0, 2.0), None], False),
        ([math.inf, None], False),
        ([-math.inf, None], False),
    ],
)
def test_float_integer_inference(
    values: list[float | None], *, can_be_integer: bool, prefer_integer: bool
) -> None:
    dataframe = pd.DataFrame({"value": pd.Series(values, dtype="Float64")})
    integer_output = can_be_integer and prefer_integer
    expected_values = (
        [int(value) if value is not None else None for value in values]
        if integer_output
        else values
    )
    expected = pd.Series(
        expected_values,
        name="value",
        dtype="Int64" if integer_output else "Float64",
    )

    result = cast_all_columns_to_numeric(dataframe, prefer_integer=prefer_integer)

    pd.testing.assert_series_equal(result["value"], expected)


@pytest.mark.parametrize("prefer_integer", [True, False])
def test_float_strings(*, prefer_integer: bool) -> None:
    dataframe = pd.DataFrame({"value": ["1.0", None]})

    result = cast_all_columns_to_numeric(dataframe, prefer_integer=prefer_integer)

    expected = pd.Series(
        [1, None], name="value", dtype="Int64" if prefer_integer else "Float64"
    )
    pd.testing.assert_series_equal(result["value"], expected)


@pytest.mark.parametrize("values", [[], [None, None]])
@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("Int32", "Int64"),
        ("UInt64", "Int64"),
        ("Float32", "Float64"),
        ("float64", "Float64"),
        ("object", "Float64"),
        ("string", "Float64"),
    ],
)
def test_empty_and_all_missing_columns(
    values: list[None], dtype: str, expected_dtype: str
) -> None:
    dataframe = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})

    result = cast_all_columns_to_numeric(dataframe)

    expected = pd.Series(values, name="value", dtype=expected_dtype)
    pd.testing.assert_series_equal(result["value"], expected)


def test_all_invalid_values_become_nullable_float() -> None:
    dataframe = pd.DataFrame({"value": ["bad", "invalid"]})

    result = cast_all_columns_to_numeric(dataframe, errors="coerce")

    expected = pd.Series([None, None], name="value", dtype="Float64")
    pd.testing.assert_series_equal(result["value"], expected)


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_preserves_booleans(dtype: str) -> None:
    values = [True, False, None] if dtype == "boolean" else [True, False]
    dataframe = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})

    result = cast_all_columns_to_numeric(dataframe)

    pd.testing.assert_frame_equal(result, dataframe)
    assert result is not dataframe


@pytest.mark.parametrize("errors", ["raise", "coerce"])
@pytest.mark.parametrize(
    "column",
    [
        pd.Series(pd.to_datetime(["2026-01-01", None])),
        pd.Series(pd.to_datetime(["2026-01-01", None], utc=True)),
        pd.Series(pd.to_timedelta([1, None], unit="s")),
        pd.Series([1 + 2j, 3 + 0j]),
        pd.Series([1 + 2j, 3 + 0j], dtype=object),
        gpd.GeoSeries(gpd.points_from_xy([0, 1], [2, 3])),
    ],
)
def test_unsupported_columns_require_ignore(
    column: pd.Series, errors: Literal["raise", "coerce"]
) -> None:
    dataframe = pd.DataFrame({"unsupported": column})

    with pytest.raises(TypeError, match=r"Column 'unsupported'.*ignore"):
        cast_all_columns_to_numeric(dataframe, errors=errors)

    result = cast_all_columns_to_numeric(
        dataframe, ignore=["unsupported"], errors=errors
    )
    pd.testing.assert_frame_equal(result, dataframe)


@pytest.mark.parametrize("empty", [True, False])
def test_rejects_duplicate_labels(*, empty: bool) -> None:
    dataframe = pd.DataFrame([] if empty else [["1", "2"]], columns=["x", "x"])

    with pytest.raises(ValueError, match="Duplicate column labels"):
        cast_all_columns_to_numeric(dataframe, ignore=["x"])


def test_validates_errors_on_empty_dataframe() -> None:
    invalid_errors = cast("Literal['raise', 'coerce']", "invalid")

    with pytest.raises(ValueError, match="errors must be"):
        cast_all_columns_to_numeric(pd.DataFrame(), errors=invalid_errors)


def test_preserves_index_column_order_and_ignored_data() -> None:
    dataframe = pd.DataFrame(
        {"value": ["1", None], "identifier": ["001", "002"]},
        index=pd.Index(["b", "b"], name="row"),
    )
    original = dataframe.copy()
    expected = dataframe.copy()
    expected["value"] = pd.Series([1, None], index=dataframe.index, dtype="Int64")

    result = cast_all_columns_to_numeric(dataframe, ignore=["identifier"])

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(dataframe, original)


def test_parsing_error_identifies_column() -> None:
    dataframe = pd.DataFrame({"amount": ["invalid"]})

    with pytest.raises(ValueError, match=r"Column 'amount'.*Unable to parse"):
        cast_all_columns_to_numeric(dataframe)

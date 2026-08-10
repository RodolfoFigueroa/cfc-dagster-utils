from typing import assert_type

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
            make_valid_int=True,
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
        make_valid_int=True,
    )

    expected = pd.DataFrame(
        {
            "identifier": ["001", "002"],
            "integer": [1, 2],
            "decimal": [1.5, 2.5],
        },
    )
    pd.testing.assert_frame_equal(result, expected)
    assert result is not dataframe
    pd.testing.assert_frame_equal(dataframe, original)


def test_cast_all_columns_to_numeric_coerces_invalid_values() -> None:
    dataframe = pd.DataFrame({"value": ["1", "invalid"]})

    result = dataframe.pipe(cast_all_columns_to_numeric, errors="coerce")

    expected = pd.DataFrame({"value": [1.0, float("nan")]})
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
        make_valid_int=True,
    )

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == geodataframe.crs
    pd.testing.assert_series_equal(result["value"], pd.Series([1, 2], name="value"))
    gpd_testing.assert_geoseries_equal(result.geometry, geodataframe.geometry)

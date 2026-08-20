from collections.abc import Sequence
from typing import Literal, overload

import geopandas as gpd
import pandas as pd


@overload
def cast_all_columns_to_numeric(
    df: gpd.GeoDataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    make_valid_int: bool = False,
    int_type: str = "int",
    float_type: str = "float",
) -> gpd.GeoDataFrame: ...


@overload
def cast_all_columns_to_numeric(
    df: pd.DataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    make_valid_int: bool = False,
    int_type: str = "int",
    float_type: str = "float",
) -> pd.DataFrame: ...


def cast_all_columns_to_numeric(  # noqa: PLR0913
    df: pd.DataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    make_valid_int: bool = False,
    int_type: str = "int",
    float_type: str = "float",
) -> pd.DataFrame:
    """Convert all columns in a DataFrame to numeric types.

    Attempts to cast all columns in a DataFrame to numeric types, with options
    to ignore specific columns and handle conversion errors.

    Args:
        df: The input DataFrame to convert.
        ignore: Column names to skip during conversion. Defaults to None.
        errors: How to handle conversion failures. ``"raise"`` raises an
            exception; ``"coerce"`` converts invalid values to NaN.
            Defaults to ``"raise"``.
        make_valid_int: If True, convert columns whose values are all whole
            numbers (and non-null) to int type. Defaults to False.

    Returns:
        A new DataFrame with converted columns. The original DataFrame is
        not modified.

    Note:
        A copy of the input DataFrame is created before modifications. If
        ``make_valid_int`` is True, columns are converted to int only if they
        contain no missing values and all numeric values are whole numbers.

    Examples:
        >>> df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['1.5', '2.5', '3.5']})
        >>> cast_all_columns_to_numeric(df)
           A    B
        0  1  1.5
        1  2  2.5
        2  3  3.5
        >>> cast_all_columns_to_numeric(df, ignore=['B'], make_valid_int=True)
           A    B
        0  1  1.5
        1  2  2.5
        2  3  3.5
    """

    if ignore is None:
        ignore = []

    df = df.copy()
    for col in df.columns:
        if col not in ignore:
            new_col = pd.to_numeric(df[col], errors=errors)
            if (
                make_valid_int
                and new_col.notna().all()
                and (new_col.to_numpy() % 1 == 0).all()
            ):
                new_col = new_col.astype(int_type)

            if pd.api.types.is_float_dtype(new_col):
                new_col = new_col.astype(float_type)

            df[col] = new_col

    return df

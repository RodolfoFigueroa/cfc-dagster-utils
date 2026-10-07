from collections.abc import Sequence
from typing import Literal, overload

import geopandas as gpd
import pandas as pd


def _numeric_column(
    column: pd.Series,
    *,
    errors: Literal["coerce", "raise"],
    prefer_integer: bool,
    integer_dtype: Literal["Int32", "Int64"],
) -> pd.Series:
    integer_stop = 2**31 if integer_dtype == "Int32" else 2**63
    integer_min = -integer_stop
    dtype = column.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return column
    if (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        or isinstance(dtype, gpd.array.GeometryDtype)
        or any(pd.api.types.is_complex(value) for value in column)
    ):
        msg = f"Unsupported dtype {dtype}; explicitly ignore this column"
        raise TypeError(msg)

    # Parse present values separately: pandas can mishandle unsigned strings
    # with nulls and retain unmasked NaNs in nullable floating-point results.
    # A temporary unique index makes restoration safe for duplicate row labels.
    values = column.reset_index(drop=True)
    numeric = pd.to_numeric(
        values.dropna(), errors=errors, dtype_backend="numpy_nullable"
    ).reindex(values.index)
    numeric.index = column.index
    if pd.api.types.is_complex_dtype(numeric.dtype):
        msg = "Complex values are unsupported; explicitly ignore this column"
        raise TypeError(msg)

    present = numeric.dropna()
    if present.empty:
        target = integer_dtype if pd.api.types.is_integer_dtype(dtype) else "Float64"
        return numeric.astype(target)

    if pd.api.types.is_integer_dtype(numeric.dtype):
        if int(present.min()) < integer_min or int(present.max()) >= integer_stop:
            msg = f"Integer values are outside the {integer_dtype} range"
            raise OverflowError(msg)
        return numeric.astype(integer_dtype)

    if (
        prefer_integer
        and present.ge(integer_min).all()
        and present.lt(integer_stop).all()
        and present.mod(1).eq(0).all()
    ):
        return numeric.astype(integer_dtype)
    return numeric.astype("Float64")


@overload
def cast_all_columns_to_numeric(
    df: gpd.GeoDataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    prefer_integer: bool = True,
    integer_dtype: Literal["Int32", "Int64"] = "Int32",
) -> gpd.GeoDataFrame: ...


@overload
def cast_all_columns_to_numeric(
    df: pd.DataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    prefer_integer: bool = True,
    integer_dtype: Literal["Int32", "Int64"] = "Int32",
) -> pd.DataFrame: ...


def cast_all_columns_to_numeric(
    df: pd.DataFrame,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    prefer_integer: bool = True,
    integer_dtype: Literal["Int32", "Int64"] = "Int32",
) -> pd.DataFrame:
    """Return a copy with numeric columns using nullable integers or Float64.

    Args:
        df: Input DataFrame or GeoDataFrame, with unique column labels.
        ignore: Column names to preserve without conversion.
        errors: Pandas parsing policy. ``"raise"`` rejects invalid values;
            ``"coerce"`` replaces them with missing values. Structural errors
            and integer overflow always raise.
        prefer_integer: Convert floating-point results to integer_dtype when every
            non-missing value is finite, exactly integral, and in range.
            If False, floating-point results stay Float64; parsed integers
            still become integer_dtype.
        integer_dtype: Nullable integer type, either "Int32" (the default) or
            "Int64". Int32 supports values from -2**31 through 2**31 - 1;
            Int64 supports values from -2**63 through 2**63 - 1.

    Returns:
        A new DataFrame preserving the index, column order, and subclass
        metadata. Missing numeric values use pd.NA. Boolean and ignored
        columns retain their original dtypes. Empty or entirely missing
        columns retain their integer/float family, normalized to integer_dtype or
        Float64; columns without a numeric input dtype become Float64.

    Raises:
        ValueError: Invalid errors policy or integer_dtype, duplicate labels,
            or failed parsing.
        TypeError: Unsupported column types, including datetime, timedelta,
            geometry, and complex columns, which must be explicitly ignored.
        OverflowError: Parsed integer values cannot be represented by integer_dtype.
            Out-of-range floating-point values remain Float64 instead.

    Notes:
        Parsing uses pandas' nullable backend to avoid routing integer strings
        with missing values through floats. Floating-point parsing still has
        pandas' precision limits: arbitrary-precision decimals are not supported,
        and precision already lost in input floats cannot be recovered.
        Conversion errors identify the affected column.

    Examples:
        >>> df = pd.DataFrame({'A': ['1', None], 'B': ['1.5', '2.5']})
        >>> cast_all_columns_to_numeric(df)
              A    B
        0     1  1.5
        1  <NA>  2.5
        >>> cast_all_columns_to_numeric(df).dtypes.astype(str).to_dict()
        {'A': 'Int32', 'B': 'Float64'}
        >>> df = pd.DataFrame({'A': ['1.0', None]})
        >>> cast_all_columns_to_numeric(df, prefer_integer=False)['A'].dtype
        Float64Dtype()
    """
    if errors not in ("raise", "coerce"):
        msg = "errors must be 'raise' or 'coerce'"
        raise ValueError(msg)
    if integer_dtype not in ("Int32", "Int64"):
        msg = "integer_dtype must be 'Int32' or 'Int64'"
        raise ValueError(msg)
    if not df.columns.is_unique:
        msg = "Duplicate column labels are unsupported"
        raise ValueError(msg)

    ignored = set(ignore or ())
    result = df.copy()
    for name in result.columns:
        if name in ignored:
            continue
        try:
            result[name] = _numeric_column(
                result[name],
                errors=errors,
                prefer_integer=prefer_integer,
                integer_dtype=integer_dtype,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            msg = f"Column {name!r}: {exc}"
            raise type(exc)(msg) from exc
    return result

"""Dtype-based schemas and bounded conversion for PostgreSQL COPY."""

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import pandas as pd
import sqlalchemy as sa
from pandas.api import types as pt
from shapely import LinearRing, LineString, set_srid, to_wkb

from cfc_dagster_utils._optional import raise_optional_dependency_error

try:
    from geoalchemy2 import Geometry
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="geoalchemy2",
        dependency_name="geoalchemy2",
        extra="postgres",
    )


UINT64_BYTES = 8


def text_value(value: object) -> str:
    return str(value)


def timestamp_value(value: object) -> object:
    if not isinstance(value, datetime):
        msg = "Expected a datetime scalar"
        raise TypeError(msg)
    return pd.Timestamp(value).as_unit("us").to_pydatetime()


def duration_value(value: object) -> int:
    if not isinstance(value, timedelta):
        msg = "Expected a timedelta scalar"
        raise TypeError(msg)
    return pd.Timedelta(value).value


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: sa.types.TypeEngine
    convert: Callable[[Any], object]
    srid: int | None = None


def geometry_srid(series: gpd.GeoSeries) -> int:
    if series.crs is not None:
        # An approximate authority match can label unchanged coordinates incorrectly.
        srid = series.crs.to_epsg(min_confidence=100)
        if srid is not None:
            return srid
        authority = series.crs.to_authority(auth_name="ESRI", min_confidence=100)
        if authority is not None:
            return int(authority[1])
    warnings.warn(
        f"Column {series.name!r}: CRS has no authority identifier; writing SRID 0",
        UserWarning,
        stacklevel=2,
    )
    return 0


def column_spec(name: str, series: pd.Series) -> Column:
    dtype = series.dtype
    if isinstance(dtype, gpd.array.GeometryDtype):
        return Column(
            name,
            Geometry(geometry_type=None, srid=-1, spatial_index=False),
            text_value,
            geometry_srid(gpd.GeoSeries(series)),
        )
    if (
        isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype))
        or pt.is_object_dtype(dtype)
        or pt.is_string_dtype(dtype)
    ):
        return Column(name, sa.Text(), text_value)
    for predicate, sql_type, convert in (
        (pt.is_bool_dtype, sa.Boolean(), bool),
        (pt.is_float_dtype, sa.Double(), float),
    ):
        if predicate(dtype):
            return Column(name, sql_type, convert)
    if pt.is_integer_dtype(dtype):
        sql_type = (
            sa.Numeric(20, 0)
            if pt.is_unsigned_integer_dtype(dtype)
            and getattr(dtype, "itemsize", None) == UINT64_BYTES
            else sa.BigInteger()
        )
        return Column(name, sql_type, int)
    if pt.is_datetime64_any_dtype(dtype):
        timezone = getattr(dtype, "tz", None)
        if timezone is None:
            timezone = getattr(getattr(dtype, "pyarrow_dtype", None), "tz", None)
        return Column(
            name, sa.TIMESTAMP(timezone=timezone is not None), timestamp_value
        )
    if pt.is_timedelta64_dtype(dtype):
        return Column(name, sa.BigInteger(), duration_value)
    msg = f"Column {name!r}: unsupported dtype {dtype}"
    raise TypeError(msg)


def frame_columns(frame: pd.DataFrame) -> list[Column]:
    if len(frame.columns) == 0:
        msg = "PostgreSQL outputs require at least one column"
        raise ValueError(msg)
    if not frame.columns.is_unique or any(
        not isinstance(name, str) or not name.strip() for name in frame.columns
    ):
        msg = "PostgreSQL outputs require unique, nonempty string column names"
        raise ValueError(msg)
    return [column_spec(str(name), series) for name, series in frame.items()]


def converted_rows(
    batch: pd.DataFrame, columns: list[Column]
) -> Iterator[tuple[object, ...]]:
    # Construct this frame only from a bounded slice. Never mutate the caller's frame.
    data = pd.DataFrame(batch, copy=True)
    for column in columns:
        if column.srid is not None:
            geometries = [
                LineString(geom) if isinstance(geom, LinearRing) else geom
                for geom in batch[column.name]
            ]
            data[column.name] = to_wkb(
                set_srid(geometries, column.srid), hex=True, include_srid=True
            )
    for row in data.itertuples(index=False, name=None):
        yield tuple(
            convert_value(value, column)
            for value, column in zip(row, columns, strict=True)
        )


def convert_value(value: object, column: Column) -> object:
    if not pt.is_scalar(value) or pt.is_complex(value):
        msg = f"Column {column.name!r}: unsupported value type {type(value).__name__}"
        raise TypeError(msg)
    if pd.isna(value):
        return None
    try:
        return column.convert(value)
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"Column {column.name!r}: cannot convert {type(value).__name__}"
        raise ValueError(msg) from error

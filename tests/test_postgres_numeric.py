"""Numeric storage widths, values, and schema validation before connection."""

from unittest.mock import MagicMock, patch

import dagster as dg
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from cfc_dagster_utils.managers._postgres import converted_rows, frame_columns
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

INTEGER_CASES = [
    ("int8", "SMALLINT"),
    ("int16", "SMALLINT"),
    ("int32", "INTEGER"),
    ("int64", "BIGINT"),
    ("uint8", "SMALLINT"),
    ("uint16", "INTEGER"),
    ("uint32", "BIGINT"),
    ("uint64", "NUMERIC(20, 0)"),
    ("Int8", "SMALLINT"),
    ("Int16", "SMALLINT"),
    ("Int32", "INTEGER"),
    ("Int64", "BIGINT"),
    ("UInt8", "SMALLINT"),
    ("UInt16", "INTEGER"),
    ("UInt32", "BIGINT"),
    ("UInt64", "NUMERIC(20, 0)"),
]
FLOAT_CASES = [
    ("float16", "REAL"),
    ("float32", "REAL"),
    ("float64", "DOUBLE PRECISION"),
    ("Float32", "REAL"),
    ("Float64", "DOUBLE PRECISION"),
]
DOUBLE_BYTES = 8


@pytest.mark.parametrize(("dtype", "expected"), INTEGER_CASES)
def test_integer_ranges(dtype: str, expected: str) -> None:
    limits = np.iinfo(dtype.lower())
    values = [int(limits.min), 0, int(limits.max)]
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    columns = frame_columns(frame)
    assert str(columns[0].sql_type.compile(dialect=postgresql.dialect())) == expected
    assert list(converted_rows(frame, columns)) == [(value,) for value in values]


@pytest.mark.parametrize(("dtype", "expected"), FLOAT_CASES)
def test_float_width_and_values(dtype: str, expected: str) -> None:
    limits = np.finfo(dtype.lower())
    values = [
        0.1,
        -0.1,
        limits.smallest_subnormal,
        limits.max,
        -limits.max,
        float("inf"),
        -float("inf"),
        float("nan"),
    ]
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    columns = frame_columns(frame)
    assert str(columns[0].sql_type.compile(dialect=postgresql.dialect())) == expected
    actual = list(converted_rows(frame, columns))
    assert actual[-1] == (None,)
    assert actual[:-1] == [(float(value),) for value in frame["value"].iloc[:-1]]


@pytest.mark.parametrize(("dtype", "expected"), INTEGER_CASES + FLOAT_CASES)
def test_empty_numeric_schema(dtype: str, expected: str) -> None:
    frame = pd.DataFrame({"value": pd.Series([], dtype=dtype)})
    columns = frame_columns(frame)
    assert str(columns[0].sql_type.compile(dialect=postgresql.dialect())) == expected
    assert list(converted_rows(frame, columns)) == []


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(dtype, sql) for dtype, sql in INTEGER_CASES + FLOAT_CASES if dtype[0].isupper()],
)
def test_nullable_schema_and_first_batch(dtype: str, expected: str) -> None:
    for values in ([None, None], [None, None, 1]):
        frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
        columns = frame_columns(frame)
        assert (
            str(columns[0].sql_type.compile(dialect=postgresql.dialect())) == expected
        )
        assert list(converted_rows(frame.iloc[:2], columns)) == [(None,), (None,)]
        if not frame.iloc[2:].empty:
            assert list(converted_rows(frame.iloc[2:], columns)) == [(1,)]


def test_longdouble_is_never_silently_narrowed() -> None:
    frame = pd.DataFrame({"precise": pd.Series([1], dtype=np.longdouble)})
    if frame["precise"].dtype.itemsize <= DOUBLE_BYTES:
        assert str(frame_columns(frame)[0].sql_type) == "DOUBLE"
        return
    with pytest.raises(TypeError, match=r"Column 'precise': unsupported numeric dtype"):
        frame_columns(frame)
    resource = PostgresResource(
        host="unused", port=5432, user="unused", password="", database="unused"
    )
    manager = PostgresIOManager(postgres_resource=resource)
    context = MagicMock(spec=dg.OutputContext)
    context.definition_metadata = {"schema": "staging", "table": "target"}
    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(TypeError, match="precise"),
    ):
        manager.handle_output(context, frame)
    begin.assert_not_called()
    context.add_output_metadata.assert_not_called()

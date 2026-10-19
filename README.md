# cfc-dagster-utils

Utilities and IO managers used by CFC Dagster projects.

## Installation

The base package contains the pandas, GeoPandas, and SQLAlchemy utilities and can be
installed without Dagster:

```shell
pip install cfc_dagster_utils
```

Install the extra for the feature you use:

```shell
pip install 'cfc_dagster_utils[dagster]'
pip install 'cfc_dagster_utils[earthengine]'
pip install 'cfc_dagster_utils[postgres]'
pip install 'cfc_dagster_utils[xarray]'
```

The `earthengine` and `xarray` extras include Dagster because their managers depend on
it. Extras can be combined, for example:

```shell
pip install 'cfc_dagster_utils[earthengine,xarray]'
```

## Imports

Optional manager classes are imported from their feature modules. The package and the
`managers` namespace can always be imported without installing any extras.

```python
from cfc_dagster_utils.managers.dataframe import DataFrameFileManager
from cfc_dagster_utils.managers.earthengine import EarthEngineManager
from cfc_dagster_utils.managers.geodataframe import GeoDataFrameFileManager
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.managers.xarray import DataArrayFileManager
```

Importing a feature module without its dependency raises an `ImportError` containing
the exact extra installation command. Import errors originating inside an installed
dependency are preserved instead of being treated as a missing optional dependency.

## Local testing

Run the complete optional-dependency matrix locally with:

```shell
uv run python scripts/test_optional_dependency_matrix.py
```

Each combination runs in a temporary isolated environment, so packages installed in
the project's `.venv` cannot hide a missing optional dependency. GitHub Actions calls
the same runner, keeping the local and CI matrices in sync.

## PostgreSQL and PostGIS outputs

Install the `postgres` extra to use psycopg 3, GeoAlchemy2, the PostgreSQL resource,
and the pandas/GeoPandas IO manager:

```shell
pip install 'cfc_dagster_utils[postgres]'
```

The IO manager is a write-only bridge for placing Python-created frames in PostgreSQL.
Each output declares its destination with flat `schema` and `table` metadata. Any
schema is supported and is created when it does not already exist. Existing tables are
replaced.

```python
import geopandas as gpd

import dagster as dg
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource


@dg.asset(
    io_manager_key="postgres_io_manager",
    metadata={"schema": "staging", "table": "cities_prepared"},
)
def cities_prepared() -> gpd.GeoDataFrame:
    return load_cities()


postgres_resource = PostgresResource(
    host=dg.EnvVar("POSTGRES_HOST"),
    port=5432,
    database=dg.EnvVar("POSTGRES_DB"),
    user=dg.EnvVar("POSTGRES_USER"),
    password=dg.EnvVar("POSTGRES_PASSWORD"),
)

defs = dg.Definitions(
    assets=[cities_prepared],
    resources={
        "postgres_resource": postgres_resource,
        "postgres_io_manager": PostgresIOManager(
            postgres_resource=postgres_resource,
            batch_size=1000,  # Positive integer; this is the default.
        ),
    },
)
```

### Streaming writes and memory

Both DataFrames and GeoDataFrames use psycopg 3 `COPY`. The writer converts only
one slice of at most `batch_size` rows at a time. It never builds full-frame
object arrays, row dictionaries, geometry encodings, or CSV buffers. The input
DataFrame remains in memory, so this bounds **additional serialization memory**,
not the asset's preparation memory or total memory across concurrent assets.

`batch_size` defaults to 1,000 and must be a positive integer (not a boolean,
string, fractional number, or `None`). Lower it for especially wide rows or
complex geometries. It limits rows, not bytes; a single very large value can
still require substantial memory. PostgreSQL server memory is separate.

The same option is available on `PostgresConnectionComponent`:

```yaml
type: cfc_dagster_utils.components.postgres.PostgresConnectionComponent
attributes:
  host: "{{ env.POSTGRES_HOST }}"
  port: "{{ env.POSTGRES_PORT }}"
  database: "{{ env.POSTGRES_DB }}"
  user: "{{ env.POSTGRES_USER }}"
  password: "{{ env.POSTGRES_PASSWORD }}"
  batch_size: 1000
```

Schema creation, table replacement, and every batch are in one transaction.
A failed conversion, COPY, or commit rolls back the operation; an existing table
is restored rather than left partially populated. Successful Dagster output
metadata is emitted only after commit. Replacement does not cascade through
dependent tables or views.

### Column types

Types come from column dtypes, not the first batch or the first non-null row:

| DataFrame dtype | PostgreSQL type |
| --- | --- |
| Boolean, including nullable | `BOOLEAN` |
| Signed 8/16-bit integer or unsigned 8-bit integer | `SMALLINT` |
| Signed 32-bit integer or unsigned 16-bit integer | `INTEGER` |
| Signed 64-bit integer or unsigned 32-bit integer | `BIGINT` |
| Unsigned 64-bit integer | `NUMERIC(20,0)` |
| 16/32-bit floating point | `REAL` |
| 64-bit floating point | `DOUBLE PRECISION` |
| Datetime without timezone | `TIMESTAMP` |
| Datetime with timezone | `TIMESTAMPTZ` |
| Timedelta | `BIGINT`, nanoseconds |
| String, categorical, object | `TEXT` |
| Geometry | Unconstrained PostGIS `geometry` |

Numeric widths apply equally to NumPy and pandas nullable dtypes: for example,
`int32` and `Int32` both produce `INTEGER`, and `float32` and `Float32` both produce
`REAL`. The default `Int32` output of `cast_all_columns_to_numeric` therefore
remains 32-bit in PostgreSQL. Unsigned integers use a type large enough for their
full range; 16-bit floats widen to `REAL` because PostgreSQL has no 16-bit float.
Types are never downcast based on observed values. Unsupported numeric widths,
including floats wider than 64 bits, raise a column-specific `TypeError` before
connecting, avoiding silent precision loss. Existing tables receive the updated
types on their next replacement materialization.

Missing values become SQL `NULL`; empty strings and floating-point infinities
are preserved. Non-null scalar values in text columns are stringified, including
object-typed numbers and dates. Explicitly cast numeric or datetime columns
before writing when you want corresponding database types. Timestamp precision
beyond microseconds is discarded to match PostgreSQL storage.

Nested containers, complex numbers, and unsupported extension dtypes raise
column-specific errors. Column names must be unique, nonempty strings. The
DataFrame index is not written. A zero-row frame creates/replaces an empty table
with the dtype-derived schema; zero-column frames are rejected.

Every geometry-typed column is encoded in batches as EWKB using its own CRS.
Mixed geometry types and dimensions, nulls, empty geometries, and all-null
geometry columns are supported. LinearRings are stored as LineStrings. EPSG or
ESRI authority identifiers with an exact match supply the SRID; a missing or
unresolvable authority
produces a warning and SRID 0. Coordinates are not reprojected.

Geometry columns have no subtype, dimension, or SRID constraint. The SRID and
coordinates are preserved in each geometry value, but catalog metadata such as
`geometry_columns` does not establish a uniform column SRID. No indexes or
constraints are created automatically. Use downstream models for spatial
indexes and stricter geometry contracts.

`PostgresIOManager` is output-only. Use `PostgresResource.connect()` for explicit
reads into downstream Python assets.

### Verification

Run the required checks and full suite (including an XML coverage report):

```shell
uv sync --locked --all-extras
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest --cov=cfc_dagster_utils --cov-report=term-missing --cov-report=xml
```

PostgreSQL tests require a **disposable** database with PostGIS enabled. They
create unique schemas and remove them afterward. For example:

```shell
docker run --detach --rm --name cfc-postgres-tests \
  --publish 127.0.0.1:55432:5432 \
  --env POSTGRES_PASSWORD=cfc_test --env POSTGRES_DB=cfc_test \
  postgis/postgis:17-3.5
export CFC_TEST_POSTGRES_URL=postgresql+psycopg://postgres:cfc_test@localhost:55432/cfc_test
export CFC_REQUIRE_POSTGRES_TESTS=1
# Wait for pg_isready to report ready before running tests.
docker exec cfc-postgres-tests pg_isready -U postgres
uv run pytest -m postgres
uv run python scripts/benchmark_postgres_memory.py
docker stop cfc-postgres-tests
```

With `CFC_TEST_POSTGRES_URL` set, the full coverage command also runs the database
tests. Without it they skip locally; setting `CFC_REQUIRE_POSTGRES_TESTS=1` makes
missing configuration a failure. CI has a required PostGIS job with both set.
The optional-dependency matrix command above remains part of verification.

The Linux memory benchmark starts fresh worker processes, measures baseline RSS
after building each input and initializing connections, then samples peak RSS
while writing and checking the result. Defaults compare 10,000 and 100,000 rows
for 200-column census-like frames and polygon GeoDataFrames, followed by both
workloads writing concurrently. Each worker uses its own temporary schema.

The benchmark passes when measured overhead stays below 128 MiB per worker and
grows by at most 64 MiB between sizes. These are fixture regression thresholds,
not a universal memory guarantee. `--batch-size`, `--columns`, and `--rows SMALL
LARGE` allow further experiments. PostgreSQL server memory is not included.

# cfc-dagster-utils

Utilities and IO managers for CFC Dagster projects, including pandas and
GeoPandas helpers, file outputs, PostgreSQL/PostGIS outputs, and optional
Earth Engine and xarray support.

## Installation

```shell
uv add cfc-dagster-utils
```

Install the extras needed by your project: `dagster`, `dvc`, `earthengine`,
`postgres`, or `xarray`. Extras can be combined:

```shell
uv add 'cfc-dagster-utils[postgres,xarray]'
```

## Imports

Import managers from their feature modules:

```python
from cfc_dagster_utils.managers.dataframe import DataFrameFileManager
from cfc_dagster_utils.managers.geodataframe import GeoDataFrameFileManager
from cfc_dagster_utils.managers.postgres import PostgresIOManager
```

## PostgreSQL outputs

With the `postgres` extra installed, use `PostgresIOManager` to save DataFrames
or GeoDataFrames. Set `schema` and `table` in each asset's metadata. Missing
schemas are created, and existing destination tables are replaced.

```python
import dagster as dg
import pandas as pd

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource


@dg.asset(
    io_manager_key="postgres",
    metadata={"schema": "ingest", "table": "cities"},
)
def cities() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Mexico City", "Monterrey"]})


postgres = PostgresResource(
    host=dg.EnvVar("POSTGRES_HOST"),
    port=5432,
    database=dg.EnvVar("POSTGRES_DB"),
    user=dg.EnvVar("POSTGRES_USER"),
    password=dg.EnvVar("POSTGRES_PASSWORD"),
)

defs = dg.Definitions(
    assets=[cities],
    resources={"postgres": PostgresIOManager(postgres_resource=postgres)},
)
```

The manager is output-only. Use `PostgresResource.connect()` for explicit reads.
GeoDataFrame outputs require PostGIS. The optional `batch_size` setting defaults
to 1,000 rows and accepts a positive integer.

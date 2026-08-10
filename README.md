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

## PostgreSQL and PostGIS assets

Install the `postgres` extra to use psycopg 3, GeoAlchemy2, the PostgreSQL resource,
and the unified pandas/GeoPandas IO manager:

```shell
pip install 'cfc_dagster_utils[postgres]'
```

Each database-backed asset declares a serializable table contract. The write mode is
configured per asset; `create` is the safe default.

```python
import geopandas as gpd

import dagster as dg
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource
from cfc_dagster_utils.types import (
    PostgresIndex,
    PostgresIndexMethod,
    PostgresRelation,
    PostgresTableSpec,
    PostgresWriteMode,
)

CITIES = PostgresTableSpec(
    relation=PostgresRelation(schema="public", name="cities"),
    write_mode=PostgresWriteMode.REPLACE,
    primary_key=("city_id",),
    indexes=(
        PostgresIndex(
            columns=("geometry",),
            method=PostgresIndexMethod.GIST,
        ),
    ),
    geometry_column="geometry",
)


@dg.asset(
    io_manager_key="postgres_io_manager",
    metadata=CITIES.to_dagster_metadata(),
)
def cities() -> gpd.GeoDataFrame:
    return load_cities()


postgres_resource = PostgresResource(
    host=dg.EnvVar("POSTGRES_HOST"),
    port=5432,
    database=dg.EnvVar("POSTGRES_DB"),
    user=dg.EnvVar("POSTGRES_USER"),
    password=dg.EnvVar("POSTGRES_PASSWORD"),
)

defs = dg.Definitions(
    assets=[cities],
    resources={
        "postgres_resource": postgres_resource,
        "postgres_io_manager": PostgresIOManager(
            postgres_resource=postgres_resource,
        ),
    },
)
```

Downstream assets can request `PostgresRelation` to receive a zero-copy table handle
instead of loading the table into memory:

```python
@dg.asset
def city_summary(cities: PostgresRelation) -> None:
    # cities is PostgresRelation(schema="public", name="cities")
    run_server_side_query(cities)
```

For an asset that performs its own server-side SQL, use
`PostgresResource.stage_dataframe()` and `publish_relation()` in the same transaction,
then return the published relation. `PostgresIOManager` verifies that the declared
relation exists without uploading it again.

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
from cfc_dagster_utils.managers.postgres import DataFramePostgresManager
from cfc_dagster_utils.managers.xarray import DataArrayFileManager
```

Importing a feature module without its dependency raises an `ImportError` containing
the exact extra installation command. Import errors originating inside an installed
dependency are preserved instead of being treated as a missing optional dependency.

## Local testing

Run the complete eight-combination optional-dependency matrix locally with:

```shell
uv run python scripts/test_optional_dependency_matrix.py
```

Each combination runs in a temporary isolated environment, so packages installed in
the project's `.venv` cannot hide a missing optional dependency. GitHub Actions calls
the same runner, keeping the local and CI matrices in sync.

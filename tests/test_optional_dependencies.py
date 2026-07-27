import importlib
import importlib.util
import os
import re

import pytest

from cfc_dagster_utils._optional import raise_optional_dependency_error

EXTRAS_ENV_VAR = "CFC_TEST_EXTRAS"
DEPENDENCY_MODULES = {
    "dagster": "dagster",
    "dvc": "dvc",
    "earthengine": "ee",
    "xarray": "xarray",
}
OPTIONAL_MODULES = {
    "cfc_dagster_utils.partitions": "dagster",
    "cfc_dagster_utils.resources": "dagster",
    "cfc_dagster_utils.managers.file": "dagster",
    "cfc_dagster_utils.managers.dataframe": "dagster",
    "cfc_dagster_utils.managers.geodataframe": "dagster",
    "cfc_dagster_utils.managers.json": "dagster",
    "cfc_dagster_utils.managers.postgres": "dagster",
    "cfc_dagster_utils.components.dvc": "dvc",
    "cfc_dagster_utils.managers.earthengine": "earthengine",
    "cfc_dagster_utils.managers.xarray": "xarray",
}


def _available_extras() -> set[str]:
    requested = os.environ.get(EXTRAS_ENV_VAR)
    if requested is None:
        return {
            extra
            for extra, module in DEPENDENCY_MODULES.items()
            if importlib.util.find_spec(module) is not None
        }

    available = set(filter(None, requested.split(",")))
    if available & {"dvc", "earthengine", "xarray"}:
        available.add("dagster")
    return available


def test_base_imports_do_not_require_extras() -> None:
    for module in (
        "cfc_dagster_utils",
        "cfc_dagster_utils.managers",
        "cfc_dagster_utils.types",
        "cfc_dagster_utils.utils",
    ):
        assert importlib.import_module(module) is not None


def test_manager_namespace_does_not_eagerly_export_classes() -> None:
    managers = importlib.import_module("cfc_dagster_utils.managers")

    assert not hasattr(managers, "DataFrameFileManager")
    assert not hasattr(managers, "EarthEngineManager")
    assert not hasattr(managers, "DataArrayFileManager")


def test_matrix_environment_contains_only_expected_optional_dependencies() -> None:
    if os.environ.get(EXTRAS_ENV_VAR) is None:
        pytest.skip("Exact dependency assertions only apply to isolated matrix runs")

    available = _available_extras()
    for extra, module in DEPENDENCY_MODULES.items():
        assert (importlib.util.find_spec(module) is not None) is (extra in available)


@pytest.mark.parametrize(("module", "extra"), OPTIONAL_MODULES.items())
def test_optional_module_import(module: str, extra: str) -> None:
    if extra in _available_extras():
        assert importlib.import_module(module) is not None
        return

    install_command = f"pip install 'cfc_dagster_utils[{extra}]'"
    with pytest.raises(ImportError, match=re.escape(install_command)):
        importlib.import_module(module)


def test_direct_missing_dependency_has_actionable_error() -> None:
    error = ModuleNotFoundError("No module named 'example'", name="example")

    with pytest.raises(
        ImportError, match=re.escape("pip install 'cfc_dagster_utils[example]'")
    ) as raised:
        raise_optional_dependency_error(
            error,
            import_name="example",
            dependency_name="example-package",
            extra="example",
        )

    assert raised.value.__cause__ is error


def test_transitive_missing_dependency_is_not_hidden() -> None:
    error = ModuleNotFoundError("No module named 'transitive'", name="transitive")

    with pytest.raises(ModuleNotFoundError) as raised:
        raise_optional_dependency_error(
            error,
            import_name="example",
            dependency_name="example-package",
            extra="example",
        )

    assert raised.value is error

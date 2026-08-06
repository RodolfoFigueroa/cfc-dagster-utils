import json
from pathlib import Path
from unittest.mock import Mock, patch

import dagster as dg
import ee
import geopandas as gpd
import pytest
import xarray as xr

from cfc_dagster_utils.managers.earthengine import EarthEngineManager
from cfc_dagster_utils.managers.file import PathResource, _BaseFileManager
from cfc_dagster_utils.managers.geodataframe import GeoDataFrameFileManager
from cfc_dagster_utils.managers.xarray import DataArrayFileManager

PATH_DAGSTER_TYPE = dg.DagsterType(
    type_check_fn=lambda _context, value: isinstance(value, Path),
    name="TestPath",
    typing_type=Path,
)
STRING_DAGSTER_TYPE = dg.DagsterType(
    type_check_fn=lambda _context, value: isinstance(value, str),
    name="TestString",
    typing_type=str,
)


def _manager(
    manager_type: type[_BaseFileManager], tmp_path: Path, extension: str
) -> _BaseFileManager:
    return manager_type(
        path_resource=PathResource(out_path=str(tmp_path)),
        extension=extension,
    )


def test_base_file_manager_returns_requested_path(tmp_path: Path) -> None:
    manager = _manager(_BaseFileManager, tmp_path, ".data")

    with dg.build_input_context(
        asset_key=dg.AssetKey(["group", "foo"]),
        dagster_type=PATH_DAGSTER_TYPE,
    ) as context:
        result = manager.load_input(context)

    assert result == tmp_path / "group" / "foo.data"


def test_base_file_manager_delegates_deserialization(tmp_path: Path) -> None:
    manager = _manager(_BaseFileManager, tmp_path, ".data")
    expected = object()

    with (
        patch.object(
            _BaseFileManager, "_load_from_path", return_value=expected
        ) as load,
        dg.build_input_context(
            asset_key="foo", dagster_type=STRING_DAGSTER_TYPE
        ) as context,
    ):
        result = manager.load_input(context)

    assert result is expected
    load.assert_called_once_with(tmp_path / "foo.data")


def test_base_file_manager_rejects_multiple_partitions_as_path(
    tmp_path: Path,
) -> None:
    manager = _manager(_BaseFileManager, tmp_path, ".data")
    partitions = dg.StaticPartitionsDefinition(["first", "second"])

    with (
        dg.build_input_context(
            asset_key="foo",
            dagster_type=PATH_DAGSTER_TYPE,
            asset_partitions_def=partitions,
            asset_partition_key_range=dg.PartitionKeyRange("first", "second"),
        ) as context,
        pytest.raises(ValueError, match="exactly one upstream asset partition"),
    ):
        manager.load_input(context)


def test_geodataframe_manager_deserializes_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = gpd.GeoDataFrame({"value": [1]})
    read_file = Mock(return_value=expected)
    monkeypatch.setattr(gpd, "read_file", read_file)
    manager = _manager(GeoDataFrameFileManager, tmp_path, ".geojson")
    fpath = tmp_path / "foo.geojson"

    result = manager._load_from_path(fpath)  # noqa: SLF001

    assert result is expected
    read_file.assert_called_once_with(fpath)


def test_data_array_manager_deserializes_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = xr.DataArray([1])
    open_dataarray = Mock(return_value=expected)
    monkeypatch.setattr(xr, "open_dataarray", open_dataarray)
    manager = _manager(DataArrayFileManager, tmp_path, ".nc")
    fpath = tmp_path / "foo.nc"

    result = manager._load_from_path(fpath)  # noqa: SLF001

    assert result is expected
    open_dataarray.assert_called_once_with(fpath)


def test_earth_engine_manager_deserializes_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fpath = tmp_path / "foo.json"
    fpath.write_text(json.dumps({"type": "Invocation"}), encoding="utf8")
    expected = Mock(spec=ee.ComputedObject)
    decode = Mock(return_value=expected)
    monkeypatch.setattr(ee.deserializer, "decode", decode)
    manager = _manager(EarthEngineManager, tmp_path, ".json")

    result = manager._load_from_path(fpath)  # noqa: SLF001

    assert result is expected
    decode.assert_called_once_with({"type": "Invocation"})

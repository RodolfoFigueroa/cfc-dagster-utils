from pathlib import Path

import dagster as dg
import pandas as pd
import pytest

from cfc_dagster_utils.managers.dataframe import (
    DataFrameEngine,
    DataFrameFileManager,
)
from cfc_dagster_utils.managers.file import PathResource


@pytest.mark.parametrize("engine", [None, "csv", "parquet"])
def test_dataframe_file_manager_accepts_supported_engines(
    tmp_path: Path,
    engine: str | None,
) -> None:
    manager = DataFrameFileManager(
        path_resource=PathResource(out_path=str(tmp_path)),
        extension=".data",
        engine=engine,
    )

    expected = DataFrameEngine(engine) if engine is not None else None
    assert manager.engine == expected


def test_dataframe_file_manager_rejects_unsupported_engine(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"csv.*parquet"):
        DataFrameFileManager(
            path_resource=PathResource(out_path=str(tmp_path)),
            extension=".data",
            engine="json",
        )


def test_dataframe_asset_can_be_loaded_as_dataframe_and_path(tmp_path: Path) -> None:
    expected = pd.DataFrame({"value": [1, 2]})

    @dg.asset(io_manager_key="dataframe_files")
    def source_dataframe() -> pd.DataFrame:
        return expected

    @dg.asset
    def dataframe_consumer(source_dataframe: pd.DataFrame) -> int:
        pd.testing.assert_frame_equal(source_dataframe, expected)
        return len(source_dataframe)

    @dg.asset
    def path_consumer(source_dataframe: Path) -> str:
        expected_path = tmp_path / "source_dataframe.csv"
        assert source_dataframe == expected_path
        assert source_dataframe.exists()
        return str(source_dataframe)

    manager = DataFrameFileManager(
        path_resource=PathResource(out_path=str(tmp_path)),
        extension=".csv",
    )

    result = dg.materialize(
        [source_dataframe, dataframe_consumer, path_consumer],
        resources={"dataframe_files": manager},
    )

    assert result.success

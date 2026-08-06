from pathlib import Path

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

from enum import StrEnum

import pandas as pd

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.managers.file import _BaseFileManager, _ResolvedFilePath

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="dagster",
    )


class DataFrameEngine(StrEnum):
    """Supported storage formats for custom DataFrame file extensions."""

    csv = "csv"
    parquet = "parquet"


class DataFrameFileManager(_BaseFileManager):
    """Dagster IO manager for reading and writing pandas DataFrames to/from local files.

    Supports ``.parquet`` and ``.csv`` file formats. For partitioned assets, loading
    returns a mapping of partition key to DataFrame.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use. Anything other than ``.parquet``
            or ``.csv`` requires manually specifying ``engine``.
        engine: The engine to use for reading/writing files. If not specified, defaults
            to the default engine for the given file format, if applicable.
    """

    engine: DataFrameEngine | None = None

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """Write a pandas DataFrame to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The pandas DataFrame to write.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv`` and
                ``engine`` is not specified.
        """
        if self.extension not in [".parquet", ".csv"]:
            if self.engine is None:
                err = "If extension is not .parquet or .csv, engine must be specified."
                raise ValueError(err)
            if self.engine not in ["csv", "parquet"]:
                err = f"Unsupported engine: {self.engine}. Must be 'csv' or 'parquet'."
                raise ValueError(err)

        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".parquet":
            obj.to_parquet(fpath)
        elif self.extension == ".csv":
            obj.to_csv(fpath, index=True)
        elif self.engine == "parquet":
            obj.to_parquet(fpath)
        elif self.engine == "csv":
            obj.to_csv(fpath, index=True)

    def _load_from_path(
        self, fpath: _ResolvedFilePath
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Load a pandas DataFrame (or a mapping of DataFrames) from a file.

        For non-partitioned or single-partition assets, returns a single DataFrame.
        For multi-partition inputs, returns a dict mapping partition key to DataFrame.

        Args:
            fpath: A single asset path or a mapping of partition keys to paths.

        Returns:
            A single DataFrame or a dict mapping partition key to DataFrame.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv``.
        """
        if self.extension == ".parquet":
            return self._dispatch_multiple_partitions(fpath, pd.read_parquet)

        if self.extension == ".csv":
            return self._dispatch_multiple_partitions(
                fpath, lambda p: pd.read_csv(p, index_col=0)
            )

        if self.engine == "parquet":
            return self._dispatch_multiple_partitions(fpath, pd.read_parquet)

        if self.engine == "csv":
            return self._dispatch_multiple_partitions(
                fpath, lambda p: pd.read_csv(p, index_col=0)
            )

        err = f"Unsupported file extension: {self.extension} and engine: {self.engine}"
        raise ValueError(err)

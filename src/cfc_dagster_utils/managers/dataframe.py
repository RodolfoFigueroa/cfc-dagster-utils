import dagster as dg
import pandas as pd

from cfc_dagster_utils.managers.file import _BaseFileManager


class DataFrameFileManager(_BaseFileManager):
    """Dagster IO manager for reading and writing pandas DataFrames to/from local files.

    Supports ``.parquet`` and ``.csv`` file formats. For partitioned assets, loading
    returns a mapping of partition key to DataFrame.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use. Must be ``.parquet`` or ``.csv``.
    """

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """Write a pandas DataFrame to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The pandas DataFrame to write.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv``.
        """
        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".parquet":
            obj.to_parquet(fpath)
        elif self.extension == ".csv":
            obj.to_csv(fpath, index=True)
        else:
            err = f"Unsupported file extension: {self.extension}"
            raise ValueError(err)

    def load_input(
        self, context: dg.InputContext
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Load a pandas DataFrame (or a mapping of DataFrames) from a file.

        For non-partitioned or single-partition assets, returns a single DataFrame.
        For multi-partition inputs, returns a dict mapping partition key to DataFrame.

        Args:
            context: The Dagster input context used to resolve the file path(s).

        Returns:
            A single DataFrame or a dict mapping partition key to DataFrame.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv``.
        """
        fpath = self._get_path(context, allow_multiple_partitions=True)

        if self.extension == ".parquet":
            return self._dispatch_multiple_partitions(fpath, pd.read_parquet)

        if self.extension == ".csv":
            return self._dispatch_multiple_partitions(
                fpath, lambda p: pd.read_csv(p, index_col=0)
            )

        err = f"Unsupported file extension: {self.extension}"
        raise ValueError(err)

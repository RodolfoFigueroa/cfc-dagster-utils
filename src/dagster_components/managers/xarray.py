import dagster as dg
import xarray as xr

from dagster_components.managers.file import _BaseFileManager


class DataArrayFileManager(_BaseFileManager):
    """Dagster IO manager for reading and writing xarray DataArrays to/from local files.

    Supports ``.nc`` file format. For partitioned assets, loading
    returns a mapping of partition key to DataArray.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use. Must be ``.nc``.
    """

    def handle_output(self, context: dg.OutputContext, obj: xr.DataArray) -> None:
        """Write a xarray DataArray to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The xarray DataArray to write.

        Raises:
            ValueError: If ``extension`` is not ``.nc``.
        """
        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".nc":
            obj.to_netcdf(fpath)
        else:
            err = f"Unsupported file extension: {self.extension}"
            raise ValueError(err)

    def load_input(
        self, context: dg.InputContext
    ) -> xr.DataArray | dict[str, xr.DataArray]:
        """Load a xarray DataArray (or a mapping of DataArrays) from a file.

        For non-partitioned or single-partition assets, returns a single DataArray.
        For multi-partition inputs, returns a dict mapping partition key to DataArray.

        Args:
            context: The Dagster input context used to resolve the file path(s).

        Returns:
            A single DataArray or a dict mapping partition key to DataArray.

        Raises:
            ValueError: If ``extension`` is not ``.nc``.
        """
        fpath = self._get_path(context, allow_multiple_partitions=True)

        if self.extension == ".nc":
            return self._dispatch_multiple_partitions(fpath, xr.open_dataarray)

        err = f"Unsupported file extension: {self.extension}"
        raise ValueError(err)

import geopandas as gpd

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


class GeoDataFrameFileManager(_BaseFileManager):
    """Dagster IO manager for reading and writing GeoDataFrames to/from local files.

    Supports ``.gpkg`` (GeoPackage) and ``.geoparquet`` file formats. For partitioned
    assets, loading returns a mapping of partition key to GeoDataFrame.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use. Must be ``.gpkg`` or ``.geoparquet``.
    """

    def handle_output(self, context: dg.OutputContext, obj: gpd.GeoDataFrame) -> None:
        """Write a GeoDataFrame to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The GeoDataFrame to write.

        Raises:
            ValueError: If ``extension`` is not ``.gpkg`` or ``.geoparquet``.
        """
        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".gpkg":
            obj.to_file(fpath, driver="GPKG")
        elif self.extension == ".geoparquet":
            obj.to_parquet(fpath, index=True)
        elif self.extension == ".geojson":
            obj.to_file(fpath, driver="GeoJSON")
        else:
            err = f"Unsupported file extension: {self.extension}"
            raise ValueError(err)

    def _load_from_path(
        self, fpath: _ResolvedFilePath
    ) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
        """Load a GeoDataFrame (or a mapping of GeoDataFrames) from a file.

        For non-partitioned or single-partition assets, returns a single GeoDataFrame.
        For multi-partition inputs, returns a dict mapping partition key to
        GeoDataFrame.

        Args:
            fpath: A single asset path or a mapping of partition keys to paths.

        Returns:
            A single GeoDataFrame or a dict mapping partition key to GeoDataFrame.

        Raises:
            ValueError: If ``extension`` is not ``.gpkg`` or ``.geoparquet``.
        """
        if self.extension == ".gpkg":
            return self._dispatch_multiple_partitions(fpath, gpd.read_file)

        if self.extension == ".geoparquet":
            return self._dispatch_multiple_partitions(fpath, gpd.read_parquet)

        if self.extension == ".geojson":
            return self._dispatch_multiple_partitions(fpath, gpd.read_file)

        err = f"Unsupported file extension: {self.extension}"
        raise ValueError(err)

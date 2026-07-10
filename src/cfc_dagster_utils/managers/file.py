from collections.abc import Callable
from pathlib import Path
from typing import Literal, overload

import dagster as dg

from cfc_dagster_utils.types import DFType, T


class PathResource(dg.ConfigurableResource):
    """Dagster resource providing a base output directory path for file-based IO
    managers.

    Args:
        out_path: The root directory path where assets are stored.

    Attributes:
        out_path: The root directory path where assets are stored.
    """

    out_path: str


class _BaseFileManager(dg.ConfigurableIOManager):
    """Base class for Dagster IO managers that read/write to/from files.

    Handles path resolution for both partitioned and non-partitioned assets.
    Subclasses must implement ``handle_output`` and ``load_input``.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use.

    Attributes:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use.
    """

    path_resource: dg.ResourceDependency[PathResource]
    extension: str

    def _get_single_partition_key_path(
        self, partition_key: str, asset_dir: Path
    ) -> Path:
        """Resolve the file path for a single partition key within an asset directory.

        Partition keys containing ``|`` are treated as multi-dimensional and mapped to
        nested subdirectory segments.

        Args:
            partition_key: The partition key string, with ``|`` as segment separator.
            asset_dir: The root directory for the asset.

        Returns:
            The resolved file path including the configured extension.
        """

        segments = partition_key.split("|")
        fpath = asset_dir / "/".join(segments)
        return fpath.with_suffix(fpath.suffix + self.extension)

    @overload
    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: Literal[False],
    ) -> Path: ...

    @overload
    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: Literal[True],
    ) -> dict[str, Path]: ...

    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: bool,
    ) -> Path | dict[str, Path]:
        """Resolve the file path(s) for a partitioned asset context.

        If the context has a single partition key, returns a single ``Path``.
        If multiple partition keys are present and ``allow_multiple_partitions`` is
        ``True``, returns a mapping of partition key to ``Path``.

        Args:
            context: The Dagster input or output context providing partition key
                information.
            asset_dir: The root directory for the asset.
            allow_multiple_partitions: Whether multiple partition paths may be returned.

        Returns:
            A single path for single-partition contexts, or a dict mapping partition key
            to path when multiple partitions are present.

        Raises:
            ValueError: If multiple partition keys are present and
                ``allow_multiple_partitions`` is ``False``.
        """
        if len(context.asset_partition_keys) == 1:
            return self._get_single_partition_key_path(
                context.asset_partition_key, asset_dir
            )

        if not allow_multiple_partitions:
            err = (
                "Multiple partition keys found for asset, but allow_multiple_partitions"
                "is False."
                f"Partition keys: {context.asset_partition_keys}"
            )
            raise ValueError(err)

        final_path = {}
        for partition_key in context.asset_partition_keys:
            final_path[partition_key] = self._get_single_partition_key_path(
                partition_key, asset_dir
            )
        return final_path

    @overload
    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: Literal[False] = False,
    ) -> Path: ...

    @overload
    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: Literal[True],
    ) -> dict[str, Path]: ...

    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: bool = False,
    ) -> Path | dict[str, Path]:
        """Resolve the output file path(s) for a given IO context.

        Combines the root path from ``path_resource`` with the asset key path to
        form a directory, then delegates to partition-aware helpers as needed.

        Args:
            context: The Dagster input or output context.
            allow_multiple_partitions: When ``True``, allows returning a
                ``dict[str, Path]`` for assets with multiple active partition keys.
                Defaults to ``False``.

        Returns:
            A single path for non-partitioned or single-partition assets, or a dict
            mapping partition key to path when multiple partitions are present.
        """
        out_path = Path(self.path_resource.out_path)
        asset_dir = out_path / "/".join(context.asset_key.path)

        if context.has_asset_partitions:
            return self._get_partitioned_asset_path(
                context, asset_dir, allow_multiple_partitions=allow_multiple_partitions
            )

        return asset_dir.with_suffix(asset_dir.suffix + self.extension)

    @overload
    def _dispatch_multiple_partitions(
        self,
        fpath: Path,
        func: Callable[[Path], T],
    ) -> T: ...

    @overload
    def _dispatch_multiple_partitions(
        self,
        fpath: dict[str, Path],
        func: Callable[[Path], T],
    ) -> dict[str, T]: ...

    def _dispatch_multiple_partitions(
        self,
        fpath: Path | dict[str, Path],
        func: Callable[[Path], T],
    ) -> T | dict[str, T]:
        """Apply a function to a single path or to each path in a partition mapping.

        Args:
            fpath: Either a single ``Path`` or a dict mapping partition keys to paths.
            func: A callable that accepts a ``Path`` and returns a value of type ``T``.

        Returns:
            The result of ``func(fpath)`` when ``fpath`` is a single ``Path``, or a dict
            mapping partition key to function result when ``fpath`` is a dict.

        Raises:
            TypeError: If ``fpath`` is neither a ``Path`` nor a ``dict``.
        """
        if isinstance(fpath, Path):
            return func(fpath)

        if isinstance(fpath, dict):
            return {k: func(v) for k, v in fpath.items()}

        err = f"Unexpected type for file path: {type(fpath)}"
        raise TypeError(err)

    def handle_output(self, context: dg.OutputContext, obj: DFType) -> None:
        """Write an object to a file. Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError

    def load_input(self, context: dg.InputContext) -> DFType:
        """Load an object from a file. Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError

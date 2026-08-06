import json
from pathlib import Path

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.managers.file import _BaseFileManager

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="dagster",
    )


class JSONManager(_BaseFileManager):
    """Dagster IO manager for reading and writing JSON files.

    Subclasses must implement ``handle_output`` and ``_load_from_path`` to define how
    objects are serialized to and deserialized from a ``dict`` before JSON encoding.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use.
    """

    def _write_serialized_json(
        self,
        serialized: dict,
        context: dg.OutputContext,
    ) -> None:
        """Write a serialized dict to a JSON file.

        Creates parent directories as needed.

        Args:
            serialized: The dict to serialize as JSON.
            context: The Dagster output context used to resolve the output file path.

        Raises:
            TypeError: If the resolved path is a dict (i.e. multiple partitions are
                active), since JSONManager does not support multiple partitions.
        """
        fpath = self._get_path(context)

        if isinstance(fpath, dict):
            err = "JSONManager does not support multiple partitions."
            raise TypeError(err)

        fpath.parent.mkdir(exist_ok=True, parents=True)

        with fpath.open("w", encoding="utf8") as f:
            json.dump(serialized, f)

    def _read_serialized_json(self, fpath: Path) -> dict:
        """Read a JSON file and return its contents as a dict.

        Args:
            fpath: The resolved input file path.

        Returns:
            The deserialized JSON contents.

        """
        with fpath.open(encoding="utf8") as f:
            return json.load(f)

import json

import dagster as dg

from cfc_dagster_utils.managers.file import _BaseFileManager


class JSONManager(_BaseFileManager):
    """Dagster IO manager for reading and writing JSON files.

    Subclasses must implement ``handle_output`` and ``load_input`` to define how objects
    are serialized to and deserialized from a ``dict`` before JSON encoding.

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

    def _read_serialized_json(self, context: dg.InputContext) -> dict:
        """Read a JSON file and return its contents as a dict.

        Args:
            context: The Dagster input context used to resolve the input file path.

        Returns:
            The deserialized JSON contents.

        Raises:
            TypeError: If the resolved path is a dict (i.e. multiple partitions are
                active), since JSONManager does not support multiple partitions.
        """
        fpath = self._get_path(context)

        if isinstance(fpath, dict):
            err = "JSONManager does not support multiple partitions."
            raise TypeError(err)

        with fpath.open(encoding="utf8") as f:
            return json.load(f)

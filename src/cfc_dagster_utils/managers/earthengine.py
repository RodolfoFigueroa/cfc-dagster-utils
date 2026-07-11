import json

from cfc_dagster_utils._optional import raise_optional_dependency_error

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="earthengine",
    )

from cfc_dagster_utils.managers.json import JSONManager

try:
    import ee
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="ee",
        dependency_name="earthengine-api",
        extra="earthengine",
    )


class EarthEngineManager(JSONManager):
    """Dagster IO manager for reading and writing Google Earth Engine objects as JSON.

    Serializes ``ee.image.Image`` and ``ee.geometry.Geometry`` objects to JSON files
    and restores them using the Earth Engine deserializer.

    Args:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use.
    """

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: ee.Image | ee.Geometry | ee.ComputedObject,
    ) -> None:
        """Serialize an Earth Engine object and write it to a JSON file.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The Earth Engine image or geometry to serialize.
        """
        serialized = json.loads(obj.serialize())
        self._write_serialized_json(serialized, context)

    def load_input(
        self,
        context: dg.InputContext,
    ) -> ee.Image | ee.Geometry | ee.ComputedObject:
        """Read a JSON file and deserialize it into an Earth Engine object.

        Args:
            context: The Dagster input context used to resolve the input file path.

        Returns:
            The deserialized Earth Engine image or geometry.

        Raises:
            TypeError: If the deserialized object is not an ``ee.image.Image`` or
                ``ee.geometry.Geometry``.
        """
        serialized = self._read_serialized_json(context)
        deserialized = ee.deserializer.decode(serialized)

        if isinstance(deserialized, (ee.Image, ee.Geometry, ee.ComputedObject)):
            return deserialized

        err = f"Unsupported type: {type(deserialized)}"
        raise TypeError(err)

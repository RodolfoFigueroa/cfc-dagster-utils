import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cfc_dagster_utils._optional import raise_optional_dependency_error

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="dvc",
    )

try:
    import dvc
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dvc",
        dependency_name="dvc",
        extra="dvc",
    )

try:
    import yaml
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="yaml",
        dependency_name="PyYAML",
        extra="dvc",
    )


@dataclass(frozen=True)
class _DvcOutput:
    path: str
    md5: str
    size: int | None
    file_count: int | None


def _optional_nonnegative_int(
    output: Mapping[object, object],
    field: str,
    descriptor: Path,
) -> int | None:
    value = output.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"DVC descriptor '{descriptor}' has an invalid '{field}' value."
        raise dg.Failure(msg)
    return value


def _load_dvc_output(descriptor: Path) -> _DvcOutput:
    try:
        document = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    except OSError as error:
        msg = f"Unable to read DVC descriptor '{descriptor}': {error}"
        raise dg.Failure(msg) from error
    except yaml.YAMLError as error:
        msg = f"Unable to parse DVC descriptor '{descriptor}': {error}"
        raise dg.Failure(msg) from error

    if not isinstance(document, dict):
        msg = f"DVC descriptor '{descriptor}' must contain a YAML mapping."
        raise dg.Failure(msg)

    outputs = document.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        msg = f"DVC descriptor '{descriptor}' must contain exactly one output."
        raise dg.Failure(msg)

    output = outputs[0]
    if not isinstance(output, dict):
        msg = f"DVC descriptor '{descriptor}' contains an invalid output."
        raise dg.Failure(msg)

    output_path = output.get("path")
    if not isinstance(output_path, str) or not output_path:
        msg = f"DVC descriptor '{descriptor}' output must contain a path."
        raise dg.Failure(msg)

    hash_algorithm = output.get("hash")
    if hash_algorithm not in (None, "md5"):
        msg = f"DVC descriptor '{descriptor}' output must use MD5."
        raise dg.Failure(msg)

    checksum = output.get("md5")
    if not isinstance(checksum, str) or not checksum:
        msg = f"DVC descriptor '{descriptor}' output must contain an MD5 checksum."
        raise dg.Failure(msg)

    return _DvcOutput(
        path=output_path,
        md5=checksum,
        size=_optional_nonnegative_int(output, "size", descriptor),
        file_count=_optional_nonnegative_int(output, "nfiles", descriptor),
    )


def _get_dvc_status(
    executable: str,
    repository_root: Path,
    dvc_file: str,
) -> dict[str, object]:
    try:
        result = subprocess.run(  # noqa: S603 - no shell; executable is resolved.
            [
                executable,
                "status",
                "--json",
                "--no-updates",
                "--",
                dvc_file,
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        msg = f"DVC status failed for '{dvc_file}'"
        if details:
            msg = f"{msg}: {details}"
        raise dg.Failure(msg) from error
    except OSError as error:
        msg = f"Unable to run DVC status for '{dvc_file}': {error}"
        raise dg.Failure(msg) from error

    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        msg = f"DVC returned invalid JSON while checking '{dvc_file}'."
        raise dg.Failure(msg) from error

    if not isinstance(status, dict) or not all(isinstance(key, str) for key in status):
        msg = f"DVC returned an unexpected status for '{dvc_file}'."
        raise dg.Failure(msg)

    return cast("dict[str, object]", status)


class DvcInputComponent(dg.Component, dg.Resolvable, dg.Model):
    """Observe one MD5-backed output described by a standalone ``.dvc`` file."""

    key: dg.ResolvedAssetKey
    dvc_file: str

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        repository_root = context.project_root
        descriptor = repository_root / self.dvc_file
        spec = dg.AssetSpec(
            key=self.key,
            group_name="raw_inputs",
            kinds={"dvc"},
        )

        @dg.multi_observable_source_asset(
            name=self.key.to_python_identifier(suffix="dvc_observe"),
            specs=[spec],
        )
        def observe() -> Iterator[dg.ObserveResult]:
            executable = shutil.which("dvc")
            if executable is None:
                msg = (
                    "The DVC executable is unavailable. Install it with "
                    "`pip install 'cfc_dagster_utils[dvc]'`."
                )
                raise dg.Failure(msg)

            changes = _get_dvc_status(
                executable,
                repository_root,
                self.dvc_file,
            )
            if changes:
                msg = "The DVC workspace does not match the tracked input version."
                raise dg.Failure(msg, metadata={"dvc/status": changes})

            output = _load_dvc_output(descriptor)
            metadata: dict[str, str | int] = {
                "dvc/checksum": output.md5,
                "dvc/hash_algorithm": "md5",
                "dvc/path": output.path,
                "dvc/descriptor": self.dvc_file,
                "dvc/version": dvc.__version__,
            }
            if output.size is not None:
                metadata["size_bytes"] = output.size
            if output.file_count is not None:
                metadata["file_count"] = output.file_count

            yield dg.ObserveResult(
                asset_key=self.key,
                data_version=dg.DataVersion(f"md5:{output.md5}"),
                metadata=metadata,
            )

        return dg.Definitions(assets=[observe])

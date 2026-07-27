import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import dagster as dg
import pytest

import cfc_dagster_utils.managers.dvc as dvc_module
from cfc_dagster_utils.managers.dvc import DvcInputComponent

pytestmark = pytest.mark.filterwarnings(
    "ignore:Function `multi_observable_source_asset` is currently in beta"
)


def _build_observation_job(
    project_root: Path,
    *,
    dvc_file: str = "input.dvc",
) -> dg.JobDefinition:
    context = replace(
        dg.ComponentTree.for_test().load_context,
        project_root=project_root,
    )
    component = DvcInputComponent(
        key=dg.AssetKey("input_data"),
        dvc_file=dvc_file,
    )
    definitions = dg.Definitions.merge(
        component.build_defs(context),
        dg.Definitions(
            jobs=[
                dg.define_asset_job(
                    "observe_dvc_input",
                    selection=[component.key],
                )
            ]
        ),
    )
    dg.Definitions.validate_loadable(definitions)
    return definitions.get_repository_def().get_job("observe_dvc_input")


def _mock_dvc_status(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: dict[str, object],
) -> Mock:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(status),
            stderr="",
        )
    )
    monkeypatch.setattr(dvc_module.shutil, "which", lambda _command: "/usr/bin/dvc")
    monkeypatch.setattr(dvc_module.subprocess, "run", run)
    return run


def test_clean_dvc_input_emits_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checksum = "d41d8cd98f00b204e9800998ecf8427e"
    size = 42
    file_count = 3
    descriptor = tmp_path / "input.dvc"
    descriptor.write_text(
        "\n".join(
            [
                "outs:",
                f"- md5: {checksum}",
                f"  size: {size}",
                f"  nfiles: {file_count}",
                "  hash: md5",
                "  path: input",
            ]
        ),
        encoding="utf-8",
    )
    run = _mock_dvc_status(monkeypatch, status={})

    result = _build_observation_job(tmp_path).execute_in_process()

    assert result.success
    event = result.get_asset_observation_events()[0]
    observation = event.asset_observation_data.asset_observation
    assert observation.asset_key == dg.AssetKey("input_data")
    assert observation.tags["dagster/data_version"] == f"md5:{checksum}"
    assert observation.metadata["dvc/checksum"].value == checksum
    assert observation.metadata["dvc/hash_algorithm"].value == "md5"
    assert observation.metadata["size_bytes"].value == size
    assert observation.metadata["file_count"].value == file_count
    run.assert_called_once_with(
        [
            "/usr/bin/dvc",
            "status",
            "--json",
            "--no-updates",
            "--",
            "input.dvc",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_changed_dvc_input_fails_without_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_dvc_status(
        monkeypatch,
        status={"input.dvc": {"changed outs": {"modified": ["input"]}}},
    )

    result = _build_observation_job(tmp_path).execute_in_process(raise_on_error=False)

    assert not result.success
    assert result.get_step_failure_events()
    assert not result.get_asset_observation_events()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            """outs:
- md5: first
  path: first
- md5: second
  path: second
""",
            "exactly one output",
        ),
        (
            """outs:
- etag: external-checksum
  path: external
""",
            "must contain an MD5 checksum",
        ),
    ],
)
def test_descriptor_rejects_unsupported_output_shapes(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    descriptor = tmp_path / "input.dvc"
    descriptor.write_text(contents, encoding="utf-8")

    with pytest.raises(dg.Failure, match=message):
        dvc_module._load_dvc_output(descriptor)  # noqa: SLF001


def test_missing_dvc_executable_has_actionable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dvc_module.shutil, "which", lambda _command: None)

    result = _build_observation_job(tmp_path).execute_in_process(raise_on_error=False)

    assert not result.success
    failure = result.get_step_failure_events()[0]
    error = failure.step_failure_data.error
    assert error is not None
    assert "cfc_dagster_utils[dvc]" in error.to_string()

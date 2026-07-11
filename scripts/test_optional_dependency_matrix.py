"""Run optional-dependency tests in isolated uv environments."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "tests/test_optional_dependencies.py"
EXTRAS_ENV_VAR = "CFC_TEST_EXTRAS"
MATRIX = (
    ("base", ()),
    ("dagster", ("dagster",)),
    ("earthengine", ("earthengine",)),
    ("xarray", ("xarray",)),
    ("dagster-earthengine", ("dagster", "earthengine")),
    ("dagster-xarray", ("dagster", "xarray")),
    ("earthengine-xarray", ("earthengine", "xarray")),
    ("all", ("dagster", "earthengine", "xarray")),
)


def _run_combination(uv: str, name: str, extras: tuple[str, ...]) -> bool:
    environment = os.environ.copy()
    environment[EXTRAS_ENV_VAR] = ",".join(extras)

    command = [
        uv,
        "run",
        "--isolated",
        "--locked",
        "--no-dev",
        "--with",
        "pytest",
    ]
    for extra in extras:
        command.extend(("--extra", extra))
    command.extend(("python", "-m", "pytest", TEST_PATH))

    LOGGER.info("Running optional-dependency combination: %s", name)
    result = subprocess.run(  # noqa: S603 - arguments are generated internally.
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    """Run every optional-dependency combination and report a summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    uv = shutil.which("uv")
    if uv is None:
        LOGGER.error("uv is required to run the optional-dependency matrix")
        return 2

    failed = [name for name, extras in MATRIX if not _run_combination(uv, name, extras)]
    if failed:
        LOGGER.error("Failed combinations: %s", ", ".join(failed))
        return 1

    LOGGER.info("All %d optional-dependency combinations passed", len(MATRIX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

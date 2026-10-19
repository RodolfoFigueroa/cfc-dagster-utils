"""Measure writer-added RSS in isolated Linux processes against disposable PostGIS."""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from uuid import uuid4

import dagster as dg
import geopandas as gpd
import pandas as pd
import sqlalchemy as sa
from shapely import Polygon

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

LOGGER = logging.getLogger(__name__)
MIB = 1024 * 1024
MAX_OVERHEAD_MIB = 128
MAX_GROWTH_MIB = 64
SAMPLE_SECONDS = 0.01
POLYGON_VERTICES = 32


def make_frame(kind: str, rows: int, columns: int) -> pd.DataFrame:
    if kind == "census":
        return pd.DataFrame(
            {
                f"value_{i}": (
                    pd.Series(["001 census"] * rows, dtype="string")
                    if i % 10 == 0
                    else pd.Series(range(rows), dtype="Int64")
                )
                for i in range(columns)
            }
        )
    polygon = Polygon(
        [
            (
                math.cos(i * math.tau / POLYGON_VERTICES),
                math.sin(i * math.tau / POLYGON_VERTICES),
            )
            for i in range(POLYGON_VERTICES)
        ]
    )
    return gpd.GeoDataFrame({"id": range(rows)}, geometry=[polygon] * rows, crs=4326)


def worker(
    pipe: Connection, kind: str, rows: int, columns: int, batch_size: int
) -> None:
    url = sa.make_url(os.environ["CFC_TEST_POSTGRES_URL"])
    resource = PostgresResource(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username or "postgres",
        password=url.password or "",
        database=url.database or "postgres",
    )
    schema = "benchmark_" + uuid4().hex
    frame = make_frame(kind, rows, columns)
    manager = PostgresIOManager(postgres_resource=resource, batch_size=batch_size)
    try:
        with (
            dg.build_init_resource_context() as init_context,
            dg.build_output_context(
                definition_metadata={"schema": schema, "table": "output"}
            ) as output_context,
        ):
            resource.setup_for_execution(init_context)
            try:
                with resource.begin() as conn:
                    conn.execute(sa.schema.CreateSchema(schema))
                    conn.execute(sa.text("SELECT PostGIS_Version()"))
                pipe.send(("ready", 0.0))
                pipe.recv()
                start = time.monotonic()
                manager.handle_output(output_context, frame)
                with resource.connect() as conn:
                    table = sa.Table(
                        "output", sa.MetaData(), schema=schema, autoload_with=conn
                    )
                    actual = conn.execute(
                        sa.select(sa.func.count()).select_from(table)
                    ).scalar_one()
                    if actual != rows:
                        msg = f"Expected {rows} rows, got {actual}"
                        raise RuntimeError(msg)
                pipe.send(("done", time.monotonic() - start))
                # Keep the source frame alive until the monitor records the final RSS.
                pipe.recv()
            finally:
                with resource.begin() as conn:
                    conn.execute(
                        sa.schema.DropSchema(schema, if_exists=True, cascade=True)
                    )
                resource.teardown_after_execution(init_context)
    finally:
        pipe.close()


def rss(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    msg = f"Cannot read RSS for process {pid}"
    raise RuntimeError(msg)


@dataclass
class Monitor:
    process: BaseProcess
    pipe: Connection
    kind: str
    baseline: int = 0
    peak: int = 0
    seconds: float | None = None

    def sample(self) -> None:
        pid = self.process.pid
        if pid is None:
            msg = "Worker has not started"
            raise RuntimeError(msg)
        self.peak = max(self.peak, rss(pid))

    def wait_ready(self) -> None:
        message, _ = self.pipe.recv()
        if message != "ready":
            msg = f"Unexpected worker message: {message}"
            raise RuntimeError(msg)
        self.sample()
        self.baseline = self.peak

    def poll_result(self) -> None:
        if self.seconds is not None:
            return
        self.sample()
        if self.pipe.poll():
            message, seconds = self.pipe.recv()
            if message != "done":
                msg = f"Unexpected worker message: {message}"
                raise RuntimeError(msg)
            self.sample()
            self.seconds = seconds

    def finish(self) -> None:
        self.pipe.send("release")
        self.process.join(timeout=30)
        if self.process.exitcode != 0:
            msg = f"Worker failed or did not exit: {self.process.exitcode}"
            raise RuntimeError(msg)


def measure(
    kinds: tuple[str, ...], rows: int, columns: int, batch_size: int
) -> list[dict[str, str | int | float]]:
    context = mp.get_context("spawn")
    monitors: list[Monitor] = []
    try:
        for kind in kinds:
            parent, child = context.Pipe()
            process = context.Process(
                target=worker, args=(child, kind, rows, columns, batch_size)
            )
            process.start()
            child.close()
            monitors.append(Monitor(process, parent, kind))
        for monitor in monitors:
            monitor.wait_ready()
        for monitor in monitors:
            monitor.pipe.send("start")
        while any(monitor.seconds is None for monitor in monitors):
            for monitor in monitors:
                monitor.poll_result()
            time.sleep(SAMPLE_SECONDS)
        results = [
            {
                "workload": monitor.kind,
                "rows": rows,
                "writers": len(kinds),
                "baseline_mib": round(monitor.baseline / MIB, 2),
                "peak_mib": round(monitor.peak / MIB, 2),
                "overhead_mib": round((monitor.peak - monitor.baseline) / MIB, 2),
                "seconds": round(monitor.seconds or 0.0, 2),
            }
            for monitor in monitors
        ]
        for monitor in monitors:
            monitor.finish()
        return results
    finally:
        for monitor in monitors:
            if monitor.process.is_alive():
                monitor.process.terminate()
            monitor.process.join(timeout=5)
            monitor.pipe.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--columns", type=int, default=200)
    parser.add_argument("--rows", type=int, nargs=2, default=(10000, 100000))
    args = parser.parse_args()
    if not os.environ.get("CFC_TEST_POSTGRES_URL"):
        parser.error("Set CFC_TEST_POSTGRES_URL to a disposable PostGIS database")
    if (
        min(args.batch_size, args.columns, *args.rows) <= 0
        or args.rows[0] >= args.rows[1]
    ):
        parser.error("Sizes must be positive and --rows must specify increasing sizes")
    results: list[dict[str, str | int | float]] = []
    for kind in ("census", "geometry"):
        for rows in args.rows:
            result = measure((kind,), rows, args.columns, args.batch_size)
            results.extend(result)
            LOGGER.info("%s", json.dumps(result))
    concurrent = measure(
        ("census", "geometry"), args.rows[1], args.columns, args.batch_size
    )
    results.extend(concurrent)
    LOGGER.info("%s", json.dumps(concurrent))
    overheads = [float(result["overhead_mib"]) for result in results]
    growth = [overheads[1] - overheads[0], overheads[3] - overheads[2]]
    if max(overheads) > MAX_OVERHEAD_MIB or max(growth) > MAX_GROWTH_MIB:
        LOGGER.error("FAIL: writer memory exceeds fixture acceptance thresholds")
        return 1
    LOGGER.info("PASS: writer memory remains within fixture acceptance thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

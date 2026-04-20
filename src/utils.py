"""Shared path helpers for local and Modal entrypoint scripts."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def timestamped_output_dir(run_name: str, output_root: str | Path = RESULTS_ROOT) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(output_dir: Path, filename: str = "terminal.log"):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
            yield log_path


def write_rows_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histories_csv(path: Path, histories: dict[str, dict], metadata: dict | None = None) -> None:
    metadata = metadata or {}
    metric_keys: list[str] = []
    for history in histories.values():
        for key in history:
            if key not in metric_keys:
                metric_keys.append(key)

    if "step" in metric_keys:
        metric_keys.remove("step")
        metric_keys.insert(0, "step")

    rows = []
    for series, history in histories.items():
        lengths = [len(values) for values in history.values() if isinstance(values, list)]
        n_rows = max(lengths, default=0)
        for i in range(n_rows):
            row = dict(metadata)
            row["series"] = series
            row["index"] = i
            for key in metric_keys:
                values = history.get(key)
                if isinstance(values, list) and i < len(values):
                    row[key] = values[i]
            rows.append(row)

    fieldnames = [*metadata.keys(), "series", "index", *metric_keys]
    write_rows_csv(path, rows, fieldnames)

"""Read and write method/run experiment artifacts.

Layout::

    artifacts/methods/<method_id>/method.json   (+ prompt.txt, script.py, attempts/)
    artifacts/runs/<run_id>/run.json            (+ predictions.jsonl, metrics.json)

The files are canonical; the Streamlit viewer reads them directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .configuration import ARTIFACTS_DIR

METHODS_DIR = ARTIFACTS_DIR / "methods"
RUNS_DIR = ARTIFACTS_DIR / "runs"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_method(
    method: dict[str, Any],
    *,
    prompt: str | None = None,
    script: str | None = None,
    attempts: Iterable[dict[str, Any]] = (),
) -> Path:
    method_id = method["method_id"]
    directory = METHODS_DIR / method_id
    directory.mkdir(parents=True, exist_ok=True)
    _dump(directory / "method.json", method)
    if prompt is not None:
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    if script is not None:
        (directory / "script.py").write_text(script, encoding="utf-8")
    for index, attempt in enumerate(attempts, start=1):
        _dump(directory / "attempts" / f"attempt-{index}.json", attempt)
    return directory


def new_run_id(method_id: str, dataset_id: str, split: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{slug(method_id)}_{slug(dataset_id)}_{split}"


def write_run(
    run: dict[str, Any],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    prompt: str | None = None,
    script: str | None = None,
) -> Path:
    directory = RUNS_DIR / run["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    _dump(directory / "run.json", {**run, "metrics": metrics})
    _dump(directory / "metrics.json", metrics)
    if prompt is not None:
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    if script is not None:
        (directory / "script.py").write_text(script, encoding="utf-8")
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return directory


def list_methods() -> list[dict[str, Any]]:
    methods = []
    if not METHODS_DIR.exists():
        return methods
    for path in sorted(METHODS_DIR.glob("*/method.json")):
        try:
            methods.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return methods


def list_runs() -> list[dict[str, Any]]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for path in sorted(RUNS_DIR.glob("*/run.json"), reverse=True):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def load_run(run_id: str) -> dict[str, Any] | None:
    path = RUNS_DIR / run_id / "run.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(run_id: str) -> list[dict[str, Any]]:
    path = RUNS_DIR / run_id / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

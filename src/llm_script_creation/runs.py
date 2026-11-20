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


def _round_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": item["round"],
        "kind": item["kind"],
        "model": item.get("model"),
        "usage": item.get("usage", {}),
        "validation_issues": item.get("issues", []),
        "empty_response": item.get("empty_response", False),
        "train_score": item.get("score"),
        "train_nonexample_score": item.get("nonexample_score"),
        "train_metrics": item.get("train_metrics", {}),
        "train_nonexample_metrics": item.get("nonexample_metrics", {}),
        "failure_count": item.get("failure_count", 0),
        "crash_count": item.get("crash_count", 0),
        "selected_failures": [
            {
                "doc_id": record.get("doc_id"),
                "status": record.get("status"),
                "error": (record.get("error") or "")[:500],
            }
            for record in item.get("selected", [])
        ],
    }


def write_run(
    run: dict[str, Any],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    prompt: str | None = None,
    script: str | None = None,
    rounds: Iterable[dict[str, Any]] = (),
) -> Path:
    directory = RUNS_DIR / run["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    round_items = list(rounds)
    round_summaries = [_round_summary(item) for item in round_items]
    _dump(
        directory / "run.json",
        {
            **run,
            "metrics": metrics,
            "rounds": run.get("rounds") or round_summaries,
        },
    )
    _dump(directory / "metrics.json", metrics)
    if prompt is not None:
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    if script is not None:
        (directory / "script.py").write_text(script, encoding="utf-8")
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    for item in round_items:
        round_dir = directory / "rounds" / f"round-{item['round']}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _dump(round_dir / "round.json", _round_summary(item))
        (round_dir / "script.py").write_text(item.get("code", ""), encoding="utf-8")
        (round_dir / "response.txt").write_text(item.get("response", ""), encoding="utf-8")
        if item.get("feedback"):
            (round_dir / "feedback.txt").write_text(item["feedback"], encoding="utf-8")
        _dump(round_dir / "messages.json", item.get("messages", []))
        with (round_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in item.get("records", []):
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return directory


def list_rounds(run_id: str) -> list[dict[str, Any]]:
    directory = RUNS_DIR / run_id / "rounds"
    if not directory.exists():
        return []
    rounds = []
    for path in directory.glob("round-*/round.json"):
        try:
            rounds.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(rounds, key=lambda item: item.get("round", 0))


def load_round(run_id: str, round_index: int) -> dict[str, Any] | None:
    path = RUNS_DIR / run_id / "rounds" / f"round-{round_index}" / "round.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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

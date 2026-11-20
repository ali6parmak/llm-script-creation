"""Run experiments and write method/run artifacts.

Currently this runs deterministic baselines; the LLM script-generation runner
uses the same artifact writer later.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any

from . import baselines, catalog, runs
from .configuration import CATALOG_PATH
from .evaluation import (
    aggregate_plan,
    aggregate_upr,
    aggregate_upr_state,
    evaluate_plan_sample,
    evaluate_upr_sample,
    evaluate_upr_state_sample,
)


def _dataset(conn: Any, dataset_id: str) -> dict[str, Any]:
    row = catalog.get_dataset(conn, dataset_id)
    if row is None:
        raise SystemExit(f"Unknown dataset: {dataset_id}")
    return dict(row)


def _records_for_split(conn: Any, dataset_id: str, split: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in catalog.iter_samples(conn, dataset_id, split=None if split == "all" else split)
    ]


def _payload(doc_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": doc_row["text"],
        "segments": json.loads(doc_row["segments_json"]),
        "page_count": doc_row["page_count"],
        "language": doc_row["language"],
    }


def run_baseline(dataset_id: str, split: str = "dev", limit: int | None = None) -> dict[str, Any]:
    baseline = baselines.for_dataset(dataset_id)
    if baseline is None:
        raise SystemExit(f"No baseline registered for dataset: {dataset_id}")

    conn = catalog.connect()
    try:
        dataset = _dataset(conn, dataset_id)
        target_schema = json.loads(dataset["target_schema_json"])
        fields = list(target_schema)
        samples = _records_for_split(conn, dataset_id, split)
        if limit is not None:
            samples = samples[:limit]

        records: list[dict[str, Any]] = []
        for sample in samples:
            doc_row = catalog.get_document(conn, dataset["db"], sample["doc_id"])
            if doc_row is None:
                continue
            doc_row = dict(doc_row)
            gold = json.loads(sample["gold_json"])
            started = time.perf_counter()
            error = None
            try:
                prediction = baseline.function(_payload(doc_row))
            except Exception as exc:  # noqa: BLE001 - baseline errors are data
                prediction = None
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            if error:
                status = "error"
                details = {}
            elif dataset_id == "upr_state":
                details = evaluate_upr_state_sample(gold, prediction)
                status = details["status"]
            elif dataset_id == "upr_recommendations":
                details = evaluate_upr_sample(gold, prediction)
                status = details["status"]
            else:
                details = evaluate_plan_sample(gold, prediction, fields)
                status = details["status"]

            records.append(
                {
                    "doc_id": sample["doc_id"],
                    "split": sample["split"],
                    "gold": gold,
                    "prediction": prediction,
                    "status": status,
                    "error": error,
                    "latency_ms": latency_ms,
                    "details": details,
                }
            )

        if dataset_id == "upr_state":
            metrics = aggregate_upr_state(records)
        elif dataset_id == "upr_recommendations":
            metrics = aggregate_upr(records)
        else:
            metrics = aggregate_plan(records, fields)

        now = datetime.now(UTC).isoformat()
        method = {
            "method_id": baseline.method_id,
            "name": baseline.name,
            "description": baseline.description,
            "type": "rule_based",
            "dataset_id": dataset_id,
            "task": dataset["task"],
            "config": {"function": baseline.function.__name__},
            "created_at": now,
        }
        runs.write_method(method)
        run_id = runs.new_run_id(baseline.method_id, dataset_id, split)
        run = {
            "run_id": run_id,
            "method_id": baseline.method_id,
            "dataset_id": dataset_id,
            "split": split,
            "status": "completed",
            "created_at": now,
            "counts": metrics.get("status_counts", {}),
        }
        runs.write_run(run, records, metrics)
        return {"run": run, "metrics": metrics, "method": method}
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        help="dataset id, or 'all' for every dataset with a registered baseline",
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="limit samples for a smoke run")
    args = parser.parse_args(argv)

    if not CATALOG_PATH.exists():
        raise SystemExit("Catalog missing. Run `uv run build-datasets` first.")

    dataset_ids = (
        [baseline.dataset_id for baseline in baselines.BASELINES]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset_id in dataset_ids:
        print(f"\n== {dataset_id} ({args.split}) ==", flush=True)
        result = run_baseline(dataset_id, args.split, args.limit)
        metrics = result["metrics"]
        run = result["run"]
        print(f"  run: {run['run_id']}", flush=True)
        if "accuracy" in metrics:
            print(f"  accuracy: {metrics['accuracy']:.3f}", flush=True)
        if "f1" in metrics:
            print(
                f"  rec P/R/F1: {metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f} "
                f"(state {metrics['state_accuracy']:.3f})",
                flush=True,
            )
        print(f"  statuses: {metrics.get('status_counts', {})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

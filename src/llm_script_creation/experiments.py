"""Run deterministic baselines over the v2 train/test split and write artifacts.

The rule-based baselines receive the full document text plus the segment list.
They do not use an InputSpec; only the LLM script pipeline does.
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
from .task_ids import V2, V3, base_dataset_id, dataset_version, versioned_dataset_id


def _dataset(conn: Any, dataset_id: str) -> dict[str, Any]:
    row = catalog.get_dataset(conn, dataset_id)
    if row is None:
        raise SystemExit(f"Unknown dataset: {dataset_id}")
    return dict(row)


def _samples(conn: Any, dataset_id: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = sorted(
        (dict(row) for row in catalog.iter_samples(conn, dataset_id, split=split)),
        key=lambda row: row["doc_id"],
    )
    return rows[:limit] if limit is not None else rows


def _payload(doc_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": doc_row["text"],
        "segments": json.loads(doc_row["segments_json"]),
        "page_count": doc_row["page_count"],
        "language": doc_row["language"],
    }


def _aggregate(dataset_id: str, records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    base_id = base_dataset_id(dataset_id)
    if base_id == "upr_state":
        return aggregate_upr_state(records)
    if base_id == "upr_recommendations":
        return aggregate_upr(records)
    return aggregate_plan(records, fields)


def _primary_score(dataset_id: str, metrics: dict[str, Any]) -> float:
    if base_dataset_id(dataset_id) == "upr_recommendations":
        return float(metrics.get("f1", 0.0))
    return float(metrics.get("accuracy", 0.0))


def _run_split(
    dataset_id: str,
    dataset: dict[str, Any],
    baseline: baselines.Baseline,
    samples: list[dict[str, Any]],
    conn: Any,
    fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        elif base_dataset_id(dataset_id) == "upr_state":
            details = evaluate_upr_state_sample(gold, prediction)
            status = details["status"]
        elif base_dataset_id(dataset_id) == "upr_recommendations":
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
    return records, _aggregate(dataset_id, records, fields)


def run_baseline(dataset_id: str, limit: int | None = None) -> dict[str, Any]:
    baseline = baselines.for_dataset(dataset_id)
    if baseline is None:
        raise SystemExit(f"No baseline registered for dataset: {dataset_id}")

    conn = catalog.connect()
    try:
        dataset = _dataset(conn, dataset_id)
        manifest = json.loads(dataset["manifest_json"] or "{}")
        fields = list(json.loads(dataset["target_schema_json"]))
        train_samples = _samples(conn, dataset_id, "train", limit)
        test_samples = _samples(conn, dataset_id, "test")

        train_records, train_metrics = _run_split(
            dataset_id, dataset, baseline, train_samples, conn, fields
        )
        test_records, test_metrics = _run_split(
            dataset_id, dataset, baseline, test_samples, conn, fields
        )
        train_score = _primary_score(dataset_id, train_metrics)
        test_score = _primary_score(dataset_id, test_metrics)
        metric_name = "f1" if base_dataset_id(dataset_id) == "upr_recommendations" else "accuracy"
        primary = {
            "metric": metric_name,
            "train": train_score,
            "train_nonexample": train_score,
            "test": test_score,
            "gap": train_score - test_score,
            "gap_nonexample": train_score - test_score,
        }
        train_failures = sum(1 for record in train_records if record["status"] != "hit")
        metrics = {
            "primary_metric": metric_name,
            "primary": primary,
            "train": train_metrics,
            "train_nonexample": train_metrics,
            "test": test_metrics,
            "status_counts": {
                "train": train_metrics.get("status_counts", {}),
                "train_nonexample": train_metrics.get("status_counts", {}),
                "test": test_metrics.get("status_counts", {}),
            },
            "repair_rounds": 0,
            "rounds": [],
            "failure_counts": {"train": train_failures, "train_crash": 0, "per_round": []},
            "token_usage": {},
            "split_policy_version": manifest.get("split_policy_version"),
            "input_spec": None,
            "examples": [],
            "train_documents": len(train_samples),
            "train_nonexample_documents": len(train_samples),
            "test_documents": len(test_samples),
        }

        now = datetime.now(UTC).isoformat()
        baseline_version = dataset_version(dataset_id) == V3
        method_id = (
            f"{baseline.method_id}_{V3}" if baseline_version else baseline.method_id
        )
        method = {
            "method_id": method_id,
            "name": f"{baseline.name} (v3)" if baseline_version else baseline.name,
            "description": baseline.description,
            "type": "rule_based",
            "dataset_id": dataset_id,
            "task": dataset["task"],
            "model": None,
            "input_spec": None,
            "examples": [],
            "repair_rounds": 0,
            "train_documents": len(train_samples),
            "test_documents": len(test_samples),
            "split_policy_version": manifest.get("split_policy_version"),
            "config": {"function": baseline.function.__name__, "input": "full text + segments"},
            "created_at": now,
            "final": primary,
        }
        runs.write_method(method)
        run_id = runs.new_run_id(method_id, dataset_id, "train_test")
        run = {
            "run_id": run_id,
            "method_id": method_id,
            "dataset_id": dataset_id,
            "split": "train_test",
            "status": "completed",
            "created_at": now,
            "counts": test_metrics.get("status_counts", {}),
            "repair_rounds": 0,
            "rounds": [],
            "split_policy_version": manifest.get("split_policy_version"),
        }
        runs.write_run(
            run,
            [*train_records, *test_records],
            metrics,
        )
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
    parser.add_argument("--limit", type=int, default=None, help="limit train samples (smoke)")
    parser.add_argument(
        "--split-version",
        choices=[V2, V3],
        default=V2,
        help="dataset split version for --dataset all (default v2)",
    )
    args = parser.parse_args(argv)

    if not CATALOG_PATH.exists():
        raise SystemExit("Catalog missing. Run `uv run build-datasets` first.")

    dataset_ids = (
        [
            versioned_dataset_id(baseline.dataset_id, args.split_version)
            for baseline in baselines.BASELINES
        ]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset_id in dataset_ids:
        print(f"\n== {dataset_id} (train/test) ==", flush=True)
        result = run_baseline(dataset_id, args.limit)
        metrics = result["metrics"]
        primary = metrics["primary"]
        print(f"  run: {result['run']['run_id']}", flush=True)
        print(
            f"  train {primary['train']:.3f} · test {primary['test']:.3f} · "
            f"gap {primary['gap']:+.3f}",
            flush=True,
        )
        print(f"  test statuses: {metrics['status_counts']['test']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score admin-agent extractor predictions and assemble method/run payloads.

Main-process safe: this module uses OUR evaluation and artifact layout only. It
does not import the admin agent. (The admin agent's own ``extraction_eval`` is
applied inside the sidecar and returned as an extra ``admin_eval`` block; that
is the only way to compare under its exact display-equality rule without
dragging its dependencies into this environment.)

Conversion rules:

* the extractor's output is ``{property: value}`` or ``None``; ``None`` becomes
  a missing prediction (never a zero or a wrong value);
* our ``evaluate_plan_sample`` receives that raw field dict directly — no
  ``{"value": ...}`` wrapper is introduced.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..evaluation import aggregate_plan, evaluate_plan_sample


def score_records(
    dataset_id: str,
    fields: list[str],
    samples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score one prediction per sample and aggregate train/test metrics.

    ``samples`` are ``{"doc_id", "split", "gold"}`` dicts (from the catalog).
    ``predictions`` maps ``doc_id`` to ``{"prediction", "error", "latency_ms"}``.
    """
    if not dataset_id.startswith("plan_") or dataset_id == "plan_all_fields":
        raise ValueError(
            f"admin-agent adapter phase 1 supports single Plan fields only, got {dataset_id!r}"
        )

    records: list[dict[str, Any]] = []
    for sample in samples:
        doc_id = str(sample["doc_id"])
        result = predictions.get(doc_id) or {}
        prediction = result.get("prediction")
        error = result.get("error")
        gold = sample.get("gold") or {}
        if error:
            status = "error"
            details: dict[str, Any] = {}
        else:
            details = evaluate_plan_sample(gold, prediction, fields)
            status = details["status"]
        records.append(
            {
                "doc_id": doc_id,
                "split": sample.get("split"),
                "gold": gold,
                "prediction": prediction,
                "status": status,
                "error": error,
                "latency_ms": result.get("latency_ms"),
                "details": details,
            }
        )

    train_records = [record for record in records if record["split"] == "train"]
    test_records = [record for record in records if record["split"] == "test"]
    train_metrics = aggregate_plan(train_records, fields)
    test_metrics = aggregate_plan(test_records, fields)
    train_score = float(train_metrics["accuracy"])
    test_score = float(test_metrics["accuracy"])

    errors = [
        {"doc_id": record["doc_id"], "split": record["split"], "error": record["error"]}
        for record in records
        if record["status"] == "error"
    ]
    coverage = {
        split: _coverage(metrics, fields) for split, metrics in (("train", train_metrics), ("test", test_metrics))
    }
    metrics: dict[str, Any] = {
        "primary_metric": "accuracy",
        "primary": {
            "metric": "accuracy",
            "train": train_score,
            "train_nonexample": None,
            "test": test_score,
            "gap": train_score - test_score,
            "gap_nonexample": None,
        },
        "train": train_metrics,
        "train_nonexample": None,
        "test": test_metrics,
        "coverage": coverage,
        "status_counts": {
            "train": dict(Counter(record["status"] for record in train_records)),
            "test": dict(Counter(record["status"] for record in test_records)),
        },
        "errors": errors[:50],
        "error_count": len(errors),
        "repair_rounds": 0,
        "rounds": [],
        "train_documents": len(train_records),
        "train_nonexample_documents": 0,
        "test_documents": len(test_records),
    }
    return records, metrics


def _coverage(metrics: dict[str, Any], fields: list[str]) -> float | None:
    """Share of judged samples where a value was produced at all."""
    total = metrics.get("total") or 0
    if not total or not fields:
        return None
    present = sum((metrics.get("fields", {}).get(field) or {}).get("present", 0) for field in fields)
    return present / (total * len(fields))


def build_method(
    *,
    method_id: str,
    name: str,
    description: str,
    dataset_id: str,
    task: str,
    model: str,
    extractor_code: str,
    prompt: str,
    manifest: dict[str, Any],
    config: dict[str, Any],
    usage: dict[str, Any],
    primary: dict[str, Any],
    train_documents: int,
    test_documents: int,
    created_at: str,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "name": name,
        "description": description,
        "type": "admin_agent_extractor",
        "dataset_id": dataset_id,
        "task": task,
        "model": model,
        "input_spec": {"representation": "text", "source": "primary-document segmentation"},
        "examples": [],
        "prompt_version": "admin-agent-v1",
        "split_policy_version": manifest.get("split_policy_version"),
        "train_documents": train_documents,
        "test_documents": test_documents,
        "usage": usage,
        "created_at": created_at,
        "config": config,
        "final": primary,
        "script_hash": _hash(extractor_code),
        "prompt_chars": len(prompt),
    }


def build_run(
    *,
    run_id: str,
    method_id: str,
    dataset_id: str,
    status: str,
    created_at: str,
    counts: dict[str, int],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "method_id": method_id,
        "dataset_id": dataset_id,
        "split": "train_test",
        "status": status,
        "created_at": created_at,
        "counts": counts,
        "repair_rounds": 0,
        "rounds": [],
        "token_usage": usage,
        "split_policy_version": manifest.get("split_policy_version"),
    }


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:12]


__all__ = ["build_method", "build_run", "score_records"]

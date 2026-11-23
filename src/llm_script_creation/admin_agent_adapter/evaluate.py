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

from ..evaluation import (
    aggregate_plan,
    aggregate_upr,
    aggregate_upr_state,
    evaluate_plan_sample,
    evaluate_upr_sample,
    evaluate_upr_state_sample,
)
from ..task_ids import base_dataset_id


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
    base_id = base_dataset_id(dataset_id)
    if base_id in {"upr_state", "upr_recommendations"}:
        kind = base_id
    elif base_id.startswith("plan_") and base_id != "plan_all_fields":
        kind = "plan"
    else:
        raise ValueError(f"admin-agent adapter does not support dataset {dataset_id!r}")

    records: list[dict[str, Any]] = []
    for sample in samples:
        doc_id = str(sample["doc_id"])
        result = predictions.get(doc_id) or {}
        prediction = _adapt_prediction(result.get("prediction"))
        error = result.get("error")
        gold = sample.get("gold") or {}
        if error:
            status = "error"
            details: dict[str, Any] = {}
        elif kind == "plan":
            details = evaluate_plan_sample(gold, prediction, fields)
            status = details["status"]
        elif kind == "upr_state":
            details = evaluate_upr_state_sample(gold, prediction)
            status = details["status"]
        else:
            details = evaluate_upr_sample(gold, prediction)
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
    if kind == "plan":
        train_metrics = aggregate_plan(train_records, fields)
        test_metrics = aggregate_plan(test_records, fields)
        metric = "accuracy"
        train_score = float(train_metrics["accuracy"])
        test_score = float(test_metrics["accuracy"])
        coverage = {
            split: _coverage(metrics, fields)
            for split, metrics in (("train", train_metrics), ("test", test_metrics))
        }
    elif kind == "upr_state":
        train_metrics = aggregate_upr_state(train_records)
        test_metrics = aggregate_upr_state(test_records)
        metric = "accuracy"
        train_score = float(train_metrics["accuracy"])
        test_score = float(test_metrics["accuracy"])
        coverage = {
            "train": _status_coverage(train_records),
            "test": _status_coverage(test_records),
        }
    else:
        train_metrics = aggregate_upr(train_records)
        test_metrics = aggregate_upr(test_records)
        metric = "f1"
        train_score = float(train_metrics["f1"])
        test_score = float(test_metrics["f1"])
        coverage = {
            "train": _status_coverage(train_records),
            "test": _status_coverage(test_records),
        }

    errors = [
        {"doc_id": record["doc_id"], "split": record["split"], "error": record["error"]}
        for record in records
        if record["status"] == "error"
    ]
    metrics: dict[str, Any] = {
        "primary_metric": metric,
        "primary": {
            "metric": metric,
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


def _adapt_prediction(prediction: Any) -> Any:
    """Accept the real Uwazi property name for the UPR state relationship."""
    if isinstance(prediction, dict) and "state_under_review" not in prediction and "state_reviewed" in prediction:
        adapted = dict(prediction)
        adapted["state_under_review"] = prediction["state_reviewed"]
        return adapted
    return prediction


def _status_coverage(records: list[dict[str, Any]]) -> float | None:
    """Share of samples where the extractor produced anything."""
    if not records:
        return None
    produced = sum(1 for record in records if record["status"] != "missing_prediction")
    return produced / len(records)


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
    prompt_version: str,
    repair_rounds: int = 0,
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
        "prompt_version": prompt_version,
        "split_policy_version": manifest.get("split_policy_version"),
        "train_documents": train_documents,
        "test_documents": test_documents,
        "usage": usage,
        "created_at": created_at,
        "repair_rounds": repair_rounds,
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
    repair_rounds: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "method_id": method_id,
        "dataset_id": dataset_id,
        "split": "train_test",
        "status": status,
        "created_at": created_at,
        "counts": counts,
        "repair_rounds": repair_rounds,
        "rounds": [],
        "token_usage": usage,
        "split_policy_version": manifest.get("split_policy_version"),
    }


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:12]


__all__ = ["build_method", "build_run", "score_records"]

"""Evaluation for Plan field extraction and UPR recommendation extraction."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .normalization import (
    canonical_state,
    normalize_code,
    normalize_text,
    parse_number,
    token_f1,
)

PLAN_FIELD_KINDS = {
    "year": "number",
    "field_month": "number",
    "document_code": "code",
    "field_document_number": "code",
    "document_type": "label",
    "legal_status": "label",
    "means_of_adoption": "label",
    "body": "label",
}


def field_kind(field: str) -> str:
    return PLAN_FIELD_KINDS.get(field, "text")


def _as_value(prediction: Any, field: str, single_field: bool) -> Any:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        if field in prediction:
            return prediction[field]
        if isinstance(prediction.get("value"), dict):
            return prediction["value"].get(field)
        if single_field and "value" in prediction:
            return prediction["value"]
        return None
    if single_field:
        return prediction
    return None


def _normalized(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "number":
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return normalize_text(value)
    if kind == "code":
        return normalize_code(value)
    return normalize_text(value)


def evaluate_plan_sample(
    gold: dict[str, Any],
    prediction: Any,
    fields: list[str],
) -> dict[str, Any]:
    """Score one Plan sample. `gold` is keyed by field name."""
    single_field = len(fields) == 1
    field_results: dict[str, Any] = {}
    for field in fields:
        gold_value = gold.get(field)
        pred_value = _as_value(prediction, field, single_field)
        kind = field_kind(field)
        gold_norm = _normalized(gold_value, kind)
        pred_norm = _normalized(pred_value, kind)
        present = pred_value is not None and not (
            isinstance(pred_value, str) and not pred_value.strip()
        )
        hit = bool(present and gold_norm == pred_norm)
        field_results[field] = {
            "gold": gold_value,
            "prediction": pred_value,
            "gold_normalized": gold_norm,
            "prediction_normalized": pred_norm,
            "kind": kind,
            "present": present,
            "hit": hit,
            "token_f1": token_f1(gold_value, pred_value)
            if kind in {"label", "text", "code"}
            else None,
        }

    present_count = sum(item["present"] for item in field_results.values())
    hit_count = sum(item["hit"] for item in field_results.values())
    if present_count == 0:
        status = "missing_prediction"
    elif hit_count == len(fields):
        status = "hit"
    elif hit_count:
        status = "partial"
    else:
        status = "wrong"
    return {"status": status, "fields": field_results}


def _macro_f1(pairs: list[tuple[Any, Any]]) -> float | None:
    labels = {gold for gold, _ in pairs} | {pred for _, pred in pairs if pred is not None}
    if not labels:
        return None
    scores = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else None


def aggregate_plan(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    field_metrics: dict[str, Any] = {}
    for field in fields:
        pairs = []
        present = 0
        token_scores = []
        for record in records:
            result = record.get("details", {}).get("fields", {}).get(field)
            if not result:
                continue
            pairs.append((result.get("gold_normalized"), result.get("prediction_normalized")))
            present += int(result.get("present", False))
            if result.get("token_f1") is not None:
                token_scores.append(result["token_f1"])
        correct = sum(1 for gold, pred in pairs if gold == pred and pred is not None)
        field_metrics[field] = {
            "kind": field_kind(field),
            "total": len(records),
            "present": present,
            "correct": correct,
            "accuracy": correct / len(records) if records else 0.0,
            "macro_f1": _macro_f1(pairs),
            "token_f1": sum(token_scores) / len(token_scores) if token_scores else None,
        }

    accuracies = [item["accuracy"] for item in field_metrics.values()]
    f1_scores = [
        item["macro_f1"] for item in field_metrics.values() if item["macro_f1"] is not None
    ]
    return {
        "fields": field_metrics,
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "status_counts": dict(Counter(record["status"] for record in records)),
        "total": len(records),
    }


# ---------------------------------------------------------------------------
# UPR


def _canonical_countries(countries: Any) -> list[str]:
    if not isinstance(countries, list):
        return []
    return sorted({canonical_state(country) for country in countries if str(country).strip()})


def _prepare_recommendations(recs: Any) -> list[dict[str, Any]]:
    prepared = []
    if not isinstance(recs, list):
        return prepared
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        number = str(rec.get("number") or "").strip()
        text = rec.get("text") or ""
        if not number:
            continue
        parsed = parse_number(number)
        prepared.append(
            {
                "number": number,
                "base": parsed[1] if parsed else number,
                "text": str(text).strip(),
                "text_normalized": normalize_text(text),
                "countries": _canonical_countries(rec.get("countries")),
            }
        )
    return prepared


def _match_recommendations(
    gold_recs: list[dict[str, Any]],
    pred_recs: list[dict[str, Any]],
    threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy match predictions to gold within the same paragraph number."""
    gold_by_base: dict[str, list[dict[str, Any]]] = {}
    pred_by_base: dict[str, list[dict[str, Any]]] = {}
    for rec in gold_recs:
        gold_by_base.setdefault(rec["base"], []).append(rec)
    for rec in pred_recs:
        pred_by_base.setdefault(rec["base"], []).append(rec)

    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for base in set(gold_by_base) | set(pred_by_base):
        gold_group = list(gold_by_base.get(base, []))
        pred_group = list(pred_by_base.get(base, []))
        used_pred: set[int] = set()
        for gold in gold_group:
            best_index = None
            best_score = -1.0
            for index, pred in enumerate(pred_group):
                if index in used_pred:
                    continue
                score = token_f1(gold["text"], pred["text"])
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= threshold:
                pred = pred_group[best_index]
                used_pred.add(best_index)
                matches.append(
                    {
                        "base": base,
                        "gold_number": gold["number"],
                        "prediction_number": pred["number"],
                        "text_f1": best_score,
                        "number_exact": pred["number"] == gold["number"],
                        "country_match": pred["countries"] == gold["countries"],
                        "gold_countries": gold["countries"],
                        "prediction_countries": pred["countries"],
                    }
                )
            else:
                missed.append({"base": base, "number": gold["number"]})
        for index, pred in enumerate(pred_group):
            if index not in used_pred:
                extras.append({"base": base, "number": pred["number"]})
    return matches, missed, extras


def evaluate_upr_sample(gold: dict[str, Any], prediction: Any) -> dict[str, Any]:
    if prediction is None:
        prediction = {}
    if not isinstance(prediction, dict):
        prediction = {}
    gold_state = canonical_state(gold.get("state_under_review"))
    pred_state = canonical_state(prediction.get("state_under_review"))
    state_correct = bool(gold_state and pred_state and gold_state == pred_state)

    gold_recs = _prepare_recommendations(gold.get("recommendations", []))
    pred_recs = _prepare_recommendations(prediction.get("recommendations", []))
    matches, missed, extras = _match_recommendations(gold_recs, pred_recs)

    matched = len(matches)
    precision = matched / len(pred_recs) if pred_recs else 0.0
    recall = matched / len(gold_recs) if gold_recs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gold_bases = {rec["base"] for rec in gold_recs}
    pred_bases = {rec["base"] for rec in pred_recs}
    paragraph_tp = len(gold_bases & pred_bases)
    paragraph_precision = paragraph_tp / len(pred_bases) if pred_bases else 0.0
    paragraph_recall = paragraph_tp / len(gold_bases) if gold_bases else 0.0
    paragraph_f1 = (
        2 * paragraph_precision * paragraph_recall / (paragraph_precision + paragraph_recall)
        if paragraph_precision + paragraph_recall
        else 0.0
    )

    country_scores = [match["country_match"] for match in matches if match["gold_countries"]]
    text_scores = [match["text_f1"] for match in matches]
    exact_numbers = sum(1 for match in matches if match["number_exact"])

    if not gold_recs and not pred_recs:
        status = "wrong"
    elif state_correct and f1 >= 0.90:
        status = "hit"
    elif matched or state_correct:
        status = "partial"
    else:
        status = "wrong"
    if not pred_recs and not pred_state:
        status = "missing_prediction"

    return {
        "status": status,
        "state_correct": state_correct,
        "gold_state": gold_state,
        "prediction_state": pred_state,
        "metrics": {
            "gold_count": len(gold_recs),
            "prediction_count": len(pred_recs),
            "matched": matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "text_f1": sum(text_scores) / len(text_scores) if text_scores else 0.0,
            "country_accuracy": sum(country_scores) / len(country_scores)
            if country_scores
            else None,
            "exact_number_accuracy": exact_numbers / matched if matched else None,
            "paragraph_precision": paragraph_precision,
            "paragraph_recall": paragraph_recall,
            "paragraph_f1": paragraph_f1,
            "paragraph_tp": paragraph_tp,
            "country_correct": sum(country_scores),
            "country_total": len(country_scores),
            "exact_numbers": exact_numbers,
        },
        "matched": matches,
        "missed": missed,
        "extras": extras,
    }


def evaluate_upr_state_sample(gold: dict[str, Any], prediction: Any) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        prediction = {}
    gold_state = canonical_state(gold.get("state_under_review"))
    pred_state = canonical_state(prediction.get("state_under_review"))
    if not pred_state:
        status = "missing_prediction"
    elif gold_state and pred_state == gold_state:
        status = "hit"
    else:
        status = "wrong"
    return {
        "status": status,
        "state_correct": bool(gold_state and pred_state == gold_state),
        "gold_state": gold_state,
        "prediction_state": pred_state,
        "metrics": {},
    }


def aggregate_upr_state(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for record in records if record.get("details", {}).get("state_correct"))
    return {
        "documents": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "status_counts": dict(Counter(record["status"] for record in records)),
    }


def aggregate_upr(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    text_score_sum = 0.0
    text_score_count = 0
    country_correct = 0
    country_total = 0
    exact_numbers = 0
    doc_f1s = []
    for record in records:
        metrics = record.get("details", {}).get("metrics", {})
        totals["gold_count"] += metrics.get("gold_count", 0)
        totals["prediction_count"] += metrics.get("prediction_count", 0)
        totals["matched"] += metrics.get("matched", 0)
        totals["paragraph_tp"] += metrics.get("paragraph_tp", 0)
        if metrics.get("matched"):
            text_score_sum += metrics.get("text_f1", 0.0) * metrics["matched"]
            text_score_count += metrics["matched"]
        country_correct += metrics.get("country_correct", 0)
        country_total += metrics.get("country_total", 0)
        exact_numbers += metrics.get("exact_numbers", 0)
        doc_f1s.append(metrics.get("f1", 0.0))

    gold_count = totals["gold_count"]
    pred_count = totals["prediction_count"]
    matched = totals["matched"]
    precision = matched / pred_count if pred_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    state_correct = sum(1 for record in records if record.get("details", {}).get("state_correct"))
    return {
        "documents": len(records),
        "gold_recommendations": gold_count,
        "predicted_recommendations": pred_count,
        "matched_recommendations": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "text_f1": text_score_sum / text_score_count if text_score_count else 0.0,
        "country_accuracy": country_correct / country_total if country_total else None,
        "exact_number_accuracy": exact_numbers / matched if matched else None,
        "state_accuracy": state_correct / len(records) if records else 0.0,
        "doc_macro_f1": sum(doc_f1s) / len(doc_f1s) if doc_f1s else 0.0,
        "status_counts": dict(Counter(record["status"] for record in records)),
    }

"""Pure helpers for the opt-in admin-agent repair loop.

The loop is deliberately simple and stays in the main process:

1. split the train split into a feedback pool (visible to the agent) and a
   frozen validation slice (never sent to the authoring sidecar);
2. author once, execute on feedback + validation, score with our evaluator;
3. pick 2-3 failures from the feedback pool only, include the current script,
   and ask the nested extractor for a revised general function;
4. choose the best round by validation score, then run test once.

This module contains no Mongo, no LLM and no admin-agent imports. Gold is used
to pick a stratified validation slice and to describe failures; validation ids
are never placed in a repair prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

SEED = 42
DEFAULT_VALIDATION_SIZE = 10
DEFAULT_REPAIR_EXAMPLES = 3

#: Failure statuses in the order the loop prefers to show them.
_FAILURE_PRIORITY = ("error", "missing_prediction", "wrong", "partial")


# ---------------------------------------------------------------------------
# Stratified feedback/validation split


def _hash_rank(dataset_id: str, doc_id: str, seed: int = SEED) -> bytes:
    return hashlib.sha256(f"{seed}:repair:{dataset_id}:{doc_id}".encode()).digest()


def _gold_label(gold: dict[str, Any], field: str) -> str:
    value = gold.get(field)
    return str(value) if value is not None and str(value).strip() else "unknown"


def _stratum(dataset_id: str, gold: dict[str, Any]) -> str:
    """The dataset's own label family, so validation keeps the class mix."""
    if dataset_id in {
        "plan_document_type",
        "plan_legal_status",
        "plan_means_of_adoption",
        "plan_body",
    }:
        return _gold_label(gold, dataset_id.removeprefix("plan_"))
    if dataset_id == "plan_all_fields":
        return _gold_label(gold, "document_type")
    if dataset_id == "plan_year":
        try:
            return f"{int(gold.get('year')) // 10 * 10}s"
        except (TypeError, ValueError):
            return "unknown"
    if dataset_id == "plan_field_month":
        try:
            return f"month-{int(gold.get('field_month')):02d}"
        except (TypeError, ValueError):
            return "unknown"
    if dataset_id == "plan_document_code":
        code = str(gold.get("document_code") or "").strip().upper()
        if not code:
            return "unknown"
        if code[0] == "C" and code[1:].isdigit():
            return "ILO C"
        parts = code.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 and parts[0] else parts[0][:3]
    if dataset_id == "plan_field_document_number":
        text = str(gold.get("field_document_number") or "").strip()
        if not text:
            return "unknown"
        if text.isdigit():
            return "numeric"
        return "mixed" if any(character.isdigit() for character in text) else "text"
    if dataset_id in {"upr_recommendations", "upr_state"}:
        recommendations = gold.get("recommendations") or []
        count = len(recommendations)
        if count < 150:
            return "recs<150"
        if count < 200:
            return "recs150-199"
        if count < 250:
            return "recs200-249"
        if count < 300:
            return "recs250-299"
        return "recs300+"
    return "all"


@dataclass(frozen=True)
class RepairSplit:
    feedback: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    strata: dict[str, dict[str, int]]
    digest: str

    def validation_ids(self) -> list[str]:
        return [str(record["doc_id"]) for record in self.validation]

    def feedback_ids(self) -> list[str]:
        return [str(record["doc_id"]) for record in self.feedback]


def _proportional_quotas(sizes: dict[str, int], slots: int) -> dict[str, int]:
    """Largest-remainder quotas capped by each stratum's size."""
    total = sum(sizes.values())
    if slots <= 0 or not total:
        return {name: 0 for name in sizes}
    quotas: dict[str, int] = {}
    remainders = []
    for name, count in sizes.items():
        exact = slots * count / total
        quotas[name] = min(int(exact), count)
        remainders.append((-(exact - int(exact)), name))
    assigned = sum(quotas.values())
    ordered = sorted(remainders)
    # Largest remainder: at most one extra slot per stratum in the first pass.
    for _, name in ordered:
        if assigned >= slots:
            break
        if quotas[name] < sizes[name]:
            quotas[name] += 1
            assigned += 1
    # Capacity fallback (a stratum can be smaller than its ideal quota).
    while assigned < slots:
        moved = False
        for _, name in ordered:
            if assigned >= slots:
                break
            if quotas[name] < sizes[name]:
                quotas[name] += 1
                assigned += 1
                moved = True
        if not moved:
            break
    return quotas


def split_validation(
    dataset_id: str,
    records: list[dict[str, Any]],
    *,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    seed: int = SEED,
) -> RepairSplit:
    """Split train records into a feedback pool and a frozen validation slice.

    Strata are the dataset's own labels (document type, year decade, ...) and
    the slice is allocated proportionally, so validation mirrors the training
    distribution instead of over-representing rare classes. Singletons stay in
    the feedback pool unless they win a proportional slot. For small pools the
    slice is capped at ~20% so a spike still has feedback documents left.
    """
    unique = {str(record.get("doc_id")): record for record in records}
    ordered = [dict(unique[doc_id]) for doc_id in sorted(unique)]
    if len(ordered) < 2:
        raise ValueError("repair needs at least two train documents")
    size = min(int(validation_size), max(1, len(ordered) // 5))
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in ordered:
        groups.setdefault(_stratum(dataset_id, record.get("gold") or {}), []).append(record)
    for ids in groups.values():
        ids.sort(key=lambda record: _hash_rank(dataset_id, str(record["doc_id"]), seed))

    ordered_groups = sorted(groups.items(), key=lambda item: item[0])
    quotas = _proportional_quotas(
        {name: len(ids) for name, ids in ordered_groups}, size
    )
    chosen: list[dict[str, Any]] = []
    for name, ids in ordered_groups:
        chosen.extend(ids[: quotas.get(name, 0)])

    if len(chosen) < size:
        selected = {str(record["doc_id"]) for record in chosen}
        rest = sorted(
            (record for record in ordered if str(record["doc_id"]) not in selected),
            key=lambda record: _hash_rank(dataset_id, str(record["doc_id"]), seed),
        )
        chosen.extend(rest[: size - len(chosen)])

    validation_ids = {str(record["doc_id"]) for record in chosen[:size]}
    validation = [record for record in ordered if str(record["doc_id"]) in validation_ids]
    feedback = [record for record in ordered if str(record["doc_id"]) not in validation_ids]
    strata = {
        name: {
            "total": len(ids),
            "validation": sum(1 for record in ids if str(record["doc_id"]) in validation_ids),
        }
        for name, ids in sorted(groups.items())
    }
    digest = hashlib.sha256(json.dumps(sorted(validation_ids)).encode()).hexdigest()[:12]
    return RepairSplit(
        feedback=tuple(feedback),
        validation=tuple(validation),
        strata=strata,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Failure selection


def _first_line(text: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _excerpt(document: dict[str, Any] | None, needle: str | None = None) -> str:
    if not document:
        return ""
    segments = document.get("segments") or []
    text = " ".join(str(segment.get("text", "")) for segment in segments)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if needle:
        position = text.casefold().find(str(needle).casefold())
        if position >= 0:
            start = max(0, position - 160)
            return ("…" if start else "") + text[start : position + 260].strip() + "…"
    return _first_line(text)


def _plan_example(record: dict[str, Any], document: dict[str, Any] | None) -> dict[str, Any]:
    fields = record.get("details", {}).get("fields") or {}
    field_name = next(iter(fields), "")
    field = fields.get(field_name, {})
    gold = field.get("gold")
    prediction = field.get("prediction")
    status = record["status"]
    if status == "missing_prediction":
        family = f"missing:{gold}"
    elif status == "error":
        family = "error"
    else:
        family = f"{status}:{gold}->{prediction}"
    detail = (
        f"gold={gold!r}, prediction={prediction!r}, "
        f"normalized_gold={field.get('gold_normalized')!r}, "
        f"normalized_prediction={field.get('prediction_normalized')!r}"
    )
    if record.get("error"):
        detail += f", error={record['error']}"
    return {
        "doc_id": str(record["doc_id"]),
        "status": status,
        "family": family,
        "field": field_name,
        "detail": detail,
        "excerpt": _excerpt(document, str(gold) if gold is not None else None),
    }


def _upr_example(record: dict[str, Any], document: dict[str, Any] | None) -> dict[str, Any]:
    details = record.get("details") or {}
    metrics = details.get("metrics") or {}
    status = record["status"]
    state_correct = bool(details.get("state_correct"))
    missed = [str(item.get("number")) for item in details.get("missed") or []]
    extras = [str(item.get("number")) for item in details.get("extras") or []]
    if status == "missing_prediction":
        family = "missing_all"
    elif not state_correct:
        family = "state"
    elif missed:
        family = "records_missing"
    elif extras:
        family = "records_extra"
    else:
        family = status
    detail = (
        f"gold_recs={metrics.get('gold_count')}, predicted_recs={metrics.get('prediction_count')}, "
        f"matched={metrics.get('matched')}, gold_state={details.get('gold_state')!r}, "
        f"prediction_state={details.get('prediction_state')!r}, "
        f"missed_numbers={missed[:5]}, extra_numbers={extras[:5]}"
    )
    return {
        "doc_id": str(record["doc_id"]),
        "status": status,
        "family": family,
        "field": "state_under_review/recommendations",
        "detail": detail,
        "excerpt": _excerpt(document, "Recommendations"),
    }


def _upr_state_example(record: dict[str, Any], document: dict[str, Any] | None) -> dict[str, Any]:
    details = record.get("details") or {}
    status = record["status"]
    family = "state" if not details.get("state_correct") else status
    return {
        "doc_id": str(record["doc_id"]),
        "status": status,
        "family": family,
        "field": "state_under_review",
        "detail": (
            f"gold_state={details.get('gold_state')!r}, "
            f"prediction_state={details.get('prediction_state')!r}"
        ),
        "excerpt": _excerpt(document, details.get("gold_state")),
    }


def select_failures(
    dataset_id: str,
    scored_records: list[dict[str, Any]],
    *,
    feedback_ids: set[str] | frozenset[str],
    max_examples: int = DEFAULT_REPAIR_EXAMPLES,
    documents: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pick diverse failures from the feedback pool (never validation docs)."""
    candidates = [
        record
        for record in scored_records
        if str(record.get("doc_id")) in feedback_ids and record.get("status") != "hit"
    ]
    candidates.sort(key=lambda record: (record.get("doc_id"), record.get("status")))
    by_status: dict[str, list[dict[str, Any]]] = {status: [] for status in _FAILURE_PRIORITY}
    for record in candidates:
        status = str(record.get("status"))
        if status in by_status:
            by_status[status].append(record)

    if dataset_id in {"upr_recommendations", "upr_state"}:
        builder = _upr_state_example if dataset_id == "upr_state" else _upr_example
    else:
        builder = _plan_example

    examples: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for status in _FAILURE_PRIORITY:
        for record in by_status[status]:
            example = builder(record, (documents or {}).get(str(record["doc_id"])))
            if example["family"] in seen_families:
                continue
            seen_families.add(example["family"])
            examples.append(example)
            if len(examples) >= max_examples:
                return examples
    return examples


# ---------------------------------------------------------------------------
# Repair prompt


def build_repair_prompt(
    *,
    base_prompt: str,
    current_code: str,
    examples: list[dict[str, Any]],
    round_index: int,
    total_rounds: int,
) -> str:
    """Task prompt for one repair round. Contains no validation documents."""
    lines = [
        base_prompt.rstrip(),
        "",
        f"REPAIR ROUND {round_index} of {total_rounds}",
        "",
        "The function below is the one you authored in the previous round. It was executed",
        "over the training documents and makes the representative errors listed after it.",
        "Revise the function so it handles these failures as GENERAL rules.",
        "",
        "CURRENT FUNCTION:",
        "```python",
        current_code.rstrip(),
        "```",
        "",
        "REPRESENTATIVE TRAINING FAILURES (from the feedback pool; do not special-case these",
        "documents):",
    ]
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                "",
                f"{index}. shared_id={example['doc_id']} · status={example['status']} · "
                f"family={example['family']}",
                f"   field: {example['field']}",
                f"   {example['detail']}",
            ]
        )
        if example.get("excerpt"):
            lines.append(f"   excerpt: {example['excerpt']}")
    lines.extend(
        [
            "",
            "INSTRUCTIONS",
            "- Fix the underlying rule, not the example: never key on a shared_id, filename, or",
            "  an exact string copied from these documents.",
            "- Keep the rules that already work; remove dead rules instead of accumulating patches.",
            "- The failing documents are in the training evidence. If you need the full text, use",
            '  `peek_segmentation_by_entity(shared_id, "en")`; run `benchmark_extractor` to check',
            "  that the revision did not break other training documents.",
            "- Return the complete updated `def extract(text, ctx)` and honest samples_total /",
            "  samples_matched counts.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPAIR_EXAMPLES",
    "DEFAULT_VALIDATION_SIZE",
    "RepairSplit",
    "build_repair_prompt",
    "select_failures",
    "split_validation",
]

"""Build the document catalog and task datasets.

`uv run build-datasets` performs two steps:

1. snapshot every segmented English document into `artifacts/catalog.db`
2. create one dataset per extraction target under `artifacts/datasets/<id>/`

Datasets share one document store; a document is never copied per task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from pymongo.database import Database

from . import catalog
from .configuration import ARTIFACTS_DIR, DATASETS_DIR, STATES_PATH
from .db import (
    PLAN_DB,
    PLAN_DOCUMENT_TEMPLATE,
    UPR_DB,
    UPR_DOCUMENT_TEMPLATE,
    UPR_RECOMMENDATION_TEMPLATE,
    UPR_STATE_TEMPLATE,
    get_db,
    iter_entities,
    load_document,
)
from .models import metadata_labels, metadata_values
from .task_ids import V2, V3, base_dataset_id, dataset_version, versioned_dataset_id
from .text import ordered_segments, render_text

SEED = 42
TRAIN_SIZE = 50
SPLIT_POLICY_VERSION = "v2-50train"

#: v3 scales the same stratified template to a larger train split. A new seed
#: means v3 is an independent experiment, not a superset of the v2 ids.
V3_SEED = 43
V3_TRAIN_SIZE = 200
V3_VALIDATION_SIZE = 40
V3_SPLIT_POLICY_VERSION = "v3-200train"
#: Small pools (notably UPR: 164/165 eligible documents) cannot supply 200 train
#: documents AND a test split. Keep this many documents out of train so the
#: frozen test split is never empty.
V3_MIN_TEST = 40

DB_TEMPLATES = {
    PLAN_DB: PLAN_DOCUMENT_TEMPLATE,
    UPR_DB: UPR_DOCUMENT_TEMPLATE,
}


@dataclass(frozen=True)
class SplitPolicy:
    """One versioned split policy (seed, train size, id suffix)."""

    version: str
    split_policy_version: str
    seed: int
    train_size: int
    validation_size: int
    suffix: str
    min_test: int

    def dataset_id(self, base_id: str) -> str:
        return versioned_dataset_id(base_id, self.version)


SPLIT_POLICIES: dict[str, SplitPolicy] = {
    V2: SplitPolicy(
        version=V2,
        split_policy_version=SPLIT_POLICY_VERSION,
        seed=SEED,
        train_size=TRAIN_SIZE,
        validation_size=10,
        suffix="",
        min_test=0,
    ),
    V3: SplitPolicy(
        version=V3,
        split_policy_version=V3_SPLIT_POLICY_VERSION,
        seed=V3_SEED,
        train_size=V3_TRAIN_SIZE,
        validation_size=V3_VALIDATION_SIZE,
        suffix="_v3",
        min_test=V3_MIN_TEST,
    ),
}

# UPR tasks share one train pool: they ask different questions about the same
# documents, so training on different documents would make the comparison noisy.
SHARED_TRAIN_GROUPS = {
    "upr_state": "upr_shared",
    "upr_recommendations": "upr_shared",
}

SPLIT_RULE = (
    "Deterministic stratified sample of {train_size} train documents per dataset: order "
    "strata rarest-first, pick one document per stratum, then fill the remaining "
    "slots proportionally to stratum size. Within a stratum, documents are ordered "
    "by sha256(seed:dataset:sharedId). Test is every remaining eligible document, "
    "minus any document that is train in another dataset of the SAME split version "
    "(cross-dataset isolation; v2 and v3 do not exclude each other). "
    "upr_state and upr_recommendations share the same {train_size}-document pool. "
    "Pools with fewer than {min_pool} eligible documents keep at least {min_test} "
    "documents for test, so they get a smaller train split."
)

SPLIT_RULE_V2 = (
    "Deterministic stratified sample of 50 train documents per dataset: order "
    "strata rarest-first, pick one document per stratum, then fill the remaining "
    "slots proportionally to stratum size. Within a stratum, documents are ordered "
    "by sha256(seed:dataset:sharedId). Test is every remaining eligible document, "
    "minus any document that is train in another dataset (cross-dataset isolation). "
    "upr_state and upr_recommendations share the same 50-document pool."
)


def _hash_rank(seed: int, dataset_key: str, doc_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{dataset_key}:{doc_id}".encode()).digest()


def select_stratified(
    samples: Iterable[dict[str, Any]],
    *,
    dataset_key: str,
    stratum_key: Callable[[dict[str, Any]], str],
    train_size: int = TRAIN_SIZE,
    seed: int = SEED,
) -> list[str]:
    """Pick `train_size` document ids deterministically, stratified by `stratum_key`.

    Every stratum is represented once before any stratum gets a second document.
    The leftover slots are allocated proportionally to stratum size (largest
    remainder), and ties are broken by the stratum name. This is deterministic
    and covers rare labels without hand-curating documents.
    """
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(stratum_key(sample), []).append(str(sample["doc_id"]))
    for ids in groups.values():
        ids.sort(key=lambda doc_id: _hash_rank(seed, dataset_key, doc_id))

    ordered = sorted(groups.items(), key=lambda item: (len(item[1]), item[0]))
    chosen: list[str] = []
    for _, ids in ordered:
        if len(chosen) >= train_size:
            break
        chosen.append(ids[0])

    remaining = train_size - len(chosen)
    if remaining > 0:
        total = sum(len(ids) for _, ids in ordered)
        quotas = []
        for key, ids in ordered:
            exact = remaining * len(ids) / total
            quotas.append([key, int(exact), exact - int(exact)])
        assigned = sum(quota[1] for quota in quotas)
        for quota in sorted(quotas, key=lambda item: (-item[2], item[0]))[
            : remaining - assigned
        ]:
            quota[1] += 1
        extras = {quota[0]: quota[1] for quota in quotas}
        for key, ids in ordered:
            chosen.extend(ids[1 : 1 + min(extras[key], len(ids) - 1)])

    if len(chosen) < train_size:
        selected = set(chosen)
        rest = sorted(
            (doc_id for ids in groups.values() for doc_id in ids if doc_id not in selected),
            key=lambda doc_id: _hash_rank(seed, dataset_key, doc_id),
        )
        chosen.extend(rest[: train_size - len(chosen)])
    return sorted(chosen[:train_size])


def _year_stratum(value: Any) -> str:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return f"{(year // 10) * 10}s"


def _month_stratum(value: Any) -> str:
    try:
        return f"month-{int(value):02d}"
    except (TypeError, ValueError):
        return "unknown"


def _code_family(value: Any) -> str:
    code = str(value or "").strip().upper()
    if not code:
        return "unknown"
    if code[0] == "C" and code[1:].isdigit():
        return "ILO C"
    parts = code.split("/")
    if len(parts) >= 2 and parts[0]:
        return "/".join(parts[:2])
    return parts[0][:3] or "unknown"


def _number_family(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if text.isdigit():
        return "numeric"
    return "mixed" if any(character.isdigit() for character in text) else "text"


def _recommendation_stratum(gold: dict[str, Any]) -> str:
    recommendations = gold.get("recommendations") or []
    count = len(recommendations)
    if count < 150:
        bucket = "119-149"
    elif count < 200:
        bucket = "150-199"
    elif count < 250:
        bucket = "200-249"
    elif count < 300:
        bucket = "250-299"
    else:
        bucket = "300+"
    subnumbered = any(str(rec.get("number", "")).count(".") >= 2 for rec in recommendations)
    return f"{bucket}|sub={int(subnumbered)}"


def dataset_stratum(dataset_id: str, gold: dict[str, Any]) -> str:
    """Stratum definition for one dataset; labels/classes where they exist."""
    dataset_id = base_dataset_id(dataset_id)
    if dataset_id in {
        "plan_document_type",
        "plan_legal_status",
        "plan_means_of_adoption",
        "plan_body",
    }:
        field = dataset_id.removeprefix("plan_")
        return str(gold.get(field) or "unknown")
    if dataset_id == "plan_all_fields":
        return str(gold.get("document_type") or "unknown")
    if dataset_id == "plan_year":
        return _year_stratum(gold.get("year"))
    if dataset_id == "plan_field_month":
        return _month_stratum(gold.get("field_month"))
    if dataset_id == "plan_document_code":
        return _code_family(gold.get("document_code"))
    if dataset_id == "plan_field_document_number":
        return _number_family(gold.get("field_document_number"))
    if dataset_id == "upr_recommendations":
        return _recommendation_stratum(gold)
    if dataset_id == "upr_state":
        return "state"
    return "all"


def stratum_description(dataset_id: str) -> str:
    dataset_id = base_dataset_id(dataset_id)
    return {
        "plan_document_type": "Uwazi document type label",
        "plan_legal_status": "legal status label",
        "plan_means_of_adoption": "means of adoption label",
        "plan_body": "issuing body label",
        "plan_all_fields": "document type label",
        "plan_year": "decade (1940s … 2020s)",
        "plan_field_month": "month 1-12",
        "plan_document_code": "document-code family (A/HRC, A/RES, S/RES, E/CN.4, ILO C, ...)",
        "plan_field_document_number": "numeric / mixed / text document number",
        "upr_recommendations": "recommendation-count bucket × sub-numbered recommendations",
        "upr_state": "shared UPR pool: recommendation-count bucket × sub-numbered recommendations",
    }.get(dataset_id, "single stratum")


# ---------------------------------------------------------------------------
# Gold extraction


def _plan_label(row: dict[str, Any], prop: str) -> str | None:
    values = metadata_labels(row["metadata"], prop)
    return values[0] if values else None


def _plan_value(row: dict[str, Any], prop: str) -> Any | None:
    values = metadata_values(row["metadata"], prop)
    return values[0] if values else None


def plan_field_gold(
    prop: str, kind: str
) -> Callable[[dict[str, Any], Database], dict[str, Any] | None]:
    def build(row: dict[str, Any], db: Database) -> dict[str, Any] | None:
        value = _plan_label(row, prop) if kind == "label" else _plan_value(row, prop)
        if value is None:
            return None
        return {prop: value}

    return build


def plan_all_fields_gold(row: dict[str, Any], db: Database) -> dict[str, Any] | None:
    fields = {
        "document_type": _plan_label(row, "document_type"),
        "legal_status": _plan_label(row, "legal_status"),
        "means_of_adoption": _plan_label(row, "means_of_adoption"),
        "year": _plan_value(row, "year"),
        "document_code": _plan_value(row, "document_code"),
    }
    if any(value is None for value in fields.values()):
        return None
    return fields


def upr_state_gold(row: dict[str, Any], db: Database) -> dict[str, Any] | None:
    state = _plan_label(row, "state_reviewed")
    return {"state_under_review": state} if state else None


def _number_key(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", number)) or (0,)


def _load_recommendations(db: Database, rec_ids: list[str]) -> list[dict[str, Any]]:
    from bson import ObjectId

    recommendations: list[dict[str, Any]] = []
    for rec in db.entities.find(
        {
            "template": ObjectId(UPR_RECOMMENDATION_TEMPLATE),
            "language": "en",
            "sharedId": {"$in": rec_ids},
        },
        {
            "sharedId": 1,
            "title": 1,
            "metadata.recommendation": 1,
            "metadata.recommending_state": 1,
        },
    ):
        text = (metadata_values(rec.get("metadata"), "recommendation") or [""])[0]
        if not text.strip():
            continue
        recommendations.append(
            {
                "number": rec.get("title", ""),
                "text": text,
                "countries": metadata_labels(rec.get("metadata"), "recommending_state"),
            }
        )
    return sorted(recommendations, key=lambda rec: _number_key(rec["number"]))


def upr_recommendations_gold(row: dict[str, Any], db: Database) -> dict[str, Any] | None:
    state = _plan_label(row, "state_reviewed")
    rec_ids = metadata_values(row["metadata"], "related_recommendations")
    recommendations = _load_recommendations(db, list(rec_ids))
    if not state or not recommendations:
        return None
    return {"state_under_review": state, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# Dataset definitions


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    name: str
    task: str
    description: str
    db: str
    template: str
    target_schema: dict[str, Any]
    known_issues: tuple[str, ...]
    gold: Callable[[dict[str, Any], Database], dict[str, Any] | None]
    primary_metric: str


PLAN_KNOWN_ISSUES = (
    "Gold is human-entered Uwazi metadata; documents without this label are excluded, not treated as negatives.",
)
UPR_RECOMMENDATION_ISSUES = (
    "Only recommendations with non-empty text are included.",
    "One paragraph can contain several recommendations; gold may use sub-numbers such as 24.112.1 while the segmentation only shows 24.112.",
    "Gold recommendation text excludes the trailing country parenthetical and semicolon.",
    "Country labels differ from document spellings and need alias normalization.",
)


def _plan_spec(
    field_name: str,
    name: str,
    task: str,
    kind: str,
    description: str,
    primary_metric: str,
) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=f"plan_{field_name}",
        name=f"Plan — {name}",
        task=task,
        description=description,
        db=PLAN_DB,
        template=PLAN_DOCUMENT_TEMPLATE,
        target_schema={field_name: {"type": "number" if kind == "value" else "string"}},
        known_issues=PLAN_KNOWN_ISSUES,
        gold=plan_field_gold(field_name, kind),
        primary_metric=primary_metric,
    )


DATASETS: tuple[DatasetSpec, ...] = (
    _plan_spec(
        "document_type",
        "Document type",
        "single_label",
        "label",
        "Classify the document into one of the nine Uwazi document types.",
        "accuracy",
    ),
    _plan_spec(
        "legal_status",
        "Legal status",
        "single_label",
        "label",
        "Classify the legal status (binding / negotiated soft law / non-negotiated soft law).",
        "accuracy",
    ),
    _plan_spec(
        "means_of_adoption",
        "Means of adoption",
        "single_label",
        "label",
        "Classify how the document was adopted (vote, consensus, N.A.).",
        "accuracy",
    ),
    _plan_spec(
        "year",
        "Year",
        "numeric",
        "value",
        "Extract the four-digit year of the document.",
        "accuracy",
    ),
    _plan_spec(
        "document_code",
        "Document code",
        "string",
        "value",
        "Extract the official document symbol/code (e.g. A/HRC/RES/14/1).",
        "accuracy",
    ),
    _plan_spec(
        "body",
        "Body",
        "single_label",
        "label",
        "Identify the issuing body (General Assembly, Human Rights Council, special procedure, ...).",
        "accuracy",
    ),
    _plan_spec(
        "field_month",
        "Month",
        "numeric",
        "value",
        "Extract the month number (1–12) associated with the document.",
        "accuracy",
    ),
    _plan_spec(
        "field_document_number",
        "Document number",
        "string",
        "value",
        "Extract the document number field used by the partner (often a resolution/page number).",
        "accuracy",
    ),
    DatasetSpec(
        dataset_id="plan_all_fields",
        name="Plan — all core metadata",
        task="multi_field",
        description="Extract the five core Plan metadata fields in one script.",
        db=PLAN_DB,
        template=PLAN_DOCUMENT_TEMPLATE,
        target_schema={
            "document_type": {"type": "string"},
            "legal_status": {"type": "string"},
            "means_of_adoption": {"type": "string"},
            "year": {"type": "number"},
            "document_code": {"type": "string"},
        },
        known_issues=PLAN_KNOWN_ISSUES,
        gold=plan_all_fields_gold,
        primary_metric="accuracy",
    ),
    DatasetSpec(
        dataset_id="upr_state",
        name="UPR — State under review",
        task="single_label",
        description="Identify the state under review from the first page of a UPR report.",
        db=UPR_DB,
        template=UPR_DOCUMENT_TEMPLATE,
        target_schema={"state_under_review": {"type": "string"}},
        known_issues=(
            "The Uwazi state label can differ from the name printed on page 1; evaluation needs aliases.",
            "The entity title contains the state but is not included in the default document payload.",
        ),
        gold=upr_state_gold,
        primary_metric="accuracy",
    ),
    DatasetSpec(
        dataset_id="upr_recommendations",
        name="UPR — Recommendations",
        task="record_extraction",
        description="Extract the recommendation number, text, recommending countries, and the state under review.",
        db=UPR_DB,
        template=UPR_DOCUMENT_TEMPLATE,
        target_schema={
            "state_under_review": {"type": "string"},
            "recommendations": {
                "type": "list",
                "items": {
                    "number": {"type": "string"},
                    "text": {"type": "string"},
                    "countries": {"type": "list", "items": {"type": "string"}},
                },
            },
        },
        known_issues=UPR_RECOMMENDATION_ISSUES,
        gold=upr_recommendations_gold,
        primary_metric="f1",
    ),
)


def dataset_specs(version: str = V2) -> tuple[DatasetSpec, ...]:
    """Return the task specs for one split version.

    v2 keeps the original dataset ids so existing rows/artifacts are untouched.
    v3 adds a ``_v3`` suffix and a version-labelled display name in the same
    catalog/artifacts root.
    """
    if version == V2:
        return DATASETS
    policy = SPLIT_POLICIES.get(version)
    if policy is None:
        raise ValueError(f"unknown split version {version!r}; choose from {tuple(SPLIT_POLICIES)}")
    return tuple(
        replace(
            spec,
            dataset_id=policy.dataset_id(spec.dataset_id),
            name=f"{spec.name} ({version}, {policy.train_size} train)",
        )
        for spec in DATASETS
    )


# ---------------------------------------------------------------------------
# Document snapshot


def _document_rows() -> Iterable[dict[str, Any]]:
    for db_name, template_id in DB_TEMPLATES.items():
        db = get_db(db_name)
        for entity in iter_entities(db, template_id=template_id, language="en"):
            document = load_document(db, entity)
            if not document.has_segmentation:
                continue
            text = render_text(document, order="layout")
            segments = [segment.as_dict() for segment in ordered_segments(document, order="layout")]
            yield {
                "db": db_name,
                "doc_id": document.shared_id,
                "language": document.language,
                "template": document.template,
                "title": document.title,
                "page_count": document.page_count,
                "segment_count": len(segments),
                "char_count": len(text),
                "token_estimate": max(1, len(text) // 4),
                "file_id": document.file_id,
                "filename": document.filename,
                "mimetype": document.mimetype,
                "text": text,
                "segments_json": json.dumps(segments, ensure_ascii=False),
                "metadata_json": json.dumps(document.metadata, ensure_ascii=False, default=str),
            }


def build_documents() -> int:
    catalog.reset()
    conn = catalog.connect()
    catalog.init_schema(conn)
    count = 0
    started = time.time()
    batch: list[dict[str, Any]] = []
    for row in _document_rows():
        batch.append(row)
        if len(batch) >= 250:
            count += catalog.insert_documents(conn, batch)
            batch.clear()
            print(f"  documents: {count:,} ({time.time() - started:.1f}s)", flush=True)
    if batch:
        count += catalog.insert_documents(conn, batch)
    print(f"  documents: {count:,} total ({time.time() - started:.1f}s)", flush=True)
    return count


# ---------------------------------------------------------------------------
# State list


def build_states() -> int:
    """Write the canonical UPR state/country list used by evaluation and prompts."""
    db = get_db(UPR_DB)
    states = []
    for entity in iter_entities(db, template_id=UPR_STATE_TEMPLATE, language="en"):
        label = entity.get("title", "")
        if not label or label == "_Unknown":
            continue
        codes = metadata_values(entity.get("metadata"), "country_code")
        states.append({"label": label, "code": codes[0] if codes else ""})
    states.sort(key=lambda item: item["label"])
    STATES_PATH.write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  states: {len(states)} candidates -> {STATES_PATH}", flush=True)
    return len(states)


# ---------------------------------------------------------------------------
# Dataset building


def _doc_rows(conn: Any, db_name: str, template_id: str) -> Iterable[dict[str, Any]]:
    for row in conn.execute(
        "SELECT db, doc_id, title, metadata_json FROM documents WHERE db = ? AND template = ?",
        (db_name, template_id),
    ):
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        yield item


def _label_distribution(spec: DatasetSpec, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if spec.task == "record_extraction":
        states = Counter()
        total = 0
        subnumbered = 0
        per_doc = []
        for sample in samples:
            recs = sample["gold"].get("recommendations", [])
            per_doc.append(len(recs))
            total += len(recs)
            states[sample["gold"].get("state_under_review", "")] += 1
            for rec in recs:
                if rec["number"].count(".") >= 2:
                    subnumbered += 1
        return {
            "recommendations_total": total,
            "recommendations_per_document": {
                "min": min(per_doc) if per_doc else 0,
                "max": max(per_doc) if per_doc else 0,
                "avg": round(sum(per_doc) / len(per_doc), 1) if per_doc else 0,
            },
            "recommendations_with_subnumber": subnumbered,
            "states": dict(states.most_common()),
        }

    distribution: dict[str, Any] = {}
    for target in spec.target_schema:
        counter = Counter()
        for sample in samples:
            value = sample["gold"].get(target)
            if value is not None:
                counter[str(value)] += 1
        distribution[target] = dict(counter.most_common(50))
    return distribution


def _effective_train_size(available: int, policy: SplitPolicy) -> int:
    """Requested train size, but never consume the whole small pool."""
    if available <= 0:
        return 0
    return max(1, min(policy.train_size, available - policy.min_test))


def select_train_ids(
    samples_by_dataset: dict[str, list[dict[str, Any]]],
    *,
    version: str = V2,
) -> dict[str, list[str]]:
    """Select the train ids per dataset for one split version.

    The UPR pool is shared inside the version only; a v2 dataset never influences
    a v3 selection (and vice versa).
    """
    policy = SPLIT_POLICIES[version]
    shared: dict[str, list[str]] = {}
    for group in sorted(set(SHARED_TRAIN_GROUPS.values())):
        members = [
            versioned_dataset_id(name, version)
            for name, group_name in SHARED_TRAIN_GROUPS.items()
            if group_name == group
        ]
        members = [dataset_id for dataset_id in members if dataset_id in samples_by_dataset]
        if not members:
            continue
        eligible_sets = [
            {sample["doc_id"] for sample in samples_by_dataset[dataset_id]}
            for dataset_id in members
        ]
        eligible = set.intersection(*eligible_sets)
        # Use the richer recommendation gold for the strata of both UPR tasks.
        base_dataset = (
            versioned_dataset_id("upr_recommendations", version)
            if versioned_dataset_id("upr_recommendations", version) in members
            else members[0]
        )
        candidates = [
            sample
            for sample in samples_by_dataset[base_dataset]
            if sample["doc_id"] in eligible
        ]
        shared[group] = select_stratified(
            candidates,
            dataset_key=f"{group}:{base_dataset}",
            stratum_key=lambda sample, base_dataset=base_dataset: dataset_stratum(
                base_dataset, sample["gold"]
            ),
            train_size=_effective_train_size(len(candidates), policy),
            seed=policy.seed,
        )

    train_ids: dict[str, list[str]] = {}
    for dataset_id, samples in samples_by_dataset.items():
        base_id = base_dataset_id(dataset_id)
        group = SHARED_TRAIN_GROUPS.get(base_id)
        if group is not None:
            train_ids[dataset_id] = list(shared[group])
            continue
        train_ids[dataset_id] = select_stratified(
            samples,
            dataset_key=dataset_id,
            stratum_key=lambda sample, dataset_id=dataset_id: dataset_stratum(
                dataset_id, sample["gold"]
            ),
            train_size=_effective_train_size(len(samples), policy),
            seed=policy.seed,
        )
    return train_ids


def assign_splits(
    samples_by_dataset: dict[str, list[dict[str, Any]]],
    train_ids: dict[str, list[str]],
    db_by_dataset: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Assign train/test and drop cross-dataset train documents from test.

    A document is identified by (db, sharedId, split version). Any document
    selected as train in one dataset is excluded from test in every other
    dataset of the same database AND the same split version, so no test document
    was seen while writing examples or repairing a script for another task. v2
    and v3 are separate experiments, so they do not exclude each other.
    """
    global_train: dict[tuple[str, str], set[str]] = {}
    for dataset_id, ids in train_ids.items():
        key = (db_by_dataset[dataset_id], dataset_version(dataset_id))
        global_train.setdefault(key, set()).update(ids)

    assigned: dict[str, list[dict[str, Any]]] = {}
    excluded_counts: dict[str, int] = {}
    for dataset_id, samples in samples_by_dataset.items():
        train = set(train_ids[dataset_id])
        train_elsewhere = global_train[(db_by_dataset[dataset_id], dataset_version(dataset_id))]
        rows: list[dict[str, Any]] = []
        excluded = 0
        for sample in samples:
            doc_id = str(sample["doc_id"])
            if doc_id in train:
                split = "train"
            elif doc_id in train_elsewhere:
                excluded += 1
                continue
            else:
                split = "test"
            rows.append({**sample, "split": split})
        rows.sort(key=lambda row: (row["split"], row["doc_id"]))
        assigned[dataset_id] = rows
        excluded_counts[dataset_id] = excluded
    return assigned, excluded_counts


def _write_dataset(
    spec: DatasetSpec,
    samples: list[dict[str, Any]],
    train_ids: list[str],
    excluded: int,
    created_at: str,
    *,
    version: str = V2,
) -> dict[str, Any]:
    policy = SPLIT_POLICIES[version]
    splits = Counter(sample["split"] for sample in samples)
    train_count = splits.get("train", 0)
    validation_count = min(policy.validation_size, max(0, train_count - 1))
    manifest = {
        "dataset_id": spec.dataset_id,
        "name": spec.name,
        "task": spec.task,
        "description": spec.description,
        "db": spec.db,
        "template": spec.template,
        "target_schema": spec.target_schema,
        "known_issues": list(spec.known_issues),
        "primary_metric": spec.primary_metric,
        "dataset_version": version,
        "split_policy_version": policy.split_policy_version,
        "split_policy": {
            "version": policy.split_policy_version,
            "dataset_version": version,
            "seed": policy.seed,
            "train_size": policy.train_size,
            "train_size_effective": train_count,
            "validation_size": validation_count,
            "min_test": policy.min_test,
            "test": (
                "all remaining eligible documents, minus documents that are train in "
                "another dataset of the same split version"
            ),
            "rule": (
                SPLIT_RULE_V2
                if version == V2
                else SPLIT_RULE.format(
                    train_size=policy.train_size,
                    min_pool=policy.train_size + policy.min_test,
                    min_test=policy.min_test,
                )
            ),
            "strata": stratum_description(spec.dataset_id),
        },
        "counts": {
            "total": len(samples),
            "train": train_count,
            "test": splits.get("test", 0),
            "train_examples": 2,
            "validation": validation_count,
            "train_feedback": train_count - validation_count,
            "excluded_train_overlap": excluded,
        },
        "train_doc_ids": sorted(train_ids),
        "label_distribution": _label_distribution(spec, samples),
        "created_at": created_at,
    }

    directory = DATASETS_DIR / spec.dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return manifest


def build_datasets(
    *,
    version: str = V2,
    only: set[str] | None = None,
) -> dict[str, int]:
    """Build one split version additively.

    ``build_datasets(version="v3")`` only inserts/overwrites ``*_v3`` dataset
    rows; v2 rows and artifact directories are not touched. The document table
    must already exist (the CLI enforces that for v3).
    """
    specs = dataset_specs(version)
    if only:
        known = {spec.dataset_id for spec in specs}
        unknown = sorted(only - known)
        if unknown:
            raise SystemExit(f"Unknown {version} dataset ids: {unknown}")
        specs = tuple(spec for spec in specs if spec.dataset_id in only)
    if not specs:
        return {}

    conn = catalog.connect()
    catalog.init_schema(conn)
    created_at = datetime.now(UTC).isoformat()
    db_cache = {db_name: get_db(db_name) for db_name in DB_TEMPLATES}
    summary: dict[str, int] = {}

    samples_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        started = time.time()
        mongo_db = db_cache[spec.db]
        samples: list[dict[str, Any]] = []
        for row in _doc_rows(conn, spec.db, spec.template):
            gold = spec.gold(row, mongo_db)
            if gold is None:
                continue
            samples.append({"doc_id": row["doc_id"], "gold": gold})
        samples_by_dataset[spec.dataset_id] = samples
        print(f"  eligible {spec.dataset_id:<28} {len(samples):>6,} ({time.time() - started:.1f}s)")

    train_ids = select_train_ids(samples_by_dataset, version=version)
    assigned, excluded_counts = assign_splits(
        samples_by_dataset,
        train_ids,
        {spec.dataset_id: spec.db for spec in specs},
    )

    for spec in specs:
        samples = assigned[spec.dataset_id]
        manifest = _write_dataset(
            spec,
            samples,
            train_ids[spec.dataset_id],
            excluded_counts[spec.dataset_id],
            created_at,
            version=version,
        )
        catalog.insert_dataset(
            conn,
            {
                "dataset_id": spec.dataset_id,
                "db": spec.db,
                "name": spec.name,
                "task": spec.task,
                "description": spec.description,
                "target_schema": spec.target_schema,
                "known_issues": list(spec.known_issues),
                "manifest": manifest,
                "created_at": created_at,
                "split_policy_version": manifest["split_policy_version"],
                "dataset_version": version,
            },
            samples,
        )
        summary[spec.dataset_id] = len(samples)
        counts = manifest["counts"]
        print(
            f"  {spec.dataset_id:<28} {len(samples):>6,} samples "
            f"(train {counts['train']:,} / test {counts['test']:,} / "
            f"validation {counts['validation']:,} / "
            f"excluded overlap {counts['excluded_train_overlap']:,})",
            flush=True,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the document catalog and task datasets.")
    parser.add_argument(
        "--skip-documents",
        action="store_true",
        help="reuse the existing catalog document table instead of rebuilding it",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="only snapshot documents; do not build datasets",
    )
    parser.add_argument(
        "--split-version",
        choices=sorted(SPLIT_POLICIES),
        default=V2,
        help=(
            "dataset split version to build. v2 is the original 50-train split; v3 is an "
            "additive 200-train split with *_v3 ids. v3 always reuses the existing catalog "
            "documents and never resets/rebuilds v2 rows."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated dataset ids (base ids are accepted for the chosen version)",
    )
    args = parser.parse_args(argv)
    version = args.split_version
    if version != V2 and not args.skip_documents:
        print(
            f"{version} is additive: reusing the existing catalog documents instead of "
            "rebuilding/resetting the catalog.",
            flush=True,
        )
        args.skip_documents = True

    print("Building document catalog…")
    if not args.skip_documents:
        build_documents()
    else:
        conn = catalog.connect()
        catalog.init_schema(conn)
        print(f"  reusing {catalog.count_documents(conn):,} documents")

    if args.documents_only:
        build_states()
        return 0

    only = None
    if args.datasets:
        requested = {item.strip() for item in args.datasets.split(",") if item.strip()}
        only = {versioned_dataset_id(item, version) for item in requested}

    build_states()
    print(f"Building {version} datasets into {ARTIFACTS_DIR / 'datasets'}…")
    summary = build_datasets(version=version, only=only)
    print(f"Built {len(summary)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

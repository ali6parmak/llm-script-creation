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
from dataclasses import dataclass
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
from .text import ordered_segments, render_text

SEED = 42
SPLIT_POLICIES = {
    PLAN_DB: {"train": 0.70, "dev": 0.15, "test": 0.15},
    UPR_DB: {"train": 0.60, "dev": 0.20, "test": 0.20},
}
DB_TEMPLATES = {
    PLAN_DB: PLAN_DOCUMENT_TEMPLATE,
    UPR_DB: UPR_DOCUMENT_TEMPLATE,
}


def split_for(db: str, doc_id: str, seed: int = SEED) -> str:
    """Deterministic global split assignment for a document."""
    policy = SPLIT_POLICIES[db]
    digest = hashlib.sha256(f"{db}:{doc_id}:{seed}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < policy["train"]:
        return "train"
    if position < policy["train"] + policy["dev"]:
        return "dev"
    return "test"


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
        "exact_match",
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
        "exact_match",
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
        primary_metric="macro_accuracy",
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
        primary_metric="recommendation_f1",
    ),
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


def _write_dataset(
    spec: DatasetSpec, samples: list[dict[str, Any]], created_at: str
) -> dict[str, Any]:
    splits = Counter(sample["split"] for sample in samples)
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
        "split_policy": {"seed": SEED, "fractions": SPLIT_POLICIES[spec.db]},
        "counts": {
            "total": len(samples),
            **{split: splits.get(split, 0) for split in ("train", "dev", "test")},
        },
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


def build_datasets() -> dict[str, int]:
    conn = catalog.connect()
    catalog.init_schema(conn)
    created_at = datetime.now(UTC).isoformat()
    db_cache = {db_name: get_db(db_name) for db_name in DB_TEMPLATES}
    summary: dict[str, int] = {}

    for spec in DATASETS:
        started = time.time()
        mongo_db = db_cache[spec.db]
        samples: list[dict[str, Any]] = []
        for row in _doc_rows(conn, spec.db, spec.template):
            gold = spec.gold(row, mongo_db)
            if gold is None:
                continue
            samples.append(
                {
                    "doc_id": row["doc_id"],
                    "split": split_for(spec.db, row["doc_id"]),
                    "gold": gold,
                }
            )
        manifest = _write_dataset(spec, samples, created_at)
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
            },
            samples,
        )
        summary[spec.dataset_id] = len(samples)
        counts = manifest["counts"]
        print(
            f"  {spec.dataset_id:<28} {len(samples):>6,} samples "
            f"(train {counts['train']:,} / dev {counts['dev']:,} / test {counts['test']:,}) "
            f"{time.time() - started:.1f}s",
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
    args = parser.parse_args(argv)

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

    build_states()
    print(f"Building datasets into {ARTIFACTS_DIR / 'datasets'}…")
    summary = build_datasets()
    print(f"Built {len(summary)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

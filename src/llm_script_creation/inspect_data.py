"""Inspect the restored `plan` and `upr` Uwazi databases.

Run with::

    uv run inspect-data            # or: uv run python -m llm_script_creation.inspect_data

Writes `reports/data_report.md` and `reports/data_report.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import ObjectId

from .configuration import REPORTS_DIR
from .db import (
    PLAN_DB,
    PLAN_DOCUMENT_TEMPLATE,
    UPR_DB,
    UPR_DOCUMENT_TEMPLATE,
    UPR_RECOMMENDATION_TEMPLATE,
    UPR_STATE_TEMPLATE,
    find_document_file,
    get_db,
    iter_entities,
    load_document,
    load_segmentation,
)
from .models import metadata_items, metadata_labels, metadata_values
from .text import ordered_segments

# Recommendation numbering inside a document: "96.1", "24.112", "24.112.1".
# It may occur mid-segment because several recommendations can share one
# segmentation cell.
RECOMMENDATION_NUMBER_RE = re.compile(r"(?<![\d.])([1-9]\d{0,2}\.\d{1,3})(?:\.(\d{1,3}))?(?![\d.])")
# The same number as used in Recommendation entity titles.
RECOMMENDATION_TITLE_RE = re.compile(r"^\s*([1-9]\d{0,2}\.\d{1,3})(?:\.(\d{1,3}))?\s*$")


def recommendation_base(number: str) -> str | None:
    """Return the paragraph-level number for a recommendation title/number."""
    match = RECOMMENDATION_TITLE_RE.match(number or "")
    return match.group(1) if match else None


def recommendation_numbers_in_text(text: str) -> list[str]:
    return [match.group(0) for match in RECOMMENDATION_NUMBER_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Small helpers


def _oid(template_id: str) -> ObjectId:
    return ObjectId(template_id)


def _template_properties(db: Any, template_id: str) -> list[dict[str, str]]:
    template = db.templates.find_one({"_id": _oid(template_id)})
    if template is None:
        return []
    return [
        {"name": p.get("name", ""), "label": p.get("label", ""), "type": p.get("type", "")}
        for p in template.get("properties", [])
    ]


def _coverage(
    db: Any, template_id: str, entity_language: str, file_language: str
) -> dict[str, int]:
    """Count documents that own a file / a segmented file for one language."""
    entity_filter = {"template": _oid(template_id), "language": entity_language}
    total = db.entities.count_documents(entity_filter)
    shared_ids = set(db.entities.distinct("sharedId", entity_filter))

    files_any = set(
        db.files.distinct("entity", {"type": "document", "entity": {"$in": list(shared_ids)}})
    )
    files_lang = set(
        db.files.distinct(
            "entity",
            {"type": "document", "language": file_language, "entity": {"$in": list(shared_ids)}},
        )
    )
    segmented_file_ids = db.segmentations.distinct("fileID", {"status": "ready"})
    segmented_files = {
        str(row["_id"]): row
        for row in db.files.find(
            {
                "_id": {"$in": segmented_file_ids},
                "type": "document",
                "entity": {"$in": list(shared_ids)},
            },
            {"entity": 1, "language": 1},
        )
    }
    segmented_any = {row["entity"] for row in segmented_files.values()}
    segmented_lang = {
        row["entity"] for row in segmented_files.values() if row.get("language") == file_language
    }
    return {
        "documents": total,
        "with_document_file": len(shared_ids & files_any),
        "with_document_file_in_language": len(shared_ids & files_lang),
        "with_segmentation": len(shared_ids & segmented_any),
        "with_segmentation_in_language": len(shared_ids & segmented_lang),
    }


def _non_empty_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in items:
        value = item.get("value")
        label = item.get("label")
        if isinstance(value, str) and not value.strip():
            value = None
        if value is None and not label:
            continue
        cleaned.append(item)
    return cleaned


def _metadata_stats(db: Any, template_id: str, language: str) -> dict[str, Any]:
    properties = _template_properties(db, template_id)
    total = db.entities.count_documents({"template": _oid(template_id), "language": language})
    stats: dict[str, Any] = {"documents": total, "properties": {}}

    counters: dict[str, Counter[str]] = {p["name"]: Counter() for p in properties}
    samples: dict[str, list[str]] = {p["name"]: [] for p in properties}
    numbers: dict[str, list[float]] = {p["name"]: [] for p in properties}

    for entity in iter_entities(
        db,
        template_id=template_id,
        language=language,
        projection={"metadata": 1},
    ):
        metadata = entity.get("metadata") or {}
        for prop in properties:
            name = prop["name"]
            items = _non_empty_items(metadata_items(metadata, name))
            if not items:
                continue
            counters[name]["__present__"] += 1
            if prop["type"] in {"numeric", "date"}:
                for item in items:
                    value = item.get("value")
                    if isinstance(value, (int, float)):
                        numbers[name].append(float(value))
            if prop["type"] in {"select", "multiselect", "relationship"}:
                for item in items:
                    label = item.get("label") or item.get("value")
                    if label:
                        counters[name][str(label)] += 1
            else:
                for item in items:
                    value = item.get("value")
                    if value is None:
                        continue
                    if len(samples[name]) < 8:
                        samples[name].append(str(value))
                    counters[name][str(value)[:120]] += 1

    for prop in properties:
        name = prop["name"]
        present = counters[name].get("__present__", 0)
        entry: dict[str, Any] = {
            "label": prop["label"],
            "type": prop["type"],
            "present": present,
            "coverage": round(present / total, 4) if total else 0.0,
        }
        if numbers[name]:
            entry["numeric"] = {
                "min": min(numbers[name]),
                "max": max(numbers[name]),
                "distinct": len(set(numbers[name])),
            }
        if prop["type"] in {"select", "multiselect", "relationship"}:
            top = [
                {"label": label, "count": count}
                for label, count in counters[name].most_common(10)
                if label != "__present__"
            ]
            entry["top_labels"] = top
            entry["distinct_labels"] = len(counters[name]) - (
                1 if "__present__" in counters[name] else 0
            )
        else:
            entry["samples"] = samples[name]
            entry["distinct_values"] = len(counters[name]) - (
                1 if "__present__" in counters[name] else 0
            )
        stats["properties"][name] = entry
    return stats


# ---------------------------------------------------------------------------
# Plan


def _plan_sample_documents(db: Any, limit: int = 3) -> list[dict[str, Any]]:
    """Pick a few well-populated documents with different issuing bodies."""
    samples: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()
    for entity in iter_entities(db, template_id=PLAN_DOCUMENT_TEMPLATE, language="en"):
        metadata = entity.get("metadata") or {}
        if not _non_empty_items(metadata_items(metadata, "document_type")):
            continue
        if not _non_empty_items(metadata_items(metadata, "document_code")):
            continue
        if not _non_empty_items(metadata_items(metadata, "year")):
            continue
        body = (metadata_labels(metadata, "body") or [""])[0]
        if body in seen_bodies:
            continue
        file_doc = find_document_file(db, entity["sharedId"], "eng")
        if file_doc is None:
            continue
        segmentation = load_segmentation(db, file_doc["_id"])
        if segmentation is None:
            continue
        document = load_document(db, entity)
        if not document.has_segmentation:
            continue
        seen_bodies.add(body)
        first_page = [s for s in ordered_segments(document, order="layout") if s.page == 1][:12]
        samples.append(
            {
                "shared_id": document.shared_id,
                "title": document.title,
                "metadata": {
                    prop: metadata_labels(document.metadata, prop)
                    or metadata_values(document.metadata, prop)
                    for prop in (
                        "body",
                        "document_type",
                        "legal_status",
                        "means_of_adoption",
                        "year",
                        "field_month",
                        "document_code",
                        "field_document_info",
                        "field_document_number",
                    )
                },
                "first_page_segments": [
                    {"type": s.type, "top": s.top, "left": s.left, "text": s.text}
                    for s in first_page
                ],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _plan_report(db: Any) -> dict[str, Any]:
    by_template = []
    for row in db.entities.aggregate(
        [{"$group": {"_id": "$template", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}]
    ):
        template = db.templates.find_one({"_id": row["_id"]})
        by_template.append(
            {
                "template": template.get("name") if template else "unknown",
                "id": str(row["_id"]),
                "entities": row["n"],
            }
        )

    languages = {
        row["_id"]: row["n"]
        for row in db.entities.aggregate(
            [
                {"$match": {"template": _oid(PLAN_DOCUMENT_TEMPLATE)}},
                {"$group": {"_id": "$language", "n": {"$sum": 1}}},
            ]
        )
    }

    return {
        "collections": {
            name: db.get_collection(name).estimated_document_count()
            for name in db.list_collection_names()
        },
        "entities_by_template": by_template,
        "document_languages": languages,
        "coverage": _coverage(db, PLAN_DOCUMENT_TEMPLATE, "en", "eng"),
        "property_stats": _metadata_stats(db, PLAN_DOCUMENT_TEMPLATE, "en"),
        "samples": _plan_sample_documents(db),
        "segment_types": {
            row["_id"]: row["n"]
            for row in db.segmentations.aggregate(
                [
                    {"$unwind": "$segmentation.paragraphs"},
                    {"$group": {"_id": "$segmentation.paragraphs.type", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                ]
            )
        },
    }


# ---------------------------------------------------------------------------
# UPR


def _recommendations_by_doc(db: Any, doc_shared_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Load English recommendations linked to the given documents."""
    doc_ids = set(doc_shared_ids)
    linked: dict[str, list[str]] = {}
    for entity in db.entities.find(
        {
            "template": _oid(UPR_DOCUMENT_TEMPLATE),
            "language": "en",
            "sharedId": {"$in": list(doc_ids)},
        },
        {"sharedId": 1, "metadata.related_recommendations": 1},
    ):
        ids = metadata_values(entity.get("metadata"), "related_recommendations")
        linked[entity["sharedId"]] = list(ids)

    all_ids = {rec_id for ids in linked.values() for rec_id in ids}
    recs: dict[str, dict[str, Any]] = {}
    for rec in db.entities.find(
        {
            "template": _oid(UPR_RECOMMENDATION_TEMPLATE),
            "language": "en",
            "sharedId": {"$in": list(all_ids)},
        },
        {
            "sharedId": 1,
            "title": 1,
            "metadata.recommendation": 1,
            "metadata.recommending_state": 1,
            "metadata.state_under_review": 1,
            "metadata.response": 1,
            "metadata.issues": 1,
            "metadata.paragraph_number": 1,
        },
    ):
        recs[rec["sharedId"]] = {
            "shared_id": rec["sharedId"],
            "title": rec.get("title", ""),
            "recommendation": metadata_values(rec.get("metadata"), "recommendation"),
            "recommending_states": metadata_labels(rec.get("metadata"), "recommending_state"),
            "state_under_review": metadata_labels(rec.get("metadata"), "state_under_review"),
            "response": metadata_labels(rec.get("metadata"), "response"),
            "issues": metadata_labels(rec.get("metadata"), "issues"),
            "paragraph_number": metadata_values(rec.get("metadata"), "paragraph_number"),
        }
    return {doc_id: [recs[rid] for rid in ids if rid in recs] for doc_id, ids in linked.items()}


def _upr_sample_document(db: Any, shared_id: str = "z702fzq18s") -> dict[str, Any]:
    entity = db.entities.find_one({"sharedId": shared_id, "language": "en"})
    if entity is None:
        return {}
    document = load_document(db, entity)
    recs = _recommendations_by_doc(db, [shared_id]).get(shared_id, [])

    segments = ordered_segments(document, order="layout")
    first_page = [s for s in segments if s.page == 1][:14]
    raw_recommendations = []
    for index, segment in enumerate(segments):
        matches = list(RECOMMENDATION_NUMBER_RE.finditer(segment.text))
        for position, match in enumerate(matches):
            number = match.group(1)
            end = (
                matches[position + 1].start() if position + 1 < len(matches) else len(segment.text)
            )
            body = segment.text[match.end() : end].strip()
            # A number cell in a table is followed by its text cell.
            if not body and index + 1 < len(segments):
                body = segments[index + 1].text.strip()
            gold = [rec for rec in recs if recommendation_base(rec["title"]) == number]
            raw_recommendations.append(
                {
                    "number": number,
                    "segment_type": segment.type,
                    "page": segment.page,
                    "top": segment.top,
                    "left": segment.left,
                    "text": body,
                    "ground_truth_count": len(gold),
                    "ground_truth": gold[0] if gold else None,
                }
            )
    missing_ground_truth = [r["number"] for r in raw_recommendations if not r["ground_truth"]]
    return {
        "shared_id": document.shared_id,
        "title": document.title,
        "state_reviewed": metadata_labels(document.metadata, "state_reviewed"),
        "cycle": metadata_labels(document.metadata, "cycle_of_the_document"),
        "session": metadata_labels(document.metadata, "session_of_the_document"),
        "pages": document.page_count,
        "recommendation_entities": len(recs),
        "recommendation_numbers_in_segmentation": len(raw_recommendations),
        "recommendation_numbers_missing_ground_truth": missing_ground_truth[:20],
        "recommendations_sharing_a_segment": sum(
            1 for r in raw_recommendations if r["ground_truth_count"] > 1
        ),
        "first_page_segments": [
            {"type": s.type, "top": s.top, "left": s.left, "text": s.text} for s in first_page
        ],
        "recommendations_sample": raw_recommendations[:8],
    }


def _upr_report(db: Any) -> dict[str, Any]:
    template_names = {}
    by_template = []
    for row in db.entities.aggregate(
        [{"$group": {"_id": "$template", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}]
    ):
        template = db.templates.find_one({"_id": row["_id"]})
        name = template.get("name") if template else "unknown"
        template_names[str(row["_id"])] = name
        by_template.append({"template": name, "id": str(row["_id"]), "entities": row["n"]})

    doc_coverage = _coverage(db, UPR_DOCUMENT_TEMPLATE, "en", "eng")

    segmented_docs: list[dict[str, Any]] = []
    for entity in iter_entities(db, template_id=UPR_DOCUMENT_TEMPLATE, language="en"):
        document = load_document(db, entity)
        if document.has_segmentation:
            segmented_docs.append(document)

    linked = _recommendations_by_doc(db, [doc.shared_id for doc in segmented_docs])
    recommendation_stats = Counter()
    recs_per_doc: list[int] = []
    raw_number_counts: list[int] = []
    missing_in_segmentation = 0
    segments_with_multiple_numbers = 0
    for document in segmented_docs:
        recs = linked.get(document.shared_id, [])
        recs_per_doc.append(len(recs))
        numbers = [rec["title"] for rec in recs if RECOMMENDATION_TITLE_RE.match(rec["title"])]
        bases = {recommendation_base(number) for number in numbers}
        raw_bases = {
            match.group(1)
            for segment in document.segments
            for match in RECOMMENDATION_NUMBER_RE.finditer(segment.text)
        }
        raw_number_counts.append(len(raw_bases))
        missing_in_segmentation += len(bases - raw_bases)
        segments_with_multiple_numbers += sum(
            1
            for segment in document.segments
            if len(RECOMMENDATION_NUMBER_RE.findall(segment.text)) > 1
        )
        for rec in recs:
            recommendation_stats["total"] += 1
            if rec["recommendation"]:
                recommendation_stats["with_text"] += 1
            title_match = RECOMMENDATION_TITLE_RE.match(rec["title"])
            if title_match:
                recommendation_stats["with_number_in_title"] += 1
                if title_match.group(2):
                    recommendation_stats["with_subnumber"] += 1
            if rec["recommending_states"]:
                recommendation_stats["with_recommending_state"] += 1
            if len(rec["recommending_states"]) > 1:
                recommendation_stats["multiple_recommending_states"] += 1
            if rec["state_under_review"]:
                recommendation_stats["with_state_under_review"] += 1
            if rec["response"]:
                recommendation_stats["with_response"] += 1

    states = [
        {
            "label": entity.get("title", ""),
            "code": (metadata_values(entity.get("metadata"), "country_code") or [""])[0],
        }
        for entity in iter_entities(db, template_id=UPR_STATE_TEMPLATE, language="en")
    ]

    return {
        "collections": {
            name: db.get_collection(name).estimated_document_count()
            for name in db.list_collection_names()
        },
        "entities_by_template": by_template,
        "document_coverage": doc_coverage,
        "segmented_documents": len(segmented_docs),
        "recommendation_stats": dict(recommendation_stats),
        "recommendations_per_document": {
            "min": min(recs_per_doc) if recs_per_doc else 0,
            "max": max(recs_per_doc) if recs_per_doc else 0,
            "avg": round(sum(recs_per_doc) / len(recs_per_doc), 1) if recs_per_doc else 0,
        },
        "recommendation_numbers_in_segmentation": {
            "min": min(raw_number_counts) if raw_number_counts else 0,
            "max": max(raw_number_counts) if raw_number_counts else 0,
            "avg": round(sum(raw_number_counts) / len(raw_number_counts), 1)
            if raw_number_counts
            else 0,
            "ground_truth_numbers_missing_from_segmentation": missing_in_segmentation,
            "segments_containing_multiple_numbers": segments_with_multiple_numbers,
        },
        "country_list": {"count": len(states), "sample": states[:8]},
        "sample_document": _upr_sample_document(db),
        "segment_types": {
            row["_id"]: row["n"]
            for row in db.segmentations.aggregate(
                [
                    {"$unwind": "$segmentation.paragraphs"},
                    {"$group": {"_id": "$segmentation.paragraphs.type", "n": {"$sum": 1}}},
                    {"$sort": {"n": -1}},
                ]
            )
        },
    }


# ---------------------------------------------------------------------------
# Markdown rendering


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def _render_plan_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "## Plan International (`plan`)",
        "",
        f"- Document entities (English): **{coverage['documents']:,}**",
        f"- With a document file: **{coverage['with_document_file']:,}**",
        f"- With an English document file: **{coverage['with_document_file_in_language']:,}**",
        f"- With a ready segmentation: **{coverage['with_segmentation']:,}**",
        f"- With an English segmentation: **{coverage['with_segmentation_in_language']:,}**",
        "",
        "### Entities by template",
        "",
        _md_table(
            ["Template", "ID", "Entities"],
            [
                [r["template"], f"`{r['id']}`", f"{r['entities']:,}"]
                for r in report["entities_by_template"]
            ],
        ),
        "",
        "### Document properties (English)",
        "",
    ]
    rows = []
    for name, prop in report["property_stats"]["properties"].items():
        rows.append(
            [
                f"`{name}`",
                prop["type"],
                f"{prop['present']:,}",
                f"{prop['coverage']:.1%}",
                prop.get("distinct_labels", prop.get("distinct_values", "")),
            ]
        )
    lines.append(_md_table(["Property", "Type", "Filled", "Coverage", "Distinct"], rows))
    lines += ["", "### Sample documents", ""]
    for sample in report["samples"]:
        lines += [f"#### {sample['title']}", "", f"`{sample['shared_id']}`", ""]
        lines.append("Ground-truth metadata:")
        lines += [f"- `{prop}`: {values}" for prop, values in sample["metadata"].items()]
        lines += ["", "First page segments:", ""]
        for segment in sample["first_page_segments"][:10]:
            text = " ".join(segment["text"].split())[:180]
            lines.append(f"- `{segment['type']}` @ {segment['top']:.0f}: {text}")
        lines.append("")
    return "\n".join(lines)


def _render_upr_markdown(report: dict[str, Any]) -> str:
    coverage = report["document_coverage"]
    stats = report["recommendation_stats"]
    lines = [
        "## UPR (`upr`)",
        "",
        f"- Document entities (English): **{coverage['documents']:,}**",
        f"- With a document file: **{coverage['with_document_file']:,}**",
        f"- With an English document file: **{coverage['with_document_file_in_language']:,}**",
        f"- With a ready segmentation: **{coverage['with_segmentation']:,}**",
        f"- Segmented documents: **{report['segmented_documents']:,}**",
        "",
        "### Entities by template",
        "",
        _md_table(
            ["Template", "ID", "Entities"],
            [
                [r["template"], f"`{r['id']}`", f"{r['entities']:,}"]
                for r in report["entities_by_template"]
            ],
        ),
        "",
        "### Recommendation statistics (English, linked to segmented documents)",
        "",
        _md_table(
            ["Metric", "Count"],
            [
                ["Recommendation entities", f"{stats.get('total', 0):,}"],
                ["With recommendation text", f"{stats.get('with_text', 0):,}"],
                ["With number in title", f"{stats.get('with_number_in_title', 0):,}"],
                ["With sub-number (e.g. 24.112.1)", f"{stats.get('with_subnumber', 0):,}"],
                ["With recommending state", f"{stats.get('with_recommending_state', 0):,}"],
                [
                    "With multiple recommending states",
                    f"{stats.get('multiple_recommending_states', 0):,}",
                ],
                ["With state under review", f"{stats.get('with_state_under_review', 0):,}"],
                ["With response", f"{stats.get('with_response', 0):,}"],
            ],
        ),
        "",
        f"- Recommendations per document: {report['recommendations_per_document']}",
        f"- Ground-truth numbers missing from the segmentation: {report['recommendation_numbers_in_segmentation']['ground_truth_numbers_missing_from_segmentation']:,}",
        f"- Segments containing more than one recommendation number: {report['recommendation_numbers_in_segmentation']['segments_containing_multiple_numbers']:,}",
        f"- Country list (English `State` entities): **{report['country_list']['count']:,}**",
        "",
        "### Sample segmented document",
        "",
    ]
    sample = report["sample_document"]
    if sample:
        lines += [
            f"#### {sample['title']} (`{sample['shared_id']}`)",
            "",
            f"- State reviewed: {sample['state_reviewed']}",
            f"- Cycle: {sample['cycle']} / session: {sample['session']}",
            f"- Pages: {sample['pages']}",
            f"- Recommendation entities: {sample['recommendation_entities']}",
            f"- Recommendation numbers found in segmentation: {sample['recommendation_numbers_in_segmentation']}",
            f"- Numbers missing ground truth (first 20): {sample['recommendation_numbers_missing_ground_truth']}",
            f"- Numbers whose paragraph has several recommendations: {sample['recommendations_sharing_a_segment']}",
            "",
            "First page:",
            "",
        ]
        for segment in sample["first_page_segments"]:
            text = " ".join(segment["text"].split())[:180]
            lines.append(f"- `{segment['type']}` @ {segment['top']:.0f}: {text}")
        lines += ["", "Recommendation alignment sample:", ""]
        for rec in sample["recommendations_sample"]:
            truth = rec["ground_truth"] or {}
            lines.append(
                f"- **{rec['number']}** (segment `{rec['segment_type']}` p{rec['page']}) "
                f"raw: {' '.join(rec['text'].split())[:140]}"
            )
            if truth:
                gold_text = (
                    " ".join(truth["recommendation"][0].split())[:120]
                    if truth["recommendation"]
                    else ""
                )
                count = rec["ground_truth_count"]
                suffix = f" (1 of {count} recommendations on this paragraph)" if count > 1 else ""
                lines.append(
                    f"  - gold{suffix}: {gold_text} "
                    f"| by: {truth['recommending_states']} | response: {truth['response']}"
                )
        lines.append("")
    return "\n".join(lines)


def _render_markdown(reports: dict[str, Any]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Uwazi data inspection report",
        "",
        f"Generated: {generated}",
        "",
        "Database: `mongodb://localhost:27017` — databases `plan` and `upr`.",
        "",
        "Join chain: `entities.sharedId` → `files.entity` → `segmentations.fileID`.",
        "",
    ]
    if "plan" in reports:
        lines.append(_render_plan_markdown(reports["plan"]))
        lines.append("")
    if "upr" in reports:
        lines.append(_render_upr_markdown(reports["upr"]))
        lines.append("")
    lines += [
        "## Data notes / gotchas",
        "",
        "- Uwazi entities are stored once per document language. Plan has `en`, `fr`, `es`; UPR has `en` and `fr`.",
        "- A single entity can own several files, one per language. Select the file whose `language` matches the entity (`en` → `eng`, `fr` → `fra`, `es` → `spa`).",
        "- Segmentations attach to files, not entities. Not every document has one.",
        "- Recommendation numbers and recommendation text can live in two adjacent segments (a `Table` cell with the number, then a `Text` cell with the body).",
        "- One segmentation cell can contain several recommendations. When several recommendations share a paragraph number, Uwazi numbers them `24.112.1`, `24.112.2`, ...; the segmentation only shows `24.112`.",
        "- UPR recommendation ground truth uses labels such as `Cote d'Ivoire`; document text may use `Côte d’Ivoire`, so country matching needs aliases/fuzzy matching.",
        "- The English `State` list contains 196 entities, but one is `_Unknown`; use the 195 real countries with ISO codes as the candidate list.",
        "- Uwazi state labels can differ from the name printed on page 1 (`Cape Verde`/`Cabo Verde`, `Macedonia FYR`/`North Macedonia`, `Laos`/`Lao People's Democratic Republic`, ...). 14 of the 165 segmented documents do not contain the label verbatim on page 1; 10 need aliases after accent/punctuation normalization, so state evaluation needs an alias map.",
        "- Existing metadata is the only ground truth we have; it is incomplete and occasionally wrong. Fields with low coverage should not be treated as negatives without review.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def build_reports(which: str = "all") -> dict[str, Any]:
    reports: dict[str, Any] = {}
    if which in {"all", "plan"}:
        reports["plan"] = _plan_report(get_db(PLAN_DB))
    if which in {"all", "upr"}:
        reports["upr"] = _upr_report(get_db(UPR_DB))
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", choices=["plan", "upr", "all"], default="all", help="database to inspect"
    )
    parser.add_argument(
        "--out", type=Path, default=REPORTS_DIR / "data_report.md", help="Markdown output path"
    )
    parser.add_argument(
        "--json-out", type=Path, default=REPORTS_DIR / "data_report.json", help="JSON output path"
    )
    args = parser.parse_args(argv)

    reports = build_reports(args.db)
    markdown = _render_markdown(reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.json_out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(markdown)
    print(f"\nWrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

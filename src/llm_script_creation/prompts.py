"""Prompt construction for LLM-written extraction scripts."""

from __future__ import annotations

import json
from typing import Any

from .evaluation import field_kind
from .normalization import load_states

SYSTEM_PROMPT = """You write one deterministic Python extraction script for Uwazi documents.

Rules:
- Output exactly one Python code block. No explanation before or after it.
- The file must define exactly one top-level function: `def extract(document: dict) -> dict`.
- Use only the Python standard library modules: re, json, unicodedata, collections, itertools,
  functools, string, datetime, math, statistics, typing, dataclasses, decimal, html.
- Do not use file access, network access, subprocesses, `eval`, `exec`, `open`, or imports
  outside that list.
- Do not print anything. Return only a JSON-serializable dict.
"""


def _field_description(field: str) -> str:
    return {
        "document_type": "The Uwazi document type (one of the supplied labels).",
        "legal_status": "The legal status (one of the supplied labels).",
        "means_of_adoption": "How the document was adopted (one of the supplied labels).",
        "year": "Four-digit year of the document.",
        "document_code": "Official document symbol/code, e.g. A/HRC/RES/14/1 or S/RES/2068.",
        "body": "Issuing body (use one of the supplied labels when possible).",
        "field_month": "Month number 1-12.",
        "field_document_number": "Document number as used by the partner (often a resolution number).",
        "state_under_review": "The state under review, read from the first page.",
        "recommendations": "List of recommendations extracted from the recommendation section.",
    }.get(field, field)


def _candidate_labels(manifest: dict[str, Any]) -> dict[str, list[str]]:
    distribution = manifest.get("label_distribution", {})
    candidates: dict[str, list[str]] = {}
    for field, values in distribution.items():
        if not isinstance(values, dict) or "recommendations_total" in distribution:
            continue
        if field_kind(field) != "label":
            continue
        cleaned = []
        seen = set()
        for label in values:
            label = str(label).strip()
            if label and label not in seen:
                seen.add(label)
                cleaned.append(label)
        candidates[field] = cleaned
    return candidates


def _task_instructions(dataset: dict[str, Any]) -> str:
    manifest = dataset.get("manifest", {})
    dataset_id = dataset["dataset_id"]
    fields = list(
        dataset.get("target_schema", {}) or json.loads(dataset.get("target_schema_json", "{}"))
    )
    lines = [
        f"DATASET: {manifest.get('name', dataset_id)}",
        f"TASK: {manifest.get('description', '')}",
        "",
        "FIELDS:",
    ]
    for field in fields:
        lines.append(f"- {field}: {_field_description(field)}")

    if dataset_id == "upr_recommendations":
        lines += [
            "",
            "OUTPUT SHAPE:",
            '{"state_under_review": "<state name>",',
            ' "recommendations": [{"number": "<number>", "text": "<text>", "countries": ["<country>"]}]}',
            "",
            "RECOMMENDATION RULES:",
            "- The recommendation section usually starts after a sentence such as "
            '"The following recommendations will be examined by <state>".',
            "- Recommendation paragraphs are separated by newlines, not necessarily by blank lines.",
            "- Scan the whole recommendation section for numbers like 96.1 or 24.112; do not only "
            "look at the beginning of a line. The text between one number and the next belongs to "
            "the recommendation.",
            "- One paragraph can contain several recommendations separated by ';'. Split them.",
            "- If several recommendations share the same paragraph number, number them "
            "24.112.1, 24.112.2, ... in reading order.",
            "- Each recommendation normally ends with a country in parentheses before the "
            "semicolon, e.g. '(New Zealand);'. Put that country in `countries` and remove the "
            "parenthetical and trailing semicolon from `text`.",
            "- Recommendations can be split across lines; join wrapped lines.",
            "- Do not invent numbers or recommendations. If the document has no recommendation "
            "section, return an empty list.",
            "- Practical algorithm: find the marker 'recommendations will be examined', then scan "
            "the rest of the text for number matches like r'(?<!\\d)(\\d{1,3}\\.\\d{1,3})(?!\\d)'. "
            "Everything between two numbers belongs to one recommendation paragraph. Inside that "
            "chunk, split on ');' if there are several '(country);' endings. The last parenthetical "
            "before the semicolon is the recommending country.",
        ]
    elif dataset_id == "upr_state":
        lines += [
            "",
            "OUTPUT SHAPE:",
            '{"state_under_review": "<state name>"}',
            "",
            "Read the state from the first page. In UPR reports the state usually appears on its "
            "own line immediately after the line containing 'Report of the Working Group on the "
            "Universal Periodic Review' (sometimes with an asterisk). Strip asterisks/footnote "
            "marks. If that fails, use the shortest section-header-like line on page 1 that is "
            "not a UN organ/session heading. The document title is not provided.",
        ]
    elif len(fields) == 1:
        lines += [
            "",
            "OUTPUT SHAPE:",
            '{"value": <string, number, or null>}',
            "",
            "Return null only if the value cannot be found.",
        ]
    else:
        lines += ["", "OUTPUT SHAPE:", json.dumps({field: None for field in fields}, indent=2)]

    if dataset_id == "plan_year":
        lines += [
            "",
            "YEAR HINT:",
            "- Prefer the document's own adoption/publication date on the first page, e.g. a line "
            "such as 'Distr.: General 2 July 2021', '30 June 2021', or 'adopted ... 30 June 2021'.",
            "- Do not simply return the first year in the whole text: opening paragraphs often cite "
            "older resolutions, conventions, and historical instruments.",
        ]
    if dataset_id == "plan_document_code":
        lines += [
            "",
            "CODE HINT:",
            "- The code often appears in the page header/footer (e.g. A/HRC/RES/14/1, S/RES/2068, "
            "A/RES/50/195, C098) and may have spaces around slashes in the extracted text.",
        ]

    candidates = _candidate_labels(manifest)
    if candidates:
        lines += ["", "CANDIDATE LABELS (prefer these; labels are human-entered and may vary):"]
        for field, labels in candidates.items():
            lines.append(f"- {field}: {json.dumps(labels, ensure_ascii=False)}")

    if dataset_id == "upr_recommendations":
        lines += [
            "",
            "STATE/COUNTRY LIST (Uwazi labels; document spellings may differ):",
        ]
        for state in load_states():
            suffix = f" ({state['code']})" if state.get("code") else ""
            lines.append(f"- {state['label']}{suffix}")
    return "\n".join(lines)


def _excerpt_from_text(text: str, dataset_id: str, limit: int = 4500) -> str:
    if dataset_id == "upr_recommendations":
        marker = text.lower().find("recommendations will be examined")
        if marker >= 0:
            start = max(0, marker - 500)
            return text[start : start + limit]
    return text[:limit]


def _segments_excerpt(
    segments: list[dict[str, Any]],
    dataset_id: str,
    *,
    include_position: bool,
    include_type: bool,
    limit: int = 18,
) -> list[dict[str, Any]]:
    start = 0
    if dataset_id == "upr_recommendations":
        import re

        for index, segment in enumerate(segments):
            if re.search(r"(?<![\d.])([1-9]\d{0,2}\.\d{1,3})(?![\d.])", segment.get("text", "")):
                start = max(0, index - 2)
                break
    selected = segments[start : start + limit]
    output = []
    for segment in selected:
        item = {"text": " ".join(str(segment.get("text", "")).split())}
        if include_type:
            item["type"] = segment.get("type")
        if include_position:
            item.update(
                page=segment.get("page"),
                top=segment.get("top"),
                left=segment.get("left"),
                width=segment.get("width"),
                height=segment.get("height"),
            )
        output.append(item)
    return output


def _example_block(
    example: dict[str, Any],
    dataset_id: str,
    fields: list[str],
    representation: str,
    include_position: bool,
    include_type: bool,
) -> str:
    document = example["document"]
    gold = example["gold"]
    if dataset_id == "upr_recommendations" and len(gold.get("recommendations", [])) > 8:
        gold = dict(gold)
        gold["recommendations"] = gold["recommendations"][:8]
        gold["_note"] = "example truncated to 8 recommendations"
    elif dataset_id not in {"upr_state", "upr_recommendations"} and len(fields) == 1:
        gold = {"value": gold.get(fields[0])}
    if representation == "text":
        body = _excerpt_from_text(document["text"], dataset_id)
        input_block = f"```text\n{body}\n```"
    else:
        segments = _segments_excerpt(
            json.loads(document["segments_json"]),
            dataset_id,
            include_position=include_position,
            include_type=include_type,
        )
        input_block = f"```json\n{json.dumps(segments, ensure_ascii=False, indent=1)}\n```"
    return (
        f"EXAMPLE {example['doc_id']}\n"
        f"Input excerpt (truncated; the real script receives the full document):\n{input_block}\n"
        f"Expected output:\n```json\n{json.dumps(gold, indent=2, ensure_ascii=False)}\n```"
    )


def build_messages(
    dataset: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    representation: str = "text",
    include_position: bool = False,
    include_type: bool = False,
) -> list[dict[str, str]]:
    dataset_id = dataset["dataset_id"]
    if representation == "text":
        input_description = (
            "`document` has keys: `id`, `language`, `page_count`, `text` (full document text, "
            "segments joined in reading order). There is no `title` key."
        )
    else:
        input_description = (
            "`document` has keys: `id`, `language`, `page_count`, `segments`. `segments` is a "
            "list of objects with `text`"
            + (", `type`" if include_type else "")
            + (", `page`, `top`, `left`, `width`, `height`" if include_position else "")
            + ". There is no `title` key."
        )
    user_prompt = "\n\n".join(
        [
            _task_instructions(dataset),
            "DOCUMENT INPUT:\n" + input_description,
            "EXAMPLES:\n"
            + "\n\n".join(
                _example_block(
                    example,
                    dataset_id,
                    list(dataset.get("target_schema", {})),
                    representation,
                    include_position,
                    include_type,
                )
                for example in examples
            ),
            "Write the complete `extract(document)` script now.",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

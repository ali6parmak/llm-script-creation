"""Prompt construction for LLM-written extraction scripts."""

from __future__ import annotations

import json
from typing import Any

from .evaluation import field_kind
from .input_spec import InputSpec, payload_excerpt, prepare_input
from .models import Document
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


def prompt_focuses(dataset_id: str) -> tuple[str, ...]:
    """Strings used to center a prompt excerpt inside the transformed input."""
    return {
        "upr_recommendations": (
            "recommendations will be examined",
            "Recommendations",
        ),
        "upr_state": ("Report of the Working Group",),
        "plan_year": ("Distr.: General", "adopted"),
        "plan_document_code": ("A/", "S/", "C0", "E/"),
    }.get(dataset_id, ())


def example_payload(document: Document, spec: InputSpec) -> dict[str, Any]:
    """Payload a prompt example is built from; identical to execution."""
    return prepare_input(document, spec)


def _example_block(
    example: dict[str, Any],
    dataset_id: str,
    fields: list[str],
    spec: InputSpec,
) -> str:
    document: Document = example["document"]
    gold = example["gold"]
    if dataset_id == "upr_recommendations" and len(gold.get("recommendations", [])) > 8:
        gold = dict(gold)
        gold["recommendations"] = gold["recommendations"][:8]
        gold["_note"] = "example truncated to 8 recommendations"
    elif dataset_id not in {"upr_state", "upr_recommendations"} and len(fields) == 1:
        gold = {"value": gold.get(fields[0])}

    payload = example_payload(document, spec)
    body, truncated = payload_excerpt(
        payload,
        spec,
        focuses=prompt_focuses(dataset_id),
    )
    language = "text" if spec.representation == "text" else "json"
    total = (
        len(payload["text"])
        if spec.representation == "text"
        else len(payload.get("segments") or [])
    )
    unit = "characters" if spec.representation == "text" else "segments"
    shown = len(body) if spec.representation == "text" else len(json.loads(body))
    if truncated:
        label = (
            f"Excerpt of the transformed input ({shown:,} of {total:,} {unit}; the script "
            f"receives {spec.describe()})."
        )
    else:
        label = (
            f"Transformed input ({total:,} {unit}; this is exactly what the script receives: "
            f"{spec.describe()})."
        )
    input_block = f"```{language}\n{body}\n```"
    return (
        f"EXAMPLE {example['doc_id']}\n"
        f"{label}\n{input_block}\n"
        f"Expected output:\n```json\n{json.dumps(gold, indent=2, ensure_ascii=False)}\n```"
    )


def build_messages(
    dataset: dict[str, Any],
    examples: list[dict[str, Any]],
    spec: InputSpec | None = None,
) -> list[dict[str, str]]:
    dataset_id = dataset["dataset_id"]
    spec = spec or InputSpec()
    if spec.representation == "text":
        input_description = (
            "`document` has keys: `id`, `language`, `page_count`, `text`. The text is "
            f"{spec.describe()} (segments joined in reading order); there is no `title` key."
        )
    else:
        input_description = (
            "`document` has keys: `id`, `language`, `page_count`, `segments`. `segments` is a "
            f"list of objects: {spec.describe()}. There is no `title` key."
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
                    spec,
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

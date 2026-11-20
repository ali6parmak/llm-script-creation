"""Small deterministic baselines used to validate the evaluation plumbing."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .normalization import (
    ANY_NUMBER_RE,
    canonical_state,
    load_states,
    normalize_text,
    strip_trailing_countries,
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DOC_CODE_RE = re.compile(r"\b[A-Z]{1,4}(?:/[A-Z0-9][A-Z0-9./-]*)+\b")
ILO_CODE_RE = re.compile(r"\bC\d{2,4}\b")


@dataclass(frozen=True)
class Baseline:
    method_id: str
    name: str
    description: str
    dataset_id: str
    function: Callable[[dict[str, Any]], dict[str, Any]]


def _text(payload: dict[str, Any]) -> str:
    return payload.get("text", "")


def _first_page_text(payload: dict[str, Any]) -> str:
    segments = payload.get("segments") or []
    return " ".join(segment.get("text", "") for segment in segments if segment.get("page") == 1)


def plan_year(payload: dict[str, Any]) -> dict[str, Any]:
    text = _text(payload)
    head = text[:4000]
    match = YEAR_RE.search(head) or YEAR_RE.search(text)
    return {"value": int(match.group(0)) if match else None}


def plan_document_code(payload: dict[str, Any]) -> dict[str, Any]:
    text = re.sub(r"\s*/\s*", "/", _text(payload))
    head = text[:6000]
    for pattern in (DOC_CODE_RE, ILO_CODE_RE):
        match = pattern.search(head)
        if match:
            return {"value": match.group(0).strip(".,;:")}
    return {"value": None}


def _state_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for state in load_states():
        label = state["label"]
        candidates.append((normalize_text(label), label))
    # Official names that differ from the Uwazi short label.
    from .normalization import COUNTRY_ALIASES

    for alias, label in COUNTRY_ALIASES.items():
        candidates.append((normalize_text(alias), label))
    return candidates


def upr_state(payload: dict[str, Any]) -> dict[str, Any]:
    page = _first_page_text(payload)
    normalized = f" {normalize_text(page)} "
    candidates = _state_candidates()
    # Prefer the longest matching name; this avoids "Congo" beating
    # "Democratic Republic of Congo".
    matches = [
        (len(key), label)
        for key, label in candidates
        if key and re.search(rf"\b{re.escape(key)}\b", normalized)
    ]
    if not matches:
        return {"state_under_review": None}
    label = max(matches)[1]
    return {"state_under_review": canonical_state(label)}


def upr_recommendations(payload: dict[str, Any]) -> dict[str, Any]:
    text = _text(payload)
    marker = re.search(r"recommendations?\s+will\s+be\s+examined", text, re.IGNORECASE)
    section = text[marker.end() :] if marker else text
    matches = list(ANY_NUMBER_RE.finditer(section))
    recommendations = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        chunk = section[match.end() : end].strip()
        # A paragraph can contain multiple recommendations: split on "); ".
        parts = [part for part in re.split(r"\)\s*;\s*", chunk) if part.strip()]
        for part_index, part in enumerate(parts):
            if not part.rstrip().endswith(")"):
                part = part.rstrip() + ")"
            clean, countries = strip_trailing_countries(part)
            if not clean or not countries:
                continue
            number = match.group(1)
            if len(parts) > 1:
                number = f"{number}.{part_index + 1}"
            recommendations.append(
                {
                    "number": number,
                    "text": clean,
                    "countries": [canonical_state(country) for country in countries],
                }
            )
    return {
        "state_under_review": upr_state(payload)["state_under_review"],
        "recommendations": recommendations,
    }


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        method_id="baseline_year_regex",
        name="Baseline — first year regex",
        description="Return the first 4-digit year found in the first 4,000 characters.",
        dataset_id="plan_year",
        function=plan_year,
    ),
    Baseline(
        method_id="baseline_document_code_regex",
        name="Baseline — document code regex",
        description="Find the first symbol (A/RES/..., S/RES/..., C098) in the first 6,000 characters.",
        dataset_id="plan_document_code",
        function=plan_document_code,
    ),
    Baseline(
        method_id="baseline_upr_state_match",
        name="Baseline — first-page state matching",
        description="Match the longest Uwazi state name (or official alias) on page 1.",
        dataset_id="upr_state",
        function=upr_state,
    ),
    Baseline(
        method_id="baseline_upr_recommendation_regex",
        name="Baseline — recommendation number regex",
        description="Split the recommendation section on paragraph numbers and trailing country parentheses.",
        dataset_id="upr_recommendations",
        function=upr_recommendations,
    ),
)


def for_dataset(dataset_id: str) -> Baseline | None:
    return next((baseline for baseline in BASELINES if baseline.dataset_id == dataset_id), None)

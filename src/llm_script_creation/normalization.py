"""Text, number and country/state normalization shared by evaluation and baselines."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
from typing import Any

from .configuration import STATES_PATH

NUMBER_RE = re.compile(r"^\s*(\d{1,3})\.(\d{1,3})(?:\.(\d{1,3}))?\s*$")
ANY_NUMBER_RE = re.compile(r"(?<![\d.])([1-9]\d{0,2}\.\d{1,3})(?:\.(\d{1,3}))?(?![\d.])")
TRAILING_COUNTRIES_RE = re.compile(r"\s*((?:\([^()]+\)\s*)+);?\s*$")
PAREN_RE = re.compile(r"\(([^()]+)\)")

# Uwazi uses short state labels; documents use official names. Keep this small
# and explicit rather than guessing. It covers the labels that differ in the
# segmented UPR corpus.
COUNTRY_ALIASES = {
    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",
    "republic of cabo verde": "Cape Verde",
    "north macedonia": "Macedonia FYR",
    "republic of north macedonia": "Macedonia FYR",
    "macedonia": "Macedonia FYR",
    "lao people's democratic republic": "Laos",
    "lao pdr": "Laos",
    "saint lucia": "St Lucia",
    "saint kitts and nevis": "St Kitts & Nevis",
    "saint vincent and the grenadines": "St Vincent & the Grenadines",
    "sao tome and principe": "Sao Tome & Principe",
    "democratic republic of the congo": "Democratic Republic of Congo",
    "democratic republic of congo": "Democratic Republic of Congo",
    "dr congo": "Democratic Republic of Congo",
    "drc": "Democratic Republic of Congo",
    "republic of the congo": "Republic of Congo",
    "congo": "Republic of Congo",
    "democratic people's republic of korea": "DPR Korea",
    "dprk": "DPR Korea",
    "north korea": "DPR Korea",
    "republic of korea": "Republic of Korea",
    "south korea": "Republic of Korea",
    "cote d'ivoire": "Cote d'Ivoire",
    "ivory coast": "Cote d'Ivoire",
    "plurinational state of bolivia": "Bolivia",
    "bolivarian republic of venezuela": "Venezuela",
    "islamic republic of iran": "Iran",
    "federated states of micronesia": "Micronesia",
    "republic of moldova": "Moldova",
    "state of palestine": "Palestine",
    "syrian arab republic": "Syria",
    "united republic of tanzania": "Tanzania",
    "united states of america": "United States of America",
    "russian federation": "Russian Federation",
    "viet nam": "Viet Nam",
    "vietnam": "Viet Nam",
}


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value: Any) -> str:
    """Casefold, strip accents/punctuation, and collapse whitespace."""
    if value is None:
        return ""
    text = strip_accents(str(value)).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_code(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(str(value)).upper())


def parse_number(value: Any) -> tuple[str, str, str] | None:
    """Return (full, base, sub) for a recommendation number."""
    if value is None:
        return None
    match = NUMBER_RE.match(str(value))
    if not match:
        return None
    base = f"{match.group(1)}.{match.group(2)}"
    sub = match.group(3) or ""
    full = f"{base}.{sub}" if sub else base
    return full, base, sub


def numbers_in_text(text: str) -> list[str]:
    return [match.group(0) for match in ANY_NUMBER_RE.finditer(text)]


def tokenize(value: Any) -> list[str]:
    return normalize_text(value).split()


def token_f1(gold: Any, prediction: Any) -> float:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(prediction)
    if not gold_tokens or not pred_tokens:
        return 0.0
    overlap = Counter(gold_tokens) & Counter(pred_tokens)
    matches = sum(overlap.values())
    if not matches:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@lru_cache(maxsize=1)
def load_states() -> tuple[dict[str, str], ...]:
    if not STATES_PATH.exists():
        return ()
    return tuple(json.loads(STATES_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _state_index() -> tuple[dict[str, str], list[str]]:
    alias_to_label: dict[str, str] = {}
    labels: list[str] = []
    for state in load_states():
        label = state["label"]
        labels.append(normalize_text(label))
        alias_to_label[normalize_text(label)] = label
        if state.get("code"):
            alias_to_label[normalize_text(state["code"])] = label
    for alias, label in COUNTRY_ALIASES.items():
        alias_to_label[normalize_text(alias)] = label
    return alias_to_label, labels


def canonical_state(value: Any) -> str:
    """Map a state/country mention to the Uwazi state label when possible."""
    key = normalize_text(value)
    if not key:
        return ""
    alias_to_label, labels = _state_index()
    if key in alias_to_label:
        return alias_to_label[key]
    # Accept explicit ISO codes and close fuzzy matches.
    close = get_close_matches(key, labels, n=1, cutoff=0.88)
    if close:
        return _label_for_normalized(close[0])
    return str(value).strip()


def _label_for_normalized(normalized_label: str) -> str:
    for state in load_states():
        if normalize_text(state["label"]) == normalized_label:
            return state["label"]
    return normalized_label


def strip_trailing_countries(text: Any) -> tuple[str, list[str]]:
    """Split '... (Country);' into recommendation text and country names."""
    if text is None:
        return "", []
    raw = str(text).strip()
    match = TRAILING_COUNTRIES_RE.search(raw)
    if not match:
        return raw.rstrip(";").strip(), []
    countries = [item.strip() for item in PAREN_RE.findall(match.group(1))]
    clean = raw[: match.start()].rstrip(";").strip()
    return clean, countries

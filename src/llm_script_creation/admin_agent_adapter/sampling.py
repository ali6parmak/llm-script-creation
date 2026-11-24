"""Deterministic authoring-sample selection for the admin-agent adapter.

The admin agent discovers its own train samples through ``query_entities``. With
the default policy ``agent`` it sees the complete train split and picks its own
8-12 documents. The restricted policies narrow that universe to a deterministic
subset (~12 documents) that covers metadata classes, layout signatures and hard
cases, so the extractor's evidence does not depend on which documents happen to
come first.

Policies
--------
``agent``
    No restriction: every train document in the authoring request (the original
    behaviour; ``agent_all_train`` is accepted as a backward-compatible alias).
``coverage``
    Greedy rare-first coverage of metadata classes (document_type, body,
    legal_status, cycle), layout buckets (pages, characters, paragraphs, header
    patterns A/RES, A/HRC, Distr., Report of, No. NN) and hard-case flags.
``similarity``
    TF-IDF over the first page/header text plus structural tokens, with
    farthest-first (k-center) selection.
``hybrid``
    Half coverage representatives, then farthest-first seeded from those
    representatives, plus a forced negative (target absent) and hard structural
    case when the universe has one.
``balanced``
    Six typical documents allocated proportionally to the target stratum (so the
    real class prior is preserved), then six rare/hard documents (rare target
    strata plus structural hard cases); the typical block comes first.
``balanced_rare_first``
    The same document set as ``balanced`` with the rare/hard block first. This is
    the ordering control: the v3 experiment showed rare-first ordering can
    recreate the biased-sample failure.

Everything here is pure stdlib: no Mongo, no LLM, no admin-agent import. Gold is
read only to recognise target-absent negatives and UPR record hard cases; no
gold value is ever written into the task prompt by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from ..datasets import dataset_stratum
from ..task_ids import base_dataset_id

POLICIES: tuple[str, ...] = (
    "agent",
    "coverage",
    "similarity",
    "hybrid",
    "balanced",
    "balanced_rare_first",
)
DEFAULT_SIZE = 12

#: Policies that keep the real class prior (6 typical) and add 6 rare/hard cases.
BALANCED_POLICIES = frozenset({"balanced", "balanced_rare_first"})

#: Policies that leave the train universe untouched.
AGENT_POLICIES = frozenset({"agent", "agent_all_train"})

#: Header/first-line signatures named in the experiment design. The order is
#: also the stable order used in specs.
HEADER_PATTERNS: tuple[str, ...] = ("A/RES", "A/HRC", "Distr.", "Report of", "No. NN")

#: Hard-case flags, rarest/most structural first. Used to force one case into
#: the selection before ordinary coverage greed.
HARD_FLAGS: tuple[str, ...] = (
    "shared_pdf",
    "multi_entry",
    "ocr_mojibake",
    "sub_numbered",
    "multi_country",
)

#: Metadata dimensions used for coverage. ``cycle`` is UPR's review cycle.
METADATA_DIMENSIONS: tuple[str, ...] = ("document_type", "body", "legal_status", "cycle")

_METADATA_PROPERTY = {
    "document_type": "document_type",
    "body": "body",
    "legal_status": "legal_status",
    "cycle": "cycle_of_the_document",
}

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_SYMBOL = re.compile(r"\b[A-Z]{1,3}(?:/[A-Z0-9][A-Z0-9.]*){1,5}\b")
_PUNCTUATION = frozenset(
    " \n\r\t.,;:!?()[]{}<>%/-_'\"“”‘’—–…&*+=@#|\\·•§°«»"
)
_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "Ð", "Ñ", "\ufffd")


# ---------------------------------------------------------------------------
# Small helpers


def normalize_policy(policy: str) -> str:
    """Accept the historical ``agent_all_train`` name for ``agent``."""
    value = (policy or "agent").strip()
    return "agent" if value in AGENT_POLICIES else value


def _rank(dataset_id: str, doc_id: str) -> str:
    return hashlib.sha256(f"{dataset_id}:{doc_id}".encode()).hexdigest()


def _page_number(segment: dict[str, Any]) -> int:
    raw = segment.get("page_number", segment.get("page"))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _metadata_label(metadata: dict[str, Any], prop: str) -> str:
    raw = metadata.get(prop)
    items = raw if isinstance(raw, list) else [raw] if raw is not None else []
    for item in items:
        if isinstance(item, dict):
            label = item.get("label")
            value = item.get("value")
            candidate = label if label not in (None, "") else value
        else:
            candidate = item
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return "unknown"


def _page_bucket(pages: int) -> str:
    if pages <= 2:
        return "1-2"
    if pages <= 5:
        return "3-5"
    if pages <= 10:
        return "6-10"
    if pages <= 25:
        return "11-25"
    if pages <= 60:
        return "26-60"
    if pages <= 150:
        return "61-150"
    return "151+"


def _length_bucket(chars: int) -> str:
    if chars <= 5_000:
        return "<5k"
    if chars <= 10_000:
        return "5-10k"
    if chars <= 20_000:
        return "10-20k"
    if chars <= 50_000:
        return "20-50k"
    if chars <= 100_000:
        return "50-100k"
    return "100k+"


def _paragraph_bucket(paragraphs: int) -> str:
    if paragraphs <= 20:
        return "<20"
    if paragraphs <= 40:
        return "20-40"
    if paragraphs <= 80:
        return "40-80"
    if paragraphs <= 150:
        return "80-150"
    if paragraphs <= 300:
        return "150-300"
    return "300+"


def _first_page_text(segments: Iterable[dict[str, Any]]) -> str:
    rows = list(segments)
    pages = [_page_number(segment) for segment in rows]
    if not pages:
        return ""
    first = min(pages)
    return "\n".join(
        str(segment.get("text", ""))
        for segment in rows
        if _page_number(segment) == first
    )


def _first_pages_text(segments: Iterable[dict[str, Any]], count: int = 3) -> str:
    rows = list(segments)
    pages = sorted({_page_number(segment) for segment in rows})
    keep = set(pages[:count])
    return "\n".join(
        str(segment.get("text", "")) for segment in rows if _page_number(segment) in keep
    )


def _headers(first_page: str) -> tuple[str, ...]:
    """Header/first-line signatures present on the first page."""
    compact = re.sub(r"\s+", "", first_page).upper()
    spaced = _normalize_ws(first_page)
    lower = spaced.lower()
    found = []
    if "A/RES" in compact or "/RES/" in compact:
        found.append("A/RES")
    if "A/HRC" in compact or "/HRC/" in compact:
        found.append("A/HRC")
    if "DISTR.:" in compact:
        found.append("Distr.")
    if "report of" in lower:
        found.append("Report of")
    if re.search(r"\bNo\.?\s*\d+", spaced, re.IGNORECASE):
        found.append("No. NN")
    return tuple(found)


def _symbols(first_pages: str) -> tuple[str, ...]:
    """Document symbols (with a digit) on the first pages, e.g. A/HRC/39/4."""
    spaced = _normalize_ws(first_pages)
    found = {
        match.upper()
        for match in _SYMBOL.findall(spaced)
        if any(character.isdigit() for character in match)
    }
    return tuple(sorted(found))


def _is_mojibake(text: str) -> bool:
    """Conservative OCR/mojibake signal (no false positives on clean PDFs)."""
    if not text:
        return False
    markers = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    if markers >= 3 or controls:
        return True
    weird = sum(
        1
        for ch in text
        if not (ch.isalnum() or ch in _PUNCTUATION)
    )
    return weird / max(1, len(text)) > 0.01


def _is_multi_entry(first_page: str, full_text: str, page_count: int) -> bool:
    """Multi-entry compilation signal.

    Long documents that collect many entries (conclusions, session supplements,
    annual reports) look different from a single resolution or report. The
    threshold is intentionally conservative: it flags only large documents with
    a repeated symbol or an explicit compilation marker.
    """
    if page_count < 40:
        return False
    lower_first_page = _normalize_ws(first_page).lower()
    lower_full = full_text.lower()
    symbols = Counter(_symbols(full_text))
    repeated = symbols.most_common(1)[0][1] if symbols else 0
    return (
        repeated >= 8
        or "conclusions adopted" in lower_full
        or "supplement no." in lower_first_page
        or "annual report" in lower_first_page
    )


def _is_negative(dataset_id: str, gold: dict[str, Any]) -> bool:
    """True when the target is absent from the gold row (a negative sample)."""
    dataset_id = base_dataset_id(dataset_id)
    if dataset_id == "upr_recommendations":
        return not gold.get("recommendations")
    if dataset_id == "upr_state":
        return not gold.get("state_under_review")
    if dataset_id.startswith("plan_"):
        field = dataset_id.removeprefix("plan_")
        if field != "all_fields":
            return not gold.get(field)
    return not gold


def _record_flags(dataset_id: str, gold: dict[str, Any]) -> tuple[str, ...]:
    """Hard cases that live in the UPR recommendation gold structure."""
    if base_dataset_id(dataset_id) != "upr_recommendations":
        return ()
    recommendations = gold.get("recommendations") or []
    flags = []
    if any(str(rec.get("number", "")).count(".") >= 2 for rec in recommendations):
        flags.append("sub_numbered")
    if any(len(rec.get("countries") or []) >= 2 for rec in recommendations):
        flags.append("multi_country")
    return tuple(flags)


# ---------------------------------------------------------------------------
# Per-document features


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    gold: dict[str, Any]
    metadata_classes: dict[str, str]
    page_count: int
    segment_count: int
    char_count: int
    page_bucket: str
    length_bucket: str
    paragraph_bucket: str
    headers: tuple[str, ...]
    symbols: tuple[str, ...]
    hard_flags: tuple[str, ...]
    negative: bool
    fingerprint: str
    file_id: str
    filename: str
    dedupe_key: str
    rank: str
    coverage_tokens: frozenset[str]
    similarity_tokens: tuple[str, ...]


def _features(dataset_id: str, record: dict[str, Any]) -> _Doc:
    segments = list(record.get("segments") or [])
    metadata = record.get("metadata") or {}
    gold = record.get("gold") or {}
    first_page = _first_page_text(segments)
    first_pages = _first_pages_text(segments)
    full_text = "\n".join(str(segment.get("text", "")) for segment in segments)
    page_count = max((_page_number(segment) for segment in segments), default=0)
    char_count = len(full_text)
    segment_count = len(segments)
    headers = _headers(first_page)
    symbols = _symbols(first_pages)
    classes = {
        dimension: _metadata_label(metadata, _METADATA_PROPERTY[dimension])
        for dimension in METADATA_DIMENSIONS
    }

    hard_flags = list(_record_flags(dataset_id, gold))
    if _is_multi_entry(first_page, full_text, page_count):
        hard_flags.append("multi_entry")
    if _is_mojibake(full_text):
        hard_flags.append("ocr_mojibake")

    fingerprint = hashlib.sha256(full_text.encode("utf-8", "replace")).hexdigest() if full_text else ""
    file_id = str(record.get("file_id") or "")
    filename = str(record.get("filename") or "")

    coverage = {
        f"md:{dimension}:{value}" for dimension, value in classes.items()
    }
    coverage.update(
        {
            f"page:{_page_bucket(page_count)}",
            f"len:{_length_bucket(char_count)}",
            f"paras:{_paragraph_bucket(segment_count)}",
        }
    )
    coverage.update(f"hdr:{header}" for header in headers)
    coverage.update(f"hard:{flag}" for flag in hard_flags)
    if _is_negative(dataset_id, gold):
        coverage.add("negative")

    similarity_tokens = [
        f"word:{word}"
        for word in _WORD.findall(_normalize_ws(first_page).lower())
    ]
    similarity_tokens.extend(f"sym:{symbol}" for symbol in symbols)
    similarity_tokens.extend(
        (
            f"page:{_page_bucket(page_count)}",
            f"paras:{_paragraph_bucket(segment_count)}",
            f"len:{_length_bucket(char_count)}",
        )
    )
    similarity_tokens.extend(f"hdr:{header}" for header in headers)

    return _Doc(
        doc_id=str(record.get("doc_id") or record.get("shared_id") or ""),
        gold=gold,
        metadata_classes=classes,
        page_count=page_count,
        segment_count=segment_count,
        char_count=char_count,
        page_bucket=_page_bucket(page_count),
        length_bucket=_length_bucket(char_count),
        paragraph_bucket=_paragraph_bucket(segment_count),
        headers=headers,
        symbols=symbols,
        hard_flags=tuple(hard_flags),
        negative=_is_negative(dataset_id, gold),
        fingerprint=fingerprint,
        file_id=file_id,
        filename=filename,
        dedupe_key=f"doc:{record.get('doc_id') or record.get('shared_id') or ''}",
        rank=_rank(dataset_id, str(record.get("doc_id") or record.get("shared_id") or "")),
        coverage_tokens=frozenset(coverage),
        similarity_tokens=tuple(similarity_tokens),
    )


def _with_shared_pdf_flags(docs: list[_Doc]) -> list[_Doc]:
    """Flag documents whose file, filename or content is shared with another row.

    Two entities can point at the same PDF (or an identical snapshot with a
    different file id), which is a hard case for extraction and should cost at
    most one selection slot.
    """
    file_counts = Counter(doc.file_id for doc in docs if doc.file_id)
    filename_counts = Counter(doc.filename for doc in docs if doc.filename)
    fingerprint_counts = Counter(doc.fingerprint for doc in docs if doc.fingerprint)
    length_counts = Counter(
        (doc.page_count, doc.char_count) for doc in docs if doc.char_count >= 50_000
    )
    flagged: list[_Doc] = []
    for doc in docs:
        shared_keys = []
        if doc.fingerprint and fingerprint_counts[doc.fingerprint] > 1:
            shared_keys.append(f"sha:{doc.fingerprint}")
        if doc.file_id and file_counts[doc.file_id] > 1:
            shared_keys.append(f"file:{doc.file_id}")
        if doc.filename and filename_counts[doc.filename] > 1:
            shared_keys.append(f"name:{doc.filename}")
        if doc.char_count >= 50_000 and length_counts[(doc.page_count, doc.char_count)] > 1:
            shared_keys.append(f"len:{doc.page_count}:{doc.char_count}")
        if shared_keys:
            flags = doc.hard_flags
            if "shared_pdf" not in flags:
                flags = ("shared_pdf", *flags)
            doc = replace(
                doc,
                hard_flags=flags,
                dedupe_key=min(shared_keys),
                coverage_tokens=doc.coverage_tokens | {"hard:shared_pdf"},
            )
        flagged.append(doc)
    return flagged


# ---------------------------------------------------------------------------
# TF-IDF + distances


def _tfidf_vectors(docs: list[_Doc]) -> dict[str, dict[str, float]]:
    document_frequency = Counter(
        token for doc in docs for token in set(doc.similarity_tokens)
    )
    total = max(1, len(docs))
    vectors: dict[str, dict[str, float]] = {}
    for doc in docs:
        counts = Counter(doc.similarity_tokens)
        vector: dict[str, float] = {}
        for token, count in counts.items():
            inverse_frequency = math.log((1.0 + total) / (1.0 + document_frequency[token])) + 1.0
            vector[token] = (1.0 + math.log(count)) * inverse_frequency
        vectors[doc.doc_id] = vector
    return vectors


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(
    left: dict[str, float], right: dict[str, float], left_norm: float, right_norm: float
) -> float:
    if not left or not right or left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    return dot / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# Selection primitives


def _rarity_weights(docs: list[_Doc]) -> dict[str, float]:
    counts = Counter(token for doc in docs for token in doc.coverage_tokens)
    return {token: 1.0 / count for token, count in counts.items()}


def _forced_docs(docs: list[_Doc]) -> list[_Doc]:
    """One target-absent negative plus one hard structural case, if available."""
    forced: list[_Doc] = []
    negatives = sorted((doc for doc in docs if doc.negative), key=lambda doc: doc.rank)
    if negatives:
        forced.append(negatives[0])

    hard = [doc for doc in docs if doc.hard_flags and doc not in forced]
    if hard:
        flag_counts = Counter(flag for doc in docs for flag in doc.hard_flags)

        def priority(doc: _Doc) -> tuple[int, int, str]:
            flag = min(doc.hard_flags, key=HARD_FLAGS.index)
            return (HARD_FLAGS.index(flag), flag_counts[flag], doc.rank)

        forced.append(min(hard, key=priority))
    return forced


def _proportional_slots(sizes: dict[str, int], slots: int) -> dict[str, int]:
    """Allocate ``slots`` proportionally to stratum sizes (largest remainder)."""
    total = sum(sizes.values())
    if slots <= 0 or not total:
        return {name: 0 for name in sizes}
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for name, count in sizes.items():
        exact = slots * count / total
        quotas[name] = min(int(exact), count)
        remainders.append((-(exact - int(exact)), name))
    assigned = sum(quotas.values())
    for _, name in sorted(remainders):
        if assigned >= slots:
            break
        if quotas[name] < sizes[name]:
            quotas[name] += 1
            assigned += 1
    while assigned < slots:
        moved = False
        for _, name in sorted(remainders):
            if assigned >= slots:
                break
            if quotas[name] < sizes[name]:
                quotas[name] += 1
                assigned += 1
                moved = True
        if not moved:
            break
    return quotas


def _balanced_selection(
    dataset_id: str, docs: list[_Doc], size: int, *, common_first: bool = True
) -> tuple[list[_Doc], dict[str, Any]]:
    """Pick half typical (prior-preserving) and half rare/hard documents.

    The typical half is allocated proportionally to the target stratum (gold
    label/decade/recommendation bucket), so it mirrors the corpus instead of
    over-representing rare classes. The rare/hard half takes one document from
    each rarest stratum first, then hard structural cases, then fills from the
    remaining rare strata. Gold is used only to decide which documents are
    typical; no gold value enters the prompt.
    """
    groups: dict[str, list[_Doc]] = {}
    for doc in docs:
        stratum = dataset_stratum(dataset_id, doc.gold)
        groups.setdefault(stratum, []).append(doc)
    for ids in groups.values():
        ids.sort(key=lambda doc: doc.rank)

    common_slots = max(1, size // 2) if size else 0
    rare_slots = max(0, size - common_slots)
    quotas = _proportional_slots({name: len(ids) for name, ids in groups.items()}, common_slots)
    common: list[_Doc] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()

    def add(doc: _Doc) -> bool:
        if doc.doc_id in seen_ids or doc.dedupe_key in seen_keys:
            return False
        common.append(doc)
        seen_ids.add(doc.doc_id)
        seen_keys.add(doc.dedupe_key)
        return True

    for name in sorted(groups, key=lambda key: (-len(groups[key]), key)):
        for doc in groups[name][: quotas.get(name, 0)]:
            add(doc)
    if len(common) < common_slots:
        for doc in sorted(docs, key=lambda item: (-len(groups[dataset_stratum(dataset_id, item.gold)]), item.rank)):
            if len(common) >= common_slots:
                break
            add(doc)

    rare: list[_Doc] = []
    rare_seen_ids = set(seen_ids)
    rare_seen_keys = set(seen_keys)

    def add_rare(doc: _Doc) -> bool:
        if doc.doc_id in rare_seen_ids or doc.dedupe_key in rare_seen_keys:
            return False
        rare.append(doc)
        rare_seen_ids.add(doc.doc_id)
        rare_seen_keys.add(doc.dedupe_key)
        return True

    rare_groups = sorted(
        (name for name in groups if set(doc.doc_id for doc in groups[name]) - rare_seen_ids),
        key=lambda name: (len(groups[name]), name),
    )
    # Reserve part of the rare/hard half for genuinely rare target strata and
    # part for structural hard cases, so the sample covers both.
    rare_target_slots = max(1, rare_slots // 2) if rare_slots else 0
    for name in rare_groups:
        if len(rare) >= rare_target_slots:
            break
        for doc in groups[name]:
            if doc.doc_id not in rare_seen_ids:
                add_rare(doc)
                break

    hard_ids: list[str] = []

    def add_hard(doc: _Doc) -> None:
        if len(rare) >= rare_slots:
            return
        if add_rare(doc):
            hard_ids.append(doc.doc_id)

    # One document per hard flag first (shared PDF, multi-entry, OCR, ...),
    # then any remaining hard case by flag priority.
    for flag in HARD_FLAGS:
        candidates = sorted(
            (doc for doc in docs if flag in doc.hard_flags and doc.doc_id not in rare_seen_ids),
            key=lambda doc: doc.rank,
        )
        if candidates:
            add_hard(candidates[0])

    hard = sorted(
        (doc for doc in docs if doc.hard_flags),
        key=lambda doc: (
            min(HARD_FLAGS.index(flag) for flag in doc.hard_flags),
            doc.rank,
        ),
    )
    for doc in hard:
        add_hard(doc)

    for name in rare_groups:
        for doc in groups[name]:
            if len(rare) >= rare_slots:
                break
            add_rare(doc)

    if len(rare) < rare_slots:
        for doc in sorted(docs, key=lambda item: item.rank):
            if len(rare) >= rare_slots:
                break
            add_rare(doc)

    rare = rare[:rare_slots]
    selected = common + rare if common_first else rare + common
    hard_ids = [doc.doc_id for doc in selected if doc.hard_flags]
    selection = {
        "order": "common_first" if common_first else "rare_first",
        "common_ids": [doc.doc_id for doc in common],
        "rare_ids": [doc.doc_id for doc in rare],
        "hard_ids": hard_ids,
        "common_target_strata": {
            name: sum(1 for doc in common if dataset_stratum(dataset_id, doc.gold) == name)
            for name in sorted({dataset_stratum(dataset_id, doc.gold) for doc in common})
        },
        "rare_target_strata": {
            name: sum(1 for doc in rare if dataset_stratum(dataset_id, doc.gold) == name)
            for name in sorted({dataset_stratum(dataset_id, doc.gold) for doc in rare})
        },
    }
    return selected, selection


def _greedy_coverage(
    docs: list[_Doc], size: int, forced: list[_Doc]
) -> list[_Doc]:
    """Rare-first greedy coverage over metadata/layout/hard-case tokens."""
    weights = _rarity_weights(docs)
    selected: list[_Doc] = []
    selected_keys: set[str] = set()
    for doc in forced:
        if len(selected) >= size:
            break
        if doc.dedupe_key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(doc.dedupe_key)

    selected_ids = {doc.doc_id for doc in selected}
    covered = {token for doc in selected for token in doc.coverage_tokens}
    remaining = [doc for doc in docs if doc.doc_id not in selected_ids]

    while len(selected) < size and remaining:
        def score(doc: _Doc) -> tuple[float, str]:
            gain = sum(weights[token] for token in doc.coverage_tokens if token not in covered)
            return (-gain, doc.rank)

        best = min(remaining, key=score)
        remaining.remove(best)
        if best.dedupe_key in selected_keys:
            continue
        selected.append(best)
        selected_keys.add(best.dedupe_key)
        covered.update(best.coverage_tokens)
    return selected


def _farthest_first(
    docs: list[_Doc],
    vectors: dict[str, dict[str, float]],
    size: int,
    seeds: list[_Doc],
) -> list[_Doc]:
    """k-center farthest-first over the first-page TF-IDF vectors."""
    norms = {doc_id: _norm(vector) for doc_id, vector in vectors.items()}
    selected = list(seeds)
    selected_keys = {doc.dedupe_key for doc in selected}
    remaining = [doc for doc in docs if doc.doc_id not in {d.doc_id for d in selected}]

    if not selected and remaining:
        centroid: dict[str, float] = {}
        for vector in vectors.values():
            for token, value in vector.items():
                centroid[token] = centroid.get(token, 0.0) + value
        centroid_norm = _norm(centroid)
        seed = min(
            remaining,
            key=lambda doc: (
                -_distance_to(
                    vectors[doc.doc_id], norms[doc.doc_id], centroid, centroid_norm
                ),
                doc.rank,
            ),
        )
        remaining.remove(seed)
        selected.append(seed)
        selected_keys.add(seed.dedupe_key)

    while len(selected) < size and remaining:
        def score(doc: _Doc) -> tuple[float, str]:
            minimum = min(
                _distance_to(
                    vectors[doc.doc_id], norms[doc.doc_id], vectors[other.doc_id], norms[other.doc_id]
                )
                for other in selected
            )
            return (-minimum, doc.rank)

        best = min(remaining, key=score)
        remaining.remove(best)
        if best.dedupe_key in selected_keys:
            continue
        selected.append(best)
        selected_keys.add(best.dedupe_key)
    return selected


def _distance_to(
    left: dict[str, float],
    left_norm: float,
    right: dict[str, float],
    right_norm: float,
) -> float:
    return 1.0 - _cosine(left, right, left_norm, right_norm)


# ---------------------------------------------------------------------------
# Result


@dataclass(frozen=True)
class SamplingResult:
    """The selected authoring universe plus its audit metadata."""

    policy: str
    doc_ids: tuple[str, ...]
    universe: int
    sample_size: int
    strata: dict[str, Any]
    hard_cases: dict[str, list[str]]
    negatives: tuple[str, ...]
    documents: tuple[dict[str, Any], ...]
    notes: tuple[str, ...]
    digest: str
    selection: dict[str, Any] | None = None

    @property
    def restricted(self) -> bool:
        return self.policy not in AGENT_POLICIES

    def prompt_note(self) -> str | None:
        """Short task note for restricted policies (never contains gold)."""
        if not self.restricted or not self.doc_ids:
            return None
        count = len(self.doc_ids)
        if self.policy in BALANCED_POLICIES:
            order = (self.selection or {}).get("order", "common_first")
            first_block = "typical" if order == "common_first" else "rare/hard"
            second_block = "rare/hard" if order == "common_first" else "typical"
            return (
                f"SAMPLE NOTE: these {count} train documents contain a typical half and a "
                f"rare/hard half. The {first_block} documents come first, then the "
                f"{second_block} ones. Do not assume the rare/hard cases represent the "
                "corpus; use the typical documents to calibrate, then inspect the hard cases."
            )
        return (
            f"SAMPLE NOTE: these {count} train documents were chosen to cover this task's "
            "layout and class variety. Peek across several of them, not just the first one "
            "or two, before committing to rules."
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "selected_shared_ids": list(self.doc_ids),
            "authoring_documents": len(self.doc_ids),
            "universe_documents": self.universe,
            "sample_size": self.sample_size,
            "digest": self.digest,
            "strata": self.strata,
            "hard_cases": self.hard_cases,
            "negatives": list(self.negatives),
            "documents": list(self.documents),
            "notes": list(self.notes),
            "selection": self.selection or {},
        }


def _summarize(
    docs: list[_Doc],
    selected: list[_Doc],
    policy: str,
    universe: int,
    size: int,
    *,
    selection: dict[str, Any] | None = None,
) -> SamplingResult:
    by_id = {doc.doc_id: doc for doc in selected}
    metadata = {
        dimension: sorted({doc.metadata_classes[dimension] for doc in selected})
        for dimension in METADATA_DIMENSIONS
    }
    layout = {
        "page_bucket": sorted({doc.page_bucket for doc in selected}),
        "length_bucket": sorted({doc.length_bucket for doc in selected}),
        "paragraph_bucket": sorted({doc.paragraph_bucket for doc in selected}),
    }
    hard_cases: dict[str, list[str]] = {}
    for doc in selected:
        for flag in doc.hard_flags:
            hard_cases.setdefault(flag, []).append(doc.doc_id)
    hard_cases = {flag: sorted(ids) for flag, ids in sorted(hard_cases.items())}
    negatives = tuple(sorted(doc.doc_id for doc in selected if doc.negative))
    documents = tuple(
        {
            "doc_id": doc.doc_id,
            "page_bucket": doc.page_bucket,
            "length_bucket": doc.length_bucket,
            "paragraph_bucket": doc.paragraph_bucket,
            "metadata_classes": dict(doc.metadata_classes),
            "headers": list(doc.headers),
            "hard_flags": list(doc.hard_flags),
            "negative": doc.negative,
        }
        for doc in selected
    )
    dedupe_keys = [doc.dedupe_key for doc in docs]
    duplicate_count = len(dedupe_keys) - len(set(dedupe_keys))
    notes = []
    if policy != "agent":
        notes.append(
            f"policy {policy}: {len(selected)}/{universe} train documents "
            f"(cap {size})"
        )
        if not negatives:
            notes.append(
                "no target-absent (negative) document exists in this train universe; "
                "the negative slot is not used"
            )
        if duplicate_count:
            notes.append(
                f"{duplicate_count} duplicate/shared document(s) in the universe; "
                "at most one of each group is selected"
            )
    digest = hashlib.sha256(
        json.dumps(
            {"policy": policy, "doc_ids": [doc.doc_id for doc in selected], "universe": universe},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    return SamplingResult(
        policy=policy,
        doc_ids=tuple(by_id.keys()),
        universe=universe,
        sample_size=size,
        strata={
            "metadata": metadata,
            "layout": layout,
            "headers": sorted({header for doc in selected for header in doc.headers}),
            "negatives": list(negatives),
        },
        hard_cases=hard_cases,
        negatives=negatives,
        documents=documents,
        notes=tuple(notes),
        digest=digest,
        selection=selection,
    )


# ---------------------------------------------------------------------------
# Public API


def sample_documents(
    dataset_id: str,
    documents: Iterable[dict[str, Any]],
    *,
    policy: str = "agent",
    size: int = DEFAULT_SIZE,
) -> SamplingResult:
    """Choose the authoring universe from the dataset's train documents.

    ``documents`` must contain train rows only. The sampler rejects an explicit
    ``split == "test"`` so a wiring mistake cannot leak test documents into the
    authoring request.
    """
    policy = normalize_policy(policy)
    if policy not in POLICIES:
        raise ValueError(f"unknown sampling policy {policy!r}; choose from {POLICIES}")
    raw = list(documents)
    if any(str(record.get("split") or "") == "test" for record in raw):
        raise ValueError("sampling is train-only: refusing documents from the test split")
    if not raw:
        return _summarize([], [], policy, 0, max(0, size))

    docs = _with_shared_pdf_flags([_features(dataset_id, record) for record in raw])
    if policy == "agent":
        selected = docs
        return _summarize(docs, selected, policy, len(docs), len(docs))

    docs.sort(key=lambda doc: doc.rank)
    size = max(0, min(int(size), len(docs)))
    if size == 0:
        return _summarize(docs, [], policy, len(docs), 0)

    if policy == "coverage":
        selected = _greedy_coverage(docs, size, _forced_docs(docs))
    elif policy == "similarity":
        selected = _farthest_first(docs, _tfidf_vectors(docs), size, [])
    elif policy in BALANCED_POLICIES:
        selected, selection = _balanced_selection(
            dataset_id,
            docs,
            size,
            common_first=policy == "balanced",
        )
        return _summarize(docs, selected, policy, len(docs), size, selection=selection)
    else:  # hybrid
        coverage_size = max(1, size // 2)
        cohort = _greedy_coverage(docs, coverage_size, _forced_docs(docs))
        selected = _farthest_first(docs, _tfidf_vectors(docs), size, cohort)
    return _summarize(docs, selected, policy, len(docs), size)


def restrict_documents(
    dataset_id: str,
    documents: Iterable[dict[str, Any]],
    doc_ids: Iterable[str],
    *,
    policy: str = "replay",
) -> SamplingResult:
    """Reconstruct a stored selection (replay) without re-running the sampler.

    The stored order is preserved. Unknown ids are an error: a replay must use
    the exact documents the method recorded.
    """
    raw = [record for record in documents]
    by_id = {str(record.get("doc_id") or record.get("shared_id")): record for record in raw}
    wanted = [str(doc_id) for doc_id in doc_ids]
    missing = [doc_id for doc_id in wanted if doc_id not in by_id]
    if missing:
        raise ValueError(f"stored selected ids are not in this train split: {missing[:5]}")
    if any(str(by_id[doc_id].get("split") or "") == "test" for doc_id in wanted):
        raise ValueError("sampling is train-only: a stored selection contains test documents")
    docs = _with_shared_pdf_flags([_features(dataset_id, by_id[doc_id]) for doc_id in wanted])
    return _summarize(docs, docs, normalize_policy(policy), len(raw), len(docs))


__all__ = [
    "AGENT_POLICIES",
    "BALANCED_POLICIES",
    "DEFAULT_SIZE",
    "HARD_FLAGS",
    "HEADER_PATTERNS",
    "POLICIES",
    "SamplingResult",
    "normalize_policy",
    "restrict_documents",
    "sample_documents",
]

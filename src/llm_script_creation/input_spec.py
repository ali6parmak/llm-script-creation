"""One shared input transformation for prompts, execution and the viewer.

The same `InputSpec` + `prepare_input` must be used everywhere a document is
turned into the payload passed to `extract(document)`:

* prompt few-shot examples,
* repair feedback examples,
* train and test execution,
* the viewer's "Input sent to the script" preview.

Gold always stays based on the full document. If a target is outside a
truncated input window the method simply fails; the gold is not moved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .models import Document
from .text import ordered_segments, render_text

Representation = Literal["text", "segments"]


@dataclass(frozen=True)
class InputSpec:
    """How a document is transformed into the script input.

    Only one window per representation is allowed: either the first N or the
    last N characters/segments, not both. With no window the full transformed
    document is used.
    """

    representation: Representation = "text"
    head_chars: int | None = None
    tail_chars: int | None = None
    head_segments: int | None = None
    tail_segments: int | None = None
    include_position: bool = False
    include_type: bool = False

    def __post_init__(self) -> None:
        if self.representation not in ("text", "segments"):
            raise ValueError(f"unknown representation: {self.representation!r}")
        if self.head_chars is not None and self.tail_chars is not None:
            raise ValueError("set at most one of head_chars and tail_chars")
        if self.head_segments is not None and self.tail_segments is not None:
            raise ValueError("set at most one of head_segments and tail_segments")
        for name in ("head_chars", "tail_chars", "head_segments", "tail_segments"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.representation == "text" and (
            self.head_segments is not None or self.tail_segments is not None
        ):
            raise ValueError("text input cannot use a segment window")
        if self.representation == "segments" and (
            self.head_chars is not None or self.tail_chars is not None
        ):
            raise ValueError("segment input cannot use a character window")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> InputSpec:
        if not value:
            return cls()
        allowed = {
            "representation",
            "head_chars",
            "tail_chars",
            "head_segments",
            "tail_segments",
            "include_position",
            "include_type",
        }
        return cls(**{key: value[key] for key in allowed if key in value})

    def describe(self) -> str:
        if self.representation == "text":
            if self.head_chars is not None:
                return f"the first {self.head_chars:,} characters of the document text"
            if self.tail_chars is not None:
                return f"the last {self.tail_chars:,} characters of the document text"
            return "the full document text"
        if self.head_segments is not None:
            window = f"the first {self.head_segments:,} segments"
        elif self.tail_segments is not None:
            window = f"the last {self.tail_segments:,} segments"
        else:
            window = "all segments"
        fields = ["text"]
        if self.include_type:
            fields.append("type")
        if self.include_position:
            fields.append("position (page, top, left, width, height)")
        return f"{window}, each with {', '.join(fields)}"

    def slug(self) -> str:
        parts = ["text" if self.representation == "text" else "segments"]
        if self.head_chars is not None:
            parts.append(f"head{self.head_chars}")
        if self.tail_chars is not None:
            parts.append(f"tail{self.tail_chars}")
        if self.head_segments is not None:
            parts.append(f"head{self.head_segments}seg")
        if self.tail_segments is not None:
            parts.append(f"tail{self.tail_segments}seg")
        if self.include_type:
            parts.append("type")
        if self.include_position:
            parts.append("pos")
        return "_".join(parts)


def _truncate_text(text: str, spec: InputSpec) -> str:
    if spec.head_chars is not None:
        return text[: spec.head_chars]
    if spec.tail_chars is not None:
        return text[-spec.tail_chars :]
    return text


def _truncate_segments(segments: list[dict[str, Any]], spec: InputSpec) -> list[dict[str, Any]]:
    if spec.head_segments is not None:
        return segments[: spec.head_segments]
    if spec.tail_segments is not None:
        return segments[-spec.tail_segments :]
    return segments


def prepare_input(document: Document, spec: InputSpec) -> dict[str, Any]:
    """Build the exact JSON-serializable payload passed to `extract`."""
    payload: dict[str, Any] = {
        "id": document.shared_id,
        "language": document.language,
        "page_count": document.page_count,
    }
    if spec.representation == "text":
        payload["text"] = _truncate_text(render_text(document, order="layout"), spec)
    else:
        segments = [
            segment.as_dict(
                include_position=spec.include_position,
                include_type=spec.include_type,
            )
            for segment in ordered_segments(document, order="layout")
        ]
        payload["segments"] = _truncate_segments(segments, spec)
    return payload


def _focus_index(text: str, focuses: Iterable[str]) -> int | None:
    lowered = text.lower()
    for focus in focuses:
        if not focus:
            continue
        index = lowered.find(str(focus).lower())
        if index >= 0:
            return index
        # PDF extraction often inserts extra spaces/newlines between words
        # ("recommendations  will  be  examined"). Match those flexibly.
        words = str(focus).split()
        if len(words) < 2:
            continue
        pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)
        match = pattern.search(text)
        if match is not None:
            return match.start()
    return None


def payload_excerpt(
    payload: dict[str, Any],
    spec: InputSpec,
    *,
    char_limit: int = 4500,
    segment_limit: int = 18,
    focuses: Iterable[str] = (),
) -> tuple[str, bool]:
    """Return an excerpt of the *transformed* payload and whether it was cut.

    Prompts, repair feedback and the viewer use this so a shown excerpt never
    comes from the untransformed document while the script runs on a window.
    """
    focuses = tuple(focuses)
    if spec.representation == "text":
        text = str(payload.get("text", ""))
        if len(text) <= char_limit:
            return text, False
        start = 0
        index = _focus_index(text, focuses)
        if index is not None:
            start = max(0, min(index - char_limit // 4, len(text) - char_limit))
        elif spec.head_chars is None and spec.tail_chars is not None:
            start = max(0, len(text) - char_limit)
        return text[start : start + char_limit], True

    segments = list(payload.get("segments") or [])
    if len(segments) <= segment_limit:
        return json.dumps(segments, ensure_ascii=False, indent=1), False
    start = 0
    index = None
    for offset, segment in enumerate(segments):
        if _focus_index(str(segment.get("text", "")), focuses) is not None:
            index = offset
            break
    if index is not None:
        start = max(0, min(index - 2, len(segments) - segment_limit))
    elif spec.head_segments is None and spec.tail_segments is not None:
        start = max(0, len(segments) - segment_limit)
    return json.dumps(segments[start : start + segment_limit], ensure_ascii=False, indent=1), True

"""Turn a segmented Uwazi document into representations for prompts/scripts.

The benchmark varies what an LLM-generated script can see:

* ``text``          -- plain text, reading order only.
* ``segments-tsv``  -- one line per segment: page + box + type + text.
* ``segments-json`` -- the same information as a JSON list.

Layout information is the experimental variable behind the "does position and
size help?" question in TASKS.md, so keep the renderers strict and explicit.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from .models import Document, Segment

Representation = Literal["text", "segments-tsv", "segments-json"]

NOISE_TYPES = frozenset({"Page header", "Page footer"})


def ordered_segments(
    document: Document, order: str = "layout", skip_noise: bool = False
) -> list[Segment]:
    """Return non-empty segments in reading order.

    ``layout`` sorts by page, vertical position and horizontal position. That
    fixes stale array order (e.g. page footers stored before body text) and
    keeps a table cell with a recommendation number before its text cell.
    ``raw`` preserves the order stored by Uwazi.
    """
    segments = [segment for segment in document.segments if segment.text.strip()]
    if skip_noise:
        segments = [segment for segment in segments if segment.type not in NOISE_TYPES]
    if order == "raw":
        return segments
    if order != "layout":
        raise ValueError(f"unknown segment order: {order!r}")
    return sorted(segments, key=lambda s: (s.page, s.top, s.left, s.width))


def render_text(
    document: Document,
    *,
    order: str = "layout",
    skip_noise: bool = False,
    page_markers: bool = False,
) -> str:
    """Render plain text, one segment per line."""
    lines: list[str] = []
    current_page: int | None = None
    for segment in ordered_segments(document, order=order, skip_noise=skip_noise):
        if page_markers and segment.page != current_page:
            if current_page is not None:
                lines.append("")
            lines.append(f"--- page {segment.page} ---")
            current_page = segment.page
        lines.append(segment.text)
    return "\n".join(lines)


def render_segments_tsv(
    document: Document,
    *,
    order: str = "layout",
    skip_noise: bool = False,
    include_position: bool = True,
    include_type: bool = True,
) -> str:
    """Render one TSV line per segment."""
    columns = []
    if include_position:
        columns += ["page", "top", "left", "width", "height"]
    if include_type:
        columns.append("type")
    columns.append("text")
    lines = ["\t".join(columns)]
    for segment in ordered_segments(document, order=order, skip_noise=skip_noise):
        values: list[str] = []
        if include_position:
            values += [
                str(segment.page),
                f"{segment.top:.0f}",
                f"{segment.left:.0f}",
                f"{segment.width:.0f}",
                f"{segment.height:.0f}",
            ]
        if include_type:
            values.append(segment.type)
        text = " ".join(segment.text.split())
        values.append(text.replace("\t", " "))
        lines.append("\t".join(values))
    return "\n".join(lines)


def render_segments_json(
    document: Document,
    *,
    order: str = "layout",
    skip_noise: bool = False,
    include_position: bool = True,
    include_type: bool = True,
) -> str:
    segments = [
        segment.as_dict(include_position=include_position, include_type=include_type)
        for segment in ordered_segments(document, order=order, skip_noise=skip_noise)
    ]
    return json.dumps(segments, ensure_ascii=False)


def render_document(
    document: Document,
    representation: Representation = "text",
    **kwargs: Any,
) -> str:
    if representation == "text":
        kwargs.pop("include_position", None)
        kwargs.pop("include_type", None)
        return render_text(document, **kwargs)
    if representation == "segments-tsv":
        return render_segments_tsv(document, **kwargs)
    if representation == "segments-json":
        return render_segments_json(document, **kwargs)
    raise ValueError(f"unknown representation: {representation!r}")


def document_payload(
    document: Document,
    representation: Representation = "text",
    *,
    include_position: bool = True,
    include_type: bool = True,
    include_title: bool = False,
    skip_noise: bool = False,
    order: str = "layout",
) -> dict[str, Any]:
    """Serializable document passed to generated extraction scripts.

    The title is omitted by default: it often contains the answer (e.g. a UPR
    document title starts with the state under review), so including it would
    silently leak the ground truth.
    """
    payload: dict[str, Any] = {
        "id": document.shared_id,
        "language": document.language,
        "page_count": document.page_count,
    }
    if include_title:
        payload["title"] = document.title
    if representation == "text":
        payload["text"] = render_text(document, order=order, skip_noise=skip_noise)
    else:
        payload["segments"] = [
            segment.as_dict(include_position=include_position, include_type=include_type)
            for segment in ordered_segments(document, order=order, skip_noise=skip_noise)
        ]
    return payload

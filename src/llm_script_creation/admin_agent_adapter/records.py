"""Convert catalog document rows to the admin agent's domain models.

Catalog rows come from ``artifacts/catalog.db`` (``documents`` table). Their
``segments_json`` uses the key ``page`` (see ``Segment.as_dict``), while Uwazi's
raw segmentation and the admin agent's :class:`Paragraph` use ``page_number``
(alias ``pageNumber``). The conversion lives here so the sidecar never guesses.

Admin-agent imports are LAZY: the pure helpers (paragraph mapping, joined text,
context shape, raw-entity shape) can be imported and unit-tested without the
admin agent's environment. The model constructors import ``uwazi_agent`` /
``uwazi_api`` only when called, which only happens in the sidecar.

Metadata is passed through in Uwazi's read shape (a property maps to a list of
``{"value": ..., "label": ...}`` entries). That is the "production context"
condition: the extractor sees the same metadata the admin agent's review table
shows, labels included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Uwazi type string -> AgentPropertyType member name. Unknown types fall back to
# TEXT rather than dropping the property: the extractor only needs the target's
# declared type, but a full template listing is more honest than a partial one.
_PROPERTY_TYPE_NAMES = {
    "text": "TEXT",
    "markdown": "MARKDOWN",
    "numeric": "NUMERIC",
    "date": "DATE",
    "daterange": "DATE_RANGE",
    "multidate": "MULTI_DATE",
    "multidaterange": "MULTI_DATE_RANGE",
    "select": "SELECT",
    "multiselect": "MULTI_SELECT",
    "link": "LINK",
    "image": "IMAGE",
    "media": "MEDIA",
    "geolocation": "GEO_LOCATION",
    "relationship": "RELATIONSHIP",
    "generatedid": "GENERATED_ID",
    "preview": "PREVIEW",
    "nested": "NESTED",
}


def load_json(value: Any, default: Any) -> Any:
    """Parse a JSON string (catalog column) or pass a decoded value through."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def paragraph_dicts(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Map catalog segments to the admin agent's paragraph field names (pure).

    Catalog segments are already stored in layout/reading order (non-empty,
    sorted by page/top/left/width), so their order is preserved: the admin
    agent's document-extraction contract joins ``paragraphs`` in this order.
    """
    paragraphs: list[dict[str, Any]] = []
    for raw in segments or []:
        page = raw.get("page_number", raw.get("page"))
        paragraphs.append(
            {
                "left": float(raw.get("left", 0) or 0),
                "top": float(raw.get("top", 0) or 0),
                "width": float(raw.get("width", 0) or 0),
                "height": float(raw.get("height", 0) or 0),
                "page_number": int(page or 0),
                "text": str(raw.get("text", "")),
                "type": str(raw.get("type", "")),
            }
        )
    return paragraphs


def joined_paragraph_text(paragraphs: list[dict[str, Any]]) -> str:
    """The admin agent's document-extraction input (pure).

    The admin contract is literally ``"\\n".join(p["text"] for p in
    seg["paragraphs"])``; no filtering, no page markers, no truncation.
    """
    return "\n".join(str(paragraph.get("text", "")) for paragraph in paragraphs)


def context_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """``ctx`` exactly as the admin agent's ``build_context`` builds it (pure)."""
    return {
        "shared_id": str(record.get("shared_id") or record.get("doc_id") or ""),
        "title": str(record.get("title") or ""),
        "metadata": record.get("metadata") or {},
    }


def raw_entity_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """The raw entity shape ``peek_entity_files`` / benchmark reads expect (pure).

    The admin agent strips files from :class:`AgentEntity`, so the raw
    repository is the only way its tools can see that a primary document exists.
    We expose one ``documents`` entry (the segmented PDF) and no attachments.
    """
    file_id = record.get("file_id")
    documents: list[dict[str, Any]] = []
    if file_id:
        documents.append(
            {
                "_id": str(file_id),
                "filename": record.get("filename") or "",
                "originalname": record.get("originalname") or record.get("filename") or "",
                "language": record.get("file_language") or "eng",
                "mimetype": record.get("mimetype") or "application/pdf",
                "size": record.get("size"),
            }
        )
    raw: dict[str, Any] = {
        "sharedId": str(record.get("shared_id") or record.get("doc_id") or ""),
        "title": record.get("title") or "",
        "language": record.get("language") or "en",
        "template": record.get("template_id") or record.get("template") or "",
        "metadata": record.get("metadata") or {},
        "documents": documents,
        "attachments": [],
    }
    if record.get("entity_id"):
        raw["_id"] = str(record["entity_id"])
    return raw


# ---------------------------------------------------------------------------
# Admin-agent domain objects (sidecar only; lazy imports)


def agent_entity_from_record(record: dict[str, Any]) -> Any:
    """Build ``uwazi_agent.domain.agent_entity.AgentEntity`` from a record."""
    from uwazi_agent.domain.agent_entity import AgentEntity

    return AgentEntity(
        shared_id=str(record.get("shared_id") or record.get("doc_id") or ""),
        title=str(record.get("title") or ""),
        template_name=str(record.get("template_name") or ""),
        metadata=record.get("metadata") or {},
        language=str(record.get("language") or "en"),
    )


def segmentation_from_record(record: dict[str, Any]) -> Any:
    """Build ``uwazi_api.domain.segmentation.Segmentation`` from a record.

    ``Paragraph.page_number`` is an alias (``pageNumber``) and the model does not
    set ``populate_by_name``, so the alias is used explicitly.
    """
    from uwazi_api.domain.segmentation import Paragraph, Segmentation

    paragraphs = paragraph_dicts(record.get("segments"))
    file_id = str(record.get("file_id") or f"{record.get('doc_id')}:file")
    return Segmentation(
        id=str(record.get("segmentation_id") or f"{file_id}:segmentation"),
        file_id=file_id,
        document_id=str(record.get("doc_id") or ""),
        filename=str(record.get("filename") or record.get("title") or ""),
        status="ready",
        paragraphs=[
            Paragraph(
                left=paragraph["left"],
                top=paragraph["top"],
                width=paragraph["width"],
                height=paragraph["height"],
                pageNumber=paragraph["page_number"],
                text=paragraph["text"],
                type=paragraph["type"],
            )
            for paragraph in paragraphs
        ],
    )


def agent_template_from_record(record: dict[str, Any]) -> Any:
    """Build ``uwazi_agent.domain.agent_template.AgentTemplate`` from a record."""
    from uwazi_agent.domain.agent_property import AgentProperty
    from uwazi_agent.domain.agent_property_type import AgentPropertyType
    from uwazi_agent.domain.agent_template import AgentTemplate

    properties = []
    for raw in record.get("properties") or []:
        type_name = _PROPERTY_TYPE_NAMES.get(str(raw.get("type") or "").casefold(), "TEXT")
        properties.append(
            AgentProperty(
                name=str(raw.get("name") or ""),
                type=getattr(AgentPropertyType, type_name),
                required=bool(raw.get("required")),
                use_as_filter=bool(raw.get("use_as_filter", raw.get("filter"))),
                show_in_card=bool(raw.get("show_in_card", raw.get("showInCard"))),
            )
        )
    return AgentTemplate(
        name=str(record.get("name") or ""),
        properties=properties,
        color=str(record.get("color") or ""),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "agent_entity_from_record",
    "agent_template_from_record",
    "context_from_record",
    "joined_paragraph_text",
    "load_json",
    "paragraph_dicts",
    "raw_entity_from_record",
    "read_json",
    "segmentation_from_record",
    "write_json",
]

"""Thin data models for Uwazi entities, files and segmentations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    """One paragraph/line as produced by Uwazi's PDF segmentation."""

    text: str
    page: int
    top: float
    left: float
    width: float
    height: float
    type: str
    page_width: float
    page_height: float

    @classmethod
    def from_mongo(cls, raw: dict[str, Any]) -> Segment:
        return cls(
            text=str(raw.get("text", "")),
            page=int(raw.get("page_number", raw.get("page", 0)) or 0),
            top=float(raw.get("top", 0) or 0),
            left=float(raw.get("left", 0) or 0),
            width=float(raw.get("width", 0) or 0),
            height=float(raw.get("height", 0) or 0),
            type=str(raw.get("type", "")),
            page_width=float(raw.get("page_width", 0) or 0),
            page_height=float(raw.get("page_height", 0) or 0),
        )

    def as_dict(self, include_position: bool = True, include_type: bool = True) -> dict[str, Any]:
        item: dict[str, Any] = {"text": self.text}
        if include_type:
            item["type"] = self.type
        if include_position:
            item.update(
                page=self.page,
                top=round(self.top, 1),
                left=round(self.left, 1),
                width=round(self.width, 1),
                height=round(self.height, 1),
            )
        return item


@dataclass
class Document:
    """An entity plus the one file/segmentation we treat as its document."""

    entity_id: str
    shared_id: str
    title: str
    language: str
    template: str
    metadata: dict[str, Any]
    file_id: str | None = None
    filename: str | None = None
    file_language: str | None = None
    mimetype: str | None = None
    total_pages: int | None = None
    segments: list[Segment] = field(default_factory=list)

    @property
    def has_segmentation(self) -> bool:
        return bool(self.segments)

    @property
    def page_count(self) -> int:
        if self.segments:
            return max(segment.page for segment in self.segments)
        return int(self.total_pages or 0)


def metadata_items(metadata: dict[str, Any] | None, prop: str) -> list[dict[str, Any]]:
    """Return the raw Uwazi metadata values for a property."""
    return list((metadata or {}).get(prop) or [])


def metadata_values(metadata: dict[str, Any] | None, prop: str) -> list[Any]:
    """Return non-empty primitive values for a property."""
    values: list[Any] = []
    for item in metadata_items(metadata, prop):
        value = item.get("value")
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        values.append(value)
    return values


def metadata_labels(metadata: dict[str, Any] | None, prop: str) -> list[str]:
    """Return non-empty labels for a property (falling back to the value)."""
    labels: list[str] = []
    for item in metadata_items(metadata, prop):
        label = item.get("label")
        value = item.get("value")
        if label:
            labels.append(str(label))
        elif value is not None and not (isinstance(value, str) and not value.strip()):
            labels.append(str(value))
    return labels

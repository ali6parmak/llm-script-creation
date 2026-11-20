"""MongoDB access for the two restored Uwazi instances.

Join chain: entities.sharedId -> files.entity -> segmentations.fileID
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from .configuration import MONGO_URI
from .models import Document, Segment

PLAN_DB = "plan"
UPR_DB = "upr"

PLAN_DOCUMENT_TEMPLATE = "5bfbb1a0471dd0fc16ada146"
UPR_DOCUMENT_TEMPLATE = "5e57cdd6f54e0a1304c0d50d"
UPR_RECOMMENDATION_TEMPLATE = "5d8ce04361cde0408222e9a8"
UPR_STATE_TEMPLATE = "5d8cdec361cde0408222d3ec"

# Uwazi entity language code -> Uwazi file language code
FILE_LANGUAGES = {"en": "eng", "fr": "fra", "es": "spa"}


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)


def get_db(name: str) -> Database:
    return get_client()[name]


def iter_entities(
    db: Database,
    template_id: str | None = None,
    language: str | None = None,
    query: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate entities in a stable order, filtered by template and language."""
    selector: dict[str, Any] = dict(query or {})
    if template_id is not None:
        from bson import ObjectId

        selector["template"] = ObjectId(template_id)
    if language is not None:
        selector["language"] = language
    yield from db.entities.find(selector, projection).sort("_id", ASCENDING)


def find_document_file(
    db: Database,
    shared_id: str,
    preferred_language: str = "eng",
) -> dict[str, Any] | None:
    """Find the document file for an entity, preferring the requested language.

    An entity can own several files (one per language, plus custom images).
    """
    base = {"entity": shared_id, "type": "document"}
    preferred = db.files.find_one(
        {**base, "language": preferred_language}, sort=[("_id", ASCENDING)]
    )
    if preferred is not None:
        return preferred
    return db.files.find_one(base, sort=[("_id", ASCENDING)])


def load_segmentation(db: Database, file_id: Any) -> dict[str, Any] | None:
    return db.segmentations.find_one({"fileID": file_id, "status": "ready"})


def load_document(
    db: Database, entity: dict[str, Any], file_language: str | None = None
) -> Document:
    """Build a `Document` for an entity, attaching its segmentation if present."""
    shared_id = entity["sharedId"]
    language = entity.get("language") or "en"
    preferred = file_language or FILE_LANGUAGES.get(language, "eng")
    file_doc = find_document_file(db, shared_id, preferred)

    document = Document(
        entity_id=str(entity["_id"]),
        shared_id=shared_id,
        title=entity.get("title") or "",
        language=language,
        template=str(entity.get("template")),
        metadata=entity.get("metadata") or {},
    )
    if file_doc is None:
        return document

    document.file_id = str(file_doc["_id"])
    document.filename = file_doc.get("filename")
    document.file_language = file_doc.get("language")
    document.mimetype = file_doc.get("mimetype")
    document.total_pages = file_doc.get("totalPages")

    segmentation = load_segmentation(db, file_doc["_id"])
    if segmentation is not None:
        document.segments = [
            Segment.from_mongo(raw)
            for raw in segmentation.get("segmentation", {}).get("paragraphs", [])
        ]
    return document


def iter_documents(
    db: Database,
    template_id: str,
    language: str = "en",
    require_segmentation: bool = True,
    limit: int | None = None,
) -> Iterable[Document]:
    """Yield documents for a template/language, optionally only segmented ones."""
    count = 0
    for entity in iter_entities(db, template_id=template_id, language=language):
        document = load_document(db, entity)
        if require_segmentation and not document.has_segmentation:
            continue
        yield document
        count += 1
        if limit is not None and count >= limit:
            break

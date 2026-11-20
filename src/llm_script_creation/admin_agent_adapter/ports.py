"""Read-only port implementations backed by the sidecar request payload.

The admin agent's extraction path reads through ports: entities, raw entities,
templates, segmentation, files and (optionally) external URLs. Production wires
those to Uwazi. Here they are backed by the JSON request our benchmark process
wrote, containing ONLY train documents (the authoring request never contains a
test document). Nothing can reach Mongo, Uwazi or the network from these ports.

Every write method raises ``NotImplementedError`` with an explicit "read-only
adapter" message, so a generated extractor cannot mutate anything even if the
model tries. Only the read paths the extractor tools actually need are real.

This module is sidecar-only: it imports the admin agent's port ABCs and domain
models, so the benchmark's main process must not import it.
"""

from __future__ import annotations

from typing import Any

from uwazi_admin_agent.ports.entity_repository_port import EntityRepositoryPort
from uwazi_admin_agent.ports.file_repository_port import FileRepositoryPort
from uwazi_admin_agent.ports.segmentation_repository_port import SegmentationRepositoryPort
from uwazi_admin_agent.ports.url_fetcher_port import UrlFetcherPort, UrlFetchError
from uwazi_admin_agent.use_cases.admin_agent_deps import AdminAgentDeps
from uwazi_agent.domain.agent_entity import AgentEntity
from uwazi_agent.domain.agent_entity_search_result import AgentEntitySearchResult
from uwazi_agent.domain.agent_entity_summary import AgentEntitySummary
from uwazi_agent.domain.agent_publish_status import AgentPublishStatus
from uwazi_agent.domain.agent_search_filter import AgentSearchFilter
from uwazi_agent.domain.agent_template import AgentTemplate
from uwazi_agent.ports.entity_api_port import EntityApiPort
from uwazi_agent.ports.template_api_port import TemplateApiPort
from uwazi_agent.ports.template_mapper_port import TemplateMapperPort
from uwazi_agent.ports.thesauri_api_port import ThesauriApiPort
from uwazi_api.domain.exceptions import DomainError, EntityNotFoundError, SegmentationNotFoundError

from . import records

_READ_ONLY = "read-only benchmark adapter: {operation} is not supported"


def _not_supported(operation: str) -> NotImplementedError:
    return NotImplementedError(_READ_ONLY.format(operation=operation))


class StaticEntityApi(EntityApiPort):
    """``EntityApiPort`` reads over the request's train entities."""

    def __init__(self, entities: list[AgentEntity], template_name: str) -> None:
        self.entities = list(entities)
        self.template_name = template_name
        self._by_id = {entity.shared_id: entity for entity in self.entities}

    @staticmethod
    def _summary(entities: list[AgentEntity], count: int | None = None) -> AgentEntitySummary:
        by_template: dict[str, int] = {}
        for entity in entities:
            by_template[entity.template_name] = by_template.get(entity.template_name, 0) + 1
        return AgentEntitySummary(
            count=len(entities) if count is None else count,
            by_template=by_template,
            sample_titles=[entity.title or "" for entity in entities[:5]],
            shared_ids=[entity.shared_id for entity in entities[:20]],
        )

    @classmethod
    def _result(cls, entities: list[AgentEntity]) -> AgentEntitySearchResult:
        result = AgentEntitySearchResult(summary=cls._summary(entities), examples=entities[:5])
        result._all_entities = list(entities)
        return result

    async def get_entities_by_shared_ids(
        self, shared_ids: list[str], language: str, limit: int = 10000
    ) -> list[AgentEntity]:
        del language, limit  # the benchmark data is English-only
        return [self._by_id[shared_id] for shared_id in shared_ids if shared_id in self._by_id]

    async def get_entities_by_template(
        self, template_name: str, language: str, limit: int
    ) -> AgentEntitySearchResult:
        del language
        entities = [entity for entity in self.entities if entity.template_name == template_name]
        if not entities:
            return self._result([])
        return self._result(entities[: max(1, int(limit))])

    async def search_entities_by_text(
        self,
        search_term: str,
        template_name: str | None,
        language: str,
        limit: int,
    ) -> AgentEntitySearchResult:
        del language
        needle = (search_term or "").casefold()
        entities = [
            entity
            for entity in self.entities
            if (template_name is None or entity.template_name == template_name)
            and (needle in (entity.title or "").casefold() or needle in entity.shared_id.casefold())
        ]
        return self._result(entities[: max(1, int(limit))])

    async def search_entities_by_filter(
        self,
        template_name: str,
        filters: list[AgentSearchFilter],
        language: str,
        limit: int,
        published: bool | None = None,
    ) -> AgentEntitySearchResult:
        del language, published
        if not filters:
            return await self.get_entities_by_template(template_name, "en", limit)

        matches = []
        for entity in self.entities:
            if entity.template_name != template_name:
                continue
            if all(self._matches_filter(entity, item) for item in filters):
                matches.append(entity)
        return self._result(matches[: max(1, int(limit))])

    @staticmethod
    def _values(entity: AgentEntity, property_name: str) -> list[str]:
        raw = entity.metadata.get(property_name)
        items = raw if isinstance(raw, list) else [raw] if raw is not None else []
        values: list[str] = []
        for item in items:
            if isinstance(item, dict):
                label = item.get("label")
                value = item.get("value")
                names = [label, value] if label is not None else [value]
            else:
                names = [item]
            values.extend(str(name) for name in names if name is not None)
        return values

    @classmethod
    def _matches_filter(cls, entity: AgentEntity, item: AgentSearchFilter) -> bool:
        if item.date_from or item.date_to:
            raise DomainError(
                "admin_agent_adapter: date filters are not supported by the read-only benchmark adapter "
                f"(property {item.property_name!r}); use by_template or by_ids instead."
            )
        wanted = {value.casefold() for value in (item.values or [])}
        if not wanted:
            return True
        present = {value.casefold() for value in cls._values(entity, item.property_name)}
        return bool(present & wanted)

    async def create_entities(self, *args: Any, **kwargs: Any) -> Any:
        raise _not_supported("create_entities")

    async def update_entities(self, *args: Any, **kwargs: Any) -> Any:
        raise _not_supported("update_entities")

    async def delete_entities_by_shared_ids(self, *args: Any, **kwargs: Any) -> Any:
        raise _not_supported("delete_entities_by_shared_ids")

    async def set_entities_publish_status(self, *args: Any, **kwargs: Any) -> Any:
        raise _not_supported("set_entities_publish_status")

    async def get_publish_status(self, shared_ids: list[str], language: str) -> list[AgentPublishStatus]:
        del language
        return [
            AgentPublishStatus(shared_id=shared_id, published=True)
            for shared_id in shared_ids
            if shared_id in self._by_id
        ]


class StaticEntityRepository(EntityRepositoryPort):
    """``EntityRepositoryPort`` reads over the request's raw entity dicts."""

    def __init__(self, raws: dict[str, dict[str, Any]]) -> None:
        self.raws = raws

    async def get_raw_by_shared_id(self, shared_id: str, language: str | None = None) -> dict[str, Any]:
        del language
        raw = self.raws.get(shared_id)
        if raw is None:
            raise EntityNotFoundError(f"entity {shared_id!r} is not in this read-only request")
        return raw

    async def get_raw_by_internal_id(self, internal_id: str) -> dict[str, Any]:
        for raw in self.raws.values():
            if str(raw.get("_id")) == str(internal_id):
                return raw
        raise EntityNotFoundError(f"entity internal id {internal_id!r} is not in this read-only request")

    async def save_raw(self, *args: Any, **kwargs: Any) -> None:
        raise _not_supported("save_raw")

    async def create_raw(self, *args: Any, **kwargs: Any) -> str:
        raise _not_supported("create_raw")

    async def delete_by_shared_id(self, *args: Any, **kwargs: Any) -> None:
        raise _not_supported("delete_by_shared_id")


class StaticFileRepository(FileRepositoryPort):
    """File bytes are deliberately unavailable: the benchmark stores no PDFs."""

    async def get_file_bytes(self, filename: str) -> bytes | None:
        del filename
        return None

    async def upload_document(self, *args: Any, **kwargs: Any) -> bool:
        raise _not_supported("upload_document")

    async def upload_attachment(self, *args: Any, **kwargs: Any) -> bool:
        raise _not_supported("upload_attachment")

    async def delete_file(self, *args: Any, **kwargs: Any) -> bool:
        raise _not_supported("delete_file")


class StaticSegmentationRepository(SegmentationRepositoryPort):
    """``SegmentationRepositoryPort`` reads over the request's train documents."""

    def __init__(self, segmentations: dict[str, Any], file_ids: dict[str, str]) -> None:
        self.segmentations = segmentations
        self.file_ids = file_ids

    async def get_by_file_id(self, file_id: str) -> Any:
        segmentation = self.segmentations.get(file_id)
        if segmentation is None:
            raise SegmentationNotFoundError(
                f"no segmentation for file {file_id!r} in this read-only request"
            )
        return segmentation

    async def get_by_shared_id(self, shared_id: str, language: str) -> Any:
        del language
        file_id = self.file_ids.get(shared_id)
        segmentation = self.segmentations.get(file_id) if file_id else None
        if segmentation is None:
            raise SegmentationNotFoundError(
                f"no segmentation for entity {shared_id!r} in this read-only request"
            )
        return segmentation


class StaticTemplateApi(TemplateApiPort):
    """``TemplateApiPort`` over the single template in the request."""

    def __init__(self, template: AgentTemplate) -> None:
        self.template = template

    async def get_templates(self) -> list[AgentTemplate]:
        return [self.template]

    async def get_templates_by_names(self, names: list[str]) -> list[AgentTemplate]:
        wanted = {name.casefold() for name in names}
        return [self.template] if self.template.name.casefold() in wanted else []

    async def get_template_names(self) -> list[str]:
        return [self.template.name]

    async def create_template(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("create_template")

    async def update_template(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("update_template")

    async def delete_template(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("delete_template")


class StaticTemplateMapper(TemplateMapperPort):
    """Mapping API templates is unused by the extractor; reads mirror the record."""

    def __init__(self, template: AgentTemplate) -> None:
        self.template = template

    def to_agent(self, api_template: Any) -> AgentTemplate:
        del api_template
        return self.template

    def to_api(self, agent_template: AgentTemplate) -> Any:
        raise _not_supported("TemplateMapperPort.to_api")


class StaticThesauriApi(ThesauriApiPort):
    """No thesaurus reads are needed to author a document extractor."""

    async def get_thesauris(self, language: str) -> list[Any]:
        del language
        return []

    async def get_thesauris_by_names(self, names: list[str], language: str) -> list[Any]:
        del names, language
        return []

    async def create_thesauri(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("create_thesauri")

    async def update_thesauri(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("update_thesauri")

    async def delete_thesauri(self, *args: Any, **kwargs: Any) -> dict:
        raise _not_supported("delete_thesauri")


class ReadOnlyUrlFetcher(UrlFetcherPort):
    """External URLs are out of scope: the benchmark extracts from Uwazi text."""

    async def fetch_text(self, url: str) -> str:
        raise UrlFetchError(f"external URL fetching is disabled in the read-only benchmark adapter: {url}")


def build_deps(request: dict[str, Any]) -> AdminAgentDeps:
    """Build :class:`AdminAgentDeps` over one author request.

    ``request['train_entities']`` is the complete universe: no test document is
    written into the authoring request, so no port can serve one.
    """
    template = records.agent_template_from_record(request["template"])
    entity_records = list(request.get("train_entities") or [])
    entities = [records.agent_entity_from_record(record) for record in entity_records]

    raws = {
        record["shared_id"]: records.raw_entity_from_record(record) for record in entity_records
    }
    segmentations: dict[str, Any] = {}
    file_ids: dict[str, str] = {}
    for record in entity_records:
        segmentation = records.segmentation_from_record(record)
        file_ids[record["shared_id"]] = segmentation.file_id
        segmentations[segmentation.file_id] = segmentation

    return AdminAgentDeps(
        thesauri_api=StaticThesauriApi(),
        template_api=StaticTemplateApi(template),
        template_mapper=StaticTemplateMapper(template),
        entity_api=StaticEntityApi(entities, template.name),
        entity_repository=StaticEntityRepository(raws),
        file_repository=StaticFileRepository(),
        segmentation_repository=StaticSegmentationRepository(segmentations, file_ids),
        url_fetcher=ReadOnlyUrlFetcher(),
    )


__all__ = ["build_deps"]

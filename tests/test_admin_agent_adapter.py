"""Pure tests for the admin-agent adapter (no Mongo, no LLM, no admin venv)."""

from __future__ import annotations

import pytest

from llm_script_creation.admin_agent_adapter import evaluate, prompt, records


def _segment(page: int, text: str, top: float = 0.0, left: float = 0.0) -> dict:
    return {
        "text": text,
        "page": page,
        "top": top,
        "left": left,
        "width": 100.0,
        "height": 10.0,
        "type": "Paragraph",
        "page_width": 600.0,
        "page_height": 800.0,
    }


def test_paragraph_dicts_maps_catalog_page_to_page_number() -> None:
    paragraphs = records.paragraph_dicts([_segment(2, "second"), _segment(1, "first", top=5.0)])

    assert [item["page_number"] for item in paragraphs] == [2, 1]
    assert [item["text"] for item in paragraphs] == ["second", "first"]
    assert paragraphs[0]["width"] == 100.0
    assert "page" not in paragraphs[0]


def test_paragraph_dicts_accepts_raw_page_number() -> None:
    raw = {**_segment(3, "raw"), "page_number": 7}
    raw.pop("page")

    assert records.paragraph_dicts([raw])[0]["page_number"] == 7


def test_joined_paragraph_text_is_the_admin_contract() -> None:
    paragraphs = records.paragraph_dicts([_segment(1, "a"), _segment(1, ""), _segment(2, "b")])

    # No filtering, no page markers: exactly "\n".join(p["text"] ...).
    assert records.joined_paragraph_text(paragraphs) == "a\n\nb"


def test_context_shape_matches_admin_build_context() -> None:
    record = {"shared_id": "abc", "title": "A title", "metadata": {"year": [{"value": 2019}]}}

    assert records.context_from_record(record) == {
        "shared_id": "abc",
        "title": "A title",
        "metadata": {"year": [{"value": 2019}]},
    }
    assert records.context_from_record({"doc_id": "x"})["title"] == ""


def test_raw_entity_exposes_the_primary_document() -> None:
    record = {
        "shared_id": "abc",
        "title": "T",
        "template_id": "tpl",
        "language": "en",
        "metadata": {"year": [{"value": 2019}]},
        "file_id": "file-1",
        "filename": "doc.pdf",
        "mimetype": "application/pdf",
    }
    raw = records.raw_entity_from_record(record)

    assert raw["sharedId"] == "abc"
    assert raw["template"] == "tpl"
    assert raw["documents"][0]["_id"] == "file-1"
    assert raw["documents"][0]["originalname"] == "doc.pdf"
    assert raw["attachments"] == []


def test_prompt_states_source_output_and_train_only() -> None:
    task = prompt.build_author_task(
        dataset_id="plan_year",
        property_name="year",
        property_type="numeric",
        template_name="Document",
        train_total=2,
    )

    assert 'Template: "Document"' in task
    assert "`year`" in task
    assert 'Return `{"year": <value>}`' in task
    assert "Return `None`" in task
    assert '`"\\n".join(p["text"] for p in seg["paragraphs"])`' in task
    assert '{"shared_id": str, "title": str, "metadata": {property: [values]}}' in task
    assert "peek_segmentation_by_entity" in task
    # No injected sample list: the extractor discovers entities itself.
    assert "Candidate train shared_ids" not in task
    assert "by_ids" not in task
    assert "test" not in task.lower().replace("latest", "")


def test_prompt_lists_allowed_select_values() -> None:
    task = prompt.build_author_task(
        dataset_id="plan_document_type",
        property_name="document_type",
        property_type="select",
        template_name="Document",
        train_total=1,
        allowed_values=["Resolution", "Report"],
    )

    assert "ALLOWED VALUES" in task
    assert "- Resolution" in task
    assert "- Report" in task


def test_score_records_maps_hit_wrong_missing_and_error() -> None:
    samples = [
        {"doc_id": "a", "split": "train", "gold": {"year": 2019}},
        {"doc_id": "b", "split": "test", "gold": {"year": 2020}},
        {"doc_id": "c", "split": "test", "gold": {"year": 2021}},
        {"doc_id": "d", "split": "test", "gold": {"year": 1999}},
    ]
    predictions = {
        "a": {"prediction": {"year": 2019}},
        "b": {"prediction": {"year": 2020}},
        "c": {"prediction": None},
        "d": {"prediction": {"year": 2000}},
    }

    scored, metrics = evaluate.score_records("plan_year", ["year"], samples, predictions)
    by_id = {record["doc_id"]: record for record in scored}

    assert by_id["a"]["status"] == "hit"
    assert by_id["b"]["status"] == "hit"
    assert by_id["c"]["status"] == "missing_prediction"
    assert by_id["d"]["status"] == "wrong"
    assert metrics["primary"]["train"] == 1.0
    assert metrics["primary"]["test"] == pytest.approx(1 / 3)
    assert metrics["status_counts"]["test"] == {"hit": 1, "missing_prediction": 1, "wrong": 1}
    assert metrics["train_documents"] == 1
    assert metrics["test_documents"] == 3


def test_score_records_marks_execution_errors() -> None:
    samples = [{"doc_id": "a", "split": "test", "gold": {"year": 2019}}]
    predictions = {"a": {"prediction": None, "error": "ValueError: boom"}}

    _, metrics = evaluate.score_records("plan_year", ["year"], samples, predictions)

    assert metrics["status_counts"]["test"] == {"error": 1}
    assert metrics["errors"][0]["doc_id"] == "a"


def test_score_records_rejects_non_phase_one_datasets() -> None:
    with pytest.raises(ValueError):
        evaluate.score_records("upr_state", ["state_under_review"], [], {})


def test_records_models_load_with_admin_agent_when_available() -> None:
    pytest.importorskip("uwazi_agent.domain.agent_entity")
    pytest.importorskip("uwazi_api.domain.segmentation")

    record = {
        "shared_id": "abc",
        "doc_id": "abc",
        "title": "T",
        "template_name": "Document",
        "language": "en",
        "metadata": {"year": [{"value": 2019}]},
        "file_id": "file-1",
        "filename": "doc.pdf",
        "segments": [_segment(1, "2019 text")],
    }
    entity = records.agent_entity_from_record(record)
    segmentation = records.segmentation_from_record(record)

    assert entity.shared_id == "abc"
    assert entity.metadata["year"] == [{"value": 2019}]
    assert segmentation.paragraphs[0].page_number == 1

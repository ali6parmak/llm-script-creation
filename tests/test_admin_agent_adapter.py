"""Pure tests for the admin-agent adapter (no Mongo, no LLM, no admin venv)."""

from __future__ import annotations

import pytest

from llm_script_creation import runs
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
    assert '{"shared_id": str, "title": "" (hidden), "metadata": {property: [values]}}' in task
    assert "Title is HIDDEN" in task
    assert "target property `year` is REMOVED" in task
    assert "peek_segmentation_by_entity" in task
    # No injected sample list: the extractor discovers entities itself.
    assert "Candidate train shared_ids" not in task
    assert "by_ids" not in task
    assert "test" not in task.lower().replace("latest", "")


def test_prompt_title_visible_variant() -> None:
    task = prompt.build_author_task(
        dataset_id="plan_year",
        property_name="year",
        property_type="numeric",
        template_name="Document",
        train_total=2,
        title_visible=True,
        target_metadata_hidden=False,
    )

    assert '{"shared_id": str, "title": str, "metadata": {property: [values]}}' in task
    assert "Title is VISIBLE" in task
    assert "Metadata is visible, including the target property" in task


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


def test_score_records_rejects_unsupported_datasets() -> None:
    with pytest.raises(ValueError):
        evaluate.score_records("plan_all_fields", ["year"], [], {})


def test_score_records_maps_upr_state_property() -> None:
    samples = [
        {"doc_id": "a", "split": "test", "gold": {"state_under_review": "Afghanistan"}},
    ]
    predictions = {"a": {"prediction": {"state_reviewed": "Afghanistan"}}}

    scored, metrics = evaluate.score_records(
        "upr_state", ["state_under_review"], samples, predictions
    )

    assert scored[0]["status"] == "hit"
    assert scored[0]["prediction"]["state_under_review"] == "Afghanistan"
    assert metrics["primary"]["metric"] == "accuracy"
    assert metrics["primary"]["test"] == 1.0


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


def test_recommendations_task_shape_and_candidate_list() -> None:
    task = prompt.build_recommendations_task(
        dataset_id="upr_recommendations",
        template_name="Document",
        train_total=50,
        state_labels=[f"State {index}" for index in range(60)],
    )

    assert '"recommendations"' in task
    assert '"number"' in task and '"text"' in task and '"countries"' in task
    assert "CANDIDATE STATE LABELS" in task
    assert "State 59" in task  # the full candidate list is shown, not just the first 20
    assert "benchmark_extractor" in task


def test_score_records_scores_upr_recommendation_records() -> None:
    gold_recommendations = [
        {"number": "95.24", "text": "End the practice.", "countries": ["France"]},
        {"number": "95.25", "text": "Ratify the treaty.", "countries": ["Spain"]},
    ]
    samples = [
        {
            "doc_id": "a",
            "split": "test",
            "gold": {"state_under_review": "Albania", "recommendations": gold_recommendations},
        }
    ]
    predictions = {
        "a": {
            "prediction": {
                "state_under_review": "Albania",
                "recommendations": gold_recommendations,
            }
        }
    }

    scored, metrics = evaluate.score_records(
        "upr_recommendations",
        ["state_under_review", "recommendations"],
        samples,
        predictions,
    )

    assert scored[0]["status"] == "hit"
    assert metrics["primary"]["metric"] == "f1"
    assert metrics["primary"]["test"] == 1.0


def test_recommendations_task_no_country_list_variant() -> None:
    task = prompt.build_recommendations_task(
        dataset_id="upr_recommendations",
        template_name="Document",
        train_total=50,
        state_labels=["France", "Spain"],
        include_state_list=False,
    )

    assert "CANDIDATE STATE LABELS" not in task
    assert "France" not in task
    assert "NO CANDIDATE STATE LIST IS PROVIDED" in task
    assert prompt.PROMPT_VERSION_RECOMMENDATIONS_NO_LIST != prompt.PROMPT_VERSION_RECOMMENDATIONS


def test_geometry_prefixes_paragraphs() -> None:
    record = {
        "shared_id": "abc",
        "segments": [
            {"text": "hello", "page": 2, "left": 10.4, "top": 20.6, "width": 100.0, "height": 9.0}
        ],
    }

    transformed = records.with_geometry(record)
    text = transformed["segments"][0]["text"]

    assert text == "[p2 x=10 y=21 w=100 h=9] hello"
    assert record["segments"][0]["text"] == "hello"  # input not mutated


def test_geometry_prompt_describes_prefix() -> None:
    task = prompt.build_author_task(
        dataset_id="plan_year",
        property_name="year",
        property_type="numeric",
        template_name="Document",
        train_total=50,
        geometry=True,
    )

    assert "GEOMETRY condition" in task
    assert "[pPAGE x=LEFT y=TOP w=WIDTH h=HEIGHT]" in task
    assert "The prefix is metadata" in task
    assert "returning any text value" in task


def test_no_metadata_prompt_condition() -> None:
    task = prompt.build_author_task(
        dataset_id="plan_year",
        property_name="year",
        property_type="numeric",
        template_name="Document",
        train_total=50,
        metadata_visible=False,
    )

    assert "`ctx['metadata']` is EMPTY" in task
    assert "Do not read any metadata" in task
    assert "target property `year` is REMOVED" not in task


def test_spec_digest_is_stable_and_sensitive_to_sampling() -> None:
    from llm_script_creation.admin_agent_adapter import cli

    base = {
        "dataset_id": "plan_year",
        "prompt_sha256": "abc",
        "condition": {"title_visible": False, "geometry": False},
        "sampling": {"policy": "agent_all_train", "selected_shared_ids": ["a", "b"]},
    }
    same = {**base, "sampling": {"policy": "agent_all_train", "selected_shared_ids": ["a", "b"]}}
    different = {**base, "sampling": {"policy": "agent_all_train", "selected_shared_ids": ["a", "c"]}}

    assert cli._spec_digest(base) == cli._spec_digest(same)
    assert cli._spec_digest(base) != cli._spec_digest(different)


def test_build_method_and_run_record_repair_rounds() -> None:
    method = evaluate.build_method(
        method_id="m1",
        name="M",
        description="D",
        dataset_id="plan_year",
        task="numeric",
        model="model",
        extractor_code="def extract(text, ctx):\n    return None\n",
        prompt="task",
        manifest={"split_policy_version": "v2"},
        config={},
        usage={},
        primary={"train": 0.5, "test": 0.4, "gap": 0.1},
        train_documents=1,
        test_documents=1,
        created_at="now",
        prompt_version="v",
        repair_rounds=2,
    )
    run = evaluate.build_run(
        run_id="r1",
        method_id="m1",
        dataset_id="plan_year",
        status="completed",
        created_at="now",
        counts={},
        metrics={},
        manifest={"split_policy_version": "v2"},
        usage={},
        repair_rounds=2,
    )

    assert method["repair_rounds"] == 2
    assert run["repair_rounds"] == 2


def test_write_method_persists_spec(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runs, "METHODS_DIR", tmp_path)
    spec = {"spec_version": 1, "spec_hash": "abc123", "sampling": {"selected_shared_ids": ["a"]}}

    runs.write_method(
        {"method_id": "m1", "name": "M"},
        prompt="task",
        script="def extract(text, ctx):\n    return None\n",
        spec=spec,
    )

    saved = (tmp_path / "m1" / "spec.json").read_text(encoding="utf-8")
    assert '"spec_hash": "abc123"' in saved
    assert (tmp_path / "m1" / "method.json").exists()
    assert (tmp_path / "m1" / "prompt.txt").exists()
    assert (tmp_path / "m1" / "script.py").exists()

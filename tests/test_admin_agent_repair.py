"""Pure tests for the admin-agent repair helpers (no Mongo, no LLM)."""

from __future__ import annotations

import pytest

from llm_script_creation.admin_agent_adapter import repair


def _record(doc_id: str, gold_type: str = "Resolution", year: int = 2010) -> dict:
    return {"doc_id": doc_id, "gold": {"document_type": gold_type, "year": year}}


def _plan_scored(
    doc_id: str,
    *,
    status: str,
    gold: str = "Resolution",
    prediction: str | None = "Resolution",
    split: str = "train",
) -> dict:
    return {
        "doc_id": doc_id,
        "split": split,
        "status": status,
        "error": "ValueError: boom" if status == "error" else None,
        "gold": {"document_type": gold},
        "prediction": None if prediction is None else {"document_type": prediction},
        "details": {
            "status": status,
            "fields": {
                "document_type": {
                    "gold": gold,
                    "prediction": prediction,
                    "gold_normalized": gold,
                    "prediction_normalized": prediction,
                    "present": prediction is not None,
                    "hit": status == "hit",
                }
            },
        },
    }


def test_split_validation_is_deterministic_and_proportional() -> None:
    records = (
        [_record(f"res{index}", "Resolution") for index in range(15)]
        + [_record(f"sp{index}", "Special Procedures' report") for index in range(6)]
        + [_record(f"ex{index}", "ExCom Conclusion") for index in range(3)]
        + [_record("rare", "Regional treaty")]
    )

    first = repair.split_validation("plan_document_type", records, validation_size=5)
    second = repair.split_validation(
        "plan_document_type", list(reversed(records)), validation_size=5
    )

    assert len(first.validation) == 5
    assert len(first.feedback) == 20
    assert first.validation_ids() == second.validation_ids()
    assert first.digest == second.digest
    assert set(first.validation_ids()).isdisjoint(first.feedback_ids())
    # proportional: the slice mirrors the mix; a singleton stays in feedback
    assert set(first.strata) == {
        "Resolution",
        "Special Procedures' report",
        "ExCom Conclusion",
        "Regional treaty",
    }
    assert first.strata["Resolution"]["validation"] >= 2
    assert first.strata["Regional treaty"]["validation"] == 0
    assert "rare" in first.feedback_ids()


def test_split_validation_caps_small_pools_and_rejects_tiny_ones() -> None:
    records = [_record(f"doc{index}") for index in range(10)]

    split = repair.split_validation("plan_document_type", records, validation_size=10)

    assert len(split.validation) == 2  # 20% cap
    assert len(split.feedback) == 8

    with pytest.raises(ValueError, match="at least two"):
        repair.split_validation("plan_document_type", records[:1])


def test_select_failures_prioritizes_status_and_deduplicates_families() -> None:
    scored = [
        _plan_scored("ok", status="hit"),
        _plan_scored("crash", status="error"),
        _plan_scored("missing", status="missing_prediction", prediction=None),
        _plan_scored("wrong-a", status="wrong", prediction="Declaration"),
        _plan_scored("wrong-b", status="wrong", prediction="Declaration"),
        _plan_scored("wrong-c", status="wrong", gold="ExCom Conclusion", prediction="Resolution"),
        _plan_scored("hidden", status="wrong", prediction="Declaration", split="validation"),
    ]

    examples = repair.select_failures(
        "plan_document_type",
        scored,
        feedback_ids={"ok", "crash", "missing", "wrong-a", "wrong-b", "wrong-c"},
        max_examples=3,
    )

    assert [example["doc_id"] for example in examples] == ["crash", "missing", "wrong-a"]
    assert len({example["family"] for example in examples}) == 3
    assert all(example["doc_id"] != "hidden" for example in examples)


def test_select_failures_never_touches_validation_ids() -> None:
    scored = [
        _plan_scored("v1", status="wrong", prediction="Declaration"),
        _plan_scored("f1", status="wrong", prediction="Declaration"),
    ]

    examples = repair.select_failures(
        "plan_document_type", scored, feedback_ids={"f1"}, max_examples=3
    )

    assert [example["doc_id"] for example in examples] == ["f1"]


def test_plan_failure_example_has_gold_prediction_and_excerpt() -> None:
    scored = [_plan_scored("f1", status="wrong", gold="Resolution", prediction="Declaration")]
    documents = {
        "f1": {
            "doc_id": "f1",
            "segments": [{"text": "A/RES/70/1 Resolution adopted by the General Assembly"}],
        }
    }

    examples = repair.select_failures(
        "plan_document_type",
        scored,
        feedback_ids={"f1"},
        max_examples=1,
        documents=documents,
    )

    assert examples[0]["family"] == "wrong:Resolution->Declaration"
    assert "gold='Resolution'" in examples[0]["detail"]
    assert "prediction='Declaration'" in examples[0]["detail"]
    assert "Resolution adopted" in examples[0]["excerpt"]


def test_upr_failure_example_reports_record_level_details() -> None:
    scored = [
        {
            "doc_id": "f1",
            "split": "train",
            "status": "partial",
            "gold": {"state_under_review": "Albania", "recommendations": []},
            "prediction": {"state_under_review": "Albania", "recommendations": []},
            "details": {
                "status": "partial",
                "state_correct": True,
                "gold_state": "Albania",
                "prediction_state": "Albania",
                "metrics": {"gold_count": 10, "prediction_count": 8, "matched": 8},
                "missed": [{"number": "12.3"}, {"number": "12.4"}],
                "extras": [],
            },
        }
    ]

    examples = repair.select_failures(
        "upr_recommendations", scored, feedback_ids={"f1"}, max_examples=1
    )

    assert examples[0]["family"] == "records_missing"
    assert "gold_recs=10" in examples[0]["detail"]
    assert "12.3" in examples[0]["detail"]


def test_upr_state_failure_example_uses_state_fields() -> None:
    scored = [
        {
            "doc_id": "f1",
            "split": "train",
            "status": "wrong",
            "gold": {"state_under_review": "Albania"},
            "prediction": {"state_under_review": "Algeria"},
            "details": {
                "status": "wrong",
                "state_correct": False,
                "gold_state": "Albania",
                "prediction_state": "Algeria",
            },
        }
    ]

    examples = repair.select_failures("upr_state", scored, feedback_ids={"f1"}, max_examples=1)

    assert examples[0]["family"] == "state"
    assert "Albania" in examples[0]["detail"]
    assert "Algeria" in examples[0]["detail"]


def test_repair_prompt_contains_code_and_examples_but_no_validation() -> None:
    examples = [
        {
            "doc_id": "f1",
            "status": "wrong",
            "family": "wrong:Resolution->Declaration",
            "field": "document_type",
            "detail": "gold='Resolution', prediction='Declaration'",
            "excerpt": "A/RES/70/1 Resolution adopted",
        }
    ]

    prompt = repair.build_repair_prompt(
        base_prompt="TASK: author an extractor.",
        current_code="def extract(text, ctx):\n    return None\n",
        examples=examples,
        round_index=1,
        total_rounds=3,
    )

    assert "REPAIR ROUND 1 of 3" in prompt
    assert "def extract(text, ctx):" in prompt
    assert "f1" in prompt
    assert "wrong:Resolution->Declaration" in prompt
    assert "never key on a shared_id" in prompt
    assert "validation" not in prompt.lower()

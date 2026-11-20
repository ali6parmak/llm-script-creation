import json

from llm_script_creation import generate
from llm_script_creation.generate import (
    _failure_group,
    _is_failure,
    primary_score,
    round_selection_key,
    select_best_round,
    select_repair_failures,
)
from llm_script_creation.llm import ChatResponse


def record(
    doc_id: str,
    status: str,
    *,
    gold: dict | None = None,
    error: str | None = None,
    f1: float = 0.0,
    state_correct: bool = False,
) -> dict:
    return {
        "doc_id": doc_id,
        "status": status,
        "error": error,
        "gold": gold or {},
        "prediction": {},
        "details": {"metrics": {"f1": f1}, "state_correct": state_correct},
    }


def test_primary_score_uses_dataset_metric() -> None:
    assert primary_score("plan_year", {"accuracy": 0.75}) == 0.75
    assert primary_score("upr_state", {"accuracy": 0.5}) == 0.5
    assert primary_score("upr_recommendations", {"f1": 0.42, "accuracy": 1.0}) == 0.42


def test_failure_detection() -> None:
    assert _is_failure("plan_year", record("a", "wrong")) is True
    assert _is_failure("plan_year", record("a", "hit")) is False
    assert _is_failure("plan_year", record("a", "error", error="boom")) is True
    assert (
        _is_failure(
            "upr_recommendations",
            record("a", "hit", f1=0.99, state_correct=True),
        )
        is True
    )
    assert (
        _is_failure(
            "upr_recommendations",
            record("a", "hit", f1=1.0, state_correct=True),
        )
        is False
    )


def test_repair_selection_starts_with_crash_and_stratifies() -> None:
    records = [
        record("a", "wrong", gold={"document_type": "Treaty"}),
        record("b", "error", error="boom"),
        record("c", "wrong", gold={"document_type": "Resolution"}),
        record("d", "wrong", gold={"document_type": "Treaty"}),
        record("e", "wrong", gold={"document_type": "Resolution"}),
    ]
    selected = select_repair_failures("plan_document_type", records, set())
    assert len(selected) == 3
    assert selected[0]["doc_id"] == "b"
    groups = {_failure_group("plan_document_type", item) for item in selected}
    assert groups == {"crash / invalid output", "Treaty", "Resolution"}


def test_repair_selection_prefers_non_example_documents() -> None:
    records = [
        record("a", "wrong", gold={"document_type": "Treaty"}),
        record("b", "error", error="boom"),
        record("c", "wrong", gold={"document_type": "Resolution"}),
        record("d", "wrong", gold={"document_type": "Treaty"}),
    ]
    selected = select_repair_failures("plan_document_type", records, {"a"})
    assert "a" not in {item["doc_id"] for item in selected}
    assert {item["doc_id"] for item in selected} == {"b", "c", "d"}


def test_best_round_is_chosen_by_nonexample_score_not_last() -> None:
    rounds = [
        {
            "round": 0,
            "issues": [],
            "score": 0.60,
            "nonexample_score": 0.55,
        },
        {
            "round": 1,
            "issues": [],
            "score": 0.70,
            "nonexample_score": 0.68,
        },
        {
            "round": 2,
            "issues": [],
            "score": 0.50,
            "nonexample_score": 0.45,
        },
        {
            "round": 3,
            "issues": ["syntax error"],
            "score": 0.0,
            "nonexample_score": 0.0,
        },
    ]
    best = select_best_round(rounds)
    assert best is not None
    assert best["round"] == 1
    assert round_selection_key(rounds[0]) < round_selection_key(rounds[1])


def test_best_round_tie_prefers_earlier_round() -> None:
    rounds = [
        {"round": 0, "issues": [], "score": 0.5, "nonexample_score": 0.5},
        {"round": 1, "issues": [], "score": 0.5, "nonexample_score": 0.5},
    ]
    best = select_best_round(rounds)
    assert best is not None
    assert best["round"] == 0


def _catalog_row(doc_id: str) -> dict:
    return {
        "db": "plan",
        "doc_id": doc_id,
        "language": "en",
        "template": "template",
        "title": doc_id,
        "page_count": 1,
        "segment_count": 1,
        "char_count": 5,
        "token_estimate": 2,
        "file_id": None,
        "filename": None,
        "mimetype": None,
        "text": "hello",
        "segments_json": json.dumps(
            [
                {
                    "text": "hello",
                    "page": 1,
                    "top": 0,
                    "left": 0,
                    "width": 10,
                    "height": 10,
                    "type": "Text",
                }
            ]
        ),
        "metadata_json": "{}",
    }


def _sample(doc_id: str, split: str) -> dict:
    row = _catalog_row(doc_id)
    row.update(split=split, gold_json=json.dumps({"year": 2020}))
    return row


def test_generate_loop_repairs_on_train_then_freezes_for_test(monkeypatch, tmp_path) -> None:
    train = [_sample(f"train-{index}", "train") for index in range(4)]
    test = [_sample(f"test-{index}", "test") for index in range(2)]
    catalog_path = tmp_path / "catalog.db"
    catalog_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(generate, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(
        generate,
        "_load_dataset",
        lambda conn, dataset_id: {
            "dataset_id": "plan_year",
            "task": "numeric",
            "db": "plan",
            "target_schema": {"year": {"type": "number"}},
            "manifest": {"split_policy_version": "v2-50train"},
        },
    )

    class Connection:
        def close(self) -> None:
            pass

    monkeypatch.setattr(generate.catalog, "connect", lambda: Connection())
    monkeypatch.setattr(
        generate.catalog,
        "iter_samples",
        lambda conn, dataset_id, split=None: train if split == "train" else test,
    )
    monkeypatch.setattr(
        generate.catalog,
        "get_document",
        lambda conn, db, doc_id: _catalog_row(doc_id),
    )

    calls = {"count": 0}

    def fake_chat(messages, **kwargs):
        calls["count"] += 1
        value = 2020 if calls["count"] > 1 else 2019
        marker = "  # GOOD" if value == 2020 else ""
        code = f'def extract(document):\n    return {{"value": {value}}}{marker}\n'
        return ChatResponse(
            content=f"```python\n{code}```",
            model="fake",
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )

    monkeypatch.setattr(generate.llm, "chat", fake_chat)
    monkeypatch.setattr(
        generate.sandbox,
        "iter_results",
        lambda script, payloads, **kwargs: (
            {"ok": True, "output": {"value": 2020 if "GOOD" in script else 2019}}
            for _ in payloads
        ),
    )

    written: dict = {}

    def fake_write_method(method, **kwargs):
        written["method"] = method

    def fake_write_run(run, records, metrics, **kwargs):
        written.update(run=run, records=records, metrics=metrics, round_count=len(kwargs["rounds"]))

    monkeypatch.setattr(generate.runs, "write_method", fake_write_method)
    monkeypatch.setattr(generate.runs, "write_run", fake_write_run)
    monkeypatch.setattr(generate.runs, "new_run_id", lambda *args: "fake-run")

    assert generate.generate("plan_year", max_repair_rounds=4) == 0
    assert calls["count"] == 2
    method = written["method"]
    assert method["repair_rounds"] == 1
    assert method["best_round"] == 1
    assert method["input_spec"] == generate.InputSpec().to_dict()
    metrics = written["metrics"]
    assert metrics["primary"]["train"] == 1.0
    assert metrics["primary"]["train_nonexample"] == 1.0
    assert metrics["primary"]["test"] == 1.0
    assert metrics["primary"]["gap"] == 0.0
    assert metrics["failure_counts"]["per_round"] == [4, 0]
    assert metrics["rounds"][0]["failure_count"] == 4
    assert metrics["rounds"][0]["selected_failures"]
    assert len(written["records"]) == 6
    assert written["round_count"] == 2

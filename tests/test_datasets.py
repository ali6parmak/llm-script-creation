from llm_script_creation.datasets import (
    plan_all_fields_gold,
    plan_field_gold,
    split_for,
    upr_state_gold,
)


def row(metadata: dict) -> dict:
    return {"doc_id": "abc", "metadata": metadata}


def test_split_is_deterministic_and_uses_the_database() -> None:
    assert split_for("plan", "abc") == split_for("plan", "abc")
    assert split_for("plan", "abc") in {"train", "dev", "test"}


def test_plan_label_gold() -> None:
    build = plan_field_gold("document_type", "label")
    assert build(row({"document_type": [{"value": "id", "label": "Resolution"}]}), None) == {
        "document_type": "Resolution"
    }
    assert build(row({"document_type": [{"value": ""}]}), None) is None


def test_plan_value_gold() -> None:
    build = plan_field_gold("year", "value")
    assert build(row({"year": [{"value": 2012}]}), None) == {"year": 2012}
    assert build(row({"year": []}), None) is None


def test_plan_all_fields_requires_every_field() -> None:
    complete = {
        "document_type": [{"value": "id", "label": "Resolution"}],
        "legal_status": [{"value": "id", "label": "Legally binding"}],
        "means_of_adoption": [{"value": "id", "label": "Vote"}],
        "year": [{"value": 2012}],
        "document_code": [{"value": "S/RES/2068"}],
    }
    assert plan_all_fields_gold(row(complete), None) == {
        "document_type": "Resolution",
        "legal_status": "Legally binding",
        "means_of_adoption": "Vote",
        "year": 2012,
        "document_code": "S/RES/2068",
    }
    incomplete = dict(complete)
    incomplete.pop("document_code")
    assert plan_all_fields_gold(row(incomplete), None) is None


def test_upr_state_gold() -> None:
    metadata = {"state_reviewed": [{"value": "id", "label": "Barbados"}]}
    assert upr_state_gold(row(metadata), None) == {"state_under_review": "Barbados"}
    assert upr_state_gold(row({}), None) is None

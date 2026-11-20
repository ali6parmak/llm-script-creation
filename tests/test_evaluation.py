from llm_script_creation.evaluation import (
    evaluate_plan_sample,
    evaluate_upr_sample,
    evaluate_upr_state_sample,
)
from llm_script_creation.normalization import (
    canonical_state,
    normalize_code,
    parse_number,
    strip_trailing_countries,
    token_f1,
)


def test_normalize_code() -> None:
    assert normalize_code("A/HRC/RES/14/1") == "AHRCRES141"
    assert normalize_code(" s/res/2068 ") == "SRES2068"


def test_parse_number() -> None:
    assert parse_number("24.112.1") == ("24.112.1", "24.112", "1")
    assert parse_number("96.1") == ("96.1", "96.1", "")
    assert parse_number("not a number") is None


def test_strip_trailing_countries_handles_several_parentheses() -> None:
    clean, countries = strip_trailing_countries("Ratify the Convention (Iceland) (New Zealand);")
    assert clean == "Ratify the Convention"
    assert countries == ["Iceland", "New Zealand"]


def test_token_f1() -> None:
    assert token_f1("Ratify the Convention", "ratify the convention") == 1.0
    assert token_f1("a b c", "x y z") == 0.0


def test_canonical_state_aliases() -> None:
    assert canonical_state("Cabo Verde") == "Cape Verde"
    assert canonical_state("Republic of the Congo") == "Republic of Congo"
    assert canonical_state("Democratic People's Republic of Korea") == "DPR Korea"


def test_plan_sample_hit_and_wrong() -> None:
    gold = {"document_type": "Resolution"}
    hit = evaluate_plan_sample(gold, {"value": "resolution"}, ["document_type"])
    assert hit["status"] == "hit"
    wrong = evaluate_plan_sample(gold, {"value": "Treaty"}, ["document_type"])
    assert wrong["status"] == "wrong"


def test_upr_state_alias_hit() -> None:
    result = evaluate_upr_state_sample(
        {"state_under_review": "Cape Verde"},
        {"state_under_review": "Cabo Verde"},
    )
    assert result["status"] == "hit"


def test_upr_recommendation_matching() -> None:
    gold = {
        "state_under_review": "Barbados",
        "recommendations": [
            {
                "number": "96.1",
                "text": "Take steps towards abolition of the death penalty",
                "countries": ["New Zealand"],
            },
            {
                "number": "96.2",
                "text": "Ratify the Second Optional Protocol",
                "countries": ["Ukraine"],
            },
        ],
    }
    prediction = {
        "state_under_review": "Barbados",
        "recommendations": [
            {
                "number": "96.1",
                "text": "Take steps towards abolition of the death penalty",
                "countries": ["New Zealand"],
            }
        ],
    }
    result = evaluate_upr_sample(gold, prediction)
    assert result["metrics"]["matched"] == 1
    assert result["metrics"]["recall"] == 0.5
    assert result["metrics"]["country_correct"] == 1
    assert result["status"] == "partial"

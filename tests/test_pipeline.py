import pytest

from llm_script_creation.generate import _output_issues, _parse_code
from llm_script_creation.sandbox import validate_script


def test_parse_code_prefers_fenced_block() -> None:
    response = "Here is the script:\n```python\ndef extract(d):\n    return {}\n```\nDone."
    assert _parse_code(response) == "def extract(d):\n    return {}"


def test_parse_code_falls_back_to_raw_response() -> None:
    assert _parse_code("def extract(d):\n    return {}\n") == "def extract(d):\n    return {}"


def test_validate_script_accepts_stdlib_script() -> None:
    source = (
        "import re\n"
        "from typing import Any\n"
        "def extract(document: dict) -> dict:\n"
        '    return {"value": re.search(r"\\d{4}", document["text"]).group(0)}\n'
    )
    assert validate_script(source) == []


def test_validate_script_blocks_io_and_bad_imports() -> None:
    assert any(
        "open" in issue
        for issue in validate_script("def extract(d):\n    return open('x').read()\n")
    )
    assert any(
        "import" in issue
        for issue in validate_script("import os\ndef extract(d):\n    return {}\n")
    )
    assert any(
        "missing top-level function" in issue
        for issue in validate_script("def not_extract(d):\n    return {}\n")
    )


def test_output_shape_validation() -> None:
    assert _output_issues("plan_year", {"value": 2019}, ["year"]) == []
    assert _output_issues("plan_year", 2019, ["year"]) == ["output is int, expected dict"]
    assert _output_issues("upr_state", {"state_under_review": "Chad"}, ["state_under_review"]) == []
    assert _output_issues("upr_recommendations", {"recommendations": []}, []) == []


@pytest.mark.parametrize("dataset", ["plan_all_fields", "upr_recommendations"])
def test_output_shape_validation_multi(dataset: str) -> None:
    if dataset == "plan_all_fields":
        assert _output_issues(
            dataset,
            {"document_type": "Resolution"},
            [
                "document_type",
                "legal_status",
                "means_of_adoption",
                "year",
                "document_code",
            ],
        )
    else:
        assert _output_issues(dataset, {"state_under_review": "x"}, [])

import json

import pytest

from llm_script_creation import generate, prompts
from llm_script_creation.input_spec import (
    InputSpec,
    payload_excerpt,
    prepare_input,
)
from llm_script_creation.models import Document, Segment


def segment(text: str, *, page: int = 1, top: float = 0, type: str = "Text") -> Segment:
    return Segment(
        text=text,
        page=page,
        top=top,
        left=0,
        width=100,
        height=10,
        type=type,
        page_width=595,
        page_height=842,
    )


def document(segments: list[Segment], *, title: str = "A document") -> Document:
    return Document(
        entity_id="entity",
        shared_id="shared-id",
        title=title,
        language="en",
        template="template",
        metadata={},
        total_pages=max((item.page for item in segments), default=0),
        segments=segments,
    )


def test_text_input_head_and_tail_windows() -> None:
    doc = document([segment("alpha"), segment("beta"), segment("gamma")])
    assert prepare_input(doc, InputSpec(head_chars=7))["text"] == "alpha\nb"
    assert prepare_input(doc, InputSpec(tail_chars=5))["text"] == "gamma"
    assert prepare_input(doc, InputSpec())["text"] == "alpha\nbeta\ngamma"


def test_segment_input_windows_and_flags() -> None:
    doc = document(
        [
            segment("first", page=1, top=10),
            segment("second", page=1, top=20),
            segment("third", page=2, top=30, type="Title"),
        ]
    )
    head = prepare_input(doc, InputSpec(representation="segments", head_segments=2))
    assert [item["text"] for item in head["segments"]] == ["first", "second"]
    assert "type" not in head["segments"][0]
    assert "page" not in head["segments"][0]

    tail = prepare_input(
        doc,
        InputSpec(
            representation="segments",
            tail_segments=1,
            include_type=True,
            include_position=True,
        ),
    )
    assert tail["segments"] == [
        {
            "text": "third",
            "type": "Title",
            "page": 2,
            "top": 30.0,
            "left": 0.0,
            "width": 100.0,
            "height": 10.0,
        }
    ]


def test_input_spec_validation_and_roundtrip() -> None:
    with pytest.raises(ValueError, match="at most one"):
        InputSpec(head_chars=10, tail_chars=10)
    with pytest.raises(ValueError, match="segment window"):
        InputSpec(head_segments=10)
    spec = InputSpec(
        representation="segments",
        tail_segments=15,
        include_position=True,
        include_type=True,
    )
    assert InputSpec.from_dict(spec.to_dict()) == spec
    assert spec.slug() == "segments_tail15seg_type_pos"


def test_example_payload_equals_execution_payload() -> None:
    doc = document([segment("hello"), segment("world")])
    spec = InputSpec(head_chars=8)
    example_payload = prompts.example_payload(doc, spec)
    execution_payload = generate.execution_payload(doc, spec)
    assert example_payload == execution_payload == prepare_input(doc, spec)
    assert example_payload["text"] == "hello\nwo"


def test_prompt_example_uses_the_transformed_input_only() -> None:
    doc = document([segment("INSIDE-" + "x" * 60), segment("OUTSIDE-MARKER-" + "y" * 60)])
    spec = InputSpec(head_chars=50)
    example = {
        "doc_id": "shared-id",
        "gold": {"value": "x"},
        "document": doc,
    }
    messages = prompts.build_messages(
        {
            "dataset_id": "plan_document_code",
            "manifest": {"name": "toy", "description": "toy"},
            "target_schema": {"document_code": {"type": "string"}},
        },
        [example],
        spec,
    )
    prompt = "\n\n".join(message["content"] for message in messages)
    transformed = prepare_input(doc, spec)["text"]
    assert transformed in prompt
    assert "OUTSIDE-MARKER" not in prompt
    assert "Excerpt of the transformed input" not in prompt
    assert "first 50 characters" in prompt


def test_prompt_excerpt_is_centered_on_focus_inside_transformed_input() -> None:
    prefix = "A" * 5000
    payload = {"text": prefix + " recommendations will be examined here " + "B" * 5000}
    excerpt, truncated = payload_excerpt(
        payload,
        InputSpec(),
        char_limit=500,
        focuses=("recommendations will be examined",),
    )
    assert truncated is True
    assert len(excerpt) == 500
    assert "recommendations will be examined" in excerpt


def test_segment_prompt_excerpt_is_json_of_transformed_segments() -> None:
    segments = [
        {"text": "number 96.1 starts here"},
        *[{"text": f"filler {index}"} for index in range(25)],
    ]
    payload = {"segments": segments}
    excerpt, truncated = payload_excerpt(
        payload,
        InputSpec(representation="segments"),
        segment_limit=5,
        focuses=("96.1",),
    )
    assert truncated is True
    shown = json.loads(excerpt)
    assert len(shown) == 5
    assert shown[0]["text"] == "number 96.1 starts here"

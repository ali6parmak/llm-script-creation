from llm_script_creation.inspect_data import (
    recommendation_base,
    recommendation_numbers_in_text,
)
from llm_script_creation.models import Document, Segment, metadata_labels, metadata_values
from llm_script_creation.text import (
    document_payload,
    ordered_segments,
    render_segments_tsv,
    render_text,
)


def segment(
    text: str,
    *,
    page: int = 1,
    top: float = 0,
    left: float = 0,
    width: float = 100,
    height: float = 10,
    type: str = "Text",
) -> Segment:
    return Segment(
        text=text,
        page=page,
        top=top,
        left=left,
        width=width,
        height=height,
        type=type,
        page_width=595,
        page_height=842,
    )


def document(segments: list[Segment], title: str = "A document") -> Document:
    return Document(
        entity_id="entity",
        shared_id="shared",
        title=title,
        language="en",
        template="template",
        metadata={},
        total_pages=max((s.page for s in segments), default=0),
        segments=segments,
    )


def test_ordered_segments_sorts_by_layout_and_drops_blank() -> None:
    doc = document(
        [
            segment("second", top=200),
            segment("  "),
            segment("first", top=100),
            segment("next page", page=2, top=50),
        ]
    )
    assert [s.text for s in ordered_segments(doc)] == ["first", "second", "next page"]


def test_ordered_segments_can_skip_page_noise() -> None:
    doc = document(
        [
            segment("header", type="Page header"),
            segment("body"),
            segment("footer", type="Page footer"),
        ]
    )
    assert [s.text for s in ordered_segments(doc, skip_noise=True)] == ["body"]


def test_render_text_joins_segments_in_reading_order() -> None:
    doc = document([segment("b", top=20), segment("a", top=10), segment("c", page=2, top=5)])
    assert render_text(doc) == "a\nb\nc"


def test_render_text_can_add_page_markers() -> None:
    doc = document([segment("a"), segment("b", page=2)])
    assert render_text(doc, page_markers=True) == "--- page 1 ---\na\n\n--- page 2 ---\nb"


def test_render_segments_tsv_has_header_and_compact_values() -> None:
    doc = document([segment("hello\tworld", page=3, top=10.4, left=20.6, width=100, height=12)])
    lines = render_segments_tsv(doc).splitlines()
    assert lines[0] == "page\ttop\tleft\twidth\theight\ttype\ttext"
    assert lines[1] == "3\t10\t21\t100\t12\tText\thello world"


def test_document_payload_omits_title_by_default() -> None:
    doc = document([segment("text")], title="Barbados - Third Cycle")
    payload = document_payload(doc)
    assert payload == {"id": "shared", "language": "en", "page_count": 1, "text": "text"}
    assert "title" not in payload


def test_document_payload_can_include_title_and_segments() -> None:
    doc = document([segment("text", top=12, left=34)], title="Title")
    payload = document_payload(doc, "segments-json", include_title=True, include_position=False)
    assert payload["title"] == "Title"
    assert payload["segments"] == [{"text": "text", "type": "Text"}]


def test_metadata_helpers_filter_blank_values() -> None:
    metadata = {
        "year": [{"value": 2019}],
        "document_code": [{"value": ""}, {"value": "A/HRC/RES/14/1"}],
        "body": [{"value": "id", "label": "United Nations General Assembly"}],
    }
    assert metadata_values(metadata, "year") == [2019]
    assert metadata_values(metadata, "document_code") == ["A/HRC/RES/14/1"]
    assert metadata_labels(metadata, "body") == ["United Nations General Assembly"]


def test_recommendation_numbers_can_be_found_mid_segment() -> None:
    text = (
        "105.92 Make legislation on abortion more lenient (France); "
        "105.93 Continue efforts to promote gender equality (Georgia);"
    )
    assert recommendation_numbers_in_text(text) == ["105.92", "105.93"]


def test_recommendation_base_ignores_sub_number() -> None:
    assert recommendation_base("24.112.1") == "24.112"
    assert recommendation_base("96.1") == "96.1"
    assert recommendation_base("not a number") is None


def test_recommendation_numbers_ignore_leading_zero_artifacts() -> None:
    # PDF extraction inserts spaces inside paragraph references: "1 02.80".
    assert recommendation_numbers_in_text("paragraphs 1 02.80 , 1 02.81 of the second cycle") == []


def test_segment_from_mongo_accepts_catalog_page_key() -> None:
    segment_item = Segment.from_mongo(
        {"text": "hello", "page": 3, "top": 10, "left": 20, "type": "Text"}
    )
    assert segment_item.page == 3
    assert segment_item.text == "hello"


def test_document_payload_distinguishes_tsv_and_json() -> None:
    doc = document([segment("hello", page=2, top=5, left=10)], title="Title")
    tsv = document_payload(doc, "segments-tsv")
    assert "segments_tsv" in tsv
    assert "segments" not in tsv
    assert tsv["segments_tsv"].startswith("page\ttop\tleft")
    as_json = document_payload(doc, "segments-json")
    assert isinstance(as_json["segments"], list)
    assert as_json["segments"][0]["page"] == 2

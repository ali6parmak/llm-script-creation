"""Pure tests for the deterministic authoring-sample selection (no Mongo/LLM)."""

from __future__ import annotations

import pytest

from llm_script_creation.admin_agent_adapter import prompt, sampling


def _segment(page: int, text: str) -> dict:
    return {"text": text, "page": page, "type": "Paragraph"}


def _doc(
    doc_id: str,
    *,
    first_page: str = "United Nations General Assembly Distr.: General",
    pages: int = 1,
    metadata: dict | None = None,
    gold: dict | None = None,
    file_id: str | None = None,
    filename: str | None = None,
    split: str = "train",
) -> dict:
    segments = [_segment(1, first_page)]
    for page in range(2, pages + 1):
        segments.append(_segment(page, f"page {page} body text for {doc_id}"))
    return {
        "doc_id": doc_id,
        "split": split,
        "gold": {"document_type": "Resolution"} if gold is None else gold,
        "metadata": metadata
        if metadata is not None
        else {
            "document_type": [{"value": "t", "label": "Resolution"}],
            "body": [{"value": "b", "label": "General Assembly"}],
            "legal_status": [{"value": "l", "label": "Negotiated soft law"}],
        },
        "segments": segments,
        "file_id": file_id,
        "filename": filename,
    }


def _docs(count: int, *, prefix: str = "doc") -> list[dict]:
    return [_doc(f"{prefix}{index:02d}", first_page=f"A/RES/70/{index} resolution text") for index in range(count)]


def test_agent_policy_keeps_all_train_documents() -> None:
    docs = _docs(20)

    result = sampling.sample_documents("plan_document_type", docs)

    assert result.policy == "agent"
    assert list(result.doc_ids) == [doc["doc_id"] for doc in docs]
    assert len(result.doc_ids) == 20
    assert result.prompt_note() is None
    assert sampling.normalize_policy("agent_all_train") == "agent"


def test_agent_all_train_alias_keeps_all_documents() -> None:
    docs = _docs(8)

    result = sampling.sample_documents("plan_document_type", docs, policy="agent_all_train")

    assert result.policy == "agent"
    assert list(result.doc_ids) == [doc["doc_id"] for doc in docs]


def test_restricted_selection_is_deterministic_and_order_independent() -> None:
    docs = _docs(16)
    for index, doc in enumerate(docs):
        doc["segments"][0]["text"] = f"{doc['segments'][0]['text']} unique{index}"

    first = sampling.sample_documents("plan_document_type", docs, policy="hybrid")
    second = sampling.sample_documents("plan_document_type", docs, policy="hybrid")
    reversed_result = sampling.sample_documents(
        "plan_document_type", list(reversed(docs)), policy="hybrid"
    )

    assert first.doc_ids == second.doc_ids
    assert first.doc_ids == reversed_result.doc_ids
    assert first.digest == second.digest == reversed_result.digest


def test_restricted_selection_respects_size_and_small_universe() -> None:
    docs = _docs(30)

    for policy in ("coverage", "similarity", "hybrid"):
        result = sampling.sample_documents("plan_document_type", docs, policy=policy)
        assert len(result.doc_ids) == sampling.DEFAULT_SIZE

    smaller = sampling.sample_documents("plan_document_type", docs, policy="hybrid", size=5)
    assert len(smaller.doc_ids) == 5

    tiny = sampling.sample_documents("plan_document_type", docs[:3], policy="coverage")
    assert len(tiny.doc_ids) == 3

    with pytest.raises(ValueError):
        sampling.sample_documents("plan_document_type", docs, policy="unknown")


def test_sampling_rejects_test_documents() -> None:
    docs = _docs(4) + [_doc("leak", split="test")]

    with pytest.raises(ValueError, match="train-only"):
        sampling.sample_documents("plan_document_type", docs, policy="coverage")


def test_restrict_documents_keeps_stored_order_and_rejects_unknown_ids() -> None:
    docs = _docs(6)
    stored = ["doc04", "doc01", "doc05"]

    result = sampling.restrict_documents("plan_document_type", docs, stored, policy="replay:agent")

    assert list(result.doc_ids) == stored
    assert result.policy == "replay:agent"

    with pytest.raises(ValueError, match="not in this train split"):
        sampling.restrict_documents("plan_document_type", docs, ["missing"])


def test_coverage_selects_rare_metadata_strata() -> None:
    docs = _docs(19)
    rare = _doc(
        "rare",
        first_page="Distr.: General No. 42 regional treaty text",
        pages=8,
        metadata={
            "document_type": [{"value": "t", "label": "Regional treaty"}],
            "body": [{"value": "b", "label": "Organization of American States"}],
            "legal_status": [{"value": "l", "label": "Legally binding"}],
        },
    )
    docs.append(rare)

    for policy in ("coverage", "hybrid"):
        result = sampling.sample_documents("plan_document_type", docs, policy=policy)
        assert "rare" in result.doc_ids
        assert "Regional treaty" in result.strata["metadata"]["document_type"]
        assert "Legally binding" in result.strata["metadata"]["legal_status"]


def test_hybrid_forces_a_negative_and_one_shared_pdf() -> None:
    docs = _docs(17)
    negative = _doc(
        "negative",
        first_page="Report of the special rapporteur without document type",
        metadata={
            "document_type": [{"value": "t", "label": "Special Procedures' report"}],
            "body": [{"value": "b", "label": "Independent Expert"}],
            "legal_status": [{"value": "l", "label": "Non-negotiated soft law"}],
        },
        gold={},
    )
    shared_text = "A/RES/70/999 Conclusions adopted by the Executive Committee"
    twin_a = _doc("twin-a", first_page=shared_text, pages=9, file_id="same-file", filename="a.pdf")
    twin_b = _doc("twin-b", first_page=shared_text, pages=9, file_id="same-file", filename="a.pdf")
    docs.extend([negative, twin_a, twin_b])

    result = sampling.sample_documents("plan_document_type", docs, policy="hybrid")

    assert "negative" in result.negatives
    assert "negative" in result.doc_ids
    assert "shared_pdf" in result.hard_cases
    assert len({"twin-a", "twin-b"} & set(result.doc_ids)) == 1
    assert len(result.doc_ids) == sampling.DEFAULT_SIZE


def test_similarity_prefers_structurally_distinct_headers() -> None:
    docs = []
    for index in range(6):
        docs.append(
            _doc(
                f"res{index}",
                first_page=f"A/RES/70/{index} resolution adopted on housing item {index}",
            )
        )
    for index in range(6):
        docs.append(
            _doc(
                f"hrc{index}",
                first_page=f"A/HRC/45/{index} report of the working group review {index}",
            )
        )
    distinct = _doc(
        "distinct",
        first_page="Distr.: General No. 42 special report of the secretary general",
        pages=60,
    )
    docs.append(distinct)

    result = sampling.sample_documents("plan_document_type", docs, policy="similarity", size=3)

    assert "distinct" in result.doc_ids
    assert any(doc_id.startswith("res") for doc_id in result.doc_ids)
    assert any(doc_id.startswith("hrc") for doc_id in result.doc_ids)


def test_result_spec_records_policy_ids_and_is_train_only() -> None:
    docs = _docs(20)

    result = sampling.sample_documents("plan_document_type", docs, policy="hybrid", size=4)
    spec = result.to_spec()

    assert spec["policy"] == "hybrid"
    assert spec["selected_shared_ids"] == list(result.doc_ids)
    assert spec["authoring_documents"] == 4
    assert spec["universe_documents"] == 20
    assert set(spec["selected_shared_ids"]) <= {doc["doc_id"] for doc in docs}
    assert any(note.startswith("policy") for note in spec["notes"])


def test_prompt_sample_note_is_optional_and_keeps_the_contract() -> None:
    note = "SAMPLE NOTE: these 12 documents cover layouts; sample broadly."
    with_note = prompt.build_author_task(
        dataset_id="plan_document_type",
        property_name="document_type",
        property_type="select",
        template_name="Document",
        train_total=12,
        allowed_values=["Resolution"],
        sample_note=note,
    )
    without_note = prompt.build_author_task(
        dataset_id="plan_document_type",
        property_name="document_type",
        property_type="select",
        template_name="Document",
        train_total=12,
        allowed_values=["Resolution"],
    )

    assert note in with_note
    assert "SAMPLE NOTE" not in without_note
    assert '`"\\n".join(p["text"] for p in seg["paragraphs"])`' in with_note
    assert "peek_segmentation_by_entity" in with_note
    assert "test" not in with_note.lower()

    recommendations = prompt.build_recommendations_task(
        dataset_id="upr_recommendations",
        template_name="Document",
        train_total=12,
        state_labels=["France"],
        sample_note=note,
    )
    assert note in recommendations
    assert '`"\\n".join(p["text"] for p in seg["paragraphs"])`' in recommendations


def _typed_docs() -> list[dict]:
    docs = [
        _doc(
            f"res{index}",
            first_page=f"A/RES/70/{index} resolution text",
            gold={"document_type": "Resolution"},
        )
        for index in range(30)
    ]
    docs += [
        _doc(
            f"sp{index}",
            first_page=f"A/HRC/45/{index} special procedures report text",
            gold={"document_type": "Special Procedures' report"},
        )
        for index in range(10)
    ]
    docs += [
        _doc(
            f"ex{index}",
            first_page=f"E/CN.4/{index} ExCom conclusion text",
            gold={"document_type": "ExCom Conclusion"},
        )
        for index in range(4)
    ]
    docs += [
        _doc(
            f"reg{index}",
            first_page=f"OAS/Ser.L/{index} regional treaty text",
            gold={"document_type": "Regional treaty"},
        )
        for index in range(2)
    ]
    docs.append(
        _doc(
            "hard-compilation",
            first_page="Conclusions adopted by the Executive Committee",
            pages=45,
            gold={"document_type": "Regional treaty"},
        )
    )
    return docs


def test_balanced_selects_six_typical_and_six_rare_hard() -> None:
    docs = _typed_docs()

    result = sampling.sample_documents("plan_document_type", docs, policy="balanced")

    assert len(result.doc_ids) == sampling.DEFAULT_SIZE == 12
    selection = result.selection
    assert selection is not None
    assert selection["order"] == "common_first"
    assert len(selection["common_ids"]) == 6
    assert len(selection["rare_ids"]) == 6
    assert list(result.doc_ids) == selection["common_ids"] + selection["rare_ids"]
    # The typical half preserves the corpus prior: Resolution dominates.
    assert selection["common_target_strata"]["Resolution"] >= 3
    # The rare half includes a genuinely rare class and a hard structural case.
    rare_golds = {doc["gold"]["document_type"] for doc in docs if doc["doc_id"] in selection["rare_ids"]}
    assert rare_golds & {"ExCom Conclusion", "Regional treaty"}
    assert "hard-compilation" in selection["hard_ids"]


def test_balanced_rare_first_flips_only_the_order() -> None:
    docs = _typed_docs()

    common = sampling.sample_documents("plan_document_type", docs, policy="balanced")
    rare_first = sampling.sample_documents(
        "plan_document_type", docs, policy="balanced_rare_first"
    )

    assert common.selection["common_ids"] == rare_first.selection["common_ids"]
    assert common.selection["rare_ids"] == rare_first.selection["rare_ids"]
    assert common.selection["hard_ids"] == rare_first.selection["hard_ids"]
    assert rare_first.selection["order"] == "rare_first"
    assert list(rare_first.doc_ids) == (
        rare_first.selection["rare_ids"] + rare_first.selection["common_ids"]
    )
    assert rare_first.doc_ids[0] in rare_first.selection["rare_ids"]


def test_balanced_is_deterministic_and_order_independent() -> None:
    docs = _typed_docs()
    first = sampling.sample_documents("plan_document_type", docs, policy="balanced")
    second = sampling.sample_documents("plan_document_type", docs, policy="balanced")
    reversed_result = sampling.sample_documents(
        "plan_document_type", list(reversed(docs)), policy="balanced"
    )

    assert first.doc_ids == second.doc_ids == reversed_result.doc_ids
    assert first.selection == second.selection == reversed_result.selection


def test_balanced_respects_size_and_dedupes_shared_pdfs() -> None:
    docs = _typed_docs() + [
        _doc(
            "twin-a",
            first_page="Shared regional treaty body",
            gold={"document_type": "Regional treaty"},
            file_id="dup",
            filename="a.pdf",
        ),
        _doc(
            "twin-b",
            first_page="Shared regional treaty body",
            gold={"document_type": "Regional treaty"},
            file_id="dup",
            filename="a.pdf",
        ),
    ]

    sized = sampling.sample_documents("plan_document_type", docs, policy="balanced", size=4)
    full = sampling.sample_documents("plan_document_type", docs, policy="balanced")

    assert len(sized.doc_ids) == 4
    assert len(sized.selection["common_ids"]) == 2
    assert len(sized.selection["rare_ids"]) == 2
    assert len({"twin-a", "twin-b"} & set(full.doc_ids)) <= 1
    assert full.to_spec()["selection"]["order"] == "common_first"


def test_balanced_prompt_note_describes_the_order_without_gold() -> None:
    docs = _typed_docs()
    common = sampling.sample_documents("plan_document_type", docs, policy="balanced")
    rare_first = sampling.sample_documents(
        "plan_document_type", docs, policy="balanced_rare_first"
    )

    common_note = common.prompt_note() or ""
    rare_note = rare_first.prompt_note() or ""
    assert "typical half" in common_note and "typical documents come first" in common_note
    assert "rare/hard documents come first" in rare_note
    assert "Resolution" not in common_note

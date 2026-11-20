from llm_script_creation.datasets import (
    SHARED_TRAIN_GROUPS,
    TRAIN_SIZE,
    assign_splits,
    dataset_stratum,
    plan_all_fields_gold,
    plan_field_gold,
    select_stratified,
    select_train_ids,
    upr_state_gold,
)


def row(metadata: dict) -> dict:
    return {"doc_id": "abc", "metadata": metadata}


def sample(doc_id: str, gold: dict) -> dict:
    return {"doc_id": doc_id, "gold": gold}


def stratified(samples: list[dict], dataset_id: str) -> list[str]:
    return select_stratified(
        samples,
        dataset_key=dataset_id,
        stratum_key=lambda item: dataset_stratum(dataset_id, item["gold"]),
    )


def test_v2_split_policy_has_no_dev() -> None:
    assert TRAIN_SIZE == 50
    assert "dev" not in SHARED_TRAIN_GROUPS
    labels = {
        dataset_stratum("plan_document_type", {"document_type": "Resolution"}),
        dataset_stratum("plan_year", {"year": 1998}),
    }
    assert labels == {"Resolution", "1990s"}


def test_stratified_selection_is_deterministic_and_covers_labels() -> None:
    samples = [
        sample(f"doc-{index:03d}", {"document_type": f"label-{index % 5}"}) for index in range(200)
    ]
    first = stratified(samples, "plan_document_type")
    assert first == stratified(list(reversed(samples)), "plan_document_type")
    assert len(first) == TRAIN_SIZE
    labels = {
        item["gold"]["document_type"] for item in samples if item["doc_id"] in set(first)
    }
    assert labels == {f"label-{index}" for index in range(5)}


def test_stratified_selection_includes_rare_strata() -> None:
    samples = [sample(f"common-{index}", {"label": "common"}) for index in range(100)]
    samples.append(sample("rare-1", {"label": "rare"}))
    chosen = select_stratified(
        samples,
        dataset_key="toy",
        stratum_key=lambda item: item["gold"]["label"],
        train_size=10,
    )
    assert "rare-1" in chosen
    assert len(chosen) == 10


def test_upr_tasks_share_one_50_document_pool() -> None:
    state = [
        sample(f"upr-{index:03d}", {"state_under_review": f"State {index}"})
        for index in range(80)
    ]
    recommendations = [
        sample(
            f"upr-{index:03d}",
            {
                "recommendations": [
                    {"number": f"96.{part}", "text": "x", "countries": []}
                    for part in range(index % 4 + 1)
                ]
            },
        )
        for index in range(75)
    ]
    train_ids = select_train_ids(
        {"upr_state": state, "upr_recommendations": recommendations}
    )
    assert train_ids["upr_state"] == train_ids["upr_recommendations"]
    assert len(train_ids["upr_state"]) == TRAIN_SIZE
    assert set(train_ids["upr_state"]) <= {item["doc_id"] for item in recommendations}


def test_cross_dataset_train_documents_are_excluded_from_other_tests() -> None:
    samples_by_dataset = {
        "plan_year": [
            sample("shared", {"year": 2019}),
            sample("year-only", {"year": 2018}),
            sample("test-only", {"year": 2017}),
        ],
        "plan_document_type": [
            sample("shared", {"document_type": "Resolution"}),
            sample("type-only", {"document_type": "Treaty"}),
            sample("test-only", {"document_type": "Resolution"}),
        ],
    }
    train_ids = {"plan_year": ["shared"], "plan_document_type": ["type-only"]}
    assigned, excluded = assign_splits(
        samples_by_dataset,
        train_ids,
        {"plan_year": "plan", "plan_document_type": "plan"},
    )
    global_train = {"shared", "type-only"}
    for dataset_id, rows in assigned.items():
        for item in rows:
            assert item["split"] in {"train", "test"}
            if item["split"] == "test":
                assert item["doc_id"] not in global_train, dataset_id
    assert excluded["plan_document_type"] == 1
    assert {row["doc_id"] for row in assigned["plan_document_type"]} == {
        "type-only",
        "test-only",
    }


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

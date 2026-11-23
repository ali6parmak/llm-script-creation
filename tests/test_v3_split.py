"""Tests for the additive v3 (200-train) split and base-task mapping."""

from __future__ import annotations

import json

from llm_script_creation import baselines, datasets
from llm_script_creation.admin_agent_adapter import cli, evaluate, repair, sampling
from llm_script_creation.task_ids import (
    V2,
    V3,
    base_dataset_id,
    dataset_version,
    versioned_dataset_id,
)


def sample(doc_id: str, gold: dict) -> dict:
    return {"doc_id": doc_id, "gold": gold}


def plan_records(count: int, *, dataset: str = "plan_document_type") -> list[dict]:
    return [
        sample(f"doc-{index:04d}", {dataset.removeprefix("plan_"): f"label-{index % 5}"})
        for index in range(count)
    ]


def year_records(count: int) -> list[dict]:
    return [
        sample(f"year-{index:04d}", {"year": 1990 + (index % 40)})
        for index in range(count)
    ]


def upr_records() -> tuple[list[dict], list[dict]]:
    state = [
        sample(f"upr-{index:03d}", {"state_under_review": f"State {index}"})
        for index in range(165)
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
        for index in range(164)
    ]
    return state, recommendations


# ---------------------------------------------------------------------------
# ID/version mapping


def test_version_suffix_maps_to_the_same_base_task() -> None:
    assert dataset_version("plan_year") == V2
    assert dataset_version("plan_year_v3") == V3
    assert base_dataset_id("plan_year_v3") == "plan_year"
    assert base_dataset_id("upr_recommendations_v3") == "upr_recommendations"
    assert base_dataset_id("plan_year") == "plan_year"
    assert versioned_dataset_id("plan_year", V3) == "plan_year_v3"
    assert versioned_dataset_id("plan_year_v3", V2) == "plan_year"


def test_v3_dataset_specs_are_additive_and_versioned() -> None:
    v2_ids = {spec.dataset_id for spec in datasets.dataset_specs(V2)}
    v3_specs = datasets.dataset_specs(V3)
    v3_ids = {spec.dataset_id for spec in v3_specs}

    assert v2_ids.isdisjoint(v3_ids)
    assert v3_ids == {f"{dataset_id}_v3" for dataset_id in v2_ids}
    assert all(spec.dataset_id.endswith("_v3") for spec in v3_specs)
    assert datasets.SPLIT_POLICIES[V3].split_policy_version == "v3-200train"
    assert datasets.SPLIT_POLICIES[V3].train_size == 200
    assert datasets.SPLIT_POLICIES[V3].validation_size == 40


def test_base_task_mapping_reaches_adapter_logic() -> None:
    assert sampling._is_negative("plan_year_v3", {"year": 2019}) is False
    assert sampling._is_negative("upr_state_v3", {"state_under_review": "Albania"}) is False
    assert repair._stratum("plan_year_v3", {"year": 1998}) == "1990s"
    assert repair._stratum("upr_recommendations_v3", {"recommendations": []}) == "recs<150"
    assert cli._target_properties("upr_recommendations_v3", "recommendations") == {
        "related_recommendations",
        "state_reviewed",
    }
    assert baselines.for_dataset("plan_year_v3").dataset_id == "plan_year"
    assert baselines.for_dataset("upr_recommendations_v3").dataset_id == "upr_recommendations"


# ---------------------------------------------------------------------------
# v3 selection


def test_v3_selection_is_deterministic_and_200_train() -> None:
    samples = {
        "plan_document_type_v3": plan_records(260),
        "plan_year_v3": year_records(260),
    }
    first = datasets.select_train_ids(samples, version=V3)
    reversed_samples = {key: list(reversed(value)) for key, value in samples.items()}
    second = datasets.select_train_ids(reversed_samples, version=V3)

    assert first == second
    assert len(first["plan_document_type_v3"]) == 200
    assert len(first["plan_year_v3"]) == 200
    assert set(first["plan_document_type_v3"]) <= {
        item["doc_id"] for item in samples["plan_document_type_v3"]
    }


def test_v3_upr_pool_is_shared_and_capped_to_leave_a_test_split() -> None:
    state, recommendations = upr_records()
    samples = {
        "upr_state_v3": state,
        "upr_recommendations_v3": recommendations,
    }

    train_ids = datasets.select_train_ids(samples, version=V3)

    assert train_ids["upr_state_v3"] == train_ids["upr_recommendations_v3"]
    # 164 eligible docs, 40 kept out of train for the frozen test split.
    assert len(train_ids["upr_state_v3"]) == 124
    assert set(train_ids["upr_state_v3"]) <= {
        item["doc_id"] for item in recommendations
    }


def test_v3_cross_dataset_isolation_stays_inside_v3() -> None:
    samples_by_dataset = {
        "plan_year_v3": [
            sample("shared", {"year": 2019}),
            sample("v3-only", {"year": 2018}),
            sample("test-only", {"year": 2017}),
        ],
        "plan_document_type_v3": [
            sample("shared", {"document_type": "Resolution"}),
            sample("v3-only", {"document_type": "Treaty"}),
            sample("test-only", {"document_type": "Resolution"}),
        ],
    }
    train_ids = {
        "plan_year_v3": ["shared"],
        "plan_document_type_v3": ["v3-only"],
    }
    assigned, excluded = datasets.assign_splits(
        samples_by_dataset,
        train_ids,
        {"plan_year_v3": "plan", "plan_document_type_v3": "plan"},
    )

    global_v3_train = {"shared", "v3-only"}
    for rows in assigned.values():
        for row in rows:
            if row["split"] == "test":
                assert row["doc_id"] not in global_v3_train
    assert excluded["plan_document_type_v3"] == 1


def test_v2_and_v3_do_not_exclude_each_other() -> None:
    shared_rows = [
        sample("shared", {"year": 2019}),
        sample("v3-only", {"year": 2018}),
    ]
    samples_by_dataset = {
        "plan_year": list(shared_rows),
        "plan_year_v3": list(shared_rows),
    }
    assigned, _ = datasets.assign_splits(
        samples_by_dataset,
        {"plan_year": ["shared"], "plan_year_v3": ["v3-only"]},
        {"plan_year": "plan", "plan_year_v3": "plan"},
    )

    v2_test = {row["doc_id"] for row in assigned["plan_year"] if row["split"] == "test"}
    v3_test = {row["doc_id"] for row in assigned["plan_year_v3"] if row["split"] == "test"}
    assert "v3-only" in v2_test
    assert "shared" in v3_test


# ---------------------------------------------------------------------------
# 160/40 repair split + manifest


def test_v3_repair_split_is_160_feedback_40_validation() -> None:
    records = plan_records(200)
    first = repair.split_validation(
        "plan_document_type_v3", records, validation_size=40, max_fraction=None
    )
    second = repair.split_validation(
        "plan_document_type_v3", list(reversed(records)), validation_size=40, max_fraction=None
    )

    assert len(first.feedback) == 160
    assert len(first.validation) == 40
    assert first.validation_ids() == second.validation_ids()
    assert set(first.validation_ids()).isdisjoint(first.feedback_ids())


def test_v3_manifest_records_version_and_split_sizes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path)
    spec = datasets.dataset_specs(V3)[0]
    records = plan_records(260)
    train_ids = [record["doc_id"] for record in records[:200]]
    samples = [
        {**record, "split": "train" if index < 200 else "test"}
        for index, record in enumerate(records)
    ]

    manifest = datasets._write_dataset(
        spec,
        samples,
        train_ids,
        excluded=0,
        created_at="now",
        version=V3,
    )

    assert manifest["dataset_id"] == "plan_document_type_v3"
    assert manifest["dataset_version"] == V3
    assert manifest["split_policy_version"] == "v3-200train"
    assert manifest["counts"]["train"] == 200
    assert manifest["counts"]["validation"] == 40
    assert manifest["counts"]["train_feedback"] == 160
    assert manifest["counts"]["test"] == 60
    saved = json.loads((tmp_path / spec.dataset_id / "manifest.json").read_text())
    assert saved["dataset_version"] == V3


# ---------------------------------------------------------------------------
# Scoring


def test_score_records_accepts_versioned_dataset_ids() -> None:
    samples = [
        {"doc_id": "a", "split": "train", "gold": {"year": 2019}},
        {"doc_id": "b", "split": "test", "gold": {"year": 2020}},
    ]
    predictions = {
        "a": {"prediction": {"year": 2019}},
        "b": {"prediction": {"year": 2020}},
    }

    _, metrics = evaluate.score_records("plan_year_v3", ["year"], samples, predictions)

    assert metrics["primary"]["metric"] == "accuracy"
    assert metrics["primary"]["test"] == 1.0

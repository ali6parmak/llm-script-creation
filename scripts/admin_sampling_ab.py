#!/usr/bin/env python
"""Summarize the admin-agent authoring-sample A/B from artifacts only.

Reads ``artifacts/methods/*/spec.json``, the matching ``method.json`` /
``attempts/attempt-1.json`` and the latest ``train_test`` run under
``artifacts/runs``. No Mongo, no LLM, no sidecar.

Reports, per run and per (dataset, policy) arm:

* test/train score, test hit/wrong/missing counts;
* author requests/tokens;
* selected-sample strata and pairwise diversity (Jaccard over signatures);
* which train documents the extractor actually peeked at, and how much of the
  selected universe that covers.

Usage::

    uv run python scripts/admin_sampling_ab.py --since 2026-11-23T17:00:00+00:00
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_script_creation import catalog, runs
from llm_script_creation.admin_agent_adapter import sampling
from llm_script_creation.configuration import REPORTS_DIR

CONDITION = {"title_visible": False, "no_metadata": True}


# ---------------------------------------------------------------------------
# Artifact loading


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _spec(method_id: str) -> dict[str, Any] | None:
    path = runs.METHODS_DIR / method_id / "spec.json"
    return _read_json(path) if path.exists() else None


def _latest_run(method_id: str) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs.list_runs()
        if run.get("method_id") == method_id
        and run.get("split") == "train_test"
        and run.get("status") == "completed"
    ]
    # A replay run reuses the method but has no author usage: prefer the
    # original authoring run when both exist.
    original = [run for run in candidates if not (run.get("metrics") or {}).get("replay")]
    pool = original or candidates
    return sorted(pool, key=lambda run: run.get("run_id", ""))[-1] if pool else None


def _peeks(method_id: str) -> list[dict[str, str]]:
    path = runs.METHODS_DIR / method_id / "attempts" / "attempt-1.json"
    if not path.exists():
        return []
    attempt = _read_json(path)
    calls: list[dict[str, str]] = []
    for item in attempt.get("transcript") or []:
        if item.get("kind") != "tool_call":
            continue
        tool = str(item.get("tool") or "")
        if "peek" not in tool:
            continue
        args = item.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        shared_id = args.get("shared_id") or args.get("doc_id") or args.get("id")
        if shared_id:
            calls.append({"tool": tool, "doc_id": str(shared_id)})
    return calls


def _train_records(dataset_id: str) -> list[dict[str, Any]]:
    """Minimal catalog-backed records, used only to backfill old specs."""
    conn = catalog.connect()
    try:
        dataset = catalog.get_dataset(conn, dataset_id)
        if dataset is None:
            return []
        records = []
        for sample in catalog.iter_samples(conn, dataset_id, split="train"):
            document = catalog.get_document(conn, dataset["db"], sample["doc_id"])
            if document is None:
                continue
            records.append(
                {
                    "doc_id": document["doc_id"],
                    "split": sample["split"],
                    "gold": json.loads(sample["gold_json"] or "{}"),
                    "metadata": json.loads(document["metadata_json"] or "{}"),
                    "segments": json.loads(document["segments_json"] or "[]"),
                    "file_id": document["file_id"],
                    "filename": document["filename"],
                }
            )
        return records
    finally:
        conn.close()


def _selected_documents(spec: dict[str, Any]) -> list[dict[str, Any]]:
    documents = (spec.get("sampling") or {}).get("documents") or []
    if documents:
        return list(documents)
    dataset_id = str(spec.get("dataset_id"))
    selected = (spec.get("sampling") or {}).get("selected_shared_ids") or []
    try:
        records = _train_records(dataset_id)
        result = sampling.restrict_documents(dataset_id, records, selected, policy="backfill")
        return list(result.to_spec()["documents"])
    except (ValueError, OSError):
        return []


_GOLD_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def _train_gold(dataset_id: str) -> dict[str, dict[str, Any]]:
    """Selected-document gold, for descriptive target-class coverage only."""
    if dataset_id in _GOLD_CACHE:
        return _GOLD_CACHE[dataset_id]
    gold: dict[str, dict[str, Any]] = {}
    conn = catalog.connect()
    try:
        for sample in catalog.iter_samples(conn, dataset_id, split="train"):
            gold[str(sample["doc_id"])] = json.loads(sample["gold_json"] or "{}")
    except OSError:
        pass
    finally:
        conn.close()
    _GOLD_CACHE[dataset_id] = gold
    return gold


def _target_class(dataset_id: str, gold: dict[str, Any]) -> str:
    """A compact target stratum, only for describing what a sample covers."""
    if dataset_id == "upr_recommendations":
        count = len(gold.get("recommendations") or [])
        if count < 150:
            return "recs<150"
        if count < 200:
            return "recs150-199"
        if count < 250:
            return "recs200-249"
        if count < 300:
            return "recs250-299"
        return "recs300+"
    field = dataset_id.removeprefix("plan_")
    value = gold.get(field)
    if field == "year":
        try:
            decade = int(value) // 10 * 10
            return f"{decade}s"
        except (TypeError, ValueError):
            return "unknown"
    return str(value) if value is not None else "missing"


# ---------------------------------------------------------------------------
# Derived quantities


def _signature(document: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for dimension, value in (document.get("metadata_classes") or {}).items():
        if value and value != "unknown":
            tokens.add(f"{dimension}:{value}")
    tokens.update(f"header:{header}" for header in document.get("headers") or [])
    for key in ("page_bucket", "length_bucket", "paragraph_bucket"):
        if document.get(key):
            tokens.add(f"{key}:{document[key]}")
    tokens.update(f"hard:{flag}" for flag in document.get("hard_flags") or [])
    return tokens


def _pairwise_distance(documents: list[dict[str, Any]]) -> float | None:
    signatures = [_signature(document) for document in documents]
    distances = []
    for left, right in itertools.combinations(signatures, 2):
        union = left | right
        if union:
            distances.append(1.0 - len(left & right) / len(union))
    return statistics.fmean(distances) if distances else None


def _strata(documents: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        for dimension, value in (document.get("metadata_classes") or {}).items():
            if value and value != "unknown":
                metadata[dimension].add(value)
    return {
        "metadata": {key: sorted(values) for key, values in sorted(metadata.items())},
        "headers": sorted(
            {header for document in documents for header in document.get("headers") or []}
        ),
        "page_buckets": sorted(
            {document.get("page_bucket") for document in documents if document.get("page_bucket")}
        ),
        "hard_flags": sorted(
            {flag for document in documents for flag in document.get("hard_flags") or []}
        ),
    }


def _target_counts(
    dataset_id: str, doc_ids: list[str], gold: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for doc_id in set(doc_ids):
        counts[_target_class(dataset_id, gold.get(doc_id, {}))] += 1
    return dict(sorted(counts.items()))


def _observation(
    method: dict[str, Any], spec: dict[str, Any], run: dict[str, Any], peeks: list[dict[str, str]]
) -> dict[str, Any]:
    metrics = run.get("metrics") or {}
    primary = metrics.get("primary") or {}
    statuses = (metrics.get("status_counts") or {}).get("test") or {}
    admin_eval = metrics.get("admin_eval") or {}
    usage = metrics.get("token_usage") or {}
    selected = list((spec.get("sampling") or {}).get("selected_shared_ids") or [])
    documents = _selected_documents(spec)
    peeked = [call["doc_id"] for call in peeks]
    peeked_selected = [doc_id for doc_id in peeked if doc_id in set(selected)]
    gold = _train_gold(str(spec.get("dataset_id")))
    target_classes = _target_counts(str(spec.get("dataset_id")), selected, gold)
    peeked_target_classes = _target_counts(str(spec.get("dataset_id")), peeked, gold)
    first_two = selected[:2]
    documents_by_id = {document["doc_id"]: document for document in documents}
    peeked_documents = [documents_by_id[doc_id] for doc_id in set(peeked) if doc_id in documents_by_id]
    return {
        "method_id": method["method_id"],
        "short_id": f"{str(spec.get('spec_hash') or '')[:8]}@{str(spec.get('created_at') or '')[11:19]}",
        "dataset_id": spec.get("dataset_id"),
        "policy": (spec.get("sampling") or {}).get("policy"),
        "prompt_version": spec.get("prompt_version"),
        "created_at": spec.get("created_at"),
        "train": primary.get("train"),
        "test": primary.get("test"),
        "metric": primary.get("metric"),
        "hit": statuses.get("hit", 0),
        "wrong": statuses.get("wrong", 0),
        "missing": statuses.get("missing_prediction", 0),
        "error": statuses.get("error", 0),
        "admin_accuracy": admin_eval.get("accuracy"),
        "admin_coverage": admin_eval.get("coverage"),
        "requests": usage.get("requests"),
        "tokens": usage.get("total_tokens"),
        "selected": selected,
        "selected_documents": documents,
        "universe": (spec.get("sampling") or {}).get("universe_documents"),
        "peeks": peeked,
        "peeked": peeked,
        "peeked_distinct": len(set(peeked)),
        "peeked_selected": peeked_selected,
        "first_two_peeks": sum(1 for doc_id in peeked if doc_id in first_two),
        "pairwise_distance": _pairwise_distance(documents),
        "peeked_pairwise_distance": _pairwise_distance(peeked_documents),
        "strata": _strata(documents),
        "target_classes": target_classes,
        "peeked_target_classes": peeked_target_classes,
    }


# ---------------------------------------------------------------------------
# Report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _score_min(observations: list[dict[str, Any]]) -> float | None:
    values = [obs["test"] for obs in observations if obs["test"] is not None]
    return min(values) if values else None


def _score_max(observations: list[dict[str, Any]]) -> float | None:
    values = [obs["test"] for obs in observations if obs["test"] is not None]
    return max(values) if values else None


def build_report(observations: list[dict[str, Any]]) -> str:
    lines = ["# Admin-agent authoring-sample A/B", ""]
    lines.append(
        "Fixed condition: no title, `--no-metadata`, country list on for UPR. "
        "One method per authoring run; scores are single runs, so treat ranges as noise, not CI."
    )
    lines.append("")

    lines += [
        "## Runs",
        "",
        "| Dataset | Policy | Method | Test | Train | Hit | Wrong | Missing | Requests | Tokens | Peeked docs | Selected dist |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in sorted(observations, key=lambda item: (item["dataset_id"], item["policy"], item["created_at"])):
        lines.append(
            "| {dataset} | {policy} | `{method}` | {test} | {train} | {hit} | {wrong} | {missing} | "
            "{requests} | {tokens} | {peeked}/{selected} | {distance} |".format(
                dataset=obs["dataset_id"],
                policy=obs["policy"],
                method=obs["short_id"],
                test=_fmt(obs["test"]),
                train=_fmt(obs["train"]),
                hit=obs["hit"],
                wrong=obs["wrong"],
                missing=obs["missing"],
                requests=_fmt(obs["requests"]),
                tokens=_fmt(obs["tokens"]),
                peeked=obs["peeked_distinct"],
                selected=len(obs["selected"]),
                distance=_fmt(obs["pairwise_distance"]),
            )
        )
    lines.append("")

    lines += [
        "## Policy summary (per dataset)",
        "",
        "| Dataset | Policy | n | Test mean | min | max | Harms (wrong+missing) | Requests | Tokens | Peeked docs | Peeked pairwise dist |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        groups[(str(obs["dataset_id"]), str(obs["policy"]))].append(obs)
    for (dataset_id, policy), group in sorted(groups.items()):
        lines.append(
            "| {dataset} | {policy} | {n} | {mean} | {low} | {high} | {harms} | {requests} | "
            "{tokens} | {peeked} | {distance} |".format(
                dataset=dataset_id,
                policy=policy,
                n=len(group),
                mean=_fmt(_mean([obs["test"] for obs in group])),
                low=_fmt(_score_min(group)),
                high=_fmt(_score_max(group)),
                harms=_fmt(_mean([obs["wrong"] + obs["missing"] for obs in group])),
                requests=_fmt(_mean([obs["requests"] for obs in group])),
                tokens=_fmt(_mean([obs["tokens"] for obs in group])),
                peeked=_fmt(_mean([obs["peeked_distinct"] for obs in group]), 1),
                distance=_fmt(_mean([obs["peeked_pairwise_distance"] for obs in group])),
            )
        )
    lines.append("")

    lines += [
        "## Selected-sample structure",
        "",
        "Distinct values among the documents offered to the extractor (agent arm: all 50).",
        "",
    ]
    for obs in sorted(observations, key=lambda item: (item["dataset_id"], item["policy"], item["created_at"])):
        strata = obs["strata"]
        metadata = ", ".join(
            f"{key}={len(values)}" for key, values in strata["metadata"].items()
        ) or "—"
        lines.append(
            f"- `{obs['dataset_id']}` / {obs['policy']} / `{obs['short_id']}`: "
            f"{len(obs['selected'])} selected, metadata classes {metadata}; "
            f"headers {strata['headers']}; page buckets {strata['page_buckets']}; "
            f"hard cases {strata['hard_flags']}."
        )
    lines.append("")

    lines += [
        "## Target-label spread of the selected docs",
        "",
        "Descriptive only (the sampler does not use the target value; metadata/layout/hard "
        "cases drive it). For `plan_year` this shows whether a selection misses decades.",
        "",
        "| Dataset | Policy | Method | Selected target classes |",
        "|---|---|---|---|",
    ]
    for obs in sorted(observations, key=lambda item: (item["dataset_id"], item["policy"], item["created_at"])):
        lines.append(
            "| {dataset} | {policy} | `{method}` | {classes} |".format(
                dataset=obs["dataset_id"],
                policy=obs["policy"],
                method=obs["short_id"],
                classes=", ".join(f"{key}×{value}" for key, value in obs["target_classes"].items())
                or "—",
            )
        )
    lines.append("")

    lines += [
        "## What the extractor peeked at",
        "",
        "| Dataset | Policy | Method | Peek calls | Distinct docs | In selected universe | First-two docs | Peeked metadata classes | Peeked headers | Peeked target classes |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for obs in sorted(observations, key=lambda item: (item["dataset_id"], item["policy"], item["created_at"])):
        documents_by_id = {document["doc_id"]: document for document in obs["selected_documents"]}
        peeked_documents = [
            documents_by_id[doc_id] for doc_id in obs["peeked"] if doc_id in documents_by_id
        ]
        peeked_strata = _strata(peeked_documents)
        classes = sum(len(values) for values in peeked_strata["metadata"].values())
        lines.append(
            "| {dataset} | {policy} | `{method}` | {calls} | {distinct} | {inside}/{selected} | {first_two} | "
            "{classes} | {headers} | {targets} |".format(
                dataset=obs["dataset_id"],
                policy=obs["policy"],
                method=obs["short_id"],
                calls=len(obs["peeks"]),
                distinct=obs["peeked_distinct"],
                inside=len(set(obs["peeked_selected"])),
                selected=len(obs["selected"]),
                first_two=obs["first_two_peeks"],
                classes=classes,
                headers=", ".join(peeked_strata["headers"]) or "—",
                targets=", ".join(
                    f"{key}×{value}" for key, value in obs["peeked_target_classes"].items()
                )
                or "—",
            )
        )
    lines.append("")

    lines += ["## Arm deltas (descriptive only)", ""]
    for dataset_id in sorted({str(obs["dataset_id"]) for obs in observations}):
        policies = sorted(
            {str(obs["policy"]) for obs in observations if str(obs["dataset_id"]) == dataset_id}
        )
        if "agent" not in policies or "hybrid" not in policies:
            continue
        agent = [obs for obs in observations if obs["dataset_id"] == dataset_id and obs["policy"] == "agent"]
        hybrid = [obs for obs in observations if obs["dataset_id"] == dataset_id and obs["policy"] == "hybrid"]
        agent_test = _mean([obs["test"] for obs in agent])
        hybrid_test = _mean([obs["test"] for obs in hybrid])
        lines.append(
            f"- `{dataset_id}`: test agent {_fmt(agent_test)} vs hybrid {_fmt(hybrid_test)} "
            f"(Δ {_fmt(None if agent_test is None or hybrid_test is None else hybrid_test - agent_test)}); "
            f"requests {_fmt(_mean([obs['requests'] for obs in agent]))} vs "
            f"{_fmt(_mean([obs['requests'] for obs in hybrid]))}; "
            f"tokens {_fmt(_mean([obs['tokens'] for obs in agent]))} vs "
            f"{_fmt(_mean([obs['tokens'] for obs in hybrid]))}."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def _parse_since(raw: str | None) -> tuple[datetime | None, dict[str, datetime]]:
    global_since = None
    by_dataset: dict[str, datetime] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            dataset_id, timestamp = item.split("=", 1)
            by_dataset[dataset_id.strip()] = datetime.fromisoformat(timestamp.strip())
        else:
            global_since = datetime.fromisoformat(item)
    return global_since, by_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="plan_document_type,upr_recommendations,plan_year",
        help="comma-separated dataset ids",
    )
    parser.add_argument(
        "--policies",
        default="agent,hybrid,coverage,similarity",
        help="comma-separated sampling policies to include",
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "only specs created at/after this ISO timestamp; accepts either one value or "
            "comma-separated dataset=timestamp pairs "
            "(e.g. upr_recommendations=2026-11-23T13:56:36+00:00)"
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated explicit method ids (overrides the other filters)",
    )
    parser.add_argument(
        "--out",
        default=str(REPORTS_DIR / "admin_sampling_ab.md"),
        help="markdown output path",
    )
    args = parser.parse_args(argv)

    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    policies = {item.strip() for item in args.policies.split(",") if item.strip()}
    global_since, since_by_dataset = _parse_since(args.since)
    explicit = {item.strip() for item in (args.methods or "").split(",") if item.strip()}

    observations: list[dict[str, Any]] = []
    for method in runs.list_methods():
        method_id = str(method.get("method_id") or "")
        if explicit and method_id not in explicit:
            continue
        spec = _spec(method_id)
        if not spec:
            continue
        if not explicit:
            if spec.get("dataset_id") not in datasets:
                continue
            if (spec.get("sampling") or {}).get("policy") not in policies:
                continue
            condition = spec.get("condition") or {}
            if any(condition.get(key) != value for key, value in CONDITION.items()):
                continue
            created_at = spec.get("created_at")
            threshold = since_by_dataset.get(str(spec.get("dataset_id")), global_since)
            if threshold is not None and created_at and datetime.fromisoformat(created_at) < threshold:
                continue
        run = _latest_run(method_id)
        if run is None:
            continue
        observations.append(_observation(method, spec, run, _peeks(method_id)))

    if not observations:
        raise SystemExit("No matching method/run observations. Check --since/--datasets.")

    report = build_report(observations)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

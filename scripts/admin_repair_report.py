#!/usr/bin/env python
"""Summarize admin-agent repair runs from artifacts only (no Mongo/LLM).

Reads ``artifacts/methods/*/spec.json`` + ``attempts/attempt-*.json`` and the
latest completed run. Reports per-round feedback/validation scores, the best
round, final test metrics, and a leak check that no validation shared_id
appears in a repair prompt.

Usage::

    uv run python scripts/admin_repair_report.py --since 2026-11-23T14:45:00+00:00
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_script_creation import runs
from llm_script_creation.configuration import REPORTS_DIR


def _spec(method_id: str) -> dict[str, Any] | None:
    path = runs.METHODS_DIR / method_id / "spec.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _attempts(method_id: str) -> list[dict[str, Any]]:
    directory = runs.METHODS_DIR / method_id / "attempts"
    attempts = []
    for path in sorted(directory.glob("attempt-*.json")):
        attempts.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(attempts, key=lambda item: item.get("round", 0))


def _latest_run(method_id: str) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs.list_runs()
        if run.get("method_id") == method_id
        and run.get("split") == "train_test"
        and run.get("status") == "completed"
        and not (run.get("metrics") or {}).get("replay")
    ]
    return sorted(candidates, key=lambda run: run.get("run_id", ""))[-1] if candidates else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _baseline_rows(since: datetime | None, datasets: set[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for method in runs.list_methods():
        method_id = str(method.get("method_id") or "")
        spec = _spec(method_id)
        if not spec or (spec.get("repair") or {}).get("enabled"):
            continue
        dataset_id = str(spec.get("dataset_id"))
        if dataset_id not in datasets:
            continue
        condition = spec.get("condition") or {}
        if condition.get("title_visible") is not False or condition.get("no_metadata") is not True:
            continue
        if (spec.get("sampling") or {}).get("policy") != "agent":
            continue
        created_at = spec.get("created_at")
        if since is not None and created_at and datetime.fromisoformat(created_at) < since:
            continue
        run = _latest_run(method_id)
        if run is None:
            continue
        rows.setdefault(dataset_id, []).append(run["metrics"]["primary"])
    return rows


def build_report(
    repair_methods: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    baseline: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# Admin-agent repair loop",
        "",
        "Opt-in loop: 40-document feedback pool, frozen stratified validation slice "
        "(never sent to authoring), 2-3 failures per round, best round on validation, "
        "test executed once at the end.",
        "",
        "## Task matrix",
        "",
        "Control = same 40/10 holdout with `--repair-rounds 0`; repair = 3 rounds. "
        "Means over the runs shown (n is small; treat deltas as a first pass).",
        "",
        "| Dataset | Control test | Repair test | Δ mean | Best rounds |",
        "|---|---:|---:|---:|---|",
    ]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for spec, run, _outcome, _attempts in repair_methods:
        dataset_id = str(spec.get("dataset_id"))
        mode = "control" if not (spec.get("repair") or {}).get("rounds_requested") else "repair"
        groups.setdefault(dataset_id, {"control": [], "repair": []})[mode].append(run)
    for dataset_id, modes in sorted(groups.items()):
        control_values = [
            (run.get("metrics") or {}).get("primary", {}).get("test")
            for run in modes["control"]
        ]
        repair_values = [
            (run.get("metrics") or {}).get("primary", {}).get("test")
            for run in modes["repair"]
        ]
        control_mean = _mean(control_values)
        repair_mean = _mean(repair_values)
        delta = (
            None
            if control_mean is None or repair_mean is None
            else repair_mean - control_mean
        )
        best_rounds = [
            str((spec.get("repair_outcome") or {}).get("best_round"))
            for spec, run, outcome, _attempts in repair_methods
            if spec.get("dataset_id") == dataset_id
            and (spec.get("repair") or {}).get("rounds_requested")
        ]
        lines.append(
            f"| {dataset_id} | {_fmt(control_mean)} (n={len(control_values)}) | "
            f"{_fmt(repair_mean)} (n={len(repair_values)}) | {_fmt(delta)} | "
            f"{', '.join(best_rounds) or '—'} |"
        )
    lines += [
        "",
        "## Repair runs",
        "",
        "| Dataset | Mode | Method | Val docs | Rounds | Best | Stop | Test | Train | Hit/wrong/missing |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for spec, run, outcome, _attempts in repair_methods:
        metrics = run.get("metrics") or {}
        primary = metrics.get("primary") or {}
        statuses = (metrics.get("status_counts") or {}).get("test") or {}
        lines.append(
            "| {dataset} | {mode} | `{method}` | {val} | {rounds} | {best} | {stop} | {test} | {train} | "
            "{hit}/{wrong}/{missing} |".format(
                dataset=spec.get("dataset_id"),
                mode="control" if not (spec.get("repair") or {}).get("rounds_requested") else "repair",
                method=f"{str(spec.get('spec_hash') or '')[:8]}@{str(spec.get('created_at') or '')[11:19]}",
                val=(spec.get("repair") or {}).get("validation_size"),
                rounds=outcome.get("rounds_run"),
                best=outcome.get("best_round"),
                stop=outcome.get("stop_reason"),
                test=_fmt(primary.get("test")),
                train=_fmt(primary.get("train")),
                hit=statuses.get("hit", 0),
                wrong=statuses.get("wrong", 0),
                missing=statuses.get("missing_prediction", 0),
            )
        )
    lines.append("")

    lines += [
        "## Rounds",
        "",
        "| Dataset | Method | Round | Feedback failures fed | Train (feedback) | Validation | Requests | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec, _run, _outcome, attempts in repair_methods:
        for attempt in attempts:
            usage = attempt.get("usage") or {}
            lines.append(
                "| {dataset} | `{method}` | {round} | {feedback} | {train} | {val} | {requests} | "
                "{tokens} |".format(
                    dataset=spec.get("dataset_id"),
                    method=str(spec.get("spec_hash") or "")[:8],
                    round=attempt.get("round"),
                    feedback=len(attempt.get("feedback") or []),
                    train=_fmt(attempt.get("train_score")),
                    val=_fmt(attempt.get("validation_score")),
                    requests=_fmt(usage.get("requests")),
                    tokens=_fmt(usage.get("total_tokens")),
                )
            )
    lines.append("")

    lines += [
        "## Feedback families",
        "",
        "| Dataset | Method | Round | Families |",
        "|---|---|---:|---|",
    ]
    for spec, _run, _outcome, attempts in repair_methods:
        for attempt in attempts:
            families = [item.get("family") for item in attempt.get("feedback") or []]
            if families:
                lines.append(
                    f"| {spec.get('dataset_id')} | `{str(spec.get('spec_hash') or '')[:8]}` | "
                    f"{attempt.get('round')} | {', '.join(families)} |"
                )
    lines.append("")

    lines += [
        "## Baseline comparison (agent arm, no repair, same no-metadata condition)",
        "",
        "| Dataset | Mode | Baseline mean [min–max] | Repair test | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for spec, run, _outcome, _attempts in repair_methods:
        dataset_id = str(spec.get("dataset_id"))
        values = [row.get("test") for row in baseline.get(dataset_id, [])]
        mean = _mean(values)
        test = (run.get("metrics") or {}).get("primary", {}).get("test")
        delta = None if mean is None or test is None else test - mean
        low = min(values) if values else None
        high = max(values) if values else None
        mode = "control" if not (spec.get("repair") or {}).get("rounds_requested") else "repair"
        lines.append(
            f"| {dataset_id} | {mode} | {_fmt(mean)} [{_fmt(low)}–{_fmt(high)}] (n={len(values)}) | "
            f"{_fmt(test)} | {_fmt(delta)} |"
        )
    lines.append("")

    lines += [
        "## Leak check",
        "",
        "Validation shared_ids must never appear in an authoring/repair prompt.",
        "",
        "| Dataset | Method | Rounds checked | Validation ids in prompts |",
        "|---|---|---:|---|",
    ]
    for spec, _run, _outcome, attempts in repair_methods:
        validation_ids = set((spec.get("repair") or {}).get("validation_ids") or [])
        leaked = False
        for attempt in attempts:
            prompt = str(attempt.get("prompt") or "")
            if any(doc_id in prompt for doc_id in validation_ids):
                leaked = True
        lines.append(
            f"| {spec.get('dataset_id')} | `{str(spec.get('spec_hash') or '')[:8]}` | "
            f"{len(attempts)} | {'YES' if leaked else 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="plan_document_type,plan_year,upr_recommendations,upr_state",
    )
    parser.add_argument("--since", default=None, help="only repair specs created at/after ISO time")
    parser.add_argument(
        "--baseline-since",
        default="2026-11-23T14:09:15+00:00",
        help="created-at floor for the no-repair agent baseline methods",
    )
    parser.add_argument("--out", default=str(REPORTS_DIR / "admin_repair.md"))
    args = parser.parse_args(argv)

    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    since = datetime.fromisoformat(args.since) if args.since else None
    baseline_since = datetime.fromisoformat(args.baseline_since) if args.baseline_since else None

    repair_methods = []
    for method in runs.list_methods():
        method_id = str(method.get("method_id") or "")
        spec = _spec(method_id)
        if not spec or not (spec.get("repair") or {}).get("enabled"):
            continue
        if spec.get("dataset_id") not in datasets:
            continue
        created_at = spec.get("created_at")
        if since is not None and created_at and datetime.fromisoformat(created_at) < since:
            continue
        run = _latest_run(method_id)
        if run is None:
            continue
        outcome = spec.get("repair_outcome") or (run.get("metrics") or {}).get("repair") or {}
        repair_methods.append((spec, run, outcome, _attempts(method_id)))
    if not repair_methods:
        raise SystemExit("No repair methods found.")

    report = build_report(
        repair_methods, _baseline_rows(baseline_since, {str(s.get("dataset_id")) for s, *_ in repair_methods})
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

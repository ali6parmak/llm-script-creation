"""Generate an extraction script with an LLM, repair it, then evaluate it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from . import catalog, llm, prompts, runs, sandbox
from .configuration import CATALOG_PATH, MODEL
from .db import document_from_row
from .evaluation import (
    aggregate_plan,
    aggregate_upr,
    aggregate_upr_state,
    evaluate_plan_sample,
    evaluate_upr_sample,
    evaluate_upr_state_sample,
)
from .input_spec import InputSpec, payload_excerpt, prepare_input


def _load_dataset(conn: Any, dataset_id: str) -> dict[str, Any]:
    row = catalog.get_dataset(conn, dataset_id)
    if row is None:
        raise SystemExit(f"Unknown dataset: {dataset_id}")
    dataset = dict(row)
    dataset["manifest"] = json.loads(dataset["manifest_json"])
    dataset["target_schema"] = json.loads(dataset["target_schema_json"])
    return dataset


def execution_payload(document: Any, spec: InputSpec) -> dict[str, Any]:
    """Payload passed to `extract`; the single execution-side entry point."""
    return prepare_input(document, spec)


def _payload_for_row(row: dict[str, Any], spec: InputSpec) -> dict[str, Any]:
    return execution_payload(document_from_row(row), spec)


def _select_examples(conn: Any, dataset_id: str, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    samples = [dict(row) for row in catalog.iter_samples(conn, dataset_id, split="train")]
    if dataset_id == "upr_recommendations":
        samples.sort(key=lambda sample: len(json.loads(sample["gold_json"])["recommendations"]))
    else:
        samples.sort(key=lambda sample: sample["doc_id"])
    examples = []
    for sample in samples[:count]:
        row = catalog.get_document(conn, sample["db"], sample["doc_id"])
        if row is None:
            continue
        examples.append(
            {
                "doc_id": sample["doc_id"],
                "gold": json.loads(sample["gold_json"]),
                "document": document_from_row(dict(row)),
            }
        )
    return examples


def _parse_code(response: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = blocks[0] if blocks else response
    code = code.strip()
    if code.startswith("python\n"):
        code = code[len("python\n") :]
    return code


def _output_issues(dataset_id: str, output: Any, fields: list[str]) -> list[str]:
    if not isinstance(output, dict):
        return [f"output is {type(output).__name__}, expected dict"]
    if dataset_id == "upr_recommendations":
        if not isinstance(output.get("recommendations"), list):
            return ["missing list `recommendations`"]
        for index, rec in enumerate(output["recommendations"][:5]):
            if not isinstance(rec, dict):
                return [f"recommendation {index} is not an object"]
            for key in ("number", "text"):
                if key not in rec:
                    return [f"recommendation {index} missing `{key}`"]
        return []
    if dataset_id == "upr_state":
        return [] if "state_under_review" in output else ["missing `state_under_review`"]
    if len(fields) == 1:
        return [] if "value" in output or fields[0] in output else ["missing `value`"]
    return [f"missing field `{field}`" for field in fields if field not in output]


PROMPT_EXAMPLES = 2
MAX_REPAIR_ROUNDS = 4
FAILURES_PER_ROUND = 3
MAX_STALE_ROUNDS = 2


def _aggregate(dataset_id: str, records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    if dataset_id == "upr_recommendations":
        return aggregate_upr(records)
    if dataset_id == "upr_state":
        return aggregate_upr_state(records)
    return aggregate_plan(records, fields)


def primary_score(dataset_id: str, metrics: dict[str, Any]) -> float:
    """Score used to compare rounds: F1 for recommendations, accuracy otherwise."""
    if dataset_id == "upr_recommendations":
        return float(metrics.get("f1", 0.0))
    return float(metrics.get("accuracy", 0.0))


def round_selection_key(state: dict[str, Any]) -> tuple[float, float, int]:
    """Best script first on the 48 non-example docs, then all-50, then earlier."""
    return (state["nonexample_score"], state["score"], -state["round"])


def select_best_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in rounds if not item.get("issues")]
    return max(valid, key=round_selection_key) if valid else None


def _is_failure(dataset_id: str, record: dict[str, Any]) -> bool:
    if record.get("status") == "error" or record.get("error"):
        return True
    if dataset_id == "upr_recommendations":
        details = record.get("details") or {}
        metrics = details.get("metrics") or {}
        return not details.get("state_correct") or float(metrics.get("f1", 0.0)) < 0.999
    return record.get("status") != "hit"


def _failure_group(dataset_id: str, record: dict[str, Any]) -> str:
    if record.get("status") == "error" or record.get("error"):
        return "crash / invalid output"
    if dataset_id in {
        "plan_document_type",
        "plan_legal_status",
        "plan_means_of_adoption",
        "plan_body",
    }:
        field = dataset_id.removeprefix("plan_")
        return str((record.get("gold") or {}).get(field) or "unknown label")
    return str(record.get("status") or "unknown")


def select_repair_failures(
    dataset_id: str,
    records: list[dict[str, Any]],
    example_ids: set[str],
    limit: int = FAILURES_PER_ROUND,
) -> list[dict[str, Any]]:
    """Choose a small, representative batch: crash first, then one per label/status.

    Prompt examples are only used when there are not enough other failures.
    """
    failures = [record for record in records if _is_failure(dataset_id, record)]
    if not failures:
        return []

    def sort_key(record: dict[str, Any]) -> tuple[int, int, float, str]:
        crashed = record.get("status") == "error" or bool(record.get("error"))
        is_example = record.get("doc_id") in example_ids
        if dataset_id == "upr_recommendations":
            score = float((record.get("details") or {}).get("metrics", {}).get("f1", 0.0))
        else:
            score = 0.0 if record.get("status") == "hit" else 1.0
        return (0 if crashed else 1, 0 if is_example else 1, score, str(record.get("doc_id")))

    pool = sorted(failures, key=sort_key)
    non_examples = [record for record in pool if record.get("doc_id") not in example_ids]
    examples = [record for record in pool if record.get("doc_id") in example_ids]

    def stratified(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        for record in candidates:
            group = _failure_group(dataset_id, record)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            picked.append(record)
            if len(picked) >= count:
                break
        return picked

    selected = stratified(non_examples, limit)
    # Only fall back to prompt-example failures when there are not enough others.
    for record in [*non_examples, *examples]:
        if len(selected) >= limit:
            break
        if record not in selected:
            selected.append(record)
    return selected[:limit]


def _run_script(
    dataset_id: str,
    dataset: dict[str, Any],
    samples: list[dict[str, Any]],
    conn: Any,
    script: str,
    spec: InputSpec,
) -> list[dict[str, Any]]:
    """Run one script over samples, returning one evaluated record per sample."""
    fields = list(dataset["target_schema"])
    payloads: list[dict[str, Any]] = []
    for sample in samples:
        row = catalog.get_document(conn, dataset["db"], sample["doc_id"])
        payloads.append(
            _payload_for_row(dict(row), spec) if row is not None else {"id": sample["doc_id"]}
        )

    records: list[dict[str, Any]] = []
    results = sandbox.iter_results(script, payloads, chunk_size=25, timeout=240)
    for sample, result in zip(samples, results, strict=False):
        gold = json.loads(sample["gold_json"])
        if result is None:
            records.append(
                {
                    "doc_id": sample["doc_id"],
                    "split": sample["split"],
                    "gold": gold,
                    "prediction": None,
                    "status": "error",
                    "error": "script produced no result for this document",
                    "details": {},
                }
            )
            continue
        if not result.get("ok"):
            records.append(
                {
                    "doc_id": sample["doc_id"],
                    "split": sample["split"],
                    "gold": gold,
                    "prediction": None,
                    "status": "error",
                    "error": result.get("error", "script error")[:3000],
                    "details": {},
                }
            )
            continue
        prediction = result.get("output")
        issues = _output_issues(dataset_id, prediction, fields)
        if issues:
            records.append(
                {
                    "doc_id": sample["doc_id"],
                    "split": sample["split"],
                    "gold": gold,
                    "prediction": prediction,
                    "status": "error",
                    "error": "invalid output: " + "; ".join(issues),
                    "details": {},
                }
            )
            continue
        if dataset_id == "upr_recommendations":
            details = evaluate_upr_sample(gold, prediction)
        elif dataset_id == "upr_state":
            details = evaluate_upr_state_sample(gold, prediction)
        else:
            details = evaluate_plan_sample(gold, prediction, fields)
        records.append(
            {
                "doc_id": sample["doc_id"],
                "split": sample["split"],
                "gold": gold,
                "prediction": prediction,
                "status": details["status"],
                "error": None,
                "details": details,
            }
        )
    return records


def _display_gold(dataset_id: str, gold: dict[str, Any], limit: int = 8) -> str:
    value = gold
    if dataset_id == "upr_recommendations" and len(gold.get("recommendations", [])) > limit:
        value = dict(gold)
        value["recommendations"] = gold["recommendations"][:limit]
        value["_note"] = f"truncated to {limit} of {len(gold['recommendations'])} recommendations"
    return json.dumps(value, ensure_ascii=False, indent=2)


def _display_output(dataset_id: str, prediction: Any, limit: int = 8) -> str:
    value = prediction
    if (
        dataset_id == "upr_recommendations"
        and isinstance(prediction, dict)
        and len(prediction.get("recommendations") or []) > limit
    ):
        value = dict(prediction)
        value["recommendations"] = prediction["recommendations"][:limit]
        value["_note"] = (
            f"truncated to {limit} of {len(prediction['recommendations'])} recommendations"
        )
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text[:6000]


def _failure_focuses(dataset_id: str, gold: dict[str, Any]) -> tuple[str, ...]:
    focuses = list(prompts.prompt_focuses(dataset_id))
    if dataset_id == "upr_state":
        state = gold.get("state_under_review")
        if state:
            focuses.insert(0, str(state))
    elif dataset_id == "upr_recommendations":
        for recommendation in gold.get("recommendations", [])[:3]:
            number = str(recommendation.get("number") or "")
            if not number:
                continue
            focuses.append(number)
            # Gold may use sub-numbers (140.1.1); the document usually shows 140.1.
            parts = number.split(".")
            if len(parts) >= 3:
                focuses.append(".".join(parts[:2]))
    else:
        for value in gold.values():
            if isinstance(value, (str, int)):
                focuses.append(str(value))
    return tuple(focuses)


def _repair_feedback(
    dataset_id: str,
    round_index: int,
    state: dict[str, Any],
    conn: Any,
    dataset: dict[str, Any],
    spec: InputSpec,
) -> str:
    parts: list[str] = [
        f"Repair round {round_index}. The current best script scored "
        f"{state['score']:.3f} ({'F1' if dataset_id == 'upr_recommendations' else 'accuracy'}) "
        f"on {state['document_count']} train documents, with "
        f"{state['failure_count']} failures."
    ]
    if state["issues"]:
        parts.append(
            "The previous attempt did not produce a valid script:\n- "
            + "\n- ".join(state["issues"])
        )
        if state["empty_response"]:
            parts.append(
                "The model returned empty content. Return one complete Python code block."
            )

    for index, record in enumerate(state["selected"], start=1):
        gold = record.get("gold") or {}
        row = catalog.get_document(conn, dataset["db"], record["doc_id"])
        payload = (
            _payload_for_row(dict(row), spec)
            if row is not None
            else {"id": record["doc_id"]}
        )
        excerpt, truncated = payload_excerpt(
            payload,
            spec,
            char_limit=2200,
            segment_limit=12,
            focuses=_failure_focuses(dataset_id, gold),
        )
        label = "transformed input excerpt" if truncated else "transformed input"
        block = "```text" if spec.representation == "text" else "```json"
        section = [
            f"FAILURE {index} — document {record['doc_id']} — status: {record.get('status')}",
            f"{label.capitalize()} (exactly the input the script receives: {spec.describe()}):\n"
            f"{block}\n{excerpt}\n```",
            f"Gold output:\n```json\n{_display_gold(dataset_id, gold)}\n```",
        ]
        if record.get("prediction") is not None:
            section.append(
                f"Actual output:\n```json\n{_display_output(dataset_id, record['prediction'])}\n```"
            )
        if record.get("error"):
            section.append(f"Error / traceback:\n```text\n{record['error'][:2500]}\n```")
        parts.append("\n\n".join(section))

    parts.append(
        "Fix the general extraction rule, not these document-specific values. Do not hardcode "
        "document ids, branch on individual documents, or copy exact gold strings. Return the "
        "complete corrected `extract(document)` script in one python code block."
    )
    return "\n\n".join(parts)


def generate(
    dataset_id: str,
    *,
    input_spec: InputSpec | None = None,
    representation: str = "text",
    include_position: bool = False,
    include_type: bool = False,
    examples_count: int = PROMPT_EXAMPLES,
    limit: int | None = None,
    max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    temperature: float = 0.0,
    model: str = MODEL,
    max_tokens: int = 8192,
    think: bool | str | None = False,
    method_name: str | None = None,
    method_description: str | None = None,
    dry_run: bool = False,
) -> int:
    """Generate, repair on train, freeze the best script, then evaluate test."""
    if not CATALOG_PATH.exists():
        raise SystemExit("Catalog missing. Run `uv run build-datasets` first.")
    spec = input_spec or InputSpec(
        representation=representation,  # type: ignore[arg-type]
        include_position=include_position,
        include_type=include_type,
    )
    conn = catalog.connect()
    try:
        dataset = _load_dataset(conn, dataset_id)
        examples = _select_examples(conn, dataset_id, examples_count)
        example_ids = {example["doc_id"] for example in examples}
        messages = prompts.build_messages(dataset, examples, spec)
        prompt_text = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        print(
            f"Prompt: {len(prompt_text):,} chars (~{len(prompt_text) // 4:,} tokens), "
            f"{len(examples)} examples",
            flush=True,
        )
        if dry_run:
            print(prompt_text)
            return 0

        train_samples = sorted(
            (dict(row) for row in catalog.iter_samples(conn, dataset_id, split="train")),
            key=lambda sample: sample["doc_id"],
        )
        if limit is not None:
            train_samples = train_samples[:limit]
        test_samples = sorted(
            (dict(row) for row in catalog.iter_samples(conn, dataset_id, split="test")),
            key=lambda sample: sample["doc_id"],
        )
        train_ids = {sample["doc_id"] for sample in train_samples}
        nonexample_count = len(train_ids - example_ids)
        print(
            f"Split: {len(train_samples)} train ({nonexample_count} non-example) · "
            f"{len(test_samples)} test",
            flush=True,
        )

        rounds: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {}
        response_model = model
        best_state: dict[str, Any] | None = None
        feedback_source: dict[str, Any] | None = None
        best_score = -1.0
        stale_rounds = 0
        feedback: str | None = None
        history = messages

        for round_index in range(max_repair_rounds + 1):
            kind = "initial" if round_index == 0 else "repair"
            if feedback is not None and feedback_source is not None:
                history = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": feedback_source["response"] or feedback_source["code"],
                    },
                    {"role": "user", "content": feedback},
                ]
            print(f"Round {round_index} ({kind}): generating…", flush=True)
            response = llm.chat(
                history,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
            )
            response_model = response.model
            for key, value in response.usage.items():
                if isinstance(value, int):
                    total_usage[key] = total_usage.get(key, 0) + value
            code = _parse_code(response.content)
            issues = sandbox.validate_script(code)
            empty_response = not response.content.strip()

            if issues:
                records: list[dict[str, Any]] = []
                train_metrics: dict[str, Any] = {}
                nonexample_metrics: dict[str, Any] = {}
                score = 0.0
                nonexample_score = 0.0
                failures: list[dict[str, Any]] = []
                selected: list[dict[str, Any]] = []
            else:
                records = _run_script(
                    dataset_id, dataset, train_samples, conn, code, spec
                )
                train_metrics = _aggregate(dataset_id, records, list(dataset["target_schema"]))
                nonexample_records = [
                    record for record in records if record["doc_id"] not in example_ids
                ]
                nonexample_metrics = _aggregate(
                    dataset_id, nonexample_records, list(dataset["target_schema"])
                )
                score = primary_score(dataset_id, train_metrics)
                nonexample_score = primary_score(dataset_id, nonexample_metrics)
                failures = [record for record in records if _is_failure(dataset_id, record)]
                selected = select_repair_failures(dataset_id, records, example_ids)

            state = {
                "round": round_index,
                "kind": kind,
                "code": code,
                "response": response.content,
                "issues": issues,
                "empty_response": empty_response,
                "records": records,
                "train_metrics": train_metrics,
                "nonexample_metrics": nonexample_metrics,
                "score": score,
                "nonexample_score": nonexample_score,
                "failures": failures,
                "selected": selected,
                "document_count": len(train_samples),
                "failure_count": len(failures),
                "crash_count": sum(
                    1 for record in failures if record.get("status") == "error"
                ),
                "feedback": feedback,
                "messages": history,
                "usage": response.usage,
                "model": response.model,
            }
            rounds.append(state)
            attempts.append(
                {
                    "round": round_index,
                    "attempt": round_index + 1,
                    "kind": kind,
                    "model": response.model,
                    "usage": response.usage,
                    "response": response.content,
                    "code": code,
                    "validation_issues": issues,
                    "feedback": feedback,
                    "train_score": score,
                    "train_nonexample_score": nonexample_score,
                    "empty_response": empty_response,
                }
            )

            if issues:
                print(f"  invalid script: {'; '.join(issues)}", flush=True)
            else:
                print(
                    f"  train {score:.3f} · non-example {nonexample_score:.3f} · "
                    f"{len(failures)} failures ({state['crash_count']} crashes)",
                    flush=True,
                )
                selection_key = round_selection_key(state)
                if best_state is None or selection_key > best_state["selection_key"]:
                    best_state = {**state, "selection_key": selection_key}

            if round_index == 0:
                best_score = nonexample_score
            elif nonexample_score > best_score + 1e-9:
                best_score = nonexample_score
                stale_rounds = 0
            else:
                stale_rounds += 1

            if not issues and not failures:
                print("  perfect train score; stopping repair loop.", flush=True)
                break
            if round_index > 0 and stale_rounds >= MAX_STALE_ROUNDS:
                print(
                    f"  no improvement for {MAX_STALE_ROUNDS} rounds; stopping repair loop.",
                    flush=True,
                )
                break
            if round_index >= max_repair_rounds:
                break
            feedback_source = best_state if best_state is not None else state
            feedback = _repair_feedback(
                dataset_id,
                round_index + 1,
                feedback_source,
                conn,
                dataset,
                spec,
            )
            if best_state is not None and state["issues"] and state is not best_state:
                feedback += (
                    "\n\nNote: the most recent attempt did not produce a valid script "
                    f"({'; '.join(state['issues'])}). Repair the current best script instead."
                )

        now = datetime.now(UTC).isoformat()
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
        model_slug = re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")
        method_id = (
            f"llm_{dataset_id}_{spec.slug()}_{examples_count}ex_{model_slug}_{prompt_hash[:6]}"
        )
        repair_rounds = len(rounds) - 1
        method = {
            "method_id": method_id,
            "name": method_name or f"LLM script — {dataset_id} / {spec.describe()}",
            "description": method_description
            or (
                "LLM-generated extraction script: 2 prompt examples, up to 4 repair rounds "
                "on the 50-document train set, best script picked on the 48 non-example "
                "documents, then frozen and run once on test."
            ),
            "type": "llm_script",
            "dataset_id": dataset_id,
            "task": dataset["task"],
            "model": model,
            "response_model": response_model,
            "input_spec": spec.to_dict(),
            "examples": sorted(example_ids),
            "prompt_version": "v2",
            "split_policy_version": dataset["manifest"].get("split_policy_version"),
            "prompt_hash": prompt_hash,
            "attempts": len(attempts),
            "repair_rounds": repair_rounds,
            "best_round": best_state["round"] if best_state else None,
            "train_documents": len(train_samples),
            "train_nonexample_documents": nonexample_count,
            "test_documents": len(test_samples),
            "usage": total_usage,
            "created_at": now,
            "config": {
                "temperature": temperature,
                "max_repair_rounds": max_repair_rounds,
                "max_tokens": max_tokens,
                "think": think,
                "examples": examples_count,
                "limit": limit,
            },
        }
        round_summaries = [
            {
                "round": item["round"],
                "kind": item["kind"],
                "model": item["model"],
                "usage": item["usage"],
                "validation_issues": item["issues"],
                "train_score": item["score"],
                "train_nonexample_score": item["nonexample_score"],
                "failure_count": item["failure_count"],
                "crash_count": item["crash_count"],
                "selected_failures": [
                    {
                        "doc_id": record["doc_id"],
                        "status": record["status"],
                        "group": _failure_group(dataset_id, record),
                        "error": (record.get("error") or "")[:500],
                    }
                    for record in item["selected"]
                ],
            }
            for item in rounds
        ]

        if best_state is None:
            method["status"] = "failed"
            runs.write_method(method, prompt=prompt_text, attempts=attempts)
            run_id = runs.new_run_id(method_id, dataset_id, "train_test")
            metrics = {
                "error": f"no valid script after {len(rounds)} attempts",
                "repair_rounds": repair_rounds,
                "rounds": round_summaries,
                "token_usage": total_usage,
            }
            run = {
                "run_id": run_id,
                "method_id": method_id,
                "dataset_id": dataset_id,
                "split": "train_test",
                "status": "failed",
                "created_at": now,
                "counts": {},
                "repair_rounds": repair_rounds,
                "rounds": round_summaries,
            }
            runs.write_run(
                run,
                [],
                metrics,
                prompt=prompt_text,
                rounds=rounds,
            )
            print(f"No valid script after {len(rounds)} attempts. Wrote failed run {run_id}.")
            return 1

        final_code = best_state["code"]
        test_records = _run_script(dataset_id, dataset, test_samples, conn, final_code, spec)
        test_metrics = _aggregate(dataset_id, test_records, list(dataset["target_schema"]))
        train_metrics = best_state["train_metrics"]
        nonexample_metrics = best_state["nonexample_metrics"]
        train_score = primary_score(dataset_id, train_metrics)
        nonexample_score = primary_score(dataset_id, nonexample_metrics)
        test_score = primary_score(dataset_id, test_metrics)
        metric_name = "f1" if dataset_id == "upr_recommendations" else "accuracy"
        primary = {
            "metric": metric_name,
            "train": train_score,
            "train_nonexample": nonexample_score,
            "test": test_score,
            "gap": train_score - test_score,
            "gap_nonexample": nonexample_score - test_score,
        }
        metrics = {
            "primary_metric": metric_name,
            "primary": primary,
            "train": train_metrics,
            "train_nonexample": nonexample_metrics,
            "test": test_metrics,
            "status_counts": {
                "train": train_metrics.get("status_counts", {}),
                "train_nonexample": nonexample_metrics.get("status_counts", {}),
                "test": test_metrics.get("status_counts", {}),
            },
            "repair_rounds": repair_rounds,
            "best_round": best_state["round"],
            "rounds": round_summaries,
            "failure_counts": {
                "train": best_state["failure_count"],
                "train_crash": best_state["crash_count"],
                "per_round": [item["failure_count"] for item in rounds],
            },
            "token_usage": total_usage,
            "split_policy_version": method["split_policy_version"],
            "input_spec": spec.to_dict(),
            "examples": sorted(example_ids),
            "train_documents": len(train_samples),
            "train_nonexample_documents": nonexample_count,
            "test_documents": len(test_samples),
        }

        method["script_hash"] = hashlib.sha256(final_code.encode()).hexdigest()[:12]
        method["final"] = primary
        runs.write_method(method, prompt=prompt_text, script=final_code, attempts=attempts)
        run_id = runs.new_run_id(method_id, dataset_id, "train_test")
        run = {
            "run_id": run_id,
            "method_id": method_id,
            "dataset_id": dataset_id,
            "split": "train_test",
            "status": "completed",
            "created_at": now,
            "counts": test_metrics.get("status_counts", {}),
            "repair_rounds": repair_rounds,
            "best_round": best_state["round"],
            "rounds": round_summaries,
            "token_usage": total_usage,
            "split_policy_version": method["split_policy_version"],
        }
        runs.write_run(
            run,
            [*best_state["records"], *test_records],
            metrics,
            prompt=prompt_text,
            script=final_code,
            rounds=rounds,
        )
        print(f"\nRun: {run_id}")
        print(
            f"  train {train_score:.3f} ({nonexample_score:.3f} on "
            f"{nonexample_count} non-example) · test {test_score:.3f} · "
            f"gap {primary['gap']:+.3f} · repair rounds {repair_rounds} · "
            f"best round {best_state['round']} · tokens {total_usage.get('total_tokens', 0):,}",
            flush=True,
        )
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", default="text", choices=["text", "segments"])
    parser.add_argument("--head-chars", type=int, default=None)
    parser.add_argument("--tail-chars", type=int, default=None)
    parser.add_argument("--head-segments", type=int, default=None)
    parser.add_argument("--tail-segments", type=int, default=None)
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--include-type", action="store_true")
    parser.add_argument("--examples", type=int, default=PROMPT_EXAMPLES)
    parser.add_argument("--limit", type=int, default=None, help="limit train documents (smoke)")
    parser.add_argument("--max-repair-rounds", type=int, default=MAX_REPAIR_ROUNDS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--think",
        default="off",
        choices=["off", "low", "high", "max"],
        help="thinking level; 'off' sets think=false (required for cloud reasoning models)",
    )
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--method-description", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    input_spec = InputSpec(
        representation=args.representation,
        head_chars=args.head_chars,
        tail_chars=args.tail_chars,
        head_segments=args.head_segments,
        tail_segments=args.tail_segments,
        include_position=args.include_position,
        include_type=args.include_type,
    )
    return generate(
        args.dataset,
        input_spec=input_spec,
        examples_count=args.examples,
        limit=args.limit,
        max_repair_rounds=args.max_repair_rounds,
        temperature=args.temperature,
        model=args.model,
        max_tokens=args.max_tokens,
        think=False if args.think == "off" else args.think,
        method_name=args.method_name,
        method_description=args.method_description,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

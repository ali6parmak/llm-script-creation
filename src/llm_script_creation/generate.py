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
from .input_spec import InputSpec, prepare_input


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


def _smoke_check(
    script: str,
    dataset_id: str,
    fields: list[str],
    payloads: list[dict[str, Any]],
    golds: list[dict[str, Any]],
) -> tuple[str | None, int | None, str | None, str | None]:
    results = list(
        sandbox.iter_results(script, payloads, chunk_size=max(1, len(payloads)), timeout=120)
    )
    outputs = [result.get("output") for result in results]
    for index, result in enumerate(results):
        if not result.get("ok"):
            return result.get("error", "unknown script error"), index, None, None
        issues = _output_issues(dataset_id, result.get("output"), fields)
        if issues:
            return (
                "invalid output: " + "; ".join(issues),
                index,
                json.dumps(result.get("output"), ensure_ascii=False)[:800],
                json.dumps(golds[index], ensure_ascii=False)[:800] if index < len(golds) else None,
            )

    # A syntactically valid script can still be semantically wrong (e.g. a regex
    # capturing group returning "20" instead of 2020). Check a few dev examples
    # and feed the first mismatch back into the repair loop.
    hits = 0
    notes = []
    for index, output in enumerate(outputs):
        gold = golds[index] if index < len(golds) else {}
        if dataset_id == "upr_recommendations":
            details = evaluate_upr_sample(gold, output)
            if details["metrics"]["matched"] > 0:
                hits += 1
            else:
                notes.append(
                    f"document {index}: no recommendation matched; gold has "
                    f"{details['metrics']['gold_count']} recommendations"
                )
        elif dataset_id == "upr_state":
            details = evaluate_upr_state_sample(gold, output)
            if details["state_correct"]:
                hits += 1
            else:
                notes.append(
                    f"document {index}: expected state {gold.get('state_under_review')!r}, "
                    f"got {output.get('state_under_review') if isinstance(output, dict) else output!r}"
                )
        else:
            details = evaluate_plan_sample(gold, output, fields)
            if details["status"] == "hit":
                hits += 1
            else:
                notes.append(
                    f"document {index}: expected {json.dumps(gold, ensure_ascii=False)}, "
                    f"got {json.dumps(output, ensure_ascii=False)[:300]}"
                )
    if outputs and hits == 0:
        return (
            "smoke predictions are valid JSON but do not match any train gold example",
            None,
            json.dumps(outputs[0], ensure_ascii=False)[:800] if outputs else None,
            json.dumps(golds[0], ensure_ascii=False)[:800] if golds else None,
        )
    if notes:
        print(f"  smoke quality: {hits}/{len(outputs)} hit; first miss: {notes[0]}", flush=True)
    return None, None, None, None


def _feedback(
    error: str,
    payload: dict[str, Any] | None,
    output: str | None,
    expected: str | None,
) -> str:
    parts = [
        "The previous script did not pass the smoke test.",
        f"Error: {error}",
    ]
    if payload is not None:
        parts.append(
            "First failing input (truncated):\n```json\n"
            + json.dumps(payload, ensure_ascii=False)[:2500]
            + "\n```"
        )
    if output is not None:
        parts.append(f"Output produced:\n```json\n{output}\n```")
    if expected is not None:
        parts.append(f"Expected output for that example:\n```json\n{expected}\n```")
    parts.append(
        "Fix the script. It must still define `extract(document)`, use no I/O, and return "
        "the required shape. Return the complete corrected file in one python code block."
    )
    return "\n\n".join(parts)


def _evaluate(
    dataset_id: str,
    dataset: dict[str, Any],
    samples: list[dict[str, Any]],
    conn: Any,
    script: str,
    spec: InputSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = list(dataset["target_schema"])
    payloads = (
        _payload_for_row(
            dict(catalog.get_document(conn, dataset["db"], sample["doc_id"])),
            spec,
        )
        for sample in samples
    )
    records: list[dict[str, Any]] = []
    results = sandbox.iter_results(script, payloads, chunk_size=20, timeout=180)
    for sample in samples:
        result = next(results, None)
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
                    "error": result.get("error", "script error")[:2000],
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

    if dataset_id == "upr_recommendations":
        metrics = aggregate_upr(records)
    elif dataset_id == "upr_state":
        metrics = aggregate_upr_state(records)
    else:
        metrics = aggregate_plan(records, fields)
    return records, metrics


def generate(
    dataset_id: str,
    *,
    input_spec: InputSpec | None = None,
    representation: str = "text",
    include_position: bool = False,
    include_type: bool = False,
    split: str = "train",
    examples_count: int = 2,
    limit: int | None = None,
    max_attempts: int = 3,
    temperature: float = 0.0,
    model: str = MODEL,
    max_tokens: int = 8192,
    think: bool | str | None = False,
    method_name: str | None = None,
    method_description: str | None = None,
    dry_run: bool = False,
) -> int:
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
        fields = list(dataset["target_schema"])
        examples = _select_examples(conn, dataset_id, examples_count)
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

        samples = [dict(row) for row in catalog.iter_samples(conn, dataset_id, split=split)]
        if limit is not None:
            samples = samples[:limit]
        smoke_samples = samples[:3]
        smoke_golds = [json.loads(sample["gold_json"]) for sample in smoke_samples]
        smoke_payloads = [
            _payload_for_row(
                dict(catalog.get_document(conn, dataset["db"], sample["doc_id"])),
                spec,
            )
            for sample in smoke_samples
        ]

        attempts: list[dict[str, Any]] = []
        final_code = None
        response_model = model
        total_usage: dict[str, int] = {}
        history = messages
        for attempt in range(1, max_attempts + 1):
            print(f"Generation attempt {attempt}/{max_attempts}…", flush=True)
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
            error = None
            if not issues:
                error, failing_index, output, expected = _smoke_check(
                    code, dataset_id, fields, smoke_payloads, smoke_golds
                )
            else:
                failing_index, output, expected = None, None, None
            attempts.append(
                {
                    "attempt": attempt,
                    "model": response.model,
                    "usage": response.usage,
                    "response": response.content,
                    "code": code,
                    "validation_issues": issues,
                    "smoke_error": error,
                }
            )
            if not issues and error is None:
                final_code = code
                print(f"  valid script on attempt {attempt}", flush=True)
                break
            failure = error or "; ".join(issues)
            if not response.content.strip():
                failure = (
                    f"{failure}; the model returned empty content "
                    f"(finish_reason={response.finish_reason!r}, reasoning_chars={len(response.reasoning)})"
                )
            print(f"  attempt {attempt} failed: {failure}", flush=True)
            if attempt == max_attempts:
                break
            failed_payload = (
                smoke_payloads[failing_index]
                if failing_index is not None and failing_index < len(smoke_payloads)
                else None
            )
            history = [
                *messages,
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": _feedback(
                        error or "; ".join(issues), failed_payload, output, expected
                    ),
                },
            ]

        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
        now = datetime.now(UTC).isoformat()
        model_slug = re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")
        method_id = (
            f"llm_{dataset_id}_{spec.slug()}_{examples_count}ex_{model_slug}_{prompt_hash[:6]}"
        )
        method = {
            "method_id": method_id,
            "name": method_name or f"LLM script — {dataset_id} / {spec.describe()}",
            "description": method_description
            or "Generated extraction script with the configured prompt and examples.",
            "type": "llm_script",
            "dataset_id": dataset_id,
            "task": dataset["task"],
            "model": model,
            "input_spec": spec.to_dict(),
            "examples": [example["doc_id"] for example in examples],
            "prompt_version": "v2",
            "split_policy_version": dataset["manifest"].get("split_policy_version"),
            "prompt_hash": prompt_hash,
            "attempts": len(attempts),
            "response_model": response_model,
            "usage": total_usage,
            "created_at": now,
            "config": {
                "temperature": temperature,
                "max_attempts": max_attempts,
                "max_tokens": max_tokens,
                "think": think,
                "split": split,
                "limit": limit,
            },
        }

        if final_code is None:
            runs.write_method(method, prompt=prompt_text, attempts=attempts)
            run_id = runs.new_run_id(method_id, dataset_id, split)
            run = {
                "run_id": run_id,
                "method_id": method_id,
                "dataset_id": dataset_id,
                "split": split,
                "status": "failed",
                "created_at": now,
                "counts": {},
            }
            runs.write_run(
                run,
                [],
                {"error": f"no valid script after {max_attempts} attempts"},
                prompt=prompt_text,
            )
            print(f"No valid script. Wrote failed run {run_id}.")
            return 1

        method["script_hash"] = hashlib.sha256(final_code.encode()).hexdigest()[:12]
        runs.write_method(method, prompt=prompt_text, script=final_code, attempts=attempts)
        records, metrics = _evaluate(
            dataset_id,
            dataset,
            samples,
            conn,
            final_code,
            spec,
        )
        run_id = runs.new_run_id(method_id, dataset_id, split)
        run = {
            "run_id": run_id,
            "method_id": method_id,
            "dataset_id": dataset_id,
            "split": split,
            "status": "completed",
            "created_at": now,
            "counts": metrics.get("status_counts", {}),
        }
        runs.write_run(run, records, metrics, prompt=prompt_text, script=final_code)
        print(f"\nRun: {run_id}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--representation", default="text", choices=["text", "segments"])
    parser.add_argument("--head-chars", type=int, default=None)
    parser.add_argument("--tail-chars", type=int, default=None)
    parser.add_argument("--head-segments", type=int, default=None)
    parser.add_argument("--tail-segments", type=int, default=None)
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--include-type", action="store_true")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
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
        split=args.split,
        examples_count=args.examples,
        limit=args.limit,
        max_attempts=args.max_attempts,
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

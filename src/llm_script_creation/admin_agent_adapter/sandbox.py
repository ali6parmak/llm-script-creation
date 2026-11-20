"""Execute a returned extractor over payloads with the admin agent's sandbox.

Two contracts must stay exact:

* input text: ``"\\n".join(p["text"] for p in seg["paragraphs"])`` — the
  document-extraction contract in the admin agent's system prompt. No page
  markers, no filtering, no truncation;
* ``ctx``: the admin agent's own ``build_context(entity)`` — literally
  ``{"shared_id", "title", "metadata"}`` with Uwazi metadata untouched.

The function itself runs through ``build_extractor_namespace``, so it sees the
same bound helpers (``re``, ``textmatch``, ``dateparse``, ``htmlextract``) as it
did while being authored. Test documents only ever reach this module's execute
request, never the authoring request.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from uwazi_admin_agent.domain.extraction_eval import (
    MatchPolicy,
    SampleResult,
    format_score,
    score_extraction,
)
from uwazi_admin_agent.use_cases.extraction_benchmark_tool import build_context, call_candidate
from uwazi_admin_agent.use_cases.extractor_sandbox import (
    build_extractor_namespace,
    validate_extractor_source,
)

from .records import agent_entity_from_record, joined_paragraph_text, paragraph_dicts


def sample_input(sample: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the exact ``(text, ctx)`` pair the admin agent's extractor receives."""
    entity = agent_entity_from_record(sample)
    text = joined_paragraph_text(paragraph_dicts(sample.get("segments")))
    # Use the admin agent's own context builder, not a local lookalike.
    return text, build_context(entity)


def _admin_score(
    predictions: list[dict[str, Any]],
    score_request: dict[str, Any],
) -> dict[str, Any]:
    property_names = list(score_request.get("property_names") or [])
    expected = score_request.get("expected") or {}
    policy = MatchPolicy(str(score_request.get("policy") or MatchPolicy.NORMALIZED.value))
    samples = [
        SampleResult(
            shared_id=item["doc_id"],
            expected=expected.get(item["doc_id"]) or {},
            predicted=item.get("prediction"),
            error=item.get("error"),
        )
        for item in predictions
    ]
    result = score_extraction(samples, property_names, policy=policy)
    return {
        "policy": result.policy.value,
        "accuracy": result.accuracy,
        "coverage": result.coverage,
        "matched": result.matched,
        "mismatched": result.mismatched,
        "missing": result.missing,
        "samples_total": result.samples_total,
        "samples_scored": result.samples_scored,
        "samples_with_error": result.samples_with_error,
        "fields": {name: field.model_dump() for name, field in result.fields.items()},
        "mismatches": [item.model_dump() for item in result.mismatches],
        "report": format_score(result),
    }


def execute_extractor(
    source: str,
    samples: Iterable[dict[str, Any]],
    score_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate ``source``, run it over ``samples``, optionally admin-score it."""
    rejection = validate_extractor_source(source)
    if rejection is not None:
        return {"mode": "execute", "status": "error", "error": rejection}

    namespace = build_extractor_namespace(source)
    extract = namespace["extract"]
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        prediction: dict[str, Any] | None = None
        error: str | None = None
        text_chars = 0
        try:
            text, context = sample_input(sample)
            text_chars = len(text)
            prediction, error = call_candidate(extract, text, context)
        except Exception as exc:  # noqa: BLE001 - a bad sample is data, not a crash
            error = f"{type(exc).__name__}: {exc}"
        predictions.append(
            {
                "doc_id": sample.get("doc_id"),
                "split": sample.get("split"),
                "prediction": prediction,
                "error": error,
                "text_chars": text_chars,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )

    response: dict[str, Any] = {
        "mode": "execute",
        "status": "ok",
        "validation": None,
        "predictions": predictions,
    }
    if score_request:
        response["admin_eval"] = _admin_score(predictions, score_request)
    return response


__all__ = ["execute_extractor", "sample_input"]

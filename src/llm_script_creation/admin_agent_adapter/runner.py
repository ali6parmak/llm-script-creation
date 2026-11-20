"""Run the admin agent's nested extractor subagent (sidecar only).

This is the phase-1 strategy: the admin agent's real extraction machinery, with
no live Uwazi. ``build_extractor_agent`` builds the nested agent; we call it
directly (rather than through ``GenerateScriptUseCase``) with read-only deps whose
documents are the train split.

The model is the admin agent's own: ``OllamaAdapter()`` with no arguments, i.e.
``uwazi_agent.configuration.MODEL`` (``deepseek-v4.1-flash:1017-cloud``) and its
configured Ollama endpoint. We do not override either.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import UsageLimits
from uwazi_admin_agent.configuration import MAX_EXTRACTOR_LLM_CALLS
from uwazi_admin_agent.use_cases.author_extractor_tool import (
    ExtractorFunction,
    build_extractor_agent,
)
from uwazi_admin_agent.use_cases.extractor_sandbox import validate_extractor_source
from uwazi_agent.adapters.llm.ollama_adapter import OllamaAdapter

from .ports import build_deps

_TRANSCRIPT_ARG_CHARS = 2_000
_TRANSCRIPT_RESULT_CHARS = 4_000
_TRANSCRIPT_MAX_PARTS = 200


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return _clip(value, _TRANSCRIPT_ARG_CHARS)


def usage_dict(usage: Any) -> dict[str, Any]:
    """Serialize pydantic-ai's ``RunUsage`` dataclass into a JSON-safe dict."""
    fields = (
        "requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_audio_read_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
    )
    result = {field: getattr(usage, field, 0) for field in fields}
    cost = getattr(usage, "cost", None)
    result["cost"] = float(cost) if cost is not None else None
    return result


def transcript(run: Any) -> list[dict[str, Any]]:
    """Compact record of the extractor's tool calls and their outputs.

    Full segmentation peeks can be 200k characters; they are clipped here. The
    point is an auditable trace of what the extractor looked at, not a replay.
    """
    items: list[dict[str, Any]] = []
    for message in getattr(run, "all_messages", lambda: [])():
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", type(part).__name__)
            if kind == "tool-call":
                items.append(
                    {
                        "kind": "tool_call",
                        "tool": getattr(part, "tool_name", ""),
                        "args": _jsonable(getattr(part, "args", None)),
                    }
                )
            elif kind == "tool-return":
                items.append(
                    {
                        "kind": "tool_return",
                        "tool": getattr(part, "tool_name", ""),
                        "content": _clip(getattr(part, "content", ""), _TRANSCRIPT_RESULT_CHARS),
                    }
                )
            elif kind == "text":
                items.append({"kind": "text", "content": _clip(getattr(part, "content", ""), 1_000)})
            if len(items) >= _TRANSCRIPT_MAX_PARTS:
                items.append({"kind": "truncated", "content": "transcript capped"})
                return items
    return items


def _extractor_payload(output: ExtractorFunction) -> dict[str, Any]:
    return {
        "python_code": output.python_code,
        "description": output.description,
        "samples_total": output.samples_total,
        "samples_matched": output.samples_matched,
        "unmatched_notes": output.unmatched_notes,
    }


async def run_authoring(request: dict[str, Any]) -> dict[str, Any]:
    """Author one extractor through the admin agent's nested agent."""
    task = str(request.get("task_prompt") or "")
    max_requests = int(request.get("max_requests") or MAX_EXTRACTOR_LLM_CALLS)
    base: dict[str, Any] = {
        "mode": "author",
        "task_prompt": task,
        "max_requests": max_requests,
    }

    try:
        llm = OllamaAdapter()
    except Exception as exc:  # noqa: BLE001 - report the concrete blocker
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    base.update({"model": llm.model_name, "base_url": llm.ollama_base_url})
    try:
        deps = build_deps(request)
        agent = build_extractor_agent(llm.get_model())
        run = await agent.run(
            task,
            deps=deps,
            usage_limits=UsageLimits(request_limit=max_requests),
        )
    except Exception as exc:  # noqa: BLE001 - the caller records the exact failure
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    usage = usage_dict(getattr(run, "usage", None))
    output = getattr(run, "output", None)
    parts = transcript(run)
    if not isinstance(output, ExtractorFunction):
        return {
            **base,
            "status": "error",
            "error": f"extractor subagent returned an unsupported output type: {type(output).__name__}",
            "empty_output": output is None,
            "usage": usage,
            "transcript": parts,
        }

    code = output.python_code
    rejection = validate_extractor_source(code)
    return {
        **base,
        "status": "ok" if rejection is None else "rejected",
        "model": llm.model_name,
        "base_url": llm.ollama_base_url,
        "extractor": _extractor_payload(output),
        "validation": rejection,
        "usage": usage,
        "tool_calls": usage.get("tool_calls", 0),
        "transcript": parts,
    }


__all__ = ["run_authoring", "transcript", "usage_dict"]

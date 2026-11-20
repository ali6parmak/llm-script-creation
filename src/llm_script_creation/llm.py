"""Minimal Ollama native chat client.

The native `/api/chat` endpoint is used instead of the OpenAI-compatible
`/v1/chat/completions` endpoint because it exposes the `think` control. Setting
`think: false` is essential for cloud reasoning models that otherwise spend the
whole completion budget in hidden reasoning and return empty content.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .configuration import MODEL, OLLAMA_API_KEY, OLLAMA_NATIVE_BASE


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _request(payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = f"{OLLAMA_NATIVE_BASE}/api/chat"
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    think: bool | str | None = False,
    timeout: int = 300,
) -> ChatResponse:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if think is not None:
        payload["think"] = think
    data = _request(payload, timeout)
    try:
        message = data["message"]
        content = message.get("content") or ""
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response: {json.dumps(data)[:1000]}") from exc

    prompt_tokens = data.get("prompt_eval_count") or 0
    completion_tokens = data.get("eval_count") or 0
    return ChatResponse(
        content=content,
        model=data.get("model", model),
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        reasoning=message.get("thinking") or "",
        finish_reason=data.get("done_reason") or "",
        raw=data,
    )

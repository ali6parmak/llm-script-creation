"""Minimal Ollama OpenAI-compatible chat client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .configuration import MODEL, OLLAMA_API_KEY, OLLAMA_BASE_URL


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    timeout: int = 300,
) -> ChatResponse:
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc

    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response: {json.dumps(payload)[:1000]}") from exc
    return ChatResponse(
        content=content,
        model=payload.get("model", model),
        usage=payload.get("usage", {}),
        reasoning=message.get("reasoning") or "",
        finish_reason=choice.get("finish_reason") or "",
        raw=payload,
    )

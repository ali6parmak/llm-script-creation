#!/usr/bin/env python
"""Admin-agent sidecar: run the real Uwazi admin agent in its own environment.

The benchmark process never imports ``uwazi_admin_agent``. It writes one JSON
request under ``work/admin_agent/...`` and invokes this script with the admin
agent's venv:

    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<benchmark>/src:<admin>/src[:<benchmark>/work/admin_agent_extra] \
    <admin>/.venv/bin/python scripts/admin_agent_sidecar.py \
        --mode author --request work/.../author_request.json \
        --response work/.../author_response.json

Modes:
  author  — build read-only deps over the request's train documents, build the
            nested extractor agent, run it once, validate the source.
  execute — execute the returned ``extract`` over the request's samples with
            the admin agent's own sandbox and context builder, optionally
            scoring with the admin agent's ``extraction_eval``.

No Mongo, no Uwazi, no network besides the model endpoint. The admin repo is
never written to (bytecode writing is disabled).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    """Stream execute samples: the full Plan test split is too large for one JSON."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["author", "execute"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    sys.dont_write_bytecode = True
    request_path = Path(args.request)
    response_path = Path(args.response)
    request = _read(request_path)

    try:
        if args.mode == "author":
            from llm_script_creation.admin_agent_adapter.runner import run_authoring

            response = asyncio.run(run_authoring(request))
        else:
            from llm_script_creation.admin_agent_adapter.sandbox import execute_extractor

            samples_path = request.get("samples_path")
            if samples_path:
                response = execute_extractor(
                    str(request.get("extractor_source") or ""),
                    _iter_jsonl(Path(str(samples_path))),
                    request.get("score"),
                )
            else:
                response = execute_extractor(
                    str(request.get("extractor_source") or ""),
                    list(request.get("samples") or []),
                    request.get("score"),
                )
    except Exception as exc:  # noqa: BLE001 - report the concrete failure to the caller
        response = {
            "mode": args.mode,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-8000:],
        }
        _write(response_path, response)
        print(f"sidecar {args.mode} error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _write(response_path, response)
    print(
        f"sidecar {args.mode}: status={response.get('status')} -> {response_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

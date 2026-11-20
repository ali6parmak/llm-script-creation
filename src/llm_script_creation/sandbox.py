"""Validate and execute LLM-generated extraction scripts in a subprocess."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .configuration import WORK_DIR

ALLOWED_IMPORTS = {
    "__future__",
    "_strptime",
    "calendar",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "functools",
    "html",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
    "time",
    "typing",
    "unicodedata",
}
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
BLOCKED_MODULES = {
    "builtins",
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "pty",
    "requests",
    "shutil",
    "smtplib",
    "socket",
    "subprocess",
    "sys",
    "telnetlib",
    "tempfile",
    "urllib",
    "webbrowser",
}

RUNNER_SOURCE = r"""
import contextlib
import importlib.util
import json
import resource
import signal
import sys
import traceback

BLOCKED = %(blocked)r

class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError("import blocked by sandbox: " + fullname)
        return None

class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False

def main():
    script_path, payload_path = sys.argv[1], sys.argv[2]
    resource.setrlimit(resource.RLIMIT_AS, (%(memory_mb)d * 1024 * 1024, %(memory_mb)d * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (%(cpu_seconds)d, %(cpu_seconds)d))
    signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("script call timed out")))
    spec = importlib.util.spec_from_file_location("generated_extractor", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extract = getattr(module, "extract")
    sys.meta_path.insert(0, ImportGuard())
    with open(payload_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                payload = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr), TimeLimit(%(call_seconds)d):
                    output = extract(payload)
                print(json.dumps({"index": index, "ok": True, "output": output}, ensure_ascii=False, default=str))
            except Exception:
                print(json.dumps({"index": index, "ok": False, "error": traceback.format_exc(limit=4)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
"""


def validate_script(source: str) -> list[str]:
    """Return static validation problems (empty means valid enough to run)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    issues: list[str] = []
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if "extract" not in functions:
        issues.append("missing top-level function `extract(document)`")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    issues.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                issues.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_NAMES:
                issues.append(f"call not allowed: {node.func.id}()")
        elif isinstance(node, ast.Attribute):
            root = _root_name(node)
            if root in BLOCKED_MODULES:
                issues.append(f"module access not allowed: {root}.*")
    return sorted(set(issues))


def _root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _runner_source() -> str:
    return RUNNER_SOURCE % {
        "blocked": sorted(BLOCKED_MODULES),
        "memory_mb": 1024,
        "cpu_seconds": 60,
        "call_seconds": 10,
    }


def _run_chunk(
    script_path: Path,
    payloads: list[dict[str, Any]],
    directory: Path,
    timeout: int,
) -> list[dict[str, Any]]:
    runner_path = directory / "runner.py"
    payload_path = directory / "payloads.jsonl"
    if not runner_path.exists():
        runner_path.write_text(_runner_source(), encoding="utf-8")
    payload_path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False, default=str) for payload in payloads)
        + "\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(directory),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
    }
    command = [sys.executable, "-I", str(runner_path), str(script_path), str(payload_path)]
    try:
        completed = subprocess.run(
            command,
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [
            {"index": index, "ok": False, "error": f"script timed out after {timeout}s"}
            for index in range(len(payloads))
        ]

    results: dict[int, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            results[item["index"]] = item
    stderr = (completed.stderr or "").strip()
    for index in range(len(payloads)):
        if index not in results:
            detail = stderr[-1500:] if stderr else "no output from script runner"
            results[index] = {"index": index, "ok": False, "error": detail}
    return [results[index] for index in range(len(payloads))]


def iter_results(
    script_source: str,
    payloads: Iterable[dict[str, Any]],
    *,
    chunk_size: int = 20,
    timeout: int = 180,
) -> Iterator[dict[str, Any]]:
    """Run `extract` over payloads, yielding one result per payload.

    Each chunk starts a fresh subprocess so a bad document cannot take down the
    whole run and so only a bounded amount of text is written to disk at once.
    """
    with tempfile.TemporaryDirectory(dir=WORK_DIR) as temporary:
        directory = Path(temporary)
        script_path = directory / "script.py"
        script_path.write_text(script_source, encoding="utf-8")
        chunk: list[dict[str, Any]] = []
        for payload in payloads:
            chunk.append(payload)
            if len(chunk) >= chunk_size:
                yield from _run_chunk(script_path, chunk, directory, timeout)
                chunk = []
        if chunk:
            yield from _run_chunk(script_path, chunk, directory, timeout)

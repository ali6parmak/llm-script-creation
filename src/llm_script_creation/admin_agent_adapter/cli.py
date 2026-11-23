"""``benchmark-admin-agent``: evaluate the real admin agent's extractor strategy.

Flow (nothing is written to Mongo or Uwazi):

1. read the catalog (dataset spec + train/test samples) and the real Uwazi
   template from the local restored database (read-only);
2. write an AUTHOR request containing the train documents ONLY, run the
   sidecar with the admin agent's venv, and receive the authored extractor;
3. write an EXECUTE request with train + test documents, run the sidecar again,
   and receive one prediction per document (plus the admin agent's own scoring
   over test);
4. score with our Plan evaluator and write method/run artifacts.

Phase 1 covers single Plan fields (``plan_year`` first).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import ObjectId

from .. import catalog, runs
from .. import evaluation as evaluation_module
from .. import normalization as normalization_module
from ..configuration import CATALOG_PATH, WORK_DIR
from ..db import PLAN_DOCUMENT_TEMPLATE, UPR_DOCUMENT_TEMPLATE, get_db
from ..normalization import load_states
from . import evaluate, prompt, records

TYPE = "admin_agent_extractor"
DOCUMENT_TEMPLATE_BY_DB = {"plan": PLAN_DOCUMENT_TEMPLATE, "upr": UPR_DOCUMENT_TEMPLATE}

# Benchmark field -> real Uwazi property. Most fields share a name; UPR state is
# a relationship property called state_reviewed while our dataset gold calls it
# state_under_review.
PROPERTY_BY_FIELD = {"state_under_review": "state_reviewed"}

#: No-title condition. Title is removed from every entity view the extractor
#: sees: AgentEntity.title, raw entities, query summaries and ctx["title"].
#: A titled (or hybrid) variant can be added later as a separate method.
TITLE_VISIBLE = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() or None
    except OSError:
        return None


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def _package_version() -> str:
    try:
        return importlib.metadata.version("llm-script-creation")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _evaluation_version() -> dict[str, str | None]:
    return {
        "evaluation": _file_digest(Path(evaluation_module.__file__)),
        "normalization": _file_digest(Path(normalization_module.__file__)),
    }


def _spec_digest(core: dict[str, Any]) -> str:
    """Stable digest of a method spec's reproducibility-critical content."""
    payload = json.dumps(core, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _load_dataset(conn: Any, dataset_id: str) -> dict[str, Any]:
    row = catalog.get_dataset(conn, dataset_id)
    if row is None:
        raise SystemExit(f"Unknown dataset: {dataset_id}")
    data = dict(row)
    data["manifest"] = records.load_json(data.get("manifest_json"), {})
    data["target_schema"] = records.load_json(data.get("target_schema_json"), {})
    if data["db"] not in DOCUMENT_TEMPLATE_BY_DB:
        raise SystemExit(f"Unknown database for dataset {dataset_id!r}: {data['db']!r}")
    # The catalog keeps the dataset spec but not its template id; the document
    # template is a per-database constant in db.py.
    data["template"] = DOCUMENT_TEMPLATE_BY_DB[data["db"]]
    return data


def _load_template(dataset: dict[str, Any]) -> dict[str, Any]:
    """Read the real template from Mongo (read-only) and keep it JSON-safe."""
    template_id = dataset["template"]
    raw = get_db(dataset["db"]).templates.find_one({"_id": ObjectId(template_id)})
    if raw is None:
        raise SystemExit(f"Template {template_id!r} not found in db {dataset['db']!r}")
    properties = [
        {
            "name": prop.get("name") or "",
            "label": prop.get("label") or "",
            "type": prop.get("type") or "text",
            "required": bool(prop.get("required")),
            "filter": bool(prop.get("filter")),
            "showInCard": bool(prop.get("showInCard")),
            "content": prop.get("content"),
        }
        for prop in raw.get("properties") or []
    ]
    return {
        "name": raw.get("name") or "",
        "template_id": template_id,
        "color": raw.get("color") or "",
        "properties": properties,
    }


def _rows(conn: Any, dataset_id: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows = sorted(
        (dict(row) for row in catalog.iter_samples(conn, dataset_id, split=split)),
        key=lambda row: row["doc_id"],
    )
    return rows[:limit] if limit is not None else rows


def _record(
    conn: Any,
    dataset: dict[str, Any],
    sample: dict[str, Any],
    template: dict[str, Any],
    title_visible: bool | None = None,
) -> dict[str, Any] | None:
    doc_row = catalog.get_document(conn, dataset["db"], sample["doc_id"])
    if doc_row is None:
        return None
    doc_row = dict(doc_row)
    segments = records.load_json(doc_row.get("segments_json"), [])
    if not segments:
        return None
    filename = doc_row.get("filename") or f"{doc_row['doc_id']}.pdf"
    return {
        "doc_id": doc_row["doc_id"],
        "shared_id": doc_row["doc_id"],
        "split": sample["split"],
        "title": (
            (doc_row.get("title") or "")
            if (TITLE_VISIBLE if title_visible is None else title_visible)
            else ""
        ),
        "template_id": dataset["template"],
        "template_name": template["name"],
        "language": doc_row.get("language") or "en",
        "metadata": records.load_json(doc_row.get("metadata_json"), {}),
        "segments": segments,
        "file_id": doc_row.get("file_id"),
        "filename": filename,
        "originalname": filename,
        "mimetype": doc_row.get("mimetype") or "application/pdf",
        "gold": records.load_json(sample.get("gold_json"), {}),
    }


def _allowed_values(property_type: str, train_records: list[dict[str, Any]], field: str) -> list[str] | None:
    if property_type != "select":
        return None
    values = {
        str(gold[field])
        for record in train_records
        for gold in [record.get("gold") or {}]
        if gold.get(field) is not None and str(gold[field]).strip()
    }
    return sorted(values)


def _candidate_values(
    dataset_id: str, property_type: str, train_records: list[dict[str, Any]], field: str
) -> list[str] | None:
    """Candidate labels the task may show. UPR uses the full Uwazi State list."""
    if dataset_id in {"upr_state", "upr_recommendations"}:
        return [str(state["label"]) for state in load_states()]
    return _allowed_values(property_type, train_records, field)


def _prompt_extras(dataset_id: str) -> dict[str, str]:
    if dataset_id == "upr_state":
        return {
            "value_shape": 'the Uwazi State label verbatim, e.g. "Afghanistan" (not the sharedId)',
            "task_hint": (
                "The state under review is printed on the first page (header/title). "
                "Return the State label; document spellings may differ from the Uwazi label."
            ),
            "candidate_block_title": (
                "CANDIDATE STATE LABELS (Uwazi State entities; use the label verbatim)"
            ),
        }
    return {}


def _target_properties(dataset_id: str, field: str) -> set[str]:
    """Metadata properties that must be absent when the extractor RUNS.

    The value being extracted is empty in production, so leaving the gold in
    ``ctx.metadata`` would be a trivial shortcut. UPR targets are relationship
    properties with different names from our benchmark field.
    """
    return {
        "upr_state": {"state_reviewed"},
        "upr_recommendations": {"related_recommendations", "state_reviewed"},
    }.get(dataset_id, {field})


def _write_execute_sample(
    handle: Any,
    record: dict[str, Any],
    redact_properties: set[str] | None = None,
    drop_metadata: bool = False,
) -> None:
    """Write one payload line (everything except our gold wrapper)."""
    payload = {key: value for key, value in record.items() if key != "gold"}
    if drop_metadata:
        payload["metadata"] = {}
    elif redact_properties:
        payload["metadata"] = {
            key: value
            for key, value in (payload.get("metadata") or {}).items()
            if key not in redact_properties
        }
    handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _admin_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo = _repo_root()
    admin_src = Path(
        env.get("UWAZI_ADMIN_SRC") or (Path.home() / "projects" / "python_uwazi_API" / "src")
    )
    paths = [str(repo / "src"), str(admin_src)]
    extra = WORK_DIR / "admin_agent_extra"
    if extra.exists():
        # The admin repo's .venv predates rapidfuzz/python-dateutil in its
        # pyproject. We must not modify the read-only repo, so the locked deps
        # live in a sidecar-only target dir instead.
        paths.append(str(extra))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _admin_python() -> Path:
    raw = os.environ.get("UWAZI_ADMIN_PYTHON")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "projects" / "python_uwazi_API" / ".venv" / "bin" / "python"


def _preflight() -> None:
    """Fail with an actionable message instead of a cryptic sidecar import error."""
    python = _admin_python()
    if not python.exists():
        raise SystemExit(
            f"Admin agent venv not found at {python}. Set UWAZI_ADMIN_PYTHON or run "
            "`uv sync` in ~/projects/python_uwazi_API."
        )
    admin_src = Path(
        os.environ.get("UWAZI_ADMIN_SRC")
        or (Path.home() / "projects" / "python_uwazi_API" / "src")
    )
    if not admin_src.exists():
        raise SystemExit(f"Admin agent source not found at {admin_src}. Set UWAZI_ADMIN_SRC.")
    check = subprocess.run(
        [str(python), "-c", "import rapidfuzz, dateutil"],
        env=_admin_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise SystemExit(
            "The admin agent venv cannot import its extraction dependencies "
            "(rapidfuzz/python-dateutil). The admin repo is read-only here, so layer them "
            "into a sidecar-only directory instead:\n\n"
            "  export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}\n"
            f"  uv pip install --target {WORK_DIR / 'admin_agent_extra'} \\\n"
            f"    --python {python} \\\n"
            '    "rapidfuzz==3.14.6" "python-dateutil==2.9.0.post0"\n'
        )


def _invoke_sidecar(mode: str, request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Write a request JSON, run the sidecar with the admin venv, read the response."""
    script = _repo_root() / "scripts" / "admin_agent_sidecar.py"
    request_path = directory / f"{mode}_request.json"
    response_path = directory / f"{mode}_response.json"
    records.write_json(request_path, request)
    command = [str(_admin_python()), str(script), "--mode", mode, "--request", str(request_path), "--response", str(response_path)]
    timeout = float(os.environ.get("ADMIN_AGENT_TIMEOUT", "2400"))
    completed = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env=_admin_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise SystemExit(
            f"sidecar {mode} failed (exit {completed.returncode}).\n"
            f"request: {request_path}\nstderr tail:\n{tail}"
        )
    if not response_path.exists():
        raise SystemExit(f"sidecar {mode} produced no response at {response_path}")
    return records.read_json(response_path)


def _sidecar_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = dict(response.get("usage") or {})
    usage.setdefault("model", response.get("model"))
    usage.setdefault("base_url", response.get("base_url"))
    return usage


def run(
    dataset_id: str,
    limit_train: int | None,
    limit_test: int | None,
    include_country_list: bool = True,
    geometry: bool = False,
    no_metadata: bool = False,
    prompt_override: str | None = None,
    prompt_version_override: str | None = None,
    selected_ids: list[str] | None = None,
    max_requests_override: int | None = None,
    replay_of: str | None = None,
    method_suffix: str = "",
    sampling_policy: str = "agent_all_train",
    title_visible_override: bool | None = None,
    redact_target_metadata: bool = True,
    prompt_source_path: str | None = None,
) -> dict[str, Any]:
    if geometry and not include_country_list:
        raise SystemExit("--geometry with --no-country-list is not a supported combination.")
    title_visible = TITLE_VISIBLE if title_visible_override is None else title_visible_override
    conn = catalog.connect()
    try:
        dataset = _load_dataset(conn, dataset_id)
        template = _load_template(dataset)
        fields = list(dataset["target_schema"])
        is_records = dataset_id == "upr_recommendations"
        if not is_records and len(fields) != 1:
            raise SystemExit(f"This adapter supports single-field datasets, got {fields}")
        field = fields[0]
        if is_records:
            property_name = "recommendations"
            property_type = "record_extraction"
        else:
            property_name = PROPERTY_BY_FIELD.get(field, field)
            properties = {prop["name"]: prop for prop in template["properties"]}
            if property_name not in properties:
                raise SystemExit(
                    f"Field {field!r} maps to property {property_name!r}, which is not on "
                    f"template {template['name']!r}"
                )
            property_type = properties[property_name]["type"]

        train_rows = _rows(conn, dataset_id, "train", limit_train)
        if selected_ids:
            order = {doc_id: index for index, doc_id in enumerate(selected_ids)}
            train_rows = sorted(
                (row for row in train_rows if row["doc_id"] in order),
                key=lambda row: order[row["doc_id"]],
            )
        test_rows = _rows(conn, dataset_id, "test", limit_test)
        train_records = [
            r
            for r in (
                _record(conn, dataset, row, template, title_visible=title_visible)
                for row in train_rows
            )
            if r
        ]
        if geometry:
            train_records = [records.with_geometry(record) for record in train_records]
        if not train_records:
            raise SystemExit("No train documents with segmentation found.")
        if not test_rows:
            raise SystemExit("No test documents with segmentation found.")
    finally:
        conn.close()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    directory = WORK_DIR / "admin_agent" / f"{stamp}_{runs.slug(dataset_id)}"
    directory.mkdir(parents=True, exist_ok=True)

    redact_properties = (
        _target_properties(dataset_id, field) if redact_target_metadata else set()
    )
    if prompt_override is not None:
        task_prompt = prompt_override
        prompt_version = prompt_version_override or prompt.PROMPT_VERSION
    elif is_records:
        prompt_version = (
            prompt.PROMPT_VERSION_RECOMMENDATIONS
            if include_country_list
            else prompt.PROMPT_VERSION_RECOMMENDATIONS_NO_LIST
        )
        if geometry:
            prompt_version += "-geometry"
        if no_metadata:
            prompt_version += "-no-metadata"
        task_prompt = prompt.build_recommendations_task(
            dataset_id=dataset_id,
            template_name=template["name"],
            train_total=len(train_records),
            state_labels=_candidate_values("upr_state", "relationship", train_records, field) or [],
            title_visible=title_visible,
            target_metadata_hidden=True,
            include_state_list=include_country_list,
            geometry=geometry,
            metadata_visible=not no_metadata,
        )
    else:
        prompt_version = prompt.PROMPT_VERSION
        if geometry:
            prompt_version += "-geometry"
        if no_metadata:
            prompt_version += "-no-metadata"
        task_prompt = prompt.build_author_task(
            dataset_id=dataset_id,
            property_name=property_name,
            property_type=property_type,
            template_name=template["name"],
            train_total=len(train_records),
            allowed_values=_candidate_values(dataset_id, property_type, train_records, field),
            title_visible=title_visible,
            target_metadata_hidden=True,
            geometry=geometry,
            metadata_visible=not no_metadata,
            **_prompt_extras(dataset_id),
        )
    max_requests = max_requests_override or int(os.environ.get("ADMIN_AGENT_MAX_REQUESTS", "40"))
    author_request = {
        "mode": "author",
        "dataset_id": dataset_id,
        "max_requests": max_requests,
        "task_prompt": task_prompt,
        "template": template,
        "train_entities": [
            {key: value for key, value in record.items() if key != "gold"} for record in train_records
        ],
    }
    print(f"Authoring via admin agent on {len(train_records)} train documents …", flush=True)
    author = _invoke_sidecar("author", author_request, directory)
    if author.get("status") != "ok":
        print(json.dumps({k: author.get(k) for k in ("status", "error", "validation", "model")}, indent=2))
        raise SystemExit(f"Admin agent did not return a valid extractor (see {directory})")
    extractor = author["extractor"]
    code = extractor["python_code"]
    print(
        f"  model {author.get('model')} · requests {author.get('usage', {}).get('requests')} · "
        f"tokens {author.get('usage', {}).get('total_tokens')} · "
        f"samples {extractor.get('samples_matched')}/{extractor.get('samples_total')}",
        flush=True,
    )

    # The full Plan test split does not fit in one JSON request, so payloads are
    # streamed as JSONL: one document per line, built from the catalog one row at
    # a time and never held all in memory. Test documents appear ONLY here.
    execute_path = directory / "execute_samples.jsonl"
    expected: dict[str, dict[str, Any]] = {}
    sample_meta: list[dict[str, Any]] = []
    for record in train_records:
        expected[record["doc_id"]] = record["gold"]
        sample_meta.append(
            {"doc_id": record["doc_id"], "split": record["split"], "gold": record["gold"]}
        )
    test_count = 0
    print(f"Preparing execution payloads for {len(test_rows):,} test rows …", flush=True)
    exec_conn = catalog.connect()
    try:
        with execute_path.open("w", encoding="utf-8") as handle:
            for record in train_records:
                _write_execute_sample(handle, record, redact_properties, drop_metadata=no_metadata)
            for row in test_rows:
                record = _record(
                    exec_conn, dataset, row, template, title_visible=title_visible
                )
                if record is None:
                    continue
                if geometry:
                    record = records.with_geometry(record)
                _write_execute_sample(handle, record, redact_properties, drop_metadata=no_metadata)
                expected[record["doc_id"]] = record["gold"]
                sample_meta.append(
                    {"doc_id": record["doc_id"], "split": "test", "gold": record["gold"]}
                )
                test_count += 1
                if test_count % 1000 == 0:
                    print(f"  prepared {test_count:,} test documents", flush=True)
    finally:
        exec_conn.close()
    if not test_count:
        raise SystemExit("No test documents with segmentation found.")

    prediction_key_map = (
        {"state_reviewed": "state_under_review"}
        if is_records
        else ({property_name: field} if property_name != field else {})
    )
    score_property_names = ["state_under_review"] if is_records else [field]
    execute_request = {
        "mode": "execute",
        "extractor_source": code,
        "samples_path": str(execute_path),
        "prediction_key_map": prediction_key_map,
        "score": {
            "property_names": score_property_names,
            "expected": expected,
            "policy": "normalized",
        },
    }
    print(
        f"Executing extractor on {len(train_records)} train + {test_count:,} test documents …",
        flush=True,
    )
    execution = _invoke_sidecar("execute", execute_request, directory)
    if execution.get("status") != "ok":
        raise SystemExit(f"Extractor execution failed: {execution.get('error')}")

    predictions = {item["doc_id"]: item for item in execution.get("predictions") or []}
    samples = sample_meta
    scored_records, metrics = evaluate.score_records(dataset_id, fields, samples, predictions)
    usage = _sidecar_usage(author)
    metrics["token_usage"] = usage
    metrics["admin_eval"] = execution.get("admin_eval")
    metrics["admin_agent"] = {
        "model": author.get("model"),
        "base_url": author.get("base_url"),
        "extractor_samples_total": extractor.get("samples_total"),
        "extractor_samples_matched": extractor.get("samples_matched"),
        "unmatched_notes": extractor.get("unmatched_notes"),
        "description": extractor.get("description"),
        "usage": usage,
        "tool_calls": author.get("tool_calls"),
    }
    metrics["method_notes"] = (
        "No-title condition: ctx['title'] and entity titles are empty; "
        + ("metadata is also EMPTY at execution" if no_metadata else "other metadata is visible")
        + ". The target property is removed from ctx.metadata at execution time. "
        "Authored on train documents only; executed once on the untouched test split."
    )

    now = datetime.now(UTC).isoformat()
    model_slug = runs.slug(str(author.get("model") or "unknown"))
    prompt_slug = runs.slug(prompt_version).removeprefix("admin_agent_")
    selected_shared_ids = [record["doc_id"] for record in train_records]
    candidate_count = (
        len(_candidate_values(dataset_id, property_type, train_records, field) or [])
        if include_country_list
        else 0
    )
    spec_core = {
        "spec_version": 1,
        "adapter_version": _package_version(),
        "git_commit": _git_commit(),
        "evaluation_version": _evaluation_version(),
        "dataset_id": dataset_id,
        "db": dataset["db"],
        "task_kind": "records" if is_records else "properties",
        "fields": fields,
        "benchmark_field": field,
        "benchmark_fields": fields,
        "property_name": property_name,
        "property_type": property_type,
        "model": author.get("model"),
        "base_url": author.get("base_url"),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(task_prompt.encode()).hexdigest(),
        "prompt_chars": len(task_prompt),
        "prompt_source": {
            "kind": "file" if prompt_source_path else "builder",
            "path": prompt_source_path,
            "label": prompt_version,
        },
        "condition": {
            "title_visible": title_visible,
            "no_metadata": no_metadata,
            "metadata_visible": not no_metadata,
            "geometry": geometry,
            "target_property_visible_at_execution": not redact_target_metadata,
            "target_properties_redacted": sorted(redact_properties),
        },
        "sampling": {
            "policy": sampling_policy,
            "selected_shared_ids": selected_shared_ids,
            "authoring_documents": len(train_records),
            "limit_train": limit_train,
            "limit_test": limit_test,
        },
        "candidate_list": {
            "included": include_country_list
            and dataset_id in {"upr_state", "upr_recommendations"},
            "count": candidate_count,
        },
        "execution": {
            "property_names": score_property_names,
            "prediction_key_map": prediction_key_map,
            "score_policy": "normalized",
        },
        "template": {"name": template["name"], "id": dataset["template"]},
        "split_policy_version": dataset["manifest"].get("split_policy_version"),
        "max_requests": max_requests,
    }
    spec_hash = _spec_digest(spec_core)
    base_method_id = f"admin_agent_{dataset_id}_{model_slug}_{prompt_slug}_{spec_hash[:8]}"
    method_id = f"{base_method_id}{method_suffix}"
    if (runs.METHODS_DIR / method_id).exists():
        method_id = f"{method_id}-r{stamp}"
        spec_core["rerun_of"] = base_method_id
    spec = {
        **spec_core,
        "spec_hash": spec_hash,
        "method_id": method_id,
        "created_at": now,
    }
    if replay_of:
        spec["replay_of"] = replay_of
    config = {
        "condition": "no_title",
        "prompt_version": prompt_version,
        "title_visible": title_visible,
        "target_property_visible_at_execution": not redact_target_metadata,
        "target_properties_redacted": sorted(redact_properties),
        "source": "primary-document segmentation",
        "template_name": template["name"],
        "template_id": dataset["template"],
        "property_name": property_name,
        "benchmark_field": field,
        "benchmark_fields": fields,
        "property_type": property_type,
        "candidate_values": candidate_count,
        "spec_hash": spec_hash,
        "sampling_policy": sampling_policy,
        "selected_train_documents": len(selected_shared_ids),
        "geometry": geometry,
        "no_metadata": no_metadata,
        "metadata_visible": not no_metadata,
        "input_representation": "text+geometry" if geometry else "text",
        "admin_agent_model": author.get("model"),
        "admin_agent_base_url": author.get("base_url"),
        "max_requests": author_request["max_requests"],
        "extractor_samples_matched": extractor.get("samples_matched"),
        "extractor_samples_total": extractor.get("samples_total"),
        "authoring_documents": len(train_records),
        "execution_documents": len(train_records) + test_count,
        "request_files": str(directory),
    }
    method = evaluate.build_method(
        method_id=method_id,
        name=f"Admin agent extractor — {dataset_id} ({prompt_version})",
        description=(
            "The real Uwazi admin agent's nested extractor subagent authors `def extract(text, ctx)` "
            "from primary-document segmentation on the train split (NO-TITLE condition: entity "
            "titles are hidden, other metadata visible, and the target property is removed from "
            "ctx.metadata at execution). The frozen function runs once over train and test, scored "
            "against real Uwazi metadata with our dataset evaluator."
        ),
        dataset_id=dataset_id,
        task=dataset["task"],
        model=str(author.get("model") or ""),
        extractor_code=code,
        prompt=task_prompt,
        manifest=dataset["manifest"],
        config=config,
        usage=usage,
        primary=metrics["primary"],
        train_documents=len(train_records),
        test_documents=test_count,
        created_at=now,
        prompt_version=prompt_version,
    )
    method["input_spec"] = {
        "representation": "text",
        "source": "primary-document segmentation",
        "geometry": geometry,
    }
    method["spec_hash"] = spec_hash
    attempt = {
        "round": 1,
        "kind": "admin_agent_author",
        "model": author.get("model"),
        "usage": usage,
        "validation_issues": [] if not author.get("validation") else [author["validation"]],
        "train_score": metrics["primary"]["train"],
        "train_nonexample_score": None,
        "code": code,
        "response": extractor.get("description") or "",
        "transcript": author.get("transcript") or [],
        "extractor": {
            key: value for key, value in extractor.items() if key != "python_code"
        },
        "empty_response": bool(author.get("empty_output")),
        "selected_shared_ids": selected_shared_ids,
    }
    runs.write_method(
        method,
        prompt=task_prompt,
        script=code,
        attempts=[attempt],
        spec=spec,
    )

    run_id = runs.new_run_id(method_id, dataset_id, "train_test")
    run = evaluate.build_run(
        run_id=run_id,
        method_id=method_id,
        dataset_id=dataset_id,
        status="completed",
        created_at=now,
        counts=metrics["status_counts"]["test"],
        metrics=metrics,
        manifest=dataset["manifest"],
        usage=usage,
    )
    runs.write_run(run, scored_records, metrics, prompt=task_prompt, script=code)

    primary = metrics["primary"]
    admin_eval = execution.get("admin_eval") or {}
    print(f"\nRun: {run_id}")
    print(
        f"  train {primary['train']:.3f} · test {primary['test']:.3f} · gap {primary['gap']:+.3f} · "
        f"test statuses {metrics['status_counts']['test']}",
        flush=True,
    )
    if admin_eval:
        accuracy = admin_eval.get("accuracy")
        print(
            "  admin-agent eval: "
            + ("n/a" if accuracy is None else f"accuracy {100 * accuracy:.1f}%")
            + f" · coverage {100 * (admin_eval.get('coverage') or 0):.1f}%"
            + f" · mismatched {admin_eval.get('mismatched')} · missing {admin_eval.get('missing')}",
            flush=True,
        )
    print(f"  requests: {directory}", flush=True)
    return {"run": run, "metrics": metrics, "method": method, "request_dir": str(directory)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="dataset id, e.g. plan_year (required unless --replay is used)",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="limit train documents (spike)")
    parser.add_argument("--limit-test", type=int, default=None, help="limit test documents (spike)")
    parser.add_argument(
        "--no-country-list",
        action="store_true",
        help="UPR recommendations only: do not put the 195 candidate State labels in the task.",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Prefix every paragraph with [pPAGE x=… y=… w=… h=…] (position/size condition).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Drop ctx.metadata entirely at execution time (clean text-only condition).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Use a stored task prompt file instead of the current prompt builder.",
    )
    parser.add_argument(
        "--prompt-label",
        default=None,
        help="Prompt version label for --prompt-file (e.g. admin-agent-v1-ids).",
    )
    parser.add_argument(
        "--with-title",
        action="store_true",
        help="Title-visible condition (old v1/v2 runs).",
    )
    parser.add_argument(
        "--target-metadata-visible",
        action="store_true",
        help="Do not remove the target property from execution metadata (old v1/v2 runs).",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Replay a method from its spec.json (see --replay-mode).",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["execute", "author"],
        default="execute",
        help="execute: run the stored frozen script; author: re-author from the stored prompt.",
    )
    args = parser.parse_args(argv)

    if not CATALOG_PATH.exists():
        raise SystemExit("Catalog missing. Run `uv run build-datasets` first.")
    _preflight()
    if args.replay:
        if args.dataset:
            raise SystemExit("Use either --dataset (new run) or --replay (stored method), not both.")
        if args.replay_mode == "author":
            result = replay_author(args.replay)
            print(f"\nMethod: {result['method']['method_id']}", file=sys.stderr)
        else:
            result = replay_execute(args.replay)
            print(f"\nRun: {result['run']['run_id']}", file=sys.stderr)
        return 0
    if not args.dataset:
        raise SystemExit("--dataset is required unless --replay is used.")
    prompt_text = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        prompt_text = prompt_path.read_text(encoding="utf-8")
    result = run(
        args.dataset,
        args.limit_train,
        args.limit_test,
        include_country_list=not args.no_country_list,
        geometry=args.geometry,
        no_metadata=args.no_metadata,
        prompt_override=prompt_text,
        prompt_version_override=(
            args.prompt_label or "legacy-prompt" if prompt_text is not None else None
        ),
        title_visible_override=True if args.with_title else None,
        redact_target_metadata=not args.target_metadata_visible,
        prompt_source_path=args.prompt_file,
    )
    print(f"\nMethod: {result['method']['method_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _load_spec(method_id: str) -> dict[str, Any]:
    path = runs.METHODS_DIR / method_id / "spec.json"
    if not path.exists():
        raise SystemExit(
            f"No spec.json for method {method_id!r}. Methods created before self-contained "
            "specs cannot be replayed; re-run them with the current CLI to create one."
        )
    spec = records.read_json(path)
    if not isinstance(spec, dict) or "spec_version" not in spec:
        raise SystemExit(f"Invalid spec.json for method {method_id!r}")
    return spec


def _method_script(method_id: str) -> str:
    path = runs.METHODS_DIR / method_id / "script.py"
    if not path.exists():
        raise SystemExit(f"No script.py for method {method_id!r}")
    return path.read_text(encoding="utf-8")


def replay_execute(method_id: str) -> dict[str, Any]:
    """Run a stored method's frozen script under its recorded condition.

    Deterministic: same script, same test split, same geometry/metadata
    condition. A new run is added under the method; the method is not modified.
    """
    spec = _load_spec(method_id)
    dataset_id = str(spec["dataset_id"])
    fields = list(spec["benchmark_fields"])
    condition = spec.get("condition") or {}
    geometry = bool(condition.get("geometry"))
    no_metadata = bool(condition.get("no_metadata"))
    redact_properties = set(condition.get("target_properties_redacted") or [])
    limit_test = (spec.get("sampling") or {}).get("limit_test")
    selected_ids = list((spec.get("sampling") or {}).get("selected_shared_ids") or [])

    conn = catalog.connect()
    try:
        dataset = _load_dataset(conn, dataset_id)
        template = _load_template(dataset)
        train_rows = _rows(conn, dataset_id, "train", None)
        order = {doc_id: index for index, doc_id in enumerate(selected_ids)}
        train_rows = sorted(
            (row for row in train_rows if row["doc_id"] in order),
            key=lambda row: order[row["doc_id"]],
        )
        test_rows = _rows(conn, dataset_id, "test", limit_test)
    finally:
        conn.close()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    directory = WORK_DIR / "admin_agent" / f"replay_{stamp}_{runs.slug(method_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    execute_path = directory / "execute_samples.jsonl"
    expected: dict[str, dict[str, Any]] = {}
    sample_meta: list[dict[str, Any]] = []
    test_count = 0
    exec_conn = catalog.connect()
    try:
        with execute_path.open("w", encoding="utf-8") as handle:
            for row in train_rows:
                record = _record(
                    exec_conn,
                    dataset,
                    row,
                    template,
                    title_visible=bool(condition.get("title_visible")),
                )
                if record is None:
                    continue
                if geometry:
                    record = records.with_geometry(record)
                _write_execute_sample(
                    handle, record, redact_properties, drop_metadata=no_metadata
                )
                expected[record["doc_id"]] = record["gold"]
                sample_meta.append(
                    {"doc_id": record["doc_id"], "split": "train", "gold": record["gold"]}
                )
            for row in test_rows:
                record = _record(
                    exec_conn,
                    dataset,
                    row,
                    template,
                    title_visible=bool(condition.get("title_visible")),
                )
                if record is None:
                    continue
                if geometry:
                    record = records.with_geometry(record)
                _write_execute_sample(
                    handle, record, redact_properties, drop_metadata=no_metadata
                )
                expected[record["doc_id"]] = record["gold"]
                sample_meta.append(
                    {"doc_id": record["doc_id"], "split": "test", "gold": record["gold"]}
                )
                test_count += 1
    finally:
        exec_conn.close()
    if not test_count:
        raise SystemExit("No test documents with segmentation found.")

    execute_request = {
        "mode": "execute",
        "extractor_source": _method_script(method_id),
        "samples_path": str(execute_path),
        "prediction_key_map": (spec.get("execution") or {}).get("prediction_key_map") or {},
        "score": {
            "property_names": (spec.get("execution") or {}).get("property_names") or fields,
            "expected": expected,
            "policy": "normalized",
        },
    }
    execution = _invoke_sidecar("execute", execute_request, directory)
    if execution.get("status") != "ok":
        raise SystemExit(f"Extractor execution failed: {execution.get('error')}")
    predictions = {item["doc_id"]: item for item in execution.get("predictions") or []}
    scored_records, metrics = evaluate.score_records(dataset_id, fields, sample_meta, predictions)
    usage = {"replay": "execute"}
    metrics["token_usage"] = usage
    metrics["admin_eval"] = execution.get("admin_eval")
    metrics["replay"] = {
        "mode": "execute",
        "method_id": method_id,
        "spec_hash": spec.get("spec_hash"),
    }

    now = datetime.now(UTC).isoformat()
    run_id = runs.new_run_id(method_id, dataset_id, "replay_execute")
    run = evaluate.build_run(
        run_id=run_id,
        method_id=method_id,
        dataset_id=dataset_id,
        status="completed",
        created_at=now,
        counts=metrics["status_counts"]["test"],
        metrics=metrics,
        manifest=dataset["manifest"],
        usage=usage,
    )
    prompt_path = runs.METHODS_DIR / method_id / "prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None
    runs.write_run(
        run,
        scored_records,
        metrics,
        prompt=prompt_text,
        script=_method_script(method_id),
    )
    primary = metrics["primary"]
    print(f"\nReplay run: {run_id}")
    print(
        f"  train {primary['train']:.3f} · test {primary['test']:.3f} · "
        f"test statuses {metrics['status_counts']['test']}",
        flush=True,
    )
    return {"run": run, "metrics": metrics, "request_dir": str(directory)}


def replay_author(method_id: str) -> dict[str, Any]:
    """Re-author from a stored prompt + the same selected train documents.

    The new extractor is stochastic, so it is written as a NEW method (suffix
    ``-r<timestamp>``) rather than overwriting the replayed method.
    """
    spec = _load_spec(method_id)
    prompt_path = runs.METHODS_DIR / method_id / "prompt.txt"
    if not prompt_path.exists():
        raise SystemExit(f"No prompt.txt for method {method_id!r}")
    prompt_text = prompt_path.read_text(encoding="utf-8")
    sampling = spec.get("sampling") or {}
    condition = spec.get("condition") or {}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return run(
        str(spec["dataset_id"]),
        None,
        sampling.get("limit_test"),
        include_country_list=bool((spec.get("candidate_list") or {}).get("included", True)),
        geometry=bool(condition.get("geometry")),
        no_metadata=bool(condition.get("no_metadata")),
        prompt_override=prompt_text,
        prompt_version_override=str(spec.get("prompt_version") or prompt.PROMPT_VERSION),
        selected_ids=list(sampling.get("selected_shared_ids") or []),
        max_requests_override=int(spec.get("max_requests") or 40),
        replay_of=method_id,
        method_suffix=f"-r{stamp}",
        sampling_policy=f"replay:{sampling.get('policy', 'unknown')}",
        title_visible_override=bool(condition.get("title_visible")),
    )

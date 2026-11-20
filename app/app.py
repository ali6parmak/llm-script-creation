"""Streamlit viewer for datasets, samples, methods, runs and comparisons.

Run with::

    uv run streamlit run app/app.py
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

import pandas as pd
import streamlit as st
from theme import (
    PALETTE,
    THEMES,
    TYPE_COLORS,
    badge,
    inject_css,
    set_theme,
    status_badge,
    table_style,
    theme_names,
)

from llm_script_creation import catalog
from llm_script_creation import runs as run_artifacts
from llm_script_creation.configuration import ARTIFACTS_DIR, CATALOG_PATH
from llm_script_creation.db import document_from_row
from llm_script_creation.text import document_payload

st.set_page_config(page_title="Uwazi extraction benchmarks", layout="wide")

SPLITS = ["all", "train", "dev", "test"]
STATUS_OPTIONS = ["hit", "partial", "wrong", "missing_prediction", "error"]


# ---------------------------------------------------------------------------
# Loaders


@st.cache_data(ttl=30)
def load_datasets() -> list[dict[str, Any]]:
    conn = catalog.connect()
    try:
        return [dict(row) for row in catalog.list_datasets(conn)]
    finally:
        conn.close()


@st.cache_data(ttl=30)
def load_dataset(dataset_id: str) -> dict[str, Any] | None:
    conn = catalog.connect()
    try:
        row = catalog.get_dataset(conn, dataset_id)
        return dict(row) if row else None
    finally:
        conn.close()


@st.cache_data(ttl=30)
def load_samples(dataset_id: str) -> pd.DataFrame:
    conn = catalog.connect()
    try:
        rows = []
        for row in catalog.iter_samples(conn, dataset_id):
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "split": row["split"],
                    "pages": row["page_count"],
                    "segments": row["segment_count"],
                    "chars": row["char_count"],
                    "gold": json.loads(row["gold_json"]),
                }
            )
        return pd.DataFrame(rows)
    finally:
        conn.close()


@st.cache_data(ttl=30)
def load_document(db: str, doc_id: str) -> dict[str, Any] | None:
    conn = catalog.connect()
    try:
        row = catalog.get_document(conn, db, doc_id)
        return dict(row) if row else None
    finally:
        conn.close()


@st.cache_data(ttl=30)
def load_methods() -> list[dict[str, Any]]:
    return run_artifacts.list_methods()


@st.cache_data(ttl=30)
def load_run_index() -> list[dict[str, Any]]:
    return run_artifacts.list_runs()


@st.cache_data(ttl=60)
def load_predictions(run_id: str) -> dict[str, dict[str, Any]]:
    return {record["doc_id"]: record for record in run_artifacts.load_predictions(run_id)}


# ---------------------------------------------------------------------------
# Small helpers


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def show_json(value: Any, *, height: int | str | None = None) -> None:
    kwargs = {"height": height} if height is not None else {}
    st.code(
        json.dumps(value, indent=2, ensure_ascii=False),
        language="json",
        wrap_lines=True,
        **kwargs,
    )


def show_table(
    frame: pd.DataFrame,
    *,
    status: bool = False,
    metrics: list[str] | None = None,
    height: int | str = "auto",
    max_rows: int = 100,
) -> None:
    if frame.empty:
        st.caption("No rows.")
        return
    shown = frame.head(max_rows)
    st.table(
        table_style(shown, status_column="status" if status else None, metric_columns=metrics),
        width="stretch",
        height=height,
    )
    if len(frame) > max_rows:
        st.caption(f"Showing {max_rows:,} of {len(frame):,} rows.")


def paginate(frame: pd.DataFrame, page_size: int, key: str) -> pd.DataFrame:
    pages = max(1, math.ceil(len(frame) / page_size))
    page = st.selectbox(
        "Page",
        list(range(1, pages + 1)),
        index=0,
        key=f"page_{key}",
        format_func=lambda value: f"{value} / {pages}",
    )
    start = (int(page) - 1) * page_size
    return frame.iloc[start : start + page_size]


def metric_value(run: dict[str, Any] | None, metric: str) -> float | None:
    if not run:
        return None
    value = run.get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def primary_metric(
    dataset: dict[str, Any] | None, run: dict[str, Any] | None
) -> tuple[str, float | None]:
    if not dataset or not run:
        return ("—", None)
    manifest = json.loads(dataset.get("manifest_json") or "{}")
    metric = manifest.get("primary_metric") or "accuracy"
    value = metric_value(run, metric)
    if value is not None:
        return (metric, value)
    for fallback in ("accuracy", "f1", "macro_f1", "state_accuracy", "exact_match"):
        value = metric_value(run, fallback)
        if value is not None:
            return (fallback, value)
    return (metric, None)


def method_runs(method_id: str, run_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in run_index if run["method_id"] == method_id]


def latest_run(method_id: str, run_index: list[dict[str, Any]]) -> dict[str, Any] | None:
    runs = method_runs(method_id, run_index)
    return runs[0] if runs else None


def run_label(run: dict[str, Any]) -> str:
    return f"{run['run_id']} · {run['split']} · {run['status']}"


def split_badge(split: str) -> str:
    color = {
        "train": PALETTE["blue"],
        "dev": PALETTE["yellow"],
        "test": PALETTE["purple"],
    }.get(split, PALETTE["gray"])
    return badge(split, color)


def dataset_accent(row: dict[str, Any]) -> str:
    if row.get("db") == "upr":
        return PALETTE["blue"]
    return TYPE_COLORS.get(row.get("task", ""), PALETTE["yellow"])


def dataset_card(row: dict[str, Any], run_count: int) -> str:
    manifest = json.loads(row["manifest_json"])
    accent = dataset_accent(row)
    counts = manifest.get("counts", {})
    description = html.escape((row.get("description") or "")[:155])
    if len(row.get("description") or "") > 155:
        description += "…"
    return f"""
    <div class="uw-card" style="border-left-color:{accent}">
      <div class="uw-card-title">{html.escape(manifest.get("name", row["dataset_id"]))}</div>
      <div class="uw-card-id">{html.escape(row["dataset_id"])}</div>
      <div>{badge(row.get("task", ""), accent)}{badge(row.get("db", ""), PALETTE["gray"])}</div>
      <div class="uw-card-desc" style="margin-top:8px">{description}</div>
      <div class="uw-stat-row">
        <div class="uw-stat"><b>{counts.get("total", 0):,}</b><span>samples</span></div>
        <div class="uw-stat"><b>{counts.get("dev", 0):,}</b><span>dev</span></div>
        <div class="uw-stat"><b>{counts.get("test", 0):,}</b><span>test</span></div>
        <div class="uw-stat"><b>{run_count}</b><span>runs</span></div>
      </div>
    </div>
    """


def method_card(
    method: dict[str, Any], run_index: list[dict[str, Any]], dataset: dict[str, Any] | None
) -> str:
    color = TYPE_COLORS.get(method.get("type", ""), PALETTE["gray"])
    run = latest_run(method["method_id"], run_index)
    metric_name, value = primary_metric(dataset, run)
    model = method.get("model") or "—"
    description = html.escape((method.get("description") or "")[:150])
    return f"""
    <div class="uw-card" style="border-left-color:{color}">
      <div class="uw-card-title">{html.escape(method.get("name", method["method_id"]))}</div>
      <div class="uw-card-id">{html.escape(method["method_id"])}</div>
      <div>
        {badge(method.get("type", "method"), color)}
        {badge(method.get("dataset_id", ""), PALETTE["blue"])}
        {badge(model.replace(":cloud", ""), PALETTE["gray"])}
      </div>
      <div class="uw-card-desc" style="margin-top:8px">{description}</div>
      <div class="uw-stat-row">
        <div class="uw-stat"><b>{fmt(value)}</b><span>{html.escape(metric_name)}</span></div>
        <div class="uw-stat"><b>{len(method_runs(method["method_id"], run_index))}</b><span>runs</span></div>
      </div>
    </div>
    """


def render_gold(gold: dict[str, Any]) -> None:
    if "recommendations" in gold:
        st.markdown(
            f"State under review: {badge(gold.get('state_under_review', '—'), PALETTE['green'])}",
            unsafe_allow_html=True,
        )
        recs = pd.DataFrame(gold["recommendations"])
        if not recs.empty:
            recs["countries"] = recs["countries"].apply(lambda value: ", ".join(value))
        show_table(recs, height=400, max_rows=300)
    else:
        show_json(gold)


def render_prediction(record: dict[str, Any], dataset_id: str) -> None:
    st.markdown(
        f"Prediction status: {status_badge(record.get('status', 'unknown'))}",
        unsafe_allow_html=True,
    )
    if record.get("error"):
        st.error(record["error"])
        return

    details = record.get("details") or {}
    if dataset_id == "upr_recommendations":
        metrics = details.get("metrics", {})
        cols = st.columns(5)
        cols[0].metric("F1", fmt(metrics.get("f1")))
        cols[1].metric("Precision", fmt(metrics.get("precision")))
        cols[2].metric("Recall", fmt(metrics.get("recall")))
        cols[3].metric("Text F1", fmt(metrics.get("text_f1")))
        cols[4].metric("Country", fmt(metrics.get("country_accuracy")))
        st.caption(
            f"Gold {metrics.get('gold_count', 0)} · predicted "
            f"{metrics.get('prediction_count', 0)} · matched {metrics.get('matched', 0)} "
            f"· exact numbers {fmt(metrics.get('exact_number_accuracy'))}"
        )
        for title, key, color in (
            ("Matched", "matched", PALETTE["green"]),
            ("Missed gold", "missed", PALETTE["red"]),
            ("Extra predictions", "extras", PALETTE["purple"]),
        ):
            rows = [{"status": key, **row} for row in (details.get(key) or [])]
            st.markdown(f"**{title}** {badge(len(rows), color)}", unsafe_allow_html=True)
            if rows:
                show_table(pd.DataFrame(rows), status=True, height=300, max_rows=100)
    elif dataset_id == "upr_state":
        left, right = st.columns(2)
        left.metric("Gold", details.get("gold_state") or "—")
        right.metric("Prediction", details.get("prediction_state") or "—")
    else:
        fields = details.get("fields", {})
        rows = [
            {
                "field": field,
                "gold": item.get("gold"),
                "prediction": item.get("prediction"),
                "status": "hit" if item.get("hit") else "wrong",
                "token F1": item.get("token_f1"),
            }
            for field, item in fields.items()
        ]
        if rows:
            show_table(pd.DataFrame(rows), status=True, height=300, metrics=["token F1"])
    with st.expander("Raw prediction JSON"):
        show_json(record.get("prediction"), height=300)


def render_payload(document_row: dict[str, Any]) -> None:
    document = document_from_row(document_row)
    representation = st.radio(
        "Representation",
        ["text", "segments-tsv", "segments-json"],
        horizontal=True,
        format_func={
            "text": "Plain text",
            "segments-tsv": "Segments · TSV",
            "segments-json": "Segments · JSON",
        }.get,
        help="What a generated extract(document) script would receive.",
    )
    payload = document_payload(
        document,
        representation=representation,  # type: ignore[arg-type]
        include_title=False,
    )
    if representation == "text":
        st.caption("Plain text: segments joined in reading order; layout is lost.")
        st.code(payload["text"], language="text", height=420, wrap_lines=True)
    elif representation == "segments-tsv":
        st.caption(
            "TSV: a compact table (page, top, left, width, height, type, text) as one string."
        )
        st.code(payload["segments_tsv"], language="text", height=420, wrap_lines=True)
    else:
        st.caption("JSON: a list of structured segment objects with layout fields.")
        st.code(
            json.dumps(payload, indent=2, ensure_ascii=False),
            language="json",
            height=420,
            wrap_lines=True,
        )


def document_excerpt(doc_row: dict[str, Any], limit: int = 4000) -> None:
    st.caption(f"First {limit:,} characters. Full text is in the Sample explorer.")
    st.code(doc_row["text"][:limit], language="text", height=320, wrap_lines=True)
    with st.expander("Full document text"):
        st.code(doc_row["text"], language="text", height=500, wrap_lines=True)


# ---------------------------------------------------------------------------
# Pages


def page_datasets(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Dataset gallery")
    st.markdown(
        f'<div class="uw-section-note">{len(datasets)} task datasets · '
        f"{sum(row['sample_count'] for row in datasets):,} dataset rows · "
        "documents are shared across tasks, never copied.</div>",
        unsafe_allow_html=True,
    )
    run_counts: dict[str, int] = {}
    for run in run_index:
        run_counts[run["dataset_id"]] = run_counts.get(run["dataset_id"], 0) + 1

    columns = st.columns(3)
    for index, row in enumerate(datasets):
        with columns[index % 3]:
            st.markdown(
                dataset_card(row, run_counts.get(row["dataset_id"], 0)), unsafe_allow_html=True
            )

    selected = st.selectbox(
        "Open dataset card",
        [row["dataset_id"] for row in datasets],
        format_func=lambda value: next(
            (
                json.loads(row["manifest_json"]).get("name", value)
                for row in datasets
                if row["dataset_id"] == value
            ),
            value,
        ),
    )
    row = load_dataset(selected)
    if not row:
        return
    manifest = json.loads(row["manifest_json"])
    accent = dataset_accent(row)
    st.markdown(
        f'<div class="uw-divider"></div><h3 style="margin-bottom:4px">'
        f"{html.escape(manifest.get('name', selected))}</h3>"
        f'<div class="uw-card-id">{html.escape(selected)}</div>'
        f"<div>{badge(row.get('task', ''), accent)}{badge(row.get('db', ''), PALETTE['gray'])}"
        f"{badge(manifest.get('primary_metric', '—'), PALETTE['yellow'])}</div>",
        unsafe_allow_html=True,
    )
    st.write(manifest.get("description", ""))
    counts = manifest.get("counts", {})
    cols = st.columns(4)
    for column, key in zip(cols, ("total", "train", "dev", "test"), strict=False):
        column.metric(key.title(), f"{counts.get(key, 0):,}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Target schema**")
        show_json(manifest.get("target_schema", {}))
        st.markdown("**Split policy**")
        show_json(manifest.get("split_policy", {}))
    with right:
        st.markdown("**Label distribution**")
        distribution = manifest.get("label_distribution", {})
        if isinstance(distribution, dict) and "recommendations_total" in distribution:
            show_json(distribution)
        elif isinstance(distribution, dict):
            for key, values in distribution.items():
                st.caption(key)
                show_table(
                    pd.DataFrame(values.items(), columns=["value", "count"]),
                    height=240,
                    max_rows=60,
                )
    st.markdown("**Known issues**")
    for issue in manifest.get("known_issues", []):
        st.markdown(f"- {issue}")

    dataset_run_rows = [
        {
            "run": run["run_id"],
            "method": run["method_id"],
            "split": run["split"],
            "status": run["status"],
            **{
                key: run.get("metrics", {}).get(key)
                for key in ("accuracy", "f1", "state_accuracy")
                if key in run.get("metrics", {})
            },
        }
        for run in run_index
        if run["dataset_id"] == selected
    ]
    if dataset_run_rows:
        st.markdown("**Runs on this dataset**")
        show_table(pd.DataFrame(dataset_run_rows), status=True, height=300, max_rows=100)


def page_sample_explorer(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Sample explorer")
    dataset_id = st.selectbox(
        "Dataset",
        [row["dataset_id"] for row in datasets],
        format_func=lambda value: next(
            (
                json.loads(row["manifest_json"]).get("name", value)
                for row in datasets
                if row["dataset_id"] == value
            ),
            value,
        ),
    )
    dataset = load_dataset(dataset_id)
    if not dataset:
        return
    samples = load_samples(dataset_id).copy()

    dataset_runs = [run for run in run_index if run["dataset_id"] == dataset_id]
    run_options = ["No run overlay"] + [run_label(run) for run in dataset_runs]
    run_choice = st.selectbox("Run overlay", run_options)
    selected_run = None
    records: dict[str, dict[str, Any]] = {}
    if run_choice != "No run overlay":
        selected_run = dataset_runs[run_options.index(run_choice) - 1]
        records = load_predictions(selected_run["run_id"])
        samples["status"] = samples["doc_id"].map(
            lambda doc_id: records.get(doc_id, {}).get("status", "")
        )

    filters = st.columns([1, 1, 1, 2])
    split = filters[0].selectbox("Split", SPLITS)
    status_filter = (
        filters[1].selectbox("Status", ["all", *STATUS_OPTIONS]) if selected_run else "all"
    )
    search = filters[3].text_input("Search title or id", "")
    filtered = samples
    if split != "all":
        filtered = filtered[filtered["split"] == split]
    if status_filter != "all":
        filtered = filtered[filtered["status"] == status_filter]
    if search:
        mask = filtered["title"].str.contains(search, case=False, na=False) | filtered[
            "doc_id"
        ].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    st.caption(f"{len(filtered):,} samples")

    columns = ["doc_id", "title", "split", "pages", "segments", "chars"]
    if selected_run:
        columns.insert(3, "status")
    page_rows = paginate(filtered, 50, key=f"sample_{dataset_id}")
    show_table(
        page_rows[columns],
        status=selected_run is not None,
        height=340,
        max_rows=50,
    )
    if page_rows.empty:
        st.info("No samples match the current filters.")
        return
    options = page_rows["doc_id"].tolist()
    previous = st.session_state.get("selected_doc_id")
    index = options.index(previous) if previous in options else 0
    titles = dict(zip(page_rows["doc_id"], page_rows["title"], strict=False))
    doc_id = st.selectbox(
        "Select sample",
        options,
        index=index,
        format_func=lambda value: titles.get(value, value),
    )
    st.session_state["selected_doc_id"] = doc_id

    sample = filtered[filtered["doc_id"] == doc_id].iloc[0]
    doc_row = load_document(dataset["db"], doc_id)
    if not doc_row:
        st.error(f"Document {doc_id} is missing from the catalog.")
        return

    st.markdown('<div class="uw-divider"></div>', unsafe_allow_html=True)
    status_html = ""
    if selected_run:
        status_html = status_badge(records.get(doc_id, {}).get("status", "no prediction"))
    st.markdown(
        f'<h3 style="margin-bottom:2px">{html.escape(sample["title"])}</h3>'
        f'<div class="uw-card-id">{html.escape(doc_id)} · {html.escape(dataset_id)}</div>'
        f"<div>{split_badge(sample['split'])}{status_html}</div>",
        unsafe_allow_html=True,
    )
    stats = st.columns(5)
    stats[0].metric("Pages", doc_row["page_count"])
    stats[1].metric("Segments", doc_row["segment_count"])
    stats[2].metric("Characters", f"{doc_row['char_count']:,}")
    stats[3].metric("Est. tokens", f"{doc_row['token_estimate']:,}")
    stats[4].metric(
        "Gold recs",
        len(sample["gold"].get("recommendations", []))
        if "recommendations" in sample["gold"]
        else "—",
    )

    tab_names = ["Document text", "Segments", "Gold"]
    if selected_run:
        tab_names.append("Prediction")
    tab_names.append("Script payload")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.code(doc_row["text"], language="text", height=560, wrap_lines=True)
    with tabs[1]:
        segments = pd.DataFrame(json.loads(doc_row["segments_json"]))
        show_table(
            segments[["page", "text", "type", "top", "left", "width", "height"]],
            height=520,
            max_rows=400,
        )
    with tabs[2]:
        render_gold(sample["gold"])
    next_tab = 3
    if selected_run:
        with tabs[next_tab]:
            record = records.get(doc_id)
            if record:
                render_prediction(record, dataset_id)
            else:
                st.info("This sample has no prediction in the selected run.")
        next_tab += 1
    with tabs[next_tab]:
        render_payload(doc_row)


def method_comparison_table(
    methods: list[dict[str, Any]], run_index: list[dict[str, Any]], dataset: dict[str, Any]
) -> None:
    rows = []
    optional_metrics: set[str] = set()
    for method in methods:
        runs = method_runs(method["method_id"], run_index)
        if not runs:
            continue
        run = runs[0]
        metrics = run.get("metrics", {})
        counts = metrics.get("status_counts") or run.get("counts", {})
        metric_name, score = primary_metric(dataset, run)
        row = {
            "method": method.get("name", method["method_id"]),
            "type": method.get("type", ""),
            "model": method.get("model") or "—",
            "split": run["split"],
            "samples": metrics.get("documents", metrics.get("total", 0)),
            "metric": metric_name,
            "score": score,
            "hits": counts.get("hit", 0),
            "partial": counts.get("partial", 0),
            "wrong": counts.get("wrong", 0),
            "missing": counts.get("missing_prediction", 0),
            "errors": counts.get("error", 0),
            "description": (method.get("description") or "")[:110],
        }
        for key in ("precision", "recall"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                row[key] = value
                optional_metrics.add(key)
        rows.append(row)

    st.markdown('<div class="uw-divider"></div>', unsafe_allow_html=True)
    st.subheader("Method comparison")
    st.caption(
        "Latest run per method for this dataset. Status counts are document-level; "
        "score is the dataset's primary metric."
    )
    if not rows:
        st.info("No methods have run on this dataset yet.")
        return
    columns = [
        "method",
        "type",
        "model",
        "split",
        "samples",
        "metric",
        "score",
        "hits",
        "partial",
        "wrong",
        "missing",
        "errors",
        *[key for key in ("precision", "recall") if key in optional_metrics],
        "description",
    ]
    frame = pd.DataFrame(rows)[columns]
    show_table(
        frame,
        metrics=["score", *[key for key in ("precision", "recall") if key in optional_metrics]],
        height=440,
        max_rows=50,
    )


def render_method_detail(
    method: dict[str, Any],
    dataset: dict[str, Any] | None,
    run_index: list[dict[str, Any]],
    selected: str,
) -> None:
    color = TYPE_COLORS.get(method.get("type", ""), PALETTE["gray"])
    metric_name, value = primary_metric(dataset, latest_run(selected, run_index))
    st.markdown(
        f'<div class="uw-divider"></div><h3 style="margin-bottom:2px">'
        f"{html.escape(method.get('name', selected))}</h3>"
        f'<div class="uw-card-id">{html.escape(selected)}</div>'
        f"<div>{badge(method.get('type', 'method'), color)}"
        f"{badge(method.get('dataset_id', ''), PALETTE['blue'])}"
        f"{badge(method.get('model') or 'rule-based', PALETTE['gray'])}"
        f"{badge(f'{metric_name}: {fmt(value)}', PALETTE['yellow'])}</div>",
        unsafe_allow_html=True,
    )
    st.write(method.get("description", ""))

    runs = method_runs(selected, run_index)
    if not runs:
        st.info("This method has no runs yet.")
        return
    run_choice = st.selectbox("Run", [run_label(item) for item in runs])
    run = runs[[run_label(item) for item in runs].index(run_choice)]
    metrics = run.get("metrics", {})
    stats = st.columns(4)
    stats[0].metric("Documents", metrics.get("documents", metrics.get("total", 0)))
    for index, key in enumerate(("accuracy", "f1", "state_accuracy"), start=1):
        value = metrics.get(key)
        stats[index].metric(
            key.replace("_", " ").title(),
            f"{value:.3f}" if isinstance(value, (int, float)) else "—",
        )
    st.caption(
        f"`{run['run_id']}` · split `{run['split']}` · status `{run['status']}` · "
        f"method `{run['method_id']}`"
    )

    records = load_predictions(run["run_id"])
    samples = load_samples(method.get("dataset_id", "")).copy()
    rows = []
    for _, sample in samples.iterrows():
        record = records.get(sample["doc_id"])
        status = record["status"] if record else "no prediction"
        row = {
            "doc_id": sample["doc_id"],
            "title": sample["title"],
            "split": sample["split"],
            "status": status,
        }
        if record:
            details = record.get("details") or {}
            if method.get("dataset_id") == "upr_recommendations":
                rec_metrics = details.get("metrics", {})
                row.update(
                    {
                        "f1": rec_metrics.get("f1"),
                        "matched": rec_metrics.get("matched"),
                        "gold_recs": rec_metrics.get("gold_count"),
                    }
                )
            elif method.get("dataset_id") == "upr_state":
                row["prediction"] = details.get("prediction_state")
                row["gold"] = details.get("gold_state")
            else:
                fields = details.get("fields", {})
                if fields:
                    field_name, field = next(iter(fields.items()))
                    row["field"] = field_name
                    row["gold"] = field.get("gold")
                    row["prediction"] = field.get("prediction")
        rows.append(row)
    predictions_frame = pd.DataFrame(rows)

    filters = st.columns([1, 1, 2])
    status_filter = filters[0].selectbox("Filter status", ["all", *STATUS_OPTIONS, "no prediction"])
    search = filters[2].text_input("Search title or id", "")
    filtered = predictions_frame
    if status_filter != "all":
        filtered = filtered[filtered["status"] == status_filter]
    if search:
        filtered = filtered[
            filtered["title"].str.contains(search, case=False, na=False)
            | filtered["doc_id"].str.contains(search, case=False, na=False)
        ]
    display_columns = ["doc_id", "title", "split", "status"] + [
        column
        for column in ("f1", "matched", "gold_recs", "gold", "prediction")
        if column in filtered
    ]
    st.markdown(
        f'<div class="uw-section-note">{len(filtered):,} samples · '
        f"{len(records):,} predictions loaded. Select a sample below.</div>",
        unsafe_allow_html=True,
    )
    page_rows = paginate(filtered, 50, key=f"method_{selected}")
    show_table(
        page_rows[display_columns],
        status=True,
        metrics=["f1"],
        height=340,
        max_rows=50,
    )
    if page_rows.empty:
        st.info("No samples match the filters.")
        return
    options = page_rows["doc_id"].tolist()
    previous = st.session_state.get(f"method_doc_{selected}")
    index = options.index(previous) if previous in options else 0
    titles = dict(zip(page_rows["doc_id"], page_rows["title"], strict=False))
    doc_id = st.selectbox(
        "Select sample",
        options,
        index=index,
        format_func=lambda value: titles.get(value, value),
    )
    st.session_state[f"method_doc_{selected}"] = doc_id

    sample = samples[samples["doc_id"] == doc_id].iloc[0]
    record = records.get(doc_id)
    st.markdown('<div class="uw-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="margin-bottom:2px">{html.escape(sample["title"])}</h3>'
        f'<div class="uw-card-id">{html.escape(doc_id)}</div>',
        unsafe_allow_html=True,
    )
    detail_tabs = st.tabs(["Prediction", "Gold", "Document excerpt"])
    with detail_tabs[0]:
        if record:
            render_prediction(record, method.get("dataset_id", ""))
        else:
            st.info("No prediction for this sample in the selected run.")
    with detail_tabs[1]:
        render_gold(sample["gold"])
    with detail_tabs[2]:
        doc_row = load_document(dataset["db"], doc_id) if dataset else None
        if doc_row:
            document_excerpt(doc_row)

    directory = run_artifacts.METHODS_DIR / selected
    prompt_path = directory / "prompt.txt"
    script_path = directory / "script.py"
    tabs = st.tabs(["Prompt & context", "Script", "Attempts", "Metrics"])
    with tabs[0]:
        if prompt_path.exists():
            st.code(prompt_path.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Rule-based baseline: no LLM prompt.")
        with st.expander("Method configuration"):
            show_json(method.get("config", {}))
    with tabs[1]:
        if script_path.exists():
            st.code(script_path.read_text(encoding="utf-8"), language="python")
        else:
            st.info("Rule-based baseline: no generated script.")
    with tabs[2]:
        attempts = (
            sorted((directory / "attempts").glob("*.json"))
            if (directory / "attempts").exists()
            else []
        )
        if not attempts:
            st.info("No generation attempts recorded for this method.")
        else:
            attempt = st.selectbox("Attempt", [path.name for path in attempts])
            data = json.loads((directory / "attempts" / attempt).read_text(encoding="utf-8"))
            st.caption(
                f"model `{data.get('model', '—')}` · finish `{data.get('finish_reason', '—')}` "
                f"· validation issues: {data.get('validation_issues') or 'none'}"
            )
            if data.get("smoke_error"):
                st.error(data["smoke_error"])
            if data.get("code"):
                st.code(data["code"], language="python")
            with st.expander("Raw response"):
                st.code(data.get("response", ""), language="text", height=300, wrap_lines=True)
    with tabs[3]:
        show_json(metrics)


def page_methods(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Methods")
    all_methods = load_methods()
    if not all_methods:
        st.info("No method artifacts yet. Run a baseline or an LLM experiment.")
        return
    dataset_by_id = {row["dataset_id"]: row for row in datasets}
    available_datasets = [
        row
        for row in datasets
        if any(m.get("dataset_id") == row["dataset_id"] for m in all_methods)
    ]
    if not available_datasets:
        st.info("No datasets have methods yet.")
        return

    dataset_id = st.selectbox(
        "Dataset",
        [row["dataset_id"] for row in available_datasets],
        format_func=lambda value: next(
            (
                json.loads(row["manifest_json"]).get("name", value)
                for row in available_datasets
                if row["dataset_id"] == value
            ),
            value,
        ),
    )
    dataset = dataset_by_id[dataset_id]
    methods = [m for m in all_methods if m.get("dataset_id") == dataset_id]
    methods.sort(key=lambda m: m.get("name", m["method_id"]))

    show_cards = st.toggle(
        "Show method cards",
        value=True,
        help="Hide the gallery when there are too many methods.",
    )
    if show_cards and methods:
        columns = st.columns(3)
        for index, method in enumerate(methods):
            with columns[index % 3]:
                st.markdown(method_card(method, run_index, dataset), unsafe_allow_html=True)

    method_comparison_table(methods, run_index, dataset)

    st.markdown('<div class="uw-divider"></div>', unsafe_allow_html=True)
    st.subheader("Method detail")
    selected = st.selectbox(
        "Open method",
        [method["method_id"] for method in methods],
        format_func=lambda value: next(
            (method.get("name", value) for method in methods if method["method_id"] == value),
            value,
        ),
    )
    method = next(item for item in methods if item["method_id"] == selected)
    render_method_detail(method, dataset, run_index, selected)


def page_compare(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Compare")
    if not run_index:
        st.info("No runs to compare yet.")
        return
    available = sorted(
        {
            key
            for run in run_index
            for key in run.get("metrics", {})
            if key
            in {
                "accuracy",
                "f1",
                "state_accuracy",
                "text_f1",
                "country_accuracy",
                "macro_f1",
            }
        }
    )
    metric = st.selectbox("Metric", available)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for run in run_index:
        if metric_value(run, metric) is None:
            continue
        latest.setdefault((run["method_id"], run["dataset_id"]), run)
    rows = [
        {
            "method": method_id,
            "dataset": dataset_id,
            "metric": metric_value(run, metric),
            "split": run["split"],
        }
        for (method_id, dataset_id), run in latest.items()
    ]
    if not rows:
        st.info(f"No runs with metric `{metric}`.")
        return
    table = pd.DataFrame(rows).pivot_table(
        index="method", columns="dataset", values="metric", aggfunc="last"
    )
    st.caption(
        f"Latest run per method × dataset · metric `{metric}` · green is better, red is worse."
    )
    matrix = table.reset_index()
    show_table(
        matrix,
        metrics=[column for column in matrix.columns if column in table.columns],
        height=400,
        max_rows=50,
    )
    st.markdown("**Method legend**")
    legend = []
    for method_id in table.index:
        method = next((m for m in load_methods() if m["method_id"] == method_id), {})
        legend.append(
            {
                "method": method_id,
                "name": method.get("name", ""),
                "dataset": method.get("dataset_id", ""),
                "type": method.get("type", ""),
                "model": method.get("model", ""),
            }
        )
    show_table(pd.DataFrame(legend), height=300, max_rows=50)
    if len(table.columns):
        st.caption(f"Datasets in this matrix: {', '.join(sorted(table.columns))}")


def main() -> None:
    st.title("Uwazi extraction benchmarks")
    st.caption(f"Catalog: `{CATALOG_PATH}` · artifacts: `{ARTIFACTS_DIR}`")

    if not CATALOG_PATH.exists():
        st.error("No catalog found. Build it first:\n\n```\nuv run build-datasets\n```")
        return

    datasets = load_datasets()
    if not datasets:
        st.error("The catalog has no datasets. Run `uv run build-datasets`.")
        return
    run_index = load_run_index()
    methods = load_methods()

    page = st.sidebar.radio("View", ["Datasets", "Sample explorer", "Methods", "Compare"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Theme")
    theme_name = st.sidebar.selectbox("Color palette", theme_names(), key="theme_name")
    set_theme(theme_name)
    inject_css()
    st.sidebar.caption(THEMES[theme_name]["description"])
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(datasets)} datasets · {sum(row['sample_count'] for row in datasets):,} samples"
    )
    st.sidebar.caption(f"{len(methods)} methods · {len(run_index)} runs")

    if page == "Datasets":
        page_datasets(datasets, run_index)
    elif page == "Sample explorer":
        page_sample_explorer(datasets, run_index)
    elif page == "Methods":
        page_methods(datasets, run_index)
    else:
        page_compare(datasets, run_index)


main()

"""Streamlit viewer for datasets, samples and (later) experiment runs.

Run with::

    uv run streamlit run app/app.py
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from llm_script_creation import catalog
from llm_script_creation import runs as run_artifacts
from llm_script_creation.configuration import ARTIFACTS_DIR, CATALOG_PATH
from llm_script_creation.models import Document, Segment
from llm_script_creation.text import document_payload

st.set_page_config(page_title="Uwazi extraction benchmarks", layout="wide")

SPLITS = ["all", "train", "dev", "test"]


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
            gold = json.loads(row["gold_json"])
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "split": row["split"],
                    "pages": row["page_count"],
                    "segments": row["segment_count"],
                    "chars": row["char_count"],
                    "gold": gold,
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


def document_from_row(row: dict[str, Any]) -> Document:
    return Document(
        entity_id=row["doc_id"],
        shared_id=row["doc_id"],
        title=row["title"],
        language=row["language"],
        template=row["template"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        file_id=row["file_id"],
        filename=row["filename"],
        mimetype=row["mimetype"],
        total_pages=row["page_count"],
        segments=[Segment.from_mongo(item) for item in json.loads(row["segments_json"])],
    )


def stats_row(document: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Pages", document["page_count"])
    cols[1].metric("Segments", document["segment_count"])
    cols[2].metric("Characters", f"{document['char_count']:,}")
    cols[3].metric("Est. tokens", f"{document['token_estimate']:,}")
    cols[4].metric("Language", document["language"])


def render_gold(gold: dict[str, Any]) -> None:
    if "recommendations" in gold:
        st.caption(f"State under review: **{gold.get('state_under_review', '—')}**")
        recs = pd.DataFrame(gold["recommendations"])
        if not recs.empty:
            recs["countries"] = recs["countries"].apply(lambda value: ", ".join(value))
        st.dataframe(recs, width="stretch", hide_index=True)
    else:
        st.json(gold)


def render_payload(document_row: dict[str, Any]) -> None:
    document = document_from_row(document_row)
    representation = st.radio(
        "Representation",
        ["text", "segments-tsv", "segments-json"],
        horizontal=True,
        help="What a generated extract(document) script would receive.",
    )
    payload = document_payload(
        document,
        representation=representation,  # type: ignore[arg-type]
        include_title=False,
    )
    st.caption("Titles are omitted from the default payload to avoid leaking the UPR state.")
    if representation == "text":
        st.text_area(
            "Payload text",
            payload["text"],
            height=420,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.json(payload)


def run_label(run: dict[str, Any]) -> str:
    return f"{run['run_id']} ({run['split']}, {run['status']})"


def metric_value(run: dict[str, Any], metric: str) -> float | None:
    value = run.get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def render_prediction(record: dict[str, Any], dataset_id: str) -> None:
    st.caption(f"Run status: **{record.get('status', 'unknown')}**")
    if record.get("error"):
        st.error(record["error"])
        return
    details = record.get("details") or {}
    if dataset_id == "upr_recommendations":
        metrics = details.get("metrics", {})
        cols = st.columns(5)
        cols[0].metric("F1", f"{metrics.get('f1', 0):.3f}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
        cols[3].metric("Text F1", f"{metrics.get('text_f1', 0):.3f}")
        cols[4].metric("Country acc.", f"{metrics.get('country_accuracy') or 0:.3f}")
        st.caption(
            f"Gold {metrics.get('gold_count', 0)} · predicted "
            f"{metrics.get('prediction_count', 0)} · matched {metrics.get('matched', 0)} "
            f"· exact numbers {metrics.get('exact_number_accuracy') or 0:.3f}"
        )
        for title, key in (
            ("Matched", "matched"),
            ("Missed gold recommendations", "missed"),
            ("Extra predictions", "extras"),
        ):
            rows = details.get(key) or []
            st.markdown(f"**{title}** ({len(rows)})")
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    elif dataset_id == "upr_state":
        st.write(f"Gold: **{details.get('gold_state')}**")
        st.write(f"Prediction: **{details.get('prediction_state')}**")
    else:
        fields = details.get("fields", {})
        rows = [
            {
                "field": field,
                "gold": item.get("gold"),
                "prediction": item.get("prediction"),
                "hit": "yes" if item.get("hit") else "no",
                "token F1": item.get("token_f1"),
            }
            for field, item in fields.items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Raw prediction JSON"):
        st.json(record.get("prediction"))


def page_datasets(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Datasets")
    st.caption(
        f"{len(datasets)} task datasets in `{ARTIFACTS_DIR}`. "
        "Each dataset shares one document snapshot; splits are stable by document id."
    )
    run_counts: dict[str, int] = {}
    for run in run_index:
        run_counts[run["dataset_id"]] = run_counts.get(run["dataset_id"], 0) + 1
    frame = pd.DataFrame(
        [
            {
                "dataset": row["dataset_id"],
                "task": row["task"],
                "samples": row["sample_count"],
                "train": row["train_count"],
                "dev": row["dev_count"],
                "test": row["test_count"],
                "runs": run_counts.get(row["dataset_id"], 0),
                "primary metric": json.loads(row["manifest_json"]).get("primary_metric", ""),
            }
            for row in datasets
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)

    selected = st.selectbox("Dataset card", [row["dataset_id"] for row in datasets])
    row = load_dataset(selected)
    if not row:
        return
    manifest = json.loads(row["manifest_json"])
    st.markdown(f"### {manifest['name']}")
    st.write(manifest.get("description", ""))
    counts = manifest.get("counts", {})
    cols = st.columns(4)
    for column, key in zip(cols, ("total", "train", "dev", "test"), strict=False):
        column.metric(key.title(), f"{counts.get(key, 0):,}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Target schema**")
        st.json(manifest.get("target_schema", {}))
        st.markdown("**Split policy**")
        st.json(manifest.get("split_policy", {}))
    with right:
        st.markdown("**Label distribution**")
        distribution = manifest.get("label_distribution", {})
        if isinstance(distribution, dict) and "recommendations_total" in distribution:
            st.json(distribution)
        elif isinstance(distribution, dict):
            for key, values in distribution.items():
                st.markdown(f"`{key}`")
                st.dataframe(
                    pd.DataFrame(values.items(), columns=["value", "count"]),
                    width="stretch",
                    hide_index=True,
                )
    st.markdown("**Known issues**")
    for issue in manifest.get("known_issues", []):
        st.markdown(f"- {issue}")


def page_sample_explorer(datasets: list[dict[str, Any]], run_index: list[dict[str, Any]]) -> None:
    st.subheader("Sample explorer")
    dataset_ids = [row["dataset_id"] for row in datasets]
    dataset_id = st.selectbox("Dataset", dataset_ids)
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

    filters = st.columns([1, 1, 2])
    split = filters[0].selectbox("Split", SPLITS)
    search = filters[2].text_input("Search title or id", "")
    filtered = samples
    if split != "all":
        filtered = filtered[filtered["split"] == split]
    if search:
        mask = filtered["title"].str.contains(search, case=False, na=False) | filtered[
            "doc_id"
        ].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    st.caption(f"{len(filtered):,} samples")

    columns = ["doc_id", "title", "split", "pages", "segments", "chars"]
    if selected_run:
        columns.insert(3, "status")
    event = st.dataframe(
        filtered[columns],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        st.session_state["selected_doc_id"] = filtered.iloc[selected_rows[0]]["doc_id"]

    doc_id = st.session_state.get("selected_doc_id")
    if doc_id not in set(filtered["doc_id"]):
        doc_id = filtered.iloc[0]["doc_id"] if not filtered.empty else None
    if not doc_id:
        st.info("No samples match the current filters.")
        return

    sample = filtered[filtered["doc_id"] == doc_id].iloc[0]
    doc_row = load_document(dataset["db"], doc_id)
    if not doc_row:
        st.error(f"Document {doc_id} is missing from the catalog.")
        return

    st.divider()
    st.markdown(f"### {sample['title']}")
    st.caption(f"`{doc_id}` · split `{sample['split']}` · dataset `{dataset_id}`")
    stats_row(doc_row)

    tab_names = ["Document text", "Segments", "Gold"]
    if selected_run:
        tab_names.append("Prediction")
    tab_names.append("Script payload")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        with st.container(height=560, border=True):
            st.text(doc_row["text"])
    with tabs[1]:
        segments = pd.DataFrame(json.loads(doc_row["segments_json"]))
        st.dataframe(
            segments[["page", "type", "top", "left", "width", "height", "text"]],
            width="stretch",
            hide_index=True,
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


def page_methods(run_index: list[dict[str, Any]]) -> None:
    st.subheader("Methods")
    methods = load_methods()
    if not methods:
        st.info("No method artifacts yet. Run a baseline or an LLM experiment.")
        return
    frame = pd.DataFrame(
        [
            {
                "method": method["method_id"],
                "name": method.get("name", ""),
                "type": method.get("type", ""),
                "dataset": method.get("dataset_id", ""),
                "description": method.get("description", ""),
            }
            for method in methods
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    selected = st.selectbox("Method detail", [method["method_id"] for method in methods])
    method = next(item for item in methods if item["method_id"] == selected)
    st.markdown(f"### {method.get('name', selected)}")
    st.write(method.get("description", ""))
    st.json(method.get("config", {}))

    directory = run_artifacts.METHODS_DIR / selected
    prompt_path = directory / "prompt.txt"
    script_path = directory / "script.py"
    if prompt_path.exists():
        with st.expander("Prompt sent to the LLM", expanded=False):
            st.code(prompt_path.read_text(encoding="utf-8"), language="text")
    if script_path.exists():
        with st.expander("Frozen extraction script", expanded=False):
            st.code(script_path.read_text(encoding="utf-8"), language="python")
    attempts = (
        sorted((directory / "attempts").glob("*.json")) if (directory / "attempts").exists() else []
    )
    if attempts:
        st.caption(f"{len(attempts)} generation/repair attempts")

    method_runs = [run for run in run_index if run["method_id"] == selected]
    if method_runs:
        st.markdown("**Runs**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": run["run_id"],
                        "dataset": run["dataset_id"],
                        "split": run["split"],
                        "status": run["status"],
                    }
                    for run in method_runs
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def page_runs(run_index: list[dict[str, Any]]) -> None:
    st.subheader("Runs")
    if not run_index:
        st.info("No runs yet. Run `uv run run-baselines` first.")
        return

    rows = []
    for run in run_index:
        metrics = run.get("metrics", {})
        rows.append(
            {
                "run": run["run_id"],
                "method": run["method_id"],
                "dataset": run["dataset_id"],
                "split": run["split"],
                "status": run["status"],
                "accuracy": metric_value(run, "accuracy"),
                "f1": metric_value(run, "f1"),
                "state_acc": metric_value(run, "state_accuracy"),
                "text_f1": metric_value(run, "text_f1"),
                "country_acc": metric_value(run, "country_accuracy"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    selected = st.selectbox("Run detail", [run_label(run) for run in run_index])
    run = run_index[[run_label(item) for item in run_index].index(selected)]
    metrics = run.get("metrics", {})
    st.markdown(f"### {run['run_id']}")
    st.caption(
        f"method `{run['method_id']}` · dataset `{run['dataset_id']}` · split `{run['split']}`"
    )
    cols = st.columns(4)
    cols[0].metric("Documents", metrics.get("documents", metrics.get("total", 0)))
    for index, key in enumerate(("accuracy", "f1", "state_accuracy"), start=1):
        if index < 4:
            value = metrics.get(key)
            cols[index].metric(
                key.replace("_", " ").title(),
                f"{value:.3f}" if isinstance(value, (int, float)) else "—",
            )
    st.json(metrics)
    if "fields" in metrics:
        st.markdown("**Per-field metrics**")
        st.dataframe(
            pd.DataFrame(
                [{"field": field, **values} for field, values in metrics["fields"].items()]
            ),
            width="stretch",
            hide_index=True,
        )

    records = list(load_predictions(run["run_id"]).values())
    if records:
        status_counts = pd.Series([record["status"] for record in records]).value_counts()
        st.markdown("**Prediction statuses**")
        st.dataframe(
            status_counts.rename_axis("status").reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        status_filter = st.multiselect(
            "Filter by status",
            sorted(status_counts.index),
            default=sorted(status_counts.index),
        )
        filtered = [record for record in records if record["status"] in status_filter]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "doc_id": record["doc_id"],
                        "split": record["split"],
                        "status": record["status"],
                        "error": record.get("error") or "",
                    }
                    for record in filtered
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if any(record.get("error") for record in records):
            with st.expander("Errors"):
                for record in records:
                    if record.get("error"):
                        st.error(f"{record['doc_id']}: {record['error']}")


def page_compare(run_index: list[dict[str, Any]]) -> None:
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
        value = metric_value(run, metric)
        if value is None:
            continue
        latest.setdefault((run["method_id"], run["dataset_id"]), run)
    rows = [
        {
            "method": method_id,
            "dataset": dataset_id,
            "metric": metric_value(run, metric),
        }
        for (method_id, dataset_id), run in latest.items()
    ]
    if not rows:
        st.info(f"No runs with metric `{metric}`.")
        return
    table = pd.DataFrame(rows).pivot_table(
        index="method", columns="dataset", values="metric", aggfunc="last"
    )
    st.dataframe(table.style.format("{:.3f}"), width="stretch")
    st.caption("Latest run per method × dataset; blank means the metric is not applicable.")


def main() -> None:
    st.title("Uwazi extraction benchmarks")
    st.caption(f"Catalog: `{CATALOG_PATH}`")

    if not CATALOG_PATH.exists():
        st.error("No catalog found. Build it first:\n\n```\nuv run build-datasets\n```")
        return

    datasets = load_datasets()
    if not datasets:
        st.error("The catalog has no datasets. Run `uv run build-datasets`.")
        return
    run_index = load_run_index()

    page = st.sidebar.radio("View", ["Datasets", "Sample explorer", "Methods", "Runs", "Compare"])
    st.sidebar.caption(
        f"{len(datasets)} datasets · {sum(row['sample_count'] for row in datasets):,} samples · "
        f"{len(load_methods())} methods · {len(run_index)} runs"
    )

    if page == "Datasets":
        page_datasets(datasets, run_index)
    elif page == "Sample explorer":
        page_sample_explorer(datasets, run_index)
    elif page == "Methods":
        page_methods(run_index)
    elif page == "Runs":
        page_runs(run_index)
    else:
        page_compare(run_index)


main()

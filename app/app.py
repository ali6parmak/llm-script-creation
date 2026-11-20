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


def page_datasets(datasets: list[dict[str, Any]]) -> None:
    st.subheader("Datasets")
    st.caption(
        f"{len(datasets)} task datasets in `{ARTIFACTS_DIR}`. "
        "Each dataset shares one document snapshot; splits are stable by document id."
    )
    frame = pd.DataFrame(
        [
            {
                "dataset": row["dataset_id"],
                "task": row["task"],
                "samples": row["sample_count"],
                "train": row["train_count"],
                "dev": row["dev_count"],
                "test": row["test_count"],
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


def page_sample_explorer(datasets: list[dict[str, Any]]) -> None:
    st.subheader("Sample explorer")
    dataset_ids = [row["dataset_id"] for row in datasets]
    dataset_id = st.selectbox("Dataset", dataset_ids)
    dataset = load_dataset(dataset_id)
    if not dataset:
        return
    samples = load_samples(dataset_id)

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

    event = st.dataframe(
        filtered[["doc_id", "title", "split", "pages", "segments", "chars"]],
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

    tabs = st.tabs(["Document text", "Segments", "Gold", "Script payload"])
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
    with tabs[3]:
        render_payload(doc_row)


def page_placeholder(title: str, detail: str) -> None:
    st.subheader(title)
    st.info(detail)


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

    page = st.sidebar.radio("View", ["Datasets", "Sample explorer", "Methods", "Runs", "Compare"])
    st.sidebar.caption(
        f"{len(datasets)} datasets · "
        f"{sum(row['sample_count'] for row in datasets):,} dataset samples"
    )

    if page == "Datasets":
        page_datasets(datasets)
    elif page == "Sample explorer":
        page_sample_explorer(datasets)
    elif page == "Methods":
        page_placeholder(
            "Methods",
            "Method cards (name, description, prompt, script, attempts) will appear here "
            "once the experiment pipeline writes artifacts.",
        )
    elif page == "Runs":
        page_placeholder(
            "Runs",
            "Run metrics, predictions and error inspection will appear here once a method "
            "has been executed on a dataset.",
        )
    else:
        page_placeholder(
            "Compare",
            "The method × dataset comparison matrix will appear here after the first runs.",
        )


main()

"""SQLite catalog for documents, datasets and samples.

The catalog is a local index/artifact so experiments and the Streamlit viewer do
not need to query MongoDB on every interaction. MongoDB remains the source of
truth and the catalog can be rebuilt at any time.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .configuration import CATALOG_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    db TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    language TEXT NOT NULL,
    template TEXT NOT NULL,
    title TEXT,
    page_count INTEGER,
    segment_count INTEGER,
    char_count INTEGER,
    token_estimate INTEGER,
    file_id TEXT,
    filename TEXT,
    mimetype TEXT,
    text TEXT,
    segments_json TEXT,
    metadata_json TEXT,
    PRIMARY KEY (db, doc_id)
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    db TEXT NOT NULL,
    name TEXT NOT NULL,
    task TEXT NOT NULL,
    description TEXT,
    target_schema_json TEXT,
    known_issues_json TEXT,
    manifest_json TEXT,
    created_at TEXT,
    split_policy_version TEXT,
    dataset_version TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    dataset_id TEXT NOT NULL,
    db TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    split TEXT NOT NULL,
    gold_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, db, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_samples_dataset_split ON samples(dataset_id, split);
CREATE INDEX IF NOT EXISTS idx_documents_db_template ON documents(db, template);
"""


def connect(path: Path = CATALOG_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add catalog columns introduced after the first build (additive only)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Existing v2 catalogs predate the dataset-version columns. Adding them is
    # additive and leaves every v2 dataset/sample row untouched.
    _ensure_columns(
        conn,
        "datasets",
        {"split_policy_version": "TEXT", "dataset_version": "TEXT"},
    )
    conn.commit()


def reset() -> None:
    """Remove the catalog so a build starts from a clean state."""
    for suffix in ("", "-wal", "-shm"):
        Path(str(CATALOG_PATH) + suffix).unlink(missing_ok=True)


def insert_documents(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    columns = (
        "db",
        "doc_id",
        "language",
        "template",
        "title",
        "page_count",
        "segment_count",
        "char_count",
        "token_estimate",
        "file_id",
        "filename",
        "mimetype",
        "text",
        "segments_json",
        "metadata_json",
    )
    sql = f"INSERT OR REPLACE INTO documents ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"
    count = 0
    for row in rows:
        conn.execute(sql, tuple(row.get(column) for column in columns))
        count += 1
    conn.commit()
    return count


def insert_dataset(
    conn: sqlite3.Connection, dataset: dict[str, Any], samples: Iterable[dict[str, Any]]
) -> int:
    conn.execute(
        """
        INSERT OR REPLACE INTO datasets
            (dataset_id, db, name, task, description, target_schema_json,
             known_issues_json, manifest_json, created_at, split_policy_version,
             dataset_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset["dataset_id"],
            dataset["db"],
            dataset["name"],
            dataset["task"],
            dataset.get("description", ""),
            json.dumps(dataset.get("target_schema", {}), ensure_ascii=False),
            json.dumps(dataset.get("known_issues", []), ensure_ascii=False),
            json.dumps(dataset.get("manifest", {}), ensure_ascii=False),
            dataset.get("created_at"),
            dataset.get("split_policy_version"),
            dataset.get("dataset_version"),
        ),
    )
    # Replace the whole sample set: rebuilt splits must not leave stale rows.
    conn.execute("DELETE FROM samples WHERE dataset_id = ?", (dataset["dataset_id"],))
    count = 0
    for sample in samples:
        conn.execute(
            "INSERT OR REPLACE INTO samples (dataset_id, db, doc_id, split, gold_json) VALUES (?, ?, ?, ?, ?)",
            (
                dataset["dataset_id"],
                dataset["db"],
                sample["doc_id"],
                sample["split"],
                json.dumps(sample["gold"], ensure_ascii=False),
            ),
        )
        count += 1
    conn.commit()
    return count


def list_datasets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT d.*,
                   COUNT(s.doc_id) AS sample_count,
                   SUM(CASE WHEN s.split = 'train' THEN 1 ELSE 0 END) AS train_count,
                   SUM(CASE WHEN s.split = 'test' THEN 1 ELSE 0 END) AS test_count
            FROM datasets d
            LEFT JOIN samples s ON s.dataset_id = d.dataset_id
            GROUP BY d.dataset_id
            ORDER BY d.dataset_id
            """
        )
    )


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()


def iter_samples(
    conn: sqlite3.Connection,
    dataset_id: str,
    split: str | None = None,
    search: str | None = None,
) -> Iterator[sqlite3.Row]:
    query = """
        SELECT s.*, d.title, d.page_count, d.segment_count, d.char_count, d.token_estimate
        FROM samples s
        JOIN documents d ON d.db = s.db AND d.doc_id = s.doc_id
        WHERE s.dataset_id = ?
    """
    params: list[Any] = [dataset_id]
    if split:
        query += " AND s.split = ?"
        params.append(split)
    if search:
        query += " AND (s.doc_id LIKE ? OR d.title LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY s.split, d.title"
    yield from conn.execute(query, params)


def get_document(conn: sqlite3.Connection, db: str, doc_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE db = ? AND doc_id = ?", (db, doc_id)
    ).fetchone()


def count_documents(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def count_samples(conn: sqlite3.Connection, dataset_id: str) -> int:
    return int(
        conn.execute("SELECT COUNT(*) FROM samples WHERE dataset_id = ?", (dataset_id,)).fetchone()[
            0
        ]
    )

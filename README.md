# LLM script creation

Benchmark whether LLMs can write deterministic extraction scripts for Uwazi documents.

Two restored Uwazi instances live in local MongoDB:

- `plan` — Plan International corpus (10,596 English `Document` entities, 10,537 with an
  English segmentation).
- `upr` — UPR corpus (676 English `Document` entities, 165 segmented; 39k linked
  recommendations).

The plan and research questions are in [PLAN.md](PLAN.md). Agent/contributor rules are in
[AGENTS.md](AGENTS.md). Data findings are in [reports/data_report.md](reports/data_report.md).

## Quickstart

```bash
# home cache can be read-only in some sandboxes; store the uv cache locally
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv sync --group dev

uv run inspect-data                 # writes reports/data_report.{md,json}
uv run inspect-data --db plan
uv run pytest -q
uv run ruff check src tests
```

`just` recipes wrap the common commands:

```bash
just sync
just inspect
just test
just lint
```

## Data access

```python
from llm_script_creation.db import PLAN_DB, PLAN_DOCUMENT_TEMPLATE, get_db, iter_documents

for doc in iter_documents(get_db(PLAN_DB), PLAN_DOCUMENT_TEMPLATE, language="en", limit=5):
    print(doc.title, len(doc.segments), doc.has_segmentation)
```

Join chain: `entities.sharedId → files.entity → segmentations.fileID`. An entity can own
several files (one per language); pick the file matching the entity language
(`en → eng`, `fr → fra`, `es → spa`).

## Layout

```
src/llm_script_creation/
  configuration.py   env, model, paths
  db.py              Mongo connection + template constants + loaders
  models.py          Segment, Document, metadata helpers
  text.py            text / TSV / JSON representations for prompts and scripts
  inspect_data.py    data report CLI (`inspect-data`)
tests/               unit tests (no Mongo required)
reports/             generated inspection reports
data/                restored Mongo dumps (gitignored)
```

## Status

M0 (data inspection) is complete. Next: dataset builder, evaluation harness, then the
LLM-generated-script pipeline. See [PLAN.md §13](PLAN.md#13-immediate-next-steps-m1).

# LLM script creation

Benchmark whether LLMs can write deterministic extraction scripts for Uwazi documents.

Two restored Uwazi instances live in local MongoDB:

- `plan` — Plan International corpus (10,596 English `Document` entities, 10,537 with an
  English segmentation).
- `upr` — UPR corpus (676 English `Document` entities, 165 segmented; 39k linked
  recommendations).

The research plan and open questions are in [PLAN.md](PLAN.md). Agent/contributor rules are
in [AGENTS.md](AGENTS.md). Data findings are in [reports/data_report.md](reports/data_report.md).

## Quickstart

```bash
# home cache can be read-only in some sandboxes; store the uv cache locally
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv sync --group dev

uv run inspect-data                 # reports/data_report.{md,json}
uv run build-datasets               # snapshot documents + build datasets (~1 min)
uv run run-baselines --split dev    # deterministic baseline runs
uv run streamlit run app/app.py     # viewer: datasets, samples, methods, runs, compare
uv run pytest -q
uv run ruff check src tests
```

Generate an LLM script and immediately evaluate it on the dev split:

```bash
uv run generate-script --dataset plan_year --split dev --examples 2
```

Cloud models run through Ollama's native `/api/chat` endpoint with `think: false` so
reasoning models return final content instead of spending the whole budget on hidden
reasoning. Use `--think low|high|max` if you want to test thinking output.

The prompt, every attempt, the frozen script, predictions, and metrics are written under
`artifacts/methods/<method_id>/` and `artifacts/runs/<run_id>/`. The viewer picks them up
automatically.

`just` recipes wrap the common commands: `just sync`, `just build`, `just baselines`,
`just app`, `just inspect`, `just test`, `just lint`.

## Artifacts

`artifacts/` is generated and gitignored:

```
artifacts/catalog.db              # documents, datasets, samples (SQLite)
artifacts/states.json             # 195 UPR state/country candidates
artifacts/datasets/<id>/          # manifest.json + samples.jsonl
artifacts/methods/<id>/           # method.json, prompt.txt, script.py, attempts/
artifacts/runs/<run_id>/          # run.json, predictions.jsonl, metrics.json
```

The catalog is the document/sample index used by experiments and the viewer. MongoDB remains
the source of truth; rebuild the catalog with `uv run build-datasets`.

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
  text.py            text / TSV / JSON representations
  inspect_data.py    data report CLI (`inspect-data`)
  catalog.py         SQLite catalog for documents/datasets/samples
  datasets.py        dataset specs, gold extraction, splits, builder
  normalization.py   text/number/country normalization
  evaluation.py      Plan field metrics + UPR recommendation matching
  baselines.py       deterministic regex/heuristic baselines
  experiments.py     baseline runner + run artifacts
  prompts.py         LLM prompt builder
  llm.py             Ollama OpenAI-compatible client
  sandbox.py         script validation + subprocess execution
  generate.py        generate → smoke test → repair → evaluate pipeline
app/app.py           Streamlit viewer
tests/               unit tests (no Mongo required)
reports/             generated inspection report
data/                restored Mongo dumps (gitignored)
```

## Status

The end-to-end prototype is working: catalog/datasets, viewer, rule-based baselines, and the
LLM-generated-script pipeline. First dev results and model notes are in
[PLAN.md §14](PLAN.md#14-implementation-status-and-first-results-2026-11-20).

Model note: `deepseek-v4.1-flash:cloud` is the default and works with `think: false`.
The OpenAI-compatible `/v1/chat/completions` endpoint ignored `think` and returned empty
content; the native endpoint fixed it. The pipeline records the actual model, reasoning,
`done_reason`, token usage, and every attempt.

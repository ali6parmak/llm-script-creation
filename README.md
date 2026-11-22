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
uv run run-baselines                # deterministic baselines on train and test
uv run streamlit run app/app.py     # viewer: datasets, samples, methods, compare
uv run pytest -q
uv run ruff check src tests app
```

Generate an LLM script, repair it on train, then evaluate the frozen script on test:

```bash
uv run generate-script --dataset plan_year --examples 2 --max-repair-rounds 4
```

## Admin-agent extractor (read-only evaluation)

This is a different strategy from `generate-script`: instead of asking the LLM to write a
script for our prompt, we run the **real Uwazi admin agent's nested extractor subagent**
(`uwazi_admin_agent.use_cases.author_extractor_tool.build_extractor_agent`) over our Plan
and UPR data, then execute the returned `def extract(text, ctx)` with the admin agent's own
sandbox.

The adapter uses the admin agent as-is (its model, its prompt, its tools). The default condition
is **no title**: entity titles and `ctx["title"]` are empty, other Uwazi metadata stays
visible, and the target property is removed from `ctx.metadata` at execution time so a
metadata shortcut cannot score. (Earlier v1/v2 runs had the title visible.) The extractor is
authored from the train split only; the untouched test split is used for one final execution.
The benchmark task prompt deliberately does **not** inject a list of train shared_ids: the
extractor discovers its own samples via `query_entities`, as it does in the real admin
workflow. The prompt version is recorded in every method artifact.

The admin agent keeps its own environment through a sidecar process
(`scripts/admin_agent_sidecar.py`, run with `~/projects/python_uwazi_API/.venv`). Our main
process never imports `uwazi_admin_agent`, never writes to MongoDB, and never calls a live
Uwazi instance.

```bash
# spike: author + execute on 5 train / 5 test documents
uv run benchmark-admin-agent --dataset plan_year --limit-train 5 --limit-test 5

# full run: the real 50-document train split and the whole test split
uv run benchmark-admin-agent --dataset plan_year

# UPR: state under review, then recommendation records (state + number + text + countries)
uv run benchmark-admin-agent --dataset upr_state
uv run benchmark-admin-agent --dataset upr_recommendations
```

`upr_state` uses the real relationship property (`state_reviewed`) and maps it to our
`state_under_review` gold field; the task receives all 195 Uwazi State labels.
`upr_recommendations` is **record extraction**, not a record write: the extractor returns a
list of `{number, text, countries}` records plus the state, which our existing UPR evaluator
scores. The full create-and-link path remains out of scope. Add `--no-country-list` to omit
the 195 candidate State labels; on the full run that drops test F1 from 0.979 to 0.939, so
the list is a real win.

The admin repo's `.venv` currently predates two dependencies declared in its `pyproject.toml`
(`rapidfuzz`, `python-dateutil`). Rather than modify that read-only repo, the sidecar layers
them into a benchmark-local directory:

```bash
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv pip install --target work/admin_agent_extra \
  --python ~/projects/python_uwazi_API/.venv/bin/python \
  "rapidfuzz==3.14.6" "python-dateutil==2.9.0.post0"
```

The method type is `admin_agent_extractor`; artifacts land under
`artifacts/methods/<method_id>/` (task prompt, extractor source, attempt with the admin
agent's tool-call transcript) and `artifacts/runs/<run_id>/` (predictions, our dataset
metrics, the admin agent's own `extraction_eval` view, and model usage). For
`upr_recommendations` the admin view covers the state property only. Nothing is written to
the `plan` / `upr` databases.

## Split policy `v2-50train`

There is no `dev` split. Each dataset has:

- **train**: 50 documents, selected deterministically and stratified (labels/classes where
  they exist, decades for years, code families for document codes, recommendation-count
  buckets plus sub-numbered recommendations for UPR; rare strata first, then proportional
  fill; within-stratum order is `sha256(seed:dataset:sharedId)`).
- **test**: every remaining eligible document, except documents that are train in another
  dataset. This strict cross-dataset isolation means no test document was seen while
  writing examples or repairing a script for another task.

`upr_state` and `upr_recommendations` share the same 50-document train pool. Seed is 42 and
the policy version (`v2-50train`) is stored in dataset manifests, methods, and runs.

## Repair loop

The LLM pipeline starts from 2 train examples, runs the script over all 50 train documents,
and scores it with the dataset's primary metric. It feeds back up to 3 representative
failures per round (crash/invalid output first, then stratified by label/error type, prompt
examples only when there are not enough other failures), for at most 4 repair rounds. It
stops after 2 consecutive rounds without improving the 48-document non-example score. The
best script is chosen on those 48 documents, then frozen and run once on test. Train (all 50
and 48 non-example), test, gap, repair rounds, failure counts, and token usage are stored in
the run artifacts.

All script inputs — prompt examples, repair feedback, train/test execution, and the viewer
preview — go through the same `InputSpec` / `prepare_input` transformation. The spec supports
text or segments, first/last N characters or segments, and position/type flags, and is stored
in `method.json`.

Cloud models run through Ollama's native `/api/chat` endpoint with `think: false` so
reasoning models return final content instead of spending the whole budget on hidden
reasoning. Use `--think low|high|max` if you want to test thinking output.

The prompt, every attempt, the frozen script, predictions, and metrics are written under
`artifacts/methods/<method_id>/` and `artifacts/runs/<run_id>/`. The viewer picks them up
automatically.

The viewer has a **Color palette** selector in the sidebar (Gruvbox Dark / Tokyo Night).
Themes are defined in `app/theme.py`; adding a palette there makes it available in the
selector without touching the pages.

`just` recipes wrap the common commands: `just sync`, `just build`, `just baselines`,
`just app`, `just inspect`, `just test`, `just lint`.

## Artifacts

`artifacts/` is generated and gitignored:

```
artifacts/catalog.db              # documents, datasets, samples (SQLite)
artifacts/states.json             # 195 UPR state/country candidates
artifacts/datasets/<id>/          # manifest.json + samples.jsonl (train/test)
artifacts/methods/<id>/           # method.json (input_spec, rounds), prompt, script, attempts/
artifacts/runs/<run_id>/          # run.json, metrics.json, predictions.jsonl, script.py
artifacts/runs/<run_id>/rounds/   # per-round script, feedback, messages, metrics, predictions
```

Old dev-split and pre-focus v2 artifacts were removed; only the final v2 datasets, methods,
and runs remain.

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
  datasets.py        dataset specs, gold extraction, v2 stratified splits, builder
  normalization.py   text/number/country normalization
  evaluation.py      Plan field metrics + UPR recommendation matching
  baselines.py       deterministic regex/heuristic baselines
  experiments.py     baseline runner (train/test) + run artifacts
  input_spec.py      shared text/segment transformation + prompt/UI excerpts
  prompts.py         LLM prompt builder
  llm.py             Ollama native `/api/chat` client (think=false)
  sandbox.py         script validation + subprocess execution
  generate.py        generate → train repair → freeze best → test pipeline
app/app.py           Streamlit viewer
app/theme.py         theme registry (Gruvbox, Tokyo Night) and semantic colors
tests/               unit tests (no Mongo required)
reports/             generated inspection report
data/                restored Mongo dumps (gitignored)
```

## Status

The v2 benchmark is working end to end: v2-50train datasets, staged train/test baselines,
the LLM generate → repair → freeze → test pipeline, and a read-only adapter that evaluates
the real Uwazi admin agent's nested extractor on the same Plan fields. Results are in
[PLAN.md §14](PLAN.md#14-implementation-status-and-v2-results).

Model note: `deepseek-v4.1-flash:cloud` is the default and works with `think: false`.
The OpenAI-compatible `/v1/chat/completions` endpoint ignored `think` and returned empty
content; the native endpoint fixed it. The pipeline records the actual model, reasoning,
`done_reason`, token usage, and every attempt.

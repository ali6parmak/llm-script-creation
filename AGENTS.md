# AGENTS.md

Guidance for AI coding agents and humans working in this repository.

## Project in one paragraph

We are benchmarking whether LLMs can write deterministic Python extraction scripts for
Uwazi documents. The data is restored locally from two Uwazi instances into MongoDB
(`plan` and `upr`). A task gives the LLM a description, optional examples/context, and asks
for `extract(document) -> dict`; the script is then run over documents and scored against
Uwazi metadata. Read `PLAN.md` first. Generated data reports live in `reports/`.

## Stack and layout

- Python 3.12, `uv` for environment/dependencies, pytest, ruff.
- MongoDB at `localhost:27017` (databases `plan`, `upr`), accessed with `pymongo`.
- LLM: Ollama native `/api/chat` API, model constant in
  `src/llm_script_creation/configuration.py` (`MODEL`), default
  `deepseek-v4.1-flash:cloud`.
- Source package: `src/llm_script_creation/`
  - `configuration.py` — env/model/paths.
  - `db.py` — Mongo connection, template IDs, entity→file→segmentation loading.
  - `models.py` — `Segment`, `Document`, metadata helpers.
  - `text.py` — document representations (`text`, `segments-tsv`, `segments-json`).
  - `inspect_data.py` — report generator / `inspect-data` CLI.
  - `catalog.py` — SQLite document/dataset/sample catalog.
  - `datasets.py` — dataset specs, gold extraction, v2 stratified splits.
  - `normalization.py` — text/number/country normalization and aliases.
  - `evaluation.py` — Plan field metrics and UPR recommendation matching.
  - `baselines.py`, `experiments.py` — deterministic baselines and run artifacts.
  - `input_spec.py` — shared input transformation for prompts, execution and the viewer.
  - `prompts.py`, `llm.py`, `sandbox.py`, `generate.py` — LLM script pipeline.
- `app/app.py` — Streamlit viewer for datasets, samples, methods, comparisons.
- `app/theme.py` — palette registry (Gruvbox Dark, Tokyo Night), CSS and status colors.
- Data dumps: `data/` (gitignored). Generated artifacts: `artifacts/`, `reports/`,
  `results/`, `work/` (all gitignored).
- Tests: `tests/`.

## Commands

```bash
# environment (sandbox note: the home uv cache is read-only in some environments)
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv sync --group dev

uv run inspect-data                 # both databases → reports/data_report.md
uv run inspect-data --db plan
uv run build-datasets               # catalog + all datasets (rebuild docs unless --skip-documents)
uv run run-baselines                # deterministic baselines on train + test
uv run generate-script --dataset plan_year   # generate → repair on train → freeze → test
uv run streamlit run app/app.py
uv run pytest -q
uv run ruff check src tests app
uv run ruff format src tests app
```

`just` recipes exist: `just sync`, `just build`, `just baselines`, `just app`,
`just inspect`, `just test`, `just lint`, `just format`.

Secrets are in `.env` (`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`). Never print, log, or commit
the API key. Do not add `.env` to artifacts. `.env` is gitignored.

## Data rules

- **Treat `plan` and `upr` as read-only.** Do not insert/update/delete in those databases.
  If an experiment needs to write, use a separate database (e.g. `benchmark`) or files.
- Use the join chain `entities.sharedId → files.entity → segmentations.fileID`.
- Segmentations attach to files; a `files` document must have `type: "document"`.
- Entity language and file language differ: `en→eng`, `fr→fra`, `es→spa`.
- We benchmark English only for now (`language: "en"` entities, `eng` files).
- Known template IDs:
  - Plan Document: `5bfbb1a0471dd0fc16ada146`
  - UPR Document: `5e57cdd6f54e0a1304c0d50d`
  - UPR Recommendation: `5d8ce04361cde0408222e9a8`
  - UPR State: `5d8cdec361cde0408222d3ec`
- UPR recommendations are linked from `Document.metadata.related_recommendations` (list of
  sharedIds), not via the `relationships` collection.
- Metadata lists can contain empty placeholders (`{"value": ""}`). Use
  `models.metadata_values` / `metadata_labels`, which filter them.
- Why we do not trust missing metadata as a negative: see PLAN.md §2.4.

## Ground truth and evaluation rules

- Gold is the existing Uwazi metadata. Do not silently "fix" gold in place.
- Evaluate Plan fields only on documents where the gold is non-empty.
- Exclude until reviewed: Plan `topics_in_related_paragraphs`, `geolocation_geolocation`,
  `field_document_info`; UPR `paragraph_number` metadata (it is mostly empty).
- UPR recommendation gold text excludes the trailing `(country);`; strip it from
  predictions before comparing.
- UPR numbers can have sub-numbers (`24.112.1`); one segmentation paragraph can contain
  several recommendations. Score paragraph-level and exact-number separately.
- Country labels in Uwazi differ from document spellings (`Cote d'Ivoire` vs
  `Côte d’Ivoire`). Normalize aliases; do not fail a prediction on accent/spacing only.
- The Uwazi `state_reviewed` label can differ from the name printed on the first page
  (`Cape Verde` vs `Cabo Verde`, `Macedonia FYR` vs `North Macedonia`, etc.). 14/165
  segmented documents have a label that does not appear verbatim on page 1; 10 need aliases
  even after accent/punctuation normalization. Use a state/country alias map for evaluation
  and for the candidate list.
- Split by document (`sharedId`), never by recommendation/segment. Keep test untouched
  until scripts are frozen.
- **Split policy `v2-50train` (seed 42):** every dataset has 50 train documents and the rest
  as test. There is no dev split. Train is deterministic and stratified (labels/classes
  where they exist; decades for years; code families for document codes;
  recommendation-count buckets + sub-numbered documents for UPR; rare strata first, then
  proportional fill; within-stratum order `sha256(seed:dataset:sharedId)`).
- **Cross-dataset isolation:** any document selected as train in one dataset is excluded
  from test in every other dataset of the same database. `upr_state` and
  `upr_recommendations` share the same 50-document train pool. The split policy version and
  train ids are stored in dataset manifests, methods, and runs.
- Inspect a sample of gold before writing a benchmark task; report known gold errors.

## Code conventions

- Type hints on public functions; prefer dataclasses in `models.py`.
- Keep data loading in `db.py`, representation/formatting in `text.py`, benchmark logic in
  `datasets.py`/`evaluation.py` (to be added), and CLI entry points in `scripts/` or
  `__main__` modules.
- Do not put benchmark-specific hacks in `db.py`. Do not scatter Mongo queries through
  experiment code; add a loader function instead.
- No notebooks for core logic. Short exploratory scripts are fine if their results can be
  regenerated; move reusable parts into the package.
- Use `pathlib.Path`, UTF-8 explicitly, and normalized paths from `configuration.py`.
- Tests must not require MongoDB. Mark integration tests with `@pytest.mark.integration`
  and skip when Mongo is unavailable. Unit tests should use synthetic `Document`/`Segment`
  objects.
- Keep `reports/data_report.md` regenerable; do not hand-edit generated reports.

## LLM / generated-script rules

- Generated code is untrusted. Run it in a subprocess with a minimal environment, import
  allowlist, and wall-clock/memory limits. Never pass `.env` or host credentials.
- Reject scripts that open files, use network/subprocess, or import outside the allowlist.
- Log every prompt, raw LLM response, generated script (including failed attempts), repair
  feedback, per-round predictions, and final frozen script under `artifacts/runs/<run_id>/`.
- Every payload must come from `input_spec.prepare_input`: prompt examples, repair feedback,
  train/test execution, and the viewer preview. Never show an excerpt from the untransformed
  document while the script runs on a truncated input. Gold stays based on the full document.
- **Repair loop:** 2 prompt examples; run over all 50 train documents; up to 3 representative
  failures per round (crash/invalid output first, then one per label/error type, prompt
  examples only if needed); at most 4 repair rounds; stop after 2 consecutive rounds without
  improving the 48-document non-example score; keep the best script on those 48 documents,
  then run it once on test.
- Do not tune prompts/examples on the test split. Examples come from train only.
- Use temperature 0 for reproducibility unless an experiment explicitly varies it; record
  the exact model id and token usage.
- Use Ollama's native `/api/chat` endpoint with `think: false` for cloud reasoning models.
  The OpenAI-compatible `/v1/chat/completions` endpoint ignored `think`; the native endpoint
  returns final content. `generate-script` defaults to `--think off` and records
  `finish_reason`, reasoning, and usage. Do not silently change the default `MODEL` constant.
- Direct per-document LLM extraction, if added, is a baseline — label it as such, do not mix
  it into script-generation results.

## Common pitfalls

- `entities.sharedId` is a string; `files.entity` is the same string, not an ObjectId.
- `files._id` is an ObjectId and `segmentations.fileID` is that ObjectId.
- Not every document has a segmentation. Filter before building datasets.
- Raw segmentation array order is not reading order. Use
  `text.ordered_segments(..., order="layout")`.
- Page headers/footers can contain useful document codes but are noisy; make it an
  experimental flag.
- Catalog `segments_json` stores `page`; raw Mongo segmentations store `page_number`.
  `Segment.from_mongo` accepts both. Do not assume one key name when reconstructing.
- Some generated scripts split headings across lines (e.g. UPR `Report of the Working
  Group ...` / `Review *` / `<state>`). A simple "next line" heuristic is fragile; provide
  segment types/positions or join adjacent header lines.
- A recommendation number may be alone in a `Table` segment; the text is in the next
  segment. Do not assume `number + text` are in one string.
- UPR document titles leak the state under review. `document_payload` omits titles by
  default; keep that default unless an experiment is explicitly testing title use.

## Current status / next tasks

The v2 benchmark is built (v2-50train datasets, viewer, train/test baselines, repair loop,
per-round artifacts). Results and known issues are in `PLAN.md` §14. Next work:

1. Fix the UPR recommendation segment script's state extraction (it returns no state on test
   documents) and harden repair against invalid Python responses.
2. Compare text vs segments/positions on the same dataset under the new loop and run
   multiple generation attempts per method.
3. Add a plain-text/markdown run summary alongside the Streamlit viewer.
4. Review gold noise (`upr_recommendations` sub-numbers, Plan field coverage) before strong
   conclusions.

When you finish a meaningful step, regenerate `reports/data_report.md` if data assumptions
changed, run `ruff` and `pytest`, and record any new data caveat in `PLAN.md` §2.4.

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
- LLM: Ollama's OpenAI-compatible API, model constant in
  `src/llm_script_creation/configuration.py` (`MODEL`), default
  `deepseek-v4.1-flash:cloud`.
- Source package: `src/llm_script_creation/`
  - `configuration.py` — env/model/paths.
  - `db.py` — Mongo connection, template IDs, entity→file→segmentation loading.
  - `models.py` — `Segment`, `Document`, metadata helpers.
  - `text.py` — document representations (`text`, `segments-tsv`, `segments-json`).
  - `inspect_data.py` — report generator / `inspect-data` CLI.
- Data dumps: `data/` (gitignored). Outputs: `reports/`, `results/`, `work/` (gitignored).
- Tests: `tests/`.

## Commands

```bash
# environment (sandbox note: the home uv cache is read-only in some environments)
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv sync --group dev

uv run inspect-data                 # both databases → reports/data_report.md
uv run inspect-data --db plan
uv run pytest -q
uv run ruff check src tests
uv run ruff format src tests
```

`just` recipes exist: `just sync`, `just inspect`, `just test`, `just lint`, `just format`.

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
  until scripts are frozen. Use seed 42 and commit split manifests.
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
  feedback, and final frozen script under `results/<run-id>/`.
- Do not tune prompts/examples on the test split. Examples come from train only.
- Use temperature 0 for reproducibility unless an experiment explicitly varies it; record
  the exact model id and token usage.
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
- A recommendation number may be alone in a `Table` segment; the text is in the next
  segment. Do not assume `number + text` are in one string.
- UPR document titles leak the state under review. `document_payload` omits titles by
  default; keep that default unless an experiment is explicitly testing title use.

## Current status / next tasks

M0 (inspection) is complete. Next milestones are in `PLAN.md` §10 and §13:

1. `datasets.py` — task definitions, gold extraction, split manifests, JSONL builder.
2. `evaluation.py` — Plan field metrics, UPR recommendation alignment/matching.
3. Tests and a rule-based baseline through the eval harness.
4. LLM client + prompt templates + sandbox + repair loop.

When you finish a meaningful step, regenerate `reports/data_report.md` if data assumptions
changed, run `ruff` and `pytest`, and record any new data caveat in `PLAN.md` §2.4.

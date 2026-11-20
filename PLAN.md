# PLAN — LLM-written extraction scripts for Uwazi documents

**Status:** end-to-end prototype built (catalog/datasets, viewer, baselines, LLM pipeline). First experiments run; next is prompt/model iteration and deeper analysis.
**Last updated:** 2026-11-20.

## 1. Context and goal

HURIDOCS partners store entities and their source documents in Uwazi. Today, extracting
metadata from those documents is manual. We want to know whether an LLM can write a small,
deterministic extraction *script* for a task — instead of extracting document-by-document —
and which context the LLM needs to write a good script.

The workflow we are benchmarking:

```
task description + examples + document/excerpt
        │
        ▼
   LLM generates Python `extract(document)` script
        │
        ▼
   script runs locally over every document
        │
        ▼
   predictions compared with Uwazi ground truth
```

We are **not** training a model or building a production extractor. We are measuring the
feasibility, failure modes, and context sensitivity of LLM-generated extraction programs.

## 2. What the data looks like (M0 findings)

Full generated report: `reports/data_report.md` (run `uv run inspect-data`).
Raw numbers: `reports/data_report.json`.

### 2.1 Instances

MongoDB at `localhost:27017`:

| Database | Purpose | Entities | Ready segmentations |
|---|---|---:|---:|
| `plan` | Plan International corpus | 1,308,420 | 31,650 |
| `upr` | Universal Periodic Review corpus | 293,972 | 349 |

The relevant join chain is:

```
entities.sharedId  →  files.entity  →  segmentations.fileID
```

- `segmentations` are attached to **files**, not entities.
- An entity may own several files (one per language plus custom images).
  Select the file whose `file.language` matches the entity language:
  `en → eng`, `fr → fra`, `es → spa`.
- `files.type == "document"` distinguishes the source PDF from custom files.
- `entity.language` is `en`/`fr`/`es` for Plan and `en`/`fr` for UPR.

### 2.2 Plan dataset

Template `Document` = `5bfbb1a0471dd0fc16ada146`.

- **10,596** English Document entities (**10,537** have an English document file with a ready
  segmentation). Most titles exist in `en`, `fr`, and `es`; we start with English only.
- Properties and ground-truth coverage over the 10,596 English documents:

| Property | Type | Filled | Coverage | Notes |
|---|---|---:|---:|---|
| `body` | select | 10,569 | 99.8% | 54 observed label variants; dictionary has 19 |
| `year` | numeric | 10,449 | 98.6% | 1948–2024 |
| `document_code` | text | 8,881 | 83.8% | e.g. `A/HRC/RES/14/1` |
| `document_type` | select | 3,170 | 29.9% | 9 labels |
| `legal_status` | select | 3,170 | 29.9% | 3 labels |
| `means_of_adoption` | select | 3,169 | 29.9% | 3 labels |
| `field_month` | numeric | 1,720 | 16.2% | 1–12 |
| `field_document_number` | text | 1,428 | 13.5% | often a page/section number |
| `field_document_info` | text | 565 | 5.3% | definition unclear; skip for now |
| `topics_in_related_paragraphs` | relationship | 10,591 | 100% | links to Paragraph entities, not direct text |
| `geolocation_geolocation` | geolocation | 0 | 0% | empty; skip |

The three select fields (`document_type`, `legal_status`, `means_of_adoption`) are filled
together for the same ~3,170 documents, which are mostly resolutions/treaties.

**Important:** the `Body` dictionary in Uwazi is incomplete/stale relative to the metadata.
The metadata contains 54 distinct labels such as
`Special Rapporteur on the right to education` while the dictionary only has generic values
like `Special Procedures`. Do not assume the dictionary is the full label set.

### 2.3 UPR dataset

Templates:

| Template | ID | Entities |
|---|---|---:|
| Recommendation | `5d8ce04361cde0408222e9a8` | 289,940 |
| Document | `5e57cdd6f54e0a1304c0d50d` | 1,352 |
| State | `5d8cdec361cde0408222d3ec` | 392 (196 English) |
| Voluntary Pledge | `5d8ce05361cde0408222eb02` | 2,244 |

- **676** English Document entities; **165** have an English segmented file. Of those,
  163 are Cycle 3 and 2 are Cycle 2. This is the UPR benchmark universe.
- **39,131** English Recommendation entities are linked to those 165 documents
  (119–389 per document, avg 237.2).
- Ground truth available for recommendations linked to segmented docs:
  - 38,774 have recommendation text
  - 38,428 have a numeric title such as `96.1`
  - 2,583 have a sub-number such as `24.112.1`
  - 38,756 have a recommending state
  - only 1 has more than one recommending state in metadata
  - 38,562 have a `response`
- **196** English `State` entities. One is `_Unknown`; the remaining 195 have ISO country
  codes. Use 195 countries as the candidate list and drop `_Unknown`.

Document structure that matters for extraction:

- Recommendation paragraphs look like
  `96.1 Take steps ... (New Zealand);`
  or, in tables, the number and body can be in separate segments.
- **One segment can contain several recommendations**: e.g.
  `24.112 Stop arbitrarily ... (Iceland); End arbitrary detention ... (Australia);`.
  Uwazi's ground truth numbers these as `24.112.1` and `24.112.2`, while the segmentation
  only shows `24.112`. There are 231 segments containing more than one recommendation
  number across the 165 documents.
- Recommendation text in ground truth excludes the trailing country parenthetical and the
  trailing semicolon. A script must strip them.
- Country spellings differ: metadata says `Cote d'Ivoire`, the document says
  `Côte d’Ivoire`; aliases/normalization are required.
- The first page almost always names the state under review, but often in a different form
  than the Uwazi label. 14/165 labels do not appear verbatim on page 1; after basic
  accent/punctuation normalization, 10 still need aliases: `Republic of Congo`/`Congo`,
  `Democratic Republic of Congo`/`Congo`, `Macedonia FYR`/`North Macedonia`,
  `Cape Verde`/`Cabo Verde`, `DPR Korea`/`Democratic People's Republic of Korea`,
  `St Lucia`/`Saint Lucia`, `Sao Tome & Principe`/`Sao Tome and Principe`,
  `Laos`/`Lao People's Democratic Republic`, `St Kitts & Nevis`/`Saint Kitts and Nevis`,
  `St Vincent & the Grenadines`/`Saint Vincent and the Grenadines`.
  An alias map is required for evaluation (and helps prompts).
- Roughly 123 recommendation texts (0.3% of 39,131) contain other paragraph references or
  merged recommendations. Spot-check this kind of gold noise before scoring text exactness;
  it is mostly from references such as "target 16.10", not a systematic merge.
- Exactly 28 ground-truth recommendation numbers could not be found in the segmentation
  text by a broad number scan (0.07% of 38,428). These are known alignment misses, not
  successful extractions.

### 2.4 Ground-truth caveats

- Uwazi metadata is the only gold we have. It was human-entered and is occasionally wrong
  or missing. Missing metadata is **not** a negative label: evaluate only documents where
  the gold field is non-empty, unless we deliberately add a separate "missing" test.
- The Plan `topics_in_related_paragraphs` field is a relationship to Paragraph entities and
  needs paragraph matching, not direct text extraction. Exclude it from the first benchmark.
- `field_document_info` has low coverage and an unclear definition. Exclude until reviewed.
- `geolocation_geolocation` is empty. Exclude.
- For UPR, the recommendation number is in the Recommendation entity `title`; the
  `paragraph_number` metadata property is mostly empty and should not be used as gold.

## 3. Research questions

- **RQ1 — Feasibility:** can current LLMs write an extraction script that runs and produces
  useful output for these real tasks?
- **RQ2 — Context:** which inputs improve the generated script? Candidate factors:
  plain text vs segment types vs page/position/size, country list, example count, prompt
  wording, explicit edge-case rules.
- **RQ3 — Failure modes:** where do scripts break? (multi-recommendation paragraphs,
  table cells, wrapped lines, page headers, aliases, empty values, regex overfitting).
- **RQ4 — Scripts vs direct extraction:** how does a generated deterministic script compare
  with asking an LLM to extract each document directly? (Optional baseline; same model and
  budget.)
- **RQ5 — Cost/reliability:** how many generation/repair attempts, tokens, and runtime are
  needed to reach a usable script?

## 4. What counts as an LLM-created script

### 4.1 Contract

Every task asks the model for one Python file defining:

```python
def extract(document: dict) -> dict:
    ...
```

The function receives a JSON-serializable document payload and returns a JSON-serializable
dict. No file I/O, no network, no subprocesses, no third-party packages.

**Document payload (default):**

```python
{
    "id": "...",             # opaque bookkeeping id
    "language": "en",
    "page_count": 18,
    "text": "...",           # text representation
    # or:
    "segments": [
        {"text": "...", "type": "Text", "page": 10, "top": 74.0,
         "left": 142.0, "width": 342.0, "height": 33.0}
    ],
}
```

The entity title is **not** included by default. UPR titles literally contain the state
under review (`Barbados - Third Cycle`), so including them would leak the answer for the
"state in the first page" task. Some variants may include it explicitly as a separate
ablation.

**Plan output:**

```python
{"value": "Resolution"}              # select / text / number
{"value": 2018}
{"value": None}                      # explicitly unknown
```

**UPR output:**

```python
{
  "state_under_review": "Barbados",
  "recommendations": [
    {
      "number": "96.1",
      "text": "Take steps towards the full abolition of the death penalty ...",
      "countries": ["New Zealand"]
    }
  ]
}
```

`countries` should use the canonical labels/ISO codes from the supplied Uwazi state list;
evaluation will normalize aliases. The script may report base numbers (`24.112`) or
sub-numbers (`24.112.1`); evaluation scores both (paragraph-level and exact-number).

### 4.2 Lifecycle

1. **Generate:** prompt → LLM → parse one Python code block.
2. **Static check:** compile the script, enforce the function contract, lint imports.
3. **Smoke test:** run on 3–5 dev documents; validate output schema and record errors.
4. **Repair:** if the script crashes, times out, or returns invalid shapes, send the error
   and a failing example back to the LLM (max 3 repairs). Keep every attempt and the errors.
5. **Freeze:** the final script is stored verbatim with its prompt and model configuration.
6. **Evaluate:** run the frozen script on the test split; never regenerate after seeing test
   results.

The key scientific unit is the **frozen script**, not a per-document LLM call. The script
can be inspected, versioned, and run cheaply over thousands of documents.

## 5. Benchmarks

### 5.1 Plan metadata tasks

One script per field (primary); one multi-field script (secondary).

| ID | Target | Type | Eval docs | Candidates |
|---|---|---|---:|---|
| P1 | `document_type` | single-label select | 3,170 | 9 labels |
| P2 | `legal_status` | single-label select | 3,170 | 3 labels |
| P3 | `means_of_adoption` | single-label select | 3,169 | 3 labels |
| P4 | `year` | integer | 10,449 | 1948–2024 |
| P5 | `document_code` | string | 8,881 | open |
| P6 | `body` | single-label select | 10,569 | observed labels |
| P7 | `field_month` | integer 1–12 | 1,720 | 12 |
| P8 | `field_document_number` | string | 1,428 | open |

P1–P3 are classification. P4/P7 are constrained numerics. P5/P8 are string extraction from
headers/footers. P6 is a hard many-class problem with label variants.

### 5.2 UPR tasks

One core script returning the full record, scored per field:

| ID | Target | Description |
|---|---|---|
| U1 | `state_under_review` | state from the first page (not the entity title) |
| U2 | `recommendations[].number` | e.g. `96.1`, `24.112`, `24.112.1` |
| U3 | `recommendations[].text` | text without trailing country/semicolon |
| U4 | `recommendations[].countries` | recommending state(s), canonicalized |

U2–U4 are one task because they come from the same recommendation paragraphs. Field-level
metrics let us see whether adding layout/country context helps number parsing but hurts
text, etc.

The core difficulty is paragraph splitting: one segment/paragraph may contain several
recommendations, and gold uses sub-numbers when they share a paragraph number.

### 5.3 Splits

- **Plan:** 70/15/15 split by `Document.sharedId`, seed 42. For per-field tasks, split the
  subset of documents with non-empty gold; stratify by field label when feasible.
- **UPR:** 60/20/20 split by `Document.sharedId`, seed 42 (~99/33/33 docs). Group by
  document so recommendations from one document never cross splits.
- Keep a **dev** split for prompt/example iteration and a **test** split that is touched
  only by frozen scripts. The split manifest is a committed artifact (IDs + seed).
- Few-shot examples are always selected from train only.
- For UPR, stratify by cycle where possible (only 2 Cycle-2 documents exist; document the
  imbalance rather than pretending the split is balanced).

## 6. Representations and experimental factors

The main question from `TASKS.md`: does providing segment position/size help?

| Factor | Levels |
|---|---|
| Representation | `text`; `segments-tsv` (no position); `segments-tsv` (+ position); `segments-json` |
| Segment type | included / excluded |
| Page headers & footers | included / excluded |
| Page markers in text | on / off |
| Country list (UPR) | on / off |
| Examples | 0 / 2 / 3 |
| Prompt variant | concise / detailed / detailed + edge cases + candidate labels |

Current renderers in `text.py` support `text`, `segments-tsv`, `segments-json` and the
position/type/noise flags. Default order is `layout` (sort by page, top, left); raw Uwazi
array order is not reading order.

A few known representation issues to study rather than hide:

- Headers can be split into pieces (`A` + `/HRC/38/12`) that are not adjacent in pure
  top-to-bottom order.
- Table segments can contain a recommendation number alone, with the text in the next
  segment.
- Multi-column PDFs can interleave when sorted by `top` then `left`. A column-aware
  renderer is a possible follow-up, not an M1 requirement.
- `segments-tsv` is compact; `segments-json` is easier for generated code but token-heavy.

## 7. Prompt design

The prompt has four parts:

1. **System/role:** "write a deterministic Python extractor, no I/O, return JSON".
2. **Task spec:** field definitions, output schema, exact contract, edge cases.
3. **Context:** candidate labels (Plan select fields), country list (UPR), 0–3 examples.
4. **Constraints:** stdlib only; no network/files; handle missing values; return one record
   per document/list item.

Task descriptions must encode the data quirks we found, e.g.:

- UPR: a paragraph can contain several recommendations separated by `;`, each ending with
  a country in parentheses; split them and assign `.1`, `.2`, ... when they share a number.
- UPR: strip the trailing country and semicolon from recommendation text.
- UPR: use the first page for the state under review.
- Plan: `document_code` often appears in the page header/footer and may be split across
  segments; normalize spacing.
- Plan: only output labels from the supplied candidate list when a list is supplied;
  otherwise output the text as it appears.

We will compare at least three prompt variants on the dev split before freezing test runs.

## 8. Script execution and safety

Generated code is untrusted. The runner will:

- execute in a separate `python` subprocess with a temporary working directory and a minimal
  environment (no `.env`, no API keys);
- restrict imports to an allowlist (`re`, `json`, `unicodedata`, `collections`, `itertools`,
  `functools`, `string`, `datetime`, `math`, `statistics`, `typing`, `dataclasses`);
- block access to `os`, `sys`, `subprocess`, `socket`, `pathlib`, `open`, `eval`, `exec`,
  `__import__` (except through the hook), and dynamic imports;
- apply wall-clock and memory limits;
- validate that output is JSON-serializable and has the expected shape.

This is a benchmark sandbox, not a hardened security boundary. Never run generated scripts
with access to the host filesystem or secrets.

## 9. Evaluation

### 9.1 Normalization

- Unicode NFKC, lowercase, collapse whitespace, normalize curly quotes/dashes, strip
  trailing punctuation.
- Plan `document_code`: remove spaces, uppercase, compare normalized forms.
- UPR text: strip trailing `(country);`, normalize spacing, compare token F1.

### 9.2 Metrics

**Plan (per field):**

- Accuracy; macro precision/recall/F1 for select fields; per-label confusion.
- Numeric exact match; report year/month separately.
- Text: exact normalized match + token F1 (report both).
- Null/abstention: predicted null on non-null gold = error; gold null is excluded by design.

**UPR:**

- **State under review:** exact match after alias normalization; top-1 accuracy.
- **Paragraph detection:** predicted base numbers vs gold base numbers (precision/recall/F1).
- **Recommendation matching:** align predictions to gold within each base number using text
  similarity + country; report:
  - recommendation precision/recall/F1 (match threshold, e.g. text token F1 ≥ 0.85),
  - text token F1 on matched pairs,
  - exact/normalized number accuracy,
  - country accuracy (set match) on matched pairs,
  - exact-number accuracy including sub-numbers.
- **Aggregation:** micro and macro over documents; bootstrap 95% CI over documents.

### 9.3 Baselines

- Majority class for Plan select fields (shows how much is trivial).
- Trivial null/empty baseline.
- Optional direct LLM extraction baseline on a subset (one call per document) to compare
  script-based vs per-document extraction under the same model.

## 10. Experiment phases

| Phase | Work | Acceptance |
|---|---|---|
| M0 | Data inspection, schema map, report | done |
| M1 | Dataset builders, splits, text renderers, eval functions, tests | deterministic dataset artifacts; unit tests green |
| M2 | LLM client, prompt templates, sandbox, repair loop, run logging | one end-to-end smoke run on Plan P4 and UPR U1 |
| M3 | Plan experiments: fields × representations × prompts | frozen scripts + metrics table + failure analysis |
| M4 | UPR experiments: core task × context factors | metrics + alignment error analysis |
| M5 | Direct-extraction baseline (optional), model replication, write-up | report with RQ answers + reproducible commands |

Work is data-first: build the harness and check a trivial rule-based extractor before
spending LLM calls.

## 11. Reproducibility and logging

Each run gets `results/<run-id>/` containing:

- `config.json`: model, prompt hash, representation, split, seed, temperature, max tokens.
- `prompt.txt`: exact prompt sent.
- `attempts/attempt-<n>.py`: every generated script, including failed ones.
- `attempts/attempt-<n>.json`: raw response, validation errors, repair feedback.
- `final_script.py`: frozen script.
- `predictions.jsonl`: one row per document with prediction, gold, latency, errors.
- `metrics.json`: all metrics + bootstrap CIs.
- `logs.txt`: human-readable run log.

Rules:

- Do not tune on test. Log every prompt/script, including failures.
- Pin the model id and record the Ollama endpoint response model.
- Use fixed seeds and deterministic split manifests.
- Keep raw LLM outputs (not just the parsed script) for audit.

## 12. Open questions / decisions needed

1. **One script per field vs one multi-field script for Plan.** Current plan: per-field
   first, multi-field as a secondary task.
2. **UPR sub-numbers.** Should the script be required to emit `24.112.1`, or is base number
   `24.112` plus one record per recommendation enough? Current plan: score both, strict
   exact-number metric reported separately.
3. **State under review leakage.** We omit the entity title from the payload; confirm that
   is the intended interpretation of "state in the first page".
4. **Plan candidate labels.** The `Body` dictionary is incomplete; use train-observed labels
   or a curated list? Current plan: include a candidate list for select fields but allow
   free text for `body`, and score against observed gold.
5. **Gold quality.** Should someone review a sample of ground truth before the benchmark?
   Recommended for UPR recommendation numbers and Plan `document_code`.
6. **Budget.** How many LLM calls / tokens are acceptable for M2–M4? Start with dev subsets
   (≤ 30 docs per smoke run) and scale only after the harness is trusted.
7. **Language scope.** Start with English; add French/Spanish later only if the benchmark
   results justify it.

## 13. Immediate next steps (M1)

1. Add `datasets.py` with:
   - `PlanTask` / `UprTask` definitions and gold extraction;
   - deterministic split manifests keyed by `sharedId`;
   - JSONL dataset builder (`documents`, `gold`, `split`).
2. Add `evaluation.py` with Plan field metrics and UPR recommendation alignment.
3. Add `normalization.py` with:
   - state/country alias maps (Uwazi label ↔ document form ↔ ISO code);
   - text, number, and punctuation normalization used by evaluation.
4. Add tests for dataset invariants and evaluation edge cases.
5. Add `scripts/build_dataset.py` / `uv run build-dataset`.
6. Add a tiny rule-based extractor and run it through the eval harness (no LLM) to validate
   the plumbing and get a lower bound.
7. Then build the LLM client + sandbox (M2).

## Appendix A — quick data access

```python
from llm_script_creation.db import (
    PLAN_DB, UPR_DB, PLAN_DOCUMENT_TEMPLATE, UPR_DOCUMENT_TEMPLATE,
    get_db, iter_documents,
)

for doc in iter_documents(get_db(PLAN_DB), PLAN_DOCUMENT_TEMPLATE, language="en", limit=5):
    print(doc.title, doc.file_id, doc.has_segmentation, len(doc.segments))
```

The inspection report is generated with:

```bash
uv run inspect-data          # writes reports/data_report.md and .json
uv run inspect-data --db plan
```


## 14. Implementation status and first results (2026-11-20)

### What is built

| Piece | Where | Status |
|---|---|---|
| Document catalog + datasets + splits + manifests | `src/llm_script_creation/catalog.py`, `datasets.py`, `artifacts/` | done |
| Streamlit viewer (datasets, samples, methods, runs, compare) | `app/app.py` | done |
| Rule-based baselines | `baselines.py`, `experiments.py` | done |
| Evaluation (Plan fields + UPR recommendations) | `evaluation.py`, `normalization.py` | done |
| LLM prompt builder + client + sandbox + repair loop | `prompts.py`, `llm.py`, `sandbox.py`, `generate.py` | done |
| Method/run artifacts | `runs.py`, `artifacts/methods`, `artifacts/runs` | done |

Commands:

```bash
uv run build-datasets
uv run run-baselines --split dev
uv run generate-script --dataset plan_year --split dev --model qwen2.5-coder:14b
uv run streamlit run app/app.py
```

The catalog is ~800 MB and lives in `artifacts/catalog.db` (gitignored). Dataset manifests
and sample ids are in `artifacts/datasets/`. Methods and runs are self-describing JSON
directories under `artifacts/methods/` and `artifacts/runs/`; the viewer reads them directly.

Dataset sizes after intersecting with segmented English documents:

| Dataset | Samples |
|---|---:|
| `plan_document_type` | 3,112 |
| `plan_legal_status` | 3,112 |
| `plan_means_of_adoption` | 3,111 |
| `plan_year` | 10,391 |
| `plan_document_code` | 8,840 |
| `plan_body` | 10,512 |
| `plan_field_month` | 1,716 |
| `plan_document_number` | 1,388 |
| `plan_all_fields` | 2,792 |
| `upr_state` | 165 |
| `upr_recommendations` | 164 (38,774 recommendations) |

### First dev-split results

Baselines (deterministic regex/heuristics):

| Dataset | Metric |
|---|---:|
| `plan_year` | accuracy 0.926 |
| `plan_document_code` | accuracy 0.944 |
| `upr_state` | accuracy 1.000 |
| `upr_recommendations` | P 0.959 / R 0.703 / F1 0.811 |

First LLM-generated scripts (dev, `qwen2.5-coder:14b` unless noted):

| Dataset | Result |
|---|---|
| `plan_year` | accuracy 0.603 (first 4-digit year anywhere; ignores date hints) |
| `plan_year` (llama3.2:3b) | accuracy 0.610 |
| `upr_state` | 0.000 — generated scripts returned empty or missed the split title line |
| `upr_recommendations` | P 0.892 / R 0.289 / F1 0.437; country 0.921; exact number 0.983; state 0.000 |

These are **not** tuned final results. They already show the main failure modes:

- LLM scripts miss the difference between "first year in the text" and "document year".
- Plain-text UPR titles can be split across lines (`Report of the Working Group ...` /
  `Review *` / `Albania`); generated scripts assume the state is the immediate next line.
- A single recommendation paragraph may contain several recommendations, and UPR state
  extraction and paragraph splitting need explicit structural instructions.
- A valid JSON shape can still be semantically wrong; the smoke repair loop now checks a
  few dev examples and feeds mismatches back to the model.

### Model note

`deepseek-v4.1-flash:cloud` (and the other cloud models tested) spent the entire completion
budget on hidden reasoning and returned empty `content` for these prompts, even at 16k
tokens. The pipeline records `finish_reason`, `reasoning`, token usage, and every attempt.
For the first experiments I used local coder/instruct models via `--model`
(`qwen2.5-coder:14b`, `llama3.2:3b`). The configured `MODEL` constant remains the default;
switching the default is a deliberate decision, not a silent change.

### Known issues / next

1. Improve prompts/representations for UPR state and recommendation extraction; segment
   types are likely the right representation for split title lines.
2. Test `segments` vs `text` on the same dataset; current UPR runs are text-only.
3. Run frozen scripts on the test split (not done yet).
4. Add a run summary/report command so results do not only live in the UI.
5. Review the `upr_recommendations` gold noise and sub-number matching before drawing strong
   conclusions.

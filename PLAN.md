# PLAN — LLM-written extraction scripts for Uwazi documents

**Status:** v2 benchmark built (v2-50train stratified splits, shared `InputSpec`, train/test
baselines, full-train repair loop, per-round artifacts). Final v2 results are in §14; next is
fixing the UPR segment-state failure and hardening repair against invalid responses.
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

1. **Generate:** prompt with 2 train examples and the shared `InputSpec` → LLM → parse one
   Python code block.
2. **Static check:** compile the script, enforce the function contract, lint imports.
3. **Train sweep:** run the script over all 50 train documents using the same `InputSpec` as
   the prompt examples; score with the dataset's primary metric.
4. **Repair:** collect failures; select up to 3 per round (crash/invalid output first, then
   one per label or error type, excluding the 2 prompt examples where possible); send a
   short excerpt of the transformed input, gold, actual output and traceback back to the
   LLM. Max 4 rounds. Stop after 2 consecutive rounds without improving the 48-document
   non-example score. Keep the best script by that score.
5. **Freeze:** the best script is stored verbatim with its prompt, `InputSpec`, every round,
   and usage.
6. **Evaluate:** run the frozen script once on the test split; never regenerate after seeing
   test results. Report train (all 50 and 48 non-example), test, and the gap.

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

### 5.3 Splits (`v2-50train`)

- **Every dataset:** 50 train documents and the rest as test. There is no dev split.
- Split by `Document.sharedId`, never by recommendation/segment. Seed 42.
- Train is deterministic and stratified, not pure random:
  - Plan select fields and `body`: gold label.
  - Plan `year`: decade; `field_month`: month.
  - Plan `document_code`: code family (`A/HRC`, `A/RES`, `S/RES`, `E/CN.4`, ILO `C...`).
  - Plan `field_document_number`: numeric / mixed / text.
  - UPR: recommendation-count bucket × sub-numbered recommendations.
  - Rule: strata sorted rarest-first; one document per stratum; remaining slots proportional
    to stratum size (largest remainder); within-stratum order
    `sha256(seed:dataset:sharedId)`. Rare/hard cases are therefore represented by design.
- **Cross-dataset isolation:** any document selected as train in one dataset is excluded from
  test in every other dataset of the same database. This is why test counts are smaller than
  the eligible counts (e.g. `plan_year` 9,965 test of 10,391 eligible).
- `upr_state` and `upr_recommendations` share the same 50-document train pool.
- Few-shot examples are always selected from train only (2 examples; they are part of the 50
  and are excluded when reporting the 48 non-example train score).
- The split policy version and train ids are stored in every dataset manifest, method, and
  run.

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

Prompt variants are compared on train (all 50 and the 48 non-example documents); the test
split is touched only once by the best frozen script. Prompt examples, repair feedback,
train/test execution, and the viewer preview all use the same `InputSpec` /
`prepare_input(...)` transformation.

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

Each method gets `artifacts/methods/<method_id>/` and each run
`artifacts/runs/<run_id>/`:

- `method.json`: model, prompt hash, `input_spec`, examples, repair rounds, best round,
  token usage, final train/test scores.
- `prompt.txt`: exact initial prompt sent.
- `attempts/attempt-<n>.json`: round, raw response, generated script, validation errors,
  repair feedback, per-round scores.
- `script.py`: frozen best script.
- `metrics.json`: primary train (all 50 and 48 non-example), test, gap, repair rounds,
  failure counts, token usage, `InputSpec`, and split-policy version.
- `predictions.jsonl`: one row per train and test document with prediction, gold, error.
- `rounds/round-<n>/`: script, response, feedback used, full messages, per-round metrics,
  and predictions for every repair round.

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
6. **Budget.** How many LLM calls / tokens are acceptable for M2–M4? The v2 loop uses 2
   examples and at most 4 repair rounds per method; use `--limit` for smoke runs and scale
   only after the harness is trusted.
7. **Repair quality.** The first v2 runs often stop improving after the first script. Is the
   bottleneck the failure selection, the feedback format, or the model's ability to
   generalise from the transformed excerpts?
8. **Language scope.** Start with English; add French/Spanish later only if the benchmark
   results justify it.

## 13. Immediate next steps (v2)

1. Investigate the repair loop: inspect failed rounds and compare failure-selection
   strategies (crash-first, label-stratified, lowest-F1 first).
2. Compare text vs segments/positions on the same dataset under the new loop; the train set
   is stratified, so train/test gaps can be negative (train is intentionally harder).
3. Run multiple generation attempts per method to separate prompt/representation effects
   from code-generation variance.
4. Add a plain-text/markdown run summary alongside the Streamlit viewer.
5. Review gold noise (`upr_recommendations` sub-numbers, Plan field coverage) before strong
   conclusions.

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


## 14. Implementation status and v2 results (2026-11-20)

### What is built

| Piece | Where | Status |
|---|---|---|
| Document catalog + v2 datasets + stratified splits + manifests | `catalog.py`, `datasets.py`, `artifacts/` | done |
| Streamlit viewer (datasets, samples, methods, compare) | `app/app.py` | done |
| Rule-based baselines on train + test | `baselines.py`, `experiments.py` | done |
| Evaluation (Plan fields + UPR recommendations) | `evaluation.py`, `normalization.py` | done |
| Shared `InputSpec` for examples/repair/execution/UI | `input_spec.py`, `prompts.py`, `generate.py`, `app/app.py` | done |
| LLM client + sandbox + full-train repair loop | `llm.py`, `sandbox.py`, `generate.py` | done |
| Per-round and final artifacts | `runs.py`, `artifacts/methods`, `artifacts/runs` | done |
| Viewer theming | `app/theme.py`, `.streamlit/config.toml` | done |
| Real admin agent's nested extractor (read-only sidecar) | `admin_agent_adapter/`, `scripts/admin_agent_sidecar.py` | phase 1 (Plan single fields) |

Commands:

```bash
export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
uv run build-datasets --skip-documents
uv run run-baselines
uv run generate-script --dataset plan_year --examples 2 --max-repair-rounds 4
uv run streamlit run app/app.py
```

The catalog is ~800 MB and lives in `artifacts/catalog.db` (gitignored). v2 dataset manifests
and sample ids are in `artifacts/datasets/`. Methods and runs are self-describing JSON
directories under `artifacts/methods/` and `artifacts/runs/`; the viewer reads them directly.
Old dev-split and pre-focus v2 artifacts were removed; `artifacts/` contains only the final
v2 datasets, methods, and runs.

Dataset sizes after v2 splitting (`train` is always 50; `test` excludes cross-dataset train
documents):

| Dataset | Total | Test |
|---|---:|---:|
| `plan_document_type` | 2,775 | 2,725 |
| `plan_legal_status` | 2,775 | 2,725 |
| `plan_means_of_adoption` | 2,774 | 2,724 |
| `plan_year` | 10,015 | 9,965 |
| `plan_document_code` | 8,508 | 8,458 |
| `plan_body` | 10,137 | 10,087 |
| `plan_field_month` | 1,565 | 1,515 |
| `plan_field_document_number` | 1,241 | 1,191 |
| `plan_all_fields` | 2,493 | 2,443 |
| `upr_state` | 165 | 115 |
| `upr_recommendations` | 164 | 114 |

### v2 baselines

| Dataset | Metric | Train (50) | Test | Gap |
|---|---|---:|---:|---:|
| `plan_year` | accuracy | 0.920 | 0.932 | -0.012 |
| `plan_document_code` | accuracy | 0.720 | 0.949 | -0.229 |
| `upr_state` | accuracy | 1.000 | 0.991 | +0.009 |
| `upr_recommendations` | F1 | 0.920 | 0.883 | +0.037 |

`upr_recommendations` test: P 0.979 / R 0.804 / F1 0.883; text F1 0.999; country 0.945;
exact number 0.938; state 0.991.

### v2 LLM scripts (`deepseek-v4.1-flash:cloud`, `think: false`, 2 examples)

All runs use the same split, the shared `InputSpec`, up to 4 repair rounds, and a best-script
choice on the 48 non-example train documents. Repair feedback and prompt examples now center
their excerpt on the relevant transformed-input content (see the focus note below).

| Dataset / input | Train (50) | Train (48 non-example) | Test | Gap (50→test) | Repairs | Best round | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `plan_year` / text | 0.720 | 0.708 | 0.826 | -0.106 | 3 | 1 | 22,302 |
| `plan_year` / segments + position/type | **0.900** | **0.896** | **0.927** | -0.027 | 4 | 3 | 48,490 |
| `plan_document_code` / text | 0.760 | 0.750 | **0.972** | -0.212 | 4 | 4 | 29,139 |
| `upr_state` / text | 1.000 | 1.000 | 0.991 | +0.009 | 0 | 0 | 3,736 |
| `upr_recommendations` / text | 0.916 | 0.918 | 0.880 | +0.037 | 3 | 1 | 63,441 |
| `upr_recommendations` / segments + position/type | **0.961** | **0.962** | **0.945** | +0.016 | 3 | 1 | 71,545 |

`upr_recommendations` test details:
- text: P 0.968 / R 0.806 / F1 0.880; text F1 0.998; country 0.955; exact number 0.982;
  state 0.895.
- segments + position/type: P 0.952 / R 0.938 / F1 0.945; text F1 0.993; country 0.927;
  exact number 0.997; state 0.000.

### Admin-agent extractor (read-only adapter, phase 1)

A different strategy from `generate-script`: instead of prompting our own pipeline, run the
real Uwazi admin agent's nested extractor subagent
(`uwazi_admin_agent.use_cases.author_extractor_tool.build_extractor_agent`) against our Plan
data, then execute the returned `def extract(text, ctx)` with the admin agent's own
`extractor_sandbox`. The admin agent is used as-is: its system prompt, its tools, its model
(`deepseek-v4.1-flash:1017-cloud` via `OllamaAdapter`), and its `benchmark_extractor`.

What is held fixed:

- **Condition (default, v3):** no title — `ctx["title"]` and every entity title are empty,
  while other Uwazi metadata remains visible. The target property is removed from
  `ctx.metadata` at execution time (production-realistic: the value being extracted is
  empty), so a metadata shortcut cannot score. Train rows still carry the target during
  authoring because they are the admin agent's labeled samples; the task tells the extractor
  not to read it. Historical v1/v2 runs had the title visible.
- **Source:** the primary-document segmentation, with exactly the admin agent's
  document-extraction input: `"\n".join(p["text"] for p in seg["paragraphs"])` (catalog
  segments are already page/reading ordered). `ctx` is the admin agent's own
  `build_context(entity)`.
- **Split:** the existing v2-50train split. The authoring request contains the 50 train
  documents only; test documents are written only to the execution payload.
- **Read-only:** the sidecar has no Mongo/Uwazi client and no network fetcher. Every port
  write raises `NotImplementedError`. The admin repo is never modified.

The admin agent keeps its own environment through `scripts/admin_agent_sidecar.py`, invoked
with `~/projects/python_uwazi_API/.venv`. Our process never imports `uwazi_admin_agent`.

```bash
# spike
uv run benchmark-admin-agent --dataset plan_year --limit-train 5 --limit-test 5
# full phase-1 run
uv run benchmark-admin-agent --dataset plan_year
```

The admin repo's `.venv` (created 2026-08-25) predates two dependencies now declared in its
`pyproject.toml`, so importing `uwazi_admin_agent.domain.text_match` fails at
`rapidfuzz`. Rather than touch the read-only repo, the sidecar layers the exact locked
versions into `work/admin_agent_extra`; the CLI preflight prints the install command if that
directory is missing.

Artifacts use method type `admin_agent_extractor`: `method.json` records the condition,
template/property, admin model and total usage, `prompt.txt` is the exact authoring task,
`script.py` is the returned extractor, `attempts/attempt-1.json` carries the extractor's
samples stats and a clipped tool-call transcript. The run records our Plan metrics, the admin
agent's own `extraction_eval` score, and per-document predictions. Nothing is written to
MongoDB.

**Self-contained methods and replay (2026-11-23).** Every method created from now on writes
a `spec.json` next to `method.json`, `prompt.txt` and `script.py`. The spec records everything
needed to identify and reproduce the method:

- dataset, fields, real Uwazi property, task kind (properties vs records);
- model id/base URL, prompt text hash/chars, prompt version;
- condition: title visibility, metadata mode, geometry, target-property redaction;
- sampling policy and the **exact selected train shared_ids**;
- candidate-list inclusion/count, execution property names/key map/score policy;
- split policy version, adapter version, git commit, evaluation/normalization file hashes;
- max author requests.

Method ids now end with a short **spec hash** (e.g. `..._0237e882`), so a different prompt,
condition or selected sample set is a different method. If the same method id already exists,
the new run gets a `-r<timestamp>` suffix instead of overwriting the method. Every method
directory is therefore immutable; new attempts add methods/runs, never replace old logic.

Replay:

```bash
# exact: run the stored frozen script under the stored condition
uv run benchmark-admin-agent --replay <method_id>

# re-author: same stored prompt + same selected train ids, new stochastic extractor
uv run benchmark-admin-agent --replay <method_id> --replay-mode author
```

`--replay-mode author` writes a **new method** (`<old_id>-r<timestamp>`) with `replay_of`
recorded in its spec; the original is untouched. Methods created before this change have no
`spec.json` and cannot be replayed, but their prompt/script/predictions are still on disk.

**Canonical results (v2-50train, seed 42; re-created 2026-11-23).** Every admin method below
was re-created after the spec/replay change, so each has a `spec.json` with its exact selected
train ids, prompt hash and condition. The old pre-spec admin methods/runs are archived under
`artifacts/archive/admin_agent_pre_spec_20261123/`. `plan_body` is skipped: its train split has
50 distinct rare labels and zero General Assembly / Human Rights Council documents, so it does
not measure the agent.

Conditions re-created:

- **v1** — title visible, 20 train ids in the stored task prompt, target metadata visible.
- **v2** — title visible, no ids, target metadata visible.
- **v3** — no title, target removed from execution metadata, other metadata visible.
- **factor runs** — v3 with `--no-country-list`, `--geometry`, `--no-metadata`, and
  `--geometry --no-metadata`.

**v1/v2 prompt replay (stored `prompt.txt`, same 50-doc train split):**

| Dataset | Prompt | Train | Test | Test hit / wrong / missing | Requests |
|---|---|---:|---:|---|---:|
| `plan_year` | v1 | 0.900 | 0.928 | 9,244 / 560 / 161 | 16 |
| `plan_document_code` | v1 | 0.760 | 0.924 | 7,818 / 197 / 443 | 11 |
| `plan_document_type` | v1 | 0.960 | 0.918 | 2,502 / 210 / 13 | 16 |
| `plan_year` | v2 | 0.960 | 0.941 | 9,382 / 580 / 3 | 17 |

**v3 canonical (no title, target redacted, other metadata visible):**

| Dataset | Train | Test | Test hit / wrong / missing | Coverage | Requests | Tokens |
|---|---:|---:|---|---:|---:|---:|
| `plan_year` | 1.000 | **0.976**¹ | 9,725 / 229 / 11 | 99.9% | 11 | 385,290 |
| `plan_document_code` | 0.860 | **0.969** | 8,196 / 159 / 103 | 98.7% | 13 | 1,962,434 |
| `plan_document_type` | 0.980 | **0.924** | 2,517 / 208 / 0 | 100.0% | 14 | 2,025,598 |
| `plan_legal_status` | 0.980 | **0.959** | 2,613 / 65 / 47 | 98.3% | 11 | 838,255 |
| `plan_means_of_adoption` | 0.510 | **0.484** | 1,319 / 438 / 967 | 64.4% | 17 | 2,219,021 |
| `plan_field_month` | 0.940 | **0.904** | 1,369 / 136 / 10 | 99.3% | 15 | 1,196,806 |
| `plan_field_document_number` | 0.580 | **0.607** | 723 / 2 / 466 | 60.8% | 15 | 2,392,215 |
| `upr_state` | 1.000 | **1.000** | 115 / 0 / 0 | 100.0% | 5 | 348,320 |
| `upr_recommendations` | 0.984 | **F1 0.971** | 112 hit · 2 partial | 100.0% | 5 | 343,502 |

¹ **`plan_year` 0.976 is a metadata shortcut, not a document-reading result.** The extractor's
first rule reads `topics_in_related_paragraphs` labels (`"<title> <year>, para. N"`). With
`--no-metadata` the same condition scores **0.931**. This is the clearest lesson of the
re-creation: production metadata can carry the answer, so the clean text-only condition is
the right one for extraction claims.

UPR state's 1.000 agrees with our canonical-state evaluation; the admin display scorer reports
98.8% only because two gold labels carry a trailing newline.

New UPR recommendation detail (with country list): P 0.971 / R 0.971 / **F1 0.971**;
26,186 of 26,964 matched; text F1 0.994; country 0.945; exact number 0.991; state 1.000.

**Factor runs, re-created (v3):**

| Condition | Task | Test | Notes |
|---|---|---:|---|
| no country list | `upr_recommendations` | F1 **0.980** | P 0.977 / R 0.982; country 0.933; state 0.965. With the list: F1 0.971, country 0.945, state 1.000. |
| geometry | `plan_year` | 0.976 | Same metadata shortcut; geometry is not the cause. |
| geometry | `upr_recommendations` | F1 **0.936** | R 0.910, country 0.984, state 1.000. Worse than text this run. |
| no metadata | `plan_year` | **0.931** | Clean text-only. |
| geometry + no metadata | `plan_year` | **0.920** | Clean text-only; no position gain. |

**What replicated, and what did not.** The old pre-spec claims do not survive re-creation:

- **Country list:** the earlier run said with-list 0.979 vs no-list 0.939. The new runs say
  with-list 0.971 vs no-list 0.980 — the direction flipped. The country-list effect is
  **not stable across single authoring runs**; we cannot claim it helps without repeats.
- **Geometry:** with metadata it still looks like 0.976/0.943, but that is the
  `topics_in_related_paragraphs` shortcut. Clean no-metadata runs are 0.931 text vs 0.920
  geometry; UPR recommendations are 0.971 text vs 0.936 geometry. There is **no consistent
  position/size gain** for the admin extractor.
- **Title:** v2 (title, no ids) scores 0.941 and v3-with-metadata 0.976 (shortcut); the clean
  no-title run is 0.931. With one run per condition, the title effect is **not separable from
  run variance and metadata leakage**.
- **Sampling variance is large:** `plan_means_of_adoption` fell from 0.709 to 0.484 with the
  same settings; the new extractor sampled 14/30 train docs and abstained on 967 test docs.
  This is exactly the problem the sampling-policy experiment should address.

**Status:** the canonical set is re-created and self-contained; the old admin results are
archived. The authoring-sample A/B is done (below): restricting the universe to 12 coverage
documents helps UPR recommendations slightly and costs less authoring, but hurts Plan
classification because the extractor calibrates on the biased sample. The opt-in repair loop
is implemented and had a neutral first pass: 10-document validation is too small (it saturates
or swings), most shown failures were singleton/gold-ambiguous, and holding out 10 of 50 docs
removes scarce learning examples. Next: systematic-only failure families, a larger/cross-val
validation signal, and production holdouts that do not shrink the authoring pool. The full
`GenerateScriptUseCase` bulk-script path remains out of scope.

### Authoring-sample policies (2026-11-23)

**Problem.** With policy `agent` the authoring request contains all 50 train documents and the
extractor picks its own 8–12 with no coverage guarantee, so the same prompt can sample a
different evidence set on every run (see the `plan_means_of_adoption` 0.709→0.484 swing
above). `sampling.py` now offers deterministic restrictions of the authoring universe, train
only, with the prompt's source/`ctx` contract unchanged:

| Policy | Authoring universe |
|---|---|
| `agent` (default) | all 50 train docs; the extractor still discovers and peeks 8–12 itself |
| `coverage` | ≤12: rare-first greedy cover of document_type/body/legal_status (cycle for UPR), page/character/paragraph buckets, header patterns A/RES, A/HRC, Distr., Report of, No. NN, and hard flags |
| `similarity` | ≤12: TF-IDF over first-page words + structural tokens, farthest-first (k-center) |
| `hybrid` | ≤12: 6 coverage representatives, then 6 farthest-first seeded from them; force-includes one target-absent negative and one hard structural case when the universe has one |

Hard cases are shared PDFs (same file/name/content, or the same page+character count for
large documents), multi-entry compilations, OCR/mojibake, and UPR sub-numbered /
multi-country recommendations. Gold is read only to recognise target-absent negatives and
those UPR record hard cases; no gold value is written into the prompt.

The selected ids, a per-document signature, stratum coverage and hard cases go into
`spec.json` under `sampling.*`. The method id hashes the spec, so every selection is an
immutable method. Restricted policies add a short "sample broadly, not just the first two"
note to the task and a `-sample-<policy>` prompt-version suffix. Replay rebuilds the stored
ids (`sampling.restrict_documents`) and never re-samples.

**A/B design (no title + `--no-metadata`, country list on for UPR).** Tasks
`plan_document_type`, `upr_recommendations`, `plan_year`; `agent` vs `hybrid`, 3 authoring
runs per arm. `scripts/admin_sampling_ab.py` reads the specs, runs and attempt transcripts and
reports test score mean/min/max, wrong+missing counts, author requests/tokens, selected-sample
strata and pairwise diversity, and the documents the extractor actually peeked at.

**Results and caveats.** 18 runs (3 per arm per dataset), frozen condition no title +
`--no-metadata`, country list on for UPR. Full tables in `reports/admin_sampling_ab.md`
(regenerate with `scripts/admin_sampling_ab.py`). Means with [min–max] over the three runs:

| Dataset | Metric | `agent` (50) | `hybrid` (12) | Δ mean |
|---|---|---:|---:|---:|
| `plan_document_type` | accuracy | **0.919** [0.889–0.953] | 0.692 [0.546–0.813] | −0.227 |
| `plan_year` | accuracy | **0.925** [0.920–0.927] | 0.860 [0.798–0.931] | −0.064 |
| `upr_recommendations` | F1 | 0.973 [0.962–0.982] | **0.977** [0.976–0.980] | +0.005 |

Wrong+missing per test split, author requests and tokens (means): `plan_document_type`
221.7 / 21.0 / 4.41M vs 840.3 / 15.3 / 1.70M; `plan_year` 751.7 / 14.7 / 1.37M vs 1394.0 /
13.0 / 2.26M; `upr_recommendations` 0 / 12.3 / 1.78M vs 0 / 9.7 / 1.33M. Hybrid halves
authoring tokens on document type and cuts requests on UPR, but it is not a uniform cost win
(plan year tokens are higher).

**The restriction hurts property classification, and the mechanism is visible.** The 12-doc
universe over-represents rare classes and layouts, and the admin extractor calibrates with
`benchmark_extractor` over that same universe, so it fits the biased sample:

- `plan_document_type`: hybrid selected 4 Resolutions of 12 (universe: 28/50). Two extractors
  over-predicted rare classes badly — one predicted `Declaration / Confererence outcome
  document` 484 times for 11 true test documents; another predicted `Special Procedures'
  report` 860 times for 552. Train stayed at 0.917–1.000 while test fell to 0.546–0.813.
- `plan_year`: hybrid selected no 1940s/1950s/1970s and only one 1990s document. 1990s test
  accuracy was 0.486 (38% missing) vs 0.914 (2% missing) for `agent`; 2000s/2010s/2020s were
  nearly identical. The missing decades are exactly the target classes the sampler does not
  cover.
- `upr_recommendations`: hybrid selected no `recs300+` document, yet F1 is flat/slightly
  better and authoring is cheaper. The list extractor is less sensitive to the document
  prior.
- Peeking: on the Plan tasks the extractor sampled 9–17 documents under `agent` and 4–12
  under `hybrid`; UPR peeks were 4 vs 3–5. All peeks stayed inside the selected universe. The
  hybrid order is rare-first, so a run that peeks only 4/12 can miss common Resolutions
  entirely; one such run is the 0.546 outlier.

**Caveat on the negative slot.** The v2 train pools contain only documents with non-empty
gold, so no target-absent negative exists (the spec records `negatives: []` and a note). The
forced hard case is a shared PDF / 209-page compilation for the Plan datasets and a
sub-numbered recommendation document for UPR.

**Takeaway.** The static coverage sampler as specified is not an accuracy win for property
extraction; it trades the true class/label prior for layout/class breadth. The next iteration
should add target strata (year decade, recommendation-count bucket) and/or preserve the prior
(e.g. 6 rare + 6 proportional documents) and test ordering (put a common representative first,
then hard cases). The active second pass remains appropriate for properties and UPR state,
not for recommendation lists.

### Repair loop (2026-11-23, opt-in)

`--repair` adds a feedback loop on top of the normal one-shot authoring:

1. split the train split into a **40-document feedback pool** and a **10-document frozen
   validation slice** (proportional by the dataset label, deterministic hash, capped at 20%);
2. author on the feedback pool only, execute on 40+10 with our evaluator;
3. feed up to 3 diverse failures from the **feedback pool only** plus the current function
   back to the nested extractor;
4. repeat up to 3 rounds, pick the best round on validation (ties broken by the feedback
   score), then run test once.

Validation docs never appear in an authoring/repair request. `spec.json` records the split,
feedback/validation ids and per-round scores separately from the spec hash (so the selection
is reproducible and the outcome does not change the method id). `--replay` execute works on
the final frozen script; `--replay-mode author` refuses repair methods.

**First-pass results** (one or two runs per cell; controls are `--repair-rounds 0`, i.e. the
same 40/10 holdout with no repair; baseline is the 3-run `agent` A/B above):

| Dataset | 50-doc baseline | 40/10 control | 40/10 repair |
|---|---:|---:|---:|
| `plan_year` | 0.925 [0.920–0.927] | **0.932** | 0.932 · 0.935 |
| `plan_document_type` | **0.919** [0.889–0.953] | 0.834 | 0.858 |

Round transcripts show why the loop did not move the score:

- `plan_year`: validation was saturated at **1.0** from round 0, so no later round could win
  on validation. Feedback stayed at 0.925 across rounds even though the shown failures
  changed (`1977`, `1980`, `2015`), i.e. revisions swapped errors rather than fixing them.
  The tie-break on feedback score had nothing to select. The repair run matched the control
  exactly (0.932).
- `plan_document_type`: feedback improved 0.975→1.000 but validation fell 0.80→0.70, so the
  loop correctly kept round 0. The recurring failure was a single `missing:Resolution` family.
  Test was 0.834 (control) / 0.858 (repair) vs 0.919 with all 50 docs: holding out 10 removes
  rare authoring examples (the validation slice contained the only ExCom/General Comment
  examples in the 40-doc pool) and that costs more than repair recovers.
- Leak check: no validation shared_id appeared in any round prompt (checked by
  `scripts/admin_repair_report.py` over stored attempts).

**Findings.** The orchestration is sound and cheap; the limiting factor is signal quality.
A 10-document validation slice is too small for model selection (saturates or swings), and
most failures shown were singleton, possibly gold-ambiguous cases, so the agent had nothing
systematic to generalise. At 50 train documents, holding out 10 also removes scarce learning
examples.

**Next:** only feed failure families that occur in at least two documents and are grounded in
the text; use a larger validation slice or cross-validation; in production, keep authoring on
the full pool and hold out a separate validation/evaluation stream rather than subtracting
20% from a small training set.

### What the v2 numbers show

- **The excerpt fix mattered a lot.** An earlier v2 pass centered prompt/repair excerpts on
  an early occurrence of "Recommendations" (or fell back to the document head) because UPR
  text often has doubled spaces. After matching whitespace-flexibly and centering on the
  actual recommendation section, the same methods improved dramatically without changing the
  split or the model: plan year text 0.660→0.720 train / 0.769→0.826 test, document code
  0.380→0.760 train / 0.915→0.972 test, UPR recommendations text 0.320→0.916 train /
  0.436→0.880 test, UPR recommendations segments 0.318→0.961 train / 0.434→0.945 test.
  Context location, not just representation, is a first-order factor.
- **Plan year benefits from segments/position**: 0.900 train / 0.927 test vs 0.720 / 0.826
  for text. It nearly matches the regex baseline (0.932 test).
- **Plan document code is now a strong result**: the LLM script reaches 0.972 test and 0.760
  train, above the regex baseline on both (0.949 / 0.720). The negative train→test gap is
  expected because the train set is deliberately stratified by code family and contains
  rare/hard codes; it is not an overfitting signal.
- **UPR recommendation extraction is now close to or above the baseline**:
  - text: F1 0.880 vs baseline 0.883 (same recall regime, better precision).
  - segments + position/type: F1 0.945 and recall 0.938 vs baseline 0.883 / 0.804. Layout
    helps the recommendation paragraphs, but the same script returns no state under review
    on any test document (state accuracy 0.000 vs 0.895 for text). The segment script builds
    text from segments but its state regex does not handle the page-1 header; this is a
    concrete, reproducible failure mode and a clear next fix.
- **UPR state is solved by the model and the baseline**: 1.000 train, 0.991 test (one wrong
  document). Segmentation layout adds nothing here.
- **Repair is now useful but still uneven**: plan year segments improved from 0.700 to 0.900
  over four rounds; document code from 0.560 to 0.760; UPR recommendations text from 0.816 to
  0.916; UPR recommendations segments from 0.887 to 0.961. Two repair rounds in the UPR text
  run produced invalid Python (syntax errors); best-script selection kept the valid round.
- **Train is intentionally hard**: stratified rare classes and UPR sub-number/count strata
  make the 50-document train score a diagnostic, not a clean estimate of test performance.
  Always report the 50-document, 48 non-example, and test scores.

### Model note / cloud fix

The native Ollama `/api/chat` endpoint exposes a `think` control. Setting
`"think": false` makes `deepseek-v4.1-flash:cloud` return final content immediately.
The OpenAI-compatible `/v1/chat/completions` endpoint ignored `think` and the reasoning
models spent the whole completion budget on hidden reasoning with empty `content`.

`llm.py` uses the native endpoint and records `message.thinking`, `done_reason`, prompt/eval
token counts, and every attempt. The configured `MODEL` remains
`deepseek-v4.1-flash:cloud`; `generate-script` exposes `--think off|low|high|max`, defaulting
to `off`.

### Known issues / next

1. Fix the UPR recommendation segment script's state extraction: it returns no state on all
   test documents although recommendations are excellent. Add an explicit state rule for
   segment input (first-page header lines, `Report of the Working Group ... Review ...`).
2. Harden repair: two UPR text repair rounds produced invalid Python. Validate the response
   and retry the same round once before counting it as a failed repair.
3. Run multiple generation attempts per method to separate prompt/representation effects
   from code-generation variance.
4. Add a plain-text/markdown run summary alongside the Streamlit viewer.
5. Review `upr_recommendations` gold noise and sub-number matching before strong conclusions.

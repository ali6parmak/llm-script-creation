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

**Results (v2-50train, seed 42, 2026-11-20):**

| Dataset | Train (50) | Test | Gap | Coverage (test) | Admin eval | Test statuses | Author requests | Tokens |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `plan_year` | 0.960 | **0.934** (9,304 / 9,965) | +0.026 | 99.7% | 93.4% (10,015 judged) | 9,304 hit · 628 wrong · 33 missing | 14 | 1,089,919 |
| `plan_document_code` | 0.800 | **0.973** (8,232 / 8,458) | −0.173 | 99.4% | 96.3% (8,508 judged) | 8,232 hit · 178 wrong · 48 missing | 13 | 1,186,141 |
| `plan_document_type` | 0.940 | **0.894** (2,437 / 2,725) | +0.046 | 91.4% | 89.5% (2,775 judged) | 2,437 hit · 53 wrong · 235 missing | 18 | 2,954,941 |

The three rows above used prompt v1, which listed up to 20 train shared_ids in the task
(the variant discussed below). The 5/5 spike scored 5/5 train and 5/5 test (admin eval 100%).
On the full run the admin
agent sampled 12 train documents with `peek_segmentation_by_entity` and used
`benchmark_extractor` three times. The returned function uses ordered fallbacks: a
compilation-entry lookup keyed on the entity's own `field_document_number` metadata (needed
because one PDF can hold many conclusions), then the official "Distr.: General <date>"
header year, then a heading-style `(YYYY)`, then the first full date in the text. That
metadata-aware compilation rule is a concrete example of the production-context condition
doing work: with title/metadata hidden it could not disambiguate a shared PDF.

The dominant error class is the year disagreement between a document's adoption date and its
distribution/header date (e.g. shared_id `5zhgytix6is`: gold 2015, header 2016). The
extractor's unmatched notes document this as a curation inconsistency. 33 test documents
return `None` (no full date found) and are counted as missing, not wrong.

On `plan_document_code` the extractor reads the repeated running header/cover symbol (with a
fallback for Commission on Human Rights resolution pages) and reaches 0.973 test over 8,458
documents; the negative train→test gap is the same stratified-train effect seen for the LLM
scripts (rare code families are concentrated in the 50 train documents).

On `plan_document_type` the extractor keys off UN header/signature phrases (resolution
headers, ExCom conclusion titles, "agreed as follows" plus a regional-body check, CSW agreed
conclusions, treaty-body general comments, SRSG/mandate reports, declarations). It samples
29/30 train documents correctly, but test coverage drops to 91.4%: 235 documents return
`None`, mostly Resolutions (182) whose first pages lack the header phrases the extractor
requires. The most common wrong labels are Resolution→Special Procedures' report (16) and
Resolution→CSW Agreed Conclusions (13). This is an authoring-sample coverage limit, not a
test-set tuning issue: the 50/30 train sample simply does not exhibit every resolution
header variant. It is reported as-is because phase 1 fixes the admin agent's prompt and
sampling behavior.

**Prompt variant v2: no injected shared_ids.** Prompt v1 listed up to 20 train shared_ids in
the task. That was an adapter convenience, not part of the admin agent's own workflow, and it
risked anchoring which documents the extractor sampled. `prompt.PROMPT_VERSION =
"admin-agent-v2-no-ids"` removes the list and the `by_ids` hint; the task now states only the
train count and tells the extractor to discover samples with
`query_entities(mode="by_template")`. Re-running `plan_year` gives:

| `plan_year` prompt | Train | Test | Test hit / wrong / missing | Author requests | Tokens |
|---|---:|---:|---|---:|---:|
| v1 (ids injected) | 0.960 | 0.934 | 9,304 / 628 / 33 | 14 | 1,089,919 |
| v2 (no ids) | 0.900 | 0.939 | 9,360 / 517 / 88 | 18 | 1,907,173 |

The +0.005 test difference is **not** a clean id-list effect: the v2 extractor chose a
different first rule — the year in the entity TITLE, e.g. `... (2011)` — before any
segmentation rule. 7,258 of the 9,965 test documents carry a `(YYYY)` in the title; v2
predicts that title year for all 7,258 and gets 7,233 right (99.7%). On the remaining 2,707
documents it scores 0.786. The v1 extractor did not use the title. So v2's test number is
mostly a **title shortcut** under the production-context condition, not a better document
reader; its lower train score and extra 55 missing predictions also show it is not simply
"better". Removing the ids was still the right call, but it cannot be evaluated in isolation
while the title is visible. The **no-title condition is now the necessary next experiment**,
not an optional one.

**No-title condition (v3).** `prompt.PROMPT_VERSION = "admin-agent-v3-no-title"` hides the
title from `AgentEntity`, raw entities, query summaries and `ctx`, and removes the target
property from `ctx.metadata` in the execution payload. Re-running `plan_year`:

| `plan_year` prompt | Train | Test | Test hit / wrong / missing | Coverage | Author requests | Tokens |
|---|---:|---:|---|---:|---:|---:|
| v1 (title visible, ids) | 0.960 | 0.934 | 9,304 / 628 / 33 | 99.7% | 14 | 1,089,919 |
| v2 (title visible, no ids) | 0.900 | 0.939 | 9,360 / 517 / 88 | 99.1% | 18 | 1,907,173 |
| v3 (no title) | 0.920 | 0.932 | 9,288 / 675 / 2 | 99.98% | 12 | 734,535 |

The v3 extractor is a clean, simple rule: **first plausible four-digit year in the document
text**, with a `dateparse` fallback on the first 2,000 characters. It never reads `ctx`.
That is effectively the regex baseline (0.932 test), so under the no-title condition the
admin agent's strategy is **baseline-level on `plan_year`**, not the metadata-aware
compilation strategy of v1. The title was doing the work in v2; without it, the agent
converges to the obvious heuristic. Coverage is 99.98% (only 2 missing), so the remaining
error is wrong-year choice, not abstention.

**Full Plan sweep under v3 (no title, target removed from execution metadata):**

| Dataset | Train | Test | Test hit / wrong / missing | Coverage | Author requests | Tokens |
|---|---:|---:|---|---:|---:|---:|
| `plan_year` | 0.920 | 0.932 | 9,288 / 675 / 2 | 99.98% | 12 | 734,535 |
| `plan_document_code` | 0.760 | 0.974 | 8,241 / 134 / 83 | 99.0% | 13 | 972,044 |
| `plan_document_type` | 0.940 | 0.885 | 2,412 / 263 / 50 | 98.2% | 21 | 2,889,624 |
| `plan_legal_status` | 1.000 | 0.968 | 2,639 / 17 / 69 | 97.5% | 13 | 1,611,385 |
| `plan_means_of_adoption` | 0.857 | 0.709 | 1,932 / 792 / 0 | 100.0% | 14 | 1,808,253 |
| `plan_body` | 0.760 | **0.055** | 550 / 1,525 / 8,012 | 20.6% | 16 | 2,018,858 |
| `plan_field_month` | 0.940 | 0.915 | 1,386 / 124 / 5 | 99.7% | 16 | 1,304,856 |
| `plan_field_document_number` | 0.540 | 0.606 | 722 / 105 / 364 | 69.1% | 21 | 3,976,425 |

Three results need context:

- **`plan_body` is not a valid capability result under the current split.** Train has 50
  documents and **50 distinct body labels** (one each, rare-first stratification) and
  **zero** General Assembly / Human Rights Council documents. Test is 89.7% those two
  classes: 6,359 GA + 2,691 HRC of 10,087. The extractor authored rules for the rare labels
  it could see (and got 0.760 on train), then returned `None` for the two dominant test
  classes. That is a split-design failure, not an agent failure; exclude `plan_body` from
  admin-agent comparisons until a body-specific split covers the dominant classes.
- **`plan_means_of_adoption` (0.709)** produced a value for every test document; the errors
  are label ambiguity, mainly N.A.→Consensus (378), Vote→N.A. (195), and Consensus→N.A.
  (104). This is a genuine three-way classification difficulty, not abstention.
- **`plan_field_document_number` (0.606, 69.1% coverage)** is the weakest legitimate field:
  identifiers are inconsistent across documents and the extractor abstained on 364 test
  documents. `plan_document_code` (0.974) and `plan_legal_status` (0.968) are strong;
  `plan_field_month` (0.915) is close behind.

**Two scoring views.** Our evaluator normalizes document codes (case/punctuation/spacing), so
a prediction `A/RES/64/74` matches gold `A/RES/64/74*`. The admin agent's own
`extraction_eval` compares rendered display strings, so it counts those 82 starred gold codes
as mismatches: 8,190 matched / 263 mismatched / 55 missing (96.26%) versus our 8,272 hit /
181 wrong / 55 missing. Neither is wrong for its purpose; our normalization is the benchmark
rule used to compare methods, and the admin view is recorded as-is. The two views agree on
every other document for this field.

**UPR state (no title, v3).** `state_reviewed` is a relationship property, so the task uses
the real property name and the sidecar maps the prediction back to our `state_under_review`
gold field. The task lists all 195 Uwazi State labels as candidates. Result: **115/115 test
(1.000)**, 5 author requests, 331,885 tokens. The extractor embeds the candidate list,
reads the state from page 1, and maps document spellings to the Uwazi label. The admin
agent's own display scorer reports 98.8% because two gold labels carry a trailing newline
(`"Trinidad and Tobago\n"`, `"Bosnia and Herzegovina\n"`); our canonical-state
normalization strips it. This is gold noise, not an extraction miss.

**UPR recommendations (record extraction, no title, v3).** This is the biggest TASKS item.
The task asks for `{"state_under_review": label, "recommendations": [{number, text,
countries}]}`, lists all 195 State labels, and tells the extractor that
`benchmark_extractor` cannot score a recommendation list (the document metadata only links
recommendation ids, not their text), so it must self-prove on sampled documents. The
extractor was authored from only **4 sampled train documents**; it parses section II of the
Working Group report, skips page furniture, merges cross-page continuations, and splits
multi-country recommendations into sub-numbered records.

| UPR recommendations | Test |
|---|---:|
| Documents | 114 |
| Gold / predicted / matched | 26,964 / 27,014 / 26,431 |
| Precision / recall / F1 | 0.978 / 0.980 / **0.979** |
| Text F1 | 0.9995 |
| Country accuracy | 0.982 |
| Exact-number accuracy | 0.991 |
| State accuracy | 0.991 |
| Statuses | 111 hit · 3 partial · 0 wrong · 0 missing |

Train F1 is 0.995; the run used 6 author requests and 545,166 tokens. This **beats our own
script pipeline** (`0.945` test with segments+position) and the deterministic baseline
(`0.883`) without using positions at all. The `admin_eval` stored for this run is
**state-only** (98.2%); it is not a recommendation score and should not be compared with the
F1 above.

**Country-list A/B (the TASKS "pass the list of countries" question).** Both runs use the
same dataset, model, no-title condition and train/test split; the only change is whether the
task includes the 195 Uwazi State labels (`--no-country-list` removes them). The no-list
extractor produced a small hand-written printed→Uwazi map (Congo, Syrian Arab Republic,
Micronesia, …) instead of the full list.

| UPR recommendations | P | R | F1 | Text F1 | Country | Number | State | Statuses | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| with 195 labels | 0.978 | 0.980 | **0.979** | 0.9995 | 0.982 | 0.991 | 0.991 | 111 hit · 3 partial | 6 | 545,166 |
| no list | 0.975 | 0.906 | 0.939 | 0.9997 | 0.956 | 0.986 | 0.904 | 89 hit · 25 partial | 14 | 1,699,722 |

The list is worth **+0.040 F1**, mostly recall (+0.074) and state accuracy (+0.087); country
accuracy improves by 2.6 points. It is also cheaper (6 vs 14 author requests), because the
agent does not have to rediscover the label universe. This is a clear win for passing the
Uwazi country list, and the main TASKS experimental factor we can answer with confidence.

Positions/sizes were not provided to either extractor; that TASKS factor is still untested
here (our own pipeline's segments+position variant remains the evidence for it).

**Status:** the no-title v3 adapter covers all eight single Plan fields with gold, UPR state,
and UPR recommendation records. The country-list A/B is done (the list clearly helps).
Remaining TASKS factors: position/size input and prompt variants. The full
`GenerateScriptUseCase` bulk-script path remains out of scope.

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

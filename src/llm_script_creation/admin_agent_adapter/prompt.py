"""Task prompt for the admin agent's nested extractor subagent.

This is the benchmark's task description, not a copy of the admin agent's system
prompt: the extractor's own contract (``def extract(text, ctx)``, no imports, no
classes, ordered fallbacks) comes from
``uwazi_admin_agent.use_cases.author_extractor_tool``. We only say WHAT to
extract and WHERE the text comes from.

The task deliberately does NOT inject a list of shared_ids. The admin agent's
extractor discovers its own samples through ``query_entities``; pre-selecting
documents in the task would anchor sampling and deviate from the real workflow.

Main-process safe: stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Version of the authoring task contract. Bumped when the task text changes so
#: method artifacts from different prompt variants stay distinguishable.
PROMPT_VERSION = "admin-agent-v3-no-title"
PROMPT_VERSION_GEOMETRY = "admin-agent-v3-no-title-geometry"

_VALUE_SHAPES = {
    "numeric": "an int, e.g. 2019 (not a string)",
    "text": "a string, e.g. \"A/HRC/RES/14/1\"",
    "select": "one of the allowed labels below, verbatim",
    "date": "an ISO \"YYYY-MM-DD\" string",
}


def _bullet_list(values: Iterable[str], limit: int = 20) -> str:
    items = list(values)
    if not items:
        return "  (none)"
    shown = items[:limit]
    lines = [f"  - {value}" for value in shown]
    if len(items) > limit:
        lines.append(f"  - … and {len(items) - limit:,} more")
    return "\n".join(lines)


def build_author_task(
    *,
    dataset_id: str,
    property_name: str,
    property_type: str,
    template_name: str,
    train_total: int,
    allowed_values: Iterable[str] | None = None,
    language: str = "en",
    source_label: str = "primary-document segmentation",
    title_visible: bool = False,
    target_metadata_hidden: bool = True,
    value_shape: str | None = None,
    task_hint: str | None = None,
    candidate_block_title: str = "ALLOWED VALUES (the property is a select thesaurus; use the label verbatim)",
    geometry: bool = False,
    metadata_visible: bool = True,
) -> str:
    """Build the precise authoring task for one property.

    Only a COUNT of train documents is disclosed; the extractor discovers the
    documents itself. No test document is ever in the authoring request.

    ``title_visible=False`` is the no-title condition: ``ctx["title"]`` is empty
    and entity summaries carry no titles. ``target_metadata_hidden=True`` tells
    the extractor that the target property is removed from ``ctx.metadata`` at
    execution time (train rows may still carry it because they are labeled).
    """
    value_shape = value_shape or _VALUE_SHAPES.get(property_type, "a JSON-serializable value")
    hint_block = f"\n- Task hint: {task_hint}" if task_hint else ""
    if geometry:
        format_block = (
            "- This is the GEOMETRY condition. Every line is `<prefix>text`, where the prefix is\n"
            "  `[pPAGE x=LEFT y=TOP w=WIDTH h=HEIGHT]` with integer PDF coordinates and the original\n"
            "  paragraph text follows after one space. The prefix is metadata: strip it before\n"
            "  returning any text value. You may use page/x/y/size to identify headers, footers,\n"
            "  columns, and reading order."
        )
    else:
        format_block = (
            "- There are no page markers, coordinate prefixes, or separators: the line is just the\n"
            "  paragraph text."
        )
    ctx_title = "str" if title_visible else '"" (hidden)'
    if title_visible:
        title_block = (
            "- Title is VISIBLE: `ctx['title']` carries the entity title. Metadata is\n"
            "  visible too, so a title/metadata shortcut is possible; prefer reading the\n"
            "  document when the value is stated there."
        )
    else:
        title_block = (
            "- Title is HIDDEN in this condition: `ctx['title']` is always the empty\n"
            "  string, and `query_entities` summaries/examples carry no titles."
        )
    if not metadata_visible:
        target_block = (
            "- At execution, `ctx['metadata']` is EMPTY. Do not read any metadata\n"
            "  (the target, the title, or any other property); use the document text."
        )
    elif target_metadata_hidden:
        target_block = (
            f"- The target property `{property_name}` is REMOVED from `ctx['metadata']`\n"
            "  when the extractor runs (production-realistic: the value being extracted is\n"
            "  empty). Train rows may still carry it because they are already labeled, but\n"
            "  a rule that reads the target from `ctx['metadata']` will return nothing on\n"
            "  the real run. Read the document text instead."
        )
    else:
        target_block = (
            f"- Metadata is visible, including the target property `{property_name}` on\n"
            "  labeled train rows. A metadata shortcut is possible; prefer document text."
        )
    allowed_block = ""
    if allowed_values:
        allowed_block = f"\n{candidate_block_title}:\n{_bullet_list(allowed_values, limit=500)}\n"

    return f"""\
TASK: author ONE deterministic extraction function for a Uwazi template property.

DATASET / TARGET
- Dataset id (for the record): {dataset_id}
- Template: "{template_name}"
- Target property: `{property_name}` (type: {property_type})
- Value shape: {value_shape}{hint_block}

OUTPUT CONTRACT (unchanged from your system prompt)
- Emit exactly one pure Python function:
      def extract(text: str, ctx: dict) -> dict | None
- Return `{{"{property_name}": <value>}}` when the document states the value.
- Return `None` when the value is absent or cannot be determined with confidence.
- Return ONLY the `{property_name}` key: no scratch keys, no wrapper object, no
  extra properties. Never guess; a missing value leaves the entity untouched.
{allowed_block}
SOURCE: {source_label}
- These entities' primary document is a PDF. The `text` argument is EXACTLY
  `"\\n".join(p["text"] for p in seg["paragraphs"])`: page-ordered paragraph
  text, one paragraph per line.
{format_block}
- `ctx` is EXACTLY:
      {{"shared_id": str, "title": {ctx_title}, "metadata": {{property: [values]}}}}
  `metadata` is Uwazi's read shape: each property is a list of entries such as
  {{"value": ..., "label": ...}}.
{title_block}
{target_block}
- Do NOT look for HTML supporting files or call `peek_file_text`: these
  entities have no HTML file. Sample their primary-document text with
  `peek_segmentation_by_entity(shared_id, "{language}")`.
- You may use the bound helpers `dateparse` and `textmatch` (documented in your
  system prompt) instead of hand-rolled date/normalization logic.

TRAINING EVIDENCE (the ONLY documents you may look at)
- {train_total} train document(s) are available through the read-only ports.
  Discover them with `query_entities(mode="by_template",
  template_name="{template_name}", language="{language}")`, then sample 8-12 of
  them with `peek_segmentation_by_entity`.
- After drafting `extract`, SELF-PROVE it on the sampled literal
  (text, ctx) pairs before emitting, as your system prompt requires. Use
  `benchmark_extractor` when comparing competing strategies; it scores against
  the values these train entities already carry.

Return the function plus honest samples_total / samples_matched counts.
"""


PROMPT_VERSION_RECOMMENDATIONS = "admin-agent-v3-records-no-title"
PROMPT_VERSION_RECOMMENDATIONS_NO_LIST = "admin-agent-v3-records-no-title-no-list"
PROMPT_VERSION_RECOMMENDATIONS_GEOMETRY = "admin-agent-v3-records-no-title-geometry"


def build_recommendations_task(
    *,
    dataset_id: str,
    template_name: str,
    train_total: int,
    state_labels: Iterable[str],
    language: str = "en",
    title_visible: bool = False,
    target_metadata_hidden: bool = True,
    include_state_list: bool = True,
    geometry: bool = False,
    metadata_visible: bool = True,
) -> str:
    """Authoring task for UPR recommendation record extraction.

    This is deliberately a different task shape from a single-property fill: one
    document yields MANY recommendation records. The function returns a dict with
    the state under review and a list of ``{number, text, countries}`` records;
    our ``evaluate_upr_sample`` scores it. The real admin agent's bulk path would
    then create/link entities — out of scope here.
    """
    title_block = (
        "- Title is HIDDEN in this condition: `ctx['title']` is always the empty string.\n"
        "  `query_entities` summaries/examples carry no titles."
        if not title_visible
        else "- Title is VISIBLE: `ctx['title']` carries the entity title."
    )
    if not metadata_visible:
        target_block = (
            "- At execution, `ctx['metadata']` is EMPTY. Do not read any metadata; use the\n"
            "  document text."
        )
    elif target_metadata_hidden:
        target_block = (
            "- The properties `state_reviewed` and `related_recommendations` are REMOVED from\n"
            "  `ctx['metadata']` when the extractor runs. Train rows may still carry them, but\n"
            "  do not read the answer from metadata; read the document text."
        )
    else:
        target_block = "- Metadata, including the linked recommendation ids, is visible."
    if geometry:
        format_block = (
            "- This is the GEOMETRY condition. Every line is `<prefix>text`, where the prefix is\n"
            "  `[pPAGE x=LEFT y=TOP w=WIDTH h=HEIGHT]` with integer PDF coordinates and the original\n"
            "  paragraph text follows after one space. Strip the prefix before returning any\n"
            "  `text` field; use page/x/y to skip page furniture and merge cross-page continuations."
        )
    else:
        format_block = (
            "- There are no page markers, coordinate prefixes, or separators: the line is just the\n"
            "  paragraph text."
        )
    if include_state_list:
        countries_instruction = (
            "- `countries`: recommending-state labels verbatim from the candidate list. Use []\n"
            "  when the text names none."
        )
        candidate_block = (
            "CANDIDATE STATE LABELS (use these verbatim for `state_under_review` and `countries`;\n"
            "document spellings may differ — use `textmatch` to map a spelling to a label):\n"
            f"{_bullet_list(state_labels, limit=500)}"
        )
    else:
        countries_instruction = (
            "- `countries`: recommending-state names as printed in the document, normalized to the\n"
            "  Uwazi State spelling from your own knowledge. No candidate list is provided. Use []\n"
            "  when the text names none."
        )
        candidate_block = (
            "NO CANDIDATE STATE LIST IS PROVIDED in this condition. Infer `state_under_review`\n"
            "and `countries` from the document text and your own knowledge of Uwazi State labels;\n"
            "do not invent a country that is not named in the text."
        )
    return f"""\
TASK: author ONE deterministic extraction function that extracts UPR recommendation RECORDS.

DATASET / TARGET
- Dataset id (for the record): {dataset_id}
- Template: "{template_name}"
- One document contains MANY recommendation records. This is NOT a single-property fill.

OUTPUT CONTRACT (still the extractor signature from your system prompt)
- Emit exactly one pure Python function:
      def extract(text: str, ctx: dict) -> dict | None
- Return:
      {{"state_under_review": "<state label>",
        "recommendations": [
          {{"number": "<number as printed>",
            "text": "<recommendation text>",
            "countries": ["<recommending state label>", ...]}},
          ...
        ]}}
- `number`: the recommendation/paragraph number that precedes the text (e.g. "95.24",
  "24.112.1"). One paragraph can contain several recommendations; numbers can have
  sub-numbers. Do not use a page number.
- `text`: the recommendation statement only. Exclude the trailing country
  parenthetical/semicolon; the recommending states belong in `countries`.
{countries_instruction}
- Return `None` only when the document contains no recommendations at all.
{title_block}
{target_block}

SOURCE: primary-document segmentation
- These entities' primary document is a PDF. The `text` argument is EXACTLY
  `"\\n".join(p["text"] for p in seg["paragraphs"])`: page-ordered paragraph text,
  one paragraph per line.
{format_block}
- `ctx` is EXACTLY:
      {{"shared_id": str, "title": "", "metadata": {{property: [values]}}}}
- Do NOT look for HTML supporting files or call `peek_file_text`; sample the
  primary-document text with `peek_segmentation_by_entity(shared_id, "{language}")`.
- The recommendation section is usually introduced by "Recommendations", "The
  recommendations will be examined", or a numbered list; the state under review is on
  page 1.

{candidate_block}

TRAINING EVIDENCE (the ONLY documents you may look at)
- {train_total} train document(s) are available through the read-only ports. Discover
  them with `query_entities(mode="by_template", template_name="{template_name}",
  language="{language}")`, then sample 8-12 with `peek_segmentation_by_entity`.
- `benchmark_extractor` CANNOT score a recommendation list here: the document metadata
  only stores linked recommendation ids, not their text. Use it only for the state
  property if you wish; self-prove the list by reading the sampled documents.
- After drafting, self-prove `extract` on the sampled literal (text, ctx) pairs before
  emitting.

Return the function plus honest samples_total / samples_matched counts.
"""

"""Task prompt for the admin agent's nested extractor subagent.

This is the benchmark's task description, not a copy of the admin agent's system
prompt: the extractor's own contract (``def extract(text, ctx)``, no imports, no
classes, ordered fallbacks) comes from
``uwazi_admin_agent.use_cases.author_extractor_tool``. We only say WHAT to
extract and WHERE the text comes from.

Main-process safe: stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable

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
    train_shared_ids: Iterable[str],
    train_total: int,
    allowed_values: Iterable[str] | None = None,
    language: str = "en",
    source_label: str = "primary-document segmentation",
) -> str:
    """Build the precise authoring task for one property.

    ``train_shared_ids`` must come from the train split only: test documents are
    never placed in the authoring request.
    """
    ids = list(train_shared_ids)
    value_shape = _VALUE_SHAPES.get(property_type, "a JSON-serializable value")
    allowed_block = ""
    if allowed_values:
        allowed_block = (
            f"\nALLOWED VALUES (the property is a select thesaurus; use the label verbatim):\n"
            f"{_bullet_list(allowed_values)}\n"
        )

    return f"""\
TASK: author ONE deterministic extraction function for a Uwazi template property.

DATASET / TARGET
- Dataset id (for the record): {dataset_id}
- Template: "{template_name}"
- Target property: `{property_name}` (type: {property_type})
- Value shape: {value_shape}

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
  text, one paragraph per line. There are no page markers or separators.
- `ctx` is EXACTLY:
      {{"shared_id": str, "title": str, "metadata": {{property: [values]}}}}
  `metadata` is Uwazi's read shape: each property is a list of entries such as
  {{"value": ..., "label": ...}}. Title and metadata are visible on purpose
  (the production-context condition). Do not assume the value appears in them.
- Do NOT look for HTML supporting files or call `peek_file_text`: these
  entities have no HTML file. Sample their primary-document text with
  `peek_segmentation_by_entity(shared_id, "{language}")`.
- You may use the bound helpers `dateparse` and `textmatch` (documented in your
  system prompt) instead of hand-rolled date/normalization logic.

TRAINING EVIDENCE (the ONLY documents you may look at)
- {train_total} train document(s) are available through the read-only ports.
  Use `query_entities(mode="by_template", template_name="{template_name}",
  language="{language}")`, then sample 8-12 of them with
  `peek_segmentation_by_entity`. `query_entities(mode="by_ids")` is capped at
  five ids per call, so prefer `peek_segmentation_by_entity` directly.
- Candidate train shared_ids:
{_bullet_list(ids)}
- After drafting `extract`, SELF-PROVE it on the sampled literal
  (text, ctx) pairs before emitting, as your system prompt requires. Use
  `benchmark_extractor` when comparing competing strategies; it scores against
  the values these train entities already carry.

Return the function plus honest samples_total / samples_matched counts.
"""

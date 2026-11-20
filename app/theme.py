"""Themes and semantic colors for the Streamlit viewer.

A theme is a palette plus a sidebar color. Switching a theme mutates the module
level `PALETTE` / `STATUS_COLORS` / `TYPE_COLORS` dictionaries in place so code
that imported them keeps working.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

GRUVBOX_DARK = {
    "bg0": "#282828",
    "bg1": "#3c3836",
    "bg2": "#504945",
    "bg3": "#665c54",
    "fg": "#ebdbb2",
    "fg2": "#d5c4a1",
    "gray": "#928374",
    "red": "#fb4934",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "blue": "#83a598",
    "purple": "#d3869b",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
    "red_dark": "#cc241d",
    "green_dark": "#98971a",
    "yellow_dark": "#d79921",
    "blue_dark": "#458588",
    "purple_dark": "#b16286",
    "aqua_dark": "#689d6a",
    "orange_dark": "#d65d0e",
}

TOKYO_NIGHT = {
    "bg0": "#1a1b26",
    "bg1": "#24283b",
    "bg2": "#414868",
    "bg3": "#565f89",
    "fg": "#c0caf5",
    "fg2": "#a9b1d6",
    "gray": "#565f89",
    "red": "#f7768e",
    "green": "#9ece6a",
    "yellow": "#e0af68",
    "blue": "#7aa2f7",
    "purple": "#bb9af7",
    "aqua": "#7dcfff",
    "orange": "#ff9e64",
    "red_dark": "#db4b4b",
    "green_dark": "#73daca",
    "yellow_dark": "#e0af68",
    "blue_dark": "#2ac3de",
    "purple_dark": "#9d7cd8",
    "aqua_dark": "#41a6b5",
    "orange_dark": "#ff9e64",
}

THEMES: dict[str, dict[str, Any]] = {
    "Gruvbox Dark": {
        "palette": GRUVBOX_DARK,
        "sidebar": "#1d2021",
        "description": "Warm retro palette (lospec.com/palette-list/gruvbox-dark).",
    },
    "Tokyo Night": {
        "palette": TOKYO_NIGHT,
        "sidebar": "#16161e",
        "description": "Cool neon-night palette (Tokyo Night Dark).",
    },
}

DEFAULT_THEME = "Gruvbox Dark"
ACTIVE_THEME = DEFAULT_THEME
SIDEBAR_COLOR = THEMES[DEFAULT_THEME]["sidebar"]

# These dictionaries are intentionally mutable and shared by importers.
PALETTE: dict[str, str] = dict(THEMES[DEFAULT_THEME]["palette"])
STATUS_COLORS: dict[str, str] = {}
TYPE_COLORS: dict[str, str] = {}


def _rebuild_semantics() -> None:
    STATUS_COLORS.clear()
    STATUS_COLORS.update(
        {
            "hit": PALETTE["green"],
            "matched": PALETTE["green"],
            "partial": PALETTE["yellow"],
            "wrong": PALETTE["red"],
            "missed": PALETTE["red"],
            "error": PALETTE["red"],
            "missing_prediction": PALETTE["orange"],
            "extra": PALETTE["purple"],
            "train": PALETTE["blue"],
            "test": PALETTE["purple"],
            "completed": PALETTE["green"],
            "failed": PALETTE["red"],
        }
    )
    TYPE_COLORS.clear()
    TYPE_COLORS.update(
        {
            "rule_based": PALETTE["blue"],
            "llm_script": PALETTE["purple"],
            "admin_agent_extractor": PALETTE["orange"],
            "single_label": PALETTE["aqua"],
            "numeric": PALETTE["yellow"],
            "string": PALETTE["orange"],
            "multi_field": PALETTE["purple"],
            "record_extraction": PALETTE["blue"],
        }
    )


_rebuild_semantics()


def theme_names() -> list[str]:
    return list(THEMES)


def set_theme(name: str) -> None:
    global ACTIVE_THEME, SIDEBAR_COLOR
    if name not in THEMES:
        name = DEFAULT_THEME
    ACTIVE_THEME = name
    SIDEBAR_COLOR = THEMES[name]["sidebar"]
    PALETTE.clear()
    PALETTE.update(THEMES[name]["palette"])
    _rebuild_semantics()


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --gb-bg0: {PALETTE["bg0"]};
            --gb-bg1: {PALETTE["bg1"]};
            --gb-bg2: {PALETTE["bg2"]};
            --gb-bg3: {PALETTE["bg3"]};
            --gb-fg: {PALETTE["fg"]};
            --gb-fg2: {PALETTE["fg2"]};
            --gb-gray: {PALETTE["gray"]};
            --gb-yellow: {PALETTE["yellow"]};
            --gb-orange: {PALETTE["orange"]};
            --gb-aqua: {PALETTE["aqua"]};
            --gb-sidebar: {SIDEBAR_COLOR};
        }}
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 20% 0%, {_rgba(PALETTE["yellow"], 0.045)}, transparent 35%),
                radial-gradient(circle at 80% 10%, {_rgba(PALETTE["blue"], 0.045)}, transparent 30%),
                var(--gb-bg0) !important;
            color: var(--gb-fg) !important;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: var(--gb-sidebar) !important;
            border-right: 1px solid var(--gb-bg2);
        }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: var(--gb-fg2);
        }}
        h1, h2, h3, h4 {{
            color: var(--gb-fg) !important;
            letter-spacing: -0.01em;
        }}
        h1 {{ border-bottom: 1px solid var(--gb-bg2); padding-bottom: .4rem; }}
        p, li, label, .stMarkdown {{
            color: var(--gb-fg2);
        }}
        [data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--gb-bg1), {_rgba(PALETTE["bg0"], 0.9)});
            border: 1px solid var(--gb-bg2);
            border-radius: 10px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--gb-yellow) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--gb-gray) !important;
        }}
        [data-testid="stExpander"] {{
            background: {_rgba(PALETTE["bg1"], 0.35)} !important;
            border: 1px solid var(--gb-bg2) !important;
            border-radius: 10px;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--gb-bg2);
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stTable"] table {{
            background: var(--gb-bg1) !important;
            color: var(--gb-fg) !important;
            border-color: var(--gb-bg2) !important;
        }}
        [data-testid="stTable"] th {{
            background: var(--gb-bg2) !important;
            color: var(--gb-fg) !important;
            border-color: var(--gb-bg3) !important;
        }}
        [data-testid="stTable"] td {{
            color: var(--gb-fg) !important;
            border-color: var(--gb-bg2) !important;
        }}
        [data-testid="stTable"] tr:nth-child(odd) td {{
            background: var(--gb-bg0);
        }}
        [data-testid="stTable"] tr:nth-child(even) td {{
            background: var(--gb-bg1);
        }}
        [data-testid="stTable"] tr:hover td {{
            background: var(--gb-bg2);
        }}
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] button,
        [data-testid="stSelectbox"] [role="group"],
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="textarea"] {{
            background-color: var(--gb-bg1) !important;
            color: var(--gb-fg) !important;
            -webkit-text-fill-color: var(--gb-fg) !important;
            border-color: var(--gb-bg2) !important;
        }}
        [data-testid="stSelectbox"] svg {{
            fill: var(--gb-fg2) !important;
            color: var(--gb-fg2) !important;
        }}
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        [data-testid="stRadio"] [data-testid="stMarkdownContainer"] {{
            color: var(--gb-fg2) !important;
        }}
        [data-testid="stRadioOption"][data-selected="true"] p,
        [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] {{
            color: var(--gb-fg) !important;
        }}
        [data-testid="stRadioOption"] div[class*="ex50key4"] {{
            border-color: var(--gb-bg3) !important;
        }}
        [data-testid="stRadioOption"] div[class*="ex50key5"] {{
            background-color: var(--gb-yellow) !important;
        }}
        [data-baseweb="select"] *,
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {{
            color: var(--gb-fg) !important;
        }}
        [data-baseweb="select"] svg {{
            fill: var(--gb-fg2) !important;
        }}
        [data-baseweb="popover"], [role="listbox"] {{
            background-color: var(--gb-bg1) !important;
            border-color: var(--gb-bg2) !important;
        }}
        [role="option"] {{
            background-color: var(--gb-bg1) !important;
            color: var(--gb-fg) !important;
        }}
        [role="option"]:hover {{
            background-color: var(--gb-bg2) !important;
        }}
        [data-testid="stTabs"] button {{
            color: var(--gb-fg2);
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--gb-yellow);
            border-color: var(--gb-yellow);
        }}
        [data-testid="stCode"] pre, [data-testid="stCode"] code,
        [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {{
            background: var(--gb-bg1) !important;
            color: var(--gb-fg) !important;
        }}
        [data-testid="stText"], [data-testid="stText"] pre,
        [data-testid="stText"] code {{
            color: var(--gb-fg) !important;
        }}
        textarea {{
            color: var(--gb-fg) !important;
            background-color: var(--gb-bg1) !important;
            caret-color: var(--gb-yellow) !important;
        }}
        button[data-testid="stCopyButton"] {{
            color: var(--gb-fg2) !important;
        }}
        button[kind="secondary"], button[kind="primary"] {{
            border-color: var(--gb-bg2) !important;
        }}
        .uw-card {{
            background: linear-gradient(180deg, var(--gb-bg1) 0%, {_rgba(PALETTE["bg0"], 0.92)} 100%);
            border: 1px solid var(--gb-bg2);
            border-left: 4px solid var(--gb-yellow);
            border-radius: 12px;
            padding: 14px 16px 12px 16px;
            margin-bottom: 10px;
            min-height: 178px;
            transition: border-color .15s ease, transform .15s ease;
        }}
        .uw-card:hover {{
            border-color: var(--gb-yellow);
            transform: translateY(-1px);
        }}
        .uw-card-title {{
            color: var(--gb-fg);
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 2px;
        }}
        .uw-card-id {{
            color: var(--gb-gray);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: .76rem;
            margin-bottom: 6px;
        }}
        .uw-card-desc {{
            color: var(--gb-fg2);
            font-size: .82rem;
            line-height: 1.35;
            min-height: 2.2em;
        }}
        .uw-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: .70rem;
            font-weight: 700;
            letter-spacing: .02em;
            margin: 4px 5px 0 0;
            border: 1px solid transparent;
        }}
        .uw-stat-row {{
            display: flex;
            gap: 18px;
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid var(--gb-bg2);
        }}
        .uw-stat {{
            display: flex;
            flex-direction: column;
        }}
        .uw-stat b {{
            color: var(--gb-fg);
            font-size: 1.02rem;
            line-height: 1.1;
        }}
        .uw-stat span {{
            color: var(--gb-gray);
            font-size: .69rem;
            text-transform: uppercase;
            letter-spacing: .05em;
        }}
        .uw-section-note {{
            color: var(--gb-gray);
            font-size: .82rem;
            margin: -4px 0 12px 0;
        }}
        .uw-status {{
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: .78rem;
        }}
        .uw-divider {{
            border-top: 1px solid var(--gb-bg2);
            margin: 16px 0 12px 0;
        }}
        code, pre {{
            border-radius: 8px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str, *, outline: bool = True) -> str:
    background = _rgba(color, 0.13) if outline else color
    foreground = color if outline else PALETTE["bg0"]
    return (
        f'<span class="uw-badge" style="background:{background};'
        f'color:{foreground};border-color:{_rgba(color, 0.35)}">{html.escape(str(text))}</span>'
    )


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, PALETTE["gray"])
    return (
        f'<span class="uw-status" style="background:{_rgba(color, 0.15)};'
        f'color:{color}">{html.escape(status)}</span>'
    )


def status_style(frame: pd.DataFrame, status_column: str = "status") -> Any:
    if status_column not in frame.columns:
        return frame

    def color(value: Any) -> str:
        hex_color = STATUS_COLORS.get(str(value), PALETTE["gray"])
        return f"background-color:{_rgba(hex_color, 0.15)}; color:{hex_color}; font-weight:600"

    return frame.style.map(color, subset=[status_column])


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _blend(start: str, end: str, amount: float) -> str:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    mixed = tuple(round(a + (b - a) * amount) for a, b in zip(start_rgb, end_rgb, strict=True))
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def score_color(value: float) -> str:
    """Red → yellow → green without pulling in matplotlib."""
    value = max(0.0, min(1.0, float(value)))
    if value < 0.5:
        return _blend(PALETTE["red"], PALETTE["yellow"], value * 2)
    return _blend(PALETTE["yellow"], PALETTE["green"], (value - 0.5) * 2)


def metric_style(frame: pd.DataFrame, columns: list[str]) -> Any:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return frame

    def color(value: Any) -> str:
        if not isinstance(value, (int, float)) or pd.isna(value):
            return ""
        hex_color = score_color(float(value))
        return f"background-color:{_rgba(hex_color, 0.15)}; color:{hex_color}; font-weight:600"

    return frame.style.map(color, subset=present)


def table_style(
    frame: pd.DataFrame,
    *,
    status_column: str | None = None,
    metric_columns: list[str] | None = None,
) -> Any:
    """Combined Styler for HTML tables (status cells + metric gradients)."""
    styler = frame.style
    if status_column and status_column in frame.columns:

        def status_color(value: Any) -> str:
            hex_color = STATUS_COLORS.get(str(value), PALETTE["gray"])
            return f"background-color:{_rgba(hex_color, 0.18)}; color:{hex_color}; font-weight:600"

        styler = styler.map(status_color, subset=[status_column])
    present = [column for column in (metric_columns or []) if column in frame.columns]
    if present:

        def metric_color(value: Any) -> str:
            if not isinstance(value, (int, float)) or pd.isna(value):
                return ""
            hex_color = score_color(float(value))
            return f"background-color:{_rgba(hex_color, 0.16)}; color:{hex_color}; font-weight:600"

        styler = styler.map(metric_color, subset=present)
    return styler

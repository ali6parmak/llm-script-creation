"""Gruvbox Dark theme and semantic colors for the Streamlit viewer."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

PALETTE = {
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

STATUS_COLORS = {
    "hit": PALETTE["green"],
    "matched": PALETTE["green"],
    "partial": PALETTE["yellow"],
    "wrong": PALETTE["red"],
    "missed": PALETTE["red"],
    "error": PALETTE["red"],
    "missing_prediction": PALETTE["orange"],
    "extra": PALETTE["purple"],
    "train": PALETTE["blue"],
    "dev": PALETTE["yellow"],
    "test": PALETTE["purple"],
    "completed": PALETTE["green"],
    "failed": PALETTE["red"],
}

TYPE_COLORS = {
    "rule_based": PALETTE["blue"],
    "llm_script": PALETTE["purple"],
    "single_label": PALETTE["aqua"],
    "numeric": PALETTE["yellow"],
    "string": PALETTE["orange"],
    "multi_field": PALETTE["purple"],
    "record_extraction": PALETTE["blue"],
}


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
        }}
        .stApp {{
            background:
                radial-gradient(circle at 20% 0%, rgba(250, 189, 47, 0.04), transparent 35%),
                radial-gradient(circle at 80% 10%, rgba(131, 165, 152, 0.04), transparent 30%),
                var(--gb-bg0);
        }}
        [data-testid="stSidebar"] {{
            background-color: #1d2021;
            border-right: 1px solid var(--gb-bg2);
        }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: var(--gb-fg2);
        }}
        h1, h2, h3 {{
            color: var(--gb-fg);
            letter-spacing: -0.01em;
        }}
        h1 {{ border-bottom: 1px solid var(--gb-bg2); padding-bottom: .4rem; }}
        [data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--gb-bg1), rgba(40, 40, 40, 0.9));
            border: 1px solid var(--gb-bg2);
            border-radius: 10px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--gb-yellow);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--gb-gray);
        }}
        [data-testid="stExpander"] {{
            background: rgba(60, 56, 54, 0.35);
            border: 1px solid var(--gb-bg2);
            border-radius: 10px;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--gb-bg2);
            border-radius: 10px;
            overflow: hidden;
        }}
        .uw-card {{
            background: linear-gradient(180deg, #32302f 0%, #2b2a29 100%);
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
    background = f"{color}22" if outline else color
    foreground = color if outline else PALETTE["bg0"]
    return (
        f'<span class="uw-badge" style="background:{background};'
        f'color:{foreground};border-color:{color}55">{html.escape(str(text))}</span>'
    )


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, PALETTE["gray"])
    return f'<span class="uw-status" style="background:{color}22;color:{color}">{html.escape(status)}</span>'


def status_style(frame: pd.DataFrame, status_column: str = "status") -> Any:
    if status_column not in frame.columns:
        return frame

    def color(value: Any) -> str:
        hex_color = STATUS_COLORS.get(str(value), PALETTE["gray"])
        return f"background-color:{hex_color}22; color:{hex_color}; font-weight:600"

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
    """Gruvbox red → yellow → green without pulling in matplotlib."""
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
        return f"background-color:{hex_color}22; color:{hex_color}; font-weight:600"

    return frame.style.map(color, subset=present)

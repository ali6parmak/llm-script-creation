import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import theme  # noqa: E402


def test_theme_registry_switches_palette() -> None:
    try:
        theme.set_theme("Tokyo Night")
        assert theme.PALETTE["bg0"] == "#1a1b26"
        assert theme.ACTIVE_THEME == "Tokyo Night"
        assert theme.score_color(0.0).lower() == theme.PALETTE["red"].lower()
        theme.set_theme("Gruvbox Dark")
        assert theme.PALETTE["bg0"] == "#282828"
    finally:
        theme.set_theme("Gruvbox Dark")

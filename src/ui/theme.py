from __future__ import annotations

from pathlib import Path

FONT_BUNDLED = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansJP-Regular.otf"
FONT_CANDIDATES = ["Meiryo", "Yu Gothic", "MS Gothic"]

COL_BG = (12, 14, 22)
COL_GRID = (38, 44, 64)
COL_WHITE = (240, 240, 240)
COL_DIM = (160, 160, 180)
COL_GREEN = (60, 205, 100)
COL_RED = (235, 70, 80)
COL_YELLOW = (240, 210, 90)
COL_BLUE = (90, 170, 250)
COL_PURPLE = (160, 120, 255)

__all__ = [
    "FONT_BUNDLED",
    "FONT_CANDIDATES",
    "COL_BG",
    "COL_GRID",
    "COL_WHITE",
    "COL_DIM",
    "COL_GREEN",
    "COL_RED",
    "COL_YELLOW",
    "COL_BLUE",
    "COL_PURPLE",
]

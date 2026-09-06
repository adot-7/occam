"""The one place the Figma palette is written down.

``occam.tcss`` declares the same values as CSS variables at the top of the file
(Textual stylesheets cannot import Python), so the Figma reference is applied by
editing these constants and the matching ten lines of ``occam.tcss``.

Colour carries meaning consistently (04 §1): green = load-bearing / pass,
red = witness / fail, amber = uncertain, dim = pruned / dead branch,
cyan = current generation, magenta = baseline.
"""

from __future__ import annotations

BACKGROUND = "#0d1011"
SURFACE = "#121617"
BORDER = "#1f2624"
FG = "#c8d3d0"
DIM = "#6b7a77"
ACCENT = "#3ddc84"
GREEN = "#3ddc84"
RED = "#ff5f5f"
AMBER = "#ffb454"
CYAN = "#56c9e8"
MAGENTA = "#c792ea"

#: Verdict vocabulary from 03 §8, styled per 04 §3.4.
VERDICT_STYLES = {
    "load_bearing": f"bold {GREEN}",
    "witness": f"bold {RED}",
    "harmful": RED,
    "uncertain": AMBER,
}

__all__ = [
    "ACCENT",
    "AMBER",
    "BACKGROUND",
    "BORDER",
    "CYAN",
    "DIM",
    "FG",
    "GREEN",
    "MAGENTA",
    "RED",
    "SURFACE",
    "VERDICT_STYLES",
]

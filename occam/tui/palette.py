"""The Figma palette translated into Rich colours.

Textual stylesheets cannot import Python, so :mod:`occam.tui.occam.tcss` keeps
the matching values as stylesheet variables.  Keep the two small lists in sync
when the design reference changes.  Colour carries meaning consistently (04
§1): green is load-bearing / pass, red is witness / fail, amber is uncertain,
dim is pruned, cyan is current, and magenta is the baseline.
"""

from __future__ import annotations

BACKGROUND = "#0c0c0c"
SURFACE = "#111111"
SELECTED = "#1a1a1a"
BORDER = "#272727"
BORDER_HI = "#3c3c3c"
WIRE = "#383838"
FG_BRIGHT = "#ebebeb"
FG = "#cacaca"
DIM = "#8c8c8c"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
GREEN_BG = "#0d2018"
RED = "#f85149"
RED_BG = "#2d0e0e"
RED_DIM = "#581414"
AMBER = "#d29922"
AMBER_HI = "#fbbf24"
AMBER_BG = "#231908"
CYAN = "#58a6ff"
CYAN_BG = "#0d1e2d"
PURPLE = "#bc8cff"
MAGENTA = PURPLE

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
    "AMBER_BG",
    "AMBER_HI",
    "BACKGROUND",
    "BORDER",
    "BORDER_HI",
    "CYAN",
    "CYAN_BG",
    "DIM",
    "FG",
    "FG_BRIGHT",
    "GREEN",
    "GREEN_BG",
    "MAGENTA",
    "PURPLE",
    "RED",
    "RED_BG",
    "RED_DIM",
    "SELECTED",
    "SURFACE",
    "VERDICT_STYLES",
    "WIRE",
]

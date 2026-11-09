"""Llama Mono's glyph grid, exported from the design system's own font (src/lib/assets/fonts/LlamaMono.woff2) in the
shape of the Llama Sans specimen's grid, so GlyphGrid draws both. Every cell is one advance wide, so the cells are
too. The icon lockup (U+E002) is left out: it is wider than a cell. Run from tools/ui:

    uvx --with brotli --from fonttools python tests/stories/brand/llama-mono/data/build_glyphs.py
"""

import json
import os

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

FONT = "src/lib/assets/fonts/LlamaMono.woff2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyphs.json")
WEIGHT = 400
GROUPS = [("Uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
          ("Lowercase", "abcdefghijklmnopqrstuvwxyz"),
          ("Figures", "0123456789"),
          ("Punctuation & symbols", ".,:;!?'\"-–—()[]/&@#%*+=$€"),
          ("For code", "{}<>|\\~^`_"),
          ("Llama icon · U+E000", "")]


def bounds(font, ch):
    gs = font.getGlyphSet()
    pen = BoundsPen(gs)
    gs[font.getBestCmap()[ord(ch)]].draw(pen)
    return pen.bounds


def path(font, ch, dx, top):
    """SVG path of a glyph, flipped and placed (dx, top) in its cell, rounded to whole units."""
    name = font.getBestCmap()[ord(ch)]
    gs = font.getGlyphSet()
    pen = SVGPathPen(gs, ntos=lambda v: f"{v:.0f}")
    gs[name].draw(TransformPen(pen, (1, 0, 0, -1, round(dx), top)))
    return pen.getCommands(), font["hmtx"][name][0]


def main():
    font = instantiateVariableFont(TTFont(FONT), {"wght": WEIGHT})
    os2 = font["OS/2"]
    asc, dsc = round(bounds(font, "l")[3]), round(bounds(font, "p")[1])
    top, bot = asc + 60, dsc - 40
    chars = [c for _, s in GROUPS for c in s]
    vw = max(font["hmtx"][font.getBestCmap()[ord(c)]][0] for c in chars) + 60

    cells = []
    for c in chars:
        _, adv = path(font, c, 0, top)
        dx = (vw - adv) / 2
        d, _ = path(font, c, dx, top)
        cells.append({"ch": c, "adv": adv, "dx": round(dx), "d": d})

    grid = {
        "variant": "Mono", "width": vw, "height": top - bot,
        "guides": {"base": top, "x": top - os2.sxHeight, "cap": top - os2.sCapHeight,
                   "asc": top - asc, "dsc": top - dsc},
        "groups": [{"title": t, "chars": list(s)} for t, s in GROUPS],
        "cells": {str(WEIGHT): cells},
    }
    with open(OUT, "w") as f:
        json.dump(grid, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()

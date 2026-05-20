"""LaTeX parsing (input) and rendering (output) for the desktop solver.

Rendering is done with matplotlib's mathtext so we have no JS / web dependency.
Parsing uses SymPy's ``parse_latex`` when available, with a graceful fallback.
"""

from __future__ import annotations

import io
from typing import Optional

import sympy as sp


def parse_latex(s: str) -> Optional[sp.Expr]:
    """Parse a LaTeX math expression into a SymPy expression.

    Returns None if SymPy's optional LaTeX parser is unavailable or parsing fails.
    """
    try:
        from sympy.parsing.latex import parse_latex as _pl  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return _pl(s)
    except Exception:  # noqa: BLE001 - parse_latex raises a wide variety
        return None


def render_latex_png(latex: str, *, font_size: int = 20, dpi: int = 160) -> bytes:
    """Render a LaTeX (math-mode) string to PNG bytes via matplotlib mathtext.

    The string is wrapped in ``$...$`` if not already in math mode.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not latex:
        latex = r"\ "
    if not latex.startswith("$"):
        latex = f"${latex}$"

    fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
    try:
        fig.text(0, 0, latex, fontsize=font_size)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1,
                    transparent=True)
        return buf.getvalue()
    finally:
        plt.close(fig)

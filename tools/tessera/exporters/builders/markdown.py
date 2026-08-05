"""Markdown builder (exporter).

Converts an ``IntermediateDocument`` to a Markdown string.

The Markdown exporter is the simplest of the builders. It
walks the intermediate in document order and emits the
canonical Markdown form for each block type:

* heading -> ``#`` prefix matching the level
* paragraph -> text
* list (unordered) -> ``- item`` per item
* list (ordered) -> ``1. item`` per item
* list (task) -> ``- [ ] item`` / ``- [x] item`` per item
* listItem -> the text (handled by the list walker)
* table -> GFM table
* codeBlock -> fenced code block
* quote -> ``> `` prefixed
* divider -> ``---``
* image -> ``![alt](source)``
* equation -> ``$$latex$$``
* callout -> ``> **emoji** text`` (the AST has no callout
  rendering in v1; we use a quote + bold to surface the
  emoji)

Lossy round-trip notes:

* Bold / italic / links / inline code round-trip cleanly.
* Strikethrough round-trips via GFM's ``~~foo~~``.
* Underline / subscript / superscript / color round-trip as
  HTML (``<u>...</u>``, ``<sub>...</sub>``); a Markdown reader
  that doesn't accept HTML will see the raw tags.
* Tables round-trip via GFM.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.tessera.importers.intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


def build_markdown(doc: IntermediateDocument) -> str:
    """Render an IntermediateDocument to a Markdown string."""
    out: list[str] = []
    for ib in doc.blocks:
        rendered = _render_block(ib)
        if rendered:
            out.append(rendered)
    return "\n\n".join(out) + "\n"


def _render_block(ib: IntermediateBlock) -> str:
    if ib.type == "heading":
        level = ib.attrs.get("level", 1)
        text = _render_runs(ib.runs)
        return f"{'#' * max(1, min(6, int(level)))} {text}"
    if ib.type == "paragraph":
        return _render_runs(ib.runs)
    if ib.type == "list":
        return _render_list(ib)
    if ib.type == "table":
        return _render_table(ib)
    if ib.type == "codeBlock":
        lang = ib.attrs.get("language") or ""
        text = "".join(r.text for r in ib.runs)
        return f"```{lang}\n{text}\n```"
    if ib.type == "quote":
        text = _render_runs(ib.runs)
        return "\n".join("> " + line for line in text.splitlines())
    if ib.type == "divider":
        return "---"
    if ib.type == "image":
        alt = ib.attrs.get("alt", "")
        src = ib.attrs.get("source", "")
        return f"![{alt}]({src})"
    if ib.type == "equation":
        return f"$${ib.attrs.get('latex', '')}$$"
    if ib.type == "callout":
        emoji = ib.attrs.get("emoji", "")
        text = _render_runs(ib.runs)
        return f"> **{emoji}** {text}".strip()
    if ib.type == "listItem":
        # Unreachable: list items are rendered via the list walker.
        return _render_runs(ib.runs)
    return _render_runs(ib.runs)


def _render_list(ib: IntermediateBlock) -> str:
    style = ib.attrs.get("style", "unordered")
    children = [c for c in ib.children if hasattr(c, "type")]
    lines: list[str] = []
    if style == "ordered":
        for i, c in enumerate(children, 1):
            lines.append(f"{i}. {_render_runs(c.runs)}")
    elif style == "task":
        for c in children:
            checked = c.meta.get("checked", False)
            mark = "[x]" if checked else "[ ]"
            lines.append(f"- {mark} {_render_runs(c.runs)}")
    else:
        for c in children:
            lines.append(f"- {_render_runs(c.runs)}")
    return "\n".join(lines)


def _render_table(ib: IntermediateBlock) -> str:
    """Render a table block as a GFM table."""
    n_cols = int(ib.attrs.get("cols", 0))
    n_rows = int(ib.attrs.get("rows", 0))
    # Reconstruct the row grid from children (which are leaf
    # cell blocks). Children are listed in row-major order.
    children = [c for c in ib.children if hasattr(c, "type")]
    grid: list[list[str]] = []
    row: list[str] = []
    for c in children:
        row.append(_render_runs(c.runs))
        if len(row) == n_cols:
            grid.append(row)
            row = []
    if row:
        # Trailing partial row (shouldn't happen with well-formed
        # ASTs; pad with empty cells).
        while len(row) < n_cols:
            row.append("")
        grid.append(row)

    if not grid:
        return ""

    out: list[str] = []
    out.append("| " + " | ".join(grid[0]) + " |")
    out.append("| " + " | ".join("---" for _ in grid[0]) + " |")
    for r in grid[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _render_runs(runs: list[IntermediateInlineRun]) -> str:
    """Render a list of inline runs to a Markdown string."""
    out: list[str] = []
    for r in runs:
        out.append(_render_run(r))
    return "".join(out)


def _render_run(r: IntermediateInlineRun) -> str:
    """Render one inline run to its Markdown form."""
    text = r.text
    if not text:
        return ""
    # Apply annotations from outermost to innermost: link first
    # (innermost wrapping), then bold / italic, then code, then
    # HTML fallbacks for the rest.
    for ann in r.annotations:
        if isinstance(ann, str):
            if ann == "bold":
                text = f"**{text}**"
            elif ann == "italic":
                text = f"*{text}*"
            elif ann == "code":
                text = f"`{text}`"
            elif ann == "strikethrough":
                text = f"~~{text}~~"
            elif ann == "underline":
                text = f"<u>{text}</u>"
            elif ann == "subscript":
                text = f"<sub>{text}</sub>"
            elif ann == "superscript":
                text = f"<sup>{text}</sup>"
        elif isinstance(ann, dict):
            if "link" in ann:
                href = ann["link"]
                text = f"[{text}]({href})"
            elif "color" in ann:
                color = ann["color"]
                text = f'<span style="color:{color}">{text}</span>'
    return text

"""Pandoc parser (the swiss-army bridge).

For formats the dedicated parsers don't cover (RST, LaTeX, ODT,
EPUB, RTF, Org-mode, ...), we delegate to Pandoc.

The strategy:

1. Read the file as bytes.
2. Shell out to ``pandoc -f <input-format> -t json`` to get the
   Pandoc JSON AST. This is the canonical, lossless form Pandoc
   defines; every input format is normalised to it.
3. Walk the JSON AST and produce an ``IntermediateDocument``.

Why JSON AST, not HTML:

* HTML loses information (Pandoc's AST preserves quote levels,
  list types, table cell scopes, etc.). The HTML representation
  is what Pandoc uses for output, not for round-trip.
* The JSON AST is small and self-describing. We can walk it
  with a few dict / list operations and no DOM.

The walker is intentionally minimal: it handles the common
constructs (headings, paragraphs, lists, tables, code blocks,
quotes, links, emphasis) and silently drops the rest. This is
the same pragmatic stance the dedicated parsers take.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


# Map our file extension to the Pandoc input format name. Pandoc
# auto-detects most of the time (``-f markdown`` etc.) but
# explicit names are more reliable.
EXT_TO_PANDOC_FORMAT: dict[str, str] = {
    ".rst": "rst",
    ".tex": "latex",
    ".ltx": "latex",
    ".odt": "odt",
    ".epub": "epub",
    ".rtf": "rtf",
    ".org": "org",
    ".txt": "markdown",
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
}


def parse_pandoc(path: Path) -> IntermediateDocument:
    """Parse a file via Pandoc.

    The caller (``pipeline.py``) only routes here when the
    dedicated parsers don't match the format, so we don't worry
    about overlap with the markdown / html parsers.
    """
    log.debug("parse_pandoc: %s", path)
    if shutil.which("pandoc") is None:
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[
                        IntermediateInlineRun(
                            text=(
                                "pandoc not found; install it (brew install pandoc) "
                                "and re-import the file."
                            )
                        )
                    ],
                )
            ],
            meta={"format_error": "pandoc not found"},
        )

    fmt = EXT_TO_PANDOC_FORMAT.get(path.suffix.lower())
    args = ["pandoc", "-t", "json"]
    if fmt:
        args += ["-f", fmt]
    args.append(str(path))

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("parse_pandoc: pandoc timeout: %s", path)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text="(Pandoc parse timed out)")],
                )
            ],
            meta={"format_error": "timeout"},
        )

    if result.returncode != 0:
        log.warning("parse_pandoc: pandoc failed: %s: %s", path, result.stderr)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text=f"(pandoc failed: {result.stderr.strip()})")],
                )
            ],
            meta={"format_error": result.stderr.strip()},
        )

    try:
        ast = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        log.warning("parse_pandoc: invalid JSON: %s: %s", path, e)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text=f"(pandoc returned invalid JSON: {e})")],
                )
            ],
            meta={"format_error": f"invalid JSON: {e}"},
        )

    blocks = _walk_blocks(ast.get("blocks", []))
    return IntermediateDocument(
        blocks=blocks,
        meta={"pandoc_format": fmt, "pandoc_meta": ast.get("meta", {})},
    )


# ---------------------------------------------------------------------------
# Pandoc JSON AST walker
# ---------------------------------------------------------------------------


def _walk_blocks(blocks: list[dict[str, Any]]) -> list[IntermediateBlock]:
    """Walk a list of Pandoc blocks and emit IntermediateBlocks."""
    out: list[IntermediateBlock] = []
    for b in blocks:
        ib = _walk_block(b)
        if ib is not None:
            out.append(ib)
    return out


def _walk_block(b: dict[str, Any]) -> IntermediateBlock | None:
    t = b.get("t")
    if t == "Header":
        level = int(b.get("level", 1))
        level = max(1, min(level, 6))
        runs = _walk_inlines(b.get("c", []))
        return IntermediateBlock(type="heading", attrs={"level": level}, runs=runs)
    if t == "Para":
        runs = _walk_inlines(b.get("c", []))
        text = "".join(r.text for r in runs)
        if not text.strip():
            return None
        return IntermediateBlock(type="paragraph", runs=runs)
    if t == "Plain":
        runs = _walk_inlines(b.get("c", []))
        text = "".join(r.text for r in runs)
        if not text.strip():
            return None
        return IntermediateBlock(type="paragraph", runs=runs)
    if t == "BulletList":
        items: list[IntermediateBlock] = []
        for item in b.get("c", []):
            ibs = _walk_blocks(item)
            if ibs:
                # A list item is one block (typically a Para).
                # We use the first block's runs as the item text.
                head = ibs[0]
                # If the first block has children (e.g. nested
                # list), flatten the runs.
                runs = list(head.runs)
                items.append(
                    IntermediateBlock(
                        type="listItem",
                        runs=runs if runs else [IntermediateInlineRun(text="")],
                    )
                )
        if not items:
            return None
        return IntermediateBlock(
            type="list",
            attrs={"style": "unordered", "items": [it.id for it in items]},
            children=items,
        )
    if t == "OrderedList":
        # OrderedList attrs: (start, [style, ...], listAttributes)
        items = []
        for item in b.get("c", []):
            ibs = _walk_blocks(item)
            if ibs:
                head = ibs[0]
                items.append(
                    IntermediateBlock(
                        type="listItem",
                        runs=list(head.runs),
                    )
                )
        if not items:
            return None
        return IntermediateBlock(
            type="list",
            attrs={"style": "ordered", "items": [it.id for it in items]},
            children=items,
        )
    if t == "BlockQuote":
        inner = _walk_blocks(b.get("c", []))
        text = "\n".join(
            "".join(r.text for r in ib.runs) for ib in inner if ib.runs
        ).strip()
        if not text:
            return None
        return IntermediateBlock(
            type="quote",
            runs=[IntermediateInlineRun(text=text)],
        )
    if t == "CodeBlock":
        info = (b.get("c", [[]])[0] or "").strip() or None
        text = b.get("c", ["", ""])[1] if len(b.get("c", [])) > 1 else ""
        return IntermediateBlock(
            type="codeBlock",
            attrs={"language": info} if info else {},
            runs=[IntermediateInlineRun(text=text)],
        )
    if t == "HorizontalRule":
        return IntermediateBlock(type="divider")
    if t == "Table":
        # Table: [attr, caption, [colwidths], [header], [body]]
        # The first non-header row is a list of cells; each cell
        # is a list of blocks.
        head_row = b.get("c", [None, None, None, [], []])[3]
        body_rows = b.get("c", [None, None, None, [], []])[4]
        grid: list[list[IntermediateBlock]] = []
        for row in head_row + body_rows:
            row_blocks: list[IntermediateBlock] = []
            for cell in row:
                cell_blocks = _walk_blocks(cell)
                if cell_blocks:
                    head_cell = cell_blocks[0]
                    row_blocks.append(
                        IntermediateBlock(
                            type="paragraph",
                            runs=list(head_cell.runs),
                        )
                    )
                else:
                    row_blocks.append(IntermediateBlock(type="paragraph", runs=[]))
            grid.append(row_blocks)
        if not grid:
            return IntermediateBlock(type="table", attrs={"rows": 0, "cols": 0, "cells": []})
        n_cols = max(len(r) for r in grid)
        for r in grid:
            while len(r) < n_cols:
                r.append(IntermediateBlock(type="paragraph", runs=[]))
        flat: list[IntermediateBlock] = [c for row in grid for c in row]
        return IntermediateBlock(
            type="table",
            attrs={
                "rows": len(grid),
                "cols": n_cols,
                "cells": [[c.id for c in row] for row in grid],
            },
            children=flat,
        )
    if t == "Div":
        # Recurse into the Div's blocks; flatten.
        return None  # The caller walks inner blocks separately.
    if t == "RawBlock":
        # Raw HTML / LaTeX: ignore in v1.
        return None
    # Unknown block type: log and skip.
    log.debug("parse_pandoc: unsupported block type %r", t)
    return None


def _walk_inlines(elements: list[Any]) -> list[IntermediateInlineRun]:
    """Walk a Pandoc inline list and produce IntermediateInlineRuns.

    Pandoc's inline representation is a list whose elements are
    either plain strings (text) or ``{"t": ..., "c": ...}`` for
    formatting constructs. We accumulate text into a buffer and
    flush as a run when the annotation stack changes.
    """
    out: list[IntermediateInlineRun] = []
    annotations: list[Any] = []
    text_parts: list[str] = []

    def flush() -> None:
        if text_parts:
            out.append(
                IntermediateInlineRun(
                    text="".join(text_parts),
                    annotations=list(annotations),
                )
            )
            text_parts.clear()

    def push_ann(ann: Any) -> None:
        flush()
        annotations.append(ann)

    def pop_ann(matcher: Any) -> None:
        # Pop the most recently pushed annotation that matches.
        for j in range(len(annotations) - 1, -1, -1):
            if annotations[j] == matcher:
                annotations.pop(j)
                flush()
                return
        flush()

    for el in elements:
        if isinstance(el, str):
            text_parts.append(el)
            continue
        if not isinstance(el, dict):
            continue
        t = el.get("t")
        c = el.get("c", [])
        if t == "Str":
            text_parts.append(c)
        elif t == "Space":
            text_parts.append(" ")
        elif t == "SoftBreak":
            text_parts.append("\n")
        elif t == "LineBreak":
            text_parts.append("\n")
        elif t == "Emph":
            push_ann("italic")
            out.extend(_walk_inlines(c))
            pop_ann("italic")
        elif t == "Strong":
            push_ann("bold")
            out.extend(_walk_inlines(c))
            pop_ann("bold")
        elif t == "Strikeout":
            push_ann("strikethrough")
            out.extend(_walk_inlines(c))
            pop_ann("strikethrough")
        elif t == "Code":
            # Inline code: [attr, text]
            text = c[1] if len(c) > 1 else (c[0] if c else "")
            flush()
            out.append(IntermediateInlineRun(text=str(text), annotations=["code"]))
        elif t == "Link":
            # Link: [attr, inlines, target]
            target = c[2] if len(c) > 2 else (c[1] if len(c) > 1 else "")
            # Push a link annotation; the text is the link text.
            push_ann({"link": target})
            out.extend(_walk_inlines(c[1] if len(c) > 1 else []))
            pop_ann("link")
        elif t == "Image":
            # Image: [attr, inlines (alt), target]
            alt_parts = _walk_inlines(c[1] if len(c) > 1 else [])
            alt = "".join(r.text for r in alt_parts)
            target = c[2] if len(c) > 2 else ""
            flush()
            out.append(
                IntermediateInlineRun(
                    text=alt,
                    annotations=[{"link": target}] if target else [],
                )
            )
        else:
            # Unknown inline: skip in v1
            pass
    flush()
    return out

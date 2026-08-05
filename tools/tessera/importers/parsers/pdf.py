"""PDF parser.

Reads a .pdf file in two passes:

1. ``pdftotext -layout`` extracts the text content with
   reasonable whitespace preservation. The text is split into
   lines; we group them into paragraphs by blank-line gaps and
   into headings by visual heuristics (short lines,
   all-caps).
2. ``weasyprint`` is NOT used here (it's an HTML-to-PDF
   renderer, not a PDF parser). For v1 we shell out to
   ``pdftotext`` only. A render-to-HTML pass is planned for v2
   (it would let us preserve visual layout, but pdftotext is
   good enough for text recovery on text-based PDFs and is
   available on every macOS dev machine via poppler).

Limitations:

* Scanned PDFs (image-only) produce no output. OCR is a v2
  feature.
* Multi-column layouts are flattened left-to-right. v2 may
  use ``pdfplumber`` to detect columns.
* Tables are best-effort. pdftotext preserves tab-separated
  cells; we detect them by tab presence.
* Footnotes / endnotes are concatenated at the end of each
  page's text.

Punted (v1):

* OCR for scanned PDFs.
* Image extraction (the PDF parser doesn't carry image bytes
  forward in v1).
* Form fields.
"""

from __future__ import annotations

import logging
import re
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


def parse_pdf(path: Path) -> IntermediateDocument:
    """Parse a .pdf file via ``pdftotext -layout``.

    The binary is part of poppler (``brew install poppler`` on
    macOS, ``apt install poppler-utils`` on Linux). When not
    present we fall back to reading the file as text and
    returning a single error block.
    """
    log.debug("parse_pdf: %s", path)
    if shutil.which("pdftotext") is None:
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[
                        IntermediateInlineRun(
                            text=(
                                "pdftotext not found; install poppler "
                                "(brew install poppler / apt install poppler-utils) "
                                "and re-import the PDF."
                            )
                        )
                    ],
                )
            ],
            meta={"format_error": "pdftotext not found"},
        )

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("parse_pdf: pdftotext timeout: %s", path)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text="(PDF parse timed out)")],
                )
            ],
            meta={"format_error": "timeout"},
        )

    if result.returncode != 0:
        log.warning("parse_pdf: pdftotext failed: %s: %s", path, result.stderr)
        return IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text=f"(pdftotext failed: {result.stderr.strip()})")],
                )
            ],
            meta={"format_error": result.stderr.strip()},
        )

    text = result.stdout
    blocks = _text_to_blocks(text)
    return IntermediateDocument(blocks=blocks, meta={"format": "pdf"})


def _text_to_blocks(text: str) -> list[IntermediateBlock]:
    """Convert pdftotext output to a list of blocks.

    Heuristics:

    * A blank line separates paragraphs.
    * A short line (<=80 chars) in ALL CAPS or with no terminal
      punctuation is treated as a heading.
    * A line that contains only ``-`` or ``=`` of length >= 10
      is the underline of a Setext-style heading; we treat the
      preceding line as a heading (level 1 for ``=``, level 2
      for ``-``).
    * A line starting with ``-`` / ``*`` / ``[0-9]+.`` is a
      list item.
    * A line with multiple tab characters is a table row; we
      collect consecutive tab-rows into one table block.
    """
    blocks: list[IntermediateBlock] = []
    paragraphs: list[str] = []
    list_items_buffer: list[str] = []
    table_rows_buffer: list[list[str]] = []

    def flush_paragraphs() -> None:
        for p in paragraphs:
            p = p.strip()
            if p:
                blocks.append(
                    IntermediateBlock(
                        type="paragraph",
                        runs=[IntermediateInlineRun(text=p)],
                    )
                )
        paragraphs.clear()

    def flush_list() -> None:
        items = [
            IntermediateBlock(
                type="listItem",
                runs=[IntermediateInlineRun(text=t)],
            )
            for t in list_items_buffer
            if t.strip()
        ]
        if items:
            blocks.append(
                IntermediateBlock(
                    type="list",
                    attrs={"style": "unordered", "items": [it.id for it in items]},
                    children=items,
                )
            )
        list_items_buffer.clear()

    def flush_table() -> None:
        if not table_rows_buffer:
            return
        # Pad rows to a uniform column count.
        n_cols = max(len(r) for r in table_rows_buffer)
        grid: list[list[IntermediateBlock]] = []
        for row in table_rows_buffer:
            cells = row + [""] * (n_cols - len(row))
            grid.append(
                [
                    IntermediateBlock(
                        type="paragraph",
                        runs=[IntermediateInlineRun(text=c.strip())] if c.strip() else [],
                    )
                    for c in cells
                ]
            )
        flat: list[IntermediateBlock] = [c for row in grid for c in row]
        blocks.append(
            IntermediateBlock(
                type="table",
                attrs={
                    "rows": len(grid),
                    "cols": n_cols,
                    "cells": [[c.id for c in row] for row in grid],
                },
                children=flat,
            )
        )
        table_rows_buffer.clear()

    def is_list_item(line: str) -> bool:
        return bool(re.match(r"^\s*([-*•]|\d+[.)])\s+", line))

    def is_table_row(line: str) -> bool:
        # A line is a table row if it has 2+ tab characters.
        return line.count("\t") >= 2

    def looks_like_heading(line: str) -> int | None:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return None
        if stripped.endswith((".", "!", "?", ":")):
            return None
        if stripped.isupper() and len(stripped.split()) <= 12:
            return 1
        if re.match(r"^(\d+\.?\d*)\s+[A-Z]", stripped) and len(stripped.split()) <= 12:
            return 2
        return None

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Setext-style heading: next line is all === or ---
        if i + 1 < len(lines) and stripped:
            nxt = lines[i + 1].strip()
            if re.fullmatch(r"=+", nxt) and len(nxt) >= 3:
                flush_paragraphs()
                flush_list()
                flush_table()
                blocks.append(
                    IntermediateBlock(
                        type="heading",
                        attrs={"level": 1},
                        runs=[IntermediateInlineRun(text=stripped)],
                    )
                )
                i += 2
                continue
            if re.fullmatch(r"-+", nxt) and len(nxt) >= 3:
                flush_paragraphs()
                flush_list()
                flush_table()
                blocks.append(
                    IntermediateBlock(
                        type="heading",
                        attrs={"level": 2},
                        runs=[IntermediateInlineRun(text=stripped)],
                    )
                )
                i += 2
                continue

        if not stripped:
            flush_paragraphs()
            flush_list()
            flush_table()
            i += 1
            continue

        # Table row
        if is_table_row(line):
            flush_paragraphs()
            flush_list()
            cells = line.split("\t")
            table_rows_buffer.append(cells)
            i += 1
            continue
        else:
            flush_table()

        # List item
        if is_list_item(line):
            flush_paragraphs()
            text_only = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line)
            list_items_buffer.append(text_only)
            i += 1
            continue
        else:
            flush_list()

        # Heading?
        level = looks_like_heading(line)
        if level is not None:
            flush_paragraphs()
            blocks.append(
                IntermediateBlock(
                    type="heading",
                    attrs={"level": level},
                    runs=[IntermediateInlineRun(text=stripped)],
                )
            )
            i += 1
            continue

        # Default: paragraph (collect contiguous non-blank lines).
        paragraphs.append(stripped)
        i += 1

    flush_paragraphs()
    flush_list()
    flush_table()
    return blocks

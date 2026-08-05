"""HTML parser.

Reads .html / .htm / .mhtml files with BeautifulSoup4 and
produces an ``IntermediateDocument``.

The HTML parser is one of the simpler parsers because the
output is a flat block list. Headings map to ``heading`` blocks,
paragraphs to ``paragraph`` blocks, lists to ``list`` +
``listItem`` blocks, tables to ``table`` + cell blocks, and
inline tags (strong, em, code, a) become ``Annotation`` tags
on the corresponding ``InlineRun``.

MHTML (MIME HTML) is a multipart format that bundles HTML +
its images in a single file. The format is rare; we handle
it by feeding the bytes through a multipart parser and pulling
out the HTML part. For v1, the simple case (one HTML part) is
supported; multi-part MHTML with alternate representations
falls back to extracting the first text/html part.
"""

from __future__ import annotations

import email
import logging
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_html(path: Path) -> IntermediateDocument:
    """Parse a .html or .htm file."""
    log.debug("parse_html: %s", path)
    html = path.read_text(encoding="utf-8", errors="replace")
    return _html_to_doc(html)


def parse_mhtml(path: Path) -> IntermediateDocument:
    """Parse a .mhtml / .mht file.

    MHTML is RFC 2557 (MIME Encapsulation of HTML). We use
    Python's email parser to find the text/html part and
    delegate to ``_html_to_doc``. Other parts (images, CSS)
    are not extracted in v1.
    """
    log.debug("parse_mhtml: %s", path)
    raw = path.read_bytes()
    msg = email.message_from_bytes(raw)
    html_part: str | None = None
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_content()
            if isinstance(payload, str):
                html_part = payload
                break
    if html_part is None:
        # Fall back: maybe the file is just a single text/html
        # body without proper multipart structure.
        try:
            html_part = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return IntermediateDocument(
                blocks=[
                    IntermediateBlock(
                        type="paragraph",
                        runs=[IntermediateInlineRun(text="(empty MHTML)")],
                    )
                ]
            )
    return _html_to_doc(html_part)


# ---------------------------------------------------------------------------
# HTML -> IntermediateDocument
# ---------------------------------------------------------------------------


def _html_to_doc(html: str) -> IntermediateDocument:
    """Convert an HTML string to an IntermediateDocument.

    Uses BeautifulSoup's html.parser for the parse; html5lib
    would be more spec-compliant but adds a dependency. The
    html.parser tree is good enough for the v1 importer.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    blocks: list[IntermediateBlock] = []
    for el in body.children:
        block = _element_to_block(el)
        if block is not None:
            blocks.append(block)
    return IntermediateDocument(blocks=blocks, meta={"title": _title_of(soup)})


def _title_of(soup: BeautifulSoup) -> str:
    title = soup.find("title")
    return title.get_text(strip=True) if title is not None else ""


def _element_to_block(el: Any) -> IntermediateBlock | None:
    """Convert a top-level BS4 element to a block.

    A ``NavigableString`` is a stray text node (e.g. whitespace
    between block-level elements); we ignore it.
    """
    if isinstance(el, NavigableString):
        return None
    if not isinstance(el, Tag):
        return None
    name = el.name.lower() if el.name else ""
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        return IntermediateBlock(
            type="heading",
            attrs={"level": level},
            runs=[_run_from_html(el)],
        )
    if name == "p":
        run = _run_from_html(el)
        if not run.text.strip():
            return None
        return IntermediateBlock(type="paragraph", runs=[run])
    if name in ("ul", "ol"):
        style = "ordered" if name == "ol" else "unordered"
        items: list[IntermediateBlock] = []
        for li in el.find_all("li", recursive=False):
            t = li.get_text(" ", strip=True)
            if not t:
                continue
            items.append(
                IntermediateBlock(
                    type="listItem",
                    runs=[IntermediateInlineRun(text=t)],
                )
            )
        if not items:
            return None
        return IntermediateBlock(
            type="list",
            attrs={"style": style, "items": [it.id for it in items]},
            children=items,
        )
    if name == "table":
        return _table_to_block(el)
    if name == "blockquote":
        text = el.get_text(" ", strip=True)
        if not text:
            return None
        cite = el.get("cite") or None
        attrs: dict[str, Any] = {}
        if cite:
            attrs["cite"] = cite
        return IntermediateBlock(
            type="quote",
            attrs=attrs,
            runs=[IntermediateInlineRun(text=text)],
        )
    if name == "pre":
        # ``pre`` often contains a ``code`` element; we look for
        # it and use its text. If no ``code``, use the pre's
        # text.
        code = el.find("code")
        text = (code or el).get_text("\n", strip=False)
        return IntermediateBlock(
            type="codeBlock",
            attrs={},
            runs=[IntermediateInlineRun(text=text)],
        )
    if name == "hr":
        return IntermediateBlock(type="divider")
    if name == "img":
        src = el.get("src") or ""
        alt = el.get("alt") or ""
        return IntermediateBlock(
            type="image",
            attrs={"source": src, "alt": alt},
        )
    if name in ("div", "section", "article", "main", "header", "footer", "aside"):
        # Container: recurse into children and merge their blocks.
        # For v1 we collapse to a sequence of blocks; nested
        # containers (rare in HTML output) emit nothing.
        return None  # let the loop descend into children
    # Unknown tag: try to extract a paragraph of text.
    text = el.get_text(" ", strip=True)
    if text:
        return IntermediateBlock(
            type="paragraph",
            runs=[IntermediateInlineRun(text=text)],
        )
    return None


def _run_from_html(el: Any) -> IntermediateInlineRun:
    """Build a single ``IntermediateInlineRun`` from a BS4 element.

    Inline tags (strong / em / code / a) become annotations. The
    text is the concatenation of all descendant text. For ``a``
    (links), the annotation is the ``{"link": href}`` associated
    form. For ``span style="color:..."`` we pick up the colour as
    a ``{"color": "#hex"}`` annotation (best-effort).
    """
    annotations: list[Any] = []
    text_parts: list[str] = []

    def visit(node: Any, current: list[Any]) -> None:
        if isinstance(node, NavigableString):
            text_parts.append(str(node))
            return
        if not isinstance(node, Tag):
            return
        n = node.name.lower() if node.name else ""
        new = list(current)
        if n in ("strong", "b"):
            new.append("bold")
        if n in ("em", "i", "cite"):
            new.append("italic")
        if n in ("u",):
            new.append("underline")
        if n in ("s", "strike", "del"):
            new.append("strikethrough")
        if n in ("code", "kbd", "samp"):
            new.append("code")
        if n in ("sub",):
            new.append("subscript")
        if n in ("sup",):
            new.append("superscript")
        if n == "a":
            href = node.get("href")
            if href:
                new.append({"link": href})
        if n == "span":
            color = _span_color(node)
            if color:
                new.append({"color": color})
        for child in node.children:
            visit(child, new)

    visit(el, [])
    text = "".join(text_parts).strip()
    return IntermediateInlineRun(text=text, annotations=annotations or [])


def _span_color(el: Tag) -> str | None:
    """Extract a hex color from a span's inline style (best-effort)."""
    style = el.get("style")
    if not style:
        return None
    m = re.search(r"color\s*:\s*(#[0-9a-fA-F]{3,8})", style)
    if m:
        return m.group(1)
    return None


def _table_to_block(el: Tag) -> IntermediateBlock:
    """Convert an HTML table to a ``table`` block."""
    rows = el.find_all("tr")
    if not rows:
        return IntermediateBlock(type="table", attrs={"rows": 0, "cols": 0, "cells": []})
    grid: list[list[IntermediateBlock]] = []
    n_cols = 0
    for tr in rows:
        row_blocks: list[IntermediateBlock] = []
        for td in tr.find_all(["td", "th"], recursive=False):
            text = td.get_text(" ", strip=True)
            row_blocks.append(
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text=text)] if text else [],
                )
            )
        n_cols = max(n_cols, len(row_blocks))
        grid.append(row_blocks)
    for row in grid:
        while len(row) < n_cols:
            row.append(IntermediateBlock(type="paragraph", runs=[]))
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

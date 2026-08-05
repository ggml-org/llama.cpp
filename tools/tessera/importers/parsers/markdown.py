"""Markdown parser.

Reads a .md / .markdown file with markdown-it-py and produces an
``IntermediateDocument``. The parser walks markdown-it-py's
token stream (not the rendered HTML) so we get a structured view
of the document.

Coverage (v1):

* ATX headings (# .. ######) -> ``heading`` blocks.
* Setext headings (=== and ---) -> ``heading`` blocks.
* Paragraphs -> ``paragraph`` blocks.
* Bullet lists (-, *, +) -> ``list`` + ``listItem`` blocks.
* Ordered lists (1. 1) ...) -> ordered ``list`` blocks.
* Task lists (GFM: ``- [x]``) -> ``list`` with style ``"task"``.
* Code blocks (fenced + indented) -> ``codeBlock``.
* Inline code (`` ` ``) -> ``code`` annotation.
* Emphasis (``*foo*``) -> ``italic`` annotation.
* Strong (``**foo**``) -> ``bold`` annotation.
* Links (``[text](href)``) -> ``{"link": href}`` annotation.
* Strikethrough (GFM: ``~~foo~~``) -> ``strikethrough``.
* Tables (GFM) -> ``table`` + cell blocks.
* Block quotes (>) -> ``quote`` blocks (a single ``>`` becomes
  one block; nested ``>`` becomes nested blocks; v1 keeps the
  outer block and concatenates the inner text).
* Horizontal rules (---) -> ``divider``.
* Images (![alt](src)) -> ``image`` blocks.

Punted (v1):

* Footnotes (markdown-it has them but the AST doesn't have a
  footnote type; v1 surfaces them as paragraphs).
* Definition lists. Rare in practice.
* Math (KaTeX / LaTeX). The AST has an ``equation`` block but
  we don't auto-detect $...$ in v1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from ..intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


def parse_markdown(path: Path) -> IntermediateDocument:
    """Parse a Markdown file into an IntermediateDocument."""
    log.debug("parse_markdown: %s", path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_markdown_string(text)


def parse_markdown_string(text: str) -> IntermediateDocument:
    """Parse a Markdown string into an IntermediateDocument.

    Uses markdown-it-py with the GFM plugin enabled so tables
    and task lists work. ``html: False`` so we don't get an
    XSS surface in the parser output (the parser doesn't
    render HTML anyway, but it's the right default).
    """
    md = MarkdownIt("commonmark", {"html": False, "breaks": False, "linkify": True})
    md.enable(["table", "strikethrough"])
    try:
        md.enable(["tasklist"])
    except Exception:  # noqa: BLE001
        # Task lists are an optional plugin; missing is fine.
        pass
    tokens = md.parse(text)
    blocks, _ = _walk(tokens, 0)
    return IntermediateDocument(blocks=blocks)


# ---------------------------------------------------------------------------
# Token walker
# ---------------------------------------------------------------------------


def _walk(tokens: list[Any], start: int) -> tuple[list[IntermediateBlock], int]:
    """Walk a list of markdown-it tokens starting at `start`.

    Returns ``(blocks, next_index)`` where ``next_index`` is the
    index of the first token AFTER the last one consumed. The
    walker is recursive: container_open / container_close pairs
    are unwrapped here.
    """
    blocks: list[IntermediateBlock] = []
    i = start
    while i < len(tokens):
        tok = tokens[i]
        kind = tok.type
        if kind == "heading_open":
            level = int(tok.tag[1])  # 'h1'..'h6'
            # Next token is inline; following that is heading_close.
            inline_tok = tokens[i + 1]
            assert inline_tok.type == "inline", f"expected inline after heading_open, got {inline_tok.type}"
            runs = _runs_from_inline(inline_tok.children or [])
            blocks.append(
                IntermediateBlock(
                    type="heading",
                    attrs={"level": level},
                    runs=runs,
                )
            )
            i += 3  # open, inline, close
            continue
        if kind == "paragraph_open":
            inline_tok = tokens[i + 1]
            assert inline_tok.type == "inline", f"expected inline, got {inline_tok.type}"
            runs = _runs_from_inline(inline_tok.children or [])
            text = "".join(r.text for r in runs)
            if text.strip():
                blocks.append(IntermediateBlock(type="paragraph", runs=runs))
            i += 3
            continue
        if kind == "bullet_list_open":
            items, i = _walk_list(tokens, i + 1, ordered=False)
            if items:
                blocks.append(
                    IntermediateBlock(
                        type="list",
                        attrs={"style": "unordered", "items": [it.id for it in items]},
                        children=items,
                    )
                )
            continue
        if kind == "ordered_list_open":
            items, i = _walk_list(tokens, i + 1, ordered=True)
            if items:
                blocks.append(
                    IntermediateBlock(
                        type="list",
                        attrs={"style": "ordered", "items": [it.id for it in items]},
                        children=items,
                    )
                )
            continue
        if kind == "task_list_open":
            items, i = _walk_list(tokens, i + 1, ordered=False, task=True)
            if items:
                blocks.append(
                    IntermediateBlock(
                        type="list",
                        attrs={"style": "task", "items": [it.id for it in items]},
                        children=items,
                    )
                )
            continue
        if kind == "fence" or kind == "code_block":
            lang = (tok.info or "").strip() or None
            blocks.append(
                IntermediateBlock(
                    type="codeBlock",
                    attrs={"language": lang} if lang else {},
                    runs=[IntermediateInlineRun(text=tok.content.rstrip("\n"))],
                )
            )
            i += 1
            continue
        if kind == "blockquote_open":
            inner, i = _walk(tokens, i + 1)
            # v1 flattens: concatenate inner paragraph texts into
            # a single quote block.
            text = "\n".join(
                "".join(r.text for r in b.runs) for b in inner if b.runs
            ).strip()
            if text:
                blocks.append(
                    IntermediateBlock(
                        type="quote",
                        runs=[IntermediateInlineRun(text=text)],
                    )
                )
            continue
        if kind == "hr":
            blocks.append(IntermediateBlock(type="divider"))
            i += 1
            continue
        if kind == "table_open":
            table_block, i = _walk_table(tokens, i)
            blocks.append(table_block)
            continue
        if kind == "inline":
            # Stray inline at top level: wrap as a paragraph.
            runs = _runs_from_inline(tok.children or [])
            text = "".join(r.text for r in runs)
            if text.strip():
                blocks.append(IntermediateBlock(type="paragraph", runs=runs))
            i += 1
            continue
        if kind.endswith("_open") or kind.endswith("_close"):
            # An open without a matching walker above is a sign
            # we've hit an unsupported construct. Skip it; the
            # next iteration handles the next token.
            i += 1
            continue
        i += 1
    return blocks, i


def _walk_list(
    tokens: list[Any], start: int, *, ordered: bool, task: bool = False
) -> tuple[list[IntermediateBlock], int]:
    """Walk a list_open .. list_close span and emit listItem blocks.

    ``task`` is set when the parent was a ``task_list_open``;
    in v1 we collapse task items into a regular listItem with
    a ``checked`` attribute in meta (the AST doesn't have a
    task-list-item type, so we carry the state in the cell's
    meta and let the editor surface it as a checkbox).
    """
    items: list[IntermediateBlock] = []
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.type in ("bullet_list_close", "ordered_list_close", "task_list_close"):
            return items, i + 1
        if tok.type == "list_item_open":
            # Skip past list_item_open, then read the inline
            # tokens until list_item_close.
            i += 1
            checked = False
            inner_blocks: list[IntermediateBlock] = []
            while i < len(tokens) and tokens[i].type != "list_item_close":
                t = tokens[i]
                if t.type == "paragraph_open":
                    inline_tok = tokens[i + 1]
                    runs = _runs_from_inline(inline_tok.children or [])
                    text = "".join(r.text for r in runs)
                    if text.strip():
                        # In a task list, a leading [ ] / [x]
                        # is in the first paragraph's text;
                        # strip it for the v1 representation.
                        if task and "[ ]" in text or task and "[x]" in text:
                            checked = "[x]" in text
                            text = text.replace("[x]", "").replace("[ ]", "").strip()
                        inner_blocks.append(
                            IntermediateBlock(type="paragraph", runs=runs)
                        )
                    i += 3
                    continue
                if t.type == "inline":
                    runs = _runs_from_inline(t.children or [])
                    inner_blocks.append(
                        IntermediateBlock(type="paragraph", runs=runs)
                    )
                    i += 1
                    continue
                i += 1
            # Flatten inner paragraphs into the listItem's runs.
            if inner_blocks:
                merged_runs: list[IntermediateInlineRun] = []
                for b in inner_blocks:
                    merged_runs.extend(b.runs)
                items.append(
                    IntermediateBlock(
                        type="listItem",
                        runs=merged_runs,
                        meta={"checked": checked} if task else {},
                    )
                )
            else:
                items.append(IntermediateBlock(type="listItem", runs=[]))
            i += 1  # past list_item_close
            continue
        i += 1
    return items, i


def _walk_table(tokens: list[Any], start: int) -> tuple[IntermediateBlock, int]:
    """Walk a table_open .. table_close span and emit a table block."""
    i = start + 1
    rows: list[list[IntermediateBlock]] = []
    current_row: list[IntermediateBlock] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "table_close":
            if current_row:
                rows.append(current_row)
            i += 1
            break
        if tok.type == "tr_open":
            current_row = []
            i += 1
            continue
        if tok.type == "tr_close":
            rows.append(current_row)
            current_row = []
            i += 1
            continue
        if tok.type in ("th_open", "td_open"):
            # th / td span contains an inline token. Read it.
            inline_tok = tokens[i + 1]
            assert inline_tok.type == "inline", f"expected inline, got {inline_tok.type}"
            runs = _runs_from_inline(inline_tok.children or [])
            text = "".join(r.text for r in runs).strip()
            current_row.append(
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text=text)] if text else [],
                )
            )
            i += 3
            continue
        i += 1
    if not rows:
        return IntermediateBlock(type="table", attrs={"rows": 0, "cols": 0, "cells": []}), i
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append(IntermediateBlock(type="paragraph", runs=[]))
    flat: list[IntermediateBlock] = [c for row in rows for c in row]
    return (
        IntermediateBlock(
            type="table",
            attrs={
                "rows": len(rows),
                "cols": n_cols,
                "cells": [[c.id for c in row] for row in rows],
            },
            children=flat,
        ),
        i,
    )


# ---------------------------------------------------------------------------
# Inline tokens -> runs
# ---------------------------------------------------------------------------


def _runs_from_inline(children: list[Any]) -> list[IntermediateInlineRun]:
    """Convert markdown-it inline children to a list of runs.

    markdown-it's inline children are themselves a token stream:
    text nodes, softbreak / hardbreak, code, em_open / em_close,
    strong_open / strong_close, link_open / link_close, image,
    s_open / s_close (strikethrough), html_inline, etc.

    We walk the tree and produce a flat list of runs where each
    contiguous text node with the same annotation set becomes
    one run.
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

    for tok in children:
        t = tok.type
        if t == "text":
            text_parts.append(tok.content)
        elif t == "softbreak":
            text_parts.append("\n")
        elif t == "hardbreak":
            text_parts.append("\n")
        elif t == "code_inline":
            # Inline code: flush any pending text, then push a
            # run with the code annotation.
            flush()
            out.append(
                IntermediateInlineRun(
                    text=tok.content,
                    annotations=["code"],
                )
            )
        elif t == "em_open":
            flush()
            annotations.append("italic")
        elif t == "strong_open":
            flush()
            annotations.append("bold")
        elif t == "s_open":
            flush()
            annotations.append("strikethrough")
        elif t == "link_open":
            flush()
            href = tok.attrGet("href") or ""
            annotations.append({"link": href})
        elif t == "image":
            # Image is its own thing; v1 surfaces as a paragraph
            # with the alt text. v2 will surface as a real image
            # block.
            flush()
            alt = tok.content or ""
            src = tok.attrGet("src") or ""
            out.append(
                IntermediateInlineRun(
                    text=alt,
                    annotations=[{"link": src}] if src else [],
                )
            )
        elif t.endswith("_close"):
            # Pop the last matching annotation. The order in
            # markdown-it is well-defined (LIFO). Important:
            # flush BEFORE popping, so the just-ended span's
            # text is written with the annotation still
            # applied. After the pop, any subsequent text
            # (e.g. the "." after a link) starts a fresh run.
            tag = t[:-6]  # strip "_close"
            mapping = {
                "em": "italic",
                "strong": "bold",
                "s": "strikethrough",
                "link": "link",
            }
            target = mapping.get(tag)
            if target is not None and annotations:
                flush()
                if target == "link":
                    for j in range(len(annotations) - 1, -1, -1):
                        if isinstance(annotations[j], dict) and "link" in annotations[j]:
                            annotations.pop(j)
                            break
                else:
                    for j in range(len(annotations) - 1, -1, -1):
                        if annotations[j] == target:
                            annotations.pop(j)
                            break
        else:
            # html_inline and unknown types: ignore for v1
            pass
    flush()
    return out

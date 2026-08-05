"""HTML builder (exporter).

Converts an ``IntermediateDocument`` to an HTML string. The
HTML is a self-contained document (``<!doctype html>`` ...
``</html>``) suitable for in-app rendering or for export
to a .html file. The output is sanitized at the structural
level (we control the tags emitted; user content is escaped
via the standard library's ``html.escape``).

Coverage matches the importer's coverage: headings, paragraphs,
lists, tables, code blocks, quotes, dividers, images, equations,
callouts, and the full annotation set.
"""

from __future__ import annotations

import html
import logging
from typing import Any

from tools.tessera.importers.intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


def build_html(doc: IntermediateDocument) -> str:
    """Render an IntermediateDocument to a self-contained HTML string."""
    body = "\n".join(_render_block(b) for b in doc.blocks if b is not None)
    title = doc.meta.get("title", "Tessera document")
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(str(title))}</title>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _render_block(ib: IntermediateBlock) -> str:
    if ib.type == "heading":
        level = max(1, min(6, int(ib.attrs.get("level", 1))))
        return f"<h{level}>{_render_runs(ib.runs)}</h{level}>"
    if ib.type == "paragraph":
        return f"<p>{_render_runs(ib.runs)}</p>"
    if ib.type == "list":
        return _render_list(ib)
    if ib.type == "table":
        return _render_table(ib)
    if ib.type == "codeBlock":
        lang = html.escape(str(ib.attrs.get("language") or ""))
        text = html.escape("".join(r.text for r in ib.runs))
        return f'<pre><code class="language-{lang}">{text}</code></pre>'
    if ib.type == "quote":
        cite = ib.attrs.get("cite")
        cite_attr = f' cite="{html.escape(cite)}"' if cite else ""
        return f"<blockquote{cite_attr}>{_render_runs(ib.runs)}</blockquote>"
    if ib.type == "divider":
        return "<hr>"
    if ib.type == "image":
        alt = html.escape(str(ib.attrs.get("alt", "")))
        src = html.escape(str(ib.attrs.get("source", "")))
        return f'<img src="{src}" alt="{alt}">'
    if ib.type == "equation":
        latex = html.escape(str(ib.attrs.get("latex", "")))
        return f'<div class="equation" data-latex="{latex}">{latex}</div>'
    if ib.type == "callout":
        emoji = html.escape(str(ib.attrs.get("emoji", "")))
        text = _render_runs(ib.runs)
        return f'<div class="callout"><span class="emoji">{emoji}</span><span>{text}</span></div>'
    if ib.type == "listItem":
        return f"<li>{_render_runs(ib.runs)}</li>"
    return _render_runs(ib.runs)


def _render_list(ib: IntermediateBlock) -> str:
    style = ib.attrs.get("style", "unordered")
    children = [c for c in ib.children if hasattr(c, "type")]
    if style == "ordered":
        tag = "ol"
    else:
        tag = "ul"
    items = "\n".join(f"  <li>{_render_runs(c.runs)}</li>" for c in children)
    return f"<{tag}>\n{items}\n</{tag}>"


def _render_table(ib: IntermediateBlock) -> str:
    n_cols = int(ib.attrs.get("cols", 0))
    children = [c for c in ib.children if hasattr(c, "type")]
    grid: list[list[str]] = []
    row: list[str] = []
    for c in children:
        row.append(_render_runs(c.runs))
        if len(row) == n_cols:
            grid.append(row)
            row = []
    if not grid:
        return ""
    out: list[str] = ["<table>", "<tbody>"]
    for r in grid:
        out.append("  <tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


def _render_runs(runs: list[IntermediateInlineRun]) -> str:
    return "".join(_render_run(r) for r in runs)


def _render_run(r: IntermediateInlineRun) -> str:
    text = html.escape(r.text)
    if not text:
        return ""
    # Apply annotations outermost first, but with the convention
    # that link goes innermost (its text is the link text, not
    # the URL). We render bold/italic/etc. as outer wrappers.
    for ann in r.annotations:
        if isinstance(ann, str):
            if ann == "bold":
                text = f"<strong>{text}</strong>"
            elif ann == "italic":
                text = f"<em>{text}</em>"
            elif ann == "code":
                text = f"<code>{text}</code>"
            elif ann == "strikethrough":
                text = f"<s>{text}</s>"
            elif ann == "underline":
                text = f"<u>{text}</u>"
            elif ann == "subscript":
                text = f"<sub>{text}</sub>"
            elif ann == "superscript":
                text = f"<sup>{text}</sup>"
        elif isinstance(ann, dict):
            if "link" in ann:
                href = html.escape(str(ann["link"]), quote=True)
                text = f'<a href="{href}">{text}</a>'
            elif "color" in ann:
                color = html.escape(str(ann["color"]))
                text = f'<span style="color:{color}">{text}</span>'
    return text

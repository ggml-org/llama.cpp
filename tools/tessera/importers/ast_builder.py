"""Build a ``DocumentAST`` from an ``IntermediateDocument``.

The intermediate shape (see ``intermediate.py``) is the unit-test
boundary between parsers and the AST layer. This module's job
is to take one ``IntermediateDocument`` (or a list of them, for
XLSX/PPTX/MBOX which produce multi-document outputs) and turn
it into one or more ``DocumentAST``s.

The mapping is mostly mechanical:

* Each ``IntermediateBlock`` becomes a ``Block`` in the AST.
* Container blocks (``list``, ``table``, ``toggle``) carry
  their child blocks as ``IntermediateBlock`` references in
  the intermediate; the builder walks the children and adds
  them to the AST under the parent's id.
* Inline runs are translated as-is. The intermediate's
  ``IntermediateInlineRun`` is structurally identical to the
  AST's ``InlineRun`` shape, so the conversion is just a
  type-level rename.

The builder also adds a few useful derived things:

* A document title (from the first heading, or the source
  filename when the document has no heading).
* An ``attributes`` enrichment for some block types (e.g. an
  ``equation`` block's ``latex`` field is preserved as-is).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import ast_schema
from .ast_schema import Block, DocumentAST, json_value
from .intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


def build(
    intermediate: IntermediateDocument,
    *,
    title: str | None = None,
) -> DocumentAST:
    """Build a single ``DocumentAST`` from an ``IntermediateDocument``.

    The intermediate is fully consumed; the AST is self-contained
    and ready to be serialised to JSON. The optional ``title``
    overrides the title inferred from the first heading; the
    receipt emitter passes the original filename as a fallback.
    """
    ast = DocumentAST.empty()
    _populate(ast, intermediate)
    if title is not None:
        ast.meta["title"] = title
    else:
        # Infer a title from the first heading, if any.
        for ib in intermediate.blocks:
            if ib.type == "heading":
                ast.meta["title"] = "".join(r.text for r in ib.runs).strip()
                break
    return ast


def build_many(
    intermediates: list[IntermediateDocument],
    *,
    base_title: str | None = None,
    override_title: bool = True,
) -> list[DocumentAST]:
    """Build one ``DocumentAST`` per ``IntermediateDocument``.

    Used by parsers that produce multi-document outputs
    (XLSX, PPTX, MBOX). The resulting list is what the
    pipeline persists; each AST becomes its own
    ``graph_entity``.

    `override_title` (default True): when True, ``base_title``
    is the title for every AST, regardless of the
    intermediate's first heading. When False, ``base_title``
    is only used as a suffix when the intermediate has no
    inferred title (no leading heading). The False mode is
    what the pipeline uses for EML / DOCX / etc. where the
    document's content already includes a natural title
    (the email subject, the first heading, etc.).
    """
    out: list[DocumentAST] = []
    for i, inter in enumerate(intermediates):
        if override_title:
            title = base_title
            if base_title is not None and len(intermediates) > 1:
                # Disambiguate by sheet / slide / message index.
                sheet = inter.meta.get("sheet_name")
                slide = inter.meta.get("slide_index")
                subject = inter.meta.get("subject")
                if sheet:
                    title = f"{base_title} — {sheet}"
                elif slide is not None:
                    title = f"{base_title} — slide {slide}"
                elif subject:
                    title = f"{base_title} — {subject}"
                else:
                    title = f"{base_title} — {i + 1}"
        else:
            # Don't override: the AST builder will infer the
            # title from the first heading (or fall back to
            # the file stem). For multi-document intermediates
            # we still disambiguate by sheet / slide / subject
            # when the AST's inferred title is empty.
            title = None
            if len(intermediates) > 1:
                sheet = inter.meta.get("sheet_name")
                slide = inter.meta.get("slide_index")
                subject = inter.meta.get("subject")
                if sheet and base_title:
                    title = f"{base_title} — {sheet}"
                elif slide is not None and base_title:
                    title = f"{base_title} — slide {slide}"
                elif subject and base_title:
                    title = f"{base_title} — {subject}"
        ast = build(inter, title=title)
        out.append(ast)
    return out


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------


def _populate(ast: DocumentAST, intermediate: IntermediateDocument) -> None:
    """Add every block from `intermediate` to `ast`.

    Top-level blocks go in ``rootChildren``. Container blocks
    (list, table, toggle) have their child blocks attached via
    ``ast.attach``. We iterate the top-level blocks in order and
    recurse into containers; the recursion adds children to the
    blocks map and to the parent.
    """
    for ib in intermediate.blocks:
        block = _to_block(ib)
        if ib.type in ("list", "table", "toggle"):
            # Container block: register it, then add its children.
            if block.id in ast.blocks:
                # Defensive: never duplicate IDs.
                continue
            ast.blocks[block.id] = block
            ast.rootChildren.append(block.id)
            for child_ib in _child_blocks(ib):
                child_block = _to_block(child_ib)
                child_block.parentID = block.id
                ast.blocks[child_block.id] = child_block
        else:
            ast.add_root(block)


def _child_blocks(ib: IntermediateBlock) -> list[IntermediateBlock]:
    """Return the child IntermediateBlocks of a container.

    A container's ``children`` is ``list[IntermediateBlock | str]``;
    we filter to the IntermediateBlock instances (the strings
    are stale IDs and aren't used by the builder).
    """
    return [c for c in ib.children if isinstance(c, IntermediateBlock)]


def _to_block(ib: IntermediateBlock) -> Block:
    """Convert one ``IntermediateBlock`` to an AST ``Block``."""
    return Block(
        id=ib.id,
        type=ib.type,
        attributes=dict(ib.attrs),
        content=[_to_run_dict(r) for r in ib.runs],
        children=[c.id for c in _child_blocks(ib)],  # children are block ids
        parentID=ib.parentID,
    )


def _to_run_dict(run: IntermediateInlineRun) -> dict[str, Any]:
    """Build the JSON-shape dict for an inline run.

    Equivalent to ``run_to_json`` in ``ast_schema`` but works on
    the intermediate's run type.
    """
    return {
        "text": run.text,
        "annotations": [
            ast_schema.annotation_to_json(a) for a in run.annotations
        ],
    }

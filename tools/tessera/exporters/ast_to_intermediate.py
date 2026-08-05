"""AST -> intermediate.

The exporter pipeline mirrors the importer. The exporters
read a ``DocumentAST`` (from the data layer) and turn it into
an ``IntermediateDocument``; the per-format builders consume
the intermediate and write a file.

This module is the AST -> intermediate step. It's the inverse
of ``importers/ast_builder.py`` (which goes intermediate -> AST).

Why a shared intermediate shape?

* The exporters for DOCX / PPTX / XLSX / HTML / Markdown / EML
  all consume the same ``IntermediateDocument`` shape. The AST
  is the canonical form, the intermediate is the form the
  builders (and python-docx / openpyxl / etc.) want.
* The round-trip is testable: build a fixture AST, run the
  intermediate step, run the builder, parse the file back,
  compare.

Lossy round-trip note:

The AST is the canonical form; the intermediate + builder
combination is the lossy translation to a specific format.
DOCX has no ``equation`` type; the Markdown builder
promotes equations to inline ``$...$`` text. HTML round-trips
nearly losslessly. The receipt chain is built on the AST, so
re-importing an exported file gives you a different AST
(UUIDs, slight format drift) but the same content hash modulo
that drift. v2 will track the drift explicitly.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.tessera.importers.ast_schema import Block, DocumentAST
from tools.tessera.importers.intermediate import (
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)

log = logging.getLogger(__name__)


def ast_to_intermediate(ast: DocumentAST) -> IntermediateDocument:
    """Convert a ``DocumentAST`` to an ``IntermediateDocument``.

    Top-level blocks are emitted in ``rootChildren`` order.
    Container blocks (list, table) carry their child blocks in
    ``children``; the intermediate's children list holds
    ``IntermediateBlock`` instances for these (matching the
    importer's intermediate shape, so the builders don't
    care which side produced it).
    """
    blocks: list[IntermediateBlock] = []
    for cid in ast.rootChildren:
        b = ast.blocks.get(cid)
        if b is None:
            continue
        blocks.append(_to_intermediate(b, ast))
    return IntermediateDocument(blocks=blocks, meta=dict(ast.meta))


def _to_intermediate(b: Block, ast: DocumentAST) -> IntermediateBlock:
    """Convert one ``Block`` to an ``IntermediateBlock``."""
    children: list[Any] = []
    for cid in b.children:
        child = ast.blocks.get(cid)
        if child is not None:
            children.append(_to_intermediate(child, ast))
    return IntermediateBlock(
        id=b.id,
        type=b.type,
        attrs=dict(b.attributes),
        runs=[IntermediateInlineRun(text=r.get("text", ""), annotations=list(r.get("annotations", []))) for r in b.content],
        children=children,
        parentID=b.parentID,
    )

"""Intermediate JSON shape used between parsers and the AST builder.

The parsers (DOCX, XLSX, PPTX, PDF, email, HTML, Markdown, Pandoc)
all produce an ``IntermediateDocument``: a flat list of
``IntermediateBlock``s plus a parallel list of ``IntermediateInlineRun``s.

The intermediate shape is decoupled from the AST module
(``ast_schema.py``) and from the third-party parser libraries.
Reasons:

1. **The intermediate is the unit test boundary.** A test that
   wants to verify "DOCX tables become table blocks" can inspect
   the intermediate without dragging python-docx in.
2. **Multiple parsers can target one intermediate.** EML and
   MBOX both produce the same intermediate (a list of
   ``email`` blocks) but with different metadata.
3. **The intermediate is a stable contract between Python and
   Swift.** If we ever build a Swift-native importer for any
   format, it can produce the same intermediate and reuse the
   same AST builder.

The intermediate uses ``uuid4`` strings for block IDs (same as
the AST). When the AST builder receives an intermediate, it
either re-uses the IDs (when present) or mints fresh ones.

Why dataclasses with ``id`` defaulted to a uuid4:

* The intermediate is a planning shape, not the canonical
  form. The canonical form is the AST. The intermediate is
  free to use plain strings and dicts and we still get
  type-safety from the dataclasses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class IntermediateInlineRun:
    """One inline run in the intermediate shape.

    Mirrors ``InlineRun`` in the AST: a string with a list of
    annotation tags. The tags use the same shape the AST expects
    (bare string for no-arg cases, single-key dict for
    associated-value cases).
    """

    text: str
    annotations: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "annotations": list(self.annotations)}


@dataclass
class IntermediateBlock:
    """One block in the intermediate shape.

    The block carries:

    * ``id``: a uuid4 string. Reused by the AST builder; the
      builder also accepts a missing id and mints one.
    * ``type``: the block type (e.g. ``"paragraph"``,
      ``"heading"``, ``"list"``, ``"table"``).
    * ``attrs``: type-specific attributes (heading level,
      list style, table rows/cols, image source, etc.).
    * ``runs``: the inline runs for leaf blocks. Empty for
      container blocks.
    * ``children``: child block IDs for container blocks
      (list, table, toggle). Empty for leaf blocks.
    * ``meta``: optional metadata the AST builder might use
      (e.g. email headers, image alt text, code language).
    """

    type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    runs: list[IntermediateInlineRun] = field(default_factory=list)
    children: list["IntermediateBlock | str"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    parentID: str | None = None

    def child_blocks(self) -> list["IntermediateBlock"]:
        """Return the children that are IntermediateBlock instances (not IDs)."""
        return [c for c in self.children if isinstance(c, IntermediateBlock)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "attrs": dict(self.attrs),
            "runs": [r.to_dict() for r in self.runs],
            "children": [
                c.id if isinstance(c, IntermediateBlock) else c for c in self.children
            ],
            "meta": dict(self.meta),
        }


@dataclass
class IntermediateDocument:
    """One document's worth of intermediate state.

    The flat list of blocks is the source of truth; the AST
    builder is responsible for promoting the in-list
    IntermediateBlock references into the tree shape the AST
    expects. Most parsers produce a single IntermediateDocument
    per file. EML / MBOX produce one per message and the parser
    returns a list.
    """

    blocks: list[IntermediateBlock] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "meta": dict(self.meta),
        }

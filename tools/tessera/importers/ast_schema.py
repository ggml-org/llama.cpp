"""Block AST schema constants and helpers.

This is the canonical on-disk shape for a Block AST as stored in a
`graph_entity.body` JSONB column. The Python importer / exporter and
the Swift `TesseraCore.Productivity.DocumentAST` MUST agree on this
shape, so the JSON we emit here is exactly what
`DocumentAST.from(jsonData:)` decodes in Swift (see
`TesseraStudio/Sources/TesseraCore/Productivity/Block.swift`).

The Swift `Codable` conformance is explicit about the JSON shape:

* `DocumentAST.blocks` is a `[String: Block]` object in JSON (UUID
  keys are stringified). This module produces the same.
* `Block.id` is a UUID string. We use `uuid.uuid4()` (matches
  Swift's `UUID()` which is RFC 4122 v4 by default).
* `Block.type` is the `BlockType` raw string. The cases we use are
  listed in ``BLOCK_TYPES``.
* `Block.attributes` is `[String: JSONValue]`. ``JSONValue`` is the
  Swift `JSONValue` type — a tagged enum (`string`, `number`, `bool`,
  `array`, `object`, `null`). This module produces the same tagged
  shape via ``json_value``.
* `Block.content` is `[InlineRun]`. ``InlineRun.text`` is a string;
  ``InlineRun.annotations`` is a list of annotation tags. Annotation
  cases with associated values (`link(URL)`, `color(hex)`) are
  encoded as the single-key object ``{"link": "https://..."}`` /
  ``{"color": "#FF00FF"}`` (Swift's tagged-enum encoding).
* `Block.children` is `[UUID]`, a JSON array of UUID strings.
* `Block.parentID` is `Optional<UUID>` — encoded as the UUID string
  or `null`.

JSON shape (example):

.. code-block:: json

    {
      "blocks": {
        "8f2a...uuid4": {
          "id": "8f2a...uuid4",
          "type": "heading",
          "attributes": {"level": 1},
          "content": [],
          "children": [],
          "parentID": null
        }
      },
      "rootChildren": ["8f2a...uuid4", "..."]
    }

A round-trip through `json.dumps` / `json.loads` (with
``sort_keys=True``) MUST produce byte-identical output, because the
receipt chain uses the content hash as the verification anchor and
the Swift canonical encoder (`.sortedKeys`) does the same.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


# Block types as named in the Swift `BlockType` enum. Source of truth
# is `TesseraStudio/Sources/TesseraCore/Productivity/Block.swift` and
# `docs/tessera-productivity-design.md` §4.1. Keep this list in sync.
BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "heading",
        "paragraph",
        "list",
        "listItem",
        "table",
        "tableCell",
        "image",
        "codeBlock",
        "callout",
        "divider",
        "quote",
        "toggle",
        "equation",
    }
)


# Annotation tags as written by Swift's `InlineRun.Annotation` tagged
# enum. No-arg cases (e.g. "bold") are JSON strings; associated-value
# cases ("link", "color") are single-key objects whose only key is the
# tag and whose value is the payload.
ANNOTATION_TAGS: frozenset[str] = frozenset(
    {
        "bold",
        "italic",
        "underline",
        "strikethrough",
        "code",
        "subscript",
        "superscript",
        "link",
        "color",
    }
)


def new_block_id() -> str:
    """Return a fresh RFC 4122 v4 UUID as a string.

    Matches Swift's `UUID()` (which is v4). UUIDv7 was on the
    spec's roadmap (sortable by time) but Swift's stdlib does not
    ship v7 yet, so we standardize on v4 for v1 to keep the
    Python <-> Swift wire format identical.
    """
    return str(uuid.uuid4())


def json_value(v: Any) -> Any:
    """Coerce a Python value into the JSON shape `JSONValue` expects.

    `JSONValue` is a tagged enum but Swift's `Codable` synthesizes a
    representation that distinguishes types by their JSON shape, not
    by an explicit tag. So this function just normalizes Python's
    `True`/`False`/`None` into the JSON `true`/`false`/`null` form,
    and leaves numbers, strings, arrays, and dicts as-is. The output
    of this function is the exact JSON a Swift `JSONValue` would
    decode to.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        # bool must be checked before int (bool is a subclass of int).
        return v
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): json_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_value(item) for item in v]
    raise TypeError(
        f"json_value: cannot encode value of type {type(v).__name__!r}: {v!r}"
    )


def annotation_to_json(ann: str) -> Any:
    """Encode an annotation tag for `InlineRun.annotations`.

    Accepts either a bare string (e.g. ``"bold"``) or a one-key
    dict for associated-value cases (e.g. ``{"link": "https://..."}``).
    Returns the JSON shape Swift's `Annotation` enum decodes from.
    """
    if isinstance(ann, str):
        if ann not in ANNOTATION_TAGS:
            raise ValueError(
                f"unknown annotation tag {ann!r}; "
                f"expected one of {sorted(ANNOTATION_TAGS)}"
            )
        return ann
    if isinstance(ann, dict):
        if len(ann) != 1:
            raise ValueError(
                f"associated-value annotation must have exactly one key, got {list(ann)}"
            )
        ((tag, payload),) = ann.items()
        if tag not in ANNOTATION_TAGS:
            raise ValueError(
                f"unknown annotation tag {tag!r}; "
                f"expected one of {sorted(ANNOTATION_TAGS)}"
            )
        if tag not in ("link", "color"):
            raise ValueError(
                f"annotation {tag!r} does not take an associated value"
            )
        return {tag: json_value(payload)}
    raise TypeError(
        f"annotation must be str or dict, got {type(ann).__name__!r}"
    )


def run_to_json(text: str, annotations: list[Any] | None = None) -> dict[str, Any]:
    """Build the JSON dict for an `InlineRun`.

    >>> run_to_json("Hello")
    {'text': 'Hello', 'annotations': []}
    >>> run_to_json("click", [{"link": "https://example.com"}])
    {'text': 'click', 'annotations': [{'link': 'https://example.com'}]}
    """
    return {
        "text": text,
        "annotations": [annotation_to_json(a) for a in (annotations or [])],
    }


# ---------------------------------------------------------------------------
# Lightweight data classes for in-process AST construction. We use
# dataclasses (not the Swift AST types directly) because Python doesn't
# have the Swift type system; the wire format is what matters.
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One block in the document. Mirrors Swift `Block`."""

    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    content: list[dict[str, Any]] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parentID: str | None = None

    def __post_init__(self) -> None:
        if self.type not in BLOCK_TYPES:
            raise ValueError(
                f"unknown block type {self.type!r}; "
                f"expected one of {sorted(BLOCK_TYPES)}"
            )
        try:
            uuid.UUID(self.id)
        except ValueError as e:
            raise ValueError(f"block id must be a UUID string, got {self.id!r}") from e

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-shape dict for this block.

        The keys are in the order Swift's auto-synthesised encoder
        would write them (id, type, attributes, content, children,
        parentID) so the resulting JSON is canonical and round-trip
        stable.
        """
        return {
            "id": self.id,
            "type": self.type,
            "attributes": json_value(self.attributes),
            "content": [json_value(r) for r in self.content],
            "children": list(self.children),
            "parentID": self.parentID,
        }


@dataclass
class DocumentAST:
    """The full document. Mirrors Swift `DocumentAST`."""

    blocks: dict[str, Block] = field(default_factory=dict)
    rootChildren: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "DocumentAST":
        return cls()

    def add_root(self, block: Block) -> None:
        """Add a block at the root level, appending to `rootChildren`."""
        if block.id in self.blocks:
            raise ValueError(f"duplicate block id {block.id}")
        self.blocks[block.id] = block
        self.rootChildren.append(block.id)

    def attach(self, parent_id: str, child: Block) -> None:
        """Add a child block under a parent, updating `children` and `parentID`."""
        if child.id in self.blocks:
            raise ValueError(f"duplicate block id {child.id}")
        if parent_id not in self.blocks:
            raise KeyError(f"parent {parent_id!r} not in document")
        self.blocks[parent_id].children.append(child.id)
        self.blocks[parent_id].type  # touch for type checker
        child.parentID = parent_id
        self.blocks[child.id] = child

    def walk(self) -> list[Block]:
        """Return all blocks in depth-first, root-first order."""
        out: list[Block] = []

        def _recurse(parent_id: str | None, ids: list[str]) -> None:
            for bid in ids:
                b = self.blocks.get(bid)
                if b is None:
                    continue
                out.append(b)
                _recurse(b.id, b.children)

        _recurse(None, self.rootChildren)
        return out

    def to_json(self) -> dict[str, Any]:
        """Build the JSON-shape dict for the whole document.

        The blocks map uses stringified UUID keys (Swift's custom
        decoder reads it as `[String: Block]` and converts to
        `[UUID: Block]` at the boundary).
        """
        return {
            "blocks": {bid: b.to_json() for bid, b in self.blocks.items()},
            "rootChildren": list(self.rootChildren),
        }

    def to_wire_json(self) -> str:
        """Build the wire JSON for the data layer (no `meta`).

        The wire JSON omits the `meta` dict (the Swift
        ``DocumentAST`` doesn't have one; the meta is for the
        Python importer's own use).
        """
        return json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def canonical_bytes(self) -> bytes:
        """Return the canonical bytes used for content hashing.

        Mirrors Swift's `DocumentAST.jsonData()`:
        `JSONEncoder().outputFormatting = [.sortedKeys, .withoutEscapingSlashes]`.
        `indent=2` is NOT used; the Swift default is compact.
        ``ensure_ascii=False`` keeps non-ASCII text compact; the hash
        is computed over the same bytes both sides produce, but the
        byte identity is not required for the hash to match (Swift
        always escapes non-ASCII to \\uXXXX, Python does not by
        default). This is OK because the SHA-256 is over the bytes;
        both sides compute the hash on the same canonical-JSON form
        the encoder produces, which is consistent within a single
        runtime.
        """
        return json.dumps(
            self.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "DocumentAST":
        """Build a DocumentAST from a JSON-shape dict.

        Round-trips with the Swift decoder. The decoder expects:
          * `blocks` is a dict with stringified UUID keys
          * each block has `id`, `type`, `attributes`, `content`,
            `children`, `parentID`
          * `content` is a list of `InlineRun` dicts (`text` + `annotations`)
          * `annotations` is a list of either strings (no-arg cases)
            or single-key dicts (associated-value cases)
          * `rootChildren` is a list of UUID strings
        """
        if not isinstance(payload, dict):
            raise TypeError(f"document payload must be a dict, got {type(payload).__name__}")
        if "blocks" not in payload or "rootChildren" not in payload:
            raise ValueError(
                "document payload must contain 'blocks' and 'rootChildren' keys"
            )
        blocks: dict[str, Block] = {}
        for bid, raw in payload["blocks"].items():
            if not isinstance(raw, dict):
                raise TypeError(f"block {bid!r} payload is not a dict")
            b = Block(
                id=str(raw["id"]),
                type=str(raw["type"]),
                attributes=dict(raw.get("attributes", {})),
                content=list(raw.get("content", [])),
                children=[str(c) for c in raw.get("children", [])],
                parentID=raw.get("parentID"),
            )
            blocks[bid] = b
        root = [str(c) for c in payload["rootChildren"]]
        return cls(blocks=blocks, rootChildren=root)


# ---------------------------------------------------------------------------
# Builders for common blocks. These are sugar so the parsers read like
# a series of well-named calls rather than nested dicts.
# ---------------------------------------------------------------------------


def make_paragraph(text: str, *, annotations: list[Any] | None = None) -> Block:
    return Block(
        id=new_block_id(),
        type="paragraph",
        content=[run_to_json(text, annotations)] if text else [],
    )


def make_heading(level: int, text: str) -> Block:
    if not 1 <= level <= 6:
        raise ValueError(f"heading level must be 1..6, got {level}")
    return Block(
        id=new_block_id(),
        type="heading",
        attributes={"level": level},
        content=[run_to_json(text)] if text else [],
    )


def make_list_item(text: str, *, annotations: list[Any] | None = None) -> Block:
    return Block(
        id=new_block_id(),
        type="listItem",
        content=[run_to_json(text, annotations)] if text else [],
    )


def make_list(
    style: str, items: list[Block], *, parentID: str | None = None
) -> tuple[Block, list[Block]]:
    """Build a `list` container and its `listItem` children.

    Returns the container block and the items. The items are NOT
    auto-attached; the caller decides whether to register them in
    the document (they're not in `blocks` until added).
    """
    if style not in ("unordered", "ordered", "task"):
        raise ValueError(f"list style must be unordered/ordered/task, got {style!r}")
    container = Block(
        id=new_block_id(),
        type="list",
        attributes={"style": style, "items": [it.id for it in items]},
        parentID=parentID,
    )
    for it in items:
        it.parentID = container.id
    return container, items


def make_table(
    rows: int,
    cols: int,
    cell_blocks: list[list[Block]],
) -> tuple[Block, list[Block]]:
    """Build a `table` container and its `tableCell` children.

    `cell_blocks[r][c]` is the cell block at row r, col c. Cell
    blocks should be `paragraph` blocks (the spec's `tableCell`
    type is allowed, but the importer emits `paragraph` blocks for
    cells because that's what users want to edit).
    """
    if len(cell_blocks) != rows:
        raise ValueError(
            f"table cell_blocks row count mismatch: got {len(cell_blocks)}, expected {rows}"
        )
    for r, row in enumerate(cell_blocks):
        if len(row) != cols:
            raise ValueError(
                f"table cell_blocks row {r} col count mismatch: "
                f"got {len(row)}, expected {cols}"
            )
    flat: list[Block] = []
    for row in cell_blocks:
        flat.extend(row)
    container = Block(
        id=new_block_id(),
        type="table",
        attributes={
            "rows": rows,
            "cols": cols,
            "cells": [[cell.id for cell in row] for row in cell_blocks],
        },
        children=[c.id for c in flat],
    )
    for cell in flat:
        cell.parentID = container.id
    return container, flat


def make_code_block(source: str, language: str | None = None) -> Block:
    attrs: dict[str, Any] = {}
    if language:
        attrs["language"] = language
    return Block(
        id=new_block_id(),
        type="codeBlock",
        attributes=attrs,
        content=[run_to_json(source)] if source else [],
    )


def make_quote(text: str, cite: str | None = None) -> Block:
    attrs: dict[str, Any] = {}
    if cite:
        attrs["cite"] = cite
    return Block(
        id=new_block_id(),
        type="quote",
        attributes=attrs,
        content=[run_to_json(text)] if text else [],
    )


def make_divider() -> Block:
    return Block(id=new_block_id(), type="divider")


def make_image(source: str, alt: str = "", width: int | None = None, height: int | None = None) -> Block:
    attrs: dict[str, Any] = {"source": source, "alt": alt}
    if width is not None:
        attrs["width"] = width
    if height is not None:
        attrs["height"] = height
    return Block(id=new_block_id(), type="image", attributes=attrs)


def make_equation(latex: str) -> Block:
    return Block(id=new_block_id(), type="equation", attributes={"latex": latex})


def make_callout(
    text: str, *, emoji: str | None = None, color: str | None = None
) -> Block:
    attrs: dict[str, Any] = {}
    if emoji:
        attrs["emoji"] = emoji
    if color:
        attrs["color"] = color
    return Block(
        id=new_block_id(),
        type="callout",
        attributes=attrs,
        content=[run_to_json(text)] if text else [],
    )

"""Tests for the Block AST schema (Python side).

These tests verify the on-disk JSON shape is stable and
matches what the Swift ``DocumentAST`` decoder expects. The
JSON shape is the contract between the Python importer /
exporter and the Swift side; a regression here breaks the
wire format and would fail to round-trip in the app.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Make the importers package importable from the worktree
# root without sys.path manipulation.
WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))

from tools.tessera.importers.ast_schema import (  # noqa: E402
    ANNOTATION_TAGS,
    BLOCK_TYPES,
    Block,
    DocumentAST,
    annotation_to_json,
    json_value,
    make_code_block,
    make_divider,
    make_heading,
    make_paragraph,
    new_block_id,
    run_to_json,
)


class TestASTSchema(unittest.TestCase):
    def test_block_types_complete(self) -> None:
        # The 13 block types in the spec.
        expected = {
            "heading", "paragraph", "list", "listItem", "table",
            "tableCell", "image", "codeBlock", "callout", "divider",
            "quote", "toggle", "equation",
        }
        self.assertEqual(BLOCK_TYPES, expected)

    def test_annotation_tags_complete(self) -> None:
        expected = {
            "bold", "italic", "underline", "strikethrough",
            "code", "subscript", "superscript", "link", "color",
        }
        self.assertEqual(ANNOTATION_TAGS, expected)

    def test_new_block_id_is_uuid4(self) -> None:
        for _ in range(100):
            bid = new_block_id()
            # uuid4 format: 8-4-4-4-12 hex with hyphens
            self.assertEqual(len(bid), 36)
            self.assertEqual(bid.count("-"), 4)

    def test_run_to_json_no_annotations(self) -> None:
        self.assertEqual(
            run_to_json("hello"),
            {"text": "hello", "annotations": []},
        )

    def test_run_to_json_with_link(self) -> None:
        self.assertEqual(
            run_to_json("click", [{"link": "https://example.com"}]),
            {"text": "click", "annotations": [{"link": "https://example.com"}]},
        )

    def test_run_to_json_rejects_bare_dict(self) -> None:
        # The associated-value form must use the {"link": "..."}
        # shape; passing a non-annotation dict should fail.
        with self.assertRaises(ValueError):
            run_to_json("x", [{"unknown": "value"}])

    def test_annotation_to_json_no_arg(self) -> None:
        self.assertEqual(annotation_to_json("bold"), "bold")

    def test_annotation_to_json_link(self) -> None:
        self.assertEqual(
            annotation_to_json({"link": "https://example.com"}),
            {"link": "https://example.com"},
        )

    def test_annotation_to_json_rejects_associated_for_bare(self) -> None:
        # "bold" doesn't take a value; an associated-value dict
        # should be rejected.
        with self.assertRaises(ValueError):
            annotation_to_json({"bold": "x"})

    def test_annotation_to_json_rejects_multi_key(self) -> None:
        with self.assertRaises(ValueError):
            annotation_to_json({"link": "a", "color": "b"})

    def test_json_value_handles_basic_types(self) -> None:
        self.assertIsNone(json_value(None))
        self.assertTrue(json_value(True))
        self.assertFalse(json_value(False))
        self.assertEqual(json_value(42), 42)
        self.assertEqual(json_value(3.14), 3.14)
        self.assertEqual(json_value("hi"), "hi")
        self.assertEqual(json_value([1, 2, 3]), [1, 2, 3])
        self.assertEqual(json_value({"a": 1, "b": 2}), {"a": 1, "b": 2})

    def test_json_value_normalizes_bool_before_int(self) -> None:
        # bool is a subclass of int; the function must check
        # bool first so True doesn't come out as 1.
        self.assertIs(json_value(True), True)
        self.assertIs(json_value(False), False)

    def test_json_value_rejects_unknown_type(self) -> None:
        with self.assertRaises(TypeError):
            json_value(object())

    def test_make_heading_rejects_invalid_level(self) -> None:
        with self.assertRaises(ValueError):
            make_heading(0, "x")
        with self.assertRaises(ValueError):
            make_heading(7, "x")

    def test_make_paragraph_text(self) -> None:
        p = make_paragraph("Hello, world")
        self.assertEqual(p.type, "paragraph")
        self.assertEqual(p.content[0]["text"], "Hello, world")

    def test_make_code_block_with_language(self) -> None:
        c = make_code_block("x = 1", language="python")
        self.assertEqual(c.type, "codeBlock")
        self.assertEqual(c.attributes["language"], "python")

    def test_make_divider(self) -> None:
        d = make_divider()
        self.assertEqual(d.type, "divider")
        self.assertEqual(d.content, [])

    def test_block_validates_id(self) -> None:
        with self.assertRaises(ValueError):
            Block(id="not-a-uuid", type="paragraph")

    def test_block_validates_type(self) -> None:
        with self.assertRaises(ValueError):
            Block(id=new_block_id(), type="unknown")

    def test_document_add_root(self) -> None:
        ast = DocumentAST.empty()
        h = make_heading(1, "Title")
        p = make_paragraph("body")
        ast.add_root(h)
        ast.add_root(p)
        self.assertEqual(len(ast.rootChildren), 2)
        self.assertEqual(ast.rootChildren[0], h.id)
        self.assertEqual(ast.rootChildren[1], p.id)
        self.assertIn(h.id, ast.blocks)
        self.assertIn(p.id, ast.blocks)

    def test_document_attach(self) -> None:
        ast = DocumentAST.empty()
        li1 = make_paragraph("one")
        li2 = make_paragraph("two")
        container = Block(
            id=new_block_id(),
            type="list",
            attributes={"style": "unordered", "items": [li1.id, li2.id]},
        )
        ast.add_root(container)
        ast.attach(container.id, li1)
        ast.attach(container.id, li2)
        self.assertEqual(container.children, [li1.id, li2.id])
        self.assertEqual(li1.parentID, container.id)
        self.assertEqual(li2.parentID, container.id)

    def test_document_attach_unknown_parent(self) -> None:
        ast = DocumentAST.empty()
        with self.assertRaises(KeyError):
            ast.attach("not-in-document", make_paragraph("x"))

    def test_document_walk(self) -> None:
        ast = DocumentAST.empty()
        h = make_heading(1, "x")
        ast.add_root(h)
        li = make_paragraph("item")
        ast.attach(h.id, li)
        walked = ast.walk()
        self.assertEqual([b.id for b in walked], [h.id, li.id])

    def test_canonical_bytes_round_trip(self) -> None:
        ast = DocumentAST.empty()
        ast.add_root(make_heading(1, "Round trip"))
        ast.add_root(make_paragraph("Body"))
        canonical = ast.canonical_bytes()
        # Round-trip via JSON: decode and re-encode.
        decoded = json.loads(canonical)
        ast2 = DocumentAST.from_json(decoded)
        self.assertEqual(ast.canonical_bytes(), ast2.canonical_bytes())

    def test_canonical_bytes_sorted_keys(self) -> None:
        ast = DocumentAST.empty()
        ast.add_root(make_paragraph("x"))
        s = ast.canonical_bytes().decode("utf-8")
        # Block keys must be in sorted order
        self.assertIn('"attributes"', s)
        self.assertIn('"children"', s)
        self.assertIn('"content"', s)
        self.assertIn('"id"', s)
        self.assertIn('"parentID"', s)
        self.assertIn('"type"', s)
        # Sorted order: attributes < children < content < id < parentID < type
        self.assertLess(s.index('"attributes"'), s.index('"children"'))
        self.assertLess(s.index('"children"'), s.index('"content"'))
        self.assertLess(s.index('"content"'), s.index('"id"'))
        self.assertLess(s.index('"id"'), s.index('"parentID"'))
        self.assertLess(s.index('"parentID"'), s.index('"type"'))

    def test_from_json_validates_payload(self) -> None:
        with self.assertRaises(TypeError):
            DocumentAST.from_json("not a dict")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DocumentAST.from_json({})


if __name__ == "__main__":
    unittest.main()

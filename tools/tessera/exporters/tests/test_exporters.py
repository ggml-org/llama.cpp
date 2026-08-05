"""Tests for the exporter pipeline.

The exporter reads an AST (from the data layer) and produces
a file. The tests use the dry-run client so the AST is
synthesized from a label, and the test asserts the output
file is well-formed for the format.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
import zipfile
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))

from tools.tessera.exporters.builders import (  # noqa: E402
    docx as docx_builder,
    email as email_builder,
    html as html_builder,
    markdown as md_builder,
)
from tools.tessera.exporters.pipeline import (  # noqa: E402
    ExportPipeline,
)
from tools.tessera.importers.ast_schema import (  # noqa: E402
    DocumentAST,
    make_heading,
    make_paragraph,
    new_block_id,
    run_to_json,
)
from tools.tessera.importers.intermediate import (  # noqa: E402
    IntermediateBlock,
    IntermediateDocument,
    IntermediateInlineRun,
)
from tools.tessera.exporters.ast_to_intermediate import (  # noqa: E402
    ast_to_intermediate,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _capturing_dry_run_client():
    """Build a DataLayerClient that returns a captured AST."""
    from tools.tessera.importers.data_layer_client import DataLayerClient

    return DataLayerClient(dry_run=True)


def _ast_from_dict(d):
    """Build a DocumentAST from a dict for testing."""
    return DocumentAST.from_json(d)


class TestMarkdownBuilder(unittest.TestCase):
    def test_heading(self) -> None:
        doc = IntermediateDocument(
            blocks=[IntermediateBlock(type="heading", attrs={"level": 2}, runs=[IntermediateInlineRun(text="Title")])]
        )
        out = md_builder.build_markdown(doc)
        self.assertEqual(out.strip(), "## Title")

    def test_paragraph_with_link(self) -> None:
        doc = IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="paragraph",
                    runs=[IntermediateInlineRun(text="click", annotations=[{"link": "https://example.com"}])],
                )
            ]
        )
        out = md_builder.build_markdown(doc)
        self.assertIn("[click](https://example.com)", out)

    def test_list(self) -> None:
        li1 = IntermediateBlock(type="listItem", runs=[IntermediateInlineRun(text="one")])
        li2 = IntermediateBlock(type="listItem", runs=[IntermediateInlineRun(text="two")])
        container = IntermediateBlock(
            type="list",
            attrs={"style": "unordered", "items": [li1.id, li2.id]},
            children=[li1, li2],
        )
        doc = IntermediateDocument(blocks=[container])
        out = md_builder.build_markdown(doc)
        self.assertIn("- one", out)
        self.assertIn("- two", out)

    def test_table(self) -> None:
        cell = IntermediateBlock(type="paragraph", runs=[IntermediateInlineRun(text="a")])
        container = IntermediateBlock(
            type="table",
            attrs={"rows": 1, "cols": 1, "cells": [[cell.id]]},
            children=[cell],
        )
        doc = IntermediateDocument(blocks=[container])
        out = md_builder.build_markdown(doc)
        self.assertIn("| a |", out)
        self.assertIn("| --- |", out)


class TestHtmlBuilder(unittest.TestCase):
    def test_paragraph(self) -> None:
        doc = IntermediateDocument(
            blocks=[IntermediateBlock(type="paragraph", runs=[IntermediateInlineRun(text="Hello")])]
        )
        out = html_builder.build_html(doc)
        self.assertIn("<p>Hello</p>", out)
        self.assertTrue(out.startswith("<!doctype html>"))

    def test_escapes_user_content(self) -> None:
        doc = IntermediateDocument(
            blocks=[IntermediateBlock(type="paragraph", runs=[IntermediateInlineRun(text="<script>")])]
        )
        out = html_builder.build_html(doc)
        # User content must be escaped.
        self.assertIn("&lt;script&gt;", out)
        self.assertNotIn("<script>", out)

    def test_table(self) -> None:
        cell = IntermediateBlock(type="paragraph", runs=[IntermediateInlineRun(text="a")])
        container = IntermediateBlock(
            type="table",
            attrs={"rows": 1, "cols": 1, "cells": [[cell.id]]},
            children=[cell],
        )
        doc = IntermediateDocument(blocks=[container])
        out = html_builder.build_html(doc)
        self.assertIn("<table>", out)
        self.assertIn("<td>a</td>", out)


class TestEmailBuilder(unittest.TestCase):
    def test_eml_with_meta(self) -> None:
        doc = IntermediateDocument(
            blocks=[
                IntermediateBlock(type="heading", attrs={"level": 1}, runs=[IntermediateInlineRun(text="Subject")]),
                IntermediateBlock(type="paragraph", runs=[IntermediateInlineRun(text="body")]),
            ],
            meta={
                "from": "alice@example.com",
                "to": "bob@example.com",
                "subject": "Subject",
                "date": "Mon, 1 Jan 2024 12:00:00 +0000",
            },
        )
        out = email_builder.build_eml(doc)
        self.assertIn("From: alice@example.com", out)
        self.assertIn("To: bob@example.com", out)
        self.assertIn("Subject: Subject", out)
        # Body content is present.
        self.assertIn("body", out)


class TestDocxBuilder(unittest.TestCase):
    def test_docx_is_valid_zip(self) -> None:
        doc = IntermediateDocument(
            blocks=[
                IntermediateBlock(
                    type="heading", attrs={"level": 1},
                    runs=[IntermediateInlineRun(text="Title")],
                ),
                IntermediateBlock(
                    type="paragraph", runs=[IntermediateInlineRun(text="Body")],
                ),
            ]
        )
        out_path = FIXTURES / "_test_export.docx"
        try:
            docx_builder.build_docx(doc, out_path)
            self.assertTrue(out_path.exists())
            # The output is a valid ZIP (DOCX is OOXML).
            with zipfile.ZipFile(out_path) as zf:
                names = zf.namelist()
                self.assertIn("[Content_Types].xml", names)
                self.assertIn("word/document.xml", names)
        finally:
            if out_path.exists():
                out_path.unlink()


class TestAstToIntermediate(unittest.TestCase):
    def test_round_trip(self) -> None:
        ast = DocumentAST.empty()
        h = make_heading(1, "Title")
        p = make_paragraph("body", annotations=[{"link": "https://example.com"}])
        ast.add_root(h)
        ast.add_root(p)
        inter = ast_to_intermediate(ast)
        # Intermediate has both blocks at the top level.
        self.assertEqual(len(inter.blocks), 2)
        # The link annotation survived.
        para = next(b for b in inter.blocks if b.type == "paragraph")
        self.assertTrue(
            any(
                isinstance(a, dict) and a.get("link") == "https://example.com"
                for a in para.runs[0].annotations
            )
        )


class TestExportPipeline(unittest.TestCase):
    def test_export_dry_run(self) -> None:
        client = _capturing_dry_run_client()
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExportPipeline(client, output_dir=Path(tmpdir))
            result = pipeline.export(
                "00000000-0000-0000-0000-000000000001", "md"
            )
            self.assertEqual(result.format, "md")
            self.assertEqual(result.entity_id, "00000000-0000-0000-0000-000000000001")
            self.assertTrue(result.output_path.exists())
            # The output is a Markdown file (starts with a
            # heading in the dry-run mode).
            content = result.output_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("# "))
            # The SHA-256 is set.
            self.assertTrue(result.sha256.startswith("sha256:"))

    def test_export_all_formats(self) -> None:
        client = _capturing_dry_run_client()
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExportPipeline(client, output_dir=Path(tmpdir))
            for fmt in ["md", "html", "eml", "docx", "xlsx", "pdf", "pptx"]:
                result = pipeline.export(
                    "00000000-0000-0000-0000-000000000002", fmt
                )
                self.assertEqual(result.format, fmt)
                self.assertTrue(
                    result.output_path.exists(),
                    f"output file should exist for {fmt}",
                )
                # The output should be non-empty.
                self.assertGreater(
                    result.size_bytes, 0, f"empty output for {fmt}"
                )

    def test_export_unsupported_format(self) -> None:
        client = _capturing_dry_run_client()
        pipeline = ExportPipeline(client)
        with self.assertRaises(ValueError):
            pipeline.export("entity-id", "unknown-format")


if __name__ == "__main__":
    unittest.main()

"""Tests for the format-specific parsers.

Each test exercises one parser with the canonical fixture
and asserts the expected block types appear. The
``IntermediateDocument`` is the unit-test boundary: the
tests don't care about the AST module; they only check
that the parser produces the right intermediate shape.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))

from tools.tessera.importers.parsers import (  # noqa: E402
    docx as docx_parser,
    email as email_parser,
    html as html_parser,
    markdown as md_parser,
    pdf as pdf_parser,
    pptx as pptx_parser,
    xlsx as xlsx_parser,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _block_types(doc):
    return [b.type for b in doc.blocks]


class TestDocxParser(unittest.TestCase):
    def test_parse_sample(self) -> None:
        doc = docx_parser.parse_docx(FIXTURES / "sample.docx")
        types = _block_types(doc)
        # At minimum we should see one heading and one
        # paragraph. The fixture has 1 heading + 1
        # paragraph + 3 list items.
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        # The list container should be present.
        self.assertIn("list", types)

    def test_parse_extracts_heading_text(self) -> None:
        doc = docx_parser.parse_docx(FIXTURES / "sample.docx")
        heading = next(b for b in doc.blocks if b.type == "heading")
        text = "".join(r.text for r in heading.runs)
        self.assertEqual(text, "Hello Tessera")

    def test_parse_extracts_list_items(self) -> None:
        doc = docx_parser.parse_docx(FIXTURES / "sample.docx")
        lst = next(b for b in doc.blocks if b.type == "list")
        self.assertEqual(lst.attrs.get("style"), "unordered")
        self.assertEqual(len(lst.children), 3)


class TestXlsxParser(unittest.TestCase):
    def test_parse_one_sheet(self) -> None:
        docs = xlsx_parser.parse_xlsx(FIXTURES / "sample.xlsx")
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIn("table", _block_types(doc))
        table = next(b for b in doc.blocks if b.type == "table")
        self.assertEqual(table.attrs.get("rows"), 3)
        self.assertEqual(table.attrs.get("cols"), 3)


class TestPptxParser(unittest.TestCase):
    def test_parse_one_slide(self) -> None:
        docs = pptx_parser.parse_pptx(FIXTURES / "sample.pptx")
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        # Title should become a heading.
        self.assertIn("heading", _block_types(doc))
        # Body text should become paragraphs.
        self.assertIn("paragraph", _block_types(doc))


class TestPdfParser(unittest.TestCase):
    def test_parse_sample(self) -> None:
        doc = pdf_parser.parse_pdf(FIXTURES / "sample.pdf")
        # The fixture has 1 heading + 2 paragraphs.
        self.assertIn("heading", _block_types(doc))
        # At least one paragraph should be present.
        self.assertIn("paragraph", _block_types(doc))

    def test_parse_extracts_text(self) -> None:
        doc = pdf_parser.parse_pdf(FIXTURES / "sample.pdf")
        all_text = " ".join(
            "".join(r.text for r in b.runs) for b in doc.blocks
        )
        # The fixture title is in ALL CAPS so the heading
        # heuristic fires.
        self.assertIn("HELLO TESSERA", all_text)
        self.assertIn("first paragraph", all_text)
        self.assertIn("second paragraph", all_text)


class TestEmailParser(unittest.TestCase):
    def test_parse_eml(self) -> None:
        doc = email_parser.parse_eml(FIXTURES / "sample.eml")
        # Subject is emitted as a heading.
        self.assertIn("heading", _block_types(doc))
        # Body is paragraphs.
        self.assertIn("paragraph", _block_types(doc))
        # Headers are in meta.
        self.assertEqual(doc.meta.get("from"), "alice@example.com")
        self.assertEqual(doc.meta.get("subject"), "Hello Tessera")

    def test_parse_mbox(self) -> None:
        docs = email_parser.parse_mbox(FIXTURES / "sample.mbox")
        # The fixture has 2 messages.
        self.assertEqual(len(docs), 2)
        # Each message has a subject heading.
        self.assertEqual(docs[0].meta.get("subject"), "First message")
        self.assertEqual(docs[1].meta.get("subject"), "Second message")


class TestHtmlParser(unittest.TestCase):
    def test_parse_sample(self) -> None:
        doc = html_parser.parse_html(FIXTURES / "sample.html")
        types = _block_types(doc)
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertIn("list", types)


class TestMarkdownParser(unittest.TestCase):
    def test_parse_sample(self) -> None:
        doc = md_parser.parse_markdown(FIXTURES / "sample.md")
        types = _block_types(doc)
        # The fixture has 1 heading, 1 paragraph, 1 list,
        # 1 table.
        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertIn("list", types)
        self.assertIn("table", types)

    def test_parse_inline_link(self) -> None:
        doc = md_parser.parse_markdown(FIXTURES / "sample.md")
        # Find the paragraph that contains the link text
        # "link" and confirm the link annotation.
        for b in doc.blocks:
            if b.type == "paragraph":
                for r in b.runs:
                    if any(
                        isinstance(a, dict) and "link" in a
                        for a in r.annotations
                    ):
                        # Found the link annotation.
                        self.assertTrue(
                            any(
                                a.get("link") == "https://example.com"
                                for a in r.annotations
                                if isinstance(a, dict)
                            )
                        )
                        return
        self.fail("expected a link annotation in the markdown")

    def test_parse_inline_bold_italic(self) -> None:
        doc = md_parser.parse_markdown(FIXTURES / "sample.md")
        # The fixture has **bold** and *italic* in the
        # paragraph.
        for b in doc.blocks:
            if b.type == "paragraph":
                runs = b.runs
                # We don't require a specific run breakdown;
                # the text must include "bold" and "italic".
                text = "".join(r.text for r in runs)
                if "bold" in text and "italic" in text:
                    self.assertIn("bold", text)
                    self.assertIn("italic", text)
                    return
        self.fail("expected a paragraph containing 'bold' and 'italic'")


if __name__ == "__main__":
    unittest.main()

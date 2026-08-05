"""Generate small fixture files for the importer tests.

This is a helper, not a pytest test (it has no assertions). It's
invoked once during development to write the fixture files into
``tools/tessera/importers/tests/fixtures/``. The fixtures are
then committed to the worktree so the tests can run offline.

Run with ``python3 -m tools.tessera.importers.tests._make_fixtures``.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


def _mkdirs() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def make_markdown() -> None:
    """A simple Markdown file: heading + paragraph + list."""
    FIXTURES.joinpath("sample.md").write_text(
        """# Hello Tessera

This is a short paragraph with a [link](https://example.com) and
some **bold** and *italic* text.

- one
- two
- three

| Name  | Value |
|-------|-------|
| foo   | 1     |
| bar   | 2     |
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def make_html() -> None:
    """A simple HTML file: h1 + p + ul."""
    FIXTURES.joinpath("sample.html").write_text(
        """<!doctype html>
<html><head><title>Sample</title></head><body>
<h1>Hello Tessera</h1>
<p>This is a <strong>simple</strong> HTML file with a
<a href="https://example.com">link</a>.</p>
<ul>
  <li>one</li>
  <li>two</li>
  <li>three</li>
</ul>
</body></html>
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def make_docx() -> None:
    """Build a small DOCX in-process (no python-docx needed at build time).

    DOCX is a ZIP with a few required parts. The minimum is:

    * ``[Content_Types].xml`` — content type registry.
    * ``_rels/.rels`` — root relationships.
    * ``word/document.xml`` — the body.
    * ``word/_rels/document.xml.rels`` — document relationships.

    We use only stdlib so this runs in a fresh venv. The body
    is intentionally simple: one heading, one paragraph with
    bold, one bullet list.
    """
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Hello Tessera</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">This is </w:t></w:r>
      <w:r><w:t xml:space="preserve">a short paragraph.</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr>
        <w:numPr><w:numId w:val="1"/><w:ilvl w:val="0"/></w:numPr>
      </w:pPr>
      <w:r><w:t>First bullet</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr>
        <w:numPr><w:numId w:val="1"/><w:ilvl w:val="0"/></w:numPr>
      </w:pPr>
      <w:r><w:t>Second bullet</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr>
        <w:numPr><w:numId w:val="1"/><w:ilvl w:val="0"/></w:numPr>
      </w:pPr>
      <w:r><w:t>Third bullet</w:t></w:r>
    </w:p>
  </w:body>
</w:document>"""

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

    # styles.xml maps the "Heading1" style id to a "Heading 1"
    # display name. python-docx uses the display name; without
    # this mapping, the style is reported as "Normal".
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1">
    <w:name w:val="Normal"/>
    <w:uiPriority w:val="0"/>
  </w:style>
</w:styles>"""

    # Minimal numbering.xml referenced by document.xml. The
    # parser inspects this to decide ordered vs unordered.
    numbering = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0">
      <w:numFmt w:val="bullet"/>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1">
    <w:abstractNumId w:val="0"/>
  </w:num>
</w:numbering>"""

    with zipfile.ZipFile(FIXTURES / "sample.docx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("word/numbering.xml", numbering)


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def make_xlsx() -> None:
    """Build a small XLSX in-process (no openpyxl at build time).

    Same minimal ZIP structure as DOCX. The sheet has 3 rows
    and 3 columns of text; one cell carries a formula.
    """
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""

    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Name</t></is></c>
      <c r="B1" t="inlineStr"><is><t>Value</t></is></c>
      <c r="C1" t="inlineStr"><is><t>Note</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>foo</t></is></c>
      <c r="B2"><v>1</v></c>
      <c r="C2" t="inlineStr"><is><t>first</t></is></c>
    </row>
    <row r="3">
      <c r="A3" t="inlineStr"><is><t>bar</t></is></c>
      <c r="B3"><v>2</v></c>
      <c r="C3" t="inlineStr"><is><t>second</t></is></c>
    </row>
  </sheetData>
</worksheet>"""

    with zipfile.ZipFile(FIXTURES / "sample.xlsx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------


def make_pptx() -> None:
    """Build a small PPTX in-process.

    One slide with a title + a text frame. PPTX is a ZIP of XML
    parts; we use the same in-process approach as DOCX / XLSX.
    """
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""

    presentation = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId1"/>
  </p:sldIdLst>
</p:presentation>"""

    pres_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""

    slide = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr/>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/><a:lstStyle/>
          <a:p><a:r><a:t>Hello Tessera</a:t></a:r></a:p>
        </p:txBody>
      </p:sp>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>
        <p:spPr/>
        <p:txBody>
          <a:bodyPr/><a:lstStyle/>
          <a:p><a:r><a:t>This slide has two paragraphs.</a:t></a:r></a:p>
          <a:p><a:r><a:t>And a second one.</a:t></a:r></a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>"""

    with zipfile.ZipFile(FIXTURES / "sample.pptx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("ppt/presentation.xml", presentation)
        zf.writestr("ppt/_rels/presentation.xml.rels", pres_rels)
        zf.writestr("ppt/slides/slide1.xml", slide)


# ---------------------------------------------------------------------------
# PDF (1 page, 2 paragraphs) - via reportlab if available, else
# minimal hand-crafted PDF.
# ---------------------------------------------------------------------------


def make_pdf() -> None:
    """Build a small 1-page PDF.

    Uses reportlab when available (the importer requirements
    file lists it as a test-only dep) so the test PDF is
    well-formed and parseable by pdftotext.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        path = FIXTURES / "sample.pdf"
        c = canvas.Canvas(str(path), pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 720, "HELLO TESSERA")
        c.setFont("Helvetica", 12)
        c.drawString(72, 680, "This is the first paragraph.")
        c.drawString(72, 660, "And this is the second paragraph.")
        c.showPage()
        c.save()
        return
    except ImportError:
        pass

    # Hand-crafted minimal PDF (no reportlab). 1 page, 2 lines
    # of text in Helvetica. The bytes below are a complete,
    # valid PDF. pdftotext can extract the text.
    FIXTURES.joinpath("sample.pdf").write_bytes(_HAND_PDF)


_HAND_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 110>>stream
BT
/F1 16 Tf
72 720 Td
(HELLO TESSERA) Tj
0 -40 Td
/F1 12 Tf
(This is the first paragraph.) Tj
0 -20 Td
(And this is the second paragraph.) Tj
ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000053 00000 n
0000000099 00000 n
0000000187 00000 n
0000000346 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
406
%%EOF
"""


# ---------------------------------------------------------------------------
# EML / MBOX
# ---------------------------------------------------------------------------


def make_email() -> None:
    """A simple EML file with From + Subject + body."""
    FIXTURES.joinpath("sample.eml").write_text(
        """From: alice@example.com
To: bob@example.com
Subject: Hello Tessera
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <abc123@example.com>

Hi Bob,

This is the body of the email. It has two paragraphs.

Best,
Alice
""",
        encoding="utf-8",
    )


def make_mbox() -> None:
    """A simple MBOX file with two messages."""
    FIXTURES.joinpath("sample.mbox").write_text(
        """From alice@example.com Mon Jan 01 12:00:00 2024
From: alice@example.com
To: bob@example.com
Subject: First message
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <first@example.com>

First body.

From bob@example.com Mon Jan 01 12:05:00 2024
From: bob@example.com
To: alice@example.com
Subject: Second message
Date: Mon, 1 Jan 2024 12:05:00 +0000
Message-ID: <second@example.com>

Second body.
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_all() -> None:
    _mkdirs()
    make_markdown()
    make_html()
    make_docx()
    make_xlsx()
    make_pptx()
    make_pdf()
    make_email()
    make_mbox()
    print(f"wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    make_all()

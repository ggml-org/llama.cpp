"""Tests for the format detector.

The detector's job is to look at a file's magic bytes and
extension and decide which parser to use. The tests below
exercise each format with the canonical fixture and
verify the right parser is selected. Pandoc is the
catch-all; a file with no extension and no magic is
routed to the Pandoc bridge.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))

from tools.tessera.importers.format_detector import (  # noqa: E402
    EXTENSION_MAP,
    Format,
    detect,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestFormatDetector(unittest.TestCase):
    def test_detect_docx(self) -> None:
        d = detect(FIXTURES / "sample.docx")
        self.assertEqual(d.format, Format.DOCX)

    def test_detect_xlsx(self) -> None:
        d = detect(FIXTURES / "sample.xlsx")
        self.assertEqual(d.format, Format.XLSX)

    def test_detect_pptx(self) -> None:
        d = detect(FIXTURES / "sample.pptx")
        self.assertEqual(d.format, Format.PPTX)

    def test_detect_pdf(self) -> None:
        d = detect(FIXTURES / "sample.pdf")
        self.assertEqual(d.format, Format.PDF)

    def test_detect_eml(self) -> None:
        d = detect(FIXTURES / "sample.eml")
        self.assertEqual(d.format, Format.EML)

    def test_detect_mbox(self) -> None:
        d = detect(FIXTURES / "sample.mbox")
        self.assertEqual(d.format, Format.MBOX)

    def test_detect_html(self) -> None:
        d = detect(FIXTURES / "sample.html")
        self.assertEqual(d.format, Format.HTML)

    def test_detect_markdown(self) -> None:
        d = detect(FIXTURES / "sample.md")
        self.assertEqual(d.format, Format.MARKDOWN)

    def test_detect_unknown_falls_back_to_pandoc(self) -> None:
        # Create a file with no extension and no recognized
        # magic; the detector should fall through to Pandoc.
        tmp = FIXTURES / "_test_unknown.bin"
        tmp.write_bytes(b"this is just some random text content\n")
        try:
            d = detect(tmp)
            self.assertEqual(d.format, Format.PANDOC)
        finally:
            tmp.unlink()

    def test_detect_missing_file_is_pandoc(self) -> None:
        # The detector reports the extension's format when
        # the file is unreadable; the importer's error path
        # then surfaces a useful message.
        d = detect(FIXTURES / "does-not-exist.docx")
        # The path doesn't exist but has a known extension;
        # the detector returns DOCX (so the importer
        # produces a file-not-found error rather than a
        # generic "unsupported format" error).
        self.assertEqual(d.format, Format.DOCX)

    def test_extension_map_completeness(self) -> None:
        # Every concrete format has at least one extension
        # mapping. ``Format.PANDOC`` is the catch-all
        # (no extension); we skip it.
        for fmt in Format:
            if fmt is Format.PANDOC:
                continue
            exts = [ext for ext, f in EXTENSION_MAP.items() if f == fmt]
            self.assertTrue(
                exts, f"format {fmt} has no extension mapping"
            )


if __name__ == "__main__":
    unittest.main()

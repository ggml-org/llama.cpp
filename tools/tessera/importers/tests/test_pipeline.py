"""End-to-end tests for the importer pipeline.

The pipeline orchestrates detection + parsing + AST build +
data layer + receipt. The tests use the dry-run client so
no HTTP traffic is required; the assertions are on the
returned ``ImportSuccess`` records (entity ids, format,
parser, receipt count).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))

from tools.tessera.importers.data_layer_client import (  # noqa: E402
    DataLayerClient,
    ImportResult,
)
from tools.tessera.importers.pipeline import (  # noqa: E402
    ImportPipeline,
)
from tools.tessera.importers.format_detector import Format  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


class _CapturingClient(DataLayerClient):
    """A dry-run client that records every call.

    Used by the pipeline tests to assert the pipeline
    invokes the data layer with the right arguments.
    """

    def __init__(self) -> None:
        super().__init__(dry_run=True)
        self.entities: list[dict] = []
        self.receipts: list[dict] = []

    def create_entity(self, entity_type, label, body, **kwargs):  # type: ignore[override]
        # Don't call super; we want to record but also
        # produce a deterministic UUID so the test can
        # assert on it.
        import uuid

        rid = str(uuid.uuid4())
        rec = {
            "entity_id": rid,
            "entity_type": entity_type,
            "label": label,
            "body": body,
            "kwargs": kwargs,
        }
        self.entities.append(rec)
        return ImportResult(entity_id=rid, entity_type=entity_type, body=body)

    def append_receipt(self, entity_id, receipt_type, payload, signature=None):  # type: ignore[override]
        import uuid

        rid = str(uuid.uuid4())
        self.receipts.append(
            {
                "receipt_id": rid,
                "entity_id": entity_id,
                "receipt_type": receipt_type,
                "payload": payload,
                "signature": signature,
            }
        )
        return rid


class TestImporterPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _CapturingClient()
        self.pipeline = ImportPipeline(self.client)

    def test_import_markdown(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.md")
        self.assertEqual(s.format, Format.MARKDOWN.value)
        self.assertEqual(s.parser, "markdown-it-py")
        self.assertEqual(len(s.entities), 1)
        self.assertEqual(len(s.receipts), 1)
        # Receipt is "import" type.
        self.assertEqual(s.receipts[0].receipt_type, "import")
        # Entity was created with the document type.
        self.assertEqual(self.client.entities[0]["entity_type"], "document")

    def test_import_xlsx_multi_entity(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.xlsx")
        # One sheet, one entity, one receipt.
        self.assertEqual(len(s.entities), 1)
        self.assertEqual(s.entities[0].entity_type, "spreadsheet")
        self.assertEqual(len(s.receipts), 1)

    def test_import_mbox_multi_entity(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.mbox")
        # Two messages, two entities, two receipts.
        self.assertEqual(len(s.entities), 2)
        self.assertEqual(len(s.receipts), 2)
        for e in s.entities:
            self.assertEqual(e.entity_type, "email")

    def test_import_pdf_creates_document(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.pdf")
        self.assertEqual(s.entities[0].entity_type, "document")
        self.assertEqual(s.parser, "pdftotext")

    def test_import_eml_metadata(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.eml")
        # Subject is the title in the meta of the entity.
        body = self.client.entities[0]["body"]
        # The body is the AST JSON; the label is set from the
        # title in the AST's meta.
        self.assertEqual(self.client.entities[0]["label"], "Hello Tessera")

    def test_import_receipt_carries_hash(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.md")
        payload = s.receipts[0].payload
        self.assertIn("ast_content_hash", payload)
        self.assertTrue(payload["ast_content_hash"].startswith("sha256:"))
        self.assertIn("parser_used", payload)
        self.assertEqual(payload["parser_used"], "markdown-it-py")

    def test_batch_import_all_formats(self) -> None:
        result = self.pipeline.import_paths(
            [
                FIXTURES / "sample.md",
                FIXTURES / "sample.html",
                FIXTURES / "sample.docx",
                FIXTURES / "sample.xlsx",
                FIXTURES / "sample.pptx",
                FIXTURES / "sample.pdf",
                FIXTURES / "sample.eml",
                FIXTURES / "sample.mbox",
            ]
        )
        self.assertTrue(result.ok, f"imports failed: {result.failures}")
        self.assertEqual(len(result.successes), 8)
        # 1 (md) + 1 (html) + 1 (docx) + 1 (xlsx) + 1 (pptx) + 1 (pdf) + 1 (eml) + 2 (mbox) = 9 entities
        self.assertEqual(len(self.client.entities), 9)
        # Same number of receipts.
        self.assertEqual(len(self.client.receipts), 9)

    def test_failed_parse_continues_batch(self) -> None:
        # A file that exists but is unparseable is a failure
        # that the pipeline collects; the batch continues.
        bad = FIXTURES / "_bad.docx"
        bad.write_bytes(b"not a real docx")
        try:
            result = self.pipeline.import_paths(
                [
                    bad,
                    FIXTURES / "sample.md",
                ]
            )
            self.assertFalse(result.ok)
            self.assertEqual(len(result.successes), 1)
            self.assertEqual(len(result.failures), 1)
            self.assertEqual(result.failures[0].path.name, "_bad.docx")
        finally:
            bad.unlink()

    def test_ast_body_is_valid_json(self) -> None:
        s = self.pipeline.import_file(FIXTURES / "sample.md")
        body = self.client.entities[0]["body"]
        # Body is a JSON string; it must parse.
        parsed = json.loads(body)
        self.assertIn("blocks", parsed)
        self.assertIn("rootChildren", parsed)

    def test_fail_fast_aborts(self) -> None:
        bad = FIXTURES / "_bad2.docx"
        bad.write_bytes(b"not a real docx")
        try:
            pipeline = ImportPipeline(self.client, fail_fast=True)
            result = pipeline.import_paths(
                [
                    bad,
                    FIXTURES / "sample.md",
                ]
            )
            # With fail_fast, the second import is skipped.
            self.assertEqual(len(result.successes), 0)
            self.assertEqual(len(result.failures), 1)
        finally:
            if bad.exists():
                bad.unlink()


if __name__ == "__main__":
    unittest.main()

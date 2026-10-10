#!/usr/bin/env python3
"""Checks for archive inventory coverage and byte-identity semantics."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from index_benchmarks import inventory


class InventoryTests(unittest.TestCase):
    def test_hidden_artifacts_and_duplicates_keep_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case").mkdir()
            (root / "oneapi-ab").mkdir()
            (root / "README.md").write_text("guide, excluded")
            (root / "FILES.tsv").write_text("old inventory, excluded")
            (root / "case/README.md").write_text("archived report, included")
            (root / "case/result.json").write_bytes(b"same")
            (root / "oneapi-ab/result.json").write_bytes(b"same")
            (root / ".hidden.log").write_bytes(b"")
            rows = {str(row["path"]): row for row in inventory(root)}
            self.assertEqual(set(rows), {"case/README.md", "case/result.json",
                                         "oneapi-ab/result.json", ".hidden.log"})
            self.assertEqual(rows["case/result.json"]["sha256"],
                             hashlib.sha256(b"same").hexdigest())
            self.assertEqual(rows["oneapi-ab/result.json"]["identical_to"], "case/result.json")
            self.assertEqual(rows["case/result.json"]["bytes"], 4)
            self.assertEqual(rows[".hidden.log"]["bytes"], 0)

    def test_links_are_recorded_without_reading_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside").symlink_to("/does/not/exist")
            (root / "loop").symlink_to(root, target_is_directory=True)
            rows = list(inventory(root))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["kind"] == "symlink" and not row["sha256"] for row in rows))
            self.assertEqual(rows[1]["link_target"], "/does/not/exist")


if __name__ == "__main__":
    unittest.main()

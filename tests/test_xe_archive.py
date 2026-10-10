"""Validate the committed xe research archive without the original host files."""
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/xe-fix-docs"


class ArchiveTests(unittest.TestCase):
    def test_manifest_targets_and_hashes(self):
        tracked = set(subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], text=True).splitlines())
        manifest = json.loads((ARCHIVE / "session/source-manifest-2026-09-30.json").read_text())
        for entry in manifest["files"]:
            path = ARCHIVE / entry["archive"]
            with self.subTest(path=entry["archive"]):
                self.assertIn(path.relative_to(ROOT).as_posix(), tracked)
                data = path.read_bytes()
                self.assertEqual(len(data), entry["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertIn("source_sha256", entry)
        for entry in manifest["omitted"]:
            self.assertNotIn((ARCHIVE / entry["archive"]).relative_to(ROOT).as_posix(), tracked)

    def test_single_raw_tree(self):
        alias = ARCHIVE / "evidence/sycl-oneapi-benchmarks-2026-09-29/raw"
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), (ARCHIVE / "evidence/raw").resolve())

    def test_single_investigation_tree(self):
        self.assertTrue((ARCHIVE / "artifacts").is_dir())
        self.assertFalse((ARCHIVE / "evidence/xe-investigation-20260929").exists())

    def test_committed_archive_text_is_ascii(self):
        paths = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "docs/xe-fix-docs"], text=True).splitlines()
        for name in paths:
            path = ROOT / name
            if path.is_symlink():
                continue
            data = path.read_bytes()
            if b"\0" in data:
                continue
            with self.subTest(path=name):
                self.assertTrue(data.isascii())


if __name__ == "__main__":
    unittest.main()

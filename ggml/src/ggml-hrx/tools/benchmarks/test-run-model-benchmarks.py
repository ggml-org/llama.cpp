#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run-model-benchmarks.py")
SPEC = importlib.util.spec_from_file_location("run_model_benchmarks", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LoomLinkSourceArgsTest(unittest.TestCase):
    def test_preserves_provider_and_source_closure_roles_for_retry(self) -> None:
        entry = {
            "corpus_dir": "kernel-corpus/kernels/loom-libs",
            "sources": ["ops/example.loom", "motifs/helper.loom"],
            "primary_sources": ["ops/example.loom"],
            "library_sources": ["motifs/helper.loom"],
        }

        args = MODULE.loom_link_provider_source_args(entry)

        corpus = MODULE.HRX_DIR / entry["corpus_dir"]
        self.assertEqual(
            args,
            [
                str(corpus / "motifs/helper.loom"),
                f"--library={corpus / 'ops/example.loom'}",
            ],
        )

    def test_supports_legacy_flat_source_manifests(self) -> None:
        entry = {
            "corpus_dir": "kernel-corpus/kernels/loom-libs",
            "sources": ["ops/example.loom", "motifs/helper.loom"],
        }

        args = MODULE.loom_link_source_args(entry)

        corpus = MODULE.HRX_DIR / entry["corpus_dir"]
        self.assertEqual(
            args,
            [str(corpus / "ops/example.loom"), str(corpus / "motifs/helper.loom")],
        )

    def test_initial_link_retains_flat_source_order(self) -> None:
        entry = {
            "corpus_dir": "kernel-corpus/kernels/loom-libs",
            "sources": ["ops/example.loom", "motifs/helper.loom"],
            "primary_sources": ["ops/example.loom"],
            "library_sources": ["motifs/helper.loom"],
        }

        args = MODULE.loom_link_source_args(entry)

        corpus = MODULE.HRX_DIR / entry["corpus_dir"]
        self.assertEqual(
            args,
            [str(corpus / "ops/example.loom"), str(corpus / "motifs/helper.loom")],
        )

    def test_retries_only_missed_exact_source_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stderr = root / "stderr.txt"
            stderr.write_text(
                "exact link selection missed reachable source symbol ref {module=0, symbol=2}\n",
                encoding="utf-8",
            )
            result = MODULE.CommandResult(
                "failed",
                "loom-link exited with status 1",
                1,
                None,
                root / "stdout.txt",
                stderr,
                root / "command.json",
            )
            self.assertTrue(MODULE.missed_exact_source_selection(result))

            stderr.write_text("unresolved exact declaration\n", encoding="utf-8")
            self.assertFalse(MODULE.missed_exact_source_selection(result))


if __name__ == "__main__":
    unittest.main()

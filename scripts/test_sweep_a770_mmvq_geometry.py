#!/usr/bin/env python3
"""Unit tests for the Arc A770 MMVQ geometry sweep orchestrator."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("sweep-a770-mmvq-geometry.py")
SPEC = importlib.util.spec_from_file_location("sweep_a770_mmvq_geometry", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load geometry sweep: {SCRIPT_PATH}")
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class GeometrySweepTests(unittest.TestCase):
    def test_added_suffix_handles_full_and_truncated_history(self) -> None:
        self.assertEqual(SWEEP.added_suffix(["a", "b"], ["a", "b", "c"]), ["c"])
        self.assertEqual(SWEEP.added_suffix(["a", "b"], ["b", "c"]), ["c"])
        self.assertEqual(SWEEP.added_suffix(["a"], ["x"]), ["x"])

    def test_empty_fuser_exit_one_proves_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 1, "", "")
        with mock.patch.object(SWEEP, "run", return_value=proc), mock.patch.object(
            SWEEP, "require_tool", return_value="fuser"
        ):
            SWEEP.require_sole_tenancy("/dev/dri/renderD128")

    def test_holder_or_fuser_error_blocks_gpu_work(self) -> None:
        cases = ((0, "1234 llama-bench"), (1, "permission denied"))
        for returncode, output in cases:
            with self.subTest(returncode=returncode, output=output):
                proc = subprocess.CompletedProcess(["fuser"], returncode, output, "")
                with mock.patch.object(SWEEP, "run", return_value=proc), mock.patch.object(
                    SWEEP, "require_tool", return_value="fuser"
                ):
                    with self.assertRaises(SWEEP.SweepError):
                        SWEEP.require_sole_tenancy("/dev/dri/renderD128")

    def test_build_directory_encodes_geometry_and_tag(self) -> None:
        self.assertEqual(
            SWEEP.build_dir(Path("/builds"), "abc123", 4, 32),
            Path("/builds/build-p58-y4-sg32-abc123"),
        )


if __name__ == "__main__":
    unittest.main()

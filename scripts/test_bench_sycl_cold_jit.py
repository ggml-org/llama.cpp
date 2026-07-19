#!/usr/bin/env python3
"""Unit tests for the SYCL cold-JIT benchmark runner."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("bench-sycl-cold-jit.py")
SPEC = importlib.util.spec_from_file_location("bench_sycl_cold_jit", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load cold-JIT runner: {SCRIPT_PATH}")
COLD_JIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLD_JIT)


class ColdJitRunnerTests(unittest.TestCase):
    def test_parse_assignments_requires_named_nonempty_values(self) -> None:
        self.assertEqual(
            COLD_JIT.parse_assignments(["mistral=/models/m.gguf", "MODE=fast"]),
            {"mistral": "/models/m.gguf", "MODE": "fast"},
        )
        for invalid in ("missing-separator", "=missing-name", "missing-value="):
            with self.subTest(invalid=invalid), self.assertRaises(COLD_JIT.ColdJitError):
                COLD_JIT.parse_assignments([invalid])

    def test_summary_reports_median_and_student_t_interval(self) -> None:
        summary = COLD_JIT.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["n"], 5)
        self.assertEqual(summary["median"], 3.0)
        self.assertLess(summary["lower95"], summary["mean"])
        self.assertGreater(summary["upper95"], summary["mean"])

    def test_summary_uses_student_t_through_thirty_degrees_of_freedom(self) -> None:
        values = [float(value) for value in range(31)]
        summary = COLD_JIT.summarize(values)
        expected_ci95 = 2.042 * COLD_JIT.statistics.stdev(values) / len(values) ** 0.5
        self.assertAlmostEqual(summary["ci95"], expected_ci95)

    def test_empty_fuser_exit_one_proves_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 1, "", "")
        with mock.patch.object(COLD_JIT.subprocess, "run", return_value=proc):
            snapshot = COLD_JIT.holder_snapshot("/dev/dri/renderD128")
        self.assertEqual(snapshot["pids"], [])

    def test_fuser_failure_is_not_mistaken_for_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 2, "", "permission denied")
        with mock.patch.object(COLD_JIT.subprocess, "run", return_value=proc):
            with self.assertRaises(COLD_JIT.ColdJitError):
                COLD_JIT.holder_snapshot("/dev/dri/renderD128")
    def test_fuser_exit_one_with_diagnostics_is_not_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 1, "", "permission denied")
        with mock.patch.object(COLD_JIT.subprocess, "run", return_value=proc):
            with self.assertRaises(COLD_JIT.ColdJitError):
                COLD_JIT.holder_snapshot("/dev/dri/renderD128")



if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the SYCL cold-JIT benchmark runner."""

from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import tempfile
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
        for invalid in (
            "missing-separator",
            "=missing-name",
            "missing-value=",
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(COLD_JIT.ColdJitError),
            ):
                COLD_JIT.parse_assignments([invalid])

    def test_summary_reports_student_t_interval(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = COLD_JIT.summarize(values)
        expected = 2.776 * statistics.stdev(values) / len(values) ** 0.5
        self.assertEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["ci95"], expected)

    def test_summary_uses_table_through_thirty_degrees(self) -> None:
        values = [float(value) for value in range(31)]
        summary = COLD_JIT.summarize(values)
        expected = 2.042 * statistics.stdev(values) / len(values) ** 0.5
        self.assertAlmostEqual(summary["ci95"], expected)

    def test_summary_approximates_student_t_above_thirty_degrees(
        self,
    ) -> None:
        values = [float(value) for value in range(41)]
        summary = COLD_JIT.summarize(values)
        standard_error = statistics.stdev(values) / len(values) ** 0.5
        critical = summary["ci95"] / standard_error
        self.assertAlmostEqual(critical, 2.021075, places=5)

    def test_empty_fuser_exit_one_proves_idle(self) -> None:
        proc = subprocess.CompletedProcess(["fuser"], 1, "", "")
        with mock.patch.object(COLD_JIT.subprocess, "run", return_value=proc):
            snapshot = COLD_JIT.holder_snapshot("/dev/dri/renderD128")
        self.assertEqual(snapshot["pids"], [])

    def test_fuser_failure_is_not_mistaken_for_idle(self) -> None:
        for returncode in (1, 2):
            proc = subprocess.CompletedProcess(
                ["fuser"], returncode, "", "permission denied"
            )
            with (
                self.subTest(returncode=returncode),
                mock.patch.object(COLD_JIT.subprocess, "run", return_value=proc),
                self.assertRaises(COLD_JIT.ColdJitError),
            ):
                COLD_JIT.holder_snapshot("/dev/dri/renderD128")

    def test_run_sample_times_out_after_stdout_eof(self) -> None:
        proc = mock.Mock()
        proc.stdout = iter(())
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd=["llama-bench"], timeout=1),
            0,
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(COLD_JIT.subprocess, "Popen", return_value=proc),
            self.assertRaisesRegex(COLD_JIT.ColdJitError, "timed out"),
        ):
            COLD_JIT.run_sample(
                bench=Path("bench"),
                bin_dir=Path("bin"),
                model=Path("model"),
                timeout_s=10,
                env_extra={},
                stderr_path=Path(directory) / "stderr.log",
            )
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)

    def test_run_sample_rejects_invalid_timing_rows(self) -> None:
        invalid_rows = (
            {},
            {"avg_ts": None},
            {"avg_ts": "1.0"},
            {"avg_ts": float("nan")},
            {"avg_ts": float("inf")},
        )
        for row in invalid_rows:
            proc = mock.Mock()
            proc.stdout = iter((json.dumps(row) + "\n",))
            proc.wait.return_value = 0
            with (
                self.subTest(row=row),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(COLD_JIT.subprocess, "Popen", return_value=proc),
                self.assertRaisesRegex(COLD_JIT.ColdJitError, "invalid avg_ts"),
            ):
                COLD_JIT.run_sample(
                    bench=Path("bench"),
                    bin_dir=Path("bin"),
                    model=Path("model"),
                    timeout_s=10,
                    env_extra={},
                    stderr_path=Path(directory) / "stderr.log",
                )

    def test_run_sample_accepts_required_finite_rows(self) -> None:
        lines = (
            "diagnostic noise\n",
            json.dumps({"avg_ts": 3.0, "n_prompt": 512, "n_gen": 0}) + "\n",
            json.dumps({"avg_ts": 4.0, "n_prompt": 0, "n_gen": 128}) + "\n",
        )
        proc = mock.Mock()
        proc.stdout = iter(lines)
        proc.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(COLD_JIT.subprocess, "Popen", return_value=proc),
        ):
            result = COLD_JIT.run_sample(
                bench=Path("bench"),
                bin_dir=Path("bin"),
                model=Path("model"),
                timeout_s=10,
                env_extra={"LD_LIBRARY_PATH": "/custom"},
                stderr_path=Path(directory) / "stderr.log",
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["effective_ld_library_path"], "bin:/custom")

    def test_failed_rerun_removes_stale_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            product = out_dir / "product.json"
            product.write_text("stale", encoding="utf-8")
            args = mock.Mock(
                bin_dir=Path("bin"),
                model=[],
                out_dir=out_dir,
                repetitions=2,
                timeout=10,
                render_node="/dev/dri/renderD128",
                env=[],
            )
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=args,
            ):
                self.assertEqual(COLD_JIT.main(), 1)
            self.assertFalse(product.exists())

    def test_effective_library_path_uses_override(self) -> None:
        self.assertEqual(
            COLD_JIT.effective_library_path(
                Path("/build/bin"),
                {"LD_LIBRARY_PATH": "/override"},
            ),
            "/build/bin:/override",
        )


if __name__ == "__main__":
    unittest.main()

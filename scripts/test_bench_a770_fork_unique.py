#!/usr/bin/env python3
"""Unit tests for the Arc A770 product campaign harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("bench-a770-fork-unique.py")
SPEC = importlib.util.spec_from_file_location("bench_a770_fork_unique", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load benchmark harness: {SCRIPT_PATH}")
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def bench_result(pp: float | None, tg: float | None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if pp is not None:
        rows.append({"n_prompt": 512, "n_gen": 0, "avg_ts": pp})
    if tg is not None:
        rows.append({"n_prompt": 0, "n_gen": 128, "avg_ts": tg})
    return {
        "ok": True,
        "returncode": 0,
        "elapsed_s": 0.1,
        "stdout": json.dumps(rows),
        "stderr": "",
    }


class ProductCampaignTests(unittest.TestCase):
    def test_selects_pp512_and_tg128_rows(self) -> None:
        rows = [
            {"n_prompt": 64, "n_gen": 0, "avg_ts": 999.0},
            {"n_prompt": 512, "n_gen": 1, "avg_ts": 777.0},
            {"n_prompt": 1, "n_gen": 128, "avg_ts": 666.0},
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 101.5},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 12.25},
        ]
        selected = HARNESS._select_product_rows(rows)
        self.assertEqual(selected["pp512"], rows[3])
        self.assertEqual(selected["tg128"], rows[4])

    def test_product_argv_uses_supported_no_warmup_flag(self) -> None:
        argv = HARNESS._product_bench_argv(
            Path("/tmp/bin"), "model.gguf", ("f16", "f16"), 0
        )
        self.assertIn("--no-warmup", argv)
        self.assertNotIn("-no-warmup", argv)
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "512")

    def test_tenancy_probe_accepts_only_empty_exit_one(self) -> None:
        cases = (
            (1, "", "", False),
            (0, "1234", "", True),
            (1, "", "permission denied", True),
            (2, "", "", True),
        )
        for returncode, stdout, stderr, should_raise in cases:
            with self.subTest(returncode=returncode, stdout=stdout, stderr=stderr):
                proc = mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)
                runner = mock.Mock(return_value=proc)
                if should_raise:
                    with self.assertRaises(HARNESS.SoleTenancyViolation):
                        HARNESS.check_sole_tenancy(runner=runner)
                else:
                    HARNESS.check_sole_tenancy(runner=runner)

    def test_candidate_binary_requires_candidate_environment(self) -> None:
        argv = [
            str(SCRIPT_PATH),
            "--campaign", "product",
            "--bin-dir", "/tmp/baseline",
            "--candidate-bin-dir", "/tmp/candidate",
            "--model", "/tmp/model.gguf",
            "--out-dir", "/tmp/output",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            HARNESS.main()
        self.assertEqual(raised.exception.code, 2)

    def test_effective_env_clears_ambient_behavior_toggles(self) -> None:
        controlled = (
            "GGML_SYCL_FA_XMX",
            "GGML_SYCL_FA_FORCE_VEC_STANDARD",
            "GGML_SYCL_FA_Q8_GQA_TILE",
            "GGML_SYCL_Q8_KV_QUANTS_FIRST",
            "LLAMA_ENABLE_INNERQ",
            "TURBO_LAYER_ADAPTIVE",
            "TURBO_AUTO_ASYMMETRIC",
        )
        with mock.patch.dict(HARNESS.os.environ, {name: "1" for name in controlled}):
            env = HARNESS._effective_env({"GGML_SYCL_FA_Q8_GQA_TILE": "1"})

        for name in controlled:
            if name == "GGML_SYCL_FA_Q8_GQA_TILE":
                self.assertEqual(env[name], "1")
            else:
                self.assertNotIn(name, env)

    def test_environment_lists_reject_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate name 'TEST_ARM'"):
            HARNESS._parse_env_list(
                ["TEST_ARM=baseline", "TEST_ARM=candidate"]
            )

    def test_product_input_lists_reject_empty_incomplete_and_negative_entries(self) -> None:
        for value in ("", "0,", "-1"):
            with self.subTest(depths=value), self.assertRaises(ValueError):
                HARNESS._parse_depths(value)
        for value in ("", "q8_0/q8_0,", "q8_0", "/q8_0", "q8_0/"):
            with self.subTest(kv_types=value), self.assertRaises(ValueError):
                HARNESS._parse_kv_types(value)

    def test_non_finite_timing_invalidates_aligned_samples(self) -> None:
        samples = {
            "baseline": [{"rep": 1, "pp512_ts": 100.0}],
            "candidate": [{"rep": 1, "pp512_ts": float("nan")}],
        }

        _, failures = HARNESS._align_cell_samples_by_rep(samples, [1], "pp512")

        self.assertTrue(any("non-finite" in failure for failure in failures))

    def test_product_requires_two_retained_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            bench.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)
            ns.repetitions = 2

            rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, 2)
            self.assertFalse(out_dir.exists())



    def test_run_product_cell_requires_two_retained_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "repetitions must be >= 3"):
            HARNESS.run_product_cell(
                bin_dir=Path("bin"),
                model_path="model.gguf",
                kv=("q8_0", "q8_0"),
                depth=0,
                baseline_env={},
                candidate_env=None,
                repetitions=2,
                timeout_s=5,
                samples_dir=Path("samples"),
                cell_idx=1,
            )

    def test_product_argv_uses_supported_no_warmup_flag(self) -> None:
        argv = HARNESS._product_bench_argv(
            Path("/tmp/bin"), "model.gguf", ("q8_0", "q8_0"), 4096
        )

        self.assertIn("--no-warmup", argv)
        self.assertNotIn("-no-warmup", argv)
        self.assertEqual(argv[argv.index("-n") + 1], "128")
        self.assertEqual(argv[argv.index("-b") + 1], "512")
        self.assertEqual(argv[argv.index("-ub") + 1], "512")
        self.assertEqual(argv[-2:], ["-d", "4096"])

    def test_discards_sample_zero_and_pairs_baseline_candidate(self) -> None:
        calls = {"baseline": 0, "candidate": 0}

        def fake_run(argv, env_extra, timeout_s, cwd=None):
            del argv, timeout_s, cwd
            arm = "candidate" if env_extra else "baseline"
            rep = calls[arm]
            calls[arm] += 1
            offset = 10.0 if arm == "candidate" else 0.0
            return bench_result(100.0 + offset + rep, 20.0 + offset + rep)

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            HARNESS, "check_sole_tenancy"
        ), mock.patch.object(HARNESS, "run", side_effect=fake_run):
            cell = HARNESS.run_product_cell(
                bin_dir=Path(td),
                model_path="model.gguf",
                kv=("q8_0", "q8_0"),
                depth=0,
                baseline_env={},
                candidate_env={"TEST_ARM": "candidate"},
                repetitions=3,
                timeout_s=5,
                samples_dir=Path(td) / "samples",
                cell_idx=1,
            )

        self.assertTrue(cell["valid"])
        self.assertEqual(
            cell["metrics"]["pp512"]["retained_baseline_ts"], [101.0, 102.0]
        )
        self.assertEqual(
            cell["metrics"]["pp512"]["retained_candidate_ts"], [111.0, 112.0]
        )
        self.assertEqual(cell["metrics"]["pp512"]["paired"]["n"], 2)
        self.assertEqual(cell["metrics"]["tg128"]["paired"]["n"], 2)

    def test_missing_required_row_invalidates_whole_cell(self) -> None:
        for missing_metric in ("pp512", "tg128"):
            with self.subTest(missing_metric=missing_metric):
                calls = 0

                def fake_run(argv, env_extra, timeout_s, cwd=None):
                    nonlocal calls
                    del argv, env_extra, timeout_s, cwd
                    calls += 1
                    if calls == 2 and missing_metric == "pp512":
                        return bench_result(None, 20.0)
                    if calls == 2 and missing_metric == "tg128":
                        return bench_result(100.0, None)
                    return bench_result(100.0, 20.0)

                with tempfile.TemporaryDirectory() as td, mock.patch.object(
                    HARNESS, "check_sole_tenancy"
                ), mock.patch.object(HARNESS, "run", side_effect=fake_run):
                    cell = HARNESS.run_product_cell(
                        bin_dir=Path(td),
                        model_path="model.gguf",
                        kv=("q8_0", "q8_0"),
                        depth=0,
                        baseline_env={},
                        candidate_env=None,
                        repetitions=3,
                        timeout_s=5,
                        samples_dir=Path(td) / "samples",
                        cell_idx=1,
                    )

                self.assertFalse(cell["valid"])
                for metric in cell["metrics"].values():
                    self.assertEqual(metric["retained_baseline_ts"], [])
                    self.assertEqual(metric["baseline_stats"]["n"], 0)

    def test_zero_byte_model_rejected_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            bench.chmod(0o755)
            model = root / "empty.gguf"
            model.touch()
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)

            rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, 2)
            self.assertFalse(out_dir.exists())

    def test_tenancy_holder_returns_70(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            bench.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)
            violation = HARNESS.SoleTenancyViolation(["1234 /usr/bin/gpu-user"])

            with mock.patch.object(
                HARNESS, "check_sole_tenancy", side_effect=violation
            ):
                rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, HARNESS.SOLE_TENANCY_EXIT)
            self.assertIn(
                "1234", (out_dir / "sole-tenancy-violation.txt").read_text()
            )

    def test_nonzero_sample_invalidates_cell(self) -> None:
        calls = 0

        def fake_run(argv, env_extra, timeout_s, cwd=None):
            nonlocal calls
            del argv, env_extra, timeout_s, cwd
            calls += 1
            result = bench_result(100.0, 20.0)
            if calls == 2:
                result["ok"] = False
                result["returncode"] = 3
            return result

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            HARNESS, "check_sole_tenancy"
        ), mock.patch.object(HARNESS, "run", side_effect=fake_run):
            cell = HARNESS.run_product_cell(
                bin_dir=Path(td),
                model_path="model.gguf",
                kv=("q8_0", "q8_0"),
                depth=0,
                baseline_env={},
                candidate_env=None,
                repetitions=3,
                timeout_s=5,
                samples_dir=Path(td) / "samples",
                cell_idx=1,
            )

        self.assertFalse(cell["valid"])
        self.assertEqual(cell["metrics"]["pp512"]["baseline_stats"]["n"], 0)

    def test_sample_artifacts_preserve_raw_output_and_both_env_maps(self) -> None:
        prefix = "raw-prefix-" + ("x" * 5000)

        def fake_run(argv, env_extra, timeout_s, cwd=None):
            del argv, timeout_s, cwd
            result = bench_result(100.0, 20.0)
            result["stdout"] = prefix + result["stdout"]
            result["stderr"] = f"TEST_ARM: {env_extra['TEST_ARM']}\n" + ("y" * 5000)
            return result

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            HARNESS, "check_sole_tenancy"
        ), mock.patch.object(HARNESS, "run", side_effect=fake_run):
            samples_dir = Path(td) / "samples"
            HARNESS.run_product_cell(
                bin_dir=Path(td),
                model_path="model.gguf",
                kv=("q8_0", "q8_0"),
                depth=0,
                baseline_env={"TEST_ARM": "baseline"},
                candidate_env={"TEST_ARM": "candidate"},
                repetitions=3,
                timeout_s=5,
                samples_dir=samples_dir,
                cell_idx=1,
            )
            for sample_path in samples_dir.glob("*.json"):
                sample = json.loads(sample_path.read_text())
                self.assertTrue(sample["stdout"].startswith(prefix))
                self.assertGreater(len(sample["stderr"]), 5000)
                self.assertEqual(sample["baseline_env"], {"TEST_ARM": "baseline"})
                self.assertEqual(sample["candidate_env"], {"TEST_ARM": "candidate"})

    def test_q8_effective_requested_kv_bandwidth_formula(self) -> None:
        layers, heads, depth, head_dim = 32, 32, 16384, 128
        expected = 2 * layers * heads * depth * head_dim * (34 / 32)
        actual = HARNESS._effective_kv_read_bytes_per_step(
            ("q8_0", "q8_0"), depth, layers, heads, head_dim
        )
        self.assertEqual(actual, expected)
        cell = {
            "kv": ["q8_0", "q8_0"],
            "depth": depth,
            "metrics": {
                "tg128": {
                    "baseline_stats": {"median": 20.0},
                    "candidate_stats": {"median": 22.0, "n": 1},
                }
            },
        }
        HARNESS._annotate_effective_kv_bandwidth(
            cell, (layers, heads, head_dim)
        )
        self.assertEqual(cell["effective_kv_read_bytes_per_step"], expected)
        self.assertEqual(cell["effective_kv_gbps"]["baseline"], expected * 20 / 1e9)

    def test_campaign_routes_distinct_envs_and_asserts_logged_value(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            candidate_bin_dir = root / "candidate-bin"
            candidate_bin_dir.mkdir()
            candidate_bench = candidate_bin_dir / "llama-bench"
            candidate_bench.write_text("#!/bin/sh\n", encoding="utf-8")
            candidate_bench.chmod(0o755)
            bench.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)
            ns.candidate_bin_dir = str(candidate_bin_dir)
            ns.repetitions = 3
            ns.baseline_env = ["TEST_ARM=baseline"]
            ns.env = ["TEST_ARM=candidate"]
            seen_envs = []
            seen_bench_paths = []

            def fake_run(argv, env_extra, timeout_s, cwd=None):
                del timeout_s, cwd
                seen_bench_paths.append(argv[0])
                seen_envs.append(dict(env_extra))
                result = bench_result(100.0, 20.0)
                result["stderr"] = f"TEST_ARM: {env_extra['TEST_ARM']}\n"
                return result

            def clean_dmesg(path, *args, **kwargs):
                del args, kwargs
                path.write_text("", encoding="utf-8")
                return 0

            with mock.patch.object(HARNESS, "check_sole_tenancy"), \
                 mock.patch.object(HARNESS, "run", side_effect=fake_run), \
                 mock.patch.object(HARNESS, "capture_dmesg", side_effect=clean_dmesg), \
                 mock.patch.object(HARNESS, "collect_product_provenance", return_value={}):
                rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, 0)
            self.assertIn({"TEST_ARM": "baseline"}, seen_envs)
            self.assertIn({"TEST_ARM": "candidate"}, seen_envs)
            self.assertIn(str(bench), seen_bench_paths)
            self.assertIn(str(candidate_bench), seen_bench_paths)
            summary = json.loads((out_dir / "product.json").read_text())
            self.assertEqual(summary["baseline_env"], {"TEST_ARM": "baseline"})
            self.assertEqual(
                summary["candidate_bin_dir"], str(candidate_bin_dir.resolve())
            )
            self.assertEqual(summary["candidate_env"], {"TEST_ARM": "candidate"})
            self.assertTrue(
                summary["candidate_env_log_assertions"]["TEST_ARM"]["valid"]
            )

    def test_new_i915_dmesg_fault_invalidates_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            bench.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)
            ns.repetitions = 3
            captures = 0

            def capture(path, *args, **kwargs):
                nonlocal captures
                del args, kwargs
                captures += 1
                line = (
                    "[now] i915 GPU HANG detected\n" if captures == 2 else ""
                )
                path.write_text(line, encoding="utf-8")
                return 1 if line else 0

            with mock.patch.object(HARNESS, "check_sole_tenancy"), \
                 mock.patch.object(HARNESS, "run", return_value=bench_result(100.0, 20.0)), \
                 mock.patch.object(HARNESS, "capture_dmesg", side_effect=capture), \
                 mock.patch.object(HARNESS, "collect_product_provenance", return_value={}):
                rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, 1)
            summary = json.loads((out_dir / "product.json").read_text())
            self.assertFalse(summary["all_cells_valid"])
            self.assertEqual(
                summary["dmesg_new_matches"], ["[now] i915 GPU HANG detected"]
            )

    def test_partial_model_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bench = bin_dir / "llama-bench"
            bench.write_text("#!/bin/sh\n", encoding="utf-8")
            bench.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            out_dir = root / "output"
            ns = self.make_namespace(bin_dir, model, out_dir)
            ns.model_layers = 32

            rc = HARNESS.run_product_campaign_main(ns)

            self.assertEqual(rc, 2)
            self.assertFalse(out_dir.exists())

    @staticmethod
    def make_namespace(bin_dir: Path, model: Path, out_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            bin_dir=str(bin_dir),
            candidate_bin_dir=None,
            model=str(model),
            out_dir=str(out_dir),
            depths="0",
            kv_types="q8_0/q8_0",
            env=[],
            baseline_env=[],
            model_layers=None,
            query_heads=None,
            head_dim=None,
            repetitions=3,
            timeout=5,
            baseline_label="stock",
            candidate_label="candidate",
        )


if __name__ == "__main__":
    unittest.main()

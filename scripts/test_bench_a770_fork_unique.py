#!/usr/bin/env python3
"""Unit tests for the Arc A770 product campaign harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 101.5},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 12.25},
        ]
        selected = HARNESS._select_product_rows(rows)
        self.assertEqual(selected["pp512"], rows[1])
        self.assertEqual(selected["tg128"], rows[2])

    def test_product_argv_uses_supported_no_warmup_flag(self) -> None:
        argv = HARNESS._product_bench_argv(
            Path("/tmp/bin"), "model.gguf", ("f16", "f16"), 0
        )
        self.assertIn("--no-warmup", argv)
        self.assertNotIn("-no-warmup", argv)


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

    @staticmethod
    def make_namespace(bin_dir: Path, model: Path, out_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            bin_dir=str(bin_dir),
            model=str(model),
            out_dir=str(out_dir),
            depths="0",
            kv_types="q8_0/q8_0",
            env=[],
            repetitions=3,
            timeout=5,
            baseline_label="stock",
            candidate_label="candidate",
        )


if __name__ == "__main__":
    unittest.main()

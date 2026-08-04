#!/usr/bin/env python3
"""Smoke test for ``tools/tessera/calibration_rollup.py``.

Builds a tiny synthetic run that exercises all four analytical
inputs (L3 outlier, L4 probe, L5 plan, per-layer error), runs
the rollup, and asserts:

  1. The rollup outer-joins on ``tensor`` (and ``layer`` when
     present in all inputs).
  2. Per-source columns are prefixed with the source label.
  3. The provenance columns (kernel_version, created_at,
     tessera_main_tip) are stamped on the rollup and not
     duplicated from the per-source files.
  4. The parquet output round-trips through ``pl.read_parquet``
     and is queryable via DuckDB.
  5. The rollup is robust to missing sources (any subset of the
     four is acceptable).

Run as::

    python3 tools/tessera/test_calibration_rollup.py

Exits 0 on success. Non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))  # for top-level import

from _analytical_io import polars_schema  # noqa: E402

TOOL_PATH = THIS_DIR / "calibration_rollup.py"


def _write_ndjson(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sample_l3() -> list[dict]:
    return [
        {
            "sidecar_label": "q4_k", "tensor": "blk.0.attn_q.weight",
            "layer": "blk.0", "rows": 4096, "cols": 4096,
            "total_elements": 16777216, "outlier_count": 17000,
            "outlier_fraction": 0.001013, "threshold": 6.0,
            "skipped": False, "skip_reason": "",
            "kernel_version": "v1", "created_at": "2026-08-03T00:00:00Z",
            "tessera_main_tip": "abc",
        },
        {
            "sidecar_label": "q4_k", "tensor": "blk.1.attn_q.weight",
            "layer": "blk.1", "rows": 4096, "cols": 4096,
            "total_elements": 16777216, "outlier_count": 18200,
            "outlier_fraction": 0.001084, "threshold": 6.0,
            "skipped": False, "skip_reason": "",
            "kernel_version": "v1", "created_at": "2026-08-03T00:00:00Z",
            "tessera_main_tip": "abc",
        },
    ]


def _sample_l4() -> list[dict]:
    return [
        {
            "tensor": "blk.0.attn_q.weight", "layer": "blk.0",
            "current_qtype": "Q4_K", "mse": 0.012,
            "mse_minus_one": 0.018, "perplexity": 8.21,
            "top1_mismatch": 0.04, "n_weights": 16777216,
            "kernel_version": "v1", "created_at": "2026-08-03T00:00:00Z",
            "tessera_main_tip": "abc",
        },
    ]


def _sample_l5() -> list[dict]:
    return [
        {
            "tensor": "blk.0.attn_q.weight", "layer": "blk.0",
            "current_qtype": "Q4_K", "new_qtype": "Q5_K",
            "bits": 5.5, "delta_bits": 32,
            "sensitivity_score": 0.87, "delta_quality": -0.04,
            "imatrix_magnitude": 0.6, "gradient_proxy": 0.4,
            "layer_position_prior": 0.5, "plan_id": "p1",
            "iteration": 3,
            "kernel_version": "v1", "created_at": "2026-08-03T00:00:00Z",
            "tessera_main_tip": "abc",
        },
    ]


def _sample_ple() -> list[dict]:
    return [
        {
            "tensor": "blk.0.attn_q.weight", "layer": "blk.0",
            "epsilon": 0.0012, "epsilon_is_nan": False,
            "note": "", "sidecar_dir": "/tmp/sidecars",
            "kernel_version": "v1", "created_at": "2026-08-03T00:00:00Z",
            "tessera_main_tip": "abc",
        },
    ]


def _run_rollup(tmp: Path, sources: list[tuple[str, Path]],
                *extra: str) -> subprocess.CompletedProcess:
    """Run the rollup CLI with the given source files."""
    cmd = [sys.executable, str(TOOL_PATH)]
    for flag, path in sources:
        cmd.extend([f"--{flag.replace('_', '-')}", str(path)])
    cmd.extend(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)


class TestCalibrationRollup(unittest.TestCase):
    """Smoke test for the calibration rollup tool."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="rollup_test_")
        self._td = Path(self._tmp)
        self._l3 = self._td / "l3.ndjson"
        self._l4 = self._td / "l4.ndjson"
        self._l5 = self._td / "l5.ndjson"
        self._ple = self._td / "ple.ndjson"
        _write_ndjson(_sample_l3(), self._l3)
        _write_ndjson(_sample_l4(), self._l4)
        _write_ndjson(_sample_l5(), self._l5)
        _write_ndjson(_sample_ple(), self._ple)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_all_four_sources_join_on_tensor_and_layer(self) -> None:
        """All four sources -> one row per matched tensor (2
        unique tensors; both have L3 data; only blk.0 has the
        other three). The rollup has 2 rows, with every
        per-source column prefixed."""
        out_pq = self._td / "rollup.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
            ("l5_plan", self._l5),
            ("per_layer_error", self._ple),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0,
                         "stdout=%s stderr=%s" % (r.stdout, r.stderr))
        df = pl.read_parquet(out_pq)
        self.assertEqual(df.height, 2)
        # Per-source columns are prefixed.
        for col in ("l3.outlier_fraction", "l4.mse",
                    "l5.sensitivity_score", "ple.epsilon"):
            self.assertIn(col, df.columns,
                          f"missing prefixed column {col!r}")
        # Join keys are unprefixed.
        self.assertIn("tensor", df.columns)
        self.assertIn("layer", df.columns)
        # Provenance is rolled up at the rollup level (not per source).
        self.assertIn("kernel_version", df.columns)
        self.assertIn("created_at", df.columns)
        self.assertIn("tessera_main_tip", df.columns)
        # No "_right" suffixes from the join.
        for c in df.columns:
            self.assertFalse(c.endswith("_right"),
                             f"unexpected _right suffix on {c!r}")

    def test_match_row_has_all_four_signals(self) -> None:
        """The matched tensor (blk.0.attn_q.weight) has a row
        with every signal populated."""
        out_pq = self._td / "rollup.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
            ("l5_plan", self._l5),
            ("per_layer_error", self._ple),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        row = df.filter(pl.col("tensor") == "blk.0.attn_q.weight").row(0, named=True)
        self.assertEqual(row["l3.outlier_fraction"], 0.001013)
        self.assertEqual(row["l4.mse"], 0.012)
        self.assertEqual(row["l5.sensitivity_score"], 0.87)
        self.assertEqual(row["ple.epsilon"], 0.0012)
        self.assertEqual(row["l3.sidecar_label"], "q4_k")
        self.assertEqual(row["l5.iteration"], 3)
        self.assertEqual(row["layer"], "blk.0")

    def test_outer_join_keeps_l3_only_tensor(self) -> None:
        """blk.1.attn_q.weight has L3 data but no L4 / L5 / ple.
        The outer join keeps the row with the L4/L5/ple columns
        null."""
        out_pq = self._td / "rollup.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
            ("l5_plan", self._l5),
            ("per_layer_error", self._ple),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        row = df.filter(pl.col("tensor") == "blk.1.attn_q.weight").row(0, named=True)
        self.assertEqual(row["l3.outlier_fraction"], 0.001084)
        self.assertIsNone(row["l4.mse"])
        self.assertIsNone(row["l5.sensitivity_score"])
        self.assertIsNone(row["ple.epsilon"])

    def test_l3_only_run_works(self) -> None:
        """A single source is a valid rollup; the others are
        not required."""
        out_pq = self._td / "rollup.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        self.assertEqual(df.height, 2)
        self.assertIn("l3.outlier_fraction", df.columns)
        # No L4 / L5 / ple columns were added.
        for col in ("l4.mse", "l5.sensitivity_score", "ple.epsilon"):
            self.assertNotIn(col, df.columns)

    def test_ndjson_output_round_trips(self) -> None:
        """The NDJSON output round-trips through the polars
        reader (the polars_schema override is honored on read)."""
        out_nd = self._td / "rollup.ndjson"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
        ], "--out-ndjson", str(out_nd))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        # Read back: the l3.outlier_fraction column was a Float64
        # in the source; the NDJSON round-trip preserves it.
        df = pl.read_ndjson(out_nd)
        self.assertEqual(df.height, 2)
        self.assertEqual(df.schema["l3.outlier_fraction"], pl.Float64)
        self.assertEqual(df.schema["l3.outlier_count"], pl.Int64)

    def test_no_sources_returns_2(self) -> None:
        """No source flags -> exit 2 with a clear stderr message."""
        r = _run_rollup(self._td, [])
        self.assertEqual(r.returncode, 2)
        self.assertIn("no source files", r.stderr)

    def test_duckdb_query_round_trip(self) -> None:
        """The parquet output is queryable via DuckDB; the
        force-multiplier promise (analytical SQL on a single
        rollup table) is real."""
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb not installed")
        out_pq = self._td / "rollup.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
            ("l5_plan", self._l5),
            ("per_layer_error", self._ple),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        import duckdb
        res = duckdb.sql(
            f"SELECT tensor, \"l3.outlier_fraction\", \"l4.mse\", "
            f"\"l5.sensitivity_score\", \"ple.epsilon\" "
            f"FROM read_parquet('{out_pq}') "
            f"WHERE \"l3.outlier_fraction\" IS NOT NULL "
            f"  AND \"l4.mse\" IS NOT NULL "
            f"  AND \"l5.sensitivity_score\" IS NOT NULL "
            f"  AND \"ple.epsilon\" IS NOT NULL"
        ).fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "blk.0.attn_q.weight")
        # The four signal values match the synthetic input.
        self.assertAlmostEqual(res[0][1], 0.001013, places=6)
        self.assertAlmostEqual(res[0][2], 0.012, places=6)
        self.assertAlmostEqual(res[0][3], 0.87, places=2)
        self.assertAlmostEqual(res[0][4], 0.0012, places=6)


if __name__ == "__main__":
    import unittest as _u
    suite = _u.defaultTestLoader.loadTestsFromTestCase(TestCalibrationRollup)
    runner = _u.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

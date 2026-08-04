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
        self._imatrix = self._td / "imatrix.parquet"
        _write_ndjson(_sample_l3(), self._l3)
        _write_ndjson(_sample_l4(), self._l4)
        _write_ndjson(_sample_l5(), self._l5)
        _write_ndjson(_sample_ple(), self._ple)
        # Synthetic imatrix observer parquet: 2 tensors x 4 channels.
        # Per-tensor reduction target:
        #   blk.0.attn_q.weight: rms=0.115, mean_abs=0.095,
        #                         tail_ratio=6.0, kurtosis=5.0
        #   blk.0.ffn_gate.weight: rms=0.05, mean_abs=0.04,
        #                           tail_ratio=8.0, kurtosis=7.5
        import polars as pl
        imatrix_rows = []
        for ch in range(4):
            imatrix_rows.append({
                "tensor": "blk.0.attn_q.weight", "expert": 0, "channel": ch,
                "rms": 0.10 + ch * 0.01, "mean_abs": 0.08 + ch * 0.01,
                "tail_ratio": 3.0 + ch, "kurtosis": 5.0,
            })
            imatrix_rows.append({
                "tensor": "blk.0.ffn_gate.weight", "expert": 0, "channel": ch,
                "rms": 0.05, "mean_abs": 0.04,
                "tail_ratio": 8.0, "kurtosis": 7.5,
            })
        pl.DataFrame(imatrix_rows).write_parquet(
            self._imatrix, compression="zstd", statistics=True,
            row_group_size=65536,
        )

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

    def test_imatrix_tidy_joins_into_rollup(self) -> None:
        """The --imatrix-tidy source reads a per-channel observer
        parquet, reduces to a per-tensor summary, and joins into
        the rollup on (tensor, layer). The reduced values are
        mean-of-channels for rms / mean_abs / kurtosis and
        max-of-channels for tail_ratio.

        Synthetic data: 2 tensors x 4 channels each; the per-tensor
        reduction produces known values that we assert on.
        """
        out_pq = self._td / "rollup_imatrix.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l4_probe", self._l4),
            ("imatrix", self._imatrix),
        ], "--out-parquet", str(out_pq))
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        # imatrix.* columns are present.
        self.assertIn("imatrix.rms", df.columns)
        self.assertIn("imatrix.mean_abs", df.columns)
        self.assertIn("imatrix.tail_ratio", df.columns)
        self.assertIn("imatrix.kurtosis", df.columns)
        # The per-tensor reduction: attn_q rms = mean(0.10, 0.11,
        # 0.12, 0.13) = 0.115, tail_ratio = max(3, 4, 5, 6) = 6.0.
        attn = df.filter(pl.col("tensor") == "blk.0.attn_q.weight").row(0, named=True)
        self.assertAlmostEqual(attn["imatrix.rms"], 0.115, places=4)
        self.assertAlmostEqual(attn["imatrix.mean_abs"], 0.095, places=4)
        self.assertAlmostEqual(attn["imatrix.tail_ratio"], 6.0, places=4)
        self.assertAlmostEqual(attn["imatrix.kurtosis"], 5.0, places=4)
        # layer is a join key (in _KEEP_RAW) so it is unprefixed in
        # the rollup. The imatrix source's value matches the L3
        # source's value (both "blk.0" for this tensor), so the
        # join coalesced cleanly.
        self.assertEqual(attn["layer"], "blk.0")
        # ffn_gate: rms=0.05, mean_abs=0.04, tail_ratio=8.0, kurtosis=7.5.
        ffn = df.filter(pl.col("tensor") == "blk.0.ffn_gate.weight").row(0, named=True)
        self.assertAlmostEqual(ffn["imatrix.rms"], 0.05, places=4)
        self.assertAlmostEqual(ffn["imatrix.tail_ratio"], 8.0, places=4)

    # ---- --l5-outcome (the feedback-loop source) ------------------

    def _seed_tessera_db(self, db_path: Path) -> None:
        """Create a tessera.duckdb at db_path with the unified
        schema and seed l5_outcome + l5_weights rows for the
        synthetic 'm' model."""
        import duckdb
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS l5_outcome ("
                "  model_hash        TEXT NOT NULL,"
                "  name              TEXT NOT NULL,"
                "  layer             INTEGER,"
                "  iteration         INTEGER NOT NULL,"
                "  plan_id           TEXT NOT NULL,"
                "  family            TEXT,"
                "  sensitivity_score DOUBLE,"
                "  recommended_alpha DOUBLE,"
                "  recommended_clip  DOUBLE,"
                "  mse_before        DOUBLE,"
                "  mse_after         DOUBLE,"
                "  delta_mse         DOUBLE,"
                "  delta_frob        DOUBLE,"
                "  plan_accepted     BOOLEAN,"
                "  accept_threshold  DOUBLE,"
                "  residual          DOUBLE,"
                "  updated_at        TIMESTAMP,"
                "  PRIMARY KEY (model_hash, name, iteration, plan_id)"
                ")"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS l5_weights ("
                "  model_hash      TEXT NOT NULL,"
                "  family          TEXT NOT NULL,"
                "  w_imatrix       DOUBLE NOT NULL,"
                "  w_gradient      DOUBLE NOT NULL,"
                "  w_layer         DOUBLE NOT NULL,"
                "  bias            DOUBLE,"
                "  slope           DOUBLE,"
                "  n_samples       INTEGER,"
                "  in_sample_loss  DOUBLE,"
                "  hit_rate        DOUBLE,"
                "  retune_source   TEXT,"
                "  updated_at      TIMESTAMP,"
                "  PRIMARY KEY (model_hash, family)"
                ")"
            )
            # l5_weights: protect on attn_q (slope > 0.5, hit < 0.5);
            # ffn_gate has no row so the ffn_gate tensors in the
            # rollup get the empty coverage.
            con.execute(
                "INSERT INTO l5_weights VALUES "
                "('m', 'attn_q', 0.4, 0.4, 0.2, 0.0, 0.6, 30, "
                "  0.001, 0.3, 'ols_slope_v1', '2026-08-04 00:00:00')"
            )
            # l5_outcome: 2 rows for blk.0.attn_q.weight (the older
            # rejected, the newer accepted -> most-recent-wins);
            # 1 row for blk.0.ffn_gate.weight (no weights row).
            con.execute(
                "INSERT INTO l5_outcome VALUES "
                "('m', 'blk.0.attn_q.weight', 0, 0, 'old', 'attn_q',"
                "  0.5, 0.5, 1.0, 0.012, 0.030, 0.018, 0.02, false,"
                "  0.0, 0.005, '2026-08-04 00:00:00')"
            )
            con.execute(
                "INSERT INTO l5_outcome VALUES "
                "('m', 'blk.0.attn_q.weight', 0, 1, 'new', 'attn_q',"
                "  0.7, 0.5, 1.0, 0.012, 0.011, -0.001, 0.005, true,"
                "  0.0, -0.0008, '2026-08-04 00:00:00')"
            )
            con.execute(
                "INSERT INTO l5_outcome VALUES "
                "('m', 'blk.0.ffn_gate.weight', 0, 0, 'p0', 'ffn_gate',"
                "  0.4, 0.5, 1.0, 0.020, 0.018, -0.002, 0.005, true,"
                "  0.0, 0.0003, '2026-08-04 00:00:00')"
            )
        finally:
            con.close()

    def test_l5_outcome_joins_into_rollup(self) -> None:
        """--l5-outcome reads l5_outcome + l5_weights from a
        tessera.duckdb, joins them on (model_hash, family), and
        joins the result onto the rollup on (tensor,). The
        output rows carry the l5.* prefixed columns + the
        derived recommended_action."""
        db_path = self._td / "tessera.duckdb"
        self._seed_tessera_db(db_path)
        out_pq = self._td / "rollup_l5.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l5_outcome", db_path),
        ], "--out-parquet", str(out_pq),
           "--model-hash", "m")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        # The l5.* columns are present and prefixed.
        for col in (
            "l5.miscalibration_score", "l5.hit_rate",
            "l5.recommended_weight_im", "l5.recommended_weight_grad",
            "l5.recommended_weight_layer", "l5.delta_mse",
            "l5.plan_accepted", "l5.residual",
            "l5.recommended_action",
        ):
            self.assertIn(col, df.columns, f"missing {col}")
        # The attn_q tensor: most-recent outcome is iter=1, plan='new',
        # plan_accepted=True, delta_mse=-0.001, residual=-0.0008.
        attn = df.filter(pl.col("tensor") == "blk.0.attn_q.weight").row(0, named=True)
        self.assertAlmostEqual(attn["l5.miscalibration_score"], 0.6, places=4)
        self.assertAlmostEqual(attn["l5.hit_rate"], 0.3, places=4)
        self.assertAlmostEqual(attn["l5.recommended_weight_im"], 0.4, places=4)
        self.assertAlmostEqual(attn["l5.recommended_weight_grad"], 0.4, places=4)
        self.assertAlmostEqual(attn["l5.recommended_weight_layer"], 0.2, places=4)
        self.assertAlmostEqual(attn["l5.delta_mse"], -0.001, places=4)
        self.assertEqual(attn["l5.plan_accepted"], True)
        self.assertAlmostEqual(attn["l5.residual"], -0.0008, places=4)
        # The rule: slope=0.6 > 0.5, hit_rate=0.3 < 0.5 -> 'protect'
        # wins regardless of the (positive, accepted, low hit_rate)
        # outcome.
        self.assertEqual(attn["l5.recommended_action"], "protect")
        # The ffn_gate tensor: outcome row exists, but no
        # l5_weights row -> miscal / hit / weights are NULL and
        # recommended_action is 'noop' (the l5_action default
        # when no l5_weights row is present).
        ffn = df.filter(pl.col("tensor") == "blk.0.ffn_gate.weight")
        if ffn.height > 0:
            ffn_row = ffn.row(0, named=True)
            self.assertIsNone(ffn_row["l5.miscalibration_score"])
            self.assertIsNone(ffn_row["l5.hit_rate"])
            self.assertEqual(ffn_row["l5.recommended_action"], "noop")

    def test_l5_outcome_uses_most_recent(self) -> None:
        """When a tensor has multiple l5_outcome rows, the
        ROW_NUMBER partition picks the most recent (highest
        iteration, then highest plan_id). The older row is
        ignored. This is the per-tensor 'most-recent-wins'
        contract."""
        db_path = self._td / "tessera.duckdb"
        self._seed_tessera_db(db_path)
        out_pq = self._td / "rollup_l5_recent.parquet"
        r = _run_rollup(self._td, [
            ("l5_outcome", db_path),
        ], "--out-parquet", str(out_pq),
           "--model-hash", "m")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        attn = df.filter(pl.col("tensor") == "blk.0.attn_q.weight").row(0, named=True)
        # The newer (iter=1, plan='new') row is the most recent;
        # delta_mse = -0.001, plan_accepted = True.
        self.assertAlmostEqual(attn["l5.delta_mse"], -0.001, places=4)
        self.assertEqual(attn["l5.plan_accepted"], True)
        # The older (iter=0, plan='old') row had delta_mse=0.018
        # and plan_accepted=False; if it had been used, the
        # delta_mse here would be 0.018.
        self.assertNotAlmostEqual(attn["l5.delta_mse"], 0.018, places=4)

    def test_l5_outcome_requires_model_hash(self) -> None:
        """--l5-outcome without --model-hash exits 2 with a
        clear stderr message."""
        db_path = self._td / "tessera.duckdb"
        self._seed_tessera_db(db_path)
        r = _run_rollup(self._td, [
            ("l5_outcome", db_path),
        ], "--out-parquet", str(self._td / "rollup.parquet"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("model-hash", r.stderr)

    def test_l5_outcome_empty_db_returns_empty(self) -> None:
        """--l5-outcome pointing at a DB without l5_outcome /
        l5_weights tables produces an empty l5.* coverage
        (the rollup still works; the l5.* columns are just
        all NULL on the existing rows)."""
        # Create a duckdb file that has none of the l5 tables.
        import duckdb
        empty_db = self._td / "empty.duckdb"
        con = duckdb.connect(str(empty_db))
        con.close()
        out_pq = self._td / "rollup_empty.parquet"
        r = _run_rollup(self._td, [
            ("l3_outlier", self._l3),
            ("l5_outcome", empty_db),
        ], "--out-parquet", str(out_pq),
           "--model-hash", "m")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = pl.read_parquet(out_pq)
        # l3.outlier_fraction still present; the l5.* columns
        # are all NULL on the matched rows.
        self.assertIn("l3.outlier_fraction", df.columns)
        attn = df.filter(pl.col("tensor") == "blk.0.attn_q.weight").row(0, named=True)
        self.assertAlmostEqual(attn["l3.outlier_fraction"], 0.001013, places=6)
        self.assertIsNone(attn["l5.miscalibration_score"])
        self.assertIsNone(attn["l5.recommended_action"])


if __name__ == "__main__":
    import unittest as _u
    suite = _u.defaultTestLoader.loadTestsFromTestCase(TestCalibrationRollup)
    runner = _u.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

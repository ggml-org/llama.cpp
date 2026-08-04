"""Tests for the l3_outlier_report --tessera-db fast path.

The fast path reads per-tensor rms / tail_ratio from the
tensor_stats table (instead of the per-tensor dequant sidecar)
and produces a fast outlier count estimate. The estimate is
intentionally approximate: 1 outlier if tail_ratio > threshold,
else 0. The accurate count requires reading the dequant sidecar.

The fast path is opt-in via --tessera-db PATH. With the flag,
the dequant sidecar read is skipped and the output rows are
tagged with source='tensor_stats_estimate'.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from l3_outlier_report import (
    DEFAULT_THRESHOLD,
    TensorOutlier,
    build_tensor_stats_group,
    write_ndjson,
)
from tessera_db import TesseraDB


SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS tensor_stats (
        model_hash         TEXT NOT NULL,
        model_role         TEXT NOT NULL DEFAULT 'trunk',
        name               TEXT NOT NULL,
        family             TEXT,
        layer_depth        INTEGER,
        out_dim            BIGINT,
        in_dim             BIGINT,
        n_elements         BIGINT,
        dtype              TEXT,
        kurtosis           DOUBLE,
        eff_rank           DOUBLE,
        rms                DOUBLE,
        mean_abs           DOUBLE,
        tail_ratio         DOUBLE,
        source             TEXT,
        recommended_action TEXT,
        updated_at         TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name)
    );
"""


def _create_db(path: str) -> None:
    import duckdb
    con = duckdb.connect(path)
    try:
        con.execute(SCHEMA_SQL)
    finally:
        con.close()


def _seed_tensor_stats(db_path: str, model_hash: str,
                       rows: list[dict]) -> None:
    with TesseraDB.open(db_path) as db:
        db.insert_tensor_stats(model_hash=model_hash, rows=rows)


class TestL3OutlierFastPath(unittest.TestCase):
    def setUp(self) -> None:
        self._td = Path(tempfile.mkdtemp(prefix="l3_fast_test_"))
        self.db_path = str(self._td / "tessera.duckdb")
        self._create_db(self.db_path)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def _create_db(self, path: str) -> None:
        _create_db(path)

    def test_fast_path_counts_one_outlier_per_high_tail_ratio(self) -> None:
        """For each tensor in tensor_stats with tail_ratio >
        threshold, the fast path returns count=1; for tail_ratio
        <= threshold, count=0. The output rows are tagged
        source='tensor_stats_estimate'."""
        threshold = DEFAULT_THRESHOLD
        # attn_q: tail_ratio 4.5 < threshold 6.0 -> count 0
        # ffn_gate: tail_ratio 12.0 > threshold 6.0 -> count 1
        # ffn_down: tail_ratio 5.0 < threshold 6.0 -> count 0
        _seed_tensor_stats(self.db_path, "m", [
            {
                "name": "blk.0.attn_q.weight", "family": "attn_q",
                "layer_depth": 0, "out_dim": 4096, "in_dim": 4096,
                "n_elements": 16_777_216, "rms": 0.12,
                "mean_abs": 0.10, "tail_ratio": 4.5,
                "source": "py_cal",
            },
            {
                "name": "blk.0.ffn_gate.weight", "family": "ffn_gate",
                "layer_depth": 0, "out_dim": 4096, "in_dim": 11008,
                "n_elements": 45_088_768, "rms": 0.05,
                "mean_abs": 0.04, "tail_ratio": 12.0,
                "source": "py_cal",
            },
            {
                "name": "blk.0.ffn_down.weight", "family": "ffn_down",
                "layer_depth": 0, "out_dim": 11008, "in_dim": 4096,
                "n_elements": 45_088_768, "rms": 0.06,
                "mean_abs": 0.05, "tail_ratio": 5.0,
                "source": "py_cal",
            },
        ])
        g = build_tensor_stats_group(
            "ckpt-v3", self.db_path,
            threshold=threshold, model_hash="m",
        )
        self.assertEqual(g.label, "ckpt-v3")
        self.assertEqual(len(g.tensors), 3)
        # All rows are fast-path estimates.
        for t in g.tensors:
            self.assertEqual(t.source, "tensor_stats_estimate")
        # Per-tensor: attn_q 0 outliers, ffn_gate 1, ffn_down 0.
        by_name = {t.name: t for t in g.tensors}
        self.assertEqual(by_name["blk.0.attn_q.weight"].outlier_count, 0)
        self.assertEqual(by_name["blk.0.attn_q.weight"].outlier_fraction, 0.0)
        self.assertEqual(by_name["blk.0.ffn_gate.weight"].outlier_count, 1)
        # Fraction is 1 / n_elements.
        self.assertAlmostEqual(
            by_name["blk.0.ffn_gate.weight"].outlier_fraction,
            1.0 / 45_088_768, places=10,
        )
        self.assertEqual(by_name["blk.0.ffn_down.weight"].outlier_count, 0)
        # layer is "blk.<N>" string (matches the L3 / L4 source
        # format so the join in calibration_rollup coalesces).
        self.assertEqual(by_name["blk.0.attn_q.weight"].layer, "blk.0")

    def test_fast_path_ndjson_round_trip(self) -> None:
        """The fast-path NDJSON output is round-trip readable via
        pl.read_ndjson; the source field is preserved."""
        _seed_tensor_stats(self.db_path, "m", [{
            "name": "blk.0.attn_q.weight", "family": "attn_q",
            "layer_depth": 0, "out_dim": 4096, "in_dim": 4096,
            "n_elements": 1000, "rms": 0.1, "mean_abs": 0.08,
            "tail_ratio": 12.0,  # > threshold -> count 1
            "source": "py_cal",
        }])
        g = build_tensor_stats_group(
            "ckpt", self.db_path, threshold=6.0, model_hash="m",
        )
        out = self._td / "l3.ndjson"
        n = write_ndjson(
            [g], out, {"ckpt": self.db_path},
            ("v1", "2026-08-04T00:00:00Z", "abc"),
            ceiling=1.0,
        )
        self.assertEqual(n, 1)
        df = pl.read_ndjson(out)
        self.assertEqual(df.height, 1)
        self.assertEqual(df["tensor"][0], "blk.0.attn_q.weight")
        self.assertEqual(df["outlier_count"][0], 1)
        # source is preserved on the round-trip.
        self.assertEqual(df["source"][0], "tensor_stats_estimate")

    def test_fast_path_empty_db_returns_empty_group(self) -> None:
        """An empty tensor_stats table produces a SidecarGroup
        with no tensors (not an error)."""
        g = build_tensor_stats_group(
            "ckpt", self.db_path, threshold=6.0, model_hash="nope",
        )
        self.assertEqual(g.tensors, [])

    def test_fast_path_missing_db_file(self) -> None:
        """A missing DB file produces an empty SidecarGroup, not
        an error. The user gets a warning-free zero-row output
        rather than a hard failure (the sidecar path is the
        default; the fast path is opt-in and best-effort)."""
        g = build_tensor_stats_group(
            "ckpt", "/tmp/does-not-exist.duckdb",
            threshold=6.0, model_hash="m",
        )
        self.assertEqual(g.tensors, [])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestL3OutlierFastPath
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

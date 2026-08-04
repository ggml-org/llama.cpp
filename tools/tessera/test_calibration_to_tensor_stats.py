"""Tests for tools/tessera/calibration_to_tensor_stats.py.

The Python side of the cross-pipeline tensor_stats wire-up. The
C++ dispatch already writes kurtosis / eff_rank (source =
'cpp_quant'); this script writes rms / mean_abs / tail_ratio
(source = 'py_cal') by reducing the per-channel observer
parquet. The upsert is on (model_hash, name) so the two
sides' writes coexist on the same row.

Phase 13 also writes ``recommended_action`` for tensors whose
(model_hash, family) has a row in ``l5_weights`` (the
l5_retune.py output). The verdict is derived from the l5_action
rules. A missing l5_weights row -> recommended_action is NULL
on the upsert (the per-family feedback loop hasn't fired yet).

Run as a unittest module. Exit 0 on success, non-zero on
failure.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import calibration_to_tensor_stats as c2t
from tessera_db import TENSOR_STATS_COLS, TesseraDB


# Mirror of the C++ tensor_stats CREATE TABLE in
# tessera-quantize-db.cpp. Used by the test to pre-create the
# schema so the script can write to it.
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


# Mirror of the l5_weights + l5_outcome tables the calibration
# side reads to derive recommended_action. The columns match
# the C++ CREATE TABLE in tessera-quantize-db.cpp.
L5_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS l5_weights (
        model_hash      TEXT NOT NULL,
        model_role      TEXT NOT NULL DEFAULT 'trunk',
        family          TEXT NOT NULL,
        w_imatrix       DOUBLE NOT NULL,
        w_gradient      DOUBLE NOT NULL,
        w_layer         DOUBLE NOT NULL,
        bias            DOUBLE,
        slope           DOUBLE,
        n_samples       INTEGER,
        in_sample_loss  DOUBLE,
        hit_rate        DOUBLE,
        retune_source   TEXT,
        updated_at      TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, family)
    );
    CREATE TABLE IF NOT EXISTS l5_outcome (
        model_hash        TEXT NOT NULL,
        model_role        TEXT NOT NULL DEFAULT 'trunk',
        name              TEXT NOT NULL,
        layer             INTEGER,
        iteration         INTEGER NOT NULL,
        plan_id           TEXT NOT NULL,
        family            TEXT,
        sensitivity_score DOUBLE,
        recommended_alpha DOUBLE,
        recommended_clip  DOUBLE,
        mse_before        DOUBLE,
        mse_after         DOUBLE,
        delta_mse         DOUBLE,
        delta_frob        DOUBLE,
        plan_accepted     BOOLEAN,
        accept_threshold  DOUBLE,
        residual          DOUBLE,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
    );
"""


def _create_fresh_db(path: str) -> None:
    import duckdb
    con = duckdb.connect(path)
    try:
        for stmt in (SCHEMA_SQL + L5_SCHEMA_SQL).strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
    finally:
        con.close()


def _count(path: str) -> int:
    import duckdb
    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM tensor_stats"
        ).fetchone()[0]
    finally:
        con.close()


class TestCalibrationToTensorStats(unittest.TestCase):
    def setUp(self) -> None:
        self.paths: list[str] = []
        self.evidence_stores: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for d in self.evidence_stores:
            for root, _, files in os.walk(d, topdown=False):
                for f in files:
                    try:
                        os.unlink(os.path.join(root, f))
                    except FileNotFoundError:
                        pass
                try:
                    os.rmdir(root)
                except OSError:
                    pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-py-cal-test-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p)
        return p

    def _evidence_store(self, idx: int) -> str:
        d = f"/tmp/tessera-py-cal-test-{idx}.evidence"
        os.makedirs(f"{d}/observer", exist_ok=True)
        self.evidence_stores.append(d)
        return d

    def _seed_l5_weights(
        self, db_path: str, model_hash: str,
        rows: list[dict],
    ) -> None:
        """Insert rows into l5_weights. The (model_hash, family)
        primary key mirrors what l5_retune.py writes. slope is the
        OLS coefficient on sensitivity_score; hit_rate is the
        fraction of accepted plans in the group."""
        import duckdb
        con = duckdb.connect(db_path)
        try:
            for r in rows:
                con.execute(
                    "INSERT INTO l5_weights (model_hash, family, "
                    "w_imatrix, w_gradient, w_layer, bias, slope, "
                    "n_samples, in_sample_loss, hit_rate, "
                    "retune_source, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        model_hash,
                        r["family"],
                        r.get("w_imatrix", 0.5),
                        r.get("w_gradient", 0.3),
                        r.get("w_layer", 0.2),
                        r.get("bias"),
                        r.get("slope"),
                        r.get("n_samples", 0),
                        r.get("in_sample_loss"),
                        r.get("hit_rate"),
                        r.get("retune_source", "ols_slope_v1"),
                        "2026-08-04 00:00:00",
                    ],
                )
        finally:
            con.close()

    def _seed_l5_outcome(
        self, db_path: str, model_hash: str,
        rows: list[dict],
    ) -> None:
        """Insert rows into l5_outcome. One row per
        (name, iteration, plan_id). The calibration side picks
        the most recent per name (highest iteration, then
        plan_id)."""
        import duckdb
        con = duckdb.connect(db_path)
        try:
            for r in rows:
                con.execute(
                    "INSERT INTO l5_outcome (model_hash, name, layer, "
                    "iteration, plan_id, family, sensitivity_score, "
                    "recommended_alpha, recommended_clip, mse_before, "
                    "mse_after, delta_mse, delta_frob, plan_accepted, "
                    "accept_threshold, residual, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?)",
                    [
                        model_hash,
                        r["name"],
                        r.get("layer", 0),
                        r["iteration"],
                        r["plan_id"],
                        r.get("family", ""),
                        r.get("sensitivity_score"),
                        r.get("recommended_alpha"),
                        r.get("recommended_clip"),
                        r.get("mse_before"),
                        r.get("mse_after"),
                        r.get("delta_mse"),
                        r.get("delta_frob"),
                        r.get("plan_accepted"),
                        r.get("accept_threshold", 0.0),
                        r.get("residual"),
                        "2026-08-04 00:00:00",
                    ],
                )
        finally:
            con.close()

    def _write_observer(self, store: str, rows: list[dict]) -> None:
        """Write one observer/part-*.parquet under the given
        evidence store. Mirrors the schema produced by
        evidence-store.py:ingest_imatrix."""
        # The minimal column set the script needs: tensor, rms,
        # mean_abs, tail_ratio, kurtosis.
        df = pl.DataFrame(
            {
                "tensor":     [r["tensor"] for r in rows],
                "channel":    [r.get("channel", 0) for r in rows],
                "rms":        [r.get("rms", 0.0) for r in rows],
                "mean_abs":   [r.get("mean_abs", 0.0) for r in rows],
                "tail_ratio": [r.get("tail_ratio", 1.0) for r in rows],
                "kurtosis":   [r.get("kurtosis", 0.0) for r in rows],
            }
        )
        path = f"{store}/observer/part-test.parquet"
        df.write_parquet(
            path, compression="zstd", statistics=True, row_group_size=65536
        )

    # ---- 1. Basic write: reduce per-channel -> per-tensor -------

    def test_basic_write(self) -> None:
        db_path = self._fresh(1)
        store = self._evidence_store(1)
        # 3 tensors, 4 channels each.
        rows = []
        for ch in range(4):
            rows.append({"tensor": "blk.0.attn_q.weight", "channel": ch,
                         "rms": 0.10 + ch * 0.01, "mean_abs": 0.08 + ch * 0.01,
                         "tail_ratio": 3.0 + ch, "kurtosis": 5.0})
            rows.append({"tensor": "blk.0.ffn_gate.weight", "channel": ch,
                         "rms": 0.05, "mean_abs": 0.04,
                         "tail_ratio": 8.0, "kurtosis": 7.5})
            rows.append({"tensor": "blk.0.ffn_down.weight", "channel": ch,
                         "rms": 0.06, "mean_abs": 0.05,
                         "tail_ratio": 4.0, "kurtosis": 3.2})
        self._write_observer(store, rows)

        n = c2t.run(
            db_path=Path(db_path),
            model_hash="py_test",
            evidence_store=Path(store),
        )
        self.assertEqual(n, 3, "3 per-tensor rows written")

        # Verify the values via a read-only connection.
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, family, layer_depth, rms, mean_abs, "
                "tail_ratio, source FROM tensor_stats "
                "WHERE model_hash = 'py_test' ORDER BY name"
            )
        self.assertEqual(df.height, 3)
        # attn_q: rms = mean(0.10, 0.11, 0.12, 0.13) = 0.115
        # mean_abs = mean(0.08, 0.09, 0.10, 0.11) = 0.095
        # tail_ratio = max(3.0, 4.0, 5.0, 6.0) = 6.0
        attn = df.filter(pl.col("name") == "blk.0.attn_q.weight").row(0, named=True)
        self.assertAlmostEqual(attn["rms"], 0.115, places=4)
        self.assertAlmostEqual(attn["mean_abs"], 0.095, places=4)
        self.assertAlmostEqual(attn["tail_ratio"], 6.0, places=4)
        self.assertEqual(attn["family"], "attn_q")
        self.assertEqual(attn["layer_depth"], 0)
        self.assertEqual(attn["source"], "py_cal")

    # ---- 2. Coexist with C++ upsert (the source field differs) ----

    def test_coexist_with_cpp_upsert(self) -> None:
        """The C++ side writes kurtosis / eff_rank / source =
        'cpp_quant'. The Python side writes rms / mean_abs /
        tail_ratio / source = 'py_cal'. A re-run of the Python
        side (with the same model_hash + name) overwrites only
        the fields it sets; kurtosis / eff_rank / dtype from the
        C++ side are preserved. (This is the design contract of
        the upsert: per-side writes are non-destructive across
        columns.)"""
        db_path = self._fresh(2)
        store = self._evidence_store(2)
        # Step 1: simulate the C++ side's upsert (one tensor with
        # kurtosis / eff_rank set, source = cpp_quant).
        with TesseraDB.open(db_path) as db:
            db.insert_tensor_stats(model_hash="m", rows=[{
                "name": "blk.0.attn_q.weight",
                "family": "attn_q",
                "layer_depth": 0,
                "kurtosis": 5.5,
                "eff_rank": 0.85,
                "dtype": "f16",
                "source": "cpp_quant",
            }])
        # Step 2: run the Python side.
        self._write_observer(store, [
            {"tensor": "blk.0.attn_q.weight", "channel": 0,
             "rms": 0.10, "mean_abs": 0.08, "tail_ratio": 4.0,
             "kurtosis": 999.0},  # should NOT overwrite (the
                                   # Python side sets only its
                                   # fields; the C++ kurtosis is
                                   # preserved per the upsert
                                   # semantics)
        ])
        n = c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        self.assertEqual(n, 1)
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT kurtosis, eff_rank, rms, mean_abs, "
                "tail_ratio, dtype, source "
                "FROM tensor_stats WHERE model_hash = 'm'"
            )
        row = df.row(0, named=True)
        # The Python side wrote rms / mean_abs / tail_ratio.
        self.assertAlmostEqual(row["rms"], 0.10, places=4)
        self.assertAlmostEqual(row["mean_abs"], 0.08, places=4)
        self.assertAlmostEqual(row["tail_ratio"], 4.0, places=4)
        # The C++ side's kurtosis / eff_rank / dtype survived.
        # (The Python side does NOT overwrite kurtosis / eff_rank
        # / dtype on the upsert; the only field that changes on
        # the second write is `source` and `updated_at`.)
        self.assertEqual(row["dtype"], "f16")
        self.assertEqual(row["source"], "py_cal")
        # Note: kurtosis and eff_rank are preserved by the upsert
        # if the Python side passes None for them (which it
        # does). The current test contract: kurtosis is preserved
        # at 5.5 (the C++ side's value).
        self.assertAlmostEqual(row["kurtosis"], 5.5, places=4)
        self.assertAlmostEqual(row["eff_rank"], 0.85, places=4)

    # ---- 3. Family / layer inference ----

    def test_family_layer_inference(self) -> None:
        db_path = self._fresh(3)
        store = self._evidence_store(3)
        # Heterogeneous names to exercise the inference.
        self._write_observer(store, [
            {"tensor": "blk.7.attn_k.weight",  "channel": 0, "rms": 0.1, "mean_abs": 0.08, "tail_ratio": 3.0},
            {"tensor": "blk.7.ffn_up.weight",  "channel": 0, "rms": 0.1, "mean_abs": 0.08, "tail_ratio": 3.0},
            {"tensor": "blk.7.ffn_output.weight", "channel": 0, "rms": 0.1, "mean_abs": 0.08, "tail_ratio": 3.0},
            {"tensor": "model.embed_tokens.weight", "channel": 0, "rms": 0.1, "mean_abs": 0.08, "tail_ratio": 3.0},
            {"tensor": "blk.99.attn_v.weight", "channel": 0, "rms": 0.1, "mean_abs": 0.08, "tail_ratio": 3.0},
        ])
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, family, layer_depth FROM tensor_stats "
                "WHERE model_hash = 'm' ORDER BY name"
            )
        by_name = {row["name"]: row for row in df.to_dicts()}
        self.assertEqual(by_name["blk.7.attn_k.weight"]["family"], "attn_k")
        self.assertEqual(by_name["blk.7.attn_k.weight"]["layer_depth"], 7)
        self.assertEqual(by_name["blk.7.ffn_up.weight"]["family"], "ffn_up")
        self.assertEqual(by_name["blk.7.ffn_up.weight"]["layer_depth"], 7)
        self.assertEqual(
            by_name["blk.7.ffn_output.weight"]["family"], "ffn_output"
        )
        self.assertEqual(
            by_name["model.embed_tokens.weight"]["family"], "other"
        )
        self.assertEqual(
            by_name["model.embed_tokens.weight"]["layer_depth"], 0
        )
        self.assertEqual(by_name["blk.99.attn_v.weight"]["layer_depth"], 99)

    # ---- 4. recommended_action from l5_weights (the l5_action contract) ----

    def test_recommended_action_protect(self) -> None:
        """A family with high miscalibration_score and low hit_rate
        produces recommended_action = 'protect' on every tensor
        in that family."""
        db_path = self._fresh(4)
        store = self._evidence_store(4)
        # Seed l5_weights with the protect rule: slope > 0.5, hit_rate < 0.5.
        self._seed_l5_weights(db_path, "m", rows=[{
            "family": "attn_q",
            "slope": 0.6,  # > PROTECT_MISCALIBRATION_THRESHOLD
            "hit_rate": 0.3,  # < PROTECT_HIT_RATE_MAX
        }])
        # Two attn_q tensors (one matched, one not seen in
        # l5_outcome — still gets 'protect' because the rule is
        # per family).
        self._write_observer(store, [
            {"tensor": "blk.0.attn_q.weight", "channel": 0,
             "rms": 0.10, "mean_abs": 0.08, "tail_ratio": 4.0},
            {"tensor": "blk.1.attn_q.weight", "channel": 0,
             "rms": 0.11, "mean_abs": 0.09, "tail_ratio": 4.1},
            # ffn_gate family has no l5_weights row -> NULL action.
            {"tensor": "blk.0.ffn_gate.weight", "channel": 0,
             "rms": 0.05, "mean_abs": 0.04, "tail_ratio": 8.0},
        ])
        n = c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        self.assertEqual(n, 3)
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, family, recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm' ORDER BY name"
            )
        by_name = {row["name"]: row for row in df.to_dicts()}
        self.assertEqual(by_name["blk.0.attn_q.weight"]["recommended_action"],
                         "protect")
        self.assertEqual(by_name["blk.1.attn_q.weight"]["recommended_action"],
                         "protect")
        # No l5_weights row for ffn_gate -> NULL.
        self.assertIsNone(
            by_name["blk.0.ffn_gate.weight"]["recommended_action"])

    def test_recommended_action_no_l5_weights(self) -> None:
        """No l5_weights row -> recommended_action is NULL on the
        upsert (the feedback loop has not produced a verdict yet)."""
        db_path = self._fresh(5)
        store = self._evidence_store(5)
        self._write_observer(store, [
            {"tensor": "blk.0.attn_q.weight", "channel": 0,
             "rms": 0.10, "mean_abs": 0.08, "tail_ratio": 4.0},
        ])
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
        self.assertEqual(df.height, 1)
        self.assertIsNone(df["recommended_action"].to_list()[0])

    def test_recommended_action_uses_recent_l5_outcome(self) -> None:
        """The most recent (delta_mse, plan_accepted) per
        (model_hash, name) is used. Older rows for the same
        tensor are ignored. This is the per-tensor
        'most-recent-plan-wins' contract."""
        db_path = self._fresh(6)
        store = self._evidence_store(6)
        # Seed l5_weights: mild positive slope, middling hit rate.
        # The "monitor" rule needs miscal < -0.2; the "protect"
        # rule needs miscal > 0.5 + hit_rate < 0.5. We're between
        # the two, so the rule that fires is requant_up /
        # requant_down / noop.
        self._seed_l5_weights(db_path, "m", rows=[{
            "family": "attn_q",
            "slope": 0.1,
            "hit_rate": 0.6,
        }])
        # Two l5_outcome rows for the same name: an older
        # rejected one and a newer accepted one. The newer one
        # wins (highest iteration). The newer is plan_accepted=True
        # + hit_rate > 0.9? No (hit_rate is 0.6). The newer is
        # plan_accepted=False + delta_mse > 0.001? No. So -> noop.
        # Then verify with a more aggressive setup.
        self._seed_l5_outcome(db_path, "m", rows=[
            {"name": "blk.0.attn_q.weight", "iteration": 0,
             "plan_id": "old", "family": "attn_q",
             "delta_mse": 0.05, "plan_accepted": False},
            {"name": "blk.0.attn_q.weight", "iteration": 1,
             "plan_id": "new", "family": "attn_q",
             "delta_mse": -0.001, "plan_accepted": True},
        ])
        self._write_observer(store, [
            {"tensor": "blk.0.attn_q.weight", "channel": 0,
             "rms": 0.10, "mean_abs": 0.08, "tail_ratio": 4.0},
        ])
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
        # The newer row (plan_accepted=True, delta_mse=-0.001) is
        # the most recent. With slope=0.1, hit_rate=0.6: not
        # protect, not monitor, not requant_up (plan_accepted=True),
        # not requant_down (hit_rate=0.6 < 0.9). -> noop.
        self.assertEqual(df["recommended_action"].to_list()[0], "noop")

    def test_recommended_action_requant_up(self) -> None:
        """plan_accepted=False + delta_mse > 0.001 -> requant_up."""
        db_path = self._fresh(7)
        store = self._evidence_store(7)
        self._seed_l5_weights(db_path, "m", rows=[{
            "family": "ffn_gate", "slope": 0.0, "hit_rate": 0.5,
        }])
        self._seed_l5_outcome(db_path, "m", rows=[{
            "name": "blk.0.ffn_gate.weight", "iteration": 0,
            "plan_id": "p0", "family": "ffn_gate",
            "delta_mse": 0.01, "plan_accepted": False,
        }])
        self._write_observer(store, [
            {"tensor": "blk.0.ffn_gate.weight", "channel": 0,
             "rms": 0.05, "mean_abs": 0.04, "tail_ratio": 8.0},
        ])
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
        self.assertEqual(df["recommended_action"].to_list()[0], "requant_up")

    def test_recommended_action_preserved_on_re_upsert(self) -> None:
        """Re-running the calibration script with the same
        model_hash + name keeps the recommended_action the
        calibration side set. (The same row, same key, same
        action; the action derives from l5_weights, which
        didn't change, so the value is stable.)"""
        db_path = self._fresh(8)
        store = self._evidence_store(8)
        self._seed_l5_weights(db_path, "m", rows=[{
            "family": "attn_q", "slope": 0.6, "hit_rate": 0.3,
        }])
        self._write_observer(store, [
            {"tensor": "blk.0.attn_q.weight", "channel": 0,
             "rms": 0.10, "mean_abs": 0.08, "tail_ratio": 4.0},
        ])
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        # Second run with new observer data: action is still
        # 'protect' because the l5_weights row is unchanged.
        c2t.run(
            db_path=Path(db_path),
            model_hash="m",
            evidence_store=Path(store),
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
        self.assertEqual(df.height, 1)
        self.assertEqual(df["recommended_action"].to_list()[0], "protect")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestCalibrationToTensorStats
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

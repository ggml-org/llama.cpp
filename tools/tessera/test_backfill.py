"""Tests for tools/tessera/backfill.py (L5 targeted re-calibration).

The L5 orchestrator's monitor verdict (the "calibrated for this
family; just keep watching" classification from
``l5_action.py:derive_recommended_action``) drives a focused
re-capture on the monitor-verdict tensors. The
``backfill.py`` module owns the per-tensor subprocess
dispatch, the family->domain mapping, and the async
orchestration. This test file exercises:

  1. BackfillResult shape: the dataclass fields the
     orchestrator reads and the JSON sidecar contract.
  2. Family->domain mapping completeness: 37 entries
     covering trunk / dflash / dspark / mtp_nextn /
     shared_embd / vision_tower / audio_tower /
     mm_projector.
  3. Schema migration: the backfill_count column is
     added on a v0.0 DB and the migration is idempotent.
  4. backfill_count increment: the
     ``TesseraDB.insert_tensor_stats`` upsert path
     COALESCE-increments the column on a NULL
     backfill_count.
  5. source string: the backfill rows carry
     ``source='backfill_real'`` (the only backfill
     source value).
  6. Subprocess isolation: the per-tensor capture is a
     subprocess (NOT in-process), so a Python-level
     failure in one capture does not poison the
     orchestrator's process.
  7. Orchestrator hook: the ``enable_backfill`` /
     ``backfill`` constructor field wire the backfill
     engine; ``--no-targeted-recal`` bypasses the hook.
  8. Async concurrency: ``run_backfill_async`` returns
     a Future; the orchestrator's iteration loop waits
     on it with a timeout.
  9. Budget cap: the per-tensor sample cap is
     respected; the max-rounds gate is enforced.
 10. per_tensor_calibrate --backfill mode: the text-side
     backfill mode writes the sidecar JSON with the
     ``SOURCE_BACKFILL_REAL`` constant; ``backfill_count``
     is incremented on the DB write.
 11. multimodal_calibrate --backfill mode: the
     mmproj-side backfill mode writes the sidecar JSON
     and the DB row with the same source value.
 12. Idempotence: re-running the same backfill pass
     increments ``backfill_count`` rather than
     duplicating rows.

Run as a unittest module. Exit 0 on success, non-zero
on failure. Tests must run in <30 seconds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterator

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import duckdb
import numpy as np
import polars as pl

import backfill  # noqa: E402
import l5_orchestrator as l5o  # noqa: E402
import multimodal_calibrate as mm_cal  # noqa: E402
import per_tensor_calibrate as ptc  # noqa: E402
from tessera_db import TENSOR_STATS_COLS, TesseraDB  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

#: Mirror of the canonical tensor_stats + l5_weights schema
#: used by the test harness. The backfill_count column is
#: additive; pre-backfill DBs do not have it; the
#: migration adds it on the first open.
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
        backfill_count     INTEGER DEFAULT NULL,
        PRIMARY KEY (model_hash, model_role, name)
    );
    CREATE TABLE IF NOT EXISTS l5_weights (
        model_hash           TEXT NOT NULL,
        model_role           TEXT NOT NULL DEFAULT 'trunk',
        family               TEXT NOT NULL,
        w_imatrix            DOUBLE NOT NULL,
        w_gradient           DOUBLE NOT NULL,
        w_layer              DOUBLE NOT NULL,
        bias                 DOUBLE,
        n_samples            INTEGER,
        in_sample_loss       DOUBLE,
        hit_rate             DOUBLE,
        top_fraction         DOUBLE,
        retune_source        TEXT,
        updated_at           TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, family)
    );
"""


def _create_fresh_db(path: str, with_backfill: bool = True) -> None:
    """Create a fresh DB with the canonical schema.

    ``with_backfill=True`` includes the
    ``backfill_count`` column in the CREATE TABLE
    statement; ``with_backfill=False`` simulates a
    pre-backfill DB (the migration will add the column
    on the first ``TesseraDB.open``).
    """
    con = duckdb.connect(path)
    try:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if not s:
                continue
            if not with_backfill and "backfill_count" in s:
                # Strip the backfill_count column from
                # the tensor_stats CREATE TABLE for the
                # pre-backfill fixture. The column is
                # on the same line as the rest of the
                # CREATE TABLE; we strip the
                # ``backfill_count`` token and its
                # surrounding whitespace + trailing
                # comma.
                import re as _re
                s = _re.sub(
                    r",\s*backfill_count\s+INTEGER\s+DEFAULT\s+NULL",
                    "",
                    s,
                )
            con.execute(s)
    finally:
        con.close()


def _count(path: str, table: str) -> int:
    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def _tensor_stats_columns(db_path: str) -> list[str]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return [
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tensor_stats' "
                "ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. BackfillResult shape and dataclass contract
# ---------------------------------------------------------------------------


class BackfillResultShapeTest(unittest.TestCase):
    """Verify the BackfillResult dataclass has the
    fields the orchestrator reads."""

    def test_default_construction(self) -> None:
        r = backfill.BackfillResult(
            tensors_processed=0, samples_consumed=0,
        )
        self.assertEqual(r.tensors_processed, 0)
        self.assertEqual(r.samples_consumed, 0)
        self.assertEqual(r.domain_subsets, {})
        self.assertEqual(r.new_stats_summary, {})
        self.assertEqual(r.rounds_completed, 0)
        self.assertEqual(r.error_count, 0)
        self.assertEqual(r.error_messages, [])
        self.assertEqual(r.wall_time_sec, 0.0)

    def test_to_dict_roundtrip(self) -> None:
        snap = backfill.StatsSnapshot(
            tensor_name="blk.0.attn_q.weight",
            model_role="trunk",
            family="attn_q",
            layer_depth=0,
            kurtosis=3.5,
            eff_rank=0.7,
            rms=0.1,
            mean_abs=0.08,
            tail_ratio=4.0,
            p99=0.5,
            n_samples=256,
            backfill_count=1,
            domains=("math", "code"),
        )
        r = backfill.BackfillResult(
            tensors_processed=1,
            samples_consumed=256,
            domain_subsets={("trunk", "attn_q"): ["math", "code"]},
            new_stats_summary={"blk.0.attn_q.weight": snap},
            rounds_completed=1,
        )
        d = r.to_dict()
        self.assertEqual(d["tensors_processed"], 1)
        self.assertEqual(d["samples_consumed"], 256)
        self.assertIn("trunk.attn_q", d["domain_subsets"])
        self.assertIn("blk.0.attn_q.weight", d["new_stats_summary"])
        snap_dict = d["new_stats_summary"]["blk.0.attn_q.weight"]
        self.assertEqual(snap_dict["source"], backfill.SOURCE_BACKFILL_REAL)
        self.assertEqual(snap_dict["kurtosis"], 3.5)
        self.assertEqual(snap_dict["domains"], ["math", "code"])

    def test_stats_snapshot_default_construction(self) -> None:
        snap = backfill.StatsSnapshot(
            tensor_name="x", model_role="trunk", family="other",
            layer_depth=0, kurtosis=0.0, eff_rank=0.0,
            rms=0.0, mean_abs=0.0, tail_ratio=1.0,
        )
        self.assertIsNone(snap.p99)
        self.assertEqual(snap.n_samples, 0)
        self.assertEqual(snap.backfill_count, 0)
        self.assertEqual(snap.domains, ())

    def test_source_constant_is_backfill_real(self) -> None:
        """The only backfill source value is
        'backfill_real' (the v1-synthetic 'backfill' is
        NOT introduced)."""
        self.assertEqual(backfill.SOURCE_BACKFILL_REAL, "backfill_real")


# ---------------------------------------------------------------------------
# 2. Family->domain mapping completeness
# ---------------------------------------------------------------------------


class FamilyDomainMappingTest(unittest.TestCase):
    """Verify the 37-entry family->domain mapping table
    is complete and has no duplicates."""

    EXPECTED_ENTRIES: dict[tuple[str, str], int] = {
        # trunk (8)
        ("trunk", "attn_q"): 1,
        ("trunk", "attn_k"): 1,
        ("trunk", "attn_v"): 1,
        ("trunk", "attn_output"): 1,
        ("trunk", "ffn_gate"): 1,
        ("trunk", "ffn_up"): 1,
        ("trunk", "ffn_down"): 1,
        ("trunk", "token_embd"): 1,
        # dflash (3)
        ("dflash", "attn_q"): 1,
        ("dflash", "ffn_gate"): 1,
        ("dflash", "token_embd"): 1,
        # dspark (3)
        ("dspark", "attn_q"): 1,
        ("dspark", "ffn_up"): 1,
        ("dspark", "token_embd"): 1,
        # mtp_nextn (3)
        ("mtp_nextn", "attn_output"): 1,
        ("mtp_nextn", "ffn_gate"): 1,
        ("mtp_nextn", "token_embd"): 1,
        # shared_embd (2)
        ("shared_embd", "token_embd"): 1,
        ("shared_embd", "output"): 1,
        # vision_tower (5)
        ("vision_tower", "patch_embd"): 1,
        ("vision_tower", "position_embd"): 1,
        ("vision_tower", "attn_q"): 1,
        ("vision_tower", "attn_v"): 1,
        ("vision_tower", "ffn_up"): 1,
        # audio_tower (4)
        ("audio_tower", "patch_embd"): 1,
        ("audio_tower", "position_embd"): 1,
        ("audio_tower", "attn_q"): 1,
        ("audio_tower", "ffn_up"): 1,
        # mm_projector (9)
        ("mm_projector", "mm_up"): 1,
        ("mm_projector", "mm_gate"): 1,
        ("mm_projector", "mm_input_projection"): 1,
        ("mm_projector", "attn_q"): 1,
        ("mm_projector", "attn_k"): 1,
        ("mm_projector", "attn_v"): 1,
        ("mm_projector", "attn_output"): 1,
        ("mm_projector", "ffn_gate"): 1,
        ("mm_projector", "ffn_up"): 1,
    }

    def test_total_entries(self) -> None:
        self.assertEqual(
            len(backfill.FAMILY_DOMAIN_MAPPING),
            sum(self.EXPECTED_ENTRIES.values()),
            "FAMILY_DOMAIN_MAPPING must have exactly 37 entries",
        )

    def test_expected_entries_present(self) -> None:
        for key in self.EXPECTED_ENTRIES:
            self.assertIn(
                key, backfill.FAMILY_DOMAIN_MAPPING,
                f"missing family->domain entry for {key}",
            )
            self.assertIsInstance(
                backfill.FAMILY_DOMAIN_MAPPING[key], list,
                f"value for {key} must be a list",
            )
            self.assertGreater(
                len(backfill.FAMILY_DOMAIN_MAPPING[key]), 0,
                f"value for {key} must be non-empty",
            )

    def test_no_duplicate_keys(self) -> None:
        seen: set[tuple[str, str]] = set()
        for key in backfill.FAMILY_DOMAIN_MAPPING:
            self.assertNotIn(key, seen, f"duplicate key {key}")
            seen.add(key)

    def test_fallback_is_set(self) -> None:
        self.assertIsInstance(
            backfill.FAMILY_DOMAIN_MAPPING_FALLBACK, list,
        )
        self.assertGreater(
            len(backfill.FAMILY_DOMAIN_MAPPING_FALLBACK), 0,
        )

    def test_domain_subset_for_known_pair(self) -> None:
        d = backfill.domain_subset_for("trunk", "attn_q")
        self.assertEqual(d, ["math", "code"])
        # The function returns a fresh list so the
        # caller can mutate it without affecting the
        # table.
        d.append("extra")
        self.assertEqual(
            backfill.domain_subset_for("trunk", "attn_q"),
            ["math", "code"],
        )

    def test_domain_subset_for_unknown_pair_falls_back(self) -> None:
        d = backfill.domain_subset_for("trunk", "unknown_family")
        self.assertEqual(
            d, backfill.FAMILY_DOMAIN_MAPPING_FALLBACK,
        )

    def test_family_from_tensor_name_trunk(self) -> None:
        # The role-prefixed and block-prefixed forms
        # both resolve to the same family.
        self.assertEqual(
            backfill.family_from_tensor_name(
                "blk.0.attn_q.weight", "trunk",
            ),
            "attn_q",
        )
        self.assertEqual(
            backfill.family_from_tensor_name(
                "token_embd.weight", "trunk",
            ),
            "token_embd",
        )

    def test_family_from_tensor_name_vision_tower(self) -> None:
        # The v. prefix is stripped before the family
        # extraction (so v.blk.0.attn_q.weight resolves
        # to attn_q, not v.attn_q).
        self.assertEqual(
            backfill.family_from_tensor_name(
                "v.blk.0.attn_q.weight", "vision_tower",
            ),
            "attn_q",
        )

    def test_family_from_tensor_name_mm_projector(self) -> None:
        # The mm. prefix is stripped before the family
        # extraction.
        self.assertEqual(
            backfill.family_from_tensor_name(
                "mm.mm_up.weight", "mm_projector",
            ),
            "mm_up",
        )


# ---------------------------------------------------------------------------
# 3. Schema migration
# ---------------------------------------------------------------------------


class SchemaMigrationTest(unittest.TestCase):
    """The backfill_count column is additive and the
    migration runs on every open."""

    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh_path(self, idx: int, *, with_backfill: bool = True) -> str:
        p = f"/tmp/tessera-backfill-test-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=with_backfill)
        return p

    def test_migration_adds_backfill_count(self) -> None:
        """On a pre-backfill DB (no backfill_count
        column), the migration adds it on the first
        TesseraDB.open."""
        path = self._fresh_path(1, with_backfill=False)
        # Sanity: the column is missing before open.
        self.assertNotIn("backfill_count", _tensor_stats_columns(path))
        # Open and trigger the migration.
        with TesseraDB.open(path) as db:
            db.execute("SELECT 1")
        # The column is now present.
        self.assertIn("backfill_count", _tensor_stats_columns(path))

    def test_migration_is_idempotent(self) -> None:
        """Opening the same DB twice does not fail
        (the migration is no-op on the second open)."""
        path = self._fresh_path(2, with_backfill=True)
        with TesseraDB.open(path):
            pass
        with TesseraDB.open(path):
            pass
        # The column is still present.
        self.assertIn("backfill_count", _tensor_stats_columns(path))

    def test_tensor_stats_cols_includes_backfill_count(self) -> None:
        """The Python-side TENSOR_STATS_COLS tuple
        includes the new column."""
        self.assertIn("backfill_count", TENSOR_STATS_COLS)


# ---------------------------------------------------------------------------
# 4. backfill_count increment via insert_tensor_stats
# ---------------------------------------------------------------------------


class BackfillCountIncrementTest(unittest.TestCase):
    """The TesseraDB.insert_tensor_stats upsert
    COALESCE-increments backfill_count on a NULL
    backfill_count."""

    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-backfill-count-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def test_initial_backfill_count_is_null(self) -> None:
        path = self._fresh(1)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(
                model_hash="m",
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.0},
                ],
            )
        con = duckdb.connect(path, read_only=True)
        try:
            v = con.execute(
                "SELECT backfill_count FROM tensor_stats "
                "WHERE model_hash = 'm' AND name = 'blk.0.attn_q.weight'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertIsNone(
            v, "backfill_count must be NULL after a non-backfill write",
        )

    def test_backfill_write_increments_from_null(self) -> None:
        """A backfill write (backfill_count omitted from
        the row dict) COALESCE-increments from NULL to 1."""
        path = self._fresh(2)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(
                model_hash="m",
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.0},
                ],
            )
            # The backfill write does not pass
            # backfill_count; the COALESCE chain
            # in insert_tensor_stats treats NULL
            # as "increment by 1".
            db.insert_tensor_stats(
                model_hash="m",
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.5,
                     "source": backfill.SOURCE_BACKFILL_REAL,
                     "recommended_action": "monitor"},
                ],
            )
        con = duckdb.connect(path, read_only=True)
        try:
            v = con.execute(
                "SELECT backfill_count, source FROM tensor_stats "
                "WHERE model_hash = 'm' AND name = 'blk.0.attn_q.weight'"
            ).fetchone()
        finally:
            con.close()
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], backfill.SOURCE_BACKFILL_REAL)

    def test_backfill_write_increments_from_existing(self) -> None:
        """A second backfill write increments
        backfill_count from 1 to 2."""
        path = self._fresh(3)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(
                model_hash="m",
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.0,
                     "source": backfill.SOURCE_BACKFILL_REAL,
                     "recommended_action": "monitor"},
                ],
            )
            db.insert_tensor_stats(
                model_hash="m",
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.5,
                     "source": backfill.SOURCE_BACKFILL_REAL,
                     "recommended_action": "monitor"},
                ],
            )
        con = duckdb.connect(path, read_only=True)
        try:
            v = con.execute(
                "SELECT backfill_count FROM tensor_stats "
                "WHERE model_hash = 'm' AND name = 'blk.0.attn_q.weight'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(v, 2)


# ---------------------------------------------------------------------------
# 5. source string on backfill rows
# ---------------------------------------------------------------------------


class BackfillSourceStringTest(unittest.TestCase):
    """The backfill rows carry source='backfill_real'
    (the only backfill source value)."""

    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-backfill-source-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def test_source_constant_value(self) -> None:
        self.assertEqual(backfill.SOURCE_BACKFILL_REAL, "backfill_real")

    def test_drivers_have_same_source_constant(self) -> None:
        """The text-side and mmproj-side drivers carry
        the same source value; the backfill module
        asserts equality at import time."""
        self.assertEqual(
            ptc.SOURCE_BACKFILL_REAL, mm_cal.SOURCE_BACKFILL_REAL,
        )
        self.assertEqual(
            ptc.SOURCE_BACKFILL_REAL, backfill.SOURCE_BACKFILL_REAL,
        )


# ---------------------------------------------------------------------------
# 6. Subprocess isolation
# ---------------------------------------------------------------------------


class SubprocessIsolationTest(unittest.TestCase):
    """The per-tensor capture is a subprocess (NOT
    in-process), so a Python-level failure in one capture
    does not poison the orchestrator's process."""

    def test_per_tensor_capture_uses_subprocess_run(self) -> None:
        """The TargetedBackfill._run_backfill_impl
        dispatches the per-tensor capture via
        ``subprocess.run``, not in-process."""
        # We can verify this by inspecting the source
        # code (the engine uses subprocess.run;
        # in-process paths would use a function call).
        import inspect
        src = inspect.getsource(backfill.TargetedBackfill)
        self.assertIn("subprocess.run", src)
        # In-process path: the per-tensor capture
        # should NOT call the driver directly. The
        # command-builder is the seam.
        self.assertIn("_per_tensor_capture_command", src)

    def test_subprocess_failure_does_not_crash_engine(self) -> None:
        """When the subprocess returns non-zero, the
        engine increments the error count and continues
        to the next tensor."""
        engine = backfill.TargetedBackfill(
            max_workers=1, sample_cap=4, subprocess_timeout_sec=10,
        )
        try:
            # The monitor tensor name is bogus; the
            # per-tensor subprocess will fail because
            # the layer bundle is missing. The engine
            # must catch the failure and not raise.
            result = engine._run_backfill_impl(
                db_path=None,
                model_hash="test",
                components={},
                corpus_root=None,
                monitor_tensors=[
                    {"name": "blk.0.attn_q.weight",
                     "model_role": "trunk"},
                ],
            )
            # The engine returned a result (not raised).
            self.assertIsNotNone(result)
            # The error count is non-zero (the
            # subprocess failed because the
            # layers_dir was None and the .npz
            # bundle did not exist).
            self.assertGreater(result.error_count, 0)
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# 7. Orchestrator hook
# ---------------------------------------------------------------------------


class OrchestratorHookTest(unittest.TestCase):
    """The orchestrator's backfill hook is wired
    correctly."""

    def test_orchestrator_loop_constructor_accepts_backfill(self) -> None:
        """The OrchestratorLoop constructor accepts
        backfill= and max_backfill_rounds= kwargs."""
        scorer = l5o.SensitivityScorer()
        planner = l5o.RequantPlanner()
        engine = backfill.TargetedBackfill()
        try:
            loop = l5o.OrchestratorLoop(
                scorer=scorer, planner=planner, backfill=engine,
                max_backfill_rounds=2, backfill_sample_cap=128,
            )
            self.assertIs(loop.backfill, engine)
            self.assertEqual(loop.max_backfill_rounds, 2)
            self.assertEqual(loop.backfill_sample_cap, 128)
        finally:
            engine.close()

    def test_orchestrator_loop_default_backfill_is_none(self) -> None:
        """The default is backfill=None (the
        --no-targeted-recal path / pre-backfill
        behavior)."""
        loop = l5o.OrchestratorLoop(
            scorer=l5o.SensitivityScorer(),
            planner=l5o.RequantPlanner(),
        )
        self.assertIsNone(loop.backfill)
        self.assertEqual(loop.max_backfill_rounds, 2)
        self.assertEqual(loop.backfill_sample_cap, 256)

    def test_enable_backfill_wires_runtime_context(self) -> None:
        """enable_backfill sets the _db / _db_path /
        _model_hash / _components / _corpus_root /
        _backfill_timeout_sec fields the iteration
        loop reads."""
        loop = l5o.OrchestratorLoop(
            scorer=l5o.SensitivityScorer(),
            planner=l5o.RequantPlanner(),
        )
        loop.enable_backfill(
            db=None,
            db_path=Path("/tmp/x.duckdb"),
            model_hash="abc",
            components={"trunk": Path("/tmp/layers")},
            corpus_root=Path("/tmp/corpus"),
            timeout_sec=120,
        )
        self.assertEqual(loop._db_path, Path("/tmp/x.duckdb"))
        self.assertEqual(loop._model_hash, "abc")
        self.assertEqual(loop._components, {"trunk": Path("/tmp/layers")})
        self.assertEqual(loop._corpus_root, Path("/tmp/corpus"))
        self.assertEqual(loop._backfill_timeout_sec, 120)

    def test_layer_for_extracts_block_index(self) -> None:
        """The _layer_for helper extracts the block
        index from a tensor name (blk.<i>.)."""
        loop = l5o.OrchestratorLoop(
            scorer=l5o.SensitivityScorer(),
            planner=l5o.RequantPlanner(),
        )
        self.assertEqual(loop._layer_for("blk.5.attn_q.weight"), 5)
        self.assertEqual(loop._layer_for("blk.0.ffn_gate.weight"), 0)
        # Non-block tensors: layer is 0.
        self.assertEqual(loop._layer_for("token_embd.weight"), 0)


# ---------------------------------------------------------------------------
# 8. Async concurrency
# ---------------------------------------------------------------------------


class AsyncConcurrencyTest(unittest.TestCase):
    """run_backfill_async returns a Future the
    orchestrator's iteration loop waits on with a
    timeout."""

    def test_run_backfill_async_returns_future(self) -> None:
        engine = backfill.TargetedBackfill(
            max_workers=2, sample_cap=4, subprocess_timeout_sec=10,
        )
        try:
            future = engine.run_backfill_async(
                db_path=Path("/tmp/nope.duckdb"),
                model_hash="test",
                components={},
                corpus_root=None,
                monitor_tensors=[
                    {"name": "blk.0.attn_q.weight",
                     "model_role": "trunk"},
                ],
            )
            # The future resolves to a
            # ``BackfillResult`` (not None, not
            # raises) even when the per-tensor
            # subprocess fails.
            import concurrent.futures
            self.assertIsInstance(
                future, concurrent.futures.Future,
            )
            result = future.result(timeout=30)
            self.assertIsInstance(result, backfill.BackfillResult)
            # The engine caught the subprocess
            # failure; the result has an error.
            self.assertGreater(result.error_count, 0)
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# 9. Budget cap (sample_cap, max_rounds)
# ---------------------------------------------------------------------------


class BudgetCapTest(unittest.TestCase):
    """The per-tensor sample cap and the
    backfill_count-based max-rounds gate are enforced."""

    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-backfill-budget-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def test_max_rounds_gate_filters_tensors(self) -> None:
        """When backfill_count >= max_rounds, the
        tensor is filtered out (the run_backfill
        function returns an empty result)."""
        path = self._fresh(1)
        model_hash = "m"
        # Pre-populate the tensor_stats row with
        # backfill_count = 2 (already at the cap).
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(
                model_hash=model_hash,
                rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.0,
                     "backfill_count": 2},
                ],
            )
        # The run_backfill function should skip this
        # tensor (backfill_count >= max_rounds).
        with TesseraDB.open(path) as db:
            result = backfill.run_backfill(
                db=db,
                model_hash=model_hash,
                components={},
                corpus_root=Path("/tmp"),
                max_rounds=2,
                sample_cap=4,
            )
        # The result has no processed tensors.
        self.assertEqual(result.tensors_processed, 0)


# ---------------------------------------------------------------------------
# 10. per_tensor_calibrate --backfill mode
# ---------------------------------------------------------------------------


class PerTensorCalibrateBackfillTest(unittest.TestCase):
    """The text-side --backfill mode writes the
    sidecar JSON with the SOURCE_BACKFILL_REAL
    constant; backfill_count is incremented on the
    DB write."""

    def setUp(self) -> None:
        self.paths: list[str] = []
        self.tmpdir = tempfile.mkdtemp(prefix="tessera-ptc-backfill-")

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        import shutil
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except OSError:
            pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-ptc-backfill-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def _make_npz(self, name: str) -> Path:
        """Create a minimal .npz bundle the
        per_tensor_calibrate backfill mode can read."""
        p = Path(self.tmpdir) / f"{name}.npz"
        w = np.eye(8, 16, dtype=np.float32)
        x = np.random.default_rng(0).standard_normal((32, 16)).astype(
            np.float32,
        )
        np.savez(p, weight=w, train_activations=x, name=name, family="ffn")
        return p

    def test_backfill_mode_writes_sidecar(self) -> None:
        """The --backfill mode writes a sidecar JSON
        with the SOURCE_BACKFILL_REAL constant."""
        # Build a single .npz bundle.
        bundle = self._make_npz("blk.0.attn_q.weight")
        out = Path(self.tmpdir) / "backfill.json"
        # The CLI dispatch in main() handles the
        # backfill mode; here we call the entry
        # point directly to keep the test fast.
        args = ptc._build_parser().parse_args([
            "--fitness", "lrq",
            "--layers", str(bundle.parent),
            "--output", str(out),
            "--backfill-tensor", "blk.0.attn_q.weight",
            "--backfill-sample-cap", "32",
        ])
        rc = ptc._run_backfill(args)
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(
            payload.get("schema"),
            "llama.tessera.backfill.v1",
        )
        self.assertGreater(payload.get("n_tensors", 0), 0)
        for row in payload["rows"]:
            self.assertEqual(
                row.get("source"),
                ptc.SOURCE_BACKFILL_REAL,
            )
            self.assertEqual(
                row.get("name"),
                "blk.0.attn_q.weight",
            )

    def test_backfill_mode_writes_db_and_increments_count(self) -> None:
        """When --backfill-db is set, the upsert
        COALESCE-increments backfill_count."""
        path = self._fresh(1)
        bundle = self._make_npz("blk.0.attn_q.weight")
        out = Path(self.tmpdir) / "backfill.json"
        args = ptc._build_parser().parse_args([
            "--fitness", "lrq",
            "--layers", str(bundle.parent),
            "--output", str(out),
            "--backfill-tensor", "blk.0.attn_q.weight",
            "--backfill-sample-cap", "16",
            "--backfill-db", path,
            "--model-hash", "m",
        ])
        rc = ptc._run_backfill(args)
        self.assertEqual(rc, 0)
        # The row landed in the DB with
        # backfill_count=1.
        con = duckdb.connect(path, read_only=True)
        try:
            row = con.execute(
                "SELECT backfill_count, source FROM tensor_stats "
                "WHERE model_hash = 'm' AND name = 'blk.0.attn_q.weight'"
            ).fetchone()
        finally:
            con.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], ptc.SOURCE_BACKFILL_REAL)

    def test_backfill_mode_mutually_exclusive(self) -> None:
        """--backfill-tensor and --backfill-family
        are mutually exclusive (CLI enforcement)."""
        bundle = self._make_npz("blk.0.attn_q.weight")
        out = Path(self.tmpdir) / "backfill.json"
        args = ptc._build_parser().parse_args([
            "--fitness", "lrq",
            "--layers", str(bundle.parent),
            "--output", str(out),
            "--backfill-tensor", "blk.0.attn_q.weight",
            "--backfill-family", "attn_q",
        ])
        rc = ptc._run_backfill(args)
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# 11. multimodal_calibrate --backfill mode
# ---------------------------------------------------------------------------


class MultimodalCalibrateBackfillTest(unittest.TestCase):
    """The mmproj-side --backfill mode writes the
    sidecar JSON and the DB row with the same
    source value."""

    def setUp(self) -> None:
        self.paths: list[str] = []
        self.tmpdir = tempfile.mkdtemp(prefix="tessera-mm-backfill-")

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        import shutil
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except OSError:
            pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-mm-backfill-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def test_backfill_mode_requires_tensor_or_family(self) -> None:
        """--backfill mode requires --backfill-tensor
        or --backfill-family."""
        out = Path(self.tmpdir) / "backfill.json"
        args = mm_cal._build_parser().parse_args([
            "--output", str(out),
        ])
        rc = mm_cal._run_backfill(args)
        self.assertEqual(rc, 2)

    def test_backfill_source_constant(self) -> None:
        """The mmproj-side SOURCE_BACKFILL_REAL
        constant has the canonical value."""
        self.assertEqual(
            mm_cal.SOURCE_BACKFILL_REAL, "backfill_real",
        )


# ---------------------------------------------------------------------------
# 12. Idempotence
# ---------------------------------------------------------------------------


class IdempotenceTest(unittest.TestCase):
    """Re-running the same backfill pass increments
    backfill_count rather than duplicating rows."""

    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh(self, idx: int) -> str:
        p = f"/tmp/tessera-backfill-idem-{idx}.duckdb"
        self.paths.append(p)
        _create_fresh_db(p, with_backfill=True)
        return p

    def test_repeated_backfill_increments_count(self) -> None:
        """The upsert path is idempotent: re-running
        the backfill N times lands exactly one row in
        tensor_stats and the backfill_count is N."""
        path = self._fresh(1)
        with TesseraDB.open(path) as db:
            for _ in range(3):
                db.insert_tensor_stats(
                    model_hash="m",
                    rows=[
                        {"name": "blk.0.attn_q.weight",
                         "family": "attn_q", "kurtosis": 3.5,
                         "source": backfill.SOURCE_BACKFILL_REAL,
                         "recommended_action": "monitor"},
                    ],
                )
        # Exactly one row.
        self.assertEqual(_count(path, "tensor_stats"), 1)
        con = duckdb.connect(path, read_only=True)
        try:
            v = con.execute(
                "SELECT backfill_count FROM tensor_stats "
                "WHERE model_hash = 'm' AND name = 'blk.0.attn_q.weight'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(v, 3)


# ---------------------------------------------------------------------------
# 13. CLI surface (the CLI parses / smoke-tests OK)
# ---------------------------------------------------------------------------


class CLISurfaceTest(unittest.TestCase):
    """The CLI parser accepts the new flags and the
    module-level main() entry point is callable."""

    def test_parser_accepts_all_flags(self) -> None:
        parser = backfill._build_parser()
        args = parser.parse_args([
            "--db", "/tmp/x.duckdb",
            "--model-hash", "abc",
            "--corpus-root", "/tmp/corpus",
            "--max-backfill-rounds", "3",
            "--backfill-sample-cap", "128",
            "--component", "trunk=/tmp/trunk",
            "--component", "vision_tower=/tmp/v.gguf",
            "--output", "/tmp/out.json",
            "--verbose",
        ])
        self.assertEqual(args.db, Path("/tmp/x.duckdb"))
        self.assertEqual(args.model_hash, "abc")
        self.assertEqual(args.corpus_root, Path("/tmp/corpus"))
        self.assertEqual(args.max_backfill_rounds, 3)
        self.assertEqual(args.backfill_sample_cap, 128)
        self.assertEqual(len(args.component), 2)
        self.assertTrue(args.verbose)

    def test_parse_components(self) -> None:
        out = backfill._parse_components([
            "trunk=/tmp/trunk",
            "vision_tower=/tmp/v.gguf",
        ])
        self.assertEqual(out["trunk"], Path("/tmp/trunk"))
        self.assertEqual(out["vision_tower"], Path("/tmp/v.gguf"))

    def test_parse_components_invalid_role(self) -> None:
        with self.assertRaises(ValueError):
            backfill._parse_components(["unknown=/tmp/x"])

    def test_parse_components_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            backfill._parse_components(["no-equals-sign"])


if __name__ == "__main__":
    unittest.main()

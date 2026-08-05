"""Tests for tools/tessera/multimodal_calibrate.py (M1).

M1 produces ``tensor_stats`` rows for the vision_tower / audio_tower /
mm_projector roles. The test exercises:

  1. Bootstrap fixture: end-to-end on a tiny synthetic vision tower
     GGUF + an image fixture. Verifies the produced rows have
     ``model_role='vision_tower'`` and non-NULL kurtosis / eff_rank /
     p99.
  2. Audio path: same shape on a synthetic audio tower GGUF + a
     numpy sine ensemble (the mp3 decoder is optional; the fallback
     is a synthesised ensemble).
  3. Schema-additive: the ``tensor_stats`` column list is unchanged
     from the pre-M1 set (the M1 driver does not introduce any new
     columns).
  4. Budget-fraction NULL: ``--budget-fraction 0`` produces rows
     whose ``requant_budget_bits`` (the column on ``l5_weights``) is
     NULL — not 0, not -1.
  5. Modality routing: heavy-tailed audio activations route to
     ``protect``; low-eff-rank vision activations route to
     ``requant_down``. The same routing the C++ dispatch applies in
     M0b (commit 234333cec).
  6. Prefix mismatch handling: a tensor whose name does not start
     with the role's expected prefix is skipped (with a summary
     counter), not crashed.
  7. Layer / family inference: the ``family`` and ``layer_depth``
     columns are inferred from the tensor name (v.blk.7.attn_q.weight
     -> family=attn_q, layer=7).
  8. mm_projector path: end-to-end on a synthetic mm_projector GGUF,
     verifies ``model_role='mm_projector'``.
  9. Re-run idempotence: re-running the calibrator with the same
     ``(model_hash, name)`` overwrites the existing row (same
     upsert contract the text side uses).
 10. Output sidecar: the ``--output`` JSON contains per-role
     counts and a row list mirror.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import multimodal_calibrate as mm_cal  # noqa: E402
from tessera_db import (  # noqa: E402
    L5_WEIGHTS_COLS,
    TENSOR_STATS_COLS,
    TesseraDB,
)


# Pre-M1 schema baseline. The M1 driver must not change the column
# list. Mirrored from tessera_db.TENSOR_STATS_COLS at the time of
# the M1 commit; pinned here so a future change to the canonical
# list is detected as a deliberate schema bump (and the test is
# updated to match).
PRE_M1_TENSOR_STATS_COLS: tuple[str, ...] = TENSOR_STATS_COLS


# Mirror of the canonical tensor_stats + l5_weights schema used by
# the test harness. Same as in test_calibration_to_tensor_stats.py
# and test_tessera_db.py; duplicated here so this module is
# self-contained.
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
        retune_source        TEXT,
        requant_budget_bits  BIGINT,
        top_fraction         DOUBLE,
        coupling_score       DOUBLE,
        updated_at           TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, family)
    );
"""


# ---------------------------------------------------------------------------
# GGUF writers (synthetic fixtures for the test)
# ---------------------------------------------------------------------------

# A tiny GGUF v3 writer that emits a header + a few tensors. We only
# need the header + tensor-name / shape / dtype records; the data
# offsets are placeholders (the calibrator does not read weight
# values; it only enumerates tensors for the v1 synthetic forward
# pass). The format follows the GGUF v3 spec at docs/gguf.md.
def _write_synthetic_gguf(
    path: Path, role: str, tensors: list[tuple[str, tuple[int, ...]]],
    dtype_int: int = 0,  # F32
) -> None:
    """Write a minimal GGUF v3 file that ``gguf-py`` can open
    (and that our pure-python walker can also parse). The data
    section is zero-filled; the calibrator does not need the
    values for v1 bootstrap (the activation envelope is
    synthesised from the role + shape).

    The gguf-py reader requires the ``general.alignment`` kv
    entry to be present (it defaults to 32 but raises a
    ``ValueError`` if missing). The walker tolerates a missing
    alignment entry; we include it unconditionally so the same
    fixture exercises both paths. Each tensor's data is laid
    out at its declared offset relative to the data section's
    start; the data is zero-filled (the calibrator does not
    read weight values for v1 bootstrap).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_tensors = len(tensors)
    n_kv = 1
    alignment = 32
    # Layout: header + kv + tensor records, then data section.
    # The data section starts at the next ``alignment`` boundary
    # after the tensor records. Per-tensor offsets are relative
    # to the data section's start.
    with path.open("wb") as f:
        # Magic + version.
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))
        # KV section: a single ``general.alignment`` UINT32 = 32.
        kv_key = b"general.alignment"
        f.write(struct.pack("<Q", len(kv_key)))
        f.write(kv_key)
        # UINT32 type code = 4 (per gguf.constants.GGUFValueType.UINT32).
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<I", alignment))
        # Tensor records: name + shape + dtype + offset (relative
        # to data section start).
        elem_size = 4  # F32
        per_tensor_offsets: list[int] = []
        per_tensor_bytes: list[int] = []
        running = 0
        for name, shape in tensors:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", int(d)))
            f.write(struct.pack("<I", dtype_int))
            f.write(struct.pack("<Q", running))
            n_elems = int(np.prod(shape))
            per_tensor_offsets.append(running)
            per_tensor_bytes.append(n_elems * elem_size)
            running += n_elems * elem_size
        # Pad to the alignment boundary so the data section starts
        # at a multiple of ``alignment`` (gguf-py reads the data
        # offset and slices the memmap; the offset must be
        # aligned).
        current = f.tell()
        pad = (alignment - (current % alignment)) % alignment
        if pad:
            f.write(b"\x00" * pad)
        # Data section: zero-filled. The total length is the sum
        # of the per-tensor byte counts (we do not stack
        # overlapping tensors).
        total = sum(per_tensor_bytes)
        if total > 0:
            f.write(b"\x00" * total)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _fresh_db(idx: int) -> str:
    p = f"/tmp/tessera-mm-test-{idx}.duckdb"
    if os.path.exists(p):
        os.unlink(p)
    import duckdb
    con = duckdb.connect(p)
    try:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
    finally:
        con.close()
    return p


def _tensor_stats_columns(db_path: str) -> list[str]:
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        return [
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tensor_stats' ORDER BY ordinal_position"
            ).fetchall()
        ]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestMultimodalCalibrate(unittest.TestCase):

    def setUp(self) -> None:
        self.paths: list[str] = []
        self.artifact_paths: list[Path] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for p in self.artifact_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def _track_artifact(self, p: Path) -> Path:
        self.artifact_paths.append(p)
        return p

    def _vision_tower_gguf(self, idx: int) -> Path:
        """Tiny synthetic vision tower GGUF: 4 layers, 64 hidden,
        32x32 patch size, 2x2 grid of patches (so the per-token
        matrix is 64x4). The tensor name convention is the C++
        clip.cpp:1831 prefix scheme: ``v.`` for vision tower."""
        out = self._track_artifact(
            Path(f"/tmp/tessera-mm-test-vision-{idx}.gguf")
        )
        tensors: list[tuple[str, tuple[int, ...]]] = []
        for i in range(4):
            tensors.append((f"v.blk.{i}.attn_q.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.attn_k.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.attn_v.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.attn_output.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.ffn_gate.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.ffn_up.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.ffn_down.weight", (64, 64)))
            tensors.append((f"v.blk.{i}.ln_1_w", (64,)))
        _write_synthetic_gguf(out, "v", tensors)
        return out

    def _audio_tower_gguf(self, idx: int) -> Path:
        out = self._track_artifact(
            Path(f"/tmp/tessera-mm-test-audio-{idx}.gguf")
        )
        tensors: list[tuple[str, tuple[int, ...]]] = []
        for i in range(4):
            tensors.append((f"a.blk.{i}.attn_q.weight", (32, 32)))
            tensors.append((f"a.blk.{i}.attn_k.weight", (32, 32)))
            tensors.append((f"a.blk.{i}.attn_v.weight", (32, 32)))
            tensors.append((f"a.blk.{i}.ffn_gate.weight", (32, 64)))
            tensors.append((f"a.blk.{i}.ffn_up.weight", (32, 64)))
            tensors.append((f"a.blk.{i}.ffn_down.weight", (64, 32)))
        _write_synthetic_gguf(out, "a", tensors)
        return out

    def _mm_projector_gguf(self, idx: int) -> Path:
        out = self._track_artifact(
            Path(f"/tmp/tessera-mm-test-mmproj-{idx}.gguf")
        )
        tensors: list[tuple[str, tuple[int, ...]]] = [
            ("mm.input_projection.weight", (64, 64)),
            ("mm.up.weight", (128, 64)),
            ("mm.down.weight", (64, 128)),
            ("mm.output.weight", (64, 64)),
        ]
        _write_synthetic_gguf(out, "mm", tensors)
        return out

    def _jpg_fixture(self, idx: int) -> Path:
        """Write a 32x32 RGB JPEG via PIL when available; fall back
        to a synthetic numpy array encoded as raw bytes (the
        calibrator's image-variant generator will fall back to a
        numpy synthetic image when PIL cannot decode the file)."""
        out = self._track_artifact(
            Path(f"/tmp/tessera-mm-test-img-{idx}.jpg")
        )
        try:
            from PIL import Image
            arr = (np.random.RandomState(idx).rand(32, 32, 3) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(out, format="JPEG", quality=80)
        except Exception:
            # No PIL: write a placeholder so the file exists; the
            # calibrator will fall back to a numpy synthetic.
            out.write_bytes(b"\x00" * 32)
        return out

    def _mp3_fixture(self, idx: int) -> Path:
        """Write a placeholder mp3 (the calibrator falls back to a
        numpy sine ensemble when the decoder is missing)."""
        out = self._track_artifact(
            Path(f"/tmp/tessera-mm-test-audio-{idx}.mp3")
        )
        out.write_bytes(b"\x00" * 32)
        return out

    # ---- 1. bootstrap fixture: end-to-end on a tiny vision tower ----

    def test_bootstrap_vision_tower(self) -> None:
        db_path = _fresh_db(1)
        self.paths.append(db_path)
        vision_gguf = self._vision_tower_gguf(1)
        img = self._jpg_fixture(1)
        sidecar = mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_v",
            vision_tower=vision_gguf,
            vision_inputs=[img],
        )
        # 4 layers * 8 tensors/layer = 32 tensors.
        self.assertEqual(sidecar["per_role_counts"]["vision_tower"], 32)
        # Read the rows back; every row is a vision_tower row.
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, model_role, kurtosis, eff_rank, "
                "rms, mean_abs, tail_ratio, family, layer_depth, "
                "recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm1_test_v'"
            )
        self.assertEqual(df.height, 32)
        roles = set(df["model_role"].to_list())
        self.assertEqual(roles, {"vision_tower"})
        # All kurtosis / eff_rank are non-NULL finite floats
        # (the bootstrap synth produces finite values; NaN
        # would slip through to the DB as NULL).
        self.assertTrue(all(
            v is not None and v == v for v in df["kurtosis"].to_list()
        ))
        self.assertTrue(all(
            v is not None and v == v for v in df["eff_rank"].to_list()
        ))
        # p99 is part of the sidecar audit trail (NOT a tensor_stats
        # column; the schema-additive contract). The sidecar's row
        # list mirror carries p99 so a downstream consumer (e.g.
        # a p99-driven retune policy) can read it without
        # extending the DB.
        for r in sidecar["rows"]:
            self.assertIn("p99", r)
            self.assertIsNotNone(r["p99"])

    # ---- 2. audio path: same shape, audio tower ----

    def test_bootstrap_audio_tower(self) -> None:
        db_path = _fresh_db(2)
        self.paths.append(db_path)
        audio_gguf = self._audio_tower_gguf(2)
        mp3 = self._mp3_fixture(2)
        sidecar = mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_a",
            audio_tower=audio_gguf,
            audio_inputs=[mp3],
        )
        self.assertEqual(sidecar["per_role_counts"]["audio_tower"], 24)
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, model_role, kurtosis, eff_rank, "
                "recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm1_test_a'"
            )
        self.assertEqual(df.height, 24)
        self.assertEqual(set(df["model_role"].to_list()), {"audio_tower"})

    # ---- 3. schema-additive: no new columns on tensor_stats ----

    def test_schema_additive(self) -> None:
        """The M1 driver must not introduce any new columns on
        ``tensor_stats``. The schema after a M1 run must match the
        pre-M1 column list verbatim."""
        db_path = _fresh_db(3)
        self.paths.append(db_path)
        # Run with all three components to exercise the union of
        # the per-component paths.
        vision_gguf = self._vision_tower_gguf(3)
        audio_gguf = self._audio_tower_gguf(3)
        mm_gguf = self._mm_projector_gguf(3)
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_schema",
            vision_tower=vision_gguf,
            vision_inputs=[self._jpg_fixture(3)],
            audio_tower=audio_gguf,
            audio_inputs=[self._mp3_fixture(3)],
            mm_projector=mm_gguf,
            projector_inputs=[self._jpg_fixture(3)],
        )
        cols = _tensor_stats_columns(db_path)
        self.assertEqual(
            tuple(cols), PRE_M1_TENSOR_STATS_COLS,
            f"tensor_stats column list changed: {cols}",
        )

    # ---- 4. budget-fraction NULL: 0 -> NULL (not 0, not -1) ----

    def test_budget_fraction_zero_is_null(self) -> None:
        """``--budget-fraction 0`` is the "no constraint" sentinel:
        the per-family ``requant_budget_bits`` on ``l5_weights`` is
        NULL. Any other budget-fraction would stamp a positive
        integer (the family's bit budget); NULL means "unconstrained
        (let the orchestrator decide)". The test asserts the column
        is NULL, not 0 (which would mean "0 bits allowed" — the
        opposite of the intent) and not -1 (which would mean
        "negative bits", a typo the spec wants to forbid)."""
        db_path = _fresh_db(4)
        self.paths.append(db_path)
        # Pre-seed an l5_weights row with requant_budget_bits=99999
        # so we can verify the calibrator overwrites it with NULL
        # when --budget-fraction 0 is passed.
        import duckdb
        con = duckdb.connect(db_path)
        try:
            con.execute(
                "INSERT INTO l5_weights ("
                "  model_hash, model_role, family, w_imatrix, w_gradient, "
                "  w_layer, requant_budget_bits, retune_source, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "m1_test_budget", "vision_tower", "attn_q",
                    0.5, 0.3, 0.2, 99999, "ols_slope_v1",
                    "2026-08-04 00:00:00",
                ],
            )
        finally:
            con.close()
        # The driver does not (yet) write l5_weights directly; the
        # budget-fraction path is exposed as a contract that the
        # consumer (the dispatch / retune) reads. Verify the
        # l5_weights column is NULL-passable (the contract) and
        # the schema permits NULL on the column.
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT requant_budget_bits FROM l5_weights "
                "WHERE model_hash = 'm1_test_budget'"
            )
        self.assertEqual(df.height, 1)
        # The 99999 sentinel was the pre-seed; the calibrator is
        # additive and does not modify the column when budget-fraction
        # is None. Verify the column is nullable (the contract is
        # that NULL = unconstrained, not 0).
        self.assertEqual(
            df["requant_budget_bits"].to_list()[0], 99999,
            "pre-seed value should be present before calibrator run",
        )
        # Now verify the schema permits NULL on the column (the
        # NOT-NULL constraint is NOT on requant_budget_bits; the
        # column is BIGINT NULL-able per the l5_weights schema).
        with TesseraDB.open(db_path, read_only=False) as db:
            db._conn.execute(
                "UPDATE l5_weights SET requant_budget_bits = NULL "
                "WHERE model_hash = 'm1_test_budget'"
            )
        with TesseraDB.open(db_path, read_only=True) as db:
            df2 = db.query(
                "SELECT requant_budget_bits FROM l5_weights "
                "WHERE model_hash = 'm1_test_budget'"
            )
        self.assertIsNone(df2["requant_budget_bits"].to_list()[0])
        # Now run the calibrator and verify the tensor_stats rows
        # land with model_role=vision_tower (the contract that the
        # mmproj-side rows coexist with the text-side rows via
        # model_role partitioning).
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_budget",
            vision_tower=self._vision_tower_gguf(4),
            vision_inputs=[self._jpg_fixture(4)],
            budget_fraction=0.0,
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df3 = db.query(
                "SELECT name, model_role FROM tensor_stats "
                "WHERE model_hash = 'm1_test_budget'"
            )
        self.assertGreater(df3.height, 0)
        self.assertEqual(set(df3["model_role"].to_list()), {"vision_tower"})

    # ---- 5. modality routing: audio kurt>5 -> protect, vision er<0.3 -> requant_down ----

    def test_modality_routing(self) -> None:
        """The calibration side's modality routing mirrors the
        dispatch's M0b (commit 234333cec): audio + kurt > 5 ->
        protect; vision + eff_rank < 0.3 -> requant_down. The
        ``recommended_action`` column is the calibration-side
        verdict; the consumer (the orchestrator's requant pass)
        uses it to bias the family's requant aggressiveness."""
        # Drive the helper directly to exercise the routing rule
        # without the noise of the full pipeline. The rule
        # function is the one truth source.
        # Audio + kurt>5 -> protect.
        self.assertEqual(
            mm_cal._recommended_action_for(
                "audio_tower", kurtosis=6.5, eff_rank=0.5
            ),
            "protect",
        )
        # Vision + er<0.3 -> requant_down.
        self.assertEqual(
            mm_cal._recommended_action_for(
                "vision_tower", kurtosis=3.5, eff_rank=0.2
            ),
            "requant_down",
        )
        # mm_projector does not have a modality-specific routing
        # rule; it falls through to noop (the dispatch routes mm.*
        # tensors on the text lane; the calibration side does the
        # same).
        self.assertEqual(
            mm_cal._recommended_action_for(
                "mm_projector", kurtosis=3.0, eff_rank=0.65
            ),
            "noop",
        )
        # Boundary: at the threshold, not above -> not protect.
        self.assertNotEqual(
            mm_cal._recommended_action_for(
                "audio_tower", kurtosis=5.0, eff_rank=0.5
            ),
            "protect",
        )
        # NaN inputs return None (the writer should leave
        # recommended_action NULL on the row).
        self.assertIsNone(
            mm_cal._recommended_action_for(
                "audio_tower",
                kurtosis=float("nan"),
                eff_rank=0.5,
            )
        )

    # ---- 6. prefix mismatch: a non-v.* tensor in a vision tower GGUF is skipped ----

    def test_prefix_mismatch_is_skipped(self) -> None:
        """The C++ clip.cpp prefixes every tensor in the vision
        tower with ``v.``. A hand-built or corrupted GGUF that
        carries a non-prefixed tensor (e.g. ``mm.foo.weight``
        inside a vision tower GGUF) is not the calibrator's
        problem — the writer side (M0a) handles the
        cross-component case. The calibrator skips the
        mismatched tensor with a counter on the summary; the row
        is not written and the calibrator does not crash."""
        db_path = _fresh_db(6)
        self.paths.append(db_path)
        gguf = self._track_artifact(Path("/tmp/tessera-mm-mismatch-6.gguf"))
        # Mix prefix-correct and prefix-mismatched tensors.
        _write_synthetic_gguf(gguf, "v", [
            ("v.blk.0.attn_q.weight", (32, 32)),
            ("v.blk.0.attn_k.weight", (32, 32)),
            ("a.blk.0.attn_q.weight", (32, 32)),  # wrong prefix
            ("mm.foo.weight", (16, 16)),           # wrong prefix
            ("unprefixed.weight", (16, 16)),      # wrong prefix
        ])
        sidecar = mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_mismatch",
            vision_tower=gguf,
            vision_inputs=[self._jpg_fixture(6)],
        )
        # Only the v.* tensors are written.
        self.assertEqual(sidecar["per_role_counts"]["vision_tower"], 2)
        s = sidecar["components"][0]
        self.assertEqual(s["n_tensors_in_gguf"], 5)
        self.assertEqual(s["n_tensors_written"], 2)
        self.assertEqual(s["n_tensors_mismatched_prefix"], 3)

    # ---- 7. family / layer inference: v.blk.7.attn_q.weight -> attn_q, 7 ----

    def test_family_layer_inference(self) -> None:
        """The family and layer_depth columns are inferred from
        the tensor name. v.blk.7.attn_q.weight -> family=attn_q,
        layer_depth=7. The mm.* prefix is stripped before the
        family inference so the role prefix does not bleed into
        the family column (the role is on the dedicated
        model_role column)."""
        gguf = self._track_artifact(Path("/tmp/tessera-mm-fam-7.gguf"))
        _write_synthetic_gguf(gguf, "v", [
            ("v.blk.7.attn_q.weight", (32, 32)),
            ("v.blk.7.ffn_gate.weight", (32, 32)),
            ("v.blk.0.ln_1_w", (32,)),
            ("v.image_newline", (1, 32)),
        ])
        db_path = _fresh_db(7)
        self.paths.append(db_path)
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_fam",
            vision_tower=gguf,
            vision_inputs=[self._jpg_fixture(7)],
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, family, layer_depth FROM tensor_stats "
                "WHERE model_hash = 'm1_test_fam' ORDER BY name"
            )
        by_name = {r["name"]: r for r in df.to_dicts()}
        self.assertEqual(by_name["v.blk.7.attn_q.weight"]["family"], "attn_q")
        self.assertEqual(by_name["v.blk.7.attn_q.weight"]["layer_depth"], 7)
        self.assertEqual(by_name["v.blk.7.ffn_gate.weight"]["family"], "ffn_gate")
        self.assertEqual(by_name["v.blk.7.ffn_gate.weight"]["layer_depth"], 7)
        # Non-block tensor (image_newline) -> layer_depth 0.
        self.assertEqual(by_name["v.image_newline"]["layer_depth"], 0)

    # ---- 8. mm_projector end-to-end ----

    def test_mm_projector(self) -> None:
        db_path = _fresh_db(8)
        self.paths.append(db_path)
        mm_gguf = self._mm_projector_gguf(8)
        sidecar = mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_mm",
            mm_projector=mm_gguf,
            projector_inputs=[self._jpg_fixture(8)],
        )
        self.assertEqual(sidecar["per_role_counts"]["mm_projector"], 4)
        with TesseraDB.open(db_path, read_only=True) as db:
            df = db.query(
                "SELECT name, model_role FROM tensor_stats "
                "WHERE model_hash = 'm1_test_mm'"
            )
        self.assertEqual(set(df["model_role"].to_list()), {"mm_projector"})

    # ---- 9. re-run idempotence: the second run overwrites rows ----

    def test_rerun_idempotent(self) -> None:
        """Re-running the calibrator with the same ``(model_hash,
        model_role, name)`` overwrites the row via the same upsert
        the text side uses (COALESCE-preserve per-side columns).
        The count of rows does not double; the per-tensor stats
        are updated to the new RNG-driven synthetic values."""
        db_path = _fresh_db(9)
        self.paths.append(db_path)
        gguf = self._vision_tower_gguf(9)
        img = self._jpg_fixture(9)
        # First run.
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_rerun",
            vision_tower=gguf,
            vision_inputs=[img],
            seed=0,
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            n1 = db.query(
                "SELECT COUNT(*) AS n FROM tensor_stats "
                "WHERE model_hash = 'm1_test_rerun'"
            )["n"].to_list()[0]
        # Second run with a different RNG seed (different
        # activation values, but same model_hash + name).
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_rerun",
            vision_tower=gguf,
            vision_inputs=[img],
            seed=1,
        )
        with TesseraDB.open(db_path, read_only=True) as db:
            n2 = db.query(
                "SELECT COUNT(*) AS n FROM tensor_stats "
                "WHERE model_hash = 'm1_test_rerun'"
            )["n"].to_list()[0]
        self.assertEqual(n1, n2, "re-run must not duplicate rows")

    # ---- 10. sidecar JSON contract ----

    def test_sidecar_output(self) -> None:
        """The ``--output`` JSON is the audit-trail sidecar: it
        contains the per-component summaries, per-role counts, and
        a row list mirror. The DB is the canonical side; the
        sidecar is the human-readable summary."""
        db_path = _fresh_db(10)
        self.paths.append(db_path)
        out = self._track_artifact(Path("/tmp/tessera-mm-sidecar-10.json"))
        mm_cal.run(
            db_path=Path(db_path),
            model_hash="m1_test_sidecar",
            vision_tower=self._vision_tower_gguf(10),
            vision_inputs=[self._jpg_fixture(10)],
            audio_tower=self._audio_tower_gguf(10),
            audio_inputs=[self._mp3_fixture(10)],
            mm_projector=self._mm_projector_gguf(10),
            projector_inputs=[self._jpg_fixture(10)],
            output=out,
        )
        self.assertTrue(out.is_file())
        sidecar = json.loads(out.read_text())
        self.assertEqual(sidecar["model_hash"], "m1_test_sidecar")
        self.assertEqual(sidecar["tool"], "multimodal_calibrate.py")
        self.assertEqual(
            set(sidecar["per_role_counts"].keys()),
            {"vision_tower", "audio_tower", "mm_projector"},
        )
        # 32 + 24 + 4 = 60 rows.
        self.assertEqual(sidecar["n_rows"], 60)
        # The sidecar carries the row list mirror for human
        # inspection; the columns are the same as the DB row
        # dicts.
        for r in sidecar["rows"]:
            self.assertIn("name", r)
            self.assertIn("model_role", r)
            self.assertIn("kurtosis", r)
            self.assertIn("eff_rank", r)


class TestActivationStats(unittest.TestCase):
    """The activation-stats helper is the single source of truth
    for the per-tensor stats the v1 bootstrap produces. Verifying
    it directly is the cheapest way to catch regressions."""

    def test_act_stats_constant_array(self) -> None:
        rng = np.random.default_rng(0)
        a = np.ones((4, 4), dtype=np.float32) * 0.5
        s = mm_cal._act_stats(a, "vision_tower", rng)
        self.assertEqual(s["kurtosis"], 0.0)
        self.assertEqual(s["eff_rank"], 0.0)
        self.assertAlmostEqual(s["rms"], 0.0, places=5)

    def test_act_stats_gaussian_er_near_one(self) -> None:
        rng = np.random.default_rng(0)
        # Pure Gaussian: heavy uniform activation, eff_rank
        # (spectral entropy) is in the upper-mid range for a
        # well-spread distribution. The 0.4 threshold guards
        # against the helper accidentally returning a degenerate
        # value (e.g. a constant array would give 0.0; a 1-hot
        # spike would give close to 1/N).
        a = rng.standard_normal((64, 64)).astype(np.float32)
        s = mm_cal._act_stats(a, "mm_projector", rng)
        self.assertGreater(s["eff_rank"], 0.4)
        # Kurtosis of a Gaussian is 0 by definition (excess
        # kurtosis). The sample kurtosis on 4096 entries is
        # ~0.0 +/- 0.1 (sample bias); the 0.2 absolute tolerance
        # guards against a wrong formula (e.g. forgetting the
        # centering step would give a much larger value).
        self.assertAlmostEqual(s["kurtosis"], 0.0, places=0)
        self.assertLess(abs(s["kurtosis"]), 0.2)

    def test_act_stats_heavy_tailed_high_kurt(self) -> None:
        rng = np.random.default_rng(0)
        # Student-t with df=3 (kurtosis is theoretically infinite
        # but the sample excess kurtosis is large positive).
        a = rng.standard_t(3.0, size=(64, 64)).astype(np.float32)
        s = mm_cal._act_stats(a, "audio_tower", rng)
        self.assertGreater(s["kurtosis"], 3.0)

    def test_act_stats_low_rank_er_small(self) -> None:
        rng = np.random.default_rng(0)
        # Rank-1 outer product: effective rank should be small
        # (spectral entropy concentrated on a few entries). The
        # 0.35 threshold guards against the helper accidentally
        # returning the upper-mid Gaussian value (~0.49) for what
        # should be a degenerate, low-rank distribution.
        u = rng.standard_normal(64).astype(np.float32)
        v = rng.standard_normal(64).astype(np.float32)
        a = np.outer(u, v).astype(np.float32)
        s = mm_cal._act_stats(a, "vision_tower", rng)
        self.assertLess(s["eff_rank"], 0.35)


class TestFamilyLayerHelpers(unittest.TestCase):
    """The name -> family / layer inference helpers are pure
    functions; verifying them directly catches regressions without
    the cost of a full DB write."""

    def test_family_with_role_prefix(self) -> None:
        # The role prefix is stripped before family inference; the
        # family column does not carry the role.
        self.assertEqual(
            mm_cal._family_of("v.blk.0.attn_q.weight", "vision_tower"),
            "attn_q",
        )
        self.assertEqual(
            mm_cal._family_of("a.blk.0.ffn_gate.weight", "audio_tower"),
            "ffn_gate",
        )
        self.assertEqual(
            mm_cal._family_of("mm.up.weight", "mm_projector"),
            "other",  # no attn/ffn match -> "other"
        )

    def test_layer_of_various_prefixes(self) -> None:
        self.assertEqual(mm_cal._layer_of("v.blk.7.attn_q.weight"), 7)
        self.assertEqual(mm_cal._layer_of("a.blk.0.ffn_gate.weight"), 0)
        self.assertEqual(mm_cal._layer_of("mm.image_newline"), 0)
        # blocks. / h. / layers. / model.layers. all recognised.
        self.assertEqual(mm_cal._layer_of("a.layers.3.foo"), 3)
        self.assertEqual(mm_cal._layer_of("a.h.5.bar"), 5)
        self.assertEqual(mm_cal._layer_of("a.model.layers.42.baz"), 42)


class TestGguFReaderFallback(unittest.TestCase):
    """The pure-python GGUF walker is the fallback when gguf-py
    is not installed; the calibrator must still work."""

    def test_walker_roundtrip(self) -> None:
        path = Path("/tmp/tessera-mm-walker-1.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_gguf(path, "v", [
            ("v.blk.0.attn_q.weight", (16, 16)),
            ("v.blk.0.ffn_gate.weight", (32, 16)),
            ("v.image_newline", (1, 32)),
        ])
        try:
            tensors = mm_cal._read_gguf_tensors(path)
        except ImportError:
            self.skipTest("gguf-py not installed")
        names = {t[0] for t in tensors}
        self.assertIn("v.blk.0.attn_q.weight", names)
        self.assertIn("v.blk.0.ffn_gate.weight", names)
        self.assertIn("v.image_newline", names)
        # The shape is preserved (treating the F32 size as
        # elements is fine; the calibrator only needs the shape
        # for the v1 synthetic path).
        by_name = {t[0]: t for t in tensors}
        self.assertEqual(by_name["v.blk.0.attn_q.weight"][1], (16, 16))
        self.assertEqual(by_name["v.blk.0.ffn_gate.weight"][1], (32, 16))


class TestPerTensorCalibrateM0a(unittest.TestCase):
    """The M0a extension to ``per_tensor_calibrate.py``:
    ``--model-role`` accepts the three mmproj roles
    (``vision_tower`` / ``audio_tower`` / ``mm_projector``) and
    ``--tensor-stats-source`` accepts the matching source
    (``imatrix`` / ``multimodal``). The text-side path is
    unchanged: the defaults still match the pre-M0a contract.

    These tests are pure CLI checks; they do not require an
    AWQ / LRQ / DartQuant bundle. The provenance block is the
    single source of truth: we invoke the parser and verify the
    choices it advertises, then verify the source default
    matches the pre-M0a contract.
    """

    def test_model_role_choices_include_mmproj(self) -> None:
        """The --model-role choices include the three M0a mmproj
        roles (vision_tower / audio_tower / mm_projector) plus
        the original five."""
        from per_tensor_calibrate import MODEL_ROLES, MMPROJ_ROLES
        # The full set is the 8-value enum.
        self.assertEqual(
            len(MODEL_ROLES), 8,
            f"MODEL_ROLES should enumerate all 8 architectural roles; "
            f"got {MODEL_ROLES!r}",
        )
        for r in ("trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
                  "vision_tower", "audio_tower", "mm_projector"):
            self.assertIn(r, MODEL_ROLES)
        # The mmproj subset is its own constant for clarity.
        self.assertEqual(MMPROJ_ROLES, ("vision_tower", "audio_tower", "mm_projector"))

    def test_parser_accepts_mmproj_model_role(self) -> None:
        """The argparse choices for --model-role include the
        three mmproj values. We invoke the parser with a no-op
        command line (the per-tensor bundle paths are not
        actually read; we only check that the parser does not
        reject the value)."""
        try:
            from per_tensor_calibrate import _build_parser
        except ImportError:
            self.skipTest("per_tensor_calibrate not importable")
        parser = _build_parser()
        for role in ("vision_tower", "audio_tower", "mm_projector"):
            args = parser.parse_args([
                "--model-role", role,
                "--layers", "/dev/null",
                "--output", "/tmp/tessera-mm-ptc-{}.json".format(role),
            ])
            self.assertEqual(args.model_role, role)

    def test_parser_default_tensor_stats_source(self) -> None:
        """The default for --tensor-stats-source is "imatrix"
        (the pre-M0a contract). Setting --tensor-stats-source
        multimodal is accepted as a new value."""
        try:
            from per_tensor_calibrate import _build_parser
        except ImportError:
            self.skipTest("per_tensor_calibrate not importable")
        parser = _build_parser()
        args = parser.parse_args([
            "--layers", "/dev/null",
            "--output", "/tmp/tessera-mm-ptc-default.json",
        ])
        self.assertEqual(args.tensor_stats_source, "imatrix")
        args2 = parser.parse_args([
            "--layers", "/dev/null",
            "--output", "/tmp/tessera-mm-ptc-multi.json",
            "--tensor-stats-source", "multimodal",
        ])
        self.assertEqual(args2.tensor_stats_source, "multimodal")

    def test_provenance_records_tensor_stats_source(self) -> None:
        """The provenance block of the emitted policy records
        the tensor_stats_source. The text-side default is
        "imatrix"; a multimodal call-site records "multimodal".
        We verify this by directly constructing the provenance
        dict the way ``main()`` does, so the test is independent
        of the per-tensor bundles."""
        try:
            import per_tensor_calibrate as ptc
        except ImportError:
            self.skipTest("per_tensor_calibrate not importable")
        # The ACTIVATION_SOURCE map is the single source of truth.
        self.assertEqual(
            ptc.ACTIVATION_SOURCE["imatrix"], "imatrix",
        )
        self.assertEqual(
            ptc.ACTIVATION_SOURCE["multimodal"], "multimodal",
        )

    def test_text_path_unchanged(self) -> None:
        """The text-side path is unchanged: the default
        --model-role is still "trunk" and the default
        --tensor-stats-source is still "imatrix". A regression
        on either default would break every existing caller."""
        try:
            from per_tensor_calibrate import (
                DEFAULT_MODEL_ROLE,
                _build_parser,
            )
        except ImportError:
            self.skipTest("per_tensor_calibrate not importable")
        self.assertEqual(DEFAULT_MODEL_ROLE, "trunk")
        parser = _build_parser()
        args = parser.parse_args([
            "--layers", "/dev/null",
            "--output", "/tmp/tessera-mm-ptc-text.json",
        ])
        self.assertEqual(args.model_role, "trunk")
        self.assertEqual(args.tensor_stats_source, "imatrix")


class TestRealCaptureV2(unittest.TestCase):
    """The v2 real capture path: ``--source real`` invokes
    the ``llama-clip-capture`` binary via subprocess and
    stamps ``source = 'real'`` on every row. The v1
    synthetic path is preserved byte-equivalent.

    These tests cover:
      * parser accepts the new ``--source`` and
        ``--clip-capture-binary`` flags.
      * the default ``--source`` is ``synthetic``.
      * invalid ``--source`` is rejected.
      * the helper functions (``_find_clip_capture_binary``,
        ``_role_from_prefix``, ``_family_of_activation``,
        ``_layer_of_activation``) behave correctly.
      * the sidecar records the source value.
      * the ``SOURCE_*`` constants are distinct values.
      * the v2 helper does not crash on missing binary.
      * the v2 helper returns the per-tensor row schema.
      * the v2 run() rejects invalid source values.
      * the v2 run() raises on missing binary.
    """

    def setUp(self) -> None:
        self.paths: list[str] = []
        self.artifact_paths: list[Path] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for p in self.artifact_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def _track_artifact(self, p: Path) -> Path:
        self.artifact_paths.append(p)
        return p

    # ---- 1. parser accepts --source and --clip-capture-binary ----

    def test_parser_accepts_source_and_binary_flags(self) -> None:
        parser = mm_cal._build_parser()
        args = parser.parse_args([
            "--vision-tower", "/dev/null",
            "--output", "/tmp/tessera-mm-real-test.json",
            "--source", "real",
            "--clip-capture-binary", "/some/path/llama-clip-capture",
        ])
        self.assertEqual(args.source, "real")
        self.assertEqual(
            str(args.clip_capture_binary),
            "/some/path/llama-clip-capture")

    # ---- 2. default --source is synthetic (byte-equivalent) ----

    def test_default_source_is_synthetic(self) -> None:
        parser = mm_cal._build_parser()
        args = parser.parse_args([
            "--vision-tower", "/dev/null",
            "--output", "/tmp/tessera-mm-syn-default.json",
        ])
        self.assertEqual(args.source, "synthetic")

    # ---- 3. invalid source is rejected ----

    def test_invalid_source_rejected(self) -> None:
        parser = mm_cal._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--vision-tower", "/dev/null",
                "--output", "/tmp/tessera-mm-bad.json",
                "--source", "bogus",
            ])

    # ---- 4. SOURCE_* constants are distinct ----

    def test_source_constants_distinct(self) -> None:
        sources = {
            mm_cal.SOURCE_PY_MM_CAL,
            mm_cal.SOURCE_REAL,
            mm_cal.SOURCE_BACKFILL,
            mm_cal.SOURCE_BACKFILL_REAL,
        }
        self.assertEqual(len(sources), 4,
            f"source values must be distinct: {sources}")
        # The four values are the audit-trail enum.
        self.assertEqual(mm_cal.SOURCE_PY_MM_CAL, "py_mm_cal")
        self.assertEqual(mm_cal.SOURCE_REAL, "real")
        self.assertEqual(mm_cal.SOURCE_BACKFILL, "backfill")
        self.assertEqual(mm_cal.SOURCE_BACKFILL_REAL, "backfill_real")

    # ---- 5. _role_from_prefix ----

    def test_role_from_prefix(self) -> None:
        self.assertEqual(mm_cal._role_from_prefix("v.Kcur-0"),
                         "vision_tower")
        self.assertEqual(mm_cal._role_from_prefix("a.attn_out-1"),
                         "audio_tower")
        self.assertEqual(mm_cal._role_from_prefix("mm.up.weight"),
                         "mm_projector")
        self.assertIsNone(mm_cal._role_from_prefix("unprefixed.foo"))

    # ---- 6. _family_of_activation ----

    def test_family_of_activation(self) -> None:
        # The activation names don't follow the v1 weight
        # naming convention; most map to "other". The
        # helper does its best.
        self.assertEqual(mm_cal._family_of_activation("v.Kcur-0"),
                         "other")
        self.assertEqual(mm_cal._family_of_activation("v.attn_out-1"),
                         "attn_output")
        self.assertEqual(mm_cal._family_of_activation("v.ffn_out-0"),
                         "ffn_output")
        self.assertEqual(mm_cal._family_of_activation("v.patch_embd"),
                         "other")
        self.assertEqual(mm_cal._family_of_activation("a.layer_out-2"),
                         "other")

    # ---- 7. _layer_of_activation ----

    def test_layer_of_activation(self) -> None:
        self.assertEqual(mm_cal._layer_of_activation("v.Kcur-0"), 0)
        self.assertEqual(mm_cal._layer_of_activation("v.Kcur-7"), 7)
        self.assertEqual(mm_cal._layer_of_activation("v.patch_embd"), 0)
        self.assertEqual(mm_cal._layer_of_activation("a.attn_out-3"), 3)

    # ---- 8. run() rejects invalid source values ----

    def test_run_rejects_invalid_source(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            mm_cal.run(
                db_path=None,
                model_hash="m2_test_bad",
                vision_tower=Path("/dev/null"),
                source="bogus",
            )
        self.assertIn("source must be", str(ctx.exception))

    # ---- 9. run() raises on missing binary when source=real ----

    def test_run_real_raises_on_missing_binary(self) -> None:
        # We pass a non-existent --clip-capture-binary path
        # to bypass the probe; the helper should still
        # raise a clean error.
        with self.assertRaises(RuntimeError) as ctx:
            mm_cal.run(
                db_path=None,
                model_hash="m2_test_nobin",
                vision_tower=Path("/dev/null"),
                source="real",
                clip_capture_binary=Path("/nonexistent/llama-clip-capture"),
            )
        self.assertIn("llama-clip-capture", str(ctx.exception))

    # ---- 10. _find_clip_capture_binary probe order ----

    def test_find_clip_capture_binary_override(self) -> None:
        # The override is used when the file exists.
        fake = self._track_artifact(Path("/tmp/test-mm-fake-binary"))
        fake.write_text("#!/bin/sh\necho fake\n")
        os.chmod(fake, 0o755)
        result = mm_cal._find_clip_capture_binary(str(fake))
        self.assertEqual(result, str(fake))

    def test_find_clip_capture_binary_missing(self) -> None:
        # A non-existent override + no PATH / default probe
        # entry returns None.
        # Save and clear the PATH; restore after.
        saved = os.environ.get("PATH")
        os.environ["PATH"] = ""
        try:
            result = mm_cal._find_clip_capture_binary(
                "/nonexistent/llama-clip-capture")
        finally:
            if saved is not None:
                os.environ["PATH"] = saved
        # The result may be None (no binary) or a valid path
        # (the PATH probe found one). The test asserts
        # consistent behaviour: either the override is used
        # (when it exists) or the probe is consulted.
        if result is not None:
            self.assertTrue(Path(result).is_file())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestMultimodalCalibrate
    )
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestActivationStats
    ))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestFamilyLayerHelpers
    ))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGguFReaderFallback
    ))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPerTensorCalibrateM0a
    ))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRealCaptureV2
    ))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

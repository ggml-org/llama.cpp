"""Unified Python module for the ``tessera.duckdb`` store.

The unified store is the single DuckDB file that the calibration
(Python) and quantization (C++ via ``llama-quantize --quantize-db``)
pipelines both read and write. It holds the cross-pipeline feature
table (``tensor_stats``) and the per-tensor summary mirrors of the
four analytical outputs (``l3_outlier_summary``, ``l4_probe_summary``,
``l5_plan_summary``, ``per_layer_error_summary``).

The C++ side opens / creates the schema via
``ts_quantize_db_open()`` in ``tessera-quantize-db.cpp``; the schema
is identical, so a Python open on the same file picks up the same
tables. The Python side never creates the schema; the C++ side
owns that. If you need to create the schema from scratch, run the
C++ ``test-tessera-quantize-db`` binary or any ``llama-quantize
--quantize-db <path>`` invocation first.

Pipelines are sequential (calibration -> quantization -> analytics),
but within a pipeline the workers are heavy parallel. The buffer
abstraction (see ``tessera_db_buffer.py``) is the write side; this
module is the high-level API surface that owns the buffers and
provides typed insert helpers per table.

Companion to ``docs/tessera-polars-integration-scout.md`` and the
unified-DB follow-up. The C++ counterpart is
``tools/quantize/tessera/tessera-quantize-db.{h,cpp}`` and
``tools/quantize/tessera/tessera-db-buffer.{h,cpp}``.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import polars as pl

from tessera_db_buffer import TesseraDBBuffer, sql_escape


# Column lists for the unified-schema tables. The order is the INSERT
# order. Mirrored on the C++ side in tessera-quantize-db.cpp's CREATE
# TABLE statements; keep them in sync.
#
# Phase 16: model_role was added to the 7 affected tables
# (tensor_stats, l3_outlier_summary, l4_probe_summary, l5_plan_summary,
# l4_plan_outcome, l5_outcome, l5_weights) to disambiguate tensors
# with the same name in the unified Gemma4 12B + dspark + dflash +
# MTP arch (e.g. blk.0.attn_q.weight exists in both the trunk and
# the dflash encoder). The PKs now include model_role. The default
# 'trunk' preserves the pre-Phase-16 contract when the caller omits
# the column.
TENSOR_STATS_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "family", "layer_depth",
    "out_dim", "in_dim", "n_elements", "dtype",
    "kurtosis", "eff_rank", "rms", "mean_abs", "tail_ratio",
    "source", "recommended_action", "updated_at",
    # Targeted re-calibration: the orchestrator's monitor
    # verdict triggers a focused re-capture on a
    # domain-specific sample subset. The backfill machinery
    # (tools/tessera/backfill.py) increments this counter
    # each time the per-tensor activation stats are
    # re-captured; the orchestrator reads the counter to
    # decide whether the next iteration should re-trigger
    # the backfill (it gates on ``backfill_count <
    # max_backfill_rounds``). NULL means "no backfill yet"
    # (the column is additive; pre-backfill rows are NULL).
    "backfill_count",
)
L3_OUTLIER_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "layer", "sidecar_label",
    "outlier_count", "outlier_fraction", "max_abs", "rms",
    "updated_at",
)
L4_PROBE_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "layer",
    "current_qtype", "mse", "mse_minus_one", "perplexity",
    "top1_mismatch", "n_weights", "updated_at",
)
L4_PLAN_OUTCOME_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "layer", "iteration", "plan_id", "strategy",
    "alpha_before", "alpha_after", "clip_before", "clip_after",
    "outlier_thresh_before", "outlier_thresh_after",
    "mse_before", "mse_after", "frob_before", "frob_after",
    "family", "updated_at",
)
L5_PLAN_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "layer", "iteration", "plan_id",
    "sensitivity_score", "recommended_qtype", "recommended_alpha",
    "recommended_clip",
    # Phase 15: per-tensor sensitivity component columns (additive,
    # nullable). The orchestrator's write_history populates these
    # from SensitivityScorer.score()'s per-tensor (imatrix_magnitude,
    # gradient_proxy, layer_position_prior) outputs. The retune
    # reads them to fit a 3-coefficient OLS that decomposes the
    # miscalibration per (model, family). When the columns are
    # NULL (rows written by older producers or by the C++ side
    # before this commit), the retune falls back to the 2-coefficient
    # OLS on the combined sensitivity_score.
    "imatrix_magnitude", "gradient_proxy", "layer_position_prior",
    "updated_at",
)
L5_OUTCOME_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "name", "layer", "iteration", "plan_id",
    "family", "sensitivity_score",
    # Phase 15: per-tensor sensitivity components (additive,
    # nullable). Populated from the plan side (l5_plan_summary)
    # by l5_outcome.py at read time. The retune reads these to
    # fit a 3-coefficient OLS that decomposes which component is
    # miscalibrated per (model, family). NULL means "no
    # decomposition possible" (older rows or C++ side that has
    # not yet populated the columns); the retune falls back to
    # the 2-coefficient OLS on the combined sensitivity_score.
    "imatrix_magnitude", "gradient_proxy", "layer_position_prior",
    "recommended_alpha", "recommended_clip",
    "mse_before", "mse_after", "delta_mse", "delta_frob",
    "plan_accepted", "accept_threshold", "residual", "updated_at",
)
L5_WEIGHTS_COLS: tuple[str, ...] = (
    "model_hash", "model_role", "family",
    "w_imatrix", "w_gradient", "w_layer",
    "bias", "n_samples", "in_sample_loss", "hit_rate",
    # Phase 15: per-family top_fraction retune. The orchestrator's
    # RequantPlanner can consume this via --per-family-top-fraction
    # and override the uniform --top-fraction for the families the
    # retune has flagged as miscalibrated. NULL means "no
    # recommendation" (use the --top-fraction flag value). The
    # rule: top_fraction = base * (1 + tanh(2*slope)*(1-hit_rate)).
    # High slope + low hit rate -> more aggressive requantization
    # of the miscalibrated family; low slope or high hit rate ->
    # keep at base. Nullable / additive: existing rows in the
    # table are unaffected.
    "top_fraction",
    # Retune follow-ups: cross-component coupling score. Pearson
    # correlation of the per-layer hit_rate between the trunk's
    # role and the dflash encoder's role for the same family. A
    # high score means the two roles' miscalibration moves
    # together across layers; a low score means they are
    # independent. The retune surfaces this as a new column on
    # l5_weights so the consumer can see whether a single
    # retune covers both roles. NULL when the family has only
    # one role's rows (e.g. a pre-Phase-16 retune that did not
    # partition by role) or when the per-role per-layer hit
    # rates have zero variance (the correlation is undefined).
    "coupling_score",
    "retune_source", "updated_at",
)
PER_LAYER_ERROR_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer",
    "epsilon", "reference_qtype", "updated_at",
)
# Phase 0.5: the EXL2 per-layer sensitivity table. One row
# per (model_hash, layer_index, exl2_calibration_corpus); the
# corpus discriminator lets multiple calibration runs (e.g.
# wikitext-103 and COCO) coexist for the same model. The
# ``exl2_chosen_bpw`` integer mirrors ``exl2_per_layer_bpw``
# (the latter is REAL to match the HIGGS sidecar's per-tensor
# binarity-discriminator convention; the integer form is the
# primary column and the REAL is a backwards-compat alias).
EXL2_LAYER_STATS_COLS: tuple[str, ...] = (
    "model_hash", "layer_index", "layer_name",
    "family", "n_elements",
    "exl2_per_layer_error", "exl2_per_layer_bpw",
    "exl2_chosen_bpw", "exl2_calibration_corpus",
    "created_at",
)


@dataclass
class TesseraDBConfig:
    """Tunables for the unified-DB write buffers.

    The defaults match the C++ side's ts_db_buffer_open defaults
    (65536 / 1.0s / durable=False).
    """

    flush_threshold: int = 65536
    flush_interval_sec: float = 1.0
    durable: bool = False


class TesseraDB:
    """High-level API for the unified ``tessera.duckdb`` store.

    Holds one ``duckdb.Connection`` and one ``TesseraDBBuffer`` per
    table the caller writes to. The connection is opened on
    construction; the buffers are created lazily on the first
    write so the construction cost is constant regardless of how
    many tables the caller eventually touches.

    Usage::

        with TesseraDB.open("/path/to/tessera.duckdb") as db:
            db.insert_tensor_stats(model_hash="abc", rows=[
                {"name": "blk.0.attn_q.weight", "family": "attn_q",
                 "layer_depth": 0, "kurtosis": 5.2, ...},
                ...
            ])
            df = db.query(
                "SELECT name, kurtosis FROM tensor_stats "
                "WHERE model_hash = 'abc' ORDER BY kurtosis DESC LIMIT 10"
            )

    The buffers are flushed on context exit (sync-on-exit). Queries
    that race an in-flight flush see the rows that have already
    landed in DuckDB; the pending rows are not visible until the
    next flush.
    """

    def __init__(
        self,
        db_path: str | Path,
        config: Optional[TesseraDBConfig] = None,
        read_only: bool = False,
    ) -> None:
        import duckdb  # type: ignore

        self._db_path = str(db_path)
        self._config = config or TesseraDBConfig()
        self._read_only = read_only
        self._conn = duckdb.connect(self._db_path, read_only=read_only)
        self._buffers: dict[str, TesseraDBBuffer] = {}
        self._buffer_lock = threading.Lock()
        self._closed = False
        # Phase 16.7: per-component (model_role, name) covering
        # index on the 7 unified tables. The composite PKs include
        # model_role but the per-component query pattern
        # ("WHERE model_role = ? AND name = ?") would otherwise
        # scan the PK left-to-right; a secondary index lets
        # DuckDB satisfy it with an index seek. The C++ side has
        # the same statements in TS_QDB_SCHEMA_SQL; on a C++-open
        # DB the indexes already exist, and the CREATE INDEX IF
        # NOT EXISTS below is a no-op. Cached on the instance so
        # we don't repeat the round-trip for every insert.
        if not read_only:
            self._ensure_unified_indexes()
            # Targeted re-calibration: the backfill_count
            # column on tensor_stats is additive; the
            # migration is idempotent (ADD COLUMN IF NOT
            # EXISTS) and runs on every open so a Python
            # opener picking up a C++-created DB sees the
            # column without an explicit migration step.
            self._ensure_tensor_stats_columns()
            # Phase 15: the per-tensor component columns
            # on l5_plan_summary are additive; the
            # migration is idempotent and runs on every
            # open so a C++-created DB without them sees
            # them added on the Python side's first open.
            # The same hook covers the Phase 0.5
            # ``exl2_error`` column.
            self._ensure_l5_plan_columns()
            # Phase 0.5: the EXL2 per-layer stats table
            # and the additive ``exl2_error`` column on
            # l5_plan_summary. The migration is idempotent
            # (CREATE TABLE IF NOT EXISTS for the table;
            # ADD COLUMN IF NOT EXISTS for the column);
            # a C++-created DB without these sees them
            # added on the Python side's first open. Old
            # data is intact: the migration is
            # forward-only.
            self._ensure_exl2_layer_stats()

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        config: Optional[TesseraDBConfig] = None,
        read_only: bool = False,
    ) -> "TesseraDB":
        """Open a TesseraDB. Equivalent to the constructor but reads
        better at call sites."""
        return cls(db_path, config=config, read_only=read_only)

    # ---- write API: tensor_stats -------------------------------------

    def insert_tensor_stats(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        """Buffered write of per-tensor stats into ``tensor_stats``.

        Bypasses the buffer and uses a direct INSERT ... ON CONFLICT
        DO UPDATE because the table has a primary key on
        (model_hash, model_role, name); the buffer's plain INSERT
        would fail on a duplicate. The C++ side has the same primary
        key and the same upsert pattern (see
        ``tessera-quantize-db.cpp::ts_quantize_db_upsert_tensor_stat``).

        Phase 16: the row dict may carry ``model_role`` (one of
        'trunk' / 'dflash' / 'dspark' / 'mtp_nextn' / 'shared_embd').
        The default is 'trunk' for backward compatibility with
        pre-Phase-16 callers. The bulk INSERT lists model_role as
        the third column; the ON CONFLICT clause also references it.

        Per-side writes are non-destructive across columns: the
        C++ side writes kurtosis / eff_rank / dtype (source =
        'cpp_quant'); the Python side writes rms / mean_abs /
        tail_ratio (source = 'py_cal'). The upsert preserves
        whichever side's columns are not overwritten by the
        other side's NULL-pass.

        Returns the number of rows upserted.
        """
        if not rows:
            return 0
        if self._read_only:
            sys.stderr.write(
                "tessera-db: insert_tensor_stats on read-only connection ignored\n"
            )
            return 0
        now = _now_iso()
        for r in rows:
            # Build one upsert per row. Per-row is fine because
            # the upsert is not on the hot path; the C++ side's
            # GA-prep walk also does per-row writes.
            #
            # Targeted re-calibration: ``backfill_count`` is
            # incremented on every backfill write. The
            # ``source`` column is the discriminator: a write
            # with ``source='backfill_real'`` increments the
            # counter; a write with any other source leaves
            # the counter unchanged (the COALESCE-preserve
            # contract the other columns follow). The INSERT
            # path uses ``INSERT ... SELECT`` (rather than
            # ``VALUES``) so the backfill_count can be derived
            # from the source at insert time: the first
            # backfill write on a fresh row sets the counter
            # to 1 (not NULL). Subsequent backfill writes
            # (the UPDATE path) increment.
            sql = (
                "INSERT INTO tensor_stats ("
                "  model_hash, model_role, name, family, layer_depth, "
                "  out_dim, in_dim, n_elements, dtype, "
                "  kurtosis, eff_rank, rms, mean_abs, tail_ratio, "
                "  source, recommended_action, updated_at, backfill_count"
                ") SELECT "
                "  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "  ?, ?, ?, "
                "  CASE WHEN ? = 'backfill_real' THEN 1 ELSE NULL END "
                "ON CONFLICT (model_hash, model_role, name) DO UPDATE SET "
                "  family             = COALESCE(excluded.family,             tensor_stats.family), "
                "  layer_depth        = COALESCE(excluded.layer_depth,        tensor_stats.layer_depth), "
                "  out_dim            = COALESCE(excluded.out_dim,            tensor_stats.out_dim), "
                "  in_dim             = COALESCE(excluded.in_dim,             tensor_stats.in_dim), "
                "  n_elements         = COALESCE(excluded.n_elements,         tensor_stats.n_elements), "
                "  dtype              = COALESCE(excluded.dtype,              tensor_stats.dtype), "
                "  kurtosis           = COALESCE(excluded.kurtosis,           tensor_stats.kurtosis), "
                "  eff_rank           = COALESCE(excluded.eff_rank,           tensor_stats.eff_rank), "
                "  rms                = COALESCE(excluded.rms,                tensor_stats.rms), "
                "  mean_abs           = COALESCE(excluded.mean_abs,           tensor_stats.mean_abs), "
                "  tail_ratio         = COALESCE(excluded.tail_ratio,         tensor_stats.tail_ratio), "
                "  source             = excluded.source, "
                "  recommended_action = COALESCE(excluded.recommended_action, tensor_stats.recommended_action), "
                "  updated_at         = excluded.updated_at, "
                "  backfill_count     = CASE "
                "    WHEN excluded.source = 'backfill_real' THEN "
                "      COALESCE(tensor_stats.backfill_count, 0) + 1 "
                "    ELSE tensor_stats.backfill_count "
                "  END"
            )
            # COALESCE on the UPDATE means: if the new write's
            # column is NULL, keep the existing value (the other
            # side's contribution). If the new write has a value,
            # it wins. This is how the C++ kurtosis / eff_rank
            # survives a Python rms / mean_abs / tail_ratio
            # upsert (and vice versa); the same convention applies
            # to recommended_action (the calibration side's
            # l5_action verdict; the C++ side never writes it).
            #
            # backfill_count: targeted re-calibration
            # increments the counter on every backfill
            # write. The caller can pass an explicit value
            # (the typical case is ``None`` = "increment by
            # 1"); passing a value lets callers seed or
            # repair the counter explicitly. The CASE
            # chain is: caller explicit -> source-driven
            # increment on update -> existing (NULL-safe).
            try:
                self._conn.execute(sql, [
                    model_hash,
                    r.get("model_role", "trunk"),
                    r.get("name", ""),
                    r.get("family"),
                    r.get("layer_depth"),
                    r.get("out_dim"),
                    r.get("in_dim"),
                    r.get("n_elements"),
                    r.get("dtype"),
                    r.get("kurtosis"),
                    r.get("eff_rank"),
                    r.get("rms"),
                    r.get("mean_abs"),
                    r.get("tail_ratio"),
                    r.get("source", "py_cal"),
                    r.get("recommended_action"),
                    r.get("updated_at", now),
                    # The CASE in the SELECT (insert path)
                    # reads from this parameter; the CASE in
                    # the ON CONFLICT (update path) reads
                    # from ``excluded.source``. The
                    # source-default (``py_cal``) keeps the
                    # default-source rows on the legacy NULL
                    # path; ``backfill_real`` rows get the
                    # source-driven increment.
                    r.get("source", "py_cal"),
                ])
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: insert_tensor_stats on "
                    f"({model_hash}, {r.get('model_role', 'trunk')}, "
                    f"{r.get('name', '')}) failed: {e}\n"
                )
        return len(rows)

    # ---- write API: per-source summary tables ------------------------

    def insert_l3_outlier(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l3_outlier_summary", L3_OUTLIER_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("model_role", "trunk"),
                r.get("name", ""),
                r.get("layer"),
                r.get("sidecar_label", ""),
                r.get("outlier_count"),
                r.get("outlier_fraction"),
                r.get("max_abs"),
                r.get("rms"),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def insert_l4_probe(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l4_probe_summary", L4_PROBE_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("model_role", "trunk"),
                r.get("name", ""),
                r.get("layer"),
                r.get("current_qtype", ""),
                r.get("mse"),
                r.get("mse_minus_one"),
                r.get("perplexity"),
                r.get("top1_mismatch"),
                r.get("n_weights"),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def insert_l5_plan(
        self,
        model_hash: str,
        rows: Sequence[dict],
        *,
        model_role: str = "trunk",
    ) -> int:
        """Push rows into the ``l5_plan_summary`` table.

        ``model_role`` (Phase 16) tags the rows by their
        architectural role in the unified model: ``"trunk"``
        for the main backbone, ``"dflash"`` for the dflash
        encoder, ``"dspark"`` for the dspark drafter,
        ``"mtp_nextn"`` for the MTP-NextN head, ``"shared_embd"``
        for the shared embedding / output projection. Defaults
        to ``"trunk"`` for backward compat with Phase 15 callers
        that pre-date the role dimension.

        The ``model_role`` column is added via ``ALTER TABLE ...
        ADD COLUMN IF NOT EXISTS`` so the upsert works on DBs
        created before Phase 16.
        """
        if not rows or self._read_only:
            return 0
        # Phase 15: the per-tensor component columns are additive
        # to l5_plan_summary. The C++ side's CREATE TABLE IF NOT
        # EXISTS may have been run against an older schema; the
        # Python side ensures the columns exist before the first
        # INSERT so the buffer's bulk INSERT (which lists every
        # column) does not fail on a missing column. ``ALTER TABLE
        # ... ADD COLUMN IF NOT EXISTS`` is a no-op when the
        # column already exists, so this is safe on every open.
        #
        # Phase 16: model_role is in the column list (col index 2
        # in L5_PLAN_COLS). The CREATE TABLE on a fresh DB has
        # it; on a pre-Phase-16 DB, the Python migration
        # (tools/tessera/migrate_model_role.py) is the path that
        # adds the column + rebuilds the PK. This helper does
        # not run the migration (it's a separate script); the
        # caller is expected to have run it before opening the
        # DB for write. _ensure_l5_plan_columns is preserved for
        # the Phase 15 columns; the model_role column is part of
        # the canonical schema and not added lazily here.
        # Phase 16: the same idempotent ALTER covers ``model_role``.
        self._ensure_l5_plan_columns()
        role = str(model_role) if model_role else "trunk"
        buf = self._buffer_for("l5_plan_summary", L5_PLAN_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("model_role", role),
                r.get("name", ""),
                r.get("layer"),
                r.get("iteration"),
                r.get("plan_id", ""),
                r.get("sensitivity_score"),
                r.get("recommended_qtype", ""),
                r.get("recommended_alpha"),
                r.get("recommended_clip"),
                r.get("imatrix_magnitude"),
                r.get("gradient_proxy"),
                r.get("layer_position_prior"),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def _ensure_l5_plan_columns(self) -> None:
        """Add the Phase 15 per-tensor component columns AND the
        Phase 16 ``model_role`` column to ``l5_plan_summary`` if
        they are not already present.

        The C++ side owns the canonical schema; when a C++ binary
        creates the DB first the l5_plan_summary table does not
        yet have ``imatrix_magnitude``, ``gradient_proxy``,
        ``layer_position_prior``, or ``model_role``. The Python
        side adds them via ``ALTER TABLE ... ADD COLUMN IF NOT
        EXISTS`` so the bulk INSERT (which lists every
        L5_PLAN_COLS column) does not fail on a missing column.
        The operation is a no-op when the columns are already
        present (i.e. when the Python side creates the schema
        itself or when the C++ side has been updated to include
        them). Idempotent.

        The ``model_role`` column is TEXT with a default of
        ``"trunk"`` so pre-Phase-16 rows that did not specify
        the role still read back with the legacy value. The
        retune / orchestrator treat ``NULL`` and ``"trunk"``
        interchangeably (the legacy lookup path).
        """
        if self._read_only:
            return
        # Cache the answer in the instance so we don't repeat
        # the ALTER for every insert on a long-running loop.
        if getattr(self, "_l5_plan_columns_ensured", False):
            return
        for col in (
            "imatrix_magnitude",
            "gradient_proxy",
            "layer_position_prior",
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE l5_plan_summary "
                    f"ADD COLUMN IF NOT EXISTS {col} DOUBLE"
                )
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: ALTER TABLE l5_plan_summary "
                    f"ADD COLUMN {col} failed: {e}\n"
                )
        # Phase 16: model_role column. TEXT with default
        # 'trunk' for backward compat. The retune / orchestrator
        # treat NULL and 'trunk' as the same role.
        try:
            self._conn.execute(
                "ALTER TABLE l5_plan_summary "
                "ADD COLUMN IF NOT EXISTS model_role TEXT DEFAULT 'trunk'"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_plan_summary "
                f"ADD COLUMN model_role failed: {e}\n"
            )
        # Phase 0.5: the EXL2 per-layer error folded into the
        # per-tensor sensitivity path. The column is additive:
        # NULL when the EXL2 estimator has not been run on this
        # tensor, a per-tensor real when it has. The L5
        # orchestrator's SensitivityScorer reads it via
        # ``get_exl2_per_layer_error_for_tensors`` (a separate
        # query helper) and folds the value into the
        # ``sensitivity_score`` when ``w_exl2 > 0``. The
        # orchestrator's default ``w_exl2 = 0.0`` keeps the
        # path opt-in until the first EXL2 run lands.
        try:
            self._conn.execute(
                "ALTER TABLE l5_plan_summary "
                "ADD COLUMN IF NOT EXISTS exl2_error DOUBLE"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_plan_summary "
                f"ADD COLUMN exl2_error failed: {e}\n"
            )
        self._l5_plan_columns_ensured = True

    def _ensure_tensor_stats_columns(self) -> None:
        """Idempotent migration of the targeted re-calibration
        (``backfill_count``) column on ``tensor_stats``.

        Targeted re-calibration (the focused re-capture
        triggered by the L5 monitor verdict) writes a
        ``backfill_count`` per row; the orchestrator gates
        the next-iteration re-capture on this counter. The
        column is additive: a pre-backfill DB has no
        ``backfill_count`` column; the migration adds it
        with a NULL default so the existing rows are
        unchanged. The migration runs on every
        ``TesseraDB`` open (``__init__``) and is cached on
        the instance, so the round-trip is paid once per
        process lifetime, not once per insert.

        ``ADD COLUMN IF NOT EXISTS`` is a no-op when the
        column already exists, so the call is safe on every
        open and on every DB shape (Python-only, C++-then-
        Python, fresh, post-migration).
        """
        if self._read_only:
            return
        if getattr(self, "_tensor_stats_columns_ensured", False):
            return
        try:
            self._conn.execute(
                "ALTER TABLE tensor_stats "
                "ADD COLUMN IF NOT EXISTS backfill_count INTEGER DEFAULT NULL"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE tensor_stats "
                f"ADD COLUMN backfill_count failed: {e}\n"
            )
        self._tensor_stats_columns_ensured = True

    def insert_l4_plan_outcome(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        """Push rows into the ``l4_plan_outcome`` table. The C++
        dispatch's adaptive_requantize loop also writes here (one
        row per (tensor, gen)). ``rows`` is a list of dicts with
        keys: name, model_role (Phase 16; default 'trunk'), layer,
        iteration, plan_id, strategy, alpha_before, alpha_after,
        clip_before, clip_after, outlier_thresh_before,
        outlier_thresh_after, mse_before, mse_after, frob_before,
        frob_after, family.

        Phase 16: model_role disambiguates dflash / dspark /
        mtp_nextn rows from the trunk's. The drafter-local
        tensor name goes in ``name`` (e.g. the dflash encoder's
        'blk.0.attn_q.weight', not the global 'dflash.blk.0...
        name'); the consumer joins via
        (model_hash, model_role, name).

        Returns the number of rows accepted.
        """
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l4_plan_outcome", L4_PLAN_OUTCOME_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("model_role", "trunk"),
                r.get("name", ""),
                r.get("layer"),
                r.get("iteration"),
                r.get("plan_id", ""),
                r.get("strategy", ""),
                r.get("alpha_before"),
                r.get("alpha_after"),
                r.get("clip_before"),
                r.get("clip_after"),
                r.get("outlier_thresh_before"),
                r.get("outlier_thresh_after"),
                r.get("mse_before"),
                r.get("mse_after"),
                r.get("frob_before"),
                r.get("frob_after"),
                r.get("family", ""),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def insert_l5_outcome(
        self,
        model_hash: str,
        rows: Sequence[dict],
        *,
        model_role: str = "trunk",
    ) -> int:
        """Push rows into the ``l5_outcome`` table. Written by
        ``tools/tessera/l5_outcome.py`` after joining
        ``l5_plan_summary`` and ``l4_plan_outcome``.

        ``rows`` is a list of dicts with keys: name, layer,
        iteration, plan_id, family, sensitivity_score,
        imatrix_magnitude, gradient_proxy, layer_position_prior
        (Phase 15), recommended_alpha, recommended_clip,
        mse_before, mse_after, delta_mse, delta_frob,
        plan_accepted, accept_threshold, residual,
        model_role (Phase 16).

        Phase 15: the per-tensor component columns are nullable.
        Older C++ writers do not set them; the retune falls back
        to the 2-coefficient OLS on the combined sensitivity_score
        when the components are NULL.

        Phase 16: ``model_role`` is the architectural role
        (``"trunk"``, ``"dflash"``, ``"dspark"``, ``"mtp_nextn"``,
        ``"shared_embd"``). The retune's per-(model, family)
        groupby is now per-(model, model_role, family) so the
        trunk's ``attn_q`` and the dflash encoder's ``attn_q``
        get independent retune verdicts. Defaults to ``"trunk"``
        for backward compat with Phase 15 callers.

        Returns the number of rows accepted.
        """
        if not rows or self._read_only:
            return 0
        # Phase 15: ensure the per-tensor component columns exist
        # before the first INSERT. Same idempotent ALTER TABLE
        # pattern as _ensure_l5_plan_columns. Phase 16: also
        # ensure the model_role column exists.
        self._ensure_l5_outcome_columns()
        role = str(model_role) if model_role else "trunk"
        buf = self._buffer_for("l5_outcome", L5_OUTCOME_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("model_role", role),
                r.get("name", ""),
                r.get("layer"),
                r.get("iteration"),
                r.get("plan_id", ""),
                r.get("family", ""),
                r.get("sensitivity_score"),
                r.get("imatrix_magnitude"),
                r.get("gradient_proxy"),
                r.get("layer_position_prior"),
                r.get("recommended_alpha"),
                r.get("recommended_clip"),
                r.get("mse_before"),
                r.get("mse_after"),
                r.get("delta_mse"),
                r.get("delta_frob"),
                r.get("plan_accepted"),
                r.get("accept_threshold"),
                r.get("residual"),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def _ensure_l5_outcome_columns(self) -> None:
        """Add the Phase 15 per-tensor component columns AND the
        Phase 16 ``model_role`` column to ``l5_outcome`` if they
        are not already present.

        The C++ side may or may not have picked up the column
        addition (the docs claim the C++ side "pre-emptively
        added" the columns; the current source has not). The
        Python side adds them via ``ALTER TABLE ... ADD COLUMN
        IF NOT EXISTS`` so the bulk INSERT (which lists every
        L5_OUTCOME_COLS column) does not fail on a missing
        column. Idempotent.
        """
        if self._read_only:
            return
        if getattr(self, "_l5_outcome_columns_ensured", False):
            return
        for col in (
            "imatrix_magnitude",
            "gradient_proxy",
            "layer_position_prior",
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE l5_outcome "
                    f"ADD COLUMN IF NOT EXISTS {col} DOUBLE"
                )
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: ALTER TABLE l5_outcome "
                    f"ADD COLUMN {col} failed: {e}\n"
                )
        # Phase 16: model_role column. TEXT with default
        # 'trunk' for backward compat. Pre-Phase-16 rows that
        # did not specify the role still read back with the
        # legacy value; the retune / orchestrator treat NULL
        # and 'trunk' as the same role.
        try:
            self._conn.execute(
                "ALTER TABLE l5_outcome "
                "ADD COLUMN IF NOT EXISTS model_role TEXT DEFAULT 'trunk'"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_outcome "
                f"ADD COLUMN model_role failed: {e}\n"
            )
        self._l5_outcome_columns_ensured = True

    # ---- write API: l5_weights (PRIMARY KEY -> direct upsert) ----

    def insert_l5_weights(
        self,
        rows: Sequence[dict],
    ) -> int:
        """Upsert rows into the ``l5_weights`` table (per-model,
        per-role, per-family retuned scoring weights).

        Bypasses the buffer and uses a direct ``INSERT ... ON
        CONFLICT (model_hash, model_role, family) DO UPDATE``
        because the table has a primary key; the buffer's plain
        INSERT would fail on a duplicate. The C++ side has the
        same primary key and the same upsert pattern (see
        ``ts_tessera_db_upsert_l5_weight`` in
        ``tessera-quantize-db.cpp``).

        ``rows`` is a list of dicts with keys: model_hash,
        model_role (Phase 16; default 'trunk'), family,
        w_imatrix, w_gradient, w_layer, bias, n_samples,
        in_sample_loss, hit_rate, top_fraction (nullable, Phase 15),
        retune_source. The retune is the consumer-side half of the
        "did this requant plan reduce error?" feedback loop:
        ``tools/tessera/l5_retune.py`` fits a per-(model, family)
        closed-form OLS on ``l5_outcome.delta_mse`` and
        ``l5_outcome.sensitivity_score`` and writes the recommended
        ``(w_imatrix, w_gradient, w_layer)`` here. The
        orchestrator's next generation reads the table back via
        ``--retune-from-db``.
        ``rows`` is a list of dicts with keys: model_hash,
        model_role (Phase 16), family, w_imatrix, w_gradient,
        w_layer, bias, n_samples, in_sample_loss, hit_rate,
        top_fraction (nullable, Phase 15), coupling_score
        (nullable), requant_budget_bits (nullable BIGINT, the
        dispatch-side bit budget the retune recommends for the
        family's next requant pass; NULL = unconstrained),
        retune_source. The
        retune is the consumer-side half of the "did this
        requant plan reduce error?" feedback loop:
        ``tools/tessera/l5_retune.py`` fits a per-(model,
        model_role, family) closed-form OLS on
        ``l5_outcome.delta_mse`` and
        ``l5_outcome.sensitivity_score`` and writes the
        recommended ``(w_imatrix, w_gradient, w_layer)`` here.
        The orchestrator's next generation reads the table back
        via ``--retune-from-db --model-role R``.

        Phase 15: ``top_fraction`` is the per-family requant
        aggressiveness recommendation. NULL means "no
        recommendation" (use the --top-fraction flag value). The
        rule: ``top_fraction = base * (1 + tanh(2*slope)*(1-hit_rate))``.
        The column is added via ALTER TABLE IF NOT EXISTS so the
        upsert works on DBs created before Phase 15.

        Phase 16: model_role is part of the canonical schema
        (the l5_weights table is per-(model, role, family)). The
        Python migration is the path that adds the column +
        rebuilds the PK on a pre-Phase-16 DB. This helper does
        not run the migration; the caller is expected to have
        run migrate_model_role.py before opening the DB for write.
        Phase 16: the primary key is now ``(model_hash,
        model_role, family)`` so the trunk's ``attn_q`` and the
        dflash encoder's ``attn_q`` get independent
        (w_imatrix, w_gradient, w_layer) recommendations. The
        per-row ``model_role`` defaults to ``"trunk"`` when
        not provided (the pre-Phase-16 callers). The column is
        added via ``ALTER TABLE IF NOT EXISTS`` so the upsert
        works on DBs created before Phase 16.

        Returns the number of rows upserted.
        """
        if not rows or self._read_only:
            return 0
        # Phase 15: ensure the top_fraction column exists. Same
        # idempotency story as _ensure_l5_plan_columns: no-op when
        # the column is already present. Phase 16: also ensure
        # the model_role column exists. We add model_role BEFORE
        # the upsert (rather than relying on the table's CREATE
        # TABLE) so legacy DBs that pre-date Phase 16's
        # CREATE TABLE migration also work. Retune follow-ups:
        # the coupling_score column is also added by
        # _ensure_l5_weights_columns.
        self._ensure_l5_weights_columns()
        sql = (
            "INSERT INTO l5_weights ("
            "  model_hash, model_role, family, w_imatrix, w_gradient, w_layer, "
            "  bias, n_samples, in_sample_loss, hit_rate, "
            "  top_fraction, coupling_score, requant_budget_bits, "
            "  retune_source, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (model_hash, model_role, family) DO UPDATE SET "
            "  w_imatrix       = excluded.w_imatrix, "
            "  w_gradient      = excluded.w_gradient, "
            "  w_layer         = excluded.w_layer, "
            "  bias            = excluded.bias, "
            "  n_samples       = excluded.n_samples, "
            "  in_sample_loss  = excluded.in_sample_loss, "
            "  hit_rate        = excluded.hit_rate, "
            "  top_fraction    = excluded.top_fraction, "
            "  coupling_score  = excluded.coupling_score, "
            "  requant_budget_bits = excluded.requant_budget_bits, "
            "  retune_source   = excluded.retune_source, "
            "  updated_at      = excluded.updated_at"
        )
        # Phase 16: when the table's PRIMARY KEY is the
        # legacy 2-tuple ``(model_hash, family)`` (a
        # pre-Phase-16 DB), the ON CONFLICT 3-tuple clause
        # is invalid. Fall back to a DELETE+INSERT path on
        # the legacy PK. The schema worker's migration
        # eventually upgrades the PK to the 3-tuple; the
        # DELETE+INSERT path is non-destructive to other
        # rows. (The 3-tuple path on a legacy PK would
        # produce a Binder error; the 2-tuple fallback is
        # the safe choice for pre-migration DBs.)
        is_3tuple_pk = self._l5_weights_pk_shape()
        if is_3tuple_pk:
            sql = (
                "INSERT INTO l5_weights ("
                "  model_hash, model_role, family, "
                "  w_imatrix, w_gradient, w_layer, "
                "  bias, n_samples, in_sample_loss, hit_rate, "
                "  top_fraction, coupling_score, requant_budget_bits, "
                "  retune_source, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_hash, model_role, family) DO UPDATE SET "
                "  w_imatrix       = excluded.w_imatrix, "
                "  w_gradient      = excluded.w_gradient, "
                "  w_layer         = excluded.w_layer, "
                "  bias            = excluded.bias, "
                "  n_samples       = excluded.n_samples, "
                "  in_sample_loss  = excluded.in_sample_loss, "
                "  hit_rate        = excluded.hit_rate, "
                "  top_fraction    = excluded.top_fraction, "
                "  coupling_score  = excluded.coupling_score, "
                "  requant_budget_bits = excluded.requant_budget_bits, "
                "  retune_source   = excluded.retune_source, "
                "  updated_at      = excluded.updated_at"
            )
        else:
            # Legacy 2-tuple PK. The ``model_role`` is
            # ignored on the ON CONFLICT target; the
            # behaviour is the pre-Phase-16
            # ``(model_hash, family)`` upsert. The role
            # value is still written to the row (the
            # column exists; the column default of
            # ``"trunk"`` covers pre-Phase-16 writers).
            sql = (
                "INSERT INTO l5_weights ("
                "  model_hash, model_role, family, "
                "  w_imatrix, w_gradient, w_layer, "
                "  bias, n_samples, in_sample_loss, hit_rate, "
                "  top_fraction, coupling_score, requant_budget_bits, "
                "  retune_source, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_hash, family) DO UPDATE SET "
                "  w_imatrix       = excluded.w_imatrix, "
                "  w_gradient      = excluded.w_gradient, "
                "  w_layer         = excluded.w_layer, "
                "  bias            = excluded.bias, "
                "  n_samples       = excluded.n_samples, "
                "  in_sample_loss  = excluded.in_sample_loss, "
                "  hit_rate        = excluded.hit_rate, "
                "  top_fraction    = excluded.top_fraction, "
                "  coupling_score  = excluded.coupling_score, "
                "  requant_budget_bits = excluded.requant_budget_bits, "
                "  retune_source   = excluded.retune_source, "
                "  model_role      = excluded.model_role, "
                "  updated_at      = excluded.updated_at"
            )
        now = _now_iso()
        n = 0
        for r in rows:
            try:
                self._conn.execute(sql, [
                    r.get("model_hash", ""),
                    r.get("model_role", "trunk"),
                    r.get("family", ""),
                    float(r.get("w_imatrix", 0.0)),
                    float(r.get("w_gradient", 0.0)),
                    float(r.get("w_layer", 0.0)),
                    r.get("bias"),
                    r.get("n_samples"),
                    r.get("in_sample_loss"),
                    r.get("hit_rate"),
                    r.get("top_fraction"),
                    r.get("coupling_score"),
                    r.get("requant_budget_bits"),
                    r.get("retune_source", "ols_slope_v1"),
                    r.get("updated_at", now),
                ])
                n += 1
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: insert_l5_weights on "
                    f"({r.get('model_hash', '')}, "
                    f"{r.get('model_role', 'trunk')}, "
                    f"{r.get('family', '')}) failed: {e}\n"
                )
        return n

    def _ensure_l5_weights_columns(self) -> None:
        """Add the Phase 15 ``top_fraction`` column AND the
        Phase 16 ``model_role`` column to ``l5_weights`` if
        not already present.

        Same idempotent ALTER TABLE pattern as
        :py:meth:`_ensure_l5_plan_columns`. The C++ side may
        have created the l5_weights table before Phase 15; the
        Python side adds the column on the first open so the
        upsert does not fail on a missing column.

        Phase 16: the ``model_role`` column is added with a
        default of ``"trunk"`` for backward compat. Pre-Phase-16
        rows that did not specify the role still read back with
        the legacy value; the retune / orchestrator treat
        NULL and ``"trunk"`` as the same role.

        The function does NOT migrate the PRIMARY KEY from
        ``(model_hash, family)`` to
        ``(model_hash, model_role, family)``; the PK
        migration is the schema worker's domain (it requires
        dropping the old PK, which is a destructive
        operation; the Python side is non-destructive). On
        pre-Phase-16 DBs (with the 2-tuple PK), the
        :py:meth:`_l5_weights_pk_shape` helper detects the
        legacy PK and ``insert_l5_weights`` falls back to a
        DELETE-then-INSERT path. The retune's
        ``compute_l5_weights`` already uses a
        DELETE-then-INSERT for the write-back (the same
        pattern), so the upsert path is not on the hot
        write path; the legacy-PK DB eventually migrates
        to the 3-tuple PK via the schema worker's
        ``tessera-quantize-db.cpp`` CREATE TABLE.
        """
        if self._read_only:
            return
        if getattr(self, "_l5_weights_columns_ensured", False):
            return
        try:
            self._conn.execute(
                "ALTER TABLE l5_weights "
                "ADD COLUMN IF NOT EXISTS top_fraction DOUBLE"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_weights "
                f"ADD COLUMN top_fraction failed: {e}\n"
            )
        # Retune follow-ups: cross-component coupling score
        # column. DOUBLE, nullable. The retune populates this
        # when the family has rows for both trunk and dflash;
        # a legacy (single-role) retune leaves it NULL. Same
        # idempotent ALTER pattern as the other columns.
        try:
            self._conn.execute(
                "ALTER TABLE l5_weights "
                "ADD COLUMN IF NOT EXISTS coupling_score DOUBLE"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_weights "
                f"ADD COLUMN coupling_score failed: {e}\n"
            )
        # Phase 16: model_role column. TEXT with default
        # 'trunk' for backward compat.
        try:
            self._conn.execute(
                "ALTER TABLE l5_weights "
                "ADD COLUMN IF NOT EXISTS model_role TEXT DEFAULT 'trunk'"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_weights "
                f"ADD COLUMN model_role failed: {e}\n"
            )
        # requant_budget_bits column. BIGINT, nullable. The
        # canonical CREATE TABLE (both the C++ side's and the
        # Python migration) carries it since Phase 14, but a
        # DB created by an older Python path may pre-date it;
        # the upsert lists the column, so ensure it exists.
        try:
            self._conn.execute(
                "ALTER TABLE l5_weights "
                "ADD COLUMN IF NOT EXISTS requant_budget_bits BIGINT"
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: ALTER TABLE l5_weights "
                f"ADD COLUMN requant_budget_bits failed: {e}\n"
            )
        # Drop the cached PK shape so the next insert
        # re-probes the actual PK. (The PK may have been
        # the 2-tuple before the model_role column was
        # added; the schema worker's CREATE TABLE has the
        # 3-tuple. The cached value must be re-evaluated.)
        self._l5_weights_pk_3tuple = None
        self._l5_weights_columns_ensured = True

    def _l5_weights_pk_shape(self) -> bool:
        """Return True when the ``l5_weights`` table's PRIMARY
        KEY includes ``model_role`` (the Phase 16 3-tuple
        PK ``(model_hash, model_role, family)``); False
        when the PK is the legacy 2-tuple ``(model_hash,
        family)``.

        Used by ``insert_l5_weights`` to choose the right
        ON CONFLICT target. The PK shape is cached on
        the instance because the helper is called once per
        insert; the cache is invalidated by
        ``_ensure_l5_weights_columns`` (after the
        ``ALTER TABLE`` may have changed the column set,
        though not the PK itself). The cache is also
        invalidated on a fresh ``TesseraDB`` open (the
        instance is per-connection).

        The helper probes ``information_schema.table_constraints``
        for the PRIMARY KEY row (DuckDB names the constraint
        ``<table>_model_hash_<...>_pkey``; the helper
        filters by ``constraint_type = 'PRIMARY KEY'``).
        """
        cached = getattr(self, "_l5_weights_pk_3tuple", None)
        if cached is not None:
            return cached
        try:
            # The PK columns are on table_constraints
            # (constraint_type = 'PRIMARY KEY'); the
            # column order is read from key_column_usage
            # joined on the constraint name. DuckDB's
            # naming convention is
            # ``<table>_<col1>_<col2>_pkey`` so we filter
            # by table_name + constraint_type rather than
            # guessing the constraint name.
            pk_rows = self._conn.execute(
                "SELECT constraint_name FROM "
                "information_schema.table_constraints "
                "WHERE table_schema = 'main' "
                "AND table_name = 'l5_weights' "
                "AND constraint_type = 'PRIMARY KEY'"
            ).fetchall()
            if not pk_rows:
                # No PK? Treat as 2-tuple legacy (the
                # upsert will fall back to the
                # 2-tuple ON CONFLICT).
                self._l5_weights_pk_3tuple = False
                return False
            pk_name = pk_rows[0][0]
            col_rows = self._conn.execute(
                "SELECT column_name FROM "
                "information_schema.key_column_usage "
                "WHERE table_schema = 'main' "
                "AND table_name = 'l5_weights' "
                "AND constraint_name = ? "
                "ORDER BY ordinal_position",
                [pk_name],
            ).fetchall()
        except Exception:
            # The information_schema probe can fail on
            # some DBs (no information_schema; the table
            # was created with a different convention).
            # Fall back to the 3-tuple default: the
            # production schema has the 3-tuple PK, and
            # the ON CONFLICT will surface a clean
            # Binder error if the PK is actually 2-tuple.
            # The legacy-PK path is then exercised by the
            # caller (insert_l5_weights falls back to a
            # DELETE+INSERT).
            self._l5_weights_pk_3tuple = True
            return True
        col_set = {r[0] for r in col_rows}
        is_3tuple = "model_role" in col_set
        self._l5_weights_pk_3tuple = bool(is_3tuple)
        return self._l5_weights_pk_3tuple

    def insert_per_layer_error(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("per_layer_error_summary", PER_LAYER_ERROR_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("name", ""),
                r.get("layer"),
                r.get("epsilon"),
                r.get("reference_qtype", ""),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    # ---- Phase 0.5: EXL2 per-layer stats (additive table) ---------

    def _ensure_exl2_layer_stats(self) -> None:
        """Create the ``exl2_layer_stats`` table on first open.

        The Phase 0.5 EXL2 cross-check writes per-layer
        GPTQ-derived errors to this table. The schema is
        additive (a pre-Phase-0.5 DB has no such table; the
        ``CREATE TABLE IF NOT EXISTS`` creates it without
        touching the existing 7 tables). The PK is
        ``(model_hash, layer_index, exl2_calibration_corpus)``
        so multiple corpus runs (wikitext-103, COCO, the
        ``no_calibration_diagonal_unit`` fallback) coexist
        for the same model.

        Idempotent and cheap: ``CREATE TABLE IF NOT EXISTS``
        is a no-op on a subsequent open. Cached on the
        instance so the round-trip is paid once per
        ``TesseraDB`` lifetime, not once per insert. The
        C++ side mirrors the same statement in
        ``TS_QDB_SCHEMA_SQL``; on a C++-created DB the
        table is already present and the Python side's
        statement is a no-op.
        """
        if self._read_only:
            return
        if getattr(self, "_exl2_layer_stats_ensured", False):
            return
        try:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exl2_layer_stats (
                    model_hash              TEXT NOT NULL,
                    layer_index             INTEGER NOT NULL,
                    layer_name              TEXT,
                    family                  TEXT,
                    n_elements              BIGINT,
                    exl2_per_layer_error    DOUBLE,
                    exl2_per_layer_bpw      DOUBLE,
                    exl2_chosen_bpw         INTEGER,
                    exl2_calibration_corpus TEXT NOT NULL,
                    created_at              TIMESTAMP DEFAULT now(),
                    PRIMARY KEY (
                        model_hash, layer_index, exl2_calibration_corpus
                    )
                )
                """
            )
        except Exception as e:
            sys.stderr.write(
                f"tessera-db: CREATE TABLE exl2_layer_stats "
                f"failed: {e}\n"
            )
        self._exl2_layer_stats_ensured = True

    def insert_exl2_layer_stats(
        self,
        model_hash: str,
        rows: Sequence[dict],
        *,
        calibration_corpus: str | None = None,
    ) -> int:
        """Push rows into the ``exl2_layer_stats`` table.

        Bypasses the per-table buffer and uses a direct
        ``INSERT ... ON CONFLICT DO UPDATE`` because the
        table has a primary key on ``(model_hash,
        layer_index, exl2_calibration_corpus)``. The
        upsert pattern mirrors ``insert_tensor_stats``:
        a re-run against the same ``(model, layer,
        corpus)`` tuple overwrites the prior values
        without a manual delete (the audit trail is in
        the PK; the value reflects the most recent run).

        ``calibration_corpus`` may be supplied as a
        method-level default for callers that compute one
        run; per-row ``r["exl2_calibration_corpus"]``
        overrides it when both are present (the per-row
        form lets a single insert batch carry rows from
        multiple corpus runs). When neither is supplied,
        the insert fails closed: every row must declare
        which corpus it belongs to.

        Returns the number of rows upserted.
        """
        if not rows or self._read_only:
            return 0
        # Schema migration: ensure the table exists.
        self._ensure_exl2_layer_stats()
        sql = (
            "INSERT INTO exl2_layer_stats ("
            "  model_hash, layer_index, layer_name, family, "
            "  n_elements, exl2_per_layer_error, exl2_per_layer_bpw, "
            "  exl2_chosen_bpw, exl2_calibration_corpus"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (model_hash, layer_index, exl2_calibration_corpus) "
            "DO UPDATE SET "
            "  layer_name           = excluded.layer_name, "
            "  family               = excluded.family, "
            "  n_elements           = excluded.n_elements, "
            "  exl2_per_layer_error = excluded.exl2_per_layer_error, "
            "  exl2_per_layer_bpw   = excluded.exl2_per_layer_bpw, "
            "  exl2_chosen_bpw      = excluded.exl2_chosen_bpw, "
            "  created_at           = now()"
        )
        n_upserted = 0
        for r in rows:
            corpus = (
                r.get("exl2_calibration_corpus")
                or calibration_corpus
            )
            if not corpus:
                sys.stderr.write(
                    "tessera-db: insert_exl2_layer_stats row "
                    "missing exl2_calibration_corpus; "
                    "skipped\n"
                )
                continue
            try:
                self._conn.execute(sql, [
                    str(model_hash),
                    int(r.get("layer_index", 0)),
                    str(r.get("layer_name", "")),
                    str(r.get("family", "")) or None,
                    int(r.get("n_elements", 0) or 0),
                    r.get("exl2_per_layer_error"),
                    r.get("exl2_per_layer_bpw"),
                    int(r.get("exl2_chosen_bpw", 0) or 0),
                    str(corpus),
                ])
                n_upserted += 1
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: insert_exl2_layer_stats on "
                    f"({model_hash}, {r.get('layer_index')}, "
                    f"{corpus}) failed: {e}\n"
                )
        return n_upserted

    def get_exl2_per_layer_errors(
        self,
        model_hash: str,
        *,
        calibration_corpus: str | None = None,
    ) -> dict[int, float]:
        """Read the per-layer EXL2 errors for ``model_hash``.

        Returns ``{layer_index: per_layer_error}`` for the
        most recent row per (model_hash, layer_index). When
        ``calibration_corpus`` is given, the result is
        filtered to that corpus only. The L5 orchestrator
        reads this map to fold the EXL2 per-layer error
        into the per-tensor sensitivity score (the
        ``w_exl2 > 0`` path); the empty-dict return
        signals "EXL2 has not been run on this model" and
        the fold is skipped.

        The ``most recent row per layer`` clause is
        important: a re-run against the same
        ``(model, layer, corpus)`` updates the row
        in-place (the PK upsert), so the most-recent
        value is the only value. For multiple corpus
        runs, the caller passes ``calibration_corpus``
        to disambiguate.
        """
        if self._read_only is False and not getattr(
            self, "_exl2_layer_stats_ensured", False,
        ):
            self._ensure_exl2_layer_stats()
        params: list = [str(model_hash)]
        sql = (
            "SELECT layer_index, exl2_per_layer_error "
            "FROM exl2_layer_stats WHERE model_hash = ?"
        )
        if calibration_corpus is not None:
            sql += " AND exl2_calibration_corpus = ?"
            params.append(str(calibration_corpus))
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception:
            return {}
        return {int(li): float(err) for li, err in rows}

    # ---- Phase 16.7: per-component (model_role, name) covering index ----

    def _ensure_unified_indexes(self) -> None:
        """Create the 7 per-component covering indexes on the
        unified-schema tables.

        The composite PKs on the 7 affected tables
        (``tensor_stats``, ``l3_outlier_summary``,
        ``l4_probe_summary``, ``l5_plan_summary``,
        ``l4_plan_outcome``, ``l5_outcome``, ``l5_weights``)
        include ``model_role``, but the per-component query
        pattern ("give me all `attn_q` rows for role=`dflash`")
        would otherwise scan the PK left-to-right and walk every
        (model_hash, model_role, name) tuple. A secondary index
        on ``(model_role, name)`` (or ``(model_role, family)``
        for ``l5_weights``) lets DuckDB satisfy the query with
        an index seek.

        The C++ side's ``ts_tessera_db_open()`` runs the same
        ``CREATE INDEX IF NOT EXISTS`` statements on every open;
        on a C++-created DB the indexes already exist and the
        statements are no-ops. The mirror on the Python side
        keeps a Python-only-opened DB on the same schema (e.g.
        when the calibration pipeline opens a fresh DB before
        the C++ side has touched it).

        Idempotent and cheap: ``CREATE INDEX IF NOT EXISTS``
        short-circuits when the index is already present. Cached
        on the instance so the round-trip is paid once per
        ``TesseraDB`` lifetime, not once per insert.

        ``l5_weights`` is keyed on ``(model_role, family)`` (not
        ``name``) because the l5_weights row is the per-family
        retune verdict, not per-tensor.

        On a fresh Python-only-opened DB, none of the 7 tables
        exist yet (the C++ side owns the canonical schema
        creation; the Python side only writes into it). The
        helper probes ``information_schema.tables`` for each
        table name and skips the missing ones - a missing
        table is not an error, just "we will create the index
        the next time the table is created".
        """
        if getattr(self, "_unified_indexes_ensured", False):
            return
        if self._read_only:
            return
        # (table, columns) pairs. Matches the C++ side's
        # TS_QDB_SCHEMA_SQL index block. ``l5_weights`` is the
        # one outlier: it has no `name` column (the row is per
        # family), so the covering index is on
        # (model_role, family).
        index_specs = (
            ("tensor_stats",       "model_role, name"),
            ("l3_outlier_summary", "model_role, name"),
            ("l4_probe_summary",   "model_role, name"),
            ("l5_plan_summary",    "model_role, name"),
            ("l4_plan_outcome",    "model_role, name"),
            ("l5_outcome",         "model_role, name"),
            ("l5_weights",         "model_role, family"),
        )
        # Probe existing tables + columns once. A fresh
        # Python-only DB has none of the 7 tables; a
        # pre-Phase-16 DB has the tables but without
        # ``model_role`` (the migration's job to add). The
        # helper only creates indexes for tables that have
        # the indexed columns; missing tables and missing
        # columns are both silent no-ops. The C++ side runs
        # the migration as part of ``ts_tessera_db_open``;
        # the Python side assumes the migration has been run
        # (via ``migrate_model_role.migrate()``) before the
        # open.
        existing_tables = {
            r[0]
            for r in self._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        # Probe columns for the existing tables. Cache the
        # set of (table, column) pairs that exist; the
        # index loop then asks "are both model_role and
        # name (or model_role and family for l5_weights)
        # present?" before issuing the CREATE INDEX.
        existing_columns: dict[str, set[str]] = {}
        if existing_tables:
            placeholders = ", ".join(["?"] * len(existing_tables))
            rows = self._conn.execute(
                f"SELECT table_name, column_name FROM "
                f"information_schema.columns "
                f"WHERE table_schema = 'main' "
                f"AND table_name IN ({placeholders})",
                list(existing_tables),
            ).fetchall()
            for tname, cname in rows:
                existing_columns.setdefault(tname, set()).add(cname)
        for table, cols in index_specs:
            if table not in existing_tables:
                # Table does not exist yet; skip silently.
                # The C++ side will create the table + index
                # on the next open, and the next
                # TesseraDB.open() on this same file will
                # see the table and create the index.
                continue
            # model_role must exist (the migration's job
            # to add); the other column (name / family)
            # is part of the canonical schema so it is
            # always present on a fully-created table.
            tcols = existing_columns.get(table, set())
            if "model_role" not in tcols:
                # Pre-Phase-16 DB: the migration has not
                # been run. Skip silently; the migration
                # is a separate, explicit step on the
                # Python side. Once ``migrate()`` runs and
                # adds the column, the next TesseraDB.open()
                # will pick the table up here and create
                # the index.
                continue
            idx_name = _index_name_for(table, cols)
            try:
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table}({cols})"
                )
            except Exception as e:
                # Best-effort: log and continue so a single
                # broken index does not block the whole open.
                # The query will still work (just slower), and
                # a subsequent C++ open will retry the index
                # creation.
                sys.stderr.write(
                    f"tessera-db: CREATE INDEX {idx_name} "
                    f"ON {table}({cols}) failed: {e}\n"
                )
        self._unified_indexes_ensured = True

    # ---- read API ----------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Run a SELECT and return a polars DataFrame.

        The query is executed on the underlying duckdb connection.
        Use ``polars.read_database`` if you need a connection you
        own; this method is for the read-mostly case (analytics,
        warm-start projection, calibration rollup).
        """
        return self._conn.execute(sql).pl()

    def execute(self, sql: str) -> None:
        """Run a non-SELECT (DDL, INSERT, UPDATE). Best-effort; on
        read-only connections DuckDB raises."""
        self._conn.execute(sql)

    def table_names(self) -> list[str]:
        """List of table names in the DB. Useful for diagnostics
        (e.g. ``TesseraDB.table_names()`` to confirm the unified
        schema is in place after a fresh C++ open)."""
        return [r[0] for r in self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()]

    def buffer_stats(self) -> dict[str, TesseraDBBufferStats]:
        """Snapshot of every active buffer's stats. Keyed by table."""
        with self._buffer_lock:
            return {t: b.stats() for t, b in self._buffers.items()}

    # ---- lifecycle ---------------------------------------------------

    def close(self) -> None:
        """Close all buffers (sync-on-exit) and the underlying
        connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Close buffers first so their sync-on-exit drain hits the
        # still-open connection.
        with self._buffer_lock:
            bufs = list(self._buffers.values())
            self._buffers.clear()
        for b in bufs:
            b.close()
        # Crash-safe shutdown: force a CHECKPOINT before closing the
        # connection. DuckDB will checkpoint on a clean shutdown, but
        # not on a SIGKILL (jetsam) - the WAL is left on disk and the
        # next read-only open fails until something forces a flush.
        # Issuing CHECKPOINT here guarantees a clean .duckdb and no
        # stale .wal on every TesseraDB.close, including exception
        # paths that go through __exit__.
        try:
            if not self._read_only:
                self._conn.execute("CHECKPOINT")
        except Exception as exc:  # pragma: no cover - best effort
            sys.stderr.write(
                "tessera-db: warning: CHECKPOINT before close failed: "
                f"{exc!r}\n"
            )
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "TesseraDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---- internals ---------------------------------------------------

    def _buffer_for(
        self,
        table: str,
        schema_cols: Sequence[str],
    ) -> TesseraDBBuffer:
        with self._buffer_lock:
            b = self._buffers.get(table)
            if b is not None:
                return b
            b = TesseraDBBuffer(
                self._db_path,
                table,
                schema_cols=schema_cols,
                flush_threshold=self._config.flush_threshold,
                flush_interval_sec=self._config.flush_interval_sec,
                durable=self._config.durable,
            )
            self._buffers[table] = b
            return b


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision, matching the
    C++ ``ts_now_ts()`` helper's format (``YYYY-MM-DD HH:MM:SS``)
    but as a string DuckDB TIMESTAMP accepts."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _index_name_for(table: str, cols: str) -> str:
    """Return a deterministic index name for a (table, columns) pair.

    The C++ side uses ``TS_QDB_PHASE_16_7_INDEXES_SQL`` (see
    tessera-quantize-db.cpp) which names the 7 indexes after
    the table's *short* name. The mapping is fixed (the
    short names are documented in the C++ side's
    ``kTableIndex`` test fixture): ``tensor_stats`` ->
    ``tensor_stats``, ``l3_outlier_summary`` ->
    ``l3_outlier``, ``l4_probe_summary`` -> ``l4_probe``,
    ``l5_plan_summary`` -> ``l5_plan``, ``l4_plan_outcome``
    -> ``l4_outcome`` (NOT ``l4_plan``), ``l5_outcome`` ->
    ``l5_outcome``, ``l5_weights`` -> ``l5_weights``.

    The mapping is not derivable from a simple suffix-strip
    rule (``l4_plan_outcome`` strips to ``l4_plan`` not
    ``l4_outcome``), so the test_helpers use a lookup
    table. ``l5_weights`` is the one outlier: it has no
    ``name`` column, so the index is on (model_role,
    family) and the name is ``idx_l5_weights_role_family``.
    """
    if table == "l5_weights":
        return "idx_l5_weights_role_family"
    short = {
        "tensor_stats":       "tensor_stats",
        "l3_outlier_summary": "l3_outlier",
        "l4_probe_summary":   "l4_probe",
        "l5_plan_summary":    "l5_plan",
        "l4_plan_outcome":    "l4_outcome",
        "l5_outcome":         "l5_outcome",
    }.get(table)
    if short is None:
        # Unknown table: fall back to the full name. This
        # is a no-op for the 7 affected tables; the lookup
        # is here so a future addition is graceful.
        short = table
    return f"idx_{short}_role_name"

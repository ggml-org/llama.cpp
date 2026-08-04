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
TENSOR_STATS_COLS: tuple[str, ...] = (
    "model_hash", "name", "family", "layer_depth",
    "out_dim", "in_dim", "n_elements", "dtype",
    "kurtosis", "eff_rank", "rms", "mean_abs", "tail_ratio",
    "source", "recommended_action", "updated_at",
)
L3_OUTLIER_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer", "sidecar_label",
    "outlier_count", "outlier_fraction", "max_abs", "rms",
    "updated_at",
)
L4_PROBE_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer",
    "current_qtype", "mse", "mse_minus_one", "perplexity",
    "top1_mismatch", "n_weights", "updated_at",
)
L4_PLAN_OUTCOME_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer", "iteration", "plan_id", "strategy",
    "alpha_before", "alpha_after", "clip_before", "clip_after",
    "outlier_thresh_before", "outlier_thresh_after",
    "mse_before", "mse_after", "frob_before", "frob_after",
    "family", "updated_at",
)
L5_PLAN_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer", "iteration", "plan_id",
    "sensitivity_score", "recommended_qtype", "recommended_alpha",
    "recommended_clip", "updated_at",
)
L5_OUTCOME_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer", "iteration", "plan_id",
    "family", "sensitivity_score", "recommended_alpha", "recommended_clip",
    "mse_before", "mse_after", "delta_mse", "delta_frob",
    "plan_accepted", "accept_threshold", "residual", "updated_at",
)
L5_WEIGHTS_COLS: tuple[str, ...] = (
    "model_hash", "family",
    "w_imatrix", "w_gradient", "w_layer",
    "bias", "n_samples", "in_sample_loss", "hit_rate",
    "retune_source", "updated_at",
)
PER_LAYER_ERROR_COLS: tuple[str, ...] = (
    "model_hash", "name", "layer",
    "epsilon", "reference_qtype", "updated_at",
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
        (model_hash, name); the buffer's plain INSERT would fail on
        a duplicate. The C++ side has the same primary key and the
        same upsert pattern (see ``tessera-quantize-db.cpp::
        ts_quantize_db_upsert_tensor_stat``).

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
            sql = (
                "INSERT INTO tensor_stats ("
                "  model_hash, name, family, layer_depth, "
                "  out_dim, in_dim, n_elements, dtype, "
                "  kurtosis, eff_rank, rms, mean_abs, tail_ratio, "
                "  source, recommended_action, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (model_hash, name) DO UPDATE SET "
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
                "  updated_at         = excluded.updated_at"
            )
            # COALESCE on the UPDATE means: if the new write's
            # column is NULL, keep the existing value (the other
            # side's contribution). If the new write has a value,
            # it wins. This is how the C++ kurtosis / eff_rank
            # survives a Python rms / mean_abs / tail_ratio
            # upsert (and vice versa); the same convention applies
            # to recommended_action (the calibration side's
            # l5_action verdict; the C++ side never writes it).
            try:
                self._conn.execute(sql, [
                    model_hash,
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
                ])
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: insert_tensor_stats on "
                    f"({model_hash}, {r.get('name', '')}) failed: {e}\n"
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
    ) -> int:
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l5_plan_summary", L5_PLAN_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("name", ""),
                r.get("layer"),
                r.get("iteration"),
                r.get("plan_id", ""),
                r.get("sensitivity_score"),
                r.get("recommended_qtype", ""),
                r.get("recommended_alpha"),
                r.get("recommended_clip"),
                r.get("updated_at", now),
            )
            buf.append(row)
        return len(rows)

    def insert_l4_plan_outcome(
        self,
        model_hash: str,
        rows: Sequence[dict],
    ) -> int:
        """Push rows into the ``l4_plan_outcome`` table. The C++
        dispatch's adaptive_requantize loop also writes here (one
        row per (tensor, gen)). ``rows`` is a list of dicts with
        keys: name, layer, iteration, plan_id, strategy,
        alpha_before, alpha_after, clip_before, clip_after,
        outlier_thresh_before, outlier_thresh_after, mse_before,
        mse_after, frob_before, frob_after, family.

        Returns the number of rows accepted.
        """
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l4_plan_outcome", L4_PLAN_OUTCOME_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
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
    ) -> int:
        """Push rows into the ``l5_outcome`` table. Written by
        ``tools/tessera/l5_outcome.py`` after joining
        ``l5_plan_summary`` and ``l4_plan_outcome``.

        ``rows`` is a list of dicts with keys: name, layer,
        iteration, plan_id, family, sensitivity_score,
        recommended_alpha, recommended_clip, mse_before,
        mse_after, delta_mse, delta_frob, plan_accepted,
        accept_threshold, residual.

        Returns the number of rows accepted.
        """
        if not rows or self._read_only:
            return 0
        buf = self._buffer_for("l5_outcome", L5_OUTCOME_COLS)
        now = _now_iso()
        for r in rows:
            row = (
                model_hash,
                r.get("name", ""),
                r.get("layer"),
                r.get("iteration"),
                r.get("plan_id", ""),
                r.get("family", ""),
                r.get("sensitivity_score"),
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

    # ---- write API: l5_weights (PRIMARY KEY -> direct upsert) ----

    def insert_l5_weights(
        self,
        rows: Sequence[dict],
    ) -> int:
        """Upsert rows into the ``l5_weights`` table (per-model,
        per-family retuned scoring weights).

        Bypasses the buffer and uses a direct ``INSERT ... ON
        CONFLICT (model_hash, family) DO UPDATE`` because the table
        has a primary key; the buffer's plain INSERT would fail on
        a duplicate. The C++ side has the same primary key and the
        same upsert pattern (see
        ``ts_tessera_db_upsert_l5_weight`` in
        ``tessera-quantize-db.cpp``).

        ``rows`` is a list of dicts with keys: model_hash, family,
        w_imatrix, w_gradient, w_layer, bias, n_samples,
        in_sample_loss, hit_rate, retune_source. The retune is
        the consumer-side half of the "did this requant plan
        reduce error?" feedback loop: ``tools/tessera/l5_retune.py``
        fits a per-(model, family) closed-form OLS on
        ``l5_outcome.delta_mse`` and ``l5_outcome.sensitivity_score``
        and writes the recommended
        ``(w_imatrix, w_gradient, w_layer)`` here. The
        orchestrator's next generation reads the table back via
        ``--retune-from-db``.

        Returns the number of rows upserted.
        """
        if not rows or self._read_only:
            return 0
        sql = (
            "INSERT INTO l5_weights ("
            "  model_hash, family, w_imatrix, w_gradient, w_layer, "
            "  bias, n_samples, in_sample_loss, hit_rate, "
            "  retune_source, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (model_hash, family) DO UPDATE SET "
            "  w_imatrix       = excluded.w_imatrix, "
            "  w_gradient      = excluded.w_gradient, "
            "  w_layer         = excluded.w_layer, "
            "  bias            = excluded.bias, "
            "  n_samples       = excluded.n_samples, "
            "  in_sample_loss  = excluded.in_sample_loss, "
            "  hit_rate        = excluded.hit_rate, "
            "  retune_source   = excluded.retune_source, "
            "  updated_at      = excluded.updated_at"
        )
        now = _now_iso()
        n = 0
        for r in rows:
            try:
                self._conn.execute(sql, [
                    r.get("model_hash", ""),
                    r.get("family", ""),
                    float(r.get("w_imatrix", 0.0)),
                    float(r.get("w_gradient", 0.0)),
                    float(r.get("w_layer", 0.0)),
                    r.get("bias"),
                    r.get("n_samples"),
                    r.get("in_sample_loss"),
                    r.get("hit_rate"),
                    r.get("retune_source", "ols_slope_v1"),
                    r.get("updated_at", now),
                ])
                n += 1
            except Exception as e:
                sys.stderr.write(
                    f"tessera-db: insert_l5_weights on "
                    f"({r.get('model_hash', '')}, {r.get('family', '')}) "
                    f"failed: {e}\n"
                )
        return n

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

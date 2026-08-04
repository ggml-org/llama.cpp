"""Phase 16: backfill migration for ``model_role`` on the 7
unified-schema tables.

The unified Gemma4 12B + dspark + dflash + MTP arch has tensors
with the same name in both the trunk and the drafter
(e.g. ``blk.0.attn_q.weight`` exists in both the trunk and the
dflash encoder). Phase 16 disambiguates them with a new
``model_role`` column on 7 tables:

  - tensor_stats
  - l3_outlier_summary
  - l4_probe_summary
  - l5_plan_summary
  - l4_plan_outcome
  - l5_outcome
  - l5_weights

The PK on each table is extended to include model_role. The
migration has two paths:

  1. **Fresh DB**: the new CREATE TABLE (in
     ``tessera-quantize-db.cpp`` and mirrored in
     ``test_tessera_db.py::SCHEMA_SQL``) already includes
     ``model_role``. ``migrate()`` short-circuits to a no-op.
  2. **Pre-Phase-16 DB**: the function runs the standard
     DuckDB PK-rebuild dance for each of the 7 tables:
     ``CREATE TABLE <name>__p16_new`` with the new schema ->
     ``INSERT INTO <name>__p16_new SELECT *, 'trunk' AS
     model_role FROM <name>`` -> ``DROP TABLE <name>`` ->
     ``ALTER TABLE <name>__p16_new RENAME TO <name>``. The
     existing rows are backfilled with ``model_role='trunk'``
     (the single-model-run default; the disambiguation only
     matters for new rows written by the dflash / dspark /
     mtp_nextn / shared_embd writers).

Idempotency: re-running on an already-migrated DB is a no-op
(the ``information_schema.columns`` check at the top of each
table's migration short-circuits when ``model_role`` is
present). Calling the function is safe to chain with the C++
side's own ``ts_tessera_db_migrate_model_role()`` migration
(also idempotent); whichever runs first migrates the schema,
the other no-ops.

The C++ side runs the equivalent migration on every
``ts_tessera_db_open()``; this script is the path for
Python-only-opened DBs (e.g. when the calibration pipeline
opens the DB before the C++ side has been touched). Both
sides use the same target schema (the CREATE TABLE statements
match the canonical ``TS_QDB_SCHEMA_SQL`` in
``tessera-quantize-db.cpp``).

Usage::

    # From a Python REPL or script:
    from tessera.migrate_model_role import migrate
    n = migrate("/path/to/tessera.duckdb")
    print(f"migrated: {n} rows backfilled to model_role='trunk'")

    # From the CLI:
    python3 tools/tessera/migrate_model_role.py \\
        --db /path/to/tessera.duckdb \\
        [--verbose]

Companion to ``docs/tessera-unified-db.md`` Phase 16.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence


# The 7 affected tables and their Phase 16 (post-migration)
# CREATE TABLE statements. Each entry is (table_name,
# create_new_sql, insert_select_sql). The order is leaf-first
# (the l3 / l4 / l5 outcome / plan / weight tables in their
# natural dependency order). The schema in each CREATE TABLE
# matches the canonical TS_QDB_SCHEMA_SQL in
# tessera-quantize-db.cpp (one source of truth; the C++ side
# and the Python side agree on the column list and the new PK).
_TABLES: Sequence[tuple[str, str, str]] = (
    (
        "tensor_stats",
        (
            "CREATE TABLE IF NOT EXISTS tensor_stats ("
            "    model_hash         TEXT NOT NULL,"
            "    model_role         TEXT NOT NULL DEFAULT 'trunk',"
            "    name               TEXT NOT NULL,"
            "    family             TEXT,"
            "    layer_depth        INTEGER,"
            "    out_dim            BIGINT,"
            "    in_dim             BIGINT,"
            "    n_elements         BIGINT,"
            "    dtype              TEXT,"
            "    kurtosis           DOUBLE,"
            "    eff_rank           DOUBLE,"
            "    rms                DOUBLE,"
            "    mean_abs           DOUBLE,"
            "    tail_ratio         DOUBLE,"
            "    source             TEXT,"
            "    recommended_action TEXT,"
            "    updated_at         TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name)"
            ")"
        ),
        (
            "INSERT INTO tensor_stats__p16_new "
            "(model_hash, name, family, layer_depth, out_dim, in_dim, "
            "n_elements, dtype, kurtosis, eff_rank, rms, mean_abs, "
            "tail_ratio, source, recommended_action, updated_at, model_role) "
            "SELECT model_hash, name, family, layer_depth, out_dim, in_dim, "
            "n_elements, dtype, kurtosis, eff_rank, rms, mean_abs, "
            "tail_ratio, source, recommended_action, updated_at, 'trunk' "
            "FROM tensor_stats"
        ),
    ),
    (
        "l3_outlier_summary",
        (
            "CREATE TABLE IF NOT EXISTS l3_outlier_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    sidecar_label     TEXT,"
            "    outlier_count     BIGINT,"
            "    outlier_fraction  DOUBLE,"
            "    max_abs           DOUBLE,"
            "    rms               DOUBLE,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, sidecar_label)"
            ")"
        ),
        (
            "INSERT INTO l3_outlier_summary__p16_new "
            "(model_hash, name, layer, sidecar_label, outlier_count, "
            "outlier_fraction, max_abs, rms, updated_at, model_role) "
            "SELECT model_hash, name, layer, sidecar_label, outlier_count, "
            "outlier_fraction, max_abs, rms, updated_at, 'trunk' "
            "FROM l3_outlier_summary"
        ),
    ),
    (
        "l4_probe_summary",
        (
            "CREATE TABLE IF NOT EXISTS l4_probe_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    current_qtype     TEXT,"
            "    mse               DOUBLE,"
            "    mse_minus_one     DOUBLE,"
            "    perplexity        DOUBLE,"
            "    top1_mismatch     DOUBLE,"
            "    n_weights         BIGINT,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name)"
            ")"
        ),
        (
            "INSERT INTO l4_probe_summary__p16_new "
            "(model_hash, name, layer, current_qtype, mse, mse_minus_one, "
            "perplexity, top1_mismatch, n_weights, updated_at, model_role) "
            "SELECT model_hash, name, layer, current_qtype, mse, mse_minus_one, "
            "perplexity, top1_mismatch, n_weights, updated_at, 'trunk' "
            "FROM l4_probe_summary"
        ),
    ),
    (
        "l5_plan_summary",
        (
            "CREATE TABLE IF NOT EXISTS l5_plan_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    iteration         INTEGER,"
            "    plan_id           TEXT,"
            "    sensitivity_score DOUBLE,"
            "    recommended_qtype TEXT,"
            "    recommended_alpha DOUBLE,"
            "    recommended_clip  DOUBLE,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")"
        ),
        (
            "INSERT INTO l5_plan_summary__p16_new "
            "(model_hash, name, layer, iteration, plan_id, sensitivity_score, "
            "recommended_qtype, recommended_alpha, recommended_clip, "
            "updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, sensitivity_score, "
            "recommended_qtype, recommended_alpha, recommended_clip, "
            "updated_at, 'trunk' "
            "FROM l5_plan_summary"
        ),
    ),
    (
        "l4_plan_outcome",
        (
            "CREATE TABLE IF NOT EXISTS l4_plan_outcome ("
            "    model_hash           TEXT NOT NULL,"
            "    model_role           TEXT NOT NULL DEFAULT 'trunk',"
            "    name                 TEXT NOT NULL,"
            "    layer                INTEGER,"
            "    iteration            INTEGER NOT NULL,"
            "    plan_id              TEXT NOT NULL,"
            "    strategy             TEXT,"
            "    alpha_before         DOUBLE,"
            "    alpha_after          DOUBLE,"
            "    clip_before          DOUBLE,"
            "    clip_after           DOUBLE,"
            "    outlier_thresh_before DOUBLE,"
            "    outlier_thresh_after  DOUBLE,"
            "    mse_before           DOUBLE,"
            "    mse_after            DOUBLE,"
            "    frob_before          DOUBLE,"
            "    frob_after           DOUBLE,"
            "    family               TEXT,"
            "    updated_at           TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")"
        ),
        (
            "INSERT INTO l4_plan_outcome__p16_new "
            "(model_hash, name, layer, iteration, plan_id, strategy, "
            "alpha_before, alpha_after, clip_before, clip_after, "
            "outlier_thresh_before, outlier_thresh_after, mse_before, "
            "mse_after, frob_before, frob_after, family, updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, strategy, "
            "alpha_before, alpha_after, clip_before, clip_after, "
            "outlier_thresh_before, outlier_thresh_after, mse_before, "
            "mse_after, frob_before, frob_after, family, updated_at, 'trunk' "
            "FROM l4_plan_outcome"
        ),
    ),
    (
        "l5_outcome",
        (
            "CREATE TABLE IF NOT EXISTS l5_outcome ("
            "    model_hash            TEXT NOT NULL,"
            "    model_role            TEXT NOT NULL DEFAULT 'trunk',"
            "    name                  TEXT NOT NULL,"
            "    layer                 INTEGER,"
            "    iteration             INTEGER NOT NULL,"
            "    plan_id               TEXT NOT NULL,"
            "    family                TEXT,"
            "    sensitivity_score     DOUBLE,"
            "    recommended_alpha     DOUBLE,"
            "    recommended_clip      DOUBLE,"
            "    mse_before            DOUBLE,"
            "    mse_after             DOUBLE,"
            "    delta_mse             DOUBLE,"
            "    delta_frob            DOUBLE,"
            "    plan_accepted         BOOLEAN,"
            "    accept_threshold      DOUBLE,"
            "    residual              DOUBLE,"
            "    imatrix_magnitude     DOUBLE,"
            "    gradient_proxy        DOUBLE,"
            "    layer_position_prior  DOUBLE,"
            "    updated_at            TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")"
        ),
        (
            "INSERT INTO l5_outcome__p16_new "
            "(model_hash, name, layer, iteration, plan_id, family, "
            "sensitivity_score, recommended_alpha, recommended_clip, "
            "mse_before, mse_after, delta_mse, delta_frob, plan_accepted, "
            "accept_threshold, residual, imatrix_magnitude, gradient_proxy, "
            "layer_position_prior, updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, family, "
            "sensitivity_score, recommended_alpha, recommended_clip, "
            "mse_before, mse_after, delta_mse, delta_frob, plan_accepted, "
            "accept_threshold, residual, imatrix_magnitude, gradient_proxy, "
            "layer_position_prior, updated_at, 'trunk' "
            "FROM l5_outcome"
        ),
    ),
    (
        "l5_weights",
        (
            "CREATE TABLE IF NOT EXISTS l5_weights ("
            "    model_hash            TEXT NOT NULL,"
            "    model_role            TEXT NOT NULL DEFAULT 'trunk',"
            "    family                TEXT NOT NULL,"
            "    w_imatrix             DOUBLE NOT NULL,"
            "    w_gradient            DOUBLE NOT NULL,"
            "    w_layer               DOUBLE NOT NULL,"
            "    bias                  DOUBLE,"
            "    n_samples             INTEGER,"
            "    in_sample_loss        DOUBLE,"
            "    hit_rate              DOUBLE,"
            "    retune_source         TEXT,"
            "    requant_budget_bits   BIGINT,"
            "    updated_at            TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, family)"
            ")"
        ),
        (
            "INSERT INTO l5_weights__p16_new "
            "(model_hash, family, w_imatrix, w_gradient, w_layer, bias, "
            "n_samples, in_sample_loss, hit_rate, retune_source, "
            "requant_budget_bits, updated_at, model_role) "
            "SELECT model_hash, family, w_imatrix, w_gradient, w_layer, bias, "
            "n_samples, in_sample_loss, hit_rate, retune_source, "
            "requant_budget_bits, updated_at, 'trunk' "
            "FROM l5_weights"
        ),
    ),
)


def _table_has_model_role(con, table_name: str) -> bool:
    """Return True iff ``table_name`` already has a ``model_role``
    column. Used as the idempotency guard: when the column is
    present (fresh DB or already-migrated), the rebuild is
    skipped to avoid an unnecessary data round-trip.
    """
    n = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = 'model_role'",
        [table_name],
    ).fetchone()[0]
    return n > 0


def _row_count(con, table_name: str) -> int:
    """Return the row count of ``table_name``. Used for the
    return value (number of rows backfilled) and for the
    idempotency check (a no-op migration is reported as 0
    even when the table has rows).
    """
    if not _table_exists(con, table_name):
        return 0
    return int(con.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0])


def _table_exists(con, table_name: str) -> bool:
    n = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = ?",
        [table_name],
    ).fetchone()[0]
    return n > 0


def migrate(db_path: str, verbose: bool = False) -> int:
    """Add ``model_role`` to the 7 affected tables; backfill
    existing rows with ``'trunk'``. Idempotent: re-running on
    an already-migrated DB is a no-op (the
    ``information_schema.columns`` check at the top of each
    table's migration short-circuits when ``model_role`` is
    present).

    Phase 16.7: when at least one table was actually migrated
    (i.e. a pre-Phase-16 DB was opened for the first time),
    the function also writes a ``<stem>.model_role_migration.json``
    sidecar next to the duckdb file. The sidecar records
    ``db_path``, ``model_role`` (always ``"trunk"`` for this
    migration), ``ts``, and a per-table ``n_rows_at_migration``
    count. The sidecar is the audit trail of what Phase 16
    did to a legacy DB. Re-running on an already-migrated DB
    is a no-op and the sidecar is not re-written (the
    existing file is left in place).

    Parameters
    ----------
    db_path : str
        Path to the ``tessera.duckdb`` file. The file is opened
        read-write; the caller is expected to have closed any
        other connections.
    verbose : bool, optional
        If True, log each table's migration step to stderr.

    Returns
    -------
    int
        The number of rows backfilled. For a fresh DB (no
        tables yet), the function returns 0. For a pre-Phase-16
        DB, the function returns the sum of row counts across
        the 7 affected tables. Re-running on an already-migrated
        DB returns 0 (the per-table idempotency check is the
        gate).

    Notes
    -----
    The migration is destructive on the old table's PRIMARY
    KEY: DuckDB does not support ``ALTER TABLE ... DROP
    CONSTRAINT`` in older versions, so the standard rebuild
    dance is used. The data is preserved (the INSERT ... SELECT
    backfills every existing row with ``model_role='trunk'``).
    """
    import duckdb

    con = duckdb.connect(db_path, read_only=False)
    try:
        total_backfilled = 0
        # Phase 16.7: per-table migration log. One entry per
        # table that actually needed the migration; the audit
        # sidecar is emitted only when at least one entry
        # lands here. Re-opens (no migration) leave the log
        # empty and skip the sidecar.
        migrated: list[tuple[str, int]] = []
        for table_name, create_new_sql, insert_select_sql in _TABLES:
            if not _table_exists(con, table_name):
                # Fresh DB: the table doesn't exist yet. The
                # migration's job is to add the column to an
                # existing table; the open path's CREATE TABLE
                # IF NOT EXISTS (in the C++ schema or in
                # tessera_db.py consumers) handles the fresh
                # case. We still apply the new CREATE TABLE
                # so a Python-only-opened fresh DB (one that
                # does not run the C++ schema setup) ends up
                # with the new schema. The CREATE TABLE IF NOT
                # EXISTS is a no-op on a subsequent run.
                if verbose:
                    sys.stderr.write(
                        f"migrate_model_role: {table_name} does not exist; "
                        f"applying Phase 16 schema\n"
                    )
                con.execute(create_new_sql)
                continue
            if _table_has_model_role(con, table_name):
                if verbose:
                    sys.stderr.write(
                        f"migrate_model_role: {table_name} already has "
                        f"model_role; no-op\n"
                    )
                continue
            # Pre-Phase-16: the column is missing. Run the
            # standard DuckDB PK-rebuild dance.
            n_rows = _row_count(con, table_name)
            tmp = table_name + "__p16_new"
            if verbose:
                sys.stderr.write(
                    f"migrate_model_role: rebuilding {table_name} "
                    f"({n_rows} row(s)) -> model_role='trunk'\n"
                )
            # 1. CREATE TABLE <name>__p16_new (new schema).
            # The CREATE statement has "IF NOT EXISTS
            # <name>"; rewrite to the temporary name.
            rewritten = create_new_sql.replace(
                "CREATE TABLE IF NOT EXISTS " + table_name + " (",
                "CREATE TABLE " + tmp + " (",
                1,
            )
            con.execute(rewritten)
            # 2. INSERT INTO <name>__p16_new SELECT ... FROM <name>.
            con.execute(insert_select_sql)
            # 3. DROP TABLE <name>.
            con.execute(f"DROP TABLE {table_name}")
            # 4. ALTER TABLE <name>__p16_new RENAME TO <name>.
            con.execute(f"ALTER TABLE {tmp} RENAME TO {table_name}")
            total_backfilled += n_rows
            migrated.append((table_name, n_rows))
        if migrated:
            _write_migration_sidecar(db_path, migrated, verbose=verbose)
        return total_backfilled
    finally:
        con.close()


def _sidecar_path(db_path: str) -> str:
    """Compute the audit sidecar path. ``foo.duckdb`` ->
    ``foo.model_role_migration.json``. The stem split is
    Windows / POSIX portable (split at the last dot-after-slash).
    In-memory ``:memory:`` returns an empty string; callers
    must short-circuit on it.
    """
    if not db_path or db_path == ":memory:":
        return ""
    slash = max(db_path.rfind("/"), db_path.rfind("\\"))
    dot = db_path.rfind(".")
    if dot > slash and dot != -1:
        stem = db_path[:dot]
    else:
        stem = db_path
    return stem + ".model_role_migration.json"


def _write_migration_sidecar(
    db_path: str,
    migrated: Sequence[tuple[str, int]],
    verbose: bool = False,
) -> None:
    """Write the Phase 16.7 audit sidecar next to the duckdb
    file. Format mirrors the C++ side (same JSON shape) so
    downstream tools can read either side's output:

        {
            "db_path": "<path>",
            "model_role": "trunk",
            "ts": "YYYY-MM-DD HH:MM:SS",
            "tables": [
                {"name": "tensor_stats", "n_rows_at_migration": 42},
                ...
            ]
        }

    Atomic write (write to ``<sidecar>.tmp``, then ``os.replace``
    onto the final name) so a crash mid-write cannot leave a
    half-written sidecar on disk.
    """
    target = _sidecar_path(db_path)
    if not target:
        return
    body = {
        "db_path": db_path,
        "model_role": "trunk",
        # Use a second-precision UTC stamp; matches the C++
        # side's ts_now_ts() format.
        "ts": _now_iso_seconds(),
        "tables": [
            {"name": t, "n_rows_at_migration": n}
            for t, n in migrated
        ],
    }
    tmp = target + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
            f.write("\n")
        os.replace(tmp, target)
    except OSError as e:
        # Sidecar is informational, not load-bearing. Log
        # the failure but do not raise: the migration itself
        # succeeded; only the audit trail is missing.
        if verbose:
            sys.stderr.write(
                f"migrate_model_role: sidecar write failed "
                f"({target}): {e}\n"
            )
        # Best-effort cleanup of the tmp file.
        try:
            os.remove(tmp)
        except OSError:
            pass


def _now_iso_seconds() -> str:
    """UTC timestamp at second precision, matching the C++
    side's ``ts_now_ts()`` format. Uses ``datetime`` (not
    the stdlib's lower-level ``time``) so the value is
    timezone-aware on every platform.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(
        description="Phase 16 migration: add model_role to the 7 "
                    "unified-schema tables. Idempotent.",
    )
    p.add_argument(
        "--db", required=True,
        help="Path to the tessera.duckdb file.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log each table's migration step to stderr.",
    )
    args = p.parse_args(argv)
    n = migrate(args.db, verbose=args.verbose)
    sys.stderr.write(
        f"migrate_model_role: {n} row(s) backfilled to model_role='trunk'\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

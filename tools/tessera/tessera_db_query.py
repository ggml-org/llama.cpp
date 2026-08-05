#!/usr/bin/env python3
"""Thin CLI shim for ``tessera_db.py``.

``tessera_db.py`` is the canonical Python interface to the unified
``tessera.duckdb`` store, but it is a *library*, not a CLI: it has no
``__main__`` and no ``argparse``. Tessera's Swift agent talks to the
store through ``TesseraDBQueryTool``, which shells out to a Python
script; this shim is that script.

Commands (all print a single JSON document to stdout):

  list_models [--db PATH] [--limit N]
      Return the distinct model_hashes present in tensor_stats along with
      the per-model tensor count, family coverage, and the
      recommended_action distribution. The Library view uses this to
      populate the model grid without doing directory globbing.

  list_tensor_stats [--db PATH] [--model-hash HASH] [--limit N]
      Return one row per (model_hash, model_role, name) with the
      calibration columns the Library / Inspect views care about.

  table_names [--db PATH]
      List the tables present in the DB (uses TesseraDB.table_names()).

  query [--db PATH] --sql "SELECT ..."
      Run a read-only SQL query and dump the result rows as JSON.

The shim is read-only: it opens the DB with ``read_only=True`` and
never invokes the write API. Calibration / backfill still goes
through the dedicated CLI tools (multimodal_calibrate.py,
backfill.py, etc.) which are the only callers permitted to write.

Exit code 0 on success; 2 if the DB does not exist; 3 if the query
returns no rows (for ``list_models`` and ``list_tensor_stats`` the
empty result is normal and exits 0 with ``rows: []``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

try:
    import duckdb  # type: ignore
except Exception as exc:  # pragma: no cover - duckdb is the runtime dep
    sys.stderr.write(f"tessera_db_query: duckdb import failed: {exc!r}\n")
    sys.exit(4)


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    p = Path(db_path)
    if not p.exists():
        sys.stderr.write(f"tessera_db_query: db not found: {db_path}\n")
        sys.exit(2)
    return duckdb.connect(db_path, read_only=True)


def _emit(payload: dict[str, Any]) -> None:
    """Print a single JSON document to stdout (one line for log scraping)."""
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_list_models(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()}
        if "tensor_stats" not in tables:
            _emit({"ok": True, "rows": [], "tables": sorted(tables),
                   "note": "tensor_stats table absent; DB has no calibration data yet"})
            return 0

        limit_clause = f"LIMIT {int(args.limit)}" if args.limit else ""
        sql = f"""
            SELECT
              model_hash,
              COUNT(*) AS tensor_count,
              COUNT(DISTINCT family) AS family_count,
              COUNT(DISTINCT model_role) AS role_count,
              MIN(updated_at) AS first_seen,
              MAX(updated_at) AS last_seen,
              SUM(CASE WHEN source = 'cpp_quant' THEN 1 ELSE 0 END) AS cpp_rows,
              SUM(CASE WHEN source = 'py_cal'   THEN 1 ELSE 0 END) AS py_rows
            FROM tensor_stats
            GROUP BY model_hash
            ORDER BY last_seen DESC
            {limit_clause}
        """
        rows = conn.execute(sql).fetchall()
        cols = [d[0] for d in conn.description]  # type: ignore[index]
        out = [dict(zip(cols, r)) for r in rows]
        _emit({"ok": True, "rows": out, "tables": sorted(tables),
               "count": len(out)})
        return 0
    finally:
        conn.close()


def cmd_list_tensor_stats(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        if "tensor_stats" not in {r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'").fetchall()}:
            _emit({"ok": True, "rows": [], "note": "tensor_stats table absent"})
            return 0
        where = []
        params: list[Any] = []
        if args.model_hash:
            where.append("model_hash = ?")
            params.append(args.model_hash)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        limit_clause = f"LIMIT {int(args.limit)}" if args.limit else ""
        sql = f"""
            SELECT model_hash, model_role, name, family, layer_depth,
                   out_dim, in_dim, n_elements, dtype,
                   kurtosis, eff_rank, rms, mean_abs, tail_ratio,
                   source, recommended_action, updated_at, backfill_count
            FROM tensor_stats
            {where_sql}
            ORDER BY model_hash, model_role, name
            {limit_clause}
        """
        rows = conn.execute(sql, params).fetchall()
        cols = [d[0] for d in conn.description]  # type: ignore[index]
        out = [dict(zip(cols, r)) for r in rows]
        _emit({"ok": True, "rows": out, "count": len(out)})
        return 0
    finally:
        conn.close()


def cmd_table_names(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        names = [r[0] for r in rows]
        _emit({"ok": True, "tables": names, "count": len(names)})
        return 0
    finally:
        conn.close()


def cmd_query(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        rows = conn.execute(args.sql).fetchall()
        cols = [d[0] for d in conn.description]  # type: ignore[index]
        out = [dict(zip(cols, r)) for r in rows]
        _emit({"ok": True, "rows": out, "count": len(out)})
        return 0
    except Exception as exc:
        sys.stderr.write(f"tessera_db_query: query failed: {exc!r}\n")
        return 3
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(
        prog="tessera_db_query",
        description="Read-only CLI shim for the unified tessera.duckdb store.",
    )
    p.add_argument("--db", default="tessera.duckdb",
                   help="Path to the unified tessera.duckdb file.")
    sub = p.add_subparsers(dest="cmd", required=True)

    lm = sub.add_parser("list_models",
                        help="Distinct model_hashes with per-model stats.")
    lm.add_argument("--db", default="tessera.duckdb",
                    help="Path to the unified tessera.duckdb file.")
    lm.add_argument("--limit", type=int, default=200)
    lm.set_defaults(func=cmd_list_models)

    lts = sub.add_parser("list_tensor_stats",
                         help="Per-tensor rows from tensor_stats.")
    lts.add_argument("--db", default="tessera.duckdb",
                     help="Path to the unified tessera.duckdb file.")
    lts.add_argument("--model-hash", default=None)
    lts.add_argument("--limit", type=int, default=2000)
    lts.set_defaults(func=cmd_list_tensor_stats)

    tn = sub.add_parser("table_names",
                        help="List tables in the DB.")
    tn.add_argument("--db", default="tessera.duckdb",
                    help="Path to the unified tessera.duckdb file.")
    tn.set_defaults(func=cmd_table_names)

    q = sub.add_parser("query", help="Run a read-only SQL query.")
    q.add_argument("--db", default="tessera.duckdb",
                   help="Path to the unified tessera.duckdb file.")
    q.add_argument("--sql", required=True)
    q.set_defaults(func=cmd_query)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

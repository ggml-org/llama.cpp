"""Python mirror of tessera-db-buffer.{h,cpp}.

Per-table write buffer for the unified ``tessera.duckdb`` store. Multiple
producer threads push rows via :py:meth:`TesseraDBBuffer.append`; a
single consumer (a daemon flusher thread) drains the queue and
bulk-inserts into DuckDB.

Flush triggers (whichever fires first):

* **count**: pending rows >= ``flush_threshold`` (default 65536)
* **time**:  since last flush >= ``flush_interval_sec`` (default 1.0)
* **explicit**: :py:meth:`TesseraDBBuffer.flush_now` called by the producer
* **shutdown**: :py:meth:`TesseraDBBuffer.close` forces a final drain
  (sync-on-exit, default ON)

Best-effort: a failed flush logs to ``sys.stderr``, increments
``stats.rows_dropped`` and ``stats.flush_failures``, and the producer
continues. The Python calibration pipeline treats DB logging as a
recording aid, never a correctness requirement; a corrupt DB or full
disk must never block calibration.

This is the Python-side counterpart to the C++ ``ts_db_buffer`` that
the ``tessera-dispatch`` binary uses for ``ga_evaluations``. Same
contract, same flush defaults, same observability counters (modulo
field naming). Companion to ``docs/tessera-polars-integration-scout.md``
and the unified-DB follow-up.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class TesseraDBBufferStats:
    """Snapshot of the buffer's observability counters.

    All fields are monotonic. The snapshot is eventually consistent
    (each load is independent, no cross-field transaction).
    """

    appended: int = 0
    flushed_rows: int = 0
    flushes: int = 0
    rows_dropped: int = 0
    flush_failures: int = 0
    microsec_in_flush: int = 0


class TesseraDBBuffer:
    """Per-table write buffer for parallel producers.

    Args:
        db_path: path to the DuckDB file. The buffer opens its own
            ``duckdb`` connection; multiple buffers pointing at the
            same file serialize their writes internally (DuckDB is
            single-writer per file). For the unified ``tessera.duckdb``,
            one ``TesseraDB`` instance holds one connection and
            many buffers; each buffer writes to one table.
        table: target table name (must already exist in the DB).
        schema_cols: column names in INSERT order. The buffer builds
            a polars DataFrame with these names and a single
            ``orient="row"`` row group, so the order is the INSERT
            order. The row width is checked at append time; a
            mismatched row is dropped.
        flush_threshold: row count that triggers an automatic flush
            (default 65536, matches the evidence-store
            ``row_group_size``).
        flush_interval_sec: idle interval that triggers a flush even
            if the count is not reached (default 1.0).
        durable: if True, every flush calls ``fsync`` before ack.
            Slow (10-100x for the hot path) but crash-safe. Default
            False; only enable for "I really cannot lose this row"
            workloads.
    """

    def __init__(
        self,
        db_path: str | Path,
        table: str,
        *,
        schema_cols: Sequence[str],
        flush_threshold: int = 65536,
        flush_interval_sec: float = 1.0,
        durable: bool = False,
    ) -> None:
        if not table:
            raise ValueError("table name must be non-empty")
        if not schema_cols:
            raise ValueError("schema_cols must be non-empty")
        if flush_threshold < 1:
            flush_threshold = 1

        # Lazy import so the module is importable in environments
        # without polars / duckdb installed (the calibration harness
        # container is one such case; see the polars scout §6 risk 4).
        import duckdb  # type: ignore
        import polars  # type: ignore

        self._duckdb = duckdb
        self._polars = polars
        self._db_path = str(db_path)
        self._table = table
        self._schema_cols = tuple(schema_cols)
        self._flush_threshold = flush_threshold
        self._flush_interval_sec = float(flush_interval_sec)
        self._durable = bool(durable)

        self._conn = duckdb.connect(self._db_path)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: list[tuple] = []
        self._stop = False
        self._flusher = threading.Thread(
            target=self._flusher_loop, name=f"tessera-db-buffer[{table}]",
            daemon=True,
        )

        # Atomic-style counters guarded by _lock (the read paths use
        # _lock too; a dedicated counter lock would only add
        # contention for the producer path).
        self._appended = 0
        self._flushed_rows = 0
        self._flushes = 0
        self._rows_dropped = 0
        self._flush_failures = 0
        self._microsec_in_flush = 0

        self._flusher.start()

    # ---- public API --------------------------------------------------

    def append(self, row: Sequence) -> None:
        """Push one row. Thread-safe.

        ``row`` is a sequence with one element per column declared in
        ``schema_cols``. The row is copied into the pending queue;
        the caller may free its source data after this returns. NULL
        is the special token ``None`` (it lands as SQL NULL).
        Mismatched row width is dropped (and ``rows_dropped``
        increments); the buffer stays alive.

        Best-effort: on the error paths the row is dropped and the
        counter is bumped. The flusher does not see the row, so
        there is no exception to catch.
        """
        if len(row) != len(self._schema_cols):
            sys.stderr.write(
                f"tessera-db-buffer: append size mismatch on '{self._table}' "
                f"(got {len(row)}, want {len(self._schema_cols)}); row dropped\n"
            )
            with self._lock:
                self._rows_dropped += 1
            return
        with self._cv:
            if self._stop:
                return
            # Convert to tuple so the caller can mutate the source
            # list without affecting the pending queue.
            self._pending.append(tuple(row))
            self._appended += 1
            if len(self._pending) >= self._flush_threshold:
                self._cv.notify_all()

    def flush_now(self) -> int:
        """Force an immediate flush. Thread-safe. Returns 0 on success."""
        with self._cv:
            if self._stop or not self._pending:
                return 0
            batch = self._pending
            self._pending = []
        return self._flush_batch(batch)

    def stats(self) -> TesseraDBBufferStats:
        """Snapshot the observability counters. Cheap."""
        with self._lock:
            return TesseraDBBufferStats(
                appended=self._appended,
                flushed_rows=self._flushed_rows,
                flushes=self._flushes,
                rows_dropped=self._rows_dropped,
                flush_failures=self._flush_failures,
                microsec_in_flush=self._microsec_in_flush,
            )

    def pending_for_test(self) -> int:
        """Current depth of the pending queue. Test-only."""
        with self._lock:
            return len(self._pending)

    def query(self, sql: str):
        """Run a SELECT through the buffer's connection. Returns a
        polars DataFrame.

        Tests and consumers that need to verify rows landed use
        this method instead of opening a second DuckDB connection
        to the same file (DuckDB forbids concurrent connections
        with different configurations). The buffer's own connection
        is read-write by default; reads are safe to interleave
        with the flusher's writes (DuckDB's MVCC).
        """
        import polars  # type: ignore
        return self._conn.execute(sql).pl()

    def count(self, table: str) -> int:
        """``SELECT COUNT(*) FROM <table>`` through the buffer's
        own connection. Convenience for tests / verification."""
        return int(self._conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0])

    def close(self) -> None:
        """Stop the flusher, drain the pending queue, close the connection.

        Idempotent. After close, ``append`` and ``flush_now`` are
        no-ops. The destructor also calls this, so an explicit close
        is optional but recommended for the cleanest shutdown.
        """
        with self._cv:
            if self._stop:
                return
            self._stop = True
            self._cv.notify_all()
        self._flusher.join()
        # Final drain: any rows that were enqueued between the
        # flusher's last pass and the stop signal. The flusher
        # already drained the queue on its way out, so this is
        # usually empty.
        with self._cv:
            tail = self._pending
            self._pending = []
        if tail:
            self._flush_batch(tail)
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "TesseraDBBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---- internals ---------------------------------------------------

    def _flusher_loop(self) -> None:
        while True:
            with self._cv:
                # wait_for returns True if the predicate is true
                # (stop or count) OR the timeout elapses (time
                # flush). We don't care which fired; the post-block
                # check handles all three cases.
                self._cv.wait_for(
                    lambda: self._stop
                            or len(self._pending) >= self._flush_threshold,
                    timeout=self._flush_interval_sec,
                )
                if self._stop:
                    return
                if not self._pending:
                    continue
                batch = self._pending
                self._pending = []
            self._flush_batch(batch)

    def _flush_batch(self, batch: Sequence[tuple]) -> int:
        if not batch:
            return 0
        t0 = time.monotonic_ns()
        # Build one INSERT ... VALUES (...), (...), ... statement.
        # This mirrors the C++ ts_db_buffer::flush_batch exactly.
        # Polars 1.x's write_database only accepts SQLAlchemy or
        # ADBC connections, not DuckDBPyConnection, so a raw INSERT
        # is the portable path. Each tuple's element is formatted:
        #   None            -> NULL
        #   bool            -> TRUE / FALSE
        #   int / float     -> numeric literal (unquoted)
        #   str             -> quoted with single quotes escaped
        try:
            cols_sql = ", ".join(self._schema_cols)
            values_sqls = []
            for row in batch:
                parts = []
                for v in row:
                    parts.append(_format_sql_value(v))
                values_sqls.append("(" + ", ".join(parts) + ")")
            sql = (
                f"INSERT INTO {self._table} ({cols_sql}) VALUES "
                + ", ".join(values_sqls)
            )
            self._conn.execute(sql)
            rc = 0
        except Exception as e:
            sys.stderr.write(
                f"tessera-db-buffer: INSERT on '{self._table}' failed: {e}\n"
            )
            rc = 1
        elapsed_us = (time.monotonic_ns() - t0) // 1000
        with self._lock:
            self._microsec_in_flush += int(elapsed_us)
            if rc == 0:
                self._flushed_rows += len(batch)
                self._flushes += 1
            else:
                self._flush_failures += 1
                self._rows_dropped += len(batch)
        if self._durable and rc == 0:
            try:
                self._conn.execute("CHECKPOINT")
            except Exception:
                pass
        return rc


def sql_escape(s: str) -> str:
    """SQL escape for text values, shared with the C++ side's
    ``ts_db_sql_escape`` (which doubles single quotes per the DuckDB
    convention). Sufficient for run_id, model_path, family,
    tensor_name, etc. that do not contain LIKE / backslash
    semantics.

    Numeric columns and the literal token ``"NULL"`` should be
    passed through unchanged; :py:meth:`TesseraDBBuffer.append` does
    not quote, so the caller's value reaches DuckDB verbatim. The
    buffer's :py:meth:`_flush_batch` builds the polars DataFrame
    with the declared schema, so the dtypes drive the DuckDB
    conversion (text columns are quoted by polars; numeric columns
    are not). The ``"NULL"`` token is therefore not needed in the
    Python implementation (use Python ``None`` instead).
    """
    return s.replace("'", "''")


def _format_sql_value(v) -> str:
    """Format one value for the bulk INSERT statement.

    Mirrors the C++ buffer's number-detection pass (``looks_like_int``
    / ``looks_like_float``) but with proper type dispatch (the Python
    side can introspect the value's actual type, which the C++ side
    cannot because it only sees pre-formatted strings). This lets
    the buffer correctly emit TRUE / FALSE for bools, NULL for None,
    unquoted numerics for int / float, and quoted text for str.
    """
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Numeric strings pass through unquoted (the C++ buffer does
    # this; the dispatch writes std::to_string(float) which produces
    # an unquoted numeric literal). bool and None are already
    # handled above.
    if s and (s.replace("-", "").replace("+", "").replace(".", "").replace("e", "").replace("E", "").isdigit()
              or _looks_like_float(s)):
        return s
    return "'" + sql_escape(s) + "'"


def _looks_like_float(s: str) -> bool:
    """Cheap 'looks like a number' check (matches the C++ buffer's
    ``looks_like_float`` helper for parity)."""
    if not s:
        return False
    seen_dot = False
    seen_e = False
    seen_digit = False
    i = 1 if s[0] in "+-" else 0
    for c in s[i:]:
        if c.isdigit():
            seen_digit = True
            continue
        if c == "." and not seen_dot and not seen_e:
            seen_dot = True
            continue
        if c in "eE" and not seen_e and seen_digit:
            seen_e = True
            continue
        return False
    return seen_digit

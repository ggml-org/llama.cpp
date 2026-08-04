"""Tests for tools/tessera/tessera_db_buffer.py.

Mirrors the C++ ``test_db_buffer.cpp`` cases:

  1. basic lifecycle (open / append / flush / close)
  2. count-based flush
  3. time-based flush
  4. sync-on-exit (close-time final drain)
  5. parallel producers
  6. failed flush (unknown table -> rows_dropped increments)
  7. append / flush after close is a safe no-op
  8. pending depth reflects enqueued rows

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

# Make the module importable when run from anywhere.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import duckdb
import polars as pl

from tessera_db_buffer import TesseraDBBuffer


def _fresh_path(idx: int) -> str:
    """Return a unique tmp path for one test. /tmp/ is fine on the
    dev machine; the test should run on any POSIX system."""
    return f"/tmp/tessera-db-buffer-py-test-{idx}.duckdb"


def _count_rows(db_path: str, table: str) -> int:
    """Count rows in ``table`` via a fresh read-write connection.

    The buffer's own connection is being written to by the
    flusher thread; sharing that connection with a reader thread
    is unsafe in duckdb-py (the writer's execute() can race the
    reader's fetchone() and produce a None result, observed in
    dev). The test opens its own short-lived connection to do the
    count, which is safe because the file is on disk and the
    writes are committed on each batch.

    The connection is opened in the same configuration as the
    buffer (read-write, no special flags) so DuckDB does not
    reject the second opener. We close it after each count to
    keep the connection count at 1.
    """
    con = duckdb.connect(db_path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def _make_schema(db_path: str) -> None:
    """Pre-create the tensor_stats schema the buffer writes to.
    The C++ ``tessera-quantize-db.cpp`` creates the same schema; we
    mimic it here so the test does not depend on the C++ binary.
    """
    con = duckdb.connect(db_path)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tensor_stats (
                model_hash   TEXT NOT NULL,
                name         TEXT NOT NULL,
                family       TEXT,
                layer_depth  INTEGER,
                out_dim      BIGINT,
                in_dim       BIGINT,
                n_elements   BIGINT,
                dtype        TEXT,
                kurtosis     DOUBLE,
                eff_rank     DOUBLE,
                rms          DOUBLE,
                mean_abs     DOUBLE,
                tail_ratio   DOUBLE,
                source       TEXT,
                updated_at   TIMESTAMP,
                PRIMARY KEY (model_hash, name)
            )
        """)
    finally:
        con.close()


class TestTesseraDBBuffer(unittest.TestCase):
    def setUp(self) -> None:
        # Pre-create the test schema for tests that need it. The
        # failed-flush test points at a non-existent table, so it
        # deliberately does NOT call this.
        self.db_paths: list[str] = []
        self._schemas: list[str] = []

    def tearDown(self) -> None:
        for p in self.db_paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for p in self._schemas:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _new_schema(self, idx: int) -> str:
        p = _fresh_path(idx)
        self._schemas.append(p)
        _make_schema(p)
        return p

    # ---- 1. Basic lifecycle -----------------------------------------

    def test_basic_lifecycle(self) -> None:
        db_path = self._new_schema(1)
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=32, flush_interval_sec=0.05,
        ) as buf:
            for i in range(5):
                buf.append(("hashA", f"tensor_{i}"))
            rc = buf.flush_now()
            self.assertEqual(rc, 0)
            n = _count_rows(db_path, "tensor_stats")
            self.assertEqual(n, 5)

    # ---- 2. Count-based flush ---------------------------------------

    def test_count_flush(self) -> None:
        db_path = self._new_schema(2)
        threshold = 100
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=threshold, flush_interval_sec=60.0,
        ) as buf:
            for i in range(threshold * 3 + 7):
                buf.append(("hashB", f"row_{i}"))
            # Wait for the flusher to drain. The count trigger fires
            # on the append that crosses the threshold; the flusher
            # drains the whole pending queue in one batch.
            for _ in range(200):
                time.sleep(0.05)
                if _count_rows(db_path, "tensor_stats") >= threshold * 3 + 7:
                    break
            n = _count_rows(db_path, "tensor_stats")
            self.assertEqual(n, threshold * 3 + 7)
            s = buf.stats()
            self.assertEqual(s.appended, threshold * 3 + 7)
            self.assertEqual(s.flushed_rows, threshold * 3 + 7)
            self.assertGreaterEqual(s.flushes, 1)
            self.assertEqual(s.rows_dropped, 0)

    # ---- 3. Time-based flush -----------------------------------------

    def test_time_flush(self) -> None:
        db_path = self._new_schema(3)
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=1_000_000, flush_interval_sec=0.15,
        ) as buf:
            buf.append(("hashC", "tensor_time"))
            time.sleep(0.15 * 4)
            n = _count_rows(db_path, "tensor_stats")
            self.assertEqual(n, 1)
            s = buf.stats()
            self.assertGreaterEqual(s.flushes, 1)
            self.assertEqual(s.appended, 1)
            self.assertEqual(s.flushed_rows, 1)

    # ---- 4. Sync-on-exit (close-time final drain) -------------------

    def test_sync_on_exit(self) -> None:
        db_path = self._new_schema(4)
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=1_000_000, flush_interval_sec=60.0,
        ) as buf:
            for i in range(7):
                buf.append(("hashD", f"sync_{i}"))
            # No explicit flush_now(); the context manager's __exit__
            # invokes close(), which runs the final drain.
        n = _count_rows(db_path, "tensor_stats")
        self.assertEqual(n, 7)

    # ---- 5. Parallel producers ---------------------------------------

    def test_parallel_producers(self) -> None:
        db_path = self._new_schema(5)
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=65536, flush_interval_sec=0.05,
        ) as buf:
            n_threads = 8
            per_thread = 50_000
            expected = n_threads * per_thread

            def worker(tid: int) -> None:
                for i in range(per_thread):
                    buf.append((f"hashE_{tid}", f"t{tid}_r{i}"))

            threads = [
                threading.Thread(target=worker, args=(t + 1,))
                for t in range(n_threads)
            ]
            t0 = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            t1 = time.monotonic()
            # Wait for both the table count AND the buffer's
            # flushed_rows counter to reach the expected value. The
            # counter is updated under the lock AFTER the SQL
            # execute returns, so the count can briefly be ahead of
            # the counter while the second big INSERT is committing
            # (the test's poll would otherwise exit early on the
            # count and assert on a stale counter). Polling both
            # closes that window.
            for _ in range(400):
                time.sleep(0.05)
                n_now = _count_rows(db_path, "tensor_stats")
                s_now = buf.stats()
                if n_now >= expected and s_now.flushed_rows >= expected:
                    break
            t2 = time.monotonic()
            n = _count_rows(db_path, "tensor_stats")
            s = buf.stats()
            self.assertEqual(n, expected, "all parallel rows landed")
            self.assertEqual(s.appended, expected, "appended count")
            self.assertEqual(s.flushed_rows, expected, "flushed_rows == appended")
            self.assertEqual(s.rows_dropped, 0, "no rows dropped under contention")
            print(
                f"    {expected} rows appended in "
                f"{(t1 - t0) * 1000:.0f} ms, all flushed in "
                f"{(t2 - t0) * 1000:.0f} ms"
            )

    # ---- 6. Failed flush (unknown table) ----------------------------

    def test_failed_flush(self) -> None:
        db_path = self._new_schema(6)
        threshold = 16
        with TesseraDBBuffer(
            db_path, "no_such_table_xyz",
            schema_cols=("model_hash", "name"),
            flush_threshold=threshold, flush_interval_sec=0.05,
        ) as buf:
            for i in range(3 * threshold):
                buf.append(("hashF", f"row_{i}"))
            time.sleep(0.3)
            buf.flush_now()  # should fail, return non-zero
            s = buf.stats()
            self.assertGreaterEqual(s.rows_dropped, 3 * threshold)
            self.assertGreaterEqual(s.flush_failures, 1)

    # ---- 7. Append / flush after close is a safe no-op --------------

    def test_append_after_close(self) -> None:
        db_path = self._new_schema(7)
        buf = TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=32, flush_interval_sec=0.05,
        )
        buf.close()
        # The buffer is closed but the handle is still around;
        # appends and flushes are no-ops. (The C-style API nulls
        # the handle on close; the Python version keeps the object
        # but ignores writes after close.)
        buf.append(("hashG", "should_not_land"))
        rc = buf.flush_now()
        self.assertEqual(rc, 0)
        n = _count_rows(db_path, "tensor_stats")
        self.assertEqual(n, 0)

    # ---- 8. Pending depth reflects enqueued rows -------------------

    def test_pending_depth(self) -> None:
        db_path = self._new_schema(8)
        with TesseraDBBuffer(
            db_path, "tensor_stats",
            schema_cols=("model_hash", "name"),
            flush_threshold=1_000_000, flush_interval_sec=60.0,
        ) as buf:
            for i in range(100):
                buf.append(("hashH", f"row_{i}"))
            self.assertEqual(buf.pending_for_test(), 100)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTesseraDBBuffer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

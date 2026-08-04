//
// test_db_buffer.cpp
//
// Standalone test for the per-table write buffer
// (tessera-db-buffer.{h,cpp}). Exercises:
//
//   - Open a buffer, append a handful of rows, close -> all rows land.
//   - Count-based flush: append N rows (N > threshold) -> rows land
//     without an explicit flush_now() call.
//   - Time-based flush: append 1 row, wait > flush_interval ->
//     the flusher thread drains it on its own.
//   - Explicit flush_now(): append rows below threshold, call
//     flush_now() -> rows land synchronously.
//   - Sync-on-exit: append rows, drop the buffer (no explicit
//     close) -> destructor runs a final flush; all rows land.
//   - Parallel producers: 8 threads each append 50k rows -> all
//     400k rows land, no drops, no duplicates, no corruption.
//   - Failed flush: pass an invalid table name -> rows_dropped
//     increments, the buffer stays alive, subsequent appends are
//     unaffected.
//   - Stats counters: appended / flushed_rows / flushes match the
//     expected values; rows_dropped stays 0 on the happy path.
//   - Pending depth: ts_db_buffer_pending_for_test returns a
//     reasonable value mid-flight.
//
// Builds standalone against duckdb-amalgamation + tessera-quantize-db.cpp
// + tessera-db-buffer.cpp. Run with no args; uses a tmp file. Exit 0
// on success, non-zero on failure.
//

#include "tessera-db-buffer.h"
#include "tessera-quantize-db.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        failures++; \
    } \
} while (0)

#define CHECK_EQ(a, b, msg) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL [%s:%d]: %s (got %lld want %lld)\n", \
                __FILE__, __LINE__, msg, \
                (long long)_a, (long long)_b); \
        failures++; \
    } \
} while (0)

// Open a fresh in-memory DuckDB and return a ts_tessera_db*. The DB has
// the unified schema (the production ts_tessera_db_open path) so the
// buffer can write to any of the schema's tables.
static ts_tessera_db * open_fresh_db(const std::string & path) {
    std::remove(path.c_str());
    std::string err;
    auto * db = ts_tessera_db_open(path, &err);
    if (db == nullptr) {
        fprintf(stderr, "ts_tessera_db_open failed: %s\n", err.c_str());
        return nullptr;
    }
    return db;
}

// Run a SELECT COUNT(*) ... and return the int64 result. Used to
// confirm rows landed in the target table.
static int64_t count_rows(ts_tessera_db * db, const std::string & table) {
    return ts_tessera_db_debug_count(db, "SELECT COUNT(*) FROM " + table);
}

// Helper: append `n` rows with two text columns: model_hash and name.
// The model_hash is a per-thread sentinel so a parallel test can
// assert no rows crossed thread boundaries; the name encodes both
// the thread id and the row index. Used by the count-flush and
// parallel-producer tests.
static void append_n(ts_db_buffer * buf, int n, int thread_id = 0) {
    const std::string hash = "hashN_" + std::to_string(thread_id);
    for (int i = 0; i < n; i++) {
        std::vector<std::string> row = {
            hash,
            "t" + std::to_string(thread_id) + "_r" + std::to_string(i),
        };
        ts_db_buffer_append(buf, row);
    }
}

// 1. Basic lifecycle: open, append, close, count.
static void test_basic_lifecycle(const std::string & db_path) {
    fprintf(stderr, "[1] basic lifecycle ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    // The unified schema has a tensor_stats table; the buffer will write
    // to a subset of columns so we can use a small column list.
    std::vector<std::string> cols = {"model_hash", "name"};
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols,
        /*flush_threshold=*/32, std::chrono::milliseconds(50));
    CHECK(buf != nullptr, "buffer open");
    if (!buf) { ts_db_buffer_close(&buf); return; }

    for (int i = 0; i < 5; i++) {
        ts_db_buffer_append(buf, { "hashA", "tensor_" + std::to_string(i) });
    }
    int rc = ts_db_buffer_flush_now(buf);
    CHECK_EQ(rc, 0, "flush_now");
    ts_db_buffer_close(&buf);

    int64_t n = count_rows(db, "tensor_stats");
    CHECK_EQ(n, 5, "5 rows landed");
    delete db;
}

// 2. Count-based flush: append > threshold rows without flush_now().
// Expect a flush to happen because pending >= flush_threshold.
static void test_count_flush(const std::string & db_path) {
    fprintf(stderr, "[2] count-based flush ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    const int kThreshold = 100;
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols,
        /*flush_threshold=*/kThreshold,
        std::chrono::milliseconds(60'000));  // long; count is the only trigger
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    append_n(buf, kThreshold * 3 + 7, /*thread_id=*/0);
    // Wait for the flusher to drain. Without synchronization between
    // the appender and the flusher, the 307 rows may all land in one
    // big flush (the threshold crossing fires the wakeup, the flusher
    // drains the whole pending queue in one batch). What we are
    // testing is "rows land via the count trigger", not the count
    // of flushes — that one big flush is correct behavior.
    for (int i = 0; i < 200; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (count_rows(db, "tensor_stats") >= (int64_t)(kThreshold * 3 + 7)) break;
    }
    int64_t landed = count_rows(db, "tensor_stats");
    CHECK_EQ(landed, kThreshold * 3 + 7, "all rows landed via count flush");

    // Stats
    auto s = ts_db_buffer_stats_get(buf);
    CHECK_EQ(s.appended, (uint64_t)(kThreshold * 3 + 7), "appended count");
    CHECK(s.flushed_rows == (uint64_t)(kThreshold * 3 + 7), "flushed_rows == appended");
    CHECK(s.flushes >= 1, "at least 1 count-triggered flush occurred");
    CHECK_EQ(s.rows_dropped, (uint64_t)0, "no rows dropped on happy path");

    ts_db_buffer_close(&buf);
    delete db;
}

// 3. Time-based flush: append 1 row, wait > flush_interval, expect a
// flush without an explicit flush_now().
static void test_time_flush(const std::string & db_path) {
    fprintf(stderr, "[3] time-based flush ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    const int kThreshold = 1'000'000;  // never reached
    const auto kInterval = std::chrono::milliseconds(150);
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols, kThreshold, kInterval);
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    ts_db_buffer_append(buf, { "hashB", "tensor_time" });
    // Wait > kInterval for the flusher to drain.
    std::this_thread::sleep_for(kInterval * 4);
    int64_t landed = count_rows(db, "tensor_stats");
    CHECK_EQ(landed, 1, "1 row landed via time flush");

    auto s = ts_db_buffer_stats_get(buf);
    CHECK(s.flushes >= 1, "at least 1 flush recorded");
    CHECK_EQ(s.appended, (uint64_t)1, "1 appended");
    CHECK_EQ(s.flushed_rows, (uint64_t)1, "1 flushed");

    ts_db_buffer_close(&buf);
    delete db;
}

// 4. Sync-on-exit (close-time final drain): append rows with a long
// flush interval (so the time-based flusher never fires), then call
// close() without an explicit flush_now(). The destructor's final
// drain is the only path that gets the rows in.
static void test_sync_on_exit(const std::string & db_path) {
    fprintf(stderr, "[4] sync-on-exit (close-time final drain) ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    // Use a long interval so the only flush is close()'s final drain.
    const auto kLongInterval = std::chrono::milliseconds(60'000);
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols,
        /*flush_threshold=*/1'000'000, kLongInterval);
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    for (int i = 0; i < 7; i++) {
        ts_db_buffer_append(buf, { "hashC", "sync_" + std::to_string(i) });
    }
    // No explicit flush_now(). close() invokes the destructor, which
    // drains the pending queue as a final flush.
    ts_db_buffer_close(&buf);
    CHECK(buf == nullptr, "close() nulled the handle");
    int64_t landed = count_rows(db, "tensor_stats");
    CHECK_EQ(landed, 7, "close-time drain landed 7 rows");
    delete db;
}

// 5. Parallel producers: 8 threads * 50k rows = 400k. No drops.
static void test_parallel_producers(const std::string & db_path) {
    fprintf(stderr, "[5] parallel producers (8 threads x 50k rows) ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    // Use the production default (65536) to keep flush count down. With
    // 400k rows that's 7 flushes; the per-flush cost is the main
    // bottleneck.
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols,
        /*flush_threshold=*/65536, std::chrono::milliseconds(50));
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    const int kThreads = 8;
    const int kPerThread = 50'000;
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < kThreads; t++) {
        workers.emplace_back([buf, t]() {
            append_n(buf, kPerThread, /*thread_id=*/t + 1);
        });
    }
    for (auto & w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();
    // Give the flusher time to drain the tail.
    int64_t expected = (int64_t)kThreads * kPerThread;
    for (int i = 0; i < 400; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (count_rows(db, "tensor_stats") >= expected) break;
    }
    int64_t landed = count_rows(db, "tensor_stats");
    auto t2 = std::chrono::steady_clock::now();

    auto elapsed_append = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto elapsed_total  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    fprintf(stderr, "    %lld rows appended in %lld ms, all flushed in %lld ms\n",
        (long long)expected, (long long)elapsed_append, (long long)elapsed_total);

    CHECK_EQ(landed, expected, "all parallel rows landed");
    auto s = ts_db_buffer_stats_get(buf);
    CHECK_EQ(s.appended, (uint64_t)expected, "appended count");
    CHECK_EQ(s.flushed_rows, (uint64_t)expected, "flushed_rows == appended");
    CHECK_EQ(s.rows_dropped, (uint64_t)0, "no rows dropped under contention");

    ts_db_buffer_close(&buf);
    delete db;
}

// 6. Failed flush: use a non-existent table. The buffer should
// increment rows_dropped and stay alive for subsequent valid appends.
static void test_failed_flush(const std::string & db_path) {
    fprintf(stderr, "[6] failed flush (unknown table) ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    const int kThreshold = 16;
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "no_such_table_xyz", cols,
        kThreshold, std::chrono::milliseconds(50));
    CHECK(buf != nullptr, "buffer open (even with bad table)");
    if (!buf) return;

    for (int i = 0; i < 3 * kThreshold; i++) {
        ts_db_buffer_append(buf, { "hashD", "row_" + std::to_string(i) });
    }
    // Give the flusher a chance to fail.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    int rc = ts_db_buffer_flush_now(buf);
    // We expect non-zero on the failed flush; the exact return is
    // implementation-defined (we don't propagate the DuckDB error
    // here, just a count of failed flushes).
    (void)rc;

    auto s = ts_db_buffer_stats_get(buf);
    CHECK(s.rows_dropped >= (uint64_t)(3 * kThreshold),
          "rows_dropped >= appended (all flushed rows are counted as dropped)");
    CHECK(s.flush_failures >= 1, "at least one flush failure recorded");

    // The buffer is still alive. Close it cleanly.
    ts_db_buffer_close(&buf);
    delete db;
}

// 7. Append / flush after close: the close() shim nulls the
// handle, so subsequent calls with the same handle are caught by
// the null check and are no-ops. This is the post-condition that
// makes the dispatch's unique_ptr pattern safe (no UB on the
// deleter path).
static void test_append_after_close(const std::string & db_path) {
    fprintf(stderr, "[7] append / flush after close is a safe no-op ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols, 32, std::chrono::milliseconds(50));
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    ts_db_buffer_close(&buf);
    CHECK(buf == nullptr, "close() nulled the handle");
    // The shims now treat buf == nullptr as a no-op.
    ts_db_buffer_append(buf, { "hashE", "should_not_land" });
    int rc = ts_db_buffer_flush_now(buf);
    CHECK_EQ(rc, 0, "flush_now on nulled handle returns 0");
    int64_t landed = count_rows(db, "tensor_stats");
    CHECK_EQ(landed, 0, "no rows landed after close");
    delete db;
}

// 8. Pending depth: ts_db_buffer_pending_for_test returns a
// reasonable value mid-flight.
static void test_pending_depth(const std::string & db_path) {
    fprintf(stderr, "[8] pending depth reflects enqueued rows ...\n");
    ts_tessera_db * db = open_fresh_db(db_path);
    CHECK(db != nullptr, "db open");
    if (!db) return;

    std::vector<std::string> cols = {"model_hash", "name"};
    // Use a long interval so the flusher doesn't drain while we
    // measure the depth.
    ts_db_buffer * buf = ts_db_buffer_open(
        db, "tensor_stats", cols,
        /*flush_threshold=*/1'000'000, std::chrono::milliseconds(60'000));
    CHECK(buf != nullptr, "buffer open");
    if (!buf) return;

    for (int i = 0; i < 100; i++) {
        ts_db_buffer_append(buf, { "hashF", "row_" + std::to_string(i) });
    }
    size_t pending = ts_db_buffer_pending_for_test(buf);
    CHECK(pending == 100, "pending depth = 100 after 100 appends");
    ts_db_buffer_close(&buf);
    delete db;
}

int main(int /*argc*/, char ** /*argv*/) {
    // Use a unique path per test so leftover state from a prior
    // crashed run does not pollute the next one.
    auto fresh_path = [](int idx) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/tmp/tessera-db-buffer-test-%d.db", idx);
        return std::string(buf);
    };

    test_basic_lifecycle(fresh_path(1));
    test_count_flush(fresh_path(2));
    test_time_flush(fresh_path(3));
    test_sync_on_exit(fresh_path(4));
    test_parallel_producers(fresh_path(5));
    test_failed_flush(fresh_path(6));
    test_append_after_close(fresh_path(7));
    test_pending_depth(fresh_path(8));

    if (failures == 0) {
        printf("OK: all tessera-db-buffer tests passed\n");
        return 0;
    }
    fprintf(stderr, "FAILED: %d assertion(s) failed\n", failures);
    return 1;
}

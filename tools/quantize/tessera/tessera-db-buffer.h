//
// tessera-db-buffer.h
//
// Per-table write buffer for the unified tessera.duckdb store. Multiple
// producer threads (the GA hot path has 16-64 parallel evaluators) push
// rows via append(); a single consumer thread (the flusher) drains the
// queue and bulk-inserts into DuckDB. The pattern extends the existing
// ts_tessera_db_appender design (one Appender per active tensor, batched
// flush per generation) to a unified, table-keyed, multi-producer /
// single-consumer shape that can drive any of the unified-schema tables
// (ga_evaluations, tensor_stats, l3_outlier_summary, l4_probe_summary,
// l5_plan_summary, per_layer_error_summary).
//
// Flush triggers (whichever fires first):
//   * row count:   pending_rows >= flush_threshold (default 65536)
//   * time:        since last flush >= flush_interval (default 1 second)
//   * explicit:    flush_now() called by the producer
//   * shutdown:    destructor forces a final flush (sync-on-exit, default ON)
//
// Best-effort: a failed flush logs to stderr, increments
// flush_failures and rows_dropped, and the producer continues. The
// dispatch treats DB logging as a recording aid, never a correctness
// requirement; a corrupt DB or full disk must never block quantization.
//
// Thread-safety: append() is safe to call from any number of producer
// threads simultaneously. flush_now() is also safe from any thread
// (the caller's thread takes the consumer's role for that flush).
// The destructor is single-threaded (the owning unique_ptr goes out of
// scope in a deterministic place).
//
// The buffer does NOT take ownership of ts_tessera_db. The dispatch
// owns the DB and tears it down after all buffers are closed. The
// dispatch uses unique_ptr<ts_db_buffer> with a custom deleter that
// calls ts_db_buffer_close first.
//

#pragma once

#include "tessera-quantize-db.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Observability counters. Read via ts_db_buffer_stats_get(). All fields
// are atomic; the snapshot is eventually consistent (each load is
// independent, no cross-field transaction).
struct ts_db_buffer_stats {
    uint64_t appended;         // rows successfully enqueued
    uint64_t flushed_rows;     // rows successfully written to DuckDB
    uint64_t flushes;          // number of flushes performed (count + time + explicit)
    uint64_t rows_dropped;     // rows dropped due to flush failures
    uint64_t flush_failures;   // flush attempts that returned non-zero
    uint64_t microsec_in_flush; // cumulative time in flush_locked (rough)
};

// Opaque buffer handle. Constructed via ts_db_buffer_open, torn down via
// ts_db_buffer_close. The close is idempotent (safe to call on a buffer
// that was already closed; the second call is a no-op).
struct ts_db_buffer;

// Open a buffer for `table_name` on the given DB. The buffer takes a
// reference to `column_names` (does not copy) and uses them to build
// the INSERT column list. `flush_threshold` defaults to 65536 rows;
// `flush_interval` defaults to 1 second; `durable` defaults to false
// (no fsync). Returns nullptr on allocation failure or invalid args.
ts_db_buffer * ts_db_buffer_open(
    ts_tessera_db * db,
    const std::string & table_name,
    const std::vector<std::string> & column_names,
    size_t flush_threshold = 65536,
    std::chrono::milliseconds flush_interval = std::chrono::milliseconds(1000),
    bool durable = false);

// Append one row. Thread-safe. The values are pre-formatted SQL
// literals (text columns already passed through sql_escape; numeric
// columns already passed through std::to_string). NULL is not
// supported; pass the empty string for nullable text columns or "NULL"
// as a literal. The row is copied into the pending queue; the caller
// may free its source data after the call returns. Best-effort: on
// allocation failure the row is dropped and rows_dropped is bumped.
void ts_db_buffer_append(ts_db_buffer * buf,
                         const std::vector<std::string> & values);

// Force an immediate flush. Thread-safe. The flusher thread is woken
// (or the caller's thread takes the consumer's role if the flusher is
// not yet running). Returns 0 on success, non-zero on flush failure.
// Calling flush_now() on a closed buffer is a no-op returning 0.
int ts_db_buffer_flush_now(ts_db_buffer * buf);

// Snapshot the observability counters. Cheap; atomic loads only.
ts_db_buffer_stats ts_db_buffer_stats_get(const ts_db_buffer * buf);

// Close the buffer. Idempotent. Stops the flusher thread, drains
// pending rows (final flush = sync-on-exit), and frees the handle.
// Sets *buf_p to nullptr on return so the caller can detect
// "this handle is no longer valid" without UB. Passing nullptr is
// a no-op.
void ts_db_buffer_close(ts_db_buffer ** buf_p);

// Test-only: the current depth of the pending queue. Not for
// production code paths (the value is stale by the time the caller
// reads it). Returns 0 if buf is null.
size_t ts_db_buffer_pending_for_test(const ts_db_buffer * buf);

// SQL escape helper, shared by the dispatch's eval_recorder and any
// other caller that builds a value list for ts_db_buffer_append.
// Doubles single quotes (DuckDB's escape convention); sufficient for
// run_id, model_path, family, tensor_name, etc. that do not contain
// LIKE / backslash semantics.
std::string ts_db_sql_escape(const std::string & s);

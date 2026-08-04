//
// tessera-db-buffer.cpp
//
// See tessera-db-buffer.h for the contract. This file holds the
// ts_db_buffer class implementation; the C-style public functions are
// thin shims around the class methods.
//
// Design notes:
//   - The producer (append) and the consumer (flusher_loop) share a
//     single mutex. The critical section is a swap of the pending
//     vector into a thread-local batch, which is O(1) regardless of
//     batch size.
//   - The flusher thread waits on a condition variable with the flush
//     interval as the timeout, so the time-based flush is correct
//     even when no producer ever calls append().
//   - flush_now() is implemented by setting a wakeup flag; the flusher
//     thread sees the flag on its next loop iteration and performs the
//     flush itself. This keeps the SQL execution on the flusher
//     thread (single writer to DuckDB) and lets the caller's thread
//     return immediately. The flag is cleared on the flusher thread.
//   - The destructor signals stop, joins the flusher, and runs one
//     final drain so sync-on-exit is honored even if close() was not
//     called explicitly (the dispatch's unique_ptr deleter handles
//     this).
//   - A failed flush (DuckDB throws or returns an error) logs a
//     warning, increments flush_failures, counts the batch's rows
//     as dropped, and continues. The next batch is not affected.
//

#include "tessera-db-buffer.h"

// duckdb.hpp pulls in windows.h on Windows; the MIN/MAX macros there clash
// with std::min/std::max. NOMINMAX keeps the standard library usable.
#ifdef _WIN32
#  define NOMINMAX
#endif
#include "duckdb.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <utility>

namespace {

// ts_db_buffer implementation. The C-style API in the .h is a shim.
class ts_db_buffer_impl {
public:
    ts_db_buffer_impl(ts_tessera_db * db,
                      std::string table_name,
                      std::vector<std::string> column_names,
                      size_t flush_threshold,
                      std::chrono::milliseconds flush_interval,
                      bool durable)
        : db_(db),
          table_name_(std::move(table_name)),
          column_names_(std::move(column_names)),
          column_list_(build_column_list(column_names_)),
          flush_threshold_(flush_threshold == 0 ? 1 : flush_threshold),
          flush_interval_(flush_interval),
          durable_(durable) {}

    ~ts_db_buffer_impl() {
        close();
    }

    void append(const std::vector<std::string> & values) {
        if (values.size() != column_names_.size()) {
            std::fprintf(stderr,
                "tessera-db-buffer: append size mismatch on '%s' "
                "(got %zu, want %zu); row dropped\n",
                table_name_.c_str(), values.size(), column_names_.size());
            stats_.rows_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (closed_) return;
            pending_.push_back(values);
            stats_.appended.fetch_add(1, std::memory_order_relaxed);
            if (pending_.size() >= flush_threshold_) {
                wakeup_due_to_count_ = true;
                cv_.notify_one();
            }
        }
    }

    int flush_now_public() {
        // Public synchronous flush: take a batch under the lock, then
        // run the INSERT on the caller's thread. The flusher thread
        // will see pending_ is empty and skip its next pass. This is
        // what the C-style ts_db_buffer_flush_now() shim calls so the
        // caller's thread blocks until the SQL is committed.
        std::vector<std::vector<std::string>> batch;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (closed_) return 0;
            if (pending_.empty()) return 0;
            batch.swap(pending_);
        }
        return flush_batch(batch);
    }

    int flush_now() {
        // Async flush: signal the flusher thread. Returns 0 immediately
        // if the flusher is awake. The flush completes asynchronously.
        // The C-style shim uses the synchronous variant; this one is
        // available for callers that want fire-and-forget.
        std::lock_guard<std::mutex> lk(mtx_);
        if (closed_) return 0;
        wakeup_explicit_ = true;
        cv_.notify_one();
        return 0;
    }

    ts_db_buffer_stats stats_snapshot() const {
        ts_db_buffer_stats s;
        s.appended        = stats_.appended.load(std::memory_order_relaxed);
        s.flushed_rows    = stats_.flushed_rows.load(std::memory_order_relaxed);
        s.flushes         = stats_.flushes.load(std::memory_order_relaxed);
        s.rows_dropped    = stats_.rows_dropped.load(std::memory_order_relaxed);
        s.flush_failures  = stats_.flush_failures.load(std::memory_order_relaxed);
        s.microsec_in_flush = stats_.microsec_in_flush.load(std::memory_order_relaxed);
        return s;
    }

    size_t pending_size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return pending_.size();
    }

    void start() {
        flusher_ = std::thread(&ts_db_buffer_impl::flusher_loop, this);
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (closed_) return;
            closed_ = true;
            cv_.notify_all();
        }
        if (flusher_.joinable()) flusher_.join();
        // Final drain: any rows that were enqueued between the last
        // flusher pass and the close() call. The flusher may have
        // already drained these, in which case pending_ is empty.
        std::vector<std::vector<std::string>> tail;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            tail.swap(pending_);
        }
        if (!tail.empty()) {
            int rc = flush_batch(tail);
            if (rc != 0) {
                std::fprintf(stderr,
                    "tessera-db-buffer: final flush on '%s' failed; "
                    "%zu rows dropped on exit\n",
                    table_name_.c_str(), tail.size());
            }
        }
    }

private:
    static std::string build_column_list(const std::vector<std::string> & cols) {
        std::string s;
        for (size_t i = 0; i < cols.size(); i++) {
            if (i) s += ", ";
            s += cols[i];
        }
        return s;
    }

    // DuckDB escapes single quotes by doubling them. Sufficient for
    // run_id, model_path, family, tensor_name, etc. The producer
    // already pre-escapes via the dispatch's helper, but we re-escape
    // defensively so a caller that forgets doesn't corrupt the SQL.
    static std::string sql_escape(const std::string & s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            if (c == '\'') out += "''";
            else           out += c;
        }
        return out;
    }

    // The flusher thread. Waits on the condition variable with the
    // flush interval as the timeout. Wakes on:
    //   * count: pending_ >= flush_threshold_ (set by append)
    //   * explicit: wakeup_explicit_ (set by flush_now)
    //   * interval: timeout (steady regardless of producer activity)
    //   * shutdown: closed_ flag
    void flusher_loop() {
        while (true) {
            std::vector<std::vector<std::string>> batch;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, flush_interval_, [this] {
                    return closed_ || wakeup_due_to_count_ || wakeup_explicit_;
                });
                if (closed_) {
                    // The destructor drains any remaining rows; nothing to
                    // do here.
                    return;
                }
                if (wakeup_due_to_count_ || wakeup_explicit_) {
                    wakeup_due_to_count_ = false;
                    wakeup_explicit_      = false;
                } else {
                    // Time-based wakeup. Drain whatever has accumulated.
                }
                if (pending_.empty()) continue;
                batch.swap(pending_);
            }
            flush_batch(batch);
        }
    }

    // Build one INSERT ... VALUES (...), (...) ... statement and execute
    // it. Returns 0 on success, non-zero on failure. Counts rows in the
    // batch as flushed or dropped based on the result.
    int flush_batch(const std::vector<std::vector<std::string>> & batch) {
        if (batch.empty()) return 0;
        auto t0 = std::chrono::steady_clock::now();
        std::ostringstream q;
        q << "INSERT INTO " << table_name_ << " (" << column_list_ << ") VALUES ";
        for (size_t i = 0; i < batch.size(); i++) {
            if (i) q << ", ";
            q << "(";
            for (size_t j = 0; j < batch[i].size(); j++) {
                if (j) q << ", ";
                const std::string & v = batch[i][j];
                if (v == "NULL") {
                    q << "NULL";
                } else if (looks_like_int(v) || looks_like_float(v)) {
                    q << v;
                } else {
                    q << "'" << sql_escape(v) << "'";
                }
            }
            q << ")";
        }
        int rc = 0;
        try {
            if (db_ == nullptr || db_->conn == nullptr) {
                rc = 1;
            } else {
                auto res = db_->conn->Query(q.str());
                if (res->HasError()) {
                    std::fprintf(stderr,
                        "tessera-db-buffer: INSERT on '%s' failed: %s\n",
                        table_name_.c_str(), res->GetError().c_str());
                    rc = 1;
                }
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr,
                "tessera-db-buffer: INSERT on '%s' threw: %s\n",
                table_name_.c_str(), e.what());
            rc = 1;
        } catch (...) {
            std::fprintf(stderr,
                "tessera-db-buffer: INSERT on '%s' threw unknown\n",
                table_name_.c_str());
            rc = 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        stats_.microsec_in_flush.fetch_add((uint64_t)us, std::memory_order_relaxed);
        if (rc == 0) {
            stats_.flushed_rows.fetch_add(batch.size(), std::memory_order_relaxed);
            stats_.flushes.fetch_add(1, std::memory_order_relaxed);
        } else {
            stats_.flush_failures.fetch_add(1, std::memory_order_relaxed);
            stats_.rows_dropped.fetch_add(batch.size(), std::memory_order_relaxed);
        }
        return rc;
    }

    // Cheap "looks like a number" check. Lets the bulk INSERT skip
    // the per-value quote-and-escape path. Returns true for ints,
    // floats, negatives, and the empty string (which we treat as a
    // quoted empty string, not a number, so NULL handling is the
    // caller's responsibility).
    static bool looks_like_int(const std::string & s) {
        if (s.empty()) return false;
        size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
        if (i >= s.size()) return false;
        for (; i < s.size(); i++) {
            if (s[i] < '0' || s[i] > '9') return false;
        }
        return true;
    }
    static bool looks_like_float(const std::string & s) {
        if (s.empty()) return false;
        bool seen_dot = false, seen_e = false, seen_digit = false;
        size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
        for (; i < s.size(); i++) {
            char c = s[i];
            if (c >= '0' && c <= '9') { seen_digit = true; continue; }
            if (c == '.') { if (seen_dot || seen_e) return false; seen_dot = true; continue; }
            if (c == 'e' || c == 'E') { if (seen_e || !seen_digit) return false; seen_e = true; continue; }
            return false;
        }
        return seen_digit;
    }

    ts_tessera_db * db_;
    const std::string table_name_;
    const std::vector<std::string> column_names_;
    const std::string column_list_;
    const size_t flush_threshold_;
    const std::chrono::milliseconds flush_interval_;
    const bool durable_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<std::vector<std::string>> pending_;
    bool closed_              = false;
    bool wakeup_due_to_count_ = false;
    bool wakeup_explicit_      = false;
    std::thread flusher_;

    mutable struct {
        std::atomic<uint64_t> appended{0};
        std::atomic<uint64_t> flushed_rows{0};
        std::atomic<uint64_t> flushes{0};
        std::atomic<uint64_t> rows_dropped{0};
        std::atomic<uint64_t> flush_failures{0};
        std::atomic<uint64_t> microsec_in_flush{0};
    } stats_;
};

} // namespace

// ---------------------------------------------------------------------------
// C-style public API (the .h is the contract; the shim is here).
// ---------------------------------------------------------------------------

ts_db_buffer * ts_db_buffer_open(
    ts_tessera_db * db,
    const std::string & table_name,
    const std::vector<std::string> & column_names,
    size_t flush_threshold,
    std::chrono::milliseconds flush_interval,
    bool durable)
{
    if (db == nullptr || table_name.empty() || column_names.empty()) {
        return nullptr;
    }
    auto * impl = new (std::nothrow) ts_db_buffer_impl(
        db, table_name, column_names,
        flush_threshold, flush_interval, durable);
    if (impl == nullptr) return nullptr;
    impl->start();
    return reinterpret_cast<ts_db_buffer *>(impl);
}

void ts_db_buffer_append(ts_db_buffer * buf,
                         const std::vector<std::string> & values) {
    if (buf == nullptr) return;
    auto * impl = reinterpret_cast<ts_db_buffer_impl *>(buf);
    impl->append(values);
}

int ts_db_buffer_flush_now(ts_db_buffer * buf) {
    if (buf == nullptr) return 0;
    auto * impl = reinterpret_cast<ts_db_buffer_impl *>(buf);
    // Synchronous flush on the caller's thread. The flusher will see
    // pending_ empty on its next pass and skip. See flush_now_public()
    // in the impl.
    return impl->flush_now_public();
}

ts_db_buffer_stats ts_db_buffer_stats_get(const ts_db_buffer * buf) {
    if (buf == nullptr) {
        return ts_db_buffer_stats{};
    }
    auto * impl = reinterpret_cast<const ts_db_buffer_impl *>(buf);
    return impl->stats_snapshot();
}

void ts_db_buffer_close(ts_db_buffer ** buf_p) {
    if (buf_p == nullptr || *buf_p == nullptr) return;
    auto * impl = reinterpret_cast<ts_db_buffer_impl *>(*buf_p);
    impl->close();
    delete impl;
    *buf_p = nullptr;
}

size_t ts_db_buffer_pending_for_test(const ts_db_buffer * buf) {
    if (buf == nullptr) return 0;
    auto * impl = reinterpret_cast<const ts_db_buffer_impl *>(buf);
    return impl->pending_size();
}

std::string ts_db_sql_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') out += "''";
        else           out += c;
    }
    return out;
}

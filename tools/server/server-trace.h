#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>

// Optional JSONL trace logger for all HTTP requests and responses.
//
// When enabled (non-empty directory), each request/response pair is logged as
// structured JSON records to a rotating daily file:
//   <trace_dir>/trace-YYYYMMDD.jsonl
//
// Record types sharing a request_id:
//   "request"    - emitted when the request arrives at the handler
//   "response"   - emitted when a non-streaming response is complete
//   "chunk"      - emitted per non-empty chunk for streaming responses
//   "stream_end" - emitted when a streaming response finishes
//
// Thread-safe: a mutex serialises all file writes.
// Zero cost when disabled: all methods check the enabled flag inline.
struct http_tracer {
    // dir must be a valid, existing directory path with a trailing separator.
    explicit http_tracer(const std::string & dir);
    ~http_tracer();

    // Returns a unique, roughly time-sortable request ID.
    std::string new_request_id();

    // Emit a "request" record. Always called before the handler runs, so the
    // request is logged even if the handler fails to produce a response.
    void log_request(
        const std::string & req_id,
        const std::string & method,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body
    );

    // Emit a "stream_start" record once it is known the response will stream.
    // Emitted immediately after the handler returns and before wrapping next().
    void log_stream_start(const std::string & req_id);

    // Emit a "response" record for a completed non-streaming response.
    void log_response(
        const std::string & req_id,
        int status,
        int64_t duration_ms,
        const std::string & body,
        const std::string & content_type,
        const std::map<std::string, std::string> & headers
    );

    // Replace next_fn with a wrapper that emits "chunk" records for each
    // non-empty chunk and a "stream_end" record when the stream closes.
    // Call this before handing the response to the HTTP layer.
    void wrap_streaming_response(
        const std::string & req_id,
        std::chrono::steady_clock::time_point t_start,
        std::function<bool(std::string &)> & next_fn
    );

private:
    std::string   trace_dir;
    std::mutex    mtx;
    std::ofstream trace_file;
    std::string   current_date; // YYYYMMDD of currently open file
    std::atomic<uint64_t> counter{0};

    // Open (or rotate) the trace file for the current date.
    // Must be called with mtx held.
    void ensure_file_open();

    // Append a single JSON line. Must be called with mtx held.
    void write_line(const std::string & line);
};

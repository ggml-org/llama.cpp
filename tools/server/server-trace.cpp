#include "server-trace.h"
#include "server-common.h" // json, safe_json_to_str, SRV_WRN, string_format

#include <algorithm>  // std::min, std::transform
#include <cctype>     // std::tolower
#include <cinttypes>  // PRIx64
#include <cstdio>     // std::snprintf
#include <ctime>      // std::strftime, gmtime_r / gmtime_s
#include <set>        // std::set

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// FNV-1a 64-bit hash. Fast, dependency-free, good enough for body fingerprints.
static uint64_t fnv1a_64(const std::string & data) {
    uint64_t h = UINT64_C(14695981039346656037); // offset basis
    for (unsigned char c : data) {
        h ^= static_cast<uint64_t>(c);
        h *= UINT64_C(1099511628211); // FNV prime
    }
    return h;
}

// Format a system_clock point as ISO-8601 UTC with milliseconds.
// Example: "2024-01-15T10:30:45.123Z"
static std::string format_iso_timestamp(std::chrono::system_clock::time_point tp) {
    auto t  = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;

    struct tm tm_buf = {};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char ts[28];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    char result[36];
    std::snprintf(result, sizeof(result), "%s.%03dZ", ts, (int)ms.count());
    return result;
}

// Format a system_clock point as YYYYMMDD (used for log file rotation).
static std::string format_date(std::chrono::system_clock::time_point tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    struct tm tm_buf = {};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[12];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm_buf);
    return buf;
}

// Return the first max_len printable characters of body as a UTF-8 safe excerpt.
// Non-printable bytes are replaced with '?'; trailing "..." added if truncated.
static std::string make_excerpt(const std::string & body, size_t max_len = 200) {
    if (body.empty()) {
        return "";
    }
    size_t len = std::min(body.size(), max_len);
    std::string result;
    result.reserve(len + 3);
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = body[i];
        if (c >= 0x20 && c < 0x7f) {
            result += static_cast<char>(c);
        } else if (c == '\t' || c == '\n' || c == '\r') {
            result += ' ';
        } else {
            result += '?';
        }
    }
    if (body.size() > max_len) {
        result += "...";
    }
    return result;
}

// Build a JSON object with a safe subset of the provided headers.
// Authorization / X-Api-Key values are replaced with "[REDACTED]" to avoid
// leaking credentials into the log files.
static json safe_headers(const std::map<std::string, std::string> & hdrs) {
    static const std::set<std::string> allow_verbatim = {
        "content-type", "content-length", "user-agent",
        "accept", "accept-encoding", "transfer-encoding"
    };
    static const std::set<std::string> allow_redacted = {
        "authorization", "x-api-key"
    };

    json out = json::object();
    for (const auto & [k, v] : hdrs) {
        std::string lk = k;
        std::transform(lk.begin(), lk.end(), lk.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (allow_verbatim.count(lk)) {
            out[lk] = v;
        } else if (allow_redacted.count(lk)) {
            out[lk] = "[REDACTED]";
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// http_tracer
// ---------------------------------------------------------------------------

http_tracer::http_tracer(const std::string & dir)
    : trace_dir(dir) {}

http_tracer::~http_tracer() = default;

std::string http_tracer::new_request_id() {
    // "<16-hex ms timestamp>-<8-hex monotonic counter>"
    // Lexicographically sortable by arrival time; unique within a process.
    auto now = std::chrono::system_clock::now();
    uint64_t ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
    );
    uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64 "-%08" PRIx64, ms, n & 0xFFFFFFFFULL);
    return buf;
}

void http_tracer::log_request(
    const std::string & req_id,
    const std::string & method,
    const std::string & path,
    const std::map<std::string, std::string> & headers,
    const std::string & body
) {
    // Emitted before calling the handler so the request is always recorded,
    // even if the handler fails. Whether the response streams is left implicit:
    // a "stream_start" record follows if streaming, a "response" record if not.
    auto now = std::chrono::system_clock::now();
    json rec = {
        {"type",       "request"},
        {"request_id", req_id},
        {"timestamp",  format_iso_timestamp(now)},
        {"method",     method},
        {"path",       path},
        {"headers",    safe_headers(headers)},
        {"body_size",  body.size()},
        {"body_hash",  string_format("0x%016" PRIx64, fnv1a_64(body))},
    };
    if (!body.empty()) {
        rec["body_excerpt"] = make_excerpt(body);
    }

    std::lock_guard<std::mutex> lock(mtx);
    ensure_file_open();
    write_line(safe_json_to_str(rec));
}

void http_tracer::log_stream_start(const std::string & req_id) {
    // Emitted once the handler has returned a streaming response, before the
    // first chunk is pulled. Marks the transition from request to stream phase.
    auto now = std::chrono::system_clock::now();
    json rec = {
        {"type",       "stream_start"},
        {"request_id", req_id},
        {"timestamp",  format_iso_timestamp(now)},
    };

    std::lock_guard<std::mutex> lock(mtx);
    ensure_file_open();
    write_line(safe_json_to_str(rec));
}

void http_tracer::log_response(
    const std::string & req_id,
    int status,
    int64_t duration_ms,
    const std::string & body,
    const std::string & content_type,
    const std::map<std::string, std::string> & headers
) {
    auto now = std::chrono::system_clock::now();
    // Always include content-type in the response headers
    json res_headers = safe_headers(headers);
    res_headers["content-type"] = content_type;

    json rec = {
        {"type",        "response"},
        {"request_id",  req_id},
        {"timestamp",   format_iso_timestamp(now)},
        {"status",      status},
        {"duration_ms", duration_ms},
        {"headers",     res_headers},
        {"body_size",   body.size()},
        {"body_hash",   string_format("0x%016" PRIx64, fnv1a_64(body))},
    };
    if (!body.empty()) {
        rec["body_excerpt"] = make_excerpt(body);
    }

    std::lock_guard<std::mutex> lock(mtx);
    ensure_file_open();
    write_line(safe_json_to_str(rec));
}

void http_tracer::wrap_streaming_response(
    const std::string & req_id,
    std::chrono::steady_clock::time_point t_start,
    std::function<bool(std::string &)> & next_fn
) {
    // Take ownership of the original generator
    auto orig_next = std::move(next_fn);

    // Shared state across chunk callbacks (called from one httplib worker thread,
    // but the lambda may outlive the outer scope, hence shared_ptr).
    //
    // seq counts only non-empty chunks so that:
    //   - chunk record seq numbers are dense (0, 1, 2, ...)
    //   - stream_end total_chunks == number of "chunk" records in the log
    // httplib may invoke next() with an empty result (e.g. a trailing flush
    // call), which must not skew the counters.
    auto seq         = std::make_shared<uint64_t>(0);
    auto total_bytes = std::make_shared<size_t>(0);

    next_fn = [this, req_id, t_start,
               orig_next = std::move(orig_next),
               seq, total_bytes]
              (std::string & chunk) mutable -> bool
    {
        bool has_next = orig_next(chunk);

        *total_bytes += chunk.size();

        if (!chunk.empty()) {
            uint64_t chunk_seq = (*seq)++;   // advance only for non-empty chunks
            auto now = std::chrono::system_clock::now();
            json rec = {
                {"type",       "chunk"},
                {"request_id", req_id},
                {"timestamp",  format_iso_timestamp(now)},
                {"seq",        chunk_seq},
                {"chunk_size", chunk.size()},
            };
            std::lock_guard<std::mutex> lock(mtx);
            ensure_file_open();
            write_line(safe_json_to_str(rec));
        }

        if (!has_next) {
            // Stream finished: emit a summary record with total stats
            auto t_end = std::chrono::steady_clock::now();
            int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_end - t_start).count();
            auto now = std::chrono::system_clock::now();
            json rec = {
                {"type",         "stream_end"},
                {"request_id",   req_id},
                {"timestamp",    format_iso_timestamp(now)},
                {"total_chunks", *seq},  // number of non-empty chunks logged
                {"total_bytes",  *total_bytes},
                {"duration_ms",  ms},
            };
            std::lock_guard<std::mutex> lock(mtx);
            ensure_file_open();
            write_line(safe_json_to_str(rec));
        }

        return has_next;
    };
}

void http_tracer::ensure_file_open() {
    // Must be called with mtx held.
    auto now  = std::chrono::system_clock::now();
    std::string date = format_date(now);

    if (date == current_date && trace_file.is_open()) {
        return; // Still writing to today's file
    }

    if (trace_file.is_open()) {
        trace_file.close();
    }

    std::string path = trace_dir + "trace-" + date + ".jsonl";
    // Open in append mode so multiple server restarts on the same day accumulate
    trace_file.open(path, std::ios::app | std::ios::out);
    if (!trace_file.is_open()) {
        SRV_WRN("http_tracer: failed to open trace file: %s\n", path.c_str());
        return;
    }
    current_date = date;
}

void http_tracer::write_line(const std::string & line) {
    // Must be called with mtx held and ensure_file_open() already done.
    if (!trace_file.is_open()) {
        return;
    }
    // flush() after each line keeps records visible immediately and avoids
    // partial-line corruption if the process is killed.
    trace_file << line << '\n';
    trace_file.flush();
}

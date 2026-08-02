// server-metrics.cpp: see server-metrics.h for the design rationale.
//
// ASCII only. No em-dash, no unicode arrows.

#include "server-metrics.h"

#include "log.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

// We use httplib for the optional OTLP/HTTP export path, matching the rest of
// the server. It is a header-only dependency already linked by llama-server.
// To keep server-context (the static lib) free of an httplib link dependency,
// the network send lives in a weak stub that the server binary overrides; see
// otel_http_post() below. If the symbol is unresolved we fall back to stderr.

namespace tessera_metrics {

// Bucket upper bounds in microseconds. The shape mirrors the vLLM histogram
// defaults: roughly a 1.5x ratio between successive bounds, covering
// sub-millisecond ITL up to ~30s prefill of a 128k prompt.
static const std::vector<int64_t> & k_default_bounds_us_v() {
    static const std::vector<int64_t> v = {
        100,        // 0.1 ms
        250,
        500,
        1000,       // 1 ms
        2500,
        5000,
        10000,      // 10 ms
        25000,
        50000,
        100000,     // 100 ms
        250000,
        500000,
        1000000,    // 1 s
        2500000,
        5000000,
        10000000,   // 10 s
        30000000,   // 30 s
    };
    return v;
}

const std::vector<int64_t> & default_bucket_bounds_us() {
    return k_default_bounds_us_v();
}

// ---------------------------------------------------------------------------
// latency_histogram
// ---------------------------------------------------------------------------

latency_histogram::latency_histogram(std::string name,
                                     std::string help,
                                     std::vector<int64_t> bucket_bounds_us,
                                     size_t window_samples)
    : name_(std::move(name)),
      help_(std::move(help)),
      bounds_us_(std::move(bucket_bounds_us)),
      window_capacity_(window_samples) {
    std::sort(bounds_us_.begin(), bounds_us_.end());
    // One counter per finite bound plus one for +Inf. std::atomic is not
    // movable, so the bucket array is heap-allocated and indexed manually.
    n_buckets_ = bounds_us_.size() + 1;
    buckets_   = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[n_buckets_]);
    for (size_t i = 0; i < n_buckets_; ++i) {
        buckets_[i].store(0, std::memory_order_relaxed);
    }
    if (window_capacity_ > 0) {
        window_.reserve(window_capacity_);
    }
}

size_t latency_histogram::bucket_index_for(const std::vector<int64_t> & bounds, int64_t v) {
    // Linear search is fine: bounds is ~17 entries. Keeps the hot path branch-
    // predictor friendly and avoids a binary-search call.
    for (size_t i = 0; i < bounds.size(); ++i) {
        if (v <= bounds[i]) {
            return i;
        }
    }
    return bounds.size(); // +Inf bucket
}

void latency_histogram::record(int64_t value_us) noexcept {
    if (value_us < 0) {
        value_us = 0;
    }
    // Cumulative buckets: every bound >= value gets incremented. We do the
    // increment lazily at read time (store the raw count in the matching bin,
    // accumulate forward in take_snapshot) so record() touches exactly one
    // cache line.
    const size_t idx = bucket_index_for(bounds_us_, value_us);
    buckets_[idx].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_us_.fetch_add(value_us, std::memory_order_relaxed);

    if (window_capacity_ == 0) {
        return;
    }
    // Sliding window append. The mutex is only taken on the inference thread
    // (record()) and the rare /metrics read, so contention is negligible.
    {
        std::lock_guard<std::mutex> lk(window_mu_);
        if (window_.size() < window_capacity_) {
            window_.push_back(value_us);
        } else {
            window_[window_head_] = value_us;
            window_head_ = (window_head_ + 1) % window_capacity_;
            if (window_head_ == 0) {
                window_filled_ = true;
            }
        }
    }
}

static double percentile_from_sorted(std::vector<int64_t> & sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    // Nearest-rank percentile (matches what most ops dashboards expect for
    // small sample sets).
    const size_t n = sorted.size();
    if (n == 1) {
        return (double) sorted[0];
    }
    double rank = q * (double) (n - 1);
    size_t lo = (size_t) rank;
    size_t hi = lo + 1 < n ? lo + 1 : lo;
    double frac = rank - (double) lo;
    return (double) sorted[lo] + frac * ((double) sorted[hi] - (double) sorted[lo]);
}

latency_histogram::snapshot latency_histogram::take_snapshot() const {
    snapshot s;
    s.name = name_;
    s.help = help_;
    s.bucket_bounds_us = bounds_us_;
    // Convert per-bin counts to cumulative counts.
    s.bucket_counts.resize(n_buckets_, 0);
    uint64_t running = 0;
    for (size_t i = 0; i < n_buckets_; ++i) {
        running += buckets_[i].load(std::memory_order_relaxed);
        s.bucket_counts[i] = running;
    }
    s.count   = count_.load(std::memory_order_relaxed);
    s.sum_us  = sum_us_.load(std::memory_order_relaxed);

    if (window_capacity_ > 0) {
        std::vector<int64_t> w;
        {
            std::lock_guard<std::mutex> lk(window_mu_);
            w = window_;
        }
        std::sort(w.begin(), w.end());
        s.p50_us = percentile_from_sorted(w, 0.50);
        s.p95_us = percentile_from_sorted(w, 0.95);
        s.p99_us = percentile_from_sorted(w, 0.99);
    } else {
        // No window: derive percentiles from cumulative buckets. Coarser but
        // still useful. Linear-interpolate within the bucket that crosses the
        // quantile rank.
        const uint64_t total = s.count;
        auto pct_from_buckets = [&](double q) -> double {
            if (total == 0) return 0.0;
            const double target = q * (double) total;
            int64_t prev_bound = 0;
            uint64_t prev_count = 0;
            for (size_t i = 0; i < bounds_us_.size(); ++i) {
                if ((double) s.bucket_counts[i] >= target) {
                    uint64_t in_bucket = s.bucket_counts[i] - prev_count;
                    if (in_bucket == 0) {
                        return (double) bounds_us_[i];
                    }
                    double frac = (target - (double) prev_count) / (double) in_bucket;
                    return (double) prev_bound + frac * ((double) bounds_us_[i] - (double) prev_bound);
                }
                prev_bound = bounds_us_[i];
                prev_count = s.bucket_counts[i];
            }
            return (double) prev_bound; // landed in +Inf bucket
        };
        s.p50_us = pct_from_buckets(0.50);
        s.p95_us = pct_from_buckets(0.95);
        s.p99_us = pct_from_buckets(0.99);
    }
    return s;
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

registry::registry()
    : ttft_("time_to_first_token_seconds",
            "Time from request arrival to first generated token.",
            default_bucket_bounds_us()),
      itl_("inter_token_latency_seconds",
           "Wall time between two consecutive generated tokens for one request.",
           default_bucket_bounds_us()),
      e2e_("e2e_request_latency_seconds",
           "End-to-end request latency, arrival to final token.",
           default_bucket_bounds_us()),
      prefill_("request_prefill_time_seconds",
               "Prompt evaluation wall time per request.",
               default_bucket_bounds_us()),
      decode_("request_decode_time_seconds",
              "Decode iteration wall time.",
              default_bucket_bounds_us()) {}

registry::snapshots registry::take_snapshots() const {
    snapshots s;
    s.ttft    = ttft_.take_snapshot();
    s.itl     = itl_.take_snapshot();
    s.e2e     = e2e_.take_snapshot();
    s.prefill = prefill_.take_snapshot();
    s.decode  = decode_.take_snapshot();
    return s;
}

static void render_one(std::string & out, const latency_histogram::snapshot & s) {
    // Prometheus histogram exposition format. Metric family name uses the
    // llamacpp: prefix already adopted by the existing /metrics output so a
    // single scrape picks up both the old counters and these new histograms.
    const std::string prefix = "llamacpp:" + s.name;
    out += "# HELP " + prefix + " " + s.help + "\n";
    out += "# TYPE " + prefix + " histogram\n";
    for (size_t i = 0; i < s.bucket_bounds_us.size(); ++i) {
        char line[128];
        // bounds are us; published unit is seconds.
        const double bound_s = (double) s.bucket_bounds_us[i] / 1e6;
        snprintf(line, sizeof(line), "%s_bucket{le=\"%.6g\"} %llu\n",
                 prefix.c_str(), bound_s,
                 (unsigned long long) s.bucket_counts[i]);
        out += line;
    }
    {
        char line[128];
        snprintf(line, sizeof(line), "%s_bucket{le=\"+Inf\"} %llu\n",
                 prefix.c_str(), (unsigned long long) s.bucket_counts.back());
        out += line;
    }
    {
        char line[128];
        snprintf(line, sizeof(line), "%s_sum %.6f\n", prefix.c_str(),
                 (double) s.sum_us / 1e6);
        out += line;
    }
    {
        char line[128];
        snprintf(line, sizeof(line), "%s_count %llu\n", prefix.c_str(),
                 (unsigned long long) s.count);
        out += line;
    }
    // Also publish a P50/P95/P99 gauge alongside the buckets. Operators who
    // cannot run quantile_estimation_rate on Prometheus still get the numbers
    // directly. Suffix matches vLLM's "_p50/_p95/_p99" gauge family where one
    // exists; where vLLM only publishes the histogram, the gauge is additive.
    auto pct_gauge = [&](const char * suffix, double v_us) {
        char line[160];
        snprintf(line, sizeof(line), "llamacpp:%s%s %.6f\n",
                 s.name.c_str(), suffix, v_us / 1e6);
        out += line;
    };
    pct_gauge("_p50_seconds", s.p50_us);
    pct_gauge("_p95_seconds", s.p95_us);
    pct_gauge("_p99_seconds", s.p99_us);
}

void registry::render_prometheus(std::string & out) const {
    render_one(out, ttft_.take_snapshot());
    render_one(out, itl_.take_snapshot());
    render_one(out, e2e_.take_snapshot());
    render_one(out, prefill_.take_snapshot());
    render_one(out, decode_.take_snapshot());
}

// ---------------------------------------------------------------------------
// tracer
// ---------------------------------------------------------------------------

namespace {
void hex_encode(uint8_t * dst, const uint8_t * src, size_t n) {
    static const char * k_hex = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        dst[2 * i]     = (uint8_t) k_hex[src[i] >> 4];
        dst[2 * i + 1] = (uint8_t) k_hex[src[i] & 0x0f];
    }
}

bool hex_decode_byte(uint8_t & out, char hi, char lo) {
    auto nib = [](char c, uint8_t & v) -> bool {
        if (c >= '0' && c <= '9') { v = (uint8_t)(c - '0');      return true; }
        if (c >= 'a' && c <= 'f') { v = (uint8_t)(c - 'a' + 10); return true; }
        if (c >= 'A' && c <= 'F') { v = (uint8_t)(c - 'A' + 10); return true; }
        return false;
    };
    uint8_t h, l;
    if (!nib(hi, h) || !nib(lo, l)) {
        return false;
    }
    out = (uint8_t)((h << 4) | l);
    return true;
}
} // namespace

tracer::tracer(config cfg) : cfg_(std::move(cfg)) {}

void tracer::random_bytes(uint8_t * buf, size_t n) {
    // thread_local for cheap draws; the rng is seeded from the clock + address
    // entropy so two processes started together do not collide.
    static thread_local std::mt19937_64 rng{
        (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count()
        ^ (uint64_t) buf};
    for (size_t i = 0; i + sizeof(uint64_t) <= n; i += sizeof(uint64_t)) {
        uint64_t v = rng();
        std::memcpy(buf + i, &v, sizeof(v));
    }
    if (n % sizeof(uint64_t) != 0) {
        uint64_t v = rng();
        std::memcpy(buf + (n - (n % sizeof(uint64_t))), &v, n % sizeof(uint64_t));
    }
}

bool tracer::sampled() const {
    if (cfg_.sample_rate >= 1.0) return true;
    if (cfg_.sample_rate <= 0.0) return false;
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng) < cfg_.sample_rate;
}

span_handle tracer::start_root(const std::string & name, uint8_t trace_id_out[16]) {
    if (!cfg_.enabled || !sampled()) {
        return span_handle{};
    }
    span_descriptor d;
    d.name      = name;
    d.start_us  = 0; // filled by the handle
    d.duration_us = 0;
    tracer::random_bytes(d.trace_id, 16);
    tracer::random_bytes(d.span_id, 8);
    std::memset(d.parent_span_id, 0, 8);
    if (trace_id_out) {
        std::memcpy(trace_id_out, d.trace_id, 16);
    }
    return span_handle(this, std::move(d), std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

span_handle tracer::start_child(const std::string & name, const uint8_t parent_span_id[8]) {
    if (!cfg_.enabled || !sampled()) {
        return span_handle{};
    }
    span_descriptor d;
    d.name      = name;
    std::memset(d.trace_id, 0, 16);
    tracer::random_bytes(d.trace_id, 16); // a rootless child gets a fresh trace
    tracer::random_bytes(d.span_id, 8);
    if (parent_span_id) {
        std::memcpy(d.parent_span_id, parent_span_id, 8);
    } else {
        std::memset(d.parent_span_id, 0, 8);
    }
    return span_handle(this, std::move(d), std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

span_handle tracer::start_child(const std::string & name,
                                const uint8_t parent_trace_id[16],
                                const uint8_t parent_span_id[8]) {
    if (!cfg_.enabled || !sampled()) {
        return span_handle{};
    }
    span_descriptor d;
    d.name      = name;
    if (parent_trace_id) {
        std::memcpy(d.trace_id, parent_trace_id, 16);
    } else {
        tracer::random_bytes(d.trace_id, 16);
    }
    tracer::random_bytes(d.span_id, 8);
    if (parent_span_id) {
        std::memcpy(d.parent_span_id, parent_span_id, 8);
    } else {
        std::memset(d.parent_span_id, 0, 8);
    }
    return span_handle(this, std::move(d), std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
}

// Minimal JSON string escape. We only need to handle the characters that show
// up in span names / attribute values; quotes, backslashes, and control chars.
static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 4);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Optional weak override point for HTTP export. The default definition here
// is a no-op that returns false, so emit_locked() falls back to stderr when no
// override is linked in. A binary that links httplib (e.g. llama-server) may
// provide a strong definition that actually POSTs to the OTLP/HTTP endpoint;
// the weak attribute lets that override win at link time without changing the
// call site. Defining it here (rather than only declaring it) keeps the static
// library self-contained across all link environments.
__attribute__((weak)) bool otel_http_post(const std::string & /*endpoint*/,
                                          const std::string & /*ndjson_payload*/) {
    return false;
}


void tracer::emit_locked(const span_descriptor & d) {
    // One JSON object per line, newline-delimited. This is the format the
    // OpenTelemetry collector's "file" exporter reads, and is easy to forward
    // to OTLP/HTTP with a sidecar.
    char trace_hex[33];
    char span_hex[17];
    char parent_hex[17];
    hex_encode((uint8_t *) trace_hex, d.trace_id, 16); trace_hex[32] = 0;
    hex_encode((uint8_t *) span_hex,  d.span_id,  8);  span_hex[16]  = 0;
    hex_encode((uint8_t *) parent_hex, d.parent_span_id, 8); parent_hex[16] = 0;

    std::string line;
    line.reserve(256);
    line += "{\"name\":\"";
    line += json_escape(d.name);
    line += "\",\"traceId\":\"";
    line += trace_hex;
    line += "\",\"spanId\":\"";
    line += span_hex;
    line += "\",\"parentSpanId\":\"";
    // OTel convention: all-zero parent span id means "root" and is omitted.
    bool all_zero = true;
    for (int i = 0; i < 8; ++i) {
        if (d.parent_span_id[i] != 0) { all_zero = false; break; }
    }
    if (!all_zero) {
        line += parent_hex;
    }
    line += "\",\"startTimeUnixNano\":";
    char numbuf[32];
    snprintf(numbuf, sizeof(numbuf), "%llu", (unsigned long long)(d.start_us * 1000ull));
    line += numbuf;
    line += ",\"durationNanos\":";
    snprintf(numbuf, sizeof(numbuf), "%llu", (unsigned long long)(d.duration_us * 1000ull));
    line += numbuf;
    line += ",\"resource\":{\"service.name\":\"";
    line += json_escape(cfg_.service_name);
    line += "\"}";
    if (!d.attributes.empty()) {
        line += ",\"attributes\":[";
        bool first = true;
        for (const auto & kv : d.attributes) {
            if (!first) line += ",";
            first = false;
            line += "{\"key\":\"";
            line += json_escape(kv.first);
            line += "\",\"value\":{\"stringValue\":\"";
            line += json_escape(kv.second);
            line += "\"}}";
        }
        line += "]";
    }
    line += "}\n";

    if (!cfg_.endpoint.empty() && otel_http_post && otel_http_post(cfg_.endpoint, line)) {
        return;
    }
    // Fallback: stderr, one line per span. Flush so a live `tail -f` works.
    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

void tracer::emit(span_descriptor d) {
    if (!cfg_.enabled) return;
    std::lock_guard<std::mutex> lk(mu_);
    emit_locked(d);
}

bool tracer::parse_traceparent(const std::string & header,
                               uint8_t trace_id[16],
                               uint8_t parent_span_id[8],
                               uint8_t & flags) {
    // Format: "00-<32 hex trace>-<16 hex span>-<2 hex flags>"
    if (header.size() < 55) return false;
    if (header[2] != '-' || header[35] != '-' || header[52] != '-') return false;
    for (int i = 0; i < 16; ++i) {
        if (!hex_decode_byte(trace_id[i], header[3 + 2 * i], header[3 + 2 * i + 1])) return false;
    }
    for (int i = 0; i < 8; ++i) {
        if (!hex_decode_byte(parent_span_id[i], header[36 + 2 * i], header[36 + 2 * i + 1])) return false;
    }
    uint8_t f;
    if (!hex_decode_byte(f, header[53], header[54])) return false;
    flags = f;
    return true;
}

std::string tracer::format_traceparent(const uint8_t trace_id[16],
                                       const uint8_t span_id[8],
                                       uint8_t flags) {
    char out[56];
    char trace_hex[33];
    char span_hex[17];
    hex_encode((uint8_t *) trace_hex, trace_id, 16); trace_hex[32] = 0;
    hex_encode((uint8_t *) span_hex,  span_id,  8);  span_hex[16]  = 0;
    snprintf(out, sizeof(out), "00-%s-%s-%02x", trace_hex, span_hex, flags);
    return std::string(out);
}

// ---------------------------------------------------------------------------
// span_handle
// ---------------------------------------------------------------------------

span_handle::span_handle(class tracer * o, span_descriptor d, int64_t start_us)
    : owner_(o), desc_(std::move(d)), start_us_(start_us) {}

span_handle::span_handle(span_handle && o) noexcept
    : owner_(o.owner_), desc_(std::move(o.desc_)),
      start_us_(o.start_us_), finished_(o.finished_) {
    o.owner_   = nullptr;
    o.finished_ = true;
}

span_handle & span_handle::operator=(span_handle && o) noexcept {
    if (this != &o) {
        emit();
        owner_    = o.owner_;
        desc_     = std::move(o.desc_);
        start_us_ = o.start_us_;
        finished_ = o.finished_;
        o.owner_   = nullptr;
        o.finished_ = true;
    }
    return *this;
}

span_handle::~span_handle() {
    emit();
}

void span_handle::emit() {
    if (finished_) return;
    finished_ = true;
    if (!owner_) return;
    int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    desc_.start_us    = start_us_;
    desc_.duration_us = std::max<int64_t>(0, now_us - start_us_);
    owner_->emit(std::move(desc_));
    owner_ = nullptr;
}

void span_handle::set_attribute(const std::string & key, const std::string & value) {
    if (!owner_) return;
    desc_.attributes.emplace_back(key, value);
}

} // namespace tessera_metrics

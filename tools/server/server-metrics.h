#pragma once

// server-metrics: latency distributions + OpenTelemetry-compatible tracing.
//
// This is the observability layer called out as a secondary gap in the vLLM
// concurrency study (Section 12). The existing /metrics endpoint exposes
// Prometheus counters and instantaneous gauges but no latency *distributions*,
// so P50/P99 TTFT/ITL under load is invisible. vLLM V1 ships Prometheus
// histograms plus OpenTelemetry tracing.
//
// Two pieces live here:
//
//   1. latency_histogram - a fixed-bucket cumulative histogram with a sliding
//      ring of recent samples so percentiles reflect current load, not the
//      whole process lifetime. Renders as a Prometheus histogram in /metrics.
//
//   2. server_tracer - minimal W3C trace-context propagation + structured
//      JSON span export. We deliberately do NOT pull in the OpenTelemetry
//      C++ SDK: it is a large dependency (abseil, protobuf, gRPC...) that
//      conflicts with llama.cpp's single-static-binary portability goal.
//      Instead we implement the wire-format that OTel collectors accept
//      (W3C traceparent/tracestate headers + newline-delimited JSON spans)
//      which gives end-to-end tracing with zero new link-time deps. The
//      decision is documented in tools/server/README.md.
//
// ASCII only. No em-dash, no unicode arrows.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tessera_metrics {

// Default histogram boundaries, in microseconds, shared across TTFT / ITL /
// e2e / prefill / decode. The bucket layout is exponential-ish and matches the
// shape vLLM publishes (sub-millisecond to ~30s). Values outside the range
// land in the +Inf bucket.
//
// Microseconds (not seconds) keep integer arithmetic exact for ITL, which is
// routinely a few hundred us on Apple Silicon. The /metrics exporter divides
// by 1e6 so the published unit is seconds, matching vLLM's metric names.
const std::vector<int64_t> & default_bucket_bounds_us();

// Cumulative histogram with a sliding sample window.
//
// Thread-safety: a single instance is *not* expected to be hit from many
// threads at high rate (the server drives update_slots on one thread), but
// /metrics is served from an HTTP worker, so a mutex guards the read path.
// The write path uses a relaxed atomic count per bucket plus a mutex on the
// ring buffer append. Keep record() lock-free on the hot path.
class latency_histogram {
  public:
    // window_samples caps how many recent observations the in-memory
    // percentile estimate uses. 0 = use all samples since init (no sliding
    // window, pure cumulative like a Prometheus histogram).
    explicit latency_histogram(std::string name,
                               std::string help,
                               std::vector<int64_t> bucket_bounds_us,
                               size_t window_samples = 4096);

    // Record one observation (microseconds). Cheap: a bucket bin plus an
    // atomic bump. Ring-buffer append takes a short mutex.
    void record(int64_t value_us) noexcept;

    struct snapshot {
        std::string               name;
        std::string               help;
        std::vector<int64_t>      bucket_bounds_us;   // upper bounds (sorted)
        std::vector<uint64_t>     bucket_counts;      // cumulative (# <= bound)
        uint64_t                  count       = 0;    // cumulative observations
        int64_t                   sum_us      = 0;    // cumulative sum
        // Percentiles estimated over the sliding window (or all samples if no
        // window). These are *not* the cumulative buckets; they are the
        // near-realtime distribution the operator actually wants.
        double                    p50_us      = 0.0;
        double                    p95_us      = 0.0;
        double                    p99_us      = 0.0;
    };

    snapshot take_snapshot() const;

    const std::string & name() const { return name_; }

  private:
    static size_t bucket_index_for(const std::vector<int64_t> & bounds, int64_t v);

    std::string               name_;
    std::string               help_;
    std::vector<int64_t>      bounds_us_;        // sorted upper bounds, +Inf implied
    // Cumulative bucket counts are written from the inference thread and read
    // from the HTTP thread. Each entry is a distinct atomic so record() never
    // contends with another writer. Held in a unique_ptr array because
    // std::atomic is not movable/copyable and so cannot live in a vector.
    std::unique_ptr<std::atomic<uint64_t>[]> buckets_;
    size_t                              n_buckets_ = 0;
    std::atomic<uint64_t>               count_ { 0 };
    std::atomic<int64_t>                sum_us_ { 0 };

    // Sliding window for percentile estimation. Disabled when capacity_ == 0.
    const size_t            window_capacity_;
    mutable std::mutex      window_mu_;
    std::vector<int64_t>    window_;             // ring buffer
    size_t                  window_head_ = 0;    // next write position
    bool                    window_filled_ = false;
};

// Registry owning the canonical set of histograms. Constructed once by the
// server context. The /metrics handler pulls snapshots from here.
class registry {
  public:
    registry();

    // Lifecycle hooks called from the inference thread. All take microseconds
    // measured at the call site.
    //   TTFT: arrival -> first generated token.
    //   ITL:  gap between two consecutive generated tokens.
    //   e2e:  arrival -> final token.
    //   prefill: prompt evaluation wall time.
    //   decode:   one decode iteration wall time.
    void record_ttft(int64_t us)            { ttft_.record(us); }
    void record_itl(int64_t us)             { itl_.record(us); }
    void record_e2e(int64_t us)             { e2e_.record(us); }
    void record_prefill(int64_t us)         { prefill_.record(us); }
    void record_decode(int64_t us)          { decode_.record(us); }

    struct snapshots {
        latency_histogram::snapshot ttft;
        latency_histogram::snapshot itl;
        latency_histogram::snapshot e2e;
        latency_histogram::snapshot prefill;
        latency_histogram::snapshot decode;
    };

    snapshots take_snapshots() const;

    // Render all histograms in Prometheus exposition format (one block per
    // histogram: _bucket{le="..."} lines, _sum, _count). Matches vLLM's
    // metric names (time_to_first_token_seconds, inter_token_latency_seconds,
    // e2e_request_latency_seconds, request_prefill_time_seconds,
    // request_decode_time_seconds).
    void render_prometheus(std::string & out) const;

  private:
    latency_histogram ttft_;
    latency_histogram itl_;
    latency_histogram e2e_;
    latency_histogram prefill_;
    latency_histogram decode_;
};

// ---------------------------------------------------------------------------
// Tracing
//
// Minimal W3C trace-context + newline-delimited JSON span export. See the
// header comment for why we did not adopt the OTel C++ SDK.
//
// A trace is a tree of spans. Each span has:
//   - 16-byte trace_id (hex), shared by every span in the same request.
//   - 8-byte span_id (hex).
//   - 8-byte parent_span_id (hex), or all-zero for a root span.
//   - name, start_us, duration_us.
//   - a few string attributes.
//
// Spans are emitted to a configurable sink (stderr by default, or an OTLP/HTTP
// collector when --otel-endpoint is set). The wire format is one JSON object
// per line so a collector like `otel-collector --file` can ingest directly,
// or a sidecar can forward to OTLP/HTTP.

struct span_descriptor {
    std::string name;
    int64_t     start_us;
    int64_t     duration_us;
    uint8_t     trace_id[16];
    uint8_t     span_id[8];
    uint8_t     parent_span_id[8];
    std::vector<std::pair<std::string, std::string>> attributes;
};

// RAII handle returned by tracer::start_span. On destruction it records the
// duration and enqueues the span for emission.
class span_handle {
  public:
    span_handle() = default;  // an empty handle is a no-op (move-from or disabled tracer)
    span_handle(span_handle &&) noexcept;
    span_handle & operator=(span_handle &&) noexcept;
    span_handle(const span_handle &)            = delete;
    span_handle & operator=(const span_handle &) = delete;
    ~span_handle();

    void set_attribute(const std::string & key, const std::string & value);
    bool active() const { return owner_ != nullptr; }

  private:
    friend class tracer;
    span_handle(class tracer * o, span_descriptor d, int64_t start_us);
    void emit();

    class tracer *  owner_   = nullptr;
    span_descriptor desc_;
    int64_t         start_us_ = 0;
    bool            finished_ = false;
};

class tracer {
  public:
    struct config {
        bool        enabled          = false;
        std::string service_name     = "tessera-server";
        // OTLP/HTTP endpoint (e.g. "http://collector:4318/v1/traces"). If
        // empty, spans are written to stderr as newline-delimited JSON.
        std::string endpoint;
        // Sampling probability in [0,1]. 1.0 = record everything. Keeps the
        // hot path cheap when tracing is on but only a sample is needed.
        double      sample_rate      = 1.0;
    };

    tracer() = default;
    explicit tracer(config cfg);

    // Replace the configuration in place. Provided because tracer holds a
    // mutex and is therefore not copy/move-assignable, so `t = tracer(cfg)`
    // does not compile.
    void reconfigure(config cfg) {
        std::lock_guard<std::mutex> lk(mu_);
        cfg_ = std::move(cfg);
    }

    bool enabled() const { return cfg_.enabled; }

    // Start a root span for a new request. Returns an inactive handle if the
    // tracer is disabled or the request was dropped by sampling. The returned
    // 16-byte trace id is written into trace_id_out (used to propagate the
    // W3C traceparent back in the HTTP response, if desired).
    span_handle start_root(const std::string & name, uint8_t trace_id_out[16]);

    // Start a child span of an existing active span identified by its span id.
    span_handle start_child(const std::string & name, const uint8_t parent_span_id[8]);

    // Convenience: start a child of the root span whose id we hand out at
    // request entry. parent_trace_id is the request's trace id; the first
    // span created with that trace id becomes the implicit parent.
    span_handle start_child(const std::string & name, const uint8_t parent_trace_id[16],
                            const uint8_t parent_span_id[8]);

    // Emit a finished span. Called automatically by span_handle's destructor.
    void emit(span_descriptor d);

    // Parse a W3C traceparent header ("00-<trace_id>-<span_id>-<flags>"). On
    // success returns true and fills the output buffers; the inbound span_id
    // becomes the parent of the request's root span. Used so a caller's
    // trace is continued into the server.
    static bool parse_traceparent(const std::string & header,
                                  uint8_t trace_id[16],
                                  uint8_t parent_span_id[8],
                                  uint8_t & flags);

    // Format a traceparent header for outbound propagation.
    static std::string format_traceparent(const uint8_t trace_id[16],
                                          const uint8_t span_id[8],
                                          uint8_t flags);

    // Public helpers for callers that construct span descriptors directly
    // (e.g. to backdate a span to a known arrival time). Each fills the
    // buffer with cryptographically-random bytes drawn from the same source
    // the RAII handles use.
    static void new_trace_id(uint8_t trace_id[16]) { random_bytes(trace_id, 16); }
    static void new_span_id(uint8_t span_id[8])    { random_bytes(span_id, 8); }

  private:
    static void random_bytes(uint8_t * buf, size_t n);
    bool        sampled() const;
    void        emit_locked(const span_descriptor & d);

    config            cfg_;
    mutable std::mutex mu_;
    // Last-resort network send happens under mu_ so concurrent span finishes
    // do not interleave lines on stderr or clobber a socket.
};

// Optional override point for OTLP/HTTP span export. The default definition
// (in server-metrics.cpp) is a weak no-op that returns false, causing the
// tracer to fall back to stderr. A binary that links an HTTP client may
// provide a strong definition that actually POSTs the newline-delimited JSON
// payload to the endpoint; the weak attribute lets that override win.
bool otel_http_post(const std::string & endpoint, const std::string & ndjson_payload);

} // namespace tessera_metrics

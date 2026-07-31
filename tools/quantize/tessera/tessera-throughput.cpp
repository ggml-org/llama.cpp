#include "tessera-throughput.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// -------------------------------------------------------------------------
// workload loader

int ts_throughput_workload_load(const char * path,
                                ts_throughput_workload * out,
                                int max_workloads,
                                int * n_workloads,
                                std::string * err_msg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err_msg) *err_msg = std::string("cannot open: ") + path;
        return -1;
    }
    json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        if (err_msg) *err_msg = std::string("json parse: ") + e.what();
        return -1;
    }

    if (j.value("schema_version", 0) != 1) {
        if (err_msg) *err_msg = "unsupported schema_version (expected 1)";
        return -1;
    }
    if (!j.contains("workloads") || !j["workloads"].is_array()) {
        if (err_msg) *err_msg = "missing or non-array 'workloads'";
        return -1;
    }

    const auto & arr = j["workloads"];
    if ((int)arr.size() > max_workloads) {
        if (err_msg) *err_msg = "too many workloads (max " + std::to_string(max_workloads) + ")";
        return -1;
    }

    int n = 0;
    for (const auto & w : arr) {
        ts_throughput_workload & wl = out[n];
        std::memset(&wl, 0, sizeof(wl));

        const std::string name = w.value("name", "workload_" + std::to_string(n));
        snprintf(wl.name, sizeof(wl.name), "%s", name.c_str());

        const std::string regime = w.value("regime", "dflash");
        snprintf(wl.regime, sizeof(wl.regime), "%s", regime.c_str());

        wl.batch_size   = w.value("batch_size", 1);
        wl.seq_len      = w.value("seq_len", 512);
        wl.n_iterations = w.value("n_iterations", 10);

        if (wl.batch_size < 1 || wl.seq_len < 1 || wl.n_iterations < 1) {
            if (err_msg) *err_msg = "workload '" + name + "': batch_size, seq_len, n_iterations must be >= 1";
            return -1;
        }
        n++;
    }

    if (n_workloads) *n_workloads = n;
    return 0;
}

// -------------------------------------------------------------------------
// percentile helper (nearest-rank)

static double ts_percentile(std::vector<double> & v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t)std::ceil(pct / 100.0 * (double)v.size()) - 1;
    return v[std::min(idx, v.size() - 1)];
}

// -------------------------------------------------------------------------
// run

int ts_throughput_run(const ts_throughput_workload * workloads,
                      int n_workloads,
                      ts_throughput_infer_fn infer,
                      void * infer_ctx,
                      ts_throughput_result * results,
                      std::string * err_msg) {
    for (int i = 0; i < n_workloads; i++) {
        const ts_throughput_workload & wl = workloads[i];
        ts_throughput_result & r = results[i];
        std::memset(&r, 0, sizeof(r));
        snprintf(r.name,  sizeof(r.name),  "%s", wl.name);
        snprintf(r.regime, sizeof(r.regime), "%s", wl.regime);
        r.batch_size   = wl.batch_size;
        r.seq_len      = wl.seq_len;
        r.n_iterations = wl.n_iterations;
        r.stub         = (infer == nullptr);

        std::vector<double> latencies;
        latencies.reserve(wl.n_iterations);

        for (int it = 0; it < wl.n_iterations; it++) {
            double ms;
            if (infer) {
                ms = infer(wl.batch_size, wl.seq_len, infer_ctx);
                if (ms < 0.0) {
                    if (err_msg) *err_msg = std::string("inference failed on workload '") + wl.name + "'";
                    return -1;
                }
            } else {
                // Calibrated stub: ~0.02 ms/token so the timing plumbing is
                // exercisable without a model. Clearly marked stub=true.
                const double target_ms = 0.02 * wl.seq_len * wl.batch_size;
                const auto t0 = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::microseconds((long long)(target_ms * 1000)));
                const auto t1 = std::chrono::steady_clock::now();
                ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            latencies.push_back(ms);
        }

        double sum = 0.0;
        for (double v : latencies) sum += v;
        r.mean_latency_ms = sum / (double)latencies.size();
        r.p50_latency_ms  = ts_percentile(latencies, 50.0);
        r.p95_latency_ms  = ts_percentile(latencies, 95.0);

        // tokens/sec: total tokens across all iterations / total wall time
        const double total_tokens = (double)wl.batch_size * wl.seq_len * wl.n_iterations;
        const double total_ms     = sum;
        r.tokens_per_sec = (total_ms > 0.0) ? (total_tokens / total_ms * 1000.0) : 0.0;
    }
    return 0;
}

// -------------------------------------------------------------------------
// receipt writer

static std::string ts_throughput_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

int ts_throughput_receipt_write(const char * path,
                                const ts_throughput_result * results,
                                int n_results,
                                std::string * err_msg) {
    json j;
    j["schema"]    = "llama.tessera.throughput.v1";
    j["timestamp"] = ts_throughput_timestamp();

    json arr = json::array();
    for (int i = 0; i < n_results; i++) {
        const ts_throughput_result & r = results[i];
        json e;
        e["name"]            = r.name;
        e["regime"]          = r.regime;
        e["batch_size"]      = r.batch_size;
        e["seq_len"]         = r.seq_len;
        e["n_iterations"]    = r.n_iterations;
        e["tokens_per_sec"]  = r.tokens_per_sec;
        e["mean_latency_ms"] = r.mean_latency_ms;
        e["p50_latency_ms"]  = r.p50_latency_ms;
        e["p95_latency_ms"]  = r.p95_latency_ms;
        e["stub"]            = r.stub;
        arr.push_back(e);
    }
    j["results"] = arr;

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        if (err_msg) *err_msg = std::string("cannot open for write: ") + path;
        return -1;
    }
    f << j.dump(2) << "\n";
    if (!f) {
        if (err_msg) *err_msg = std::string("write failed: ") + path;
        return -1;
    }
    return 0;
}

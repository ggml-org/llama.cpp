#pragma once

//
// tessera-throughput.h
//
// North-star batched-throughput workload harness (docs/self-improving-loop-design.md).
// Measures tokens/sec under a set of named workloads (batch size x sequence
// length x drafting regime). The actual inference call is injected via
// ts_throughput_infer_fn; when NULL a calibrated stub is used so the timing
// and receipt infrastructure can be exercised without a loaded model.
//
// Workload input schema  (schema_version 1):
//   { "schema_version": 1,
//     "workloads": [
//       { "name": "b1s512", "batch_size": 1, "seq_len": 512,
//         "n_iterations": 10, "regime": "dflash" }, ...
//     ] }
//
// Receipt output schema:
//   { "schema": "llama.tessera.throughput.v1", "timestamp": "...",
//     "results": [ { "name":..., "tokens_per_sec":..., "stub": true, ... } ] }
//

#include <string>

#define TS_THROUGHPUT_MAX_WORKLOADS 64
#define TS_THROUGHPUT_NAME_LEN     128
#define TS_THROUGHPUT_REGIME_LEN    32

struct ts_throughput_workload {
    char name[TS_THROUGHPUT_NAME_LEN];
    char regime[TS_THROUGHPUT_REGIME_LEN];
    int  batch_size;
    int  seq_len;
    int  n_iterations;
};

struct ts_throughput_result {
    char   name[TS_THROUGHPUT_NAME_LEN];
    char   regime[TS_THROUGHPUT_REGIME_LEN];
    int    batch_size;
    int    seq_len;
    int    n_iterations;
    double tokens_per_sec;
    double mean_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;
    bool   stub;   // true when no real inference backend was supplied
};

// Inference callback: run one forward pass of batch_size sequences each of
// seq_len tokens. Return wall-clock milliseconds for the pass, or a negative
// value on error. ctx is caller-supplied (e.g. a loaded llama_context*).
typedef double (*ts_throughput_infer_fn)(int batch_size, int seq_len, void * ctx);

// Parse a workload file. Fills up to max_workloads entries; *n_workloads is
// set to the count actually read. Returns 0 on success, non-zero on error
// (message in *err_msg, *out left untouched).
int ts_throughput_workload_load(const char * path,
                                ts_throughput_workload * out,
                                int max_workloads,
                                int * n_workloads,
                                std::string * err_msg);

// Run all workloads. When infer is NULL a calibrated stub is used (stub=true
// in every result). results must point to at least n_workloads entries.
// Returns 0 on success, non-zero on error.
int ts_throughput_run(const ts_throughput_workload * workloads,
                      int n_workloads,
                      ts_throughput_infer_fn infer,
                      void * infer_ctx,
                      ts_throughput_result * results,
                      std::string * err_msg);

// Write the receipt JSON. Returns 0 on success, non-zero on error.
int ts_throughput_receipt_write(const char * path,
                                const ts_throughput_result * results,
                                int n_results,
                                std::string * err_msg);

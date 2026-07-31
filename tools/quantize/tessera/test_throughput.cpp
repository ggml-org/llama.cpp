// test_throughput.cpp - offline tests for tessera-throughput (no model needed)
#include "tessera-throughput.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

static std::string write_tmp(const char * name, const std::string & content) {
    std::string path = std::string("/tmp/ts_throughput_test_") + name;
    std::ofstream f(path, std::ios::binary);
    f << content;
    return path;
}

// -------------------------------------------------------------------------
// workload loader

static void test_workload_load_ok() {
    const std::string wl = R"({
        "schema_version": 1,
        "workloads": [
            {"name":"b1s512","batch_size":1,"seq_len":512,"n_iterations":5,"regime":"dflash"},
            {"name":"b4s256","batch_size":4,"seq_len":256,"n_iterations":3,"regime":"dspark"}
        ]
    })";
    const std::string path = write_tmp("wl_ok.json", wl);

    ts_throughput_workload out[TS_THROUGHPUT_MAX_WORKLOADS];
    int n = 0;
    std::string err;
    const int rc = ts_throughput_workload_load(path.c_str(), out, TS_THROUGHPUT_MAX_WORKLOADS, &n, &err);
    CHECK(rc == 0, "load ok rc");
    CHECK(n == 2, "load ok count");
    CHECK(std::string(out[0].name) == "b1s512", "wl0 name");
    CHECK(out[0].batch_size == 1, "wl0 batch");
    CHECK(out[0].seq_len == 512, "wl0 seq");
    CHECK(out[0].n_iterations == 5, "wl0 iters");
    CHECK(std::string(out[0].regime) == "dflash", "wl0 regime");
    CHECK(std::string(out[1].name) == "b4s256", "wl1 name");
    CHECK(out[1].batch_size == 4, "wl1 batch");
    CHECK(std::string(out[1].regime) == "dspark", "wl1 regime");
}

static void test_workload_defaults() {
    // name and regime default when omitted
    const std::string wl = R"({"schema_version":1,"workloads":[{"batch_size":2,"seq_len":128,"n_iterations":1}]})";
    const std::string path = write_tmp("wl_defaults.json", wl);

    ts_throughput_workload out[4];
    int n = 0;
    std::string err;
    const int rc = ts_throughput_workload_load(path.c_str(), out, 4, &n, &err);
    CHECK(rc == 0, "defaults rc");
    CHECK(n == 1, "defaults count");
    CHECK(std::string(out[0].regime) == "dflash", "default regime is dflash");
    CHECK(std::string(out[0].name).find("workload_") == 0, "default name prefix");
}

static void test_workload_bad_schema() {
    const std::string wl = R"({"schema_version":99,"workloads":[]})";
    const std::string path = write_tmp("wl_bad_schema.json", wl);
    ts_throughput_workload out[4]; int n = 0; std::string err;
    CHECK(ts_throughput_workload_load(path.c_str(), out, 4, &n, &err) != 0, "bad schema rejected");
}

static void test_workload_missing_file() {
    ts_throughput_workload out[4]; int n = 0; std::string err;
    CHECK(ts_throughput_workload_load("/tmp/ts_nonexistent_file.json", out, 4, &n, &err) != 0, "missing file rejected");
}

static void test_workload_zero_seq() {
    const std::string wl = R"({"schema_version":1,"workloads":[{"batch_size":1,"seq_len":0,"n_iterations":1}]})";
    const std::string path = write_tmp("wl_zero_seq.json", wl);
    ts_throughput_workload out[4]; int n = 0; std::string err;
    CHECK(ts_throughput_workload_load(path.c_str(), out, 4, &n, &err) != 0, "zero seq_len rejected");
}

// -------------------------------------------------------------------------
// run (stub mode)

static void test_run_stub() {
    ts_throughput_workload wl;
    std::memset(&wl, 0, sizeof(wl));
    snprintf(wl.name, sizeof(wl.name), "stub_test");
    snprintf(wl.regime, sizeof(wl.regime), "dflash");
    wl.batch_size = 1; wl.seq_len = 64; wl.n_iterations = 3;

    ts_throughput_result res;
    std::string err;
    const int rc = ts_throughput_run(&wl, 1, nullptr, nullptr, &res, &err);
    CHECK(rc == 0, "stub run rc");
    CHECK(res.stub == true, "stub flag set");
    CHECK(res.tokens_per_sec > 0.0, "stub tps > 0");
    CHECK(res.mean_latency_ms > 0.0, "stub mean > 0");
    CHECK(res.p50_latency_ms > 0.0, "stub p50 > 0");
    CHECK(res.p95_latency_ms >= res.p50_latency_ms, "p95 >= p50");
    CHECK(res.n_iterations == 3, "stub iters echoed");
    CHECK(std::string(res.name) == "stub_test", "stub name echoed");
}

// -------------------------------------------------------------------------
// run (injected callback)

static double fake_infer(int batch_size, int seq_len, void * ctx) {
    (void)ctx;
    // deterministic: 1 ms per token
    return (double)(batch_size * seq_len) * 1.0;
}

static void test_run_injected() {
    ts_throughput_workload wl;
    std::memset(&wl, 0, sizeof(wl));
    snprintf(wl.name, sizeof(wl.name), "injected");
    snprintf(wl.regime, sizeof(wl.regime), "dspark");
    wl.batch_size = 2; wl.seq_len = 100; wl.n_iterations = 4;

    ts_throughput_result res;
    std::string err;
    const int rc = ts_throughput_run(&wl, 1, fake_infer, nullptr, &res, &err);
    CHECK(rc == 0, "injected rc");
    CHECK(res.stub == false, "injected not stub");
    // 2*100 tokens per pass, 200 ms per pass -> 1000 tps
    CHECK(std::fabs(res.tokens_per_sec - 1000.0) < 1.0, "injected tps ~1000");
    CHECK(std::fabs(res.mean_latency_ms - 200.0) < 0.01, "injected mean 200ms");
}

static double failing_infer(int, int, void *) { return -1.0; }

static void test_run_infer_error() {
    ts_throughput_workload wl;
    std::memset(&wl, 0, sizeof(wl));
    snprintf(wl.name, sizeof(wl.name), "err");
    wl.batch_size = 1; wl.seq_len = 10; wl.n_iterations = 1;
    ts_throughput_result res; std::string err;
    CHECK(ts_throughput_run(&wl, 1, failing_infer, nullptr, &res, &err) != 0, "infer error propagated");
}

// -------------------------------------------------------------------------
// receipt writer

static void test_receipt_write() {
    ts_throughput_result res;
    std::memset(&res, 0, sizeof(res));
    snprintf(res.name, sizeof(res.name), "receipt_test");
    snprintf(res.regime, sizeof(res.regime), "dflash");
    res.batch_size = 1; res.seq_len = 512; res.n_iterations = 10;
    res.tokens_per_sec = 42.5; res.mean_latency_ms = 12.0;
    res.p50_latency_ms = 11.5; res.p95_latency_ms = 15.0; res.stub = true;

    const std::string out_path = "/tmp/ts_throughput_receipt_test.json";
    std::string err;
    const int rc = ts_throughput_receipt_write(out_path.c_str(), &res, 1, &err);
    CHECK(rc == 0, "receipt write rc");

    std::ifstream f(out_path);
    CHECK(f.good(), "receipt file exists");
    json j; f >> j;
    CHECK(j["schema"] == "llama.tessera.throughput.v1", "receipt schema");
    CHECK(j.contains("timestamp"), "receipt has timestamp");
    CHECK(j["results"].size() == 1, "receipt one result");
    const json & e = j["results"][0];
    CHECK(e["name"] == "receipt_test", "receipt name");
    CHECK(e["tokens_per_sec"].get<double>() == 42.5, "receipt tps");
    CHECK(e["stub"].get<bool>() == true, "receipt stub flag");
    CHECK(e["p95_latency_ms"].get<double>() == 15.0, "receipt p95");
}

// -------------------------------------------------------------------------

int main() {
    test_workload_load_ok();
    test_workload_defaults();
    test_workload_bad_schema();
    test_workload_missing_file();
    test_workload_zero_seq();
    test_run_stub();
    test_run_injected();
    test_run_infer_error();
    test_receipt_write();

    printf("throughput: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

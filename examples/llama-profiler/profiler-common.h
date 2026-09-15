#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct bench_timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void start() { t0 = clk::now(); }
    double stop()  { return std::chrono::duration<double>(clk::now() - t0).count(); }
};

struct bench_result {
    std::string op_name;
    std::string quant_type;

    int N = 0, K = 0, B = 0;
    int n_tokens = 0, ctx_len = 0, n_heads = 0, head_dim = 0;
    int64_t n_elements = 0;

    double ops   = 0.0;
    double bytes = 0.0;
    double time_s = 0.0;

    float arithmetic_intensity = 0.0f;
    float effective_gflops     = 0.0f;
    float effective_bw_gb_s    = 0.0f;

    void calculate_derived() {
        arithmetic_intensity = (bytes > 0.0) ? (float)(ops / bytes) : 0.0f;
        effective_gflops     = (time_s > 0.0) ? (float)(ops / time_s / 1e9) : 0.0f;
        effective_bw_gb_s    = (time_s > 0.0) ? (float)(bytes / time_s / 1e9) : 0.0f;
    }

    void print_dims() const {
        if (op_name.find("MUL_MAT_ID") != std::string::npos) {
            printf(" [N=%d K=%d experts=%d/%d B=%d]", N, K, n_tokens, ctx_len, B);
        } else if (op_name.find("MUL_MAT") != std::string::npos) {
            printf(" [N=%d K=%d B=%d]", N, K, B);
        } else if (op_name.find("FLASH_ATTN") != std::string::npos) {
            printf(" [tokens=%d ctx=%d heads=%d dim=%d]", n_tokens, ctx_len, n_heads, head_dim);
        } else {
            printf(" [n=%lld]", (long long)n_elements);
        }
    }
};

struct matmul_size { int32_t N; int32_t K; };
struct moe_config  { int32_t N; int32_t K; int32_t n_experts; int32_t n_experts_used; };
struct attn_config { const char * name; int32_t n_q_heads; int32_t n_kv_heads; int32_t head_dim; };

inline std::vector<matmul_size> get_matmul_sizes(bool fast) {
    if (fast) {
        return {
            {  1024,   1024}, {  2048,   2048}, {  4096,   4096}, {  8192,   8192},
            {   512,   2048}, {  8192,   4096}, { 14336,   4096}, {  4096,  14336},
            {128256,   4096},
        };
    }
    return {
        {  1024,  1024}, {  2048,  2048}, {  4096,  4096}, {  8192,  8192}, { 16384, 16384},
        {  2048,  1024}, {  4096,  2048}, {  8192,  4096}, { 14336,  4096}, { 16384,  8192},
        { 22016,  4096}, { 28672,  8192},
        {  1024,  2048}, {  2048,  4096}, {  4096,  8192}, {  4096, 14336}, {  4096, 22016},
        {  8192, 16384}, {  8192, 28672},
        {  1024,   512}, {  1024,  4096},
        {  1536,  4096}, {  2048,  7168}, {  8192,  5120},
        { 32000,  4096}, {128256,  4096}, {128256,  8192}, {151936,  1024}, {151936,  8192},
        {248320,  4096},
    };
}

inline std::vector<moe_config> get_moe_configs(bool fast) {
    if (fast) {
        return {
            { 2048,  7168, 256,  8}, { 1536,  4096, 128,  8}, {14336,  4096,   8,  2},
            { 8192,  5120,  16,  1}, { 1024,  1024, 128,  8}, { 2048,  1024, 128,  8},
            { 4096,  4096, 128,  8},
        };
    }
    return {
        { 1024,  4096, 512, 10}, { 1536,  4096, 128,  8}, { 2048,  7168, 256,  8},
        { 8192,  5120,  16,  1}, { 8192,  5120, 128,  1}, {14336,  4096,   8,  2},
        { 1024,  1024, 128,  8}, { 2048,  1024, 128,  8}, { 4096,  4096, 128,  8},
    };
}

inline std::vector<attn_config> get_attn_configs(bool fast) {
    if (fast) {
        return {
            {"MHA",   32, 32, 128}, {"GQA-8", 32,  8, 128},
            {"GQA-4", 32,  4, 128}, {"MQA",   32,  1, 128},
        };
    }
    return {
        {"MHA",    32, 32, 128}, {"GQA-16", 32, 16, 128}, {"GQA-8", 32,  8, 128},
        {"GQA-4",  64,  4, 128}, {"GQA-2",  32,  2, 256}, {"MQA",   32,  1, 128},
    };
}

inline std::vector<int32_t> get_attn_ctx_lens(bool fast) {
    if (fast) return { 1024, 4096, 8192, 16384 };
    return { 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
}

inline std::vector<ggml_type> get_matmul_quants(bool fast) {
    if (fast) {
        return { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q2_K, GGML_TYPE_MXFP4 };
    }
    return { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q2_K, GGML_TYPE_MXFP4 };
}

inline bool parse_int_arg(const char * s, int32_t & out) {
    char * end = nullptr;
    long val = strtol(s, &end, 10);
    if (end == s || *end != '\0' || val < INT32_MIN || val > INT32_MAX) {
        return false;
    }
    out = (int32_t)val;
    return true;
}

struct ridge_result {
    std::string key;
    double peak_gflops;
    double measured_bw;
    double ridge;
};

template <typename T>
inline std::vector<ridge_result> compute_ridge_points(
        const std::vector<T> & results, double measured_bw) {
    std::map<std::string, double> groups;

    for (const auto & r : results) {
        std::string key = r.op_name + "_" + r.quant_type;
        groups[key] = std::max(groups[key], (double)r.effective_gflops);
    }

    std::vector<ridge_result> out;
    for (const auto & [key, peak] : groups) {
        double ridge = (measured_bw > 0.0) ? (peak / measured_bw) : 0.0;
        out.push_back({key, peak, measured_bw, ridge});
    }
    return out;
}

inline std::vector<uint8_t> create_quantized_data(ggml_type type, int64_t n_elements) {
    size_t quant_size = ggml_row_size(type, n_elements);
    return std::vector<uint8_t>(quant_size, 0);
}

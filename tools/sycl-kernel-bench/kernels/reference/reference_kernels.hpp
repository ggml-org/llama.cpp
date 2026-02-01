#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"

#include "../../data_generators.hpp"

namespace sycl_bench {

struct ReferenceMetrics {
    double total_us = 0.0;
    double dequant_us = 0.0;
    double gemm_us = 0.0;
    double scale_us = 0.0;
    double tflops = 0.0;
    double tops = 0.0;
    double bandwidth_gbps = 0.0;
    double arithmetic_intensity = 0.0;
};

inline bool dequantize_row_fp32(ggml_type type, const void * src, float * dst, int64_t k, std::string & error) {
    switch (type) {
        case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0 *) src, dst, k); break;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1((const block_q4_1 *) src, dst, k); break;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0((const block_q5_0 *) src, dst, k); break;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1((const block_q5_1 *) src, dst, k); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) src, dst, k); break;
        case GGML_TYPE_MXFP4: dequantize_row_mxfp4((const block_mxfp4 *) src, dst, k); break;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) src, dst, k); break;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K *) src, dst, k); break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) src, dst, k); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) src, dst, k); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) src, dst, k); break;
        default:
            error = "unsupported quant type for dequantize";
            return false;
    }
    return true;
}

bool run_onednn_fp16_gemm(const GeneratedWeights & weights,
                          const GeneratedActivations & activations,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          ggml_type quant_type,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error);

bool run_onednn_int8_gemm(const GeneratedWeights & weights,
                          const GeneratedActivations & activations,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          ggml_type quant_type,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error);

bool run_onednn_woq_gemm(const GeneratedWeights & weights,
                         const GeneratedActivations & activations,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         ggml_type quant_type,
                         int warmup,
                         int iterations,
                         sycl::queue & queue,
                         ReferenceMetrics & out,
                         std::string & error);

bool run_memory_bandwidth(size_t bytes,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error);

bool run_roofline_compute(size_t elements,
                          int ops_per_element,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error);

}  // namespace sycl_bench

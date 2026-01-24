#include "reference_kernels.hpp"

#include <chrono>
#include <unordered_map>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

namespace sycl_bench {

#if !GGML_SYCL_DNNL
bool run_onednn_fp16_gemm(const GeneratedWeights &, const GeneratedActivations &, int64_t, int64_t, int64_t,
                          ggml_type, int, int, sycl::queue &, ReferenceMetrics &, std::string & error) {
    error = "oneDNN not enabled in this build (GGML_SYCL_DNNL=0)";
    return false;
}
#else
static bool dequantize_weights_fp16(const GeneratedWeights & weights,
                                    ggml_type quant_type,
                                    int64_t m,
                                    int64_t k,
                                    std::vector<ggml_half> & out,
                                    std::string & error) {
    out.resize(static_cast<size_t>(m * k));
    std::vector<float> row_fp32(static_cast<size_t>(k));
    const size_t row_bytes = ggml_row_size(quant_type, k);

    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_ptr = weights.aos.data() + row * row_bytes;
        if (!dequantize_row_fp32(quant_type, row_ptr, row_fp32.data(), k, error)) {
            return false;
        }
        ggml_half * dst = out.data() + row * k;
        for (int64_t i = 0; i < k; ++i) {
            dst[i] = ggml_fp32_to_fp16(row_fp32[static_cast<size_t>(i)]);
        }
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
                          std::string & error) {
    if (activations.fp16.empty()) {
        error = "FP16 activations missing for onednn_fp16_gemm";
        return false;
    }

    std::vector<ggml_half> weights_fp16;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!dequantize_weights_fp16(weights, quant_type, m, k, weights_fp16, error)) {
        return false;
    }

    const size_t weight_bytes = weights_fp16.size() * sizeof(ggml_half);
    const size_t act_bytes = activations.fp16.size() * sizeof(ggml_half);
    const size_t out_bytes = static_cast<size_t>(m * n) * sizeof(ggml_half);

    ggml_half * weights_dev = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, weight_bytes, queue));
    ggml_half * act_dev = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, act_bytes, queue));
    ggml_half * out_dev = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, out_bytes, queue));

    if (!weights_dev || !act_dev || !out_dev) {
        if (weights_dev) sycl::free(weights_dev, queue);
        if (act_dev) sycl::free(act_dev, queue);
        if (out_dev) sycl::free(out_dev, queue);
        error = "device allocation failed for onednn_fp16_gemm";
        return false;
    }

    queue.memcpy(weights_dev, weights_fp16.data(), weight_bytes);
    queue.memcpy(act_dev, activations.fp16.data(), act_bytes);
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    out.dequant_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    const int64_t k_padded = activations.k_padded > 0 ? activations.k_padded : k;
    const int64_t str_a0 = 1;
    const int64_t str_a1 = k;
    const int64_t str_a2 = m * k;
    const int64_t str_b0 = 1;
    const int64_t str_b1 = k_padded;
    const int64_t str_b2 = k_padded * n;

    auto eng = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(eng, queue);

    dnnl::memory::dims a_dims = {1, m, k};
    dnnl::memory::dims b_dims = {1, k, n};
    dnnl::memory::dims c_dims = {1, m, n};

    dnnl::memory::dims a_strides = {str_a2, str_a1, str_a0};
    dnnl::memory::dims b_strides = {str_b2, str_b0, str_b1};
    dnnl::memory::dims c_strides = {static_cast<dnnl_dim_t>(m) * n, 1, static_cast<dnnl_dim_t>(m)};

    auto a_md = dnnl::memory::desc(a_dims, dnnl::memory::data_type::f16, a_strides);
    auto b_md = dnnl::memory::desc(b_dims, dnnl::memory::data_type::f16, b_strides);
    auto c_md = dnnl::memory::desc(c_dims, dnnl::memory::data_type::f16, c_strides);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    auto a_mem = dnnl::memory(a_md, eng, weights_dev);
    auto b_mem = dnnl::memory(b_md, eng, act_dev);
    auto c_mem = dnnl::memory(c_md, eng, out_dev);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, a_mem});
    args.insert({DNNL_ARG_WEIGHTS, b_mem});
    args.insert({DNNL_ARG_DST, c_mem});

    for (int i = 0; i < warmup; ++i) {
        matmul_prim.execute(stream, args);
    }
    stream.wait();

    auto g0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        matmul_prim.execute(stream, args);
    }
    stream.wait();
    auto g1 = std::chrono::high_resolution_clock::now();

    const double gemm_total_us = std::chrono::duration<double, std::micro>(g1 - g0).count();
    out.gemm_us = (iterations > 0) ? gemm_total_us / iterations : 0.0;
    out.total_us = out.dequant_us + out.gemm_us;

    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double gemm_s = out.gemm_us * 1e-6;
    out.tflops = (gemm_s > 0.0) ? (ops / gemm_s) / 1.0e12 : 0.0;

    sycl::free(weights_dev, queue);
    sycl::free(act_dev, queue);
    sycl::free(out_dev, queue);
    return true;
}
#endif

}  // namespace sycl_bench

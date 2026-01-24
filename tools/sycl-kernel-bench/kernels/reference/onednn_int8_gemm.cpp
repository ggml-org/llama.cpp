#include "reference_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

namespace sycl_bench {

#if !GGML_SYCL_DNNL
bool run_onednn_int8_gemm(const GeneratedWeights &, const GeneratedActivations &, int64_t, int64_t, int64_t,
                          ggml_type, int, int, sycl::queue &, ReferenceMetrics &, std::string & error) {
    error = "oneDNN not enabled in this build (GGML_SYCL_DNNL=0)";
    return false;
}
#else
static float quantize_int8_tensor(const float * src, size_t count, std::vector<int8_t> & out) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }
    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
    out.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const float scaled = src[i] * inv_scale;
        const int q = static_cast<int>(std::round(scaled));
        out[i] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
    }
    return scale;
}

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
                          std::string & error) {
    if (activations.fp32.empty()) {
        error = "FP32 activations missing for onednn_int8_gemm";
        return false;
    }

    const size_t weights_f32_count = static_cast<size_t>(m * k);
    std::vector<float> weights_fp32(weights_f32_count);
    std::vector<float> row_fp32(static_cast<size_t>(k));
    const size_t row_bytes = ggml_row_size(quant_type, k);

    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_ptr = weights.aos.data() + row * row_bytes;
        if (!dequantize_row_fp32(quant_type, row_ptr, row_fp32.data(), k, error)) {
            return false;
        }
        std::copy(row_fp32.begin(), row_fp32.end(), weights_fp32.begin() + row * k);
    }

    std::vector<int8_t> weights_int8;
    std::vector<int8_t> act_int8;

    auto t0 = std::chrono::high_resolution_clock::now();
    const float weight_scale = quantize_int8_tensor(weights_fp32.data(), weights_fp32.size(), weights_int8);

    const int64_t k_padded = activations.k_padded > 0 ? activations.k_padded : k;
    const size_t act_count = static_cast<size_t>(n * k_padded);
    const float act_scale = quantize_int8_tensor(activations.fp32.data(), act_count, act_int8);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.dequant_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    const size_t weight_bytes = weights_int8.size() * sizeof(int8_t);
    const size_t act_bytes = act_int8.size() * sizeof(int8_t);
    const size_t out_i32_bytes = static_cast<size_t>(m * n) * sizeof(int32_t);
    const size_t out_f32_bytes = static_cast<size_t>(m * n) * sizeof(float);

    int8_t * weights_dev = static_cast<int8_t *>(sycl::aligned_alloc_device(64, weight_bytes, queue));
    int8_t * act_dev = static_cast<int8_t *>(sycl::aligned_alloc_device(64, act_bytes, queue));
    int32_t * out_i32_dev = static_cast<int32_t *>(sycl::aligned_alloc_device(64, out_i32_bytes, queue));
    float * out_f32_dev = static_cast<float *>(sycl::aligned_alloc_device(64, out_f32_bytes, queue));

    if (!weights_dev || !act_dev || !out_i32_dev || !out_f32_dev) {
        if (weights_dev) sycl::free(weights_dev, queue);
        if (act_dev) sycl::free(act_dev, queue);
        if (out_i32_dev) sycl::free(out_i32_dev, queue);
        if (out_f32_dev) sycl::free(out_f32_dev, queue);
        error = "device allocation failed for onednn_int8_gemm";
        return false;
    }

    queue.memcpy(weights_dev, weights_int8.data(), weight_bytes);
    queue.memcpy(act_dev, act_int8.data(), act_bytes);
    queue.wait_and_throw();

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

    auto a_md = dnnl::memory::desc(a_dims, dnnl::memory::data_type::s8, a_strides);
    auto b_md = dnnl::memory::desc(b_dims, dnnl::memory::data_type::s8, b_strides);
    auto c_md = dnnl::memory::desc(c_dims, dnnl::memory::data_type::s32, c_strides);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    auto a_mem = dnnl::memory(a_md, eng, weights_dev);
    auto b_mem = dnnl::memory(b_md, eng, act_dev);
    auto c_mem = dnnl::memory(c_md, eng, out_i32_dev);
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

    const float output_scale = weight_scale * act_scale;
    auto scale_kernel = [&](int32_t * src, float * dst, size_t count) {
        return queue.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const int32_t v = src[idx];
                dst[idx] = static_cast<float>(v) * output_scale;
            });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        scale_kernel(out_i32_dev, out_f32_dev, static_cast<size_t>(m * n));
    }
    queue.wait_and_throw();

    auto s0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        scale_kernel(out_i32_dev, out_f32_dev, static_cast<size_t>(m * n));
    }
    queue.wait_and_throw();
    auto s1 = std::chrono::high_resolution_clock::now();

    const double scale_total_us = std::chrono::duration<double, std::micro>(s1 - s0).count();
    out.scale_us = (iterations > 0) ? scale_total_us / iterations : 0.0;

    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double gemm_s = out.gemm_us * 1e-6;
    out.tops = (gemm_s > 0.0) ? (ops / gemm_s) / 1.0e12 : 0.0;

    out.total_us = out.dequant_us + out.gemm_us + out.scale_us;

    sycl::free(weights_dev, queue);
    sycl::free(act_dev, queue);
    sycl::free(out_i32_dev, queue);
    sycl::free(out_f32_dev, queue);
    return true;
}
#endif

}  // namespace sycl_bench

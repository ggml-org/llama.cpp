#include "reference_kernels.hpp"

#include <chrono>
#include <unordered_map>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

namespace sycl_bench {

#if !GGML_SYCL_DNNL
bool run_onednn_woq_gemm(const GeneratedWeights &, const GeneratedActivations &, int64_t, int64_t, int64_t,
                         ggml_type, int, int, sycl::queue &, ReferenceMetrics &, std::string & error) {
    error = "oneDNN not enabled in this build (GGML_SYCL_DNNL=0)";
    return false;
}
#else
static bool pack_q4_0_to_s4(const GeneratedWeights & weights,
                            int64_t m,
                            int64_t k,
                            std::vector<uint8_t> & out_s4,
                            std::vector<float> & out_scales,
                            std::vector<int8_t> & out_zp,
                            std::string & error) {
    if (k <= 0 || m <= 0) {
        error = "Invalid dims for q4_0 pack.";
        return false;
    }
    if (k % QK4_0 != 0) {
        error = "Q4_0 WoQ requires K divisible by QK4_0.";
        return false;
    }
    const int64_t groups = k / QK4_0;
    const size_t total_elems = static_cast<size_t>(k) * static_cast<size_t>(m);
    out_s4.assign((total_elems + 1) / 2, 0);
    out_scales.assign(static_cast<size_t>(groups) * static_cast<size_t>(m), 0.0f);
    out_zp.assign(static_cast<size_t>(groups) * static_cast<size_t>(m), 0);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_ptr = weights.aos.data() + row * row_bytes;
        const block_q4_0 * blocks = reinterpret_cast<const block_q4_0 *>(row_ptr);
        for (int64_t g = 0; g < groups; ++g) {
            const block_q4_0 & blk = blocks[g];
            const float scale = static_cast<float>(blk.d);
            out_scales[static_cast<size_t>(g) * static_cast<size_t>(m) + static_cast<size_t>(row)] = scale;

            const uint8_t * qs = blk.qs;
            for (int i = 0; i < QK4_0 / 2; ++i) {
                const uint8_t byte = qs[i];
                const int8_t v0 = static_cast<int8_t>((byte & 0x0F) - 8);
                const int8_t v1 = static_cast<int8_t>((byte >> 4) - 8);

                const int64_t k0 = g * QK4_0 + i * 2;
                const int64_t k1 = k0 + 1;

                const size_t idx0 = static_cast<size_t>(k0) * static_cast<size_t>(m) + static_cast<size_t>(row);
                const size_t idx1 = static_cast<size_t>(k1) * static_cast<size_t>(m) + static_cast<size_t>(row);

                const size_t byte0 = idx0 / 2;
                const size_t byte1 = idx1 / 2;
                const uint8_t nib0 = static_cast<uint8_t>(v0) & 0x0F;
                const uint8_t nib1 = static_cast<uint8_t>(v1) & 0x0F;

                if ((idx0 & 1u) == 0) {
                    out_s4[byte0] = static_cast<uint8_t>((out_s4[byte0] & 0xF0) | nib0);
                } else {
                    out_s4[byte0] = static_cast<uint8_t>((out_s4[byte0] & 0x0F) | (nib0 << 4));
                }
                if ((idx1 & 1u) == 0) {
                    out_s4[byte1] = static_cast<uint8_t>((out_s4[byte1] & 0xF0) | nib1);
                } else {
                    out_s4[byte1] = static_cast<uint8_t>((out_s4[byte1] & 0x0F) | (nib1 << 4));
                }
            }
        }
    }
    return true;
}

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
                         std::string & error) {
    if (quant_type != GGML_TYPE_Q4_0) {
        error = "onednn_woq_gemm currently supports Q4_0 weights only.";
        return false;
    }
    if (activations.fp16.empty()) {
        error = "FP16 activations missing for onednn_woq_gemm";
        return false;
    }
    if (k % QK4_0 != 0) {
        error = "onednn_woq_gemm requires K divisible by QK4_0.";
        return false;
    }

    const int64_t groups = k / QK4_0;
    std::vector<uint8_t> weights_s4;
    std::vector<float> scales;
    std::vector<int8_t> zero_points;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!pack_q4_0_to_s4(weights, m, k, weights_s4, scales, zero_points, error)) {
        return false;
    }

    const size_t src_bytes = activations.fp16.size() * sizeof(ggml_half);
    const size_t wei_user_bytes = weights_s4.size();
    const size_t dst_bytes = static_cast<size_t>(m * n) * sizeof(ggml_half);
    const size_t scales_bytes = scales.size() * sizeof(float);
    const size_t zp_bytes = zero_points.size() * sizeof(int8_t);

    uint8_t * wei_user_dev = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, wei_user_bytes, queue));
    ggml_half * src_dev = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, src_bytes, queue));
    ggml_half * dst_dev = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, dst_bytes, queue));
    float * scales_dev = static_cast<float *>(sycl::aligned_alloc_device(64, scales_bytes, queue));
    int8_t * zp_dev = static_cast<int8_t *>(sycl::aligned_alloc_device(64, zp_bytes, queue));

    if (!wei_user_dev || !src_dev || !dst_dev || !scales_dev || !zp_dev) {
        if (wei_user_dev) sycl::free(wei_user_dev, queue);
        if (src_dev) sycl::free(src_dev, queue);
        if (dst_dev) sycl::free(dst_dev, queue);
        if (scales_dev) sycl::free(scales_dev, queue);
        if (zp_dev) sycl::free(zp_dev, queue);
        error = "device allocation failed for onednn_woq_gemm";
        return false;
    }

    queue.memcpy(wei_user_dev, weights_s4.data(), wei_user_bytes);
    queue.memcpy(src_dev, activations.fp16.data(), src_bytes);
    queue.memcpy(scales_dev, scales.data(), scales_bytes);
    queue.memcpy(zp_dev, zero_points.data(), zp_bytes);
    queue.wait_and_throw();

    auto eng = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(eng, queue);

    dnnl::memory::desc src_md({n, k}, dnnl::memory::data_type::f16, {k, 1});
    dnnl::memory::desc wei_user_md({k, m}, dnnl::memory::data_type::s4, {m, 1});
    dnnl::memory::desc dst_md({n, m}, dnnl::memory::data_type::f16, {m, 1});
    dnnl::memory::desc wei_any_md({k, m}, dnnl::memory::data_type::s4, dnnl::memory::format_tag::any);

    dnnl::primitive_attr attr;
    const int mask = (1 << 0) | (1 << 1);
    dnnl_dims_t group_dims = { QK4_0, 1 };
    if (dnnl_primitive_attr_set_scales(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                       dnnl::memory::convert_to_c(dnnl::memory::data_type::f32)) != dnnl_success) {
        throw std::runtime_error("onednn_woq_gemm: set_scales failed");
    }
    if (dnnl_primitive_attr_set_zero_points(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                            dnnl::memory::convert_to_c(dnnl::memory::data_type::s8)) != dnnl_success) {
        throw std::runtime_error("onednn_woq_gemm: set_zero_points failed");
    }
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);

    auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, wei_any_md, dst_md, attr);

    dnnl::memory src_mem(src_md, eng, src_dev);
    dnnl::memory wei_user_mem(wei_user_md, eng, wei_user_dev);
    dnnl::memory dst_mem(dst_md, eng, dst_dev);
    dnnl::memory scales_mem({{m, groups}, dnnl::memory::data_type::f32, {1, m}}, eng, scales_dev);
    dnnl::memory zp_mem({{m, groups}, dnnl::memory::data_type::s8, {1, m}}, eng, zp_dev);

    dnnl::memory wei_mem = wei_user_mem;
    uint8_t * wei_packed_dev = nullptr;
    if (matmul_pd.weights_desc() != wei_user_mem.get_desc()) {
        const size_t wei_packed_bytes = matmul_pd.weights_desc().get_size();
        wei_packed_dev = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, wei_packed_bytes, queue));
        if (!wei_packed_dev) {
            sycl::free(wei_user_dev, queue);
            sycl::free(src_dev, queue);
            sycl::free(dst_dev, queue);
            sycl::free(scales_dev, queue);
            sycl::free(zp_dev, queue);
            error = "device allocation failed for packed weights";
            return false;
        }
        wei_mem = dnnl::memory(matmul_pd.weights_desc(), eng, wei_packed_dev);
        dnnl::reorder(wei_user_mem, wei_mem).execute(stream, wei_user_mem, wei_mem);
        stream.wait();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    out.dequant_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    auto matmul_prim = dnnl::matmul(matmul_pd);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, wei_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem});
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});

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
    out.tops = (gemm_s > 0.0) ? (ops / gemm_s) / 1.0e12 : 0.0;

    sycl::free(wei_user_dev, queue);
    sycl::free(src_dev, queue);
    sycl::free(dst_dev, queue);
    sycl::free(scales_dev, queue);
    sycl::free(zp_dev, queue);
    if (wei_packed_dev) {
        sycl::free(wei_packed_dev, queue);
    }

    return true;
}
#endif

}  // namespace sycl_bench

// Unit tests for oneDNN WoQ metadata helpers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-quants.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/onednn-woq.hpp"

using ggml_sycl::onednn_woq::packed_weights;

static bool test_pack_q4_0() {
    const int64_t m = 1;
    const int64_t k = QK4_0;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);

    std::vector<uint8_t> weights(row_bytes, 0);
    block_q4_0 * blk = reinterpret_cast<block_q4_0 *>(weights.data());

    blk->d = ggml_fp32_to_fp16(2.0f);
    std::fill_n(blk->qs, QK4_0 / 2, 0);
    blk->qs[0] = 0x10; // v0=-8, v1=-7 -> 0x98
    blk->qs[1] = 0x32; // v0=-6, v1=-5 -> 0xBA

    packed_weights out;
    std::string error;
    if (!ggml_sycl::onednn_woq::pack_q4_0_aos_to_s4(weights.data(), m, k, out, &error)) {
        std::fprintf(stderr, "FAILED: pack_q4_0_aos_to_s4: %s\n", error.c_str());
        return false;
    }

    if (out.group_size != QK4_0) {
        std::fprintf(stderr, "FAILED: group_size expected %d got %lld\n", QK4_0, (long long) out.group_size);
        return false;
    }
    if (out.scales_mask != 3 || out.zero_points_mask != 3) {
        std::fprintf(stderr, "FAILED: expected scales/zp mask=3 got %d/%d\n", out.scales_mask, out.zero_points_mask);
        return false;
    }
    if (out.scales.size() != 1 || std::fabs(out.scales[0] - 2.0f) > 1e-6f) {
        std::fprintf(stderr, "FAILED: scale mismatch (size=%zu value=%f)\n", out.scales.size(),
                     out.scales.empty() ? 0.0f : out.scales[0]);
        return false;
    }
    if (out.zero_points.size() != 1 || out.zero_points[0] != 0) {
        std::fprintf(stderr, "FAILED: zero point mismatch (size=%zu value=%d)\n", out.zero_points.size(),
                     out.zero_points.empty() ? 0 : out.zero_points[0]);
        return false;
    }

    const size_t expected_bytes = (static_cast<size_t>(m) * static_cast<size_t>(k) + 1) / 2;
    if (out.s4.size() != expected_bytes) {
        std::fprintf(stderr, "FAILED: packed s4 size mismatch (expected=%zu got=%zu)\n",
                     expected_bytes, out.s4.size());
        return false;
    }
    if (out.s4[0] != 0x98 || out.s4[1] != 0xBA) {
        std::fprintf(stderr, "FAILED: packed s4 bytes mismatch (got %02x %02x)\n",
                     out.s4[0], out.s4[1]);
        return false;
    }

    return true;
}

static bool test_dequant_fp16_support() {
    struct type_case {
        ggml_type type;
        bool expect;
        const char * name;
    };
    const type_case cases[] = {
        { GGML_TYPE_Q4_0, true, "Q4_0" },
        { GGML_TYPE_Q5_K, true, "Q5_K" },
        { GGML_TYPE_Q6_K, true, "Q6_K" },
        { GGML_TYPE_Q8_0, true, "Q8_0" },
        { GGML_TYPE_IQ4_NL, true, "IQ4_NL" },
        { GGML_TYPE_MXFP4, true, "MXFP4" },
        { GGML_TYPE_Q8_1, false, "Q8_1" },
        { GGML_TYPE_TQ1_0, false, "TQ1_0" },
        { GGML_TYPE_TQ2_0, false, "TQ2_0" },
    };

    for (const auto & entry : cases) {
        const bool supported = ggml_sycl::onednn_woq::supports_dequant_fp16(entry.type);
        if (supported != entry.expect) {
            std::fprintf(stderr, "FAILED: supports_dequant_fp16(%s) expected %d got %d\n",
                         entry.name, entry.expect ? 1 : 0, supported ? 1 : 0);
            return false;
        }
    }
    return true;
}

static bool test_woq_gemm_q4_0() {
#if !GGML_SYCL_DNNL
    std::fprintf(stderr, "SKIP: oneDNN not enabled (GGML_SYCL_DNNL=0)\n");
    return true;
#else
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP: SYCL backend unavailable\n");
        return true;
    }

    auto * ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: SYCL backend context missing\n");
        ggml_backend_free(backend);
        return false;
    }
    queue_ptr stream = ctx->stream();
    if (!stream) {
        std::fprintf(stderr, "FAIL: SYCL queue missing\n");
        ggml_backend_free(backend);
        return false;
    }

    const int64_t out_rows = 4;
    const int64_t batch = 3;
    const int64_t k = QK4_0;

    std::vector<float> weights_f(out_rows * k);
    for (size_t i = 0; i < weights_f.size(); ++i) {
        const int val = static_cast<int>(i % 37) - 18;
        weights_f[i] = 0.01f * static_cast<float>(val);
    }

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<uint8_t> weights_q4(row_bytes * out_rows, 0);
    quantize_q4_0(weights_f.data(), weights_q4.data(), out_rows, k, nullptr);

    packed_weights packed;
    std::string error;
    if (!ggml_sycl::onednn_woq::pack_q4_0_aos_to_s4(weights_q4.data(), out_rows, k, packed, &error)) {
        std::fprintf(stderr, "FAILED: pack_q4_0_aos_to_s4: %s\n", error.c_str());
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> weights_deq(out_rows * k);
    for (int64_t row = 0; row < out_rows; ++row) {
        const uint8_t * row_ptr = weights_q4.data() + row * row_bytes;
        const block_q4_0 * blocks = reinterpret_cast<const block_q4_0 *>(row_ptr);
        dequantize_row_q4_0(blocks, weights_deq.data() + row * k, k);
    }

    std::vector<float> act_f32(batch * k);
    std::vector<ggml_fp16_t> act_f16(batch * k);
    for (size_t i = 0; i < act_f32.size(); ++i) {
        const int val = static_cast<int>(i % 23) - 11;
        act_f32[i] = 0.02f * static_cast<float>(val);
        act_f16[i] = ggml_fp32_to_fp16(act_f32[i]);
    }
    std::vector<float> act_ref(act_f16.size());
    for (size_t i = 0; i < act_f16.size(); ++i) {
        act_ref[i] = ggml_fp16_to_fp32(act_f16[i]);
    }

    const size_t weights_bytes = packed.s4.size();
    const size_t scales_bytes = packed.scales.size() * sizeof(float);
    const size_t zp_bytes = packed.zero_points.size() * sizeof(int8_t);
    const size_t act_bytes = act_f16.size() * sizeof(ggml_fp16_t);
    const size_t dst_bytes = static_cast<size_t>(out_rows * batch) * sizeof(float);

    void * weights_dev = sycl::aligned_alloc_device(64, weights_bytes, *stream);
    float * scales_dev = static_cast<float *>(sycl::aligned_alloc_device(64, scales_bytes, *stream));
    int8_t * zp_dev = static_cast<int8_t *>(sycl::aligned_alloc_device(64, zp_bytes, *stream));
    void * act_dev = sycl::aligned_alloc_device(64, act_bytes, *stream);
    float * dst_dev = static_cast<float *>(sycl::aligned_alloc_device(64, dst_bytes, *stream));

    if (!weights_dev || !scales_dev || !zp_dev || !act_dev || !dst_dev) {
        std::fprintf(stderr, "SKIP: device allocation failed\n");
        if (weights_dev) sycl::free(weights_dev, *stream);
        if (scales_dev) sycl::free(scales_dev, *stream);
        if (zp_dev) sycl::free(zp_dev, *stream);
        if (act_dev) sycl::free(act_dev, *stream);
        if (dst_dev) sycl::free(dst_dev, *stream);
        ggml_backend_free(backend);
        return true;
    }

    stream->memcpy(weights_dev, packed.s4.data(), weights_bytes);
    stream->memcpy(scales_dev, packed.scales.data(), scales_bytes);
    stream->memcpy(zp_dev, packed.zero_points.data(), zp_bytes);
    stream->memcpy(act_dev, act_f16.data(), act_bytes);
    stream->wait_and_throw();

    bool ok = DnnlGemmWrapper::woq_gemm_q4_0(
        *ctx, batch, out_rows, k, act_dev, DnnlGemmWrapper::to_dt<sycl::half>(), weights_dev, packed.group_size,
        scales_dev, zp_dev, dst_dev, DnnlGemmWrapper::to_dt<float>(), stream, out_rows, 1);

    std::vector<float> out(out_rows * batch);
    std::vector<ggml_fp16_t> weights_f16;
    if (!ok) {
        std::fprintf(stderr, "SKIP: oneDNN WoQ gemm unsupported; validating dequant->f16 fallback\n");
        weights_f16.resize(static_cast<size_t>(k) * static_cast<size_t>(out_rows));
        for (int64_t row = 0; row < out_rows; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                const float val = weights_deq[static_cast<size_t>(row) * static_cast<size_t>(k) +
                                              static_cast<size_t>(col)];
                weights_f16[static_cast<size_t>(col) * static_cast<size_t>(out_rows) +
                            static_cast<size_t>(row)] = ggml_fp32_to_fp16(val);
            }
        }

        const size_t weights_f16_bytes = weights_f16.size() * sizeof(ggml_fp16_t);
        void * weights_f16_dev = sycl::aligned_alloc_device(64, weights_f16_bytes, *stream);
        if (!weights_f16_dev) {
            std::fprintf(stderr, "SKIP: dequant->f16 fallback allocation failed\n");
            sycl::free(weights_dev, *stream);
            sycl::free(scales_dev, *stream);
            sycl::free(zp_dev, *stream);
            sycl::free(act_dev, *stream);
            sycl::free(dst_dev, *stream);
            ggml_backend_free(backend);
            return true;
        }
        stream->memcpy(weights_f16_dev, weights_f16.data(), weights_f16_bytes).wait();

        const size_t dst_fallback_bytes =
            static_cast<size_t>(out_rows) * static_cast<size_t>(out_rows) * sizeof(float);
        float * dst_fallback_dev =
            static_cast<float *>(sycl::aligned_alloc_device(64, dst_fallback_bytes, *stream));
        if (!dst_fallback_dev) {
            std::fprintf(stderr, "SKIP: dequant->f16 fallback output allocation failed\n");
            sycl::free(weights_f16_dev, *stream);
            sycl::free(weights_dev, *stream);
            sycl::free(scales_dev, *stream);
            sycl::free(zp_dev, *stream);
            sycl::free(act_dev, *stream);
            sycl::free(dst_dev, *stream);
            ggml_backend_free(backend);
            return true;
        }

        DnnlGemmWrapper::row_gemm(
            *ctx, batch, out_rows, k, act_dev, DnnlGemmWrapper::to_dt<sycl::half>(), weights_f16_dev,
            DnnlGemmWrapper::to_dt<sycl::half>(), dst_fallback_dev, DnnlGemmWrapper::to_dt<float>(), stream,
            out_rows);

        std::vector<float> out_raw(static_cast<size_t>(out_rows) * static_cast<size_t>(out_rows));
        stream->memcpy(out_raw.data(), dst_fallback_dev, dst_fallback_bytes).wait();
        sycl::free(weights_f16_dev, *stream);
        sycl::free(dst_fallback_dev, *stream);

        std::vector<float> out_rowmajor(out.size(), 0.0f);
        for (int64_t i = 0; i < out_rows; ++i) {
            for (int64_t j = 0; j < batch; ++j) {
                out_rowmajor[static_cast<size_t>(j) * static_cast<size_t>(out_rows) + static_cast<size_t>(i)] =
                    out_raw[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(out_rows)];
            }
        }
        out.swap(out_rowmajor);
    } else {
        stream->memcpy(out.data(), dst_dev, dst_bytes).wait();
    }

    const float * weights_ref = weights_deq.data();
    std::vector<float> weights_ref_buf;
    if (!ok) {
        weights_ref_buf.resize(weights_f16.size());
        for (size_t i = 0; i < weights_f16.size(); ++i) {
            weights_ref_buf[i] = ggml_fp16_to_fp32(weights_f16[i]);
        }
        weights_ref = weights_ref_buf.data();
    }

    std::vector<float> ref(out_rows * batch, 0.0f);
    for (int64_t i = 0; i < out_rows; ++i) {
        for (int64_t j = 0; j < batch; ++j) {
            float sum = 0.0f;
            const float * wrow = weights_ref + i * k;
            const float * arow = act_ref.data() + j * k;
            for (int64_t p = 0; p < k; ++p) {
                sum += wrow[p] * arow[p];
            }
            ref[static_cast<size_t>(j) * static_cast<size_t>(out_rows) + static_cast<size_t>(i)] = sum;
        }
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < out.size(); ++i) {
        const float diff = std::fabs(out[i] - ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 0.15f) {
            std::fprintf(stderr, "FAILED: GEMM mismatch at %zu (got=%f ref=%f diff=%f)\n",
                         i, out[i], ref[i], diff);
            sycl::free(weights_dev, *stream);
            sycl::free(scales_dev, *stream);
            sycl::free(zp_dev, *stream);
            sycl::free(act_dev, *stream);
            sycl::free(dst_dev, *stream);
            ggml_backend_free(backend);
            return false;
        }
    }

    std::fprintf(stderr, "PASS: woq_gemm_q4_0 max_diff=%f\n", max_diff);
    sycl::free(weights_dev, *stream);
    sycl::free(scales_dev, *stream);
    sycl::free(zp_dev, *stream);
    sycl::free(act_dev, *stream);
    sycl::free(dst_dev, *stream);
    ggml_backend_free(backend);
    return true;
#endif
}

int main() {
    if (!test_pack_q4_0()) {
        return 1;
    }
    if (!test_dequant_fp16_support()) {
        return 1;
    }
    if (!test_woq_gemm_q4_0()) {
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}

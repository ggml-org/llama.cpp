#include "ggml.h"

#if defined(GGML_USE_CUDA)
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-nvfp4-helpers.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct block_nvfp4_mmq_test {
    uint32_t sc4_u32[4];
    uint32_t qs_u32[32];
};

static_assert(sizeof(block_nvfp4_mmq_test) == 144, "unexpected NVFP4 MMQ activation block size");

extern void quantize_mmq_nvfp4_cuda(
    const float * x, const int32_t * ids, void * vy, const ggml_tensor * src0,
    int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

#define CUDA_CHECK_TEST(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

static float decode_diag_value(const block_nvfp4_mmq_test & blk, int idx) {
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(blk.qs_u32);
    const uint8_t q = ggml_nvfp4_get_q4(qs, idx);
    const uint8_t s = reinterpret_cast<const uint8_t *>(blk.sc4_u32)[idx / 16];
    return ggml_fp8_ue4m3_to_fp32(s) * kvalues_nvfp4_float(q);
}

static int run_case(float diag_value) {
    constexpr int k = 256;
    constexpr int n = 256;
    constexpr int padded_k = 512;

    std::vector<float> input((size_t) k * (size_t) n, 0.0f);
    for (int col = 0; col < n; ++col) {
        input[(size_t) col * (size_t) k + (size_t) col] = diag_value;
    }

    float * d_x = nullptr;
    block_nvfp4_mmq_test * d_y = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * input.size()));
    CUDA_CHECK_TEST(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(block_nvfp4_mmq_test) * 2 * n));
    CUDA_CHECK_TEST(cudaMemcpy(d_x, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(d_y, 0, sizeof(block_nvfp4_mmq_test) * 2 * n));

    ggml_tensor src0_meta = {};
    src0_meta.type = GGML_TYPE_NVFP4;
    const int op_param_idx_nvfp4_input_scale = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t) - 1;
    const float input_scale = 1.0f;
    int32_t input_scale_bits = 0;
    std::memcpy(&input_scale_bits, &input_scale, sizeof(input_scale_bits));
    src0_meta.op_params[op_param_idx_nvfp4_input_scale] = input_scale_bits;

    quantize_mmq_nvfp4_cuda(
        d_x, nullptr, d_y, &src0_meta,
        k, k, (int64_t) k * n, (int64_t) k * n,
        padded_k, n, 1, 1, nullptr);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<block_nvfp4_mmq_test> got(2 * n);
    CUDA_CHECK_TEST(cudaMemcpy(got.data(), d_y, sizeof(block_nvfp4_mmq_test) * got.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK_TEST(cudaFree(d_y));
    CUDA_CHECK_TEST(cudaFree(d_x));

    double num = 0.0;
    double den = 0.0;
    float max_abs = 0.0f;
    int idx_max = 0;

    for (int col = 0; col < n; ++col) {
        const float recon = decode_diag_value(got[col], col);
        const float expect = diag_value;
        const float diff = recon - expect;
        max_abs = std::max(max_abs, std::fabs(diff));
        if (std::fabs(diff) == max_abs) {
            idx_max = col;
        }
        num += (double) recon * (double) expect;
        den += (double) expect * (double) expect;
        if (col < 16) {
            const uint8_t s = reinterpret_cast<const uint8_t *>(got[col].sc4_u32)[col / 16];
            const uint8_t * qs = reinterpret_cast<const uint8_t *>(got[col].qs_u32);
            const uint8_t q = ggml_nvfp4_get_q4(qs, col);
            std::printf(
                "col=%3d scale=%3u scale_f=% .8g q=%u recon=% .8g ratio=% .8g\n",
                col, (unsigned) s, ggml_fp8_ue4m3_to_fp32(s), (unsigned) q,
                recon, expect != 0.0f ? recon / expect : 0.0f);
        }
    }

    const double alpha = den > 0.0 ? num / den : 0.0;
    std::printf("diag=%g producer: max_abs=%.8g idx=%d alpha=%.8g\n", diag_value, max_abs, idx_max, alpha);
    return 0;
}

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }
    ggml_free(ctx);

    if (run_case(1.0f) != 0) return 1;
    if (run_case(0.75f) != 0) return 1;
    if (run_case(0.6f) != 0) return 1;
    if (run_case(0.5f) != 0) return 1;
    return 0;
}

#else
int main() {
    std::printf("skipped: GGML_USE_CUDA not enabled\n");
    return 0;
}
#endif

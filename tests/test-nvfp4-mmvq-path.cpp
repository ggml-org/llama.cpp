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

extern "C" bool ggml_cuda_nvfp4_mmvq_path_probe(
    const void * x_host,
    const void * y_host,
    float * out_host);

extern void quantize_mmvq_nvfp4_cuda(
    const float * x, const int32_t * ids, void * vy, const ggml_tensor * src0,
    int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

struct block_nvfp4_mmq_test {
    uint32_t sc4_u32[4];
    uint32_t qs_u32[32];
};

static_assert(sizeof(block_nvfp4_mmq_test) == 144, "unexpected block_nvfp4_mmq size");

#define CUDA_CHECK_TEST(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

static float decode_weight(const block_nvfp4 & pack, int row, int idx) {
    const uint8_t q = ggml_nvfp4_get_q4(pack.qs[row], idx);
    const uint8_t s = pack.scales[row][idx / 16];
    return ggml_fp8_ue4m3_to_fp32(s) * kvalues_nvfp4_float(q);
}

static float decode_activation(const block_nvfp4_mmq_test & blk, int idx) {
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(blk.qs_u32);
    const uint8_t q = ggml_nvfp4_get_q4(qs, idx);
    const uint8_t s = reinterpret_cast<const uint8_t *>(blk.sc4_u32)[idx / 16];
    return ggml_fp8_ue4m3_to_fp32(s) * kvalues_nvfp4_float(q);
}

static float reference_dot(const block_nvfp4 & pack, int row, const block_nvfp4_mmq_test & blk) {
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        sum += decode_weight(pack, row, i) * decode_activation(blk, i);
    }
    return sum;
}

static void fill_weight_pack(block_nvfp4 & pack) {
    std::memset(&pack, 0, sizeof(pack));

    for (int row = 0; row < 4; ++row) {
        for (int sb = 0; sb < 16; ++sb) {
            const float scale = 0.0625f * float(1 + ((row * 5 + sb * 3) % 8));
            pack.scales[row][sb] = ggml_fp8_ue4m3_from_fp32(scale);
        }

        std::array<uint8_t, 256> codes = {};
        for (int i = 0; i < 256; ++i) {
            const int group = i / 8;
            codes[i] = (uint8_t) ((row * 3 + group * 5 + i) & 0xF);
        }
        ggml_nvfp4_pack_codes_256(codes.data(), pack.qs[row]);
    }
}

static int run_case(float activation_scale) {
    std::array<float, 256> input = {};
    for (int i = 0; i < 256; ++i) {
        const float base = 0.35f * std::sin(0.17f * i) + 0.11f * std::cos(0.31f * i);
        const float ramp = ((i % 19) - 9) * 0.043f;
        input[i] = activation_scale * (base + ramp);
    }

    float * d_x = nullptr;
    block_nvfp4_mmq_test * d_y = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(float) * input.size()));
    CUDA_CHECK_TEST(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(block_nvfp4_mmq_test)));
    CUDA_CHECK_TEST(cudaMemcpy(d_x, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(d_y, 0, sizeof(block_nvfp4_mmq_test)));

    ggml_tensor src0_meta = {};
    src0_meta.type = GGML_TYPE_NVFP4;
    const int op_param_idx_nvfp4_input_scale = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t) - 1;
    const float runtime_input_scale = activation_scale > 0.0f ? 1.0f / activation_scale : 1.0f;
    int32_t runtime_input_scale_bits = 0;
    std::memcpy(&runtime_input_scale_bits, &runtime_input_scale, sizeof(runtime_input_scale_bits));
    src0_meta.op_params[op_param_idx_nvfp4_input_scale] = runtime_input_scale_bits;

    quantize_mmvq_nvfp4_cuda(
        d_x, nullptr, d_y, &src0_meta,
        256, 256, 256, 256,
        256, 1, 1, 1, nullptr);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    block_nvfp4_mmq_test staged = {};
    CUDA_CHECK_TEST(cudaMemcpy(&staged, d_y, sizeof(staged), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaFree(d_y));
    CUDA_CHECK_TEST(cudaFree(d_x));

    block_nvfp4 weights = {};
    fill_weight_pack(weights);

    float got[4] = {};
    if (!ggml_cuda_nvfp4_mmvq_path_probe(&weights, &staged, got)) {
        std::fprintf(stderr, "MMVQ probe unsupported or failed\n");
        return 1;
    }

    int rc = 0;
    for (int row = 0; row < 4; ++row) {
        const float ref = reference_dot(weights, row, staged);
        const float err = std::fabs(got[row] - ref);
        if (err > 1e-4f * 256.0f) {
            std::fprintf(stderr,
                "MMVQ vecdot mismatch row=%d activation_scale=%g ref=%.8f got=%.8f err=%.8f\n",
                row, activation_scale, ref, got[row], err);
            rc = 1;
        }
    }

    return rc;
}

int main() {
    int rc = 0;
    rc |= run_case(1.0f);
    rc |= run_case(0.3125f);
    rc |= run_case(3.25f);
    return rc;
}

#else
int main() {
    return 0;
}
#endif

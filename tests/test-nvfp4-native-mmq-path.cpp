#include "ggml.h"

#if defined(GGML_USE_CUDA)
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-nvfp4-helpers.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" bool ggml_cuda_nvfp4_native_mmq_path_probe(
    const void * x_host,
    const void * y_host,
    float * out_host);

struct block_nvfp4_mmq_test {
    uint32_t sc4_u32[4];
    uint32_t qs_u32[32];
};

static_assert(sizeof(block_nvfp4_mmq_test) == 144, "unexpected block_nvfp4_mmq size");

static constexpr int k_rows = 128;
static constexpr int k_cols = 8;

static float qval(uint8_t q) {
    return kvalues_nvfp4_float(q);
}

static void zero_x(std::vector<block_nvfp4> & x) {
    std::memset(x.data(), 0, x.size() * sizeof(block_nvfp4));
}

static void zero_y(std::vector<block_nvfp4_mmq_test> & y) {
    std::memset(y.data(), 0, y.size() * sizeof(block_nvfp4_mmq_test));
}

static void set_x_row(
        block_nvfp4 & row,
        const std::array<uint8_t, 16> & scales,
        const std::array<uint8_t, 256> & codes) {
    std::memset(&row, 0, sizeof(row));
    std::memcpy(row.scales[0], scales.data(), scales.size());
    ggml_nvfp4_pack_codes_256(codes.data(), row.qs[0]);
}

static void set_y_block(
        block_nvfp4_mmq_test & blk,
        const std::array<uint8_t, 16> & scales,
        const std::array<uint8_t, 256> & codes) {
    std::memset(&blk, 0, sizeof(blk));
    std::memcpy(blk.sc4_u32, scales.data(), scales.size());
    ggml_nvfp4_pack_codes_256(codes.data(), reinterpret_cast<uint8_t *>(blk.qs_u32));
}

static float decode_x(const block_nvfp4 & row, int idx) {
    const uint8_t q = ggml_nvfp4_get_q4(row.qs[0], idx);
    const uint8_t s = row.scales[0][idx / 16];
    return ggml_fp8_ue4m3_to_fp32(s) * qval(q);
}

static float decode_y(const block_nvfp4_mmq_test & blk, int idx) {
    const uint8_t * qs = reinterpret_cast<const uint8_t *>(blk.qs_u32);
    const uint8_t q = ggml_nvfp4_get_q4(qs, idx);
    const uint8_t s = reinterpret_cast<const uint8_t *>(blk.sc4_u32)[idx / 16];
    return ggml_fp8_ue4m3_to_fp32(s) * qval(q);
}

static float dot_row_col(const block_nvfp4 & row, const block_nvfp4_mmq_test & col0) {
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        sum += decode_x(row, i) * decode_y(col0, i);
    }
    return sum;
}

static std::array<uint8_t, 256> make_chunk_fingerprint_codes() {
    std::array<uint8_t, 256> codes = {};
    for (int chunk8 = 0; chunk8 < 32; ++chunk8) {
        const uint8_t q = (uint8_t) ((chunk8 % 7) + 1); // avoid zero/sign aliasing
        for (int k = 0; k < 8; ++k) {
            codes[chunk8 * 8 + k] = q;
        }
    }
    return codes;
}

static int best_match(const std::array<float, 16> & expected, float observed, float * out_abs = nullptr) {
    int best = -1;
    float best_abs = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 16; ++i) {
        const float d = std::fabs(observed - expected[i]);
        if (d < best_abs) {
            best_abs = d;
            best = i;
        }
    }
    if (out_abs) {
        *out_abs = best_abs;
    }
    return best;
}

int main() {
    constexpr int k_blackwell_cc = 1200;

    cudaDeviceProp prop = {};
    const cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
    if (prop_err != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(prop_err));
        return 1;
    }

    const int cc = 100 * prop.major + 10 * prop.minor;
    std::printf("device=%s cc=%d\n", prop.name, cc);

    if (cc < k_blackwell_cc) {
        std::printf("skipped: requires Blackwell native NVFP4 MMQ path (cc >= %d)\n", k_blackwell_cc);
        return 0;
    }

    const std::array<float, 16> distinct_scale_vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        10.0f, 12.0f, 14.0f, 16.0f,
        20.0f, 24.0f, 28.0f, 32.0f,
    };
    std::array<uint8_t, 16> distinct_scale_bytes = {};
    for (int i = 0; i < 16; ++i) {
        distinct_scale_bytes[i] = ggml_fp8_ue4m3_from_fp32(distinct_scale_vals[i]);
    }

    const uint8_t q_one = 0x2; // +1.0
    std::array<uint8_t, 16> ones = {};
    ones.fill(ggml_fp8_ue4m3_from_fp32(1.0f));

    std::vector<block_nvfp4> x(k_rows);
    std::vector<block_nvfp4_mmq_test> y(2 * k_cols);
    std::vector<float> out(k_rows * k_cols);

    bool ok = true;

    // A-side live-path mapping: distinct x subblock scales, y isolates subblocks.
    {
        zero_x(x);
        zero_y(y);

        std::array<uint8_t, 256> x_codes = {};
        x_codes.fill(q_one);
        for (int r = 0; r < k_rows; ++r) {
            set_x_row(x[r], distinct_scale_bytes, x_codes);
        }

        std::array<float, 16> expected = {};
        for (int sb = 0; sb < 16; ++sb) {
            std::array<uint8_t, 256> y_codes = {};
            for (int i = sb * 16; i < sb * 16 + 16; ++i) {
                y_codes[i] = q_one;
            }
            block_nvfp4_mmq_test ref_blk = {};
            set_y_block(ref_blk, ones, y_codes);
            expected[sb] = dot_row_col(x[0], ref_blk);
        }

        for (int base = 0; base < 16; base += 8) {
            zero_y(y);
            for (int c = 0; c < 8; ++c) {
                std::array<uint8_t, 256> y_codes = {};
                const int sb = base + c;
                for (int i = sb * 16; i < sb * 16 + 16; ++i) {
                    y_codes[i] = q_one;
                }
                set_y_block(y[c], ones, y_codes);
            }

            std::fill(out.begin(), out.end(), 0.0f);
            if (!ggml_cuda_nvfp4_native_mmq_path_probe(x.data(), y.data(), out.data())) {
                std::fprintf(stderr, "native MMQ path probe failed (A-side case)\n");
                return 1;
            }

            for (int c = 0; c < 8; ++c) {
                const float got = out[c * k_rows + 0];
                float best_abs = 0.0f;
                const int best = best_match(expected, got, &best_abs);
                std::printf("A-map col=%d got=%.8g best_sb=%d err=%.8g\n", base + c, got, best, best_abs);
                if (best != base + c || best_abs > 1e-4f) {
                    ok = false;
                }
            }
        }
    }

    // B-side live-path mapping: distinct y subblock scales, x isolates subblocks.
    {
        zero_x(x);
        zero_y(y);

        std::array<uint8_t, 256> y_codes = {};
        y_codes.fill(q_one);
        set_y_block(y[0], distinct_scale_bytes, y_codes);

        std::array<float, 16> expected = {};
        for (int sb = 0; sb < 16; ++sb) {
            std::array<uint8_t, 256> x_codes = {};
            for (int i = sb * 16; i < sb * 16 + 16; ++i) {
                x_codes[i] = q_one;
            }
            block_nvfp4 ref_row = {};
            set_x_row(ref_row, ones, x_codes);
            expected[sb] = dot_row_col(ref_row, y[0]);
        }

        for (int r = 0; r < 16; ++r) {
            std::array<uint8_t, 256> x_codes = {};
            for (int i = r * 16; i < r * 16 + 16; ++i) {
                x_codes[i] = q_one;
            }
            set_x_row(x[r], ones, x_codes);
        }

        std::fill(out.begin(), out.end(), 0.0f);
        if (!ggml_cuda_nvfp4_native_mmq_path_probe(x.data(), y.data(), out.data())) {
            std::fprintf(stderr, "native MMQ path probe failed (B-side case)\n");
            return 1;
        }

        for (int r = 0; r < 16; ++r) {
            const float got = out[0 * k_rows + r];
            float best_abs = 0.0f;
            const int best = best_match(expected, got, &best_abs);
            std::printf("B-map row=%d got=%.8g best_sb=%d err=%.8g\n", r, got, best, best_abs);
            if (best != r || best_abs > 1e-4f) {
                ok = false;
            }
        }
    }

    // Code-order live-path mapping: nonuniform 8-value chunks should survive native staging unchanged.
    {
        zero_x(x);
        zero_y(y);

        const std::array<uint8_t, 256> fingerprint_codes = make_chunk_fingerprint_codes();
        for (int r = 0; r < k_rows; ++r) {
            set_x_row(x[r], ones, fingerprint_codes);
        }

        for (int c = 0; c < k_cols; ++c) {
            set_y_block(y[c], ones, fingerprint_codes);
        }

        std::fill(out.begin(), out.end(), 0.0f);
        if (!ggml_cuda_nvfp4_native_mmq_path_probe(x.data(), y.data(), out.data())) {
            std::fprintf(stderr, "native MMQ path probe failed (code-order case)\n");
            return 1;
        }

        const float expected = dot_row_col(x[0], y[0]);
        for (int c = 0; c < k_cols; ++c) {
            const float got = out[c * k_rows + 0];
            const float err = std::fabs(got - expected);
            std::printf("code-order col=%d got=%.8g expected=%.8g err=%.8g\n", c, got, expected, err);
            if (err > 1e-4f) {
                ok = false;
            }
        }
    }

    if (!ok) {
        std::fprintf(stderr, "native MMQ path probe: mapping mismatch detected\n");
        return 1;
    }

    std::printf("native MMQ path probe: scale and code mapping matched software expectation\n");
    return 0;
}

#else
int main() {
    std::printf("skipped: GGML_USE_CUDA not enabled\n");
    return 0;
}
#endif

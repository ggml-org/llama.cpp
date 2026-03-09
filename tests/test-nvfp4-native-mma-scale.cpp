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
#include <cstdlib>
#include <cstdio>
#include <limits>

extern "C" bool ggml_cuda_nvfp4_native_mma_scale_probe(
    uint32_t a_regs,
    uint32_t b_regs,
    uint32_t a_scale,
    uint32_t b_scale,
    float * out_acc);

struct probe_stats {
    float mean = 0.0f;
    float min  = 0.0f;
    float max  = 0.0f;
    float spread = 0.0f;
    std::array<float, 4> lane0 = { 0.0f, 0.0f, 0.0f, 0.0f };
};

struct perm_fit {
    std::array<int, 4> perm = { 0, 1, 2, 3 };
    float max_abs = std::numeric_limits<float>::infinity();
    double rms = std::numeric_limits<double>::infinity();
};

static uint32_t pack_scale_bytes(const std::array<uint8_t, 4> & bytes) {
    return
        ((uint32_t) bytes[0] <<  0) |
        ((uint32_t) bytes[1] <<  8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static std::array<uint8_t, 4> encode_scales(const std::array<float, 4> & values) {
    std::array<uint8_t, 4> out = {};
    for (int i = 0; i < 4; ++i) {
        out[i] = ggml_fp8_ue4m3_from_fp32(values[i]);
    }
    return out;
}

static std::array<float, 4> decode_scales(const std::array<uint8_t, 4> & bytes) {
    std::array<float, 4> out = {};
    for (int i = 0; i < 4; ++i) {
        out[i] = ggml_fp8_ue4m3_to_fp32(bytes[i]);
    }
    return out;
}

static probe_stats run_case(
        const std::array<uint8_t, 4> & a_scales,
        const std::array<uint8_t, 4> & b_scales,
        const uint8_t code_a,
        const uint8_t code_b) {
    const uint32_t a_regs = (uint32_t) (code_a & 0x0F) * 0x11111111u;
    const uint32_t b_regs = (uint32_t) (code_b & 0x0F) * 0x11111111u;

    std::array<float, 128> acc = {};
    if (!ggml_cuda_nvfp4_native_mma_scale_probe(
                a_regs,
                b_regs,
                pack_scale_bytes(a_scales),
                pack_scale_bytes(b_scales),
                acc.data())) {
        std::fprintf(stderr, "native MMA probe helper failed\n");
        std::exit(1);
    }

    probe_stats stats;
    stats.min =  std::numeric_limits<float>::infinity();
    stats.max = -std::numeric_limits<float>::infinity();
    double sum = 0.0;

    for (int lane = 0; lane < 32; ++lane) {
        for (int l = 0; l < 4; ++l) {
            const float v = acc[(size_t) lane * 4 + l];
            if (lane == 0) {
                stats.lane0[l] = v;
            }
            stats.min = std::min(stats.min, v);
            stats.max = std::max(stats.max, v);
            sum += v;
        }
    }

    stats.mean = (float) (sum / (32.0 * 4.0));
    stats.spread = stats.max - stats.min;
    return stats;
}

static perm_fit fit_permutation(
        const std::array<float, 4> & observed,
        const std::array<float, 4> & decoded_distinct,
        const float base) {
    perm_fit best;
    std::array<int, 4> perm = { 0, 1, 2, 3 };
    do {
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float expected = base * decoded_distinct[perm[i]];
            const float diff = observed[i] - expected;
            max_abs = std::max(max_abs, std::fabs(diff));
            sum_sq += (double) diff * (double) diff;
        }

        const double rms = std::sqrt(sum_sq / 4.0);
        if (rms < best.rms || (rms == best.rms && max_abs < best.max_abs)) {
            best.perm = perm;
            best.max_abs = max_abs;
            best.rms = rms;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return best;
}

static void print_bytes(const char * label, const std::array<uint8_t, 4> & bytes, const std::array<float, 4> & decoded) {
    std::printf(
        "%s: [%u,%u,%u,%u] decoded=[%.8g,%.8g,%.8g,%.8g]\n",
        label,
        (unsigned) bytes[0], (unsigned) bytes[1], (unsigned) bytes[2], (unsigned) bytes[3],
        decoded[0], decoded[1], decoded[2], decoded[3]);
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
        std::printf("skipped: requires Blackwell native NVFP4 MMA (cc >= %d)\n", k_blackwell_cc);
        return 0;
    }

    const uint8_t qcode_a = 0x2; // +1.0
    const uint8_t qcode_b = 0x2; // +1.0
    const float qval_a = kvalues_nvfp4_float(qcode_a);
    const float qval_b = kvalues_nvfp4_float(qcode_b);
    const float one = ggml_fp8_ue4m3_to_fp32(ggml_fp8_ue4m3_from_fp32(1.0f));
    const float per_chunk = 16.0f * qval_a * qval_b * one;

    const std::array<uint8_t, 4> ones = encode_scales({ 1.0f, 1.0f, 1.0f, 1.0f });
    const std::array<uint8_t, 4> distinct = encode_scales({ 1.0f, 2.0f, 4.0f, 8.0f });
    const std::array<float, 4> distinct_dec = decode_scales(distinct);

    print_bytes("ones", ones, decode_scales(ones));
    print_bytes("distinct", distinct, distinct_dec);
    std::printf("qval_a=%.8g qval_b=%.8g per_chunk=%.8g\n", qval_a, qval_b, per_chunk);

    const probe_stats baseline = run_case(ones, ones, qcode_a, qcode_b);
    const float baseline_expected = 4.0f * per_chunk;
    std::printf(
        "baseline: mean=%.8g spread=%.8g expected=%.8g lane0=[%.8g %.8g %.8g %.8g]\n",
        baseline.mean, baseline.spread, baseline_expected,
        baseline.lane0[0], baseline.lane0[1], baseline.lane0[2], baseline.lane0[3]);

    std::array<float, 4> obs_a_hot = {};
    std::array<float, 4> obs_b_hot = {};

    for (int i = 0; i < 4; ++i) {
        std::array<uint8_t, 4> a_hot = { 0, 0, 0, 0 };
        std::array<uint8_t, 4> b_hot = { 0, 0, 0, 0 };
        a_hot[i] = ggml_fp8_ue4m3_from_fp32(1.0f);
        b_hot[i] = ggml_fp8_ue4m3_from_fp32(1.0f);

        const probe_stats s_a = run_case(a_hot, distinct, qcode_a, qcode_b);
        const probe_stats s_b = run_case(distinct, b_hot, qcode_a, qcode_b);
        obs_a_hot[i] = s_a.mean;
        obs_b_hot[i] = s_b.mean;

        std::printf(
            "a_hot[%d]: mean=%.8g spread=%.8g lane0=[%.8g %.8g %.8g %.8g]\n",
            i, s_a.mean, s_a.spread, s_a.lane0[0], s_a.lane0[1], s_a.lane0[2], s_a.lane0[3]);
        std::printf(
            "b_hot[%d]: mean=%.8g spread=%.8g lane0=[%.8g %.8g %.8g %.8g]\n",
            i, s_b.mean, s_b.spread, s_b.lane0[0], s_b.lane0[1], s_b.lane0[2], s_b.lane0[3]);
    }

    const perm_fit fit_a = fit_permutation(obs_a_hot, distinct_dec, per_chunk);
    const perm_fit fit_b = fit_permutation(obs_b_hot, distinct_dec, per_chunk);

    std::printf(
        "fit_a_hot: perm=[%d %d %d %d] max_abs=%.8g rms=%.8g\n",
        fit_a.perm[0], fit_a.perm[1], fit_a.perm[2], fit_a.perm[3],
        fit_a.max_abs, fit_a.rms);
    std::printf(
        "fit_b_hot: perm=[%d %d %d %d] max_abs=%.8g rms=%.8g\n",
        fit_b.perm[0], fit_b.perm[1], fit_b.perm[2], fit_b.perm[3],
        fit_b.max_abs, fit_b.rms);

    bool ok = true;

    if (baseline.spread > 1e-5f) {
        std::fprintf(stderr, "baseline spread too large: %.8g\n", baseline.spread);
        ok = false;
    }
    if (std::fabs(baseline.mean - baseline_expected) > 1e-4f) {
        std::fprintf(stderr, "baseline mismatch: got %.8g expected %.8g\n", baseline.mean, baseline_expected);
        ok = false;
    }
    for (int i = 0; i < 4; ++i) {
        if (fit_a.perm[i] != i) {
            std::fprintf(stderr, "a_hot identity permutation failed at %d -> %d\n", i, fit_a.perm[i]);
            ok = false;
        }
        if (fit_b.perm[i] != i) {
            std::fprintf(stderr, "b_hot identity permutation failed at %d -> %d\n", i, fit_b.perm[i]);
            ok = false;
        }
    }
    if (fit_a.max_abs > 1e-4f) {
        std::fprintf(stderr, "a_hot max_abs too large: %.8g\n", fit_a.max_abs);
        ok = false;
    }
    if (fit_b.max_abs > 1e-4f) {
        std::fprintf(stderr, "b_hot max_abs too large: %.8g\n", fit_b.max_abs);
        ok = false;
    }

    if (!ok) {
        std::fprintf(stderr, "native MMA block-scale probe: mismatch detected\n");
        return 1;
    }

    std::printf("native MMA block-scale probe: identity byte order and software UE4M3 decode matched\n");
    return 0;
}

#else
int main() {
    std::printf("skipped: GGML_USE_CUDA not enabled\n");
    return 0;
}
#endif

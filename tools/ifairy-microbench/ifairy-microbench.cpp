#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-ifairy-lut.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--m M] [--k K] [--iters N] [--warmup N] [--seed S]\n"
                 "\n"
                 "Benchmarks ggml_ifairy_lut_qgemm_merged64() with N==1 (decode-style).\n"
                 "K must be a multiple of 256.\n",
                 argv0);
}

static bool parse_int_arg(int & out, int argc, char ** argv, int & i) {
    if (i + 1 >= argc) {
        return false;
    }
    char * end = nullptr;
    long   v   = std::strtol(argv[i + 1], &end, 10);
    if (end == argv[i + 1] || *end != '\0') {
        return false;
    }
    out = (int) v;
    ++i;
    return true;
}

static int main_impl(int argc, char ** argv) {
    int      m      = 256;
    int      k      = 4096;
    int      iters  = 200;
    int      warmup = 30;
    uint32_t seed   = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--m") == 0) {
            if (!parse_int_arg(m, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--k") == 0) {
            if (!parse_int_arg(k, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--iters") == 0) {
            if (!parse_int_arg(iters, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--warmup") == 0) {
            if (!parse_int_arg(warmup, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0) {
            int seed_i = 0;
            if (!parse_int_arg(seed_i, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            seed = (uint32_t) seed_i;
            continue;
        }

        std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
        usage(argv[0]);
        return 2;
    }

    if (m <= 0 || k <= 0 || iters <= 0 || warmup < 0) {
        usage(argv[0]);
        return 2;
    }
    if (k % QK_K != 0) {
        std::fprintf(stderr, "error: --k must be a multiple of %d\n", QK_K);
        return 2;
    }

#if !(defined(__ARM_NEON) && defined(__aarch64__))
    std::fprintf(stderr, "warning: this benchmark is intended for aarch64+NEON; running anyway.\n");
#endif

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    std::mt19937                    rng(seed);
    std::uniform_int_distribution<> idx_dist(0, 63);
    std::uniform_int_distribution<> lut_dist(-128, 127);

    std::vector<block_ifairy> qweights((size_t) m * (size_t) blocks);
    for (size_t i = 0; i < qweights.size(); ++i) {
        qweights[i].d_real = ggml_fp32_to_fp16(1.0f);
        qweights[i].d_imag = ggml_fp32_to_fp16(1.0f);
    }

    std::vector<uint8_t> indexes((size_t) m * (size_t) groups);
    for (size_t i = 0; i < indexes.size(); ++i) {
        indexes[i] = (uint8_t) idx_dist(rng);
    }

    std::vector<int8_t> lut((size_t) groups * k_ifairy_lut_merged64_group_bytes);
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = (int8_t) lut_dist(rng);
    }

    std::vector<float> lut_scales((size_t) blocks * 2);
    for (size_t i = 0; i < lut_scales.size(); ++i) {
        lut_scales[i] = 1.0f;
    }

    std::vector<float> dst((size_t) m * 2);

    auto run_once = [&]() {
        ggml_ifairy_lut_qgemm_merged64(m, k, /*n*/ 1, qweights.data(), indexes.data(), lut.data(), lut_scales.data(),
                                       dst.data(),
                                       /*dst_col_stride*/ 0,
                                       /*dst_row_stride*/ 2 * sizeof(float),
                                       /*pack_bf16*/ false, /*add*/ false);
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_once();
    }
    const auto   t1  = std::chrono::steady_clock::now();
    const auto   ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double per = (double) ns / (double) iters;

    double checksum = 0.0;
    for (size_t i = 0; i < dst.size(); ++i) {
        checksum += (double) dst[i];
    }

    std::printf("ifairy-microbench merged64 N==1: m=%d k=%d blocks=%" PRId64 " groups=%" PRId64 "\n", m, k, blocks,
                groups);
    std::printf("iters=%d warmup=%d ns/iter=%.1f checksum=%.4e\n", iters, warmup, per, checksum);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "fatal: unknown exception\n");
        return 1;
    }
}

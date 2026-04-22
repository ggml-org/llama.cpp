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
                 "usage: %s [--type ifairy|ifairy64] [--mode qgemm|fused] [--m M] [--k K] [--iters N] [--warmup N] [--seed S]\n"
                 "\n"
                 "Benchmarks iFairy LUT qgemm_lut16() or fused preprocess+qgemm with N==1 (decode-style).\n"
                 "K must be a multiple of the selected weight block size.\n",
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

static bool parse_string_arg(std::string & out, int argc, char ** argv, int & i) {
    if (i + 1 >= argc) {
        return false;
    }
    out = argv[i + 1];
    ++i;
    return true;
}

enum class ifairy_bench_type {
    ifairy,
    ifairy64,
};

enum class ifairy_bench_mode {
    qgemm,
    fused,
};

using ifairy_qgemm_fn = void (*)(int, int, int, const void *, const void *, const void *, float *, size_t, size_t, bool,
                                 bool);
using ifairy_qgemm_fused_fn = void (*)(int, int, int, const void *, const void *, size_t, void *, void *, float *,
                                       size_t, size_t, bool, bool);

template <typename wtile_type>
static int run_qgemm_bench(const char *       label,
                           const char *       mode_label,
                           int                m,
                           int                k,
                           int                iters,
                           int                warmup,
                           uint32_t           seed,
                           int64_t            block_k,
                           int64_t            groups_per_block,
                           ifairy_qgemm_fn    qgemm,
                           ifairy_qgemm_fused_fn qgemm_fused) {
    const int64_t blocks = k / block_k;
    const int64_t groups = blocks * groups_per_block;
    const int64_t tiles  = ((int64_t) m + 15) / 16;

    std::mt19937                 rng(seed);
    std::uniform_int_distribution<> idx_dist(0, 15);
    std::uniform_int_distribution<> lut_dist(-128, 127);
    std::uniform_int_distribution<> act_dist(-127, 127);

    std::vector<wtile_type> packed_w((size_t) tiles * (size_t) blocks);
    for (int64_t t = 0; t < tiles; ++t) {
        for (int64_t blk = 0; blk < blocks; ++blk) {
            auto & wt = packed_w[(size_t) t * (size_t) blocks + (size_t) blk];
            for (int lane = 0; lane < 16; ++lane) {
                wt.d_real[lane] = 1.0f;
                wt.d_imag[lane] = 1.0f;
            }
            for (int gi = 0; gi < (int) groups_per_block; gi += 2) {
                for (int lane = 0; lane < 16; ++lane) {
                    const uint8_t lo = (uint8_t) (idx_dist(rng) & 0x0f);
                    const uint8_t hi = (uint8_t) (idx_dist(rng) & 0x0f);
                    wt.qs[gi / 2][lane] = lo | (uint8_t) (hi << 4);
                }
            }
        }
    }

    std::vector<int8_t> lut((size_t) groups * k_ifairy_lut_group_bytes);
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = (int8_t) lut_dist(rng);
    }

    std::vector<float> lut_scales((size_t) blocks * 2);
    for (size_t i = 0; i < lut_scales.size(); ++i) {
        lut_scales[i] = 1.0f;
    }

    std::vector<float> dst((size_t) m * 2);
    std::vector<block_ifairy_q16> act_q((size_t) (k / QK_IFAIRY));
    for (auto & blk : act_q) {
        blk.d_real = ggml_fp32_to_fp16(1.0f);
        blk.d_imag = ggml_fp32_to_fp16(1.0f);
        for (int i = 0; i < QK_IFAIRY; ++i) {
            ((int8_t *) blk.x_real)[i] = (int8_t) act_dist(rng);
            ((int8_t *) blk.x_imag)[i] = (int8_t) act_dist(rng);
        }
    }

    auto run_once_qgemm = [&]() {
        qgemm(m, k, /*n*/ 1, packed_w.data(), lut.data(), lut_scales.data(), dst.data(), /*dst_col_stride*/ 0,
              /*dst_row_stride*/ 2 * sizeof(float), /*pack_bf16*/ false, /*add*/ false);
    };

    auto run_once_fused = [&]() {
        qgemm_fused(m, k, /*n*/ 1, packed_w.data(), act_q.data(), sizeof(block_ifairy_q16), NULL, NULL, dst.data(),
                    /*dst_col_stride*/ 0,
                    /*dst_row_stride*/ 2 * sizeof(float), /*pack_bf16*/ false, /*add*/ false);
    };

    for (int i = 0; i < warmup; ++i) {
        if (qgemm_fused) {
            run_once_fused();
        } else {
            run_once_qgemm();
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (qgemm_fused) {
            run_once_fused();
        } else {
            run_once_qgemm();
        }
    }
    const auto   t1  = std::chrono::steady_clock::now();
    const auto   ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double per = (double) ns / (double) iters;

    double checksum = 0.0;
    for (size_t i = 0; i < dst.size(); ++i) {
        checksum += (double) dst[i];
    }

    std::printf(
        "ifairy-microbench %s %s lut16 N==1: m=%d k=%d blocks=%" PRId64 " groups=%" PRId64 " tiles=%" PRId64 "\n",
        label, mode_label, m, k, blocks, groups, tiles);
    std::printf("iters=%d warmup=%d ns/iter=%.1f checksum=%.4e\n", iters, warmup, per, checksum);
    return 0;
}

static int main_impl(int argc, char ** argv) {
    ifairy_bench_type type   = ifairy_bench_type::ifairy;
    ifairy_bench_mode mode   = ifairy_bench_mode::qgemm;
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
        if (std::strcmp(argv[i], "--type") == 0) {
            std::string value;
            if (!parse_string_arg(value, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            if (value == "ifairy") {
                type = ifairy_bench_type::ifairy;
            } else if (value == "ifairy64") {
                type = ifairy_bench_type::ifairy64;
            } else {
                std::fprintf(stderr, "error: --type must be ifairy|ifairy64\n");
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--mode") == 0) {
            std::string value;
            if (!parse_string_arg(value, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            if (value == "qgemm") {
                mode = ifairy_bench_mode::qgemm;
            } else if (value == "fused") {
                mode = ifairy_bench_mode::fused;
            } else {
                std::fprintf(stderr, "error: --mode must be qgemm|fused\n");
                return 2;
            }
            continue;
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
    const int block_k = type == ifairy_bench_type::ifairy ? QK_IFAIRY : QK_IFAIRY64;
    if (k % block_k != 0) {
        std::fprintf(stderr, "error: --k must be a multiple of %d for --type=%s\n", block_k,
                     type == ifairy_bench_type::ifairy ? "ifairy" : "ifairy64");
        return 2;
    }

    if (type == ifairy_bench_type::ifairy) {
        return run_qgemm_bench<ifairy_lut_wtile_16>("ifairy", mode == ifairy_bench_mode::fused ? "fused" : "qgemm",
                                                    m, k, iters, warmup, seed, QK_IFAIRY, QK_IFAIRY_GROUPS_PER_BLOCK,
                                                    mode == ifairy_bench_mode::qgemm ? ggml_ifairy_lut_qgemm_lut16 : NULL,
                                                    mode == ifairy_bench_mode::fused ? ggml_ifairy_lut_qgemm_fused_lut16
                                                                                     : NULL);
    }

    return run_qgemm_bench<ifairy64_lut_wtile_16>("ifairy64", mode == ifairy_bench_mode::fused ? "fused" : "qgemm", m,
                                                  k, iters, warmup, seed, QK_IFAIRY64, QK_IFAIRY64_GROUPS_PER_BLOCK,
                                                  mode == ifairy_bench_mode::qgemm ? ggml_ifairy64_lut_qgemm_lut16
                                                                                   : NULL,
                                                  mode == ifairy_bench_mode::fused ? ggml_ifairy64_lut_qgemm_fused_lut16
                                                                                   : NULL);
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

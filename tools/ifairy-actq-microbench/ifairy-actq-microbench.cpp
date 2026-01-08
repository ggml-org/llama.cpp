#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

extern "C" {
#include "ggml-cpu/quants.h"
}

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
                 "usage: %s [--k K] [--iters N] [--warmup N] [--seed S] [--impl block|tensor] [--no-verify]\n"
                 "\n"
                 "Benchmarks iFairy activation quantization (bf16-pair complex -> GGML_TYPE_IFAIRY_Q16 blocks).\n"
                 "K must be a multiple of %d.\n"
                 "\n"
                 "--impl=block:  quantize_row_ifairy_q16() (per-block scale, current baseline)\n"
                 "--impl=tensor: quantize_row_ifairy_q16_tensor() (tensor-scale over the full K)\n",
                 argv0, QK_K);
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

static int main_impl(int argc, char ** argv) {
    int      k           = 1536;
    int      iters       = 200000;
    int      warmup      = 2000;
    uint32_t seed        = 1;
    bool     verify      = true;
    bool     tensor_impl = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
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
        if (std::strcmp(argv[i], "--impl") == 0) {
            std::string impl;
            if (!parse_string_arg(impl, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            if (impl == "tensor") {
                tensor_impl = true;
            } else if (impl == "block") {
                tensor_impl = false;
            } else {
                std::fprintf(stderr, "error: --impl must be block|tensor\n");
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--no-verify") == 0) {
            verify = false;
            continue;
        }

        std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
        usage(argv[0]);
        return 2;
    }

    if (k <= 0 || iters <= 0 || warmup < 0) {
        usage(argv[0]);
        return 2;
    }
    if (k % QK_K != 0) {
        std::fprintf(stderr, "error: --k must be a multiple of %d\n", QK_K);
        return 2;
    }

    const int nb = k / QK_K;

    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<uint32_t> x_words((size_t) k);
    for (int j = 0; j < k; ++j) {
        const float xr = dist(rng);
        const float xi = dist(rng);

        const ggml_bf16_t xr_b = ggml_fp32_to_bf16(xr);
        const ggml_bf16_t xi_b = ggml_fp32_to_bf16(xi);

        x_words[j] = (uint32_t) xi_b.bits << 16 | (uint32_t) xr_b.bits;
    }

    std::vector<block_ifairy_q16> q((size_t) nb);

    auto run_once = [&]() {
        if (tensor_impl) {
            quantize_row_ifairy_q16_tensor(reinterpret_cast<const float *>(x_words.data()), q.data(), k);
        } else {
            quantize_row_ifairy_q16(reinterpret_cast<const float *>(x_words.data()), q.data(), k);
        }
    };

    if (verify) {
        run_once();
        bool scales_ok = true;
        if (tensor_impl) {
            for (int ib = 1; ib < nb; ++ib) {
                if (q[ib].d_real != q[0].d_real || q[ib].d_imag != q[0].d_imag) {
                    scales_ok = false;
                    break;
                }
            }
        }
        const char * scales_uniform = "n/a";
        if (tensor_impl) {
            scales_uniform = scales_ok ? "yes" : "no";
        }
        std::printf("verify: impl=%s nb=%d scales_uniform=%s\n", tensor_impl ? "tensor" : "block", nb, scales_uniform);
    }

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_once();
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto   ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double per = (double) ns / (double) iters;

    double checksum = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        checksum += (double) ggml_fp16_to_fp32(q[ib].d_real);
        checksum += 131.0 * (double) ggml_fp16_to_fp32(q[ib].d_imag);
        checksum += (double) ((int8_t *) q[ib].x_real)[0];
        checksum += 7.0 * (double) ((int8_t *) q[ib].x_imag)[0];
    }

    std::printf("ifairy-actq-microbench: k=%d nb=%d iters=%d warmup=%d seed=%" PRIu32 " impl=%s\n", k, nb, iters,
                warmup, seed, tensor_impl ? "tensor" : "block");
    std::printf("ns/iter=%.2f checksum=%.6e\n", per, checksum);
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

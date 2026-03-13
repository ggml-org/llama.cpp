#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu/quants.h"

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
    std::fprintf(
        stderr,
        "usage: %s [--k K] [--m M] [--iters N] [--warmup N] [--seed S] [--x-scale tensor|block] [--no-verify]\n"
        "\n"
        "Benchmarks ggml_vec_dot_ifairy_q16_K() (non-LUT vec_dot path).\n"
        "K must be a multiple of %d.\n"
        "M is the number of rows (decode-style matvec uses M >> 1; each row calls vec_dot once).\n"
        "\n"
        "Notes:\n"
        "- Output is a bf16-pair (real, imag) written into the provided `float *` pointer.\n"
        "- --x-scale=tensor makes all blocks share (d_real, d_imag); --x-scale=block randomizes per block.\n",
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

static float bf16_to_f32(ggml_bf16_t v) {
    return ggml_bf16_to_fp32(v);
}

static ggml_half fp32_to_fp16(float v) {
    return ggml_fp32_to_fp16(v);
}

static int main_impl(int argc, char ** argv) {
    int      k              = 1536;
    int      m              = 1;
    int      iters          = 200000;
    int      warmup         = 2000;
    uint32_t seed           = 1;
    bool     verify         = true;
    bool     x_tensor_scale = true;

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
        if (std::strcmp(argv[i], "--m") == 0) {
            if (!parse_int_arg(m, argc, argv, i)) {
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
        if (std::strcmp(argv[i], "--x-scale") == 0) {
            std::string mode;
            if (!parse_string_arg(mode, argc, argv, i)) {
                usage(argv[0]);
                return 2;
            }
            if (mode == "tensor") {
                x_tensor_scale = true;
            } else if (mode == "block") {
                x_tensor_scale = false;
            } else {
                std::fprintf(stderr, "error: --x-scale must be tensor|block\n");
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

    if (k <= 0 || m <= 0 || iters <= 0 || warmup < 0) {
        usage(argv[0]);
        return 2;
    }
    if (k % QK_K != 0) {
        std::fprintf(stderr, "error: --k must be a multiple of %d\n", QK_K);
        return 2;
    }

    const int nb = k / QK_K;

    std::mt19937                          rng(seed);
    std::uniform_int_distribution<int>    code_dist(0, 3);
    std::uniform_int_distribution<int>    act_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.50f);

    std::vector<block_ifairy> w((size_t) m * (size_t) nb);
    {
        const float w_d_real = 1.0f;
        const float w_d_imag = 1.0f;
        for (int row = 0; row < m; ++row) {
            auto * w_row = w.data() + (size_t) row * (size_t) nb;
            for (int ib = 0; ib < nb; ++ib) {
                w_row[ib].d_real = fp32_to_fp16(w_d_real);
                w_row[ib].d_imag = fp32_to_fp16(w_d_imag);

                for (int chunk = 0; chunk < 4; ++chunk) {
                    uint8_t packed[16] = { 0 };
                    for (int lane = 0; lane < 16; ++lane) {
                        uint8_t b = 0;
                        for (int part = 0; part < 4; ++part) {
                            const uint8_t code = (uint8_t) code_dist(rng);
                            b |= (uint8_t) (code << (2 * part));
                        }
                        packed[lane] = b;
                    }
                    std::memcpy(w_row[ib].qs + chunk * 16, packed, sizeof(packed));
                }
            }
        }
    }

    std::vector<block_ifairy_q16> x((size_t) nb);
    {
        const float d_real_global = 0.125f;
        const float d_imag_global = 0.125f;

        for (int ib = 0; ib < nb; ++ib) {
            float d_real = d_real_global;
            float d_imag = d_imag_global;
            if (!x_tensor_scale) {
                d_real = scale_dist(rng);
                d_imag = scale_dist(rng);
            }
            x[ib].d_real = fp32_to_fp16(d_real);
            x[ib].d_imag = fp32_to_fp16(d_imag);

            auto * xr = (int8_t *) x[ib].x_real;
            auto * xi = (int8_t *) x[ib].x_imag;
            for (int j = 0; j < QK_K; ++j) {
                xr[j] = (int8_t) act_dist(rng);
                xi[j] = (int8_t) act_dist(rng);
            }
        }
    }

    if (verify) {
        auto verify_row = [&](int row) {
            alignas(4) ggml_bf16_t out_opt_bf16[2] = {};
            alignas(4) ggml_bf16_t out_ref_bf16[2] = {};

            auto * out_ref_f32 = reinterpret_cast<float *>(out_ref_bf16);
            auto * out_opt_f32 = reinterpret_cast<float *>(out_opt_bf16);

            ggml_vec_dot_ifairy_q16_K_generic(k, out_ref_f32, 0, w.data() + (size_t) row * (size_t) nb, 0, x.data(), 0,
                                              1);
            ggml_vec_dot_ifairy_q16_K(k, out_opt_f32, 0, w.data() + (size_t) row * (size_t) nb, 0, x.data(), 0, 1);

            const float rr  = bf16_to_f32(out_ref_bf16[0]);
            const float ri  = bf16_to_f32(out_ref_bf16[1]);
            const float orr = bf16_to_f32(out_opt_bf16[0]);
            const float ori = bf16_to_f32(out_opt_bf16[1]);

            const float diff_r = std::abs(orr - rr);
            const float diff_i = std::abs(ori - ri);
            const float diff   = std::max(diff_r, diff_i);

            std::printf("verify[row=%d]: ref=(%.6f, %.6f) opt=(%.6f, %.6f) max_abs_diff=%.6g\n", row, rr, ri, orr, ori,
                        diff);
        };

        verify_row(0);
        if (m > 1) {
            verify_row(m - 1);
        }
    }

    std::vector<uint32_t> out_words((size_t) m, 0);

    for (int i = 0; i < warmup; ++i) {
        for (int row = 0; row < m; ++row) {
            auto * out_f32 = reinterpret_cast<float *>(&out_words[(size_t) row]);
            ggml_vec_dot_ifairy_q16_K(k, out_f32, 0, w.data() + (size_t) row * (size_t) nb, 0, x.data(), 0, 1);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        for (int row = 0; row < m; ++row) {
            auto * out_f32 = reinterpret_cast<float *>(&out_words[(size_t) row]);
            ggml_vec_dot_ifairy_q16_K(k, out_f32, 0, w.data() + (size_t) row * (size_t) nb, 0, x.data(), 0, 1);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto   ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double per = (double) ns / ((double) iters * (double) m);

    double checksum = 0.0;
    for (int row = 0; row < m; ++row) {
        const uint32_t    word = out_words[(size_t) row];
        const ggml_bf16_t r    = { (uint16_t) (word & 0xffffu) };
        const ggml_bf16_t i    = { (uint16_t) (word >> 16) };
        checksum += (double) bf16_to_f32(r) + 131.0 * (double) bf16_to_f32(i);
    }

    const size_t w_bytes = w.size() * sizeof(block_ifairy);
    const size_t x_bytes = x.size() * sizeof(block_ifairy_q16);

    std::printf("ifairy-vecdot-microbench: m=%d k=%d nb=%d iters=%d warmup=%d seed=%" PRIu32 " x_scale=%s\n", m, k, nb,
                iters, warmup, seed, x_tensor_scale ? "tensor" : "block");
    std::printf("ns/vecdot=%.2f w_bytes=%zu x_bytes=%zu checksum=%.6e\n", per, w_bytes, x_bytes, checksum);
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

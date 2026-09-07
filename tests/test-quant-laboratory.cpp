// test-quant-laboratory.cpp
// Reusable testing harness for quantization experiments.
//
// Provides:
//   - Synthetic data generators (Gaussian, Laplace, uniform)
//   - Real tensor data loading (f32bin format with [nrow, ncol] header)
//   - Importance matrix loading (flat f32 array)
//   - RMSE computation
//   - Multi-approach comparison framework (quantize → dequantize → matmul error)
//   - ggml graph-level verification skeleton
//
// To add a new experiment:
//   1. Add an approach function:  void approach_xxx(const float *W, float *out,
//                                                   int64_t nrow, int64_t ncol,
//                                                   const float *imatrix)
//   2. Register it in compare_approaches()
//   3. Call test_approach_comparison() from main()

#include "../ggml/src/ggml-quants.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Helper functions
// ============================================================================

static float rmse(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double) a[i] - (double) b[i];
        sum += d * d;
    }
    return (float) sqrt(sum / n);
}

static void fill_gaussian(float * data, size_t n, std::mt19937 & gen, float sigma = 1.0f) {
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

static void fill_laplace(float * data, size_t n, std::mt19937 & gen, float b = 1.0f) {
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        float v = u(gen);
        data[i] = -b * ((v > 0) - (v < 0)) * logf(1.0f - 2.0f * fabsf(v));
    }
}

static void fill_uniform(float * data, size_t n, std::mt19937 & gen, float range = 1.0f) {
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

static void fill_offset_gaussian(float * data, size_t n, std::mt19937 & gen, float sigma = 1.0f, float offset = 2.0f) {
    std::normal_distribution<float> dist(offset, sigma);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

// ============================================================================
// Data loading
// ============================================================================
static bool load_f32_tensor(const char * path, std::vector<float> & data, int64_t & nrow, int64_t & n_per_row) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }

    int64_t header[2];
    if (fread(header, sizeof(int64_t), 2, f) != 2) {
        fclose(f);
        return false;
    }
    nrow      = header[0];
    n_per_row = header[1];

    int64_t total = nrow * n_per_row;
    data.resize(total);
    size_t nread = fread(data.data(), sizeof(float), total, f);
    fclose(f);
    if ((int64_t) nread != total) {
        return false;
    }
    return true;
}

// Load imatrix file (flat f32 array, no header, one importance value per column dimension)
// The imatrix is the sum-of-squares of activations per dimension.
static bool load_imatrix(const char * path, std::vector<float> & data, int64_t expected_dims) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }

    // Get file size to determine dimensions
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int64_t dims = file_size / sizeof(float);
    if (expected_dims > 0 && dims != expected_dims) {
        printf("  WARN: imatrix dims %lld != expected %lld\n", (long long) dims, (long long) expected_dims);
        fclose(f);
        return false;
    }

    data.resize(dims);
    size_t nread = fread(data.data(), sizeof(float), dims, f);
    fclose(f);
    if ((int64_t) nread != dims) {
        return false;
    }

    // Compute stats
    float imin = data[0], imax = data[0], isum = 0;
    for (int64_t i = 0; i < dims; i++) {
        if (data[i] < imin) imin = data[i];
        if (data[i] > imax) imax = data[i];
        isum += data[i];
    }
    printf("  Loaded imatrix: %lld dims, min=%.6f, max=%.6f, mean=%.6f\n",
           (long long) dims, imin, imax, isum / dims);

    return true;
}

// ============================================================================
// Test class
// ============================================================================

class QuantLaboratory {
  public:
    QuantLaboratory() : gen(42) {}

    // ========================================================================
    // MULTI-APPROACH COMPARISON FRAMEWORK
    //
    // Each "approach" is a function that takes float weights and produces
    // dequantized float output.  The framework computes:
    //   - Weight RMSE (dequant vs original)
    //   - Matmul error (dequant weights x real activations vs f64 reference)
    //   - Ratio vs first approach (typically Q2_K baseline)
    //
    // To add a new approach:
    //   1. Write: void approach_xxx(const float *W, float *out,
    //                               int64_t nrow, int64_t ncol,
    //                               const float *imatrix) { ... }
    //   2. Add it to the `approaches` array in compare_approaches()
    // ========================================================================

    // -- Example approach: Q2_K baseline (via ggml library) --
    // Uncomment and adapt for your experiment:
    //
    // void approach_q2k(const float * W, float * out, int64_t nrow, int64_t ncol, const float * imatrix) {
    //     size_t               rs = ggml_row_size(GGML_TYPE_Q2_K, ncol);
    //     std::vector<uint8_t> buf(nrow * rs);
    //     quantize_q2_K(W, buf.data(), nrow, ncol, imatrix);
    //     auto * tr = ggml_get_type_traits(GGML_TYPE_Q2_K);
    //     for (int64_t r = 0; r < nrow; r++) {
    //         tr->to_float(buf.data() + r * rs, out + r * ncol, ncol, NULL);
    //     }
    // }

    void compare_approaches(const float * W,
                            int64_t       w_nrow,
                            int64_t       w_ncol,
                            const float * A,
                            int64_t       a_nrow,
                            int64_t       a_ncol,
                            const char *  name,
                            const float * imatrix) {
        if (w_ncol != a_ncol) {
            return;
        }
        int64_t nr = std::min(w_nrow, (int64_t) 256);
        int64_t nc = w_ncol;

        // Reference matmul (double precision)
        std::vector<double> ref(a_nrow * nr);
        for (int64_t t = 0; t < a_nrow; t++) {
            for (int64_t r = 0; r < nr; r++) {
                double s = 0;
                for (int64_t c = 0; c < nc; c++) {
                    s += (double) A[t * a_ncol + c] * (double) W[r * nc + c];
                }
                ref[t * nr + r] = s;
            }
        }
        double ref_mag2 = 0;
        for (auto v : ref) {
            ref_mag2 += v * v;
        }
        float ref_rms = (float) sqrt(ref_mag2 / (a_nrow * nr));
        (void) ref_rms;

        struct Approach {
            const char *                                                                    name;
            float                                                                           bpw;
            std::function<void(const float *, float *, int64_t, int64_t, const float *)>    fn;
        };

        // ── Register approaches here ──
        Approach approaches[] = {
            // { "Q2_K (baseline)", 2.625f,
            //   [&](auto * W, auto * o, auto nr, auto nc, auto * im) {
            //        approach_q2k(W, o, nr, nc, im);
            //    } },
            // Add more approaches...
            { "placeholder", 0.0f, nullptr },  // remove once real approaches added
        };

        printf("\n  %-28s  %5s  %10s  %10s  %7s\n", name, "BPW", "RMSE", "MatmulErr", "vs Q2K");
        printf("  %-28s  %5s  %10s  %10s  %7s\n", "---", "---", "---", "---", "---");

        float baseline_matmul_err = 0;
        for (auto & ap : approaches) {
            if (!ap.fn) {
                continue;
            }
            std::vector<float> dec(nr * nc);
            ap.fn(W, dec.data(), nr, nc, imatrix);

            // Weight RMSE
            double werr2 = 0;
            for (int64_t i = 0; i < nr * nc; i++) {
                double d = W[i] - dec[i];
                werr2 += d * d;
            }
            float wrmse = (float) sqrt(werr2 / (nr * nc));

            // Matmul error
            double merr2 = 0;
            for (int64_t t = 0; t < a_nrow; t++) {
                for (int64_t r = 0; r < nr; r++) {
                    double s = 0;
                    for (int64_t c = 0; c < nc; c++) {
                        s += (double) A[t * a_ncol + c] * (double) dec[r * nc + c];
                    }
                    double d = s - ref[t * nr + r];
                    merr2 += d * d;
                }
            }
            float matmul_rmse = (float) sqrt(merr2 / (a_nrow * nr));

            if (baseline_matmul_err == 0) {
                baseline_matmul_err = matmul_rmse;
            }
            float ratio = (baseline_matmul_err > 1e-10f) ? matmul_rmse / baseline_matmul_err : 0;

            printf("  %-28s  %5.3f  %10.6f  %10.6f  %6.3fx\n", ap.name, ap.bpw, wrmse, matmul_rmse, ratio);
        }
    }

    // Run comparison on all tensor pairs from data directory
    int test_approach_comparison(const char * data_dir) {
        printf("\n");
        printf("=======================================================================\n");
        printf("  MULTI-APPROACH COMPARISON (real weights x real activations)\n");
        printf("=======================================================================\n");

        struct TestPair {
            const char * wf;
            const char * af;
            const char * imf;
            const char * name;
        } pairs[] = {
            { "blk_0_ffn_gate_weight.f32bin", "act_blk0_ffn_input.f32bin",      "imatrix_blk0_ffn_gate_up.f32bin", "ffn_gate" },
            { "blk_0_ffn_up_weight.f32bin",   "act_blk0_ffn_input.f32bin",      "imatrix_blk0_ffn_gate_up.f32bin", "ffn_up"   },
            { "blk_0_ffn_down_weight.f32bin", "act_blk0_ffn_down_input.f32bin", "imatrix_blk0_ffn_down.f32bin",    "ffn_down" },
            { "blk_0_attn_q_weight.f32bin",   "act_blk0_attn_input.f32bin",     "imatrix_blk0_attn_qkv.f32bin",    "attn_q"   },
        };

        for (auto & p : pairs) {
            char wp[512], ap[512], imp[512];
            snprintf(wp, sizeof(wp), "%s/%s", data_dir, p.wf);
            snprintf(ap, sizeof(ap), "%s/%s", data_dir, p.af);
            snprintf(imp, sizeof(imp), "%s/%s", data_dir, p.imf);
            std::vector<float> wd, ad, im;
            int64_t            wnr, wnc, anr, anc;
            if (!load_f32_tensor(wp, wd, wnr, wnc) || !load_f32_tensor(ap, ad, anr, anc)) {
                continue;
            }
            const float * im_ptr = nullptr;
            if (load_imatrix(imp, im, wnc)) {
                im_ptr = im.data();
            } else {
                printf("  [%s] No imatrix found, using uniform weights\n", p.name);
            }
            compare_approaches(wd.data(), wnr, wnc, ad.data(), anr, anc, p.name, im_ptr);
        }
        printf("\n");
        return 0;
    }

  private:
    std::mt19937 gen;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    QuantLaboratory lab;
    int             total_fail = 0;

    printf("Quantization Laboratory\n");
    printf("=======================\n");

    // Real data tests (from data/ directory)
    {
        const char * data_dir = "data";
        if (argc > 1) {
            data_dir = argv[1];
        }

        char probe[512];
        snprintf(probe, sizeof(probe), "%s/blk_0_ffn_gate_weight.f32bin", data_dir);
        FILE * fp = fopen(probe, "rb");
        if (fp) {
            fclose(fp);
            total_fail += lab.test_approach_comparison(data_dir);
        } else {
            printf("\n=== Real Data Tests SKIPPED ===\n");
            printf("  No data found at %s\n", data_dir);
            printf(
                "  Run: cd data && PYTHONPATH=../gguf-py python3 ../scripts/extract-tensor-data.py MODEL.gguf "
                "blk.0.ffn_gate blk.0.ffn_up blk.0.ffn_down blk.0.attn_q\n");
            printf("  And: llama-capture-layer-data -m MODEL.gguf -l 0 -o data\n");
        }
    }

    printf("\n\n=== Testing Complete: %d failures ===\n", total_fail);

    return total_fail > 0 ? 1 : 0;
}

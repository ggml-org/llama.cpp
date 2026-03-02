// test-quant-q4dpt-experiment.cpp
// Sweep quantization knobs to find the best combination for Gaussian data.

#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
void    q4dpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row,
                            const float * imatrix, int8_t levels_out[16]);
void    q4dpt_set_levels(const int8_t * levels);
void    dequantize_row_q4_dpt(const void * x, float * y, int64_t k);
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

#define Q4DPT_N_LEVELS 16
#define QK4_NL 32

// --- block_q4_dpt layout (same as block_iq4_nl) ---
#pragma pack(push, 1)
struct block_q4_dpt {
    uint16_t d;          // f16 delta
    uint8_t  qs[QK4_NL/2]; // nibbles
};
#pragma pack(pop)

// --- Configurable quantization function ---
struct QuantConfig {
    bool use_element_weights;   // weight[j] = xb[j]^2 vs uniform
    bool allow_neg_scale;       // d can be negative (IQ4_NL-style init)
    bool iq4nl_perturbation;    // id = (itry+values[0])/max vs d = d_base*(1+itry*frac)
    bool final_reassign;        // re-assign after finding best scale
    float perturb_range;        // perturbation range for generic perturbation (e.g., 0.1)
    int  ntry;                  // number of perturbation steps
    const char * name;
};

static int best_index(int n, const int8_t * values, float x) {
    int best = 0;
    float bd = fabsf(x - (float)values[0]);
    for (int k = 1; k < n; ++k) {
        float d = fabsf(x - (float)values[k]);
        if (d < bd) { bd = d; best = k; }
    }
    return best;
}

static void quantize_block_experiment(const float * xb, block_q4_dpt * out,
                                       const int8_t * values, const QuantConfig & cfg) {
    float amax = 0.0f, max_val = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) { amax = ax; max_val = xb[j]; }
    }
    if (amax < 1e-10f) {
        out->d = 0;
        memset(out->qs, 0, QK4_NL/2);
        return;
    }

    // Weights
    float weight[QK4_NL];
    if (cfg.use_element_weights) {
        for (int j = 0; j < QK4_NL; ++j) { weight[j] = xb[j] * xb[j]; }
    } else {
        for (int j = 0; j < QK4_NL; ++j) { weight[j] = 1.0f; }
    }

    // Find max abs level
    float max_abs_level = 0.0f;
    for (int k = 0; k < Q4DPT_N_LEVELS; ++k) {
        float al = fabsf((float)values[k]);
        if (al > max_abs_level) max_abs_level = al;
    }
    if (max_abs_level < 1e-10f) max_abs_level = 1.0f;

    // Initial scale
    float d;
    if (cfg.allow_neg_scale) {
        d = cfg.ntry > 0 ? -max_val / (float)values[0] : max_val / (float)values[0];
    } else {
        d = amax / max_abs_level;
    }
    float id = (fabsf(d) > 1e-20f) ? 1.0f / d : 0.0f;

    // Initial assignment
    uint8_t L[QK4_NL];
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        float al = id * xb[j];
        L[j] = (uint8_t)best_index(Q4DPT_N_LEVELS, values, al);
        float q = (float)values[L[j]];
        float w = weight[j];
        sumqx += w * q * xb[j];
        sumq2 += w * q * q;
    }
    d = (sumq2 > 1e-20f) ? sumqx / sumq2 : d;
    float best_metric = d * sumqx;  // sumqx^2/sumq2 proxy
    uint8_t best_L[QK4_NL];
    memcpy(best_L, L, QK4_NL);
    float best_d = d;

    int ntry = cfg.ntry;

    // Scale perturbation
    if (cfg.iq4nl_perturbation) {
        // IQ4_NL-style: id = (itry + values[0]) / max_val
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = ((float)itry + (float)values[0]) / max_val;
            sumqx = sumq2 = 0.0f;
            for (int j = 0; j < QK4_NL; ++j) {
                float al = id * xb[j];
                L[j] = (uint8_t)best_index(Q4DPT_N_LEVELS, values, al);
                float q = (float)values[L[j]];
                float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best_metric * sumq2) {
                d = sumqx / sumq2;
                best_metric = d * sumqx;
                best_d = d;
                memcpy(best_L, L, QK4_NL);
            }
        }
    } else {
        // Generic: d = d_base * (1 + itry * range / ntry)
        float d_base = best_d;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            d = d_base * (1.0f + (float)itry * (cfg.perturb_range / ntry));
            if (fabsf(d) < 1e-20f) continue;
            id = 1.0f / d;
            sumqx = sumq2 = 0.0f;
            for (int j = 0; j < QK4_NL; ++j) {
                float al = id * xb[j];
                L[j] = (uint8_t)best_index(Q4DPT_N_LEVELS, values, al);
                float q = (float)values[L[j]];
                float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best_metric * sumq2) {
                d = sumqx / sumq2;
                best_metric = d * sumqx;
                best_d = d;
                memcpy(best_L, L, QK4_NL);
            }
        }
    }

    // Final re-assignment
    if (cfg.final_reassign) {
        id = (fabsf(best_d) > 1e-20f) ? 1.0f / best_d : 0.0f;
        for (int j = 0; j < QK4_NL; ++j) {
            float al = id * xb[j];
            best_L[j] = (uint8_t)best_index(Q4DPT_N_LEVELS, values, al);
        }
    }

    // Store FP16 scale via memcpy of the raw half-float bits
    uint16_t d_fp16;
    {
        float f = best_d;
        // Use ggml's conversion
        ggml_fp16_t h = ggml_fp32_to_fp16(f);
        memcpy(&d_fp16, &h, sizeof(d_fp16));
    }
    out->d = d_fp16;
    for (int j = 0; j < QK4_NL/2; ++j) {
        out->qs[j] = best_L[j] | (best_L[j + QK4_NL/2] << 4);
    }
}

// ---------------------------------------------------------------------------
static float rmse_vec(const float * a, const float * b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) { double d = (double)a[i] - (double)b[i]; s += d*d; }
    return (float)std::sqrt(s / (double)n);
}

static float experiment_rmse(const float * data, size_t nrow, size_t n_per_row,
                              const int8_t * levels, const QuantConfig & cfg) {
    q4dpt_set_levels(levels);
    size_t nblock = n_per_row / QK4_NL;
    size_t total = nrow * n_per_row;
    std::vector<block_q4_dpt> qblocks(nrow * nblock);
    std::vector<float> deq(total);

    for (size_t row = 0; row < nrow; ++row) {
        for (size_t ib = 0; ib < nblock; ++ib) {
            quantize_block_experiment(
                data + row * n_per_row + ib * QK4_NL,
                &qblocks[row * nblock + ib], levels, cfg);
        }
    }

    // Dequantize
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_DPT);
    for (size_t row = 0; row < nrow; ++row) {
        tr->to_float((const void *)&qblocks[row * nblock],
                     deq.data() + row * n_per_row, (int64_t)n_per_row);
    }

    return rmse_vec(data, deq.data(), total);
}

static float iq4nl_rmse(const float * data, size_t nrow, size_t n_per_row) {
    size_t rs = ggml_row_size(GGML_TYPE_IQ4_NL, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float> dq(nrow * n_per_row);
    ggml_quantize_chunk(GGML_TYPE_IQ4_NL, data, qb.data(), 0, nrow, n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t)n_per_row);
    }
    return rmse_vec(data, dq.data(), nrow * n_per_row);
}

int main() {
    ggml_backend_load_all();

    // Generate Gaussian data
    std::mt19937 rng(0xdeadbeef);
    const size_t nrow = 64, ncol = 4096;
    std::vector<float> data(nrow * ncol);
    std::normal_distribution<float> nd(0, 0.02f);
    for (auto & v : data) { v = nd(rng); }

    float ref = iq4nl_rmse(data.data(), nrow, ncol);
    printf("IQ4_NL reference RMSE: %.6f\n\n", ref);

    // Train levels
    int8_t trained_levels[Q4DPT_N_LEVELS];
    q4dpt_train_levels(data.data(), (int64_t)nrow, (int64_t)ncol, nullptr, trained_levels);

    printf("Trained levels: ");
    for (int k = 0; k < Q4DPT_N_LEVELS; ++k) printf("%4d", trained_levels[k]);
    printf("\nIQ4_NL  levels: ");
    for (int k = 0; k < Q4DPT_N_LEVELS; ++k) printf("%4d", kvalues_iq4nl[k]);
    printf("\n\n");

    // Define configurations to test
    QuantConfig configs[] = {
        // name, elem_w, neg_scale, iq4nl_perturb, final_reassign, perturb_range, ntry
        { false, false, false, false, 0.10f, 7, "A: baseline (uniform w, pos scale, generic 10%)" },
        { true,  false, false, false, 0.10f, 7, "B: elem_w only" },
        { false, true,  true,  false, 0.0f,  7, "C: neg_scale + iq4nl_perturb" },
        { true,  true,  true,  false, 0.0f,  7, "D: elem_w + neg_scale + iq4nl_perturb" },
        { true,  true,  true,  true,  0.0f,  7, "E: D + final_reassign" },
        { false, false, false, false, 0.05f, 7, "F: uniform w, pos scale, generic 5%" },
        { false, false, false, false, 0.20f, 7, "G: uniform w, pos scale, generic 20%" },
        { false, false, false, false, 0.10f, 15,"H: baseline ntry=15" },
        { true,  false, false, false, 0.10f, 15,"I: elem_w ntry=15" },
        { false, true,  true,  true,  0.0f,  7, "J: neg_scale + iq4nl_perturb + reassign (no elem_w)" },
        { false, true,  true,  true,  0.0f,  15,"K: J with ntry=15" },
        { true,  true,  true,  true,  0.0f,  15,"L: all features ntry=15" },
        { false, false, false, true,  0.10f, 7, "M: baseline + reassign" },
        { true,  false, false, true,  0.10f, 7, "N: elem_w + reassign" },
        { false, true,  false, false, 0.10f, 7, "O: neg_scale + generic 10%" },
        { false, true,  false, true,  0.10f, 7, "P: neg_scale + generic 10% + reassign" },
        { true,  true,  false, true,  0.10f, 7, "Q: elem_w + neg_scale + generic 10% + reassign" },
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    printf("%-55s  %8s  %8s  %8s  %8s\n", "Configuration", "Trained", "IQ4_NL", "Tr.Ratio", "NL.Ratio");
    printf("%-55s  %8s  %8s  %8s  %8s\n",
           "-------------------------------------------------------", "--------", "--------", "--------", "--------");

    for (int c = 0; c < nconfigs; ++c) {
        float rmse_trained = experiment_rmse(data.data(), nrow, ncol, trained_levels, configs[c]);
        float rmse_iq4nl   = experiment_rmse(data.data(), nrow, ncol, kvalues_iq4nl, configs[c]);
        printf("%-55s  %8.6f  %8.6f  %8.4f  %8.4f\n",
               configs[c].name, rmse_trained, rmse_iq4nl,
               rmse_trained / ref, rmse_iq4nl / ref);
    }

    printf("\n(Ratio < 1.0 = better than IQ4_NL's native quantization)\n");
    return 0;
}

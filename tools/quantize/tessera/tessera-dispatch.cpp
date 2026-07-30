//
// tessera-dispatch.cpp
//
// Top-level Tessera pipeline orchestrator skeleton. Implements the
// control flow from docs/c++-port-design.md section 2.5. Real GGUF
// loading/writing is wired in a later phase; for now this walks a
// hardcoded tensor list with synthetic weights to prove the structure.
//

#include "tessera-dispatch.h"
#include "tessera-quant.h"
#include "tessera-awq.h"

#include <cstdio>
#include <cstring>
#include <random>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> ts_to_bytes_u32(const std::vector<uint32_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(uint32_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_u16(const std::vector<uint16_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(uint16_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_i8(const std::vector<int8_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(int8_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_i32(const std::vector<int32_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(int32_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

// deterministic PRNG for synthetic weights
static uint64_t ts_splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void ts_fill_random_f32(float * dst, int64_t n, uint64_t seed) {
    uint64_t state = seed;
    for (int64_t i = 0; i < n; i++) {
        // uniform in [-1, 1]
        uint64_t r = ts_splitmix64(&state);
        dst[i] = ((float)(r >> 40) / (float)(1u << 24)) * 2.0f - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

int ts_dispatch_run(const ts_dispatch_params * params,
                    ts_dispatch_result * result,
                    std::string * err_msg) {
    if (params == nullptr || result == nullptr) {
        if (err_msg) {
            *err_msg = "null params or result";
        }
        return 1;
    }

    *result = {};

    const bool verbose = params->verbose;

    // --- step 1: determine which steps to run ---
    const bool need_calibration = params->imatrix_path.empty() && params->policy_path.empty();
    const bool need_ga          = params->policy_path.empty();

    // --- step 2: calibration (placeholder) ---
    if (need_calibration) {
        if (verbose) {
            printf("tessera-dispatch: would run calibration (%s)\n",
                   params->calib_corpus.empty() ? "built-in mini-corpus" : params->calib_corpus.c_str());
        }
    }

    if (params->calibrate_only) {
        if (verbose) {
            printf("tessera-dispatch: calibrate_only, returning early\n");
        }
        result->policy_json = "{}";
        return 0;
    }

    // --- step 3: GA (placeholder) ---
    float default_alpha = 0.5f;

    if (need_ga) {
        if (verbose) {
            printf("tessera-dispatch: would run GA (seed=%llu iters=%d islands=%d pop=%d)\n",
                   (unsigned long long)params->evolve_seed,
                   params->evolve_iters,
                   params->evolve_islands,
                   params->evolve_population);
        }
    }

    if (params->evolve_only) {
        if (verbose) {
            printf("tessera-dispatch: evolve_only, returning early\n");
        }
        result->policy_json = "{}";
        return 0;
    }

    // --- step 4: resolve alpha ---
    if (params->awq_alpha != "auto" && !params->awq_alpha.empty()) {
        default_alpha = std::stof(params->awq_alpha);
    }

    // --- step 5: walk tensors and quantize ---
    // hardcoded list for skeleton testing; real path walks the GGUF
    static const char * test_tensors[] = {
        "blk.0.attn_q",
        "blk.0.attn_k",
        "blk.0.ffn_gate",
    };
    static const char * test_families[] = {
        "attn",
        "attn",
        "ffn",
    };

    const int64_t out_dim = 4;
    const int64_t in_dim  = 640;  // one full page

    ts_quant_params_2d qparams;
    qparams.alpha          = default_alpha;
    qparams.clip           = params->awq_clip;
    qparams.max_outliers   = 0;
    qparams.outlier_thresh = params->outlier_frac;
    qparams.use_imatrix    = false;
    qparams.use_septq      = false;
    qparams.awq_grid       = 20;
    qparams.seed           = (uint32_t)params->evolve_seed;

    std::vector<float> weights((size_t)(out_dim * in_dim));
    float total_mse = 0.0f;

    for (int t = 0; t < 3; t++) {
        ts_fill_random_f32(weights.data(), out_dim * in_dim,
                           params->evolve_seed + (uint64_t)t * 7919ull);

        ts_quant_result_2d qr;
        int rc = ts_quantize_2d(weights.data(),
                                nullptr,   // act_scales
                                nullptr,   // calib_X
                                nullptr,   // ref_output
                                nullptr,   // imatrix
                                out_dim, in_dim, 0,
                                &qparams, &qr);
        if (rc != 0) {
            if (err_msg) {
                *err_msg = "ts_quantize_2d failed for " + std::string(test_tensors[t]);
            }
            return 2;
        }

        ts_dispatch_tensor_result tr;
        tr.name                = test_tensors[t];
        tr.family              = test_families[t];
        tr.out_dim             = out_dim;
        tr.in_dim              = in_dim;
        tr.packed              = ts_to_bytes_u32(qr.packed);
        tr.page_scales         = ts_to_bytes_u16(qr.page_scales);
        tr.lane_scales         = ts_to_bytes_i8(qr.lane_scales);
        tr.outlier_row_offsets = ts_to_bytes_i32(qr.outlier_row_offsets);
        tr.outlier_cols        = ts_to_bytes_i32(qr.outlier_cols);
        tr.outlier_vals        = ts_to_bytes_u16(qr.outlier_vals);
        tr.act_scale           = ts_to_bytes_u16(qr.act_scale);
        tr.mse                 = qr.mse;
        tr.alpha_used          = qr.best_alpha;

        total_mse += qr.mse;
        result->tensors.push_back(std::move(tr));

        if (verbose) {
            printf("tessera-dispatch: quantized %s (mse=%.6f alpha=%.3f)\n",
                   test_tensors[t], qr.mse, qr.best_alpha);
        }
    }

    // --- step 6: populate summary ---
    result->n_tensors_quantized = (int64_t)result->tensors.size();
    result->n_tensors_skipped   = 0;
    result->total_mse           = total_mse;
    result->policy_json         = "{}";
    result->policy_sha256       = "";

    return 0;
}

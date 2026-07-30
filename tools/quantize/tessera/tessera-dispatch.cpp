//
// tessera-dispatch.cpp
//
// Top-level Tessera pipeline orchestrator. Loads the input GGUF, walks
// every tensor, routes quantizable weights through the regime classifier,
// quantizes via ts_quantize_2d / ts_quantize_3d, and writes the output
// GGUF with tessera metadata and policy JSON.
//

#include "tessera-dispatch.h"
#include "tessera-quant.h"
#include "tessera-awq.h"
#include "tessera-regime.h"
#include "tessera-gguf-writer.h"

#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

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

// A tensor is quantizable if it is a 2D or 3D F32/F16 weight matrix
// whose name maps to a known tensor family (attn, ffn, etc).
static bool ts_is_quantizable(const char * name, enum ggml_type type, int n_dims) {
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        return false;
    }
    if (n_dims != 2 && n_dims != 3) {
        return false;
    }
    std::string family = ts_regime_infer_family(name);
    if (family.empty() || family == "other") {
        return false;
    }
    return true;
}

// Convert a ggml tensor's data to a flat F32 buffer.
// Handles F32 (copy) and F16 (convert). Returns empty on unsupported type.
static std::vector<float> ts_tensor_to_f32(const struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, out.data(), n);
    } else {
        out.clear();
    }
    return out;
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

    // --- step 5: load input GGUF ---
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gparams = {
        /*no_alloc =*/ false,
        /*ctx      =*/ &ggml_ctx,
    };

    struct gguf_context * in_ctx = gguf_init_from_file(params->input_path.c_str(), gparams);
    if (in_ctx == nullptr) {
        if (err_msg) {
            *err_msg = "failed to open input GGUF: " + params->input_path;
        }
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    if (verbose) {
        printf("tessera-dispatch: loaded '%s' (%lld tensors)\n",
               params->input_path.c_str(), (long long)n_tensors);
    }

    // --- step 6: prepare output GGUF ---
    struct gguf_context * out_ctx = gguf_init_empty();
    gguf_set_kv(out_ctx, in_ctx);

    // --- step 7: walk tensors, quantize or copy through ---
    ts_quant_params_2d qparams;
    qparams.alpha          = default_alpha;
    qparams.clip           = params->awq_clip;
    qparams.max_outliers   = 0;
    qparams.outlier_thresh = params->outlier_frac;
    qparams.use_imatrix    = false;
    qparams.use_septq      = false;
    qparams.awq_grid       = 20;
    qparams.seed           = (uint32_t)params->evolve_seed;

    float total_mse = 0.0f;
    int64_t n_quantized = 0;
    int64_t n_skipped   = 0;

    // policy JSON accumulator
    std::string policy_json = "{\n  \"tensors\": [\n";
    bool first_policy_entry = true;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(in_ctx, i);
        const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);

        // count dimensions (ne[d] == 1 for d >= n_dims)
        int n_dims = GGML_MAX_DIMS;
        while (n_dims > 1 && ne[n_dims - 1] == 1) {
            n_dims--;
        }

        struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
        if (t == nullptr) {
            fprintf(stderr, "tessera-dispatch: warning: tensor '%s' not found in ggml context, skipping\n", name);
            n_skipped++;
            continue;
        }

        if (!ts_is_quantizable(name, type, n_dims)) {
            // copy through unchanged
            gguf_add_tensor(out_ctx, t);
            n_skipped++;
            if (verbose) {
                printf("tessera-dispatch: copy-through %s (%s)\n", name, ggml_type_name(type));
            }
            continue;
        }

        // quantizable weight matrix
        const std::string family = ts_regime_infer_family(name);
        const int64_t in_dim  = ne[0];
        const int64_t out_dim = ne[1];

        // convert to F32
        std::vector<float> weights = ts_tensor_to_f32(t);
        if (weights.empty()) {
            fprintf(stderr, "tessera-dispatch: warning: unsupported type for '%s', copying through\n", name);
            gguf_add_tensor(out_ctx, t);
            n_skipped++;
            continue;
        }

        // compute regime descriptor and route
        ts_regime_descriptor desc = ts_regime_compute_descriptor(
            name, weights.data(), out_dim, in_dim,
            nullptr, 0);  // imatrix lookup deferred

        ts_regime_routing routing = ts_regime_classify(&desc);

        if (verbose) {
            printf("tessera-dispatch: %s family=%s expert=%d reason='%s'\n",
                   name, family.c_str(), (int)routing.expert, routing.reason.c_str());
        }

        if (n_dims == 3) {
            // MoE expert tensor: (n_experts x out_dim x in_dim)
            const int64_t n_experts = ne[2];

            std::vector<ts_quant_result_2d> qresults;
            int rc = ts_quantize_3d(weights.data(),
                                    nullptr, nullptr, nullptr, nullptr,
                                    n_experts, out_dim, in_dim, 0,
                                    &qparams, &qresults);
            if (rc != 0) {
                if (err_msg) {
                    *err_msg = "ts_quantize_3d failed for " + std::string(name);
                }
                gguf_free(out_ctx);
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 2;
            }

            // write per-expert clusters
            for (int64_t e = 0; e < n_experts; e++) {
                char exp_name[GGML_MAX_NAME];
                snprintf(exp_name, sizeof(exp_name), "%s.%lld", name, (long long)e);
                ts_gguf_write_tensor_cluster(out_ctx, exp_name, &qresults[(size_t)e], out_dim, in_dim);
                total_mse += qresults[(size_t)e].mse;
            }

            ts_dispatch_tensor_result tr;
            tr.name    = name;
            tr.family  = family;
            tr.out_dim = out_dim;
            tr.in_dim  = in_dim;
            // aggregate first expert's blobs for the result struct
            if (!qresults.empty()) {
                tr.packed              = ts_to_bytes_u32(qresults[0].packed);
                tr.page_scales         = ts_to_bytes_u16(qresults[0].page_scales);
                tr.lane_scales         = ts_to_bytes_i8(qresults[0].lane_scales);
                tr.outlier_row_offsets = ts_to_bytes_i32(qresults[0].outlier_row_offsets);
                tr.outlier_cols        = ts_to_bytes_i32(qresults[0].outlier_cols);
                tr.outlier_vals        = ts_to_bytes_u16(qresults[0].outlier_vals);
                tr.act_scale           = ts_to_bytes_u16(qresults[0].act_scale);
                tr.mse                 = qresults[0].mse;
                tr.alpha_used          = qresults[0].best_alpha;
            }
            result->tensors.push_back(std::move(tr));
            n_quantized++;

            if (verbose) {
                printf("tessera-dispatch: quantized %s (3D, %lld experts)\n",
                       name, (long long)n_experts);
            }
        } else {
            // standard 2D weight
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
                    *err_msg = "ts_quantize_2d failed for " + std::string(name);
                }
                gguf_free(out_ctx);
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 2;
            }

            ts_gguf_write_tensor_cluster(out_ctx, name, &qr, out_dim, in_dim);

            ts_dispatch_tensor_result tr;
            tr.name                = name;
            tr.family              = family;
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
            n_quantized++;

            if (verbose) {
                printf("tessera-dispatch: quantized %s (mse=%.6f alpha=%.3f)\n",
                       name, qr.mse, qr.best_alpha);
            }
        }

        // accumulate policy entry
        if (!first_policy_entry) {
            policy_json += ",\n";
        }
        first_policy_entry = false;
        policy_json += "    {\"name\": \"" + std::string(name) + "\", "
                     + "\"family\": \"" + family + "\", "
                     + "\"expert\": " + std::to_string((int)routing.expert) + ", "
                     + "\"alpha\": " + std::to_string(default_alpha) + "}";
    }

    policy_json += "\n  ]\n}";

    // --- step 8: write tessera metadata ---
    ts_gguf_writer_params wparams;
    wparams.seed           = (uint32_t)params->evolve_seed;
    wparams.alpha          = default_alpha;
    wparams.clip           = params->awq_clip;
    wparams.outlier_frac   = params->outlier_frac;
    wparams.policy_summary = policy_json;
    wparams.policy_sha256  = "";
    wparams.build_info     = "";
    wparams.main_tip       = "";
    ts_gguf_write_metadata(out_ctx, &wparams);

    // --- step 9: write output GGUF ---
    if (!params->output_path.empty()) {
        if (!gguf_write_to_file(out_ctx, params->output_path.c_str(), false)) {
            if (err_msg) {
                *err_msg = "failed to write output GGUF: " + params->output_path;
            }
            gguf_free(out_ctx);
            gguf_free(in_ctx);
            ggml_free(ggml_ctx);
            return 3;
        }
        if (verbose) {
            printf("tessera-dispatch: wrote '%s'\n", params->output_path.c_str());
        }
    }

    // --- step 10: write policy JSON alongside ---
    if (!params->policy_out_path.empty()) {
        std::ofstream pf(params->policy_out_path);
        if (pf.is_open()) {
            pf << policy_json << "\n";
            if (verbose) {
                printf("tessera-dispatch: wrote policy '%s'\n", params->policy_out_path.c_str());
            }
        } else {
            fprintf(stderr, "tessera-dispatch: warning: could not write policy to '%s'\n",
                    params->policy_out_path.c_str());
        }
    }

    // --- step 11: populate summary ---
    result->n_tensors_quantized = n_quantized;
    result->n_tensors_skipped   = n_skipped;
    result->total_mse           = total_mse;
    result->policy_json         = policy_json;
    result->policy_sha256       = "";

    // --- cleanup ---
    gguf_free(out_ctx);
    gguf_free(in_ctx);
    ggml_free(ggml_ctx);

    return 0;
}

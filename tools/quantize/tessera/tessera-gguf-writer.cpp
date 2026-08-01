#include "tessera-gguf-writer.h"
#include "tessera-quant.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>

#define TS_PAGE_SIZE      640
#define TS_WORDS_PER_PAGE 32

void ts_gguf_write_metadata(struct gguf_context * ctx, const ts_gguf_writer_params * params) {
    gguf_set_val_u32(ctx, "tessera.version", 1);
    gguf_set_val_u64(ctx, "tessera.quantize.seed", (uint64_t)params->seed);

    if (params->alpha > 0.0f) {
        gguf_set_val_f32(ctx, "tessera.quantize.alpha", params->alpha);
    } else {
        gguf_set_val_str(ctx, "tessera.quantize.alpha", "auto");
    }

    gguf_set_val_f32(ctx, "tessera.quantize.clip", params->clip);
    gguf_set_val_f32(ctx, "tessera.quantize.outlier_frac", params->outlier_frac);

    if (!params->policy_summary.empty()) {
        gguf_set_val_str(ctx, "tessera.calibration.policy", params->policy_summary.c_str());
    }
    if (!params->policy_sha256.empty()) {
        gguf_set_val_str(ctx, "tessera.calibration.sha256", params->policy_sha256.c_str());
    }
    if (!params->build_info.empty()) {
        gguf_set_val_str(ctx, "tessera.provenance.build_info", params->build_info.c_str());
    }
    if (!params->main_tip.empty()) {
        gguf_set_val_str(ctx, "tessera.provenance.main_tip", params->main_tip.c_str());
    }

    // S9 W4A4 activation quantization metadata (additive; absent when disabled)
    if (params->w4a4_enabled) {
        gguf_set_val_bool(ctx, "tessera.w4a4.enabled", true);
        gguf_set_val_u32(ctx, "tessera.w4a4.activation_bits", (uint32_t)params->w4a4_activation_bits);
        gguf_set_val_str(ctx, "tessera.w4a4.scale_mode",
                         params->w4a4_scale_mode.empty() ? "per_token" : params->w4a4_scale_mode.c_str());
        gguf_set_val_f32(ctx, "tessera.w4a4.outlier_thresh", params->w4a4_outlier_thresh);
    }
}

void ts_gguf_write_tensor_cluster(struct gguf_context * ctx,
                                  struct ggml_context * gctx,
                                  const char * base_name,
                                  const void * result,
                                  int64_t out_dim, int64_t in_dim) {
    const auto * res = static_cast<const ts_quant_result_2d *>(result);

    const int64_t pages_per_row = (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;
    const int64_t packed_cols   = pages_per_row * TS_WORDS_PER_PAGE;
    const int64_t lane_cols     = pages_per_row * 32;
    const int64_t n_outliers    = (int64_t)res->outlier_cols.size();

    struct ggml_tensor * t;

    // weight_packed: i32 [out_dim, pages_per_row * 32]
    t = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, packed_cols, out_dim);
    ggml_format_name(t, "%s.weight_packed", base_name);
    t->data = (void *)res->packed.data();
    gguf_add_tensor(ctx, t);

    // weight_page_scales: f16 [out_dim, pages_per_row]
    t = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, pages_per_row, out_dim);
    ggml_format_name(t, "%s.weight_page_scales", base_name);
    t->data = (void *)res->page_scales.data();
    gguf_add_tensor(ctx, t);

    // weight_lane_scales: i8 [out_dim, pages_per_row * 32]
    t = ggml_new_tensor_2d(gctx, GGML_TYPE_I8, lane_cols, out_dim);
    ggml_format_name(t, "%s.weight_lane_scales", base_name);
    t->data = (void *)res->lane_scales.data();
    gguf_add_tensor(ctx, t);

    // weight_outlier_row_offsets: i32 [out_dim + 1]
    t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, out_dim + 1);
    ggml_format_name(t, "%s.weight_outlier_row_offsets", base_name);
    t->data = (void *)res->outlier_row_offsets.data();
    gguf_add_tensor(ctx, t);

    // weight_outlier_cols: i32 [total_outliers]. When there are no outliers the
    // vector is empty and .data() may return nullptr, which trips gguf's
    // non-null assertion at write time; point at a static zero in that case so
    // the length-1 placeholder tensor has a valid backing buffer.
    static const int32_t empty_i32 = 0;
    static const ggml_fp16_t empty_f16 = 0;
    t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_outliers > 0 ? n_outliers : 1);
    ggml_format_name(t, "%s.weight_outlier_cols", base_name);
    t->data = (n_outliers > 0) ? (void *)res->outlier_cols.data()
                               : (void *)&empty_i32;
    gguf_add_tensor(ctx, t);

    // weight_outlier_vals: f16 [total_outliers]
    t = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, n_outliers > 0 ? n_outliers : 1);
    ggml_format_name(t, "%s.weight_outlier_vals", base_name);
    t->data = (n_outliers > 0) ? (void *)res->outlier_vals.data()
                               : (void *)&empty_f16;
    gguf_add_tensor(ctx, t);

    // weight_act_scale: f16 [in_dim] (optional)
    if (!res->act_scale.empty()) {
        t = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, in_dim);
        ggml_format_name(t, "%s.weight_act_scale", base_name);
        t->data = (void *)res->act_scale.data();
        gguf_add_tensor(ctx, t);
    }
}

int ts_gguf_repoint_tensor_cluster(struct ggml_context * gctx,
                                   const char * base_name,
                                   const void * result) {
    const auto * res = static_cast<const ts_quant_result_2d *>(result);
    if (gctx == nullptr || base_name == nullptr || res == nullptr) {
        return 0;
    }

    // Pairs of (suffix, new data pointer). Order matches the cluster writer.
    struct entry { const char * suffix; const void * data; };
    const entry entries[] = {
        { "weight_packed",           res->packed.data() },
        { "weight_page_scales",      res->page_scales.data() },
        { "weight_lane_scales",      res->lane_scales.data() },
        { "weight_outlier_row_offsets", res->outlier_row_offsets.data() },
        { "weight_outlier_cols",     res->outlier_cols.data() },
        { "weight_outlier_vals",     res->outlier_vals.data() },
        { "weight_act_scale",        res->act_scale.data() },
    };

    int repointed = 0;
    for (const auto & e : entries) {
        // Skip the optional act_scale buffer when the refined result dropped it
        // (act_scale is empty when AWQ alpha resolves to 0); the descriptor would
        // not have been written in that case.
        if (e.data == nullptr) {
            continue;
        }
        char want[GGML_MAX_NAME];
        snprintf(want, sizeof(want), "%s.%s", base_name, e.suffix);
        struct ggml_tensor * t = ggml_get_tensor(gctx, want);
        if (t == nullptr) {
            // For act_scale this is expected when it was absent at first write;
            // for the others it indicates the cluster was never written, which
            // is a caller bug. Either way, skip rather than create a new tensor.
            continue;
        }
        t->data = const_cast<void *>(e.data);
        repointed++;
    }
    return repointed;
}

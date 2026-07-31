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
                                  const char * base_name,
                                  const void * result,
                                  int64_t out_dim, int64_t in_dim) {
    const auto * res = static_cast<const ts_quant_result_2d *>(result);

    const int64_t pages_per_row = (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;
    const int64_t packed_cols   = pages_per_row * TS_WORDS_PER_PAGE;
    const int64_t lane_cols     = pages_per_row * 32;
    const int64_t n_outliers    = (int64_t)res->outlier_cols.size();

    struct ggml_init_params gparams = {
        /*mem_size   =*/ 16 * 1024,
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    struct ggml_context * gctx = ggml_init(gparams);
    if (!gctx) {
        std::fprintf(stderr, "%s: ggml_init failed\n", __func__);
        return;
    }

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

    // weight_outlier_cols: i32 [total_outliers]
    t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_outliers > 0 ? n_outliers : 1);
    ggml_format_name(t, "%s.weight_outlier_cols", base_name);
    t->data = (void *)res->outlier_cols.data();
    gguf_add_tensor(ctx, t);

    // weight_outlier_vals: f16 [total_outliers]
    t = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, n_outliers > 0 ? n_outliers : 1);
    ggml_format_name(t, "%s.weight_outlier_vals", base_name);
    t->data = (void *)res->outlier_vals.data();
    gguf_add_tensor(ctx, t);

    // weight_act_scale: f16 [in_dim] (optional)
    if (!res->act_scale.empty()) {
        t = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, in_dim);
        ggml_format_name(t, "%s.weight_act_scale", base_name);
        t->data = (void *)res->act_scale.data();
        gguf_add_tensor(ctx, t);
    }

    ggml_free(gctx);
}

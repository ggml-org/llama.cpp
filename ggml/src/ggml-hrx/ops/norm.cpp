#include "ggml-hrx-ops.h"

#include <climits>
#include <cstring>
#include <string>

bool ggml_hrx_supports_rms_norm(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    if (!ggml_hrx_elem_strides(src, ss) || !ggml_hrx_elem_strides(op, ds) || ss[0] != 1 || ds[0] != 1) {
        return false;
    }
    float eps;
    memcpy(&eps, op->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return false;
    }
    // The kernel needs all offsets below 2^31 - 8.
    const int64_t max_elems = INT_MAX - 1024;
    return src->ne[0] % 4 == 0 && src->ne[0] <= dev->tune->rms_norm_rows_max && ggml_nrows(op) <= max_elems &&
           (int64_t) (ggml_nbytes(src) / 4) <= max_elems && (int64_t) (ggml_nbytes(op) / 4) <= max_elems;
}

bool ggml_hrx_op_rms_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (ggml_nelements(node) == 0) {
        return true;
    }

    int64_t ss[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_elem_strides(src, ss);
    ggml_hrx_elem_strides(node, ds);

    uint32_t eps_bits;
    memcpy(&eps_bits, node->op_params, sizeof(uint32_t));

    const ggml_hrx_tune * tune = ctx->device->tune;
    const int64_t rows = ggml_nrows(node);
    const int64_t ne0  = src->ne[0];
    int64_t lanes = 16;
    while (lanes < tune->rms_norm_lanes_max && 4 * lanes < ne0) {
        lanes *= 2;
    }
    const int64_t vecs           = (ne0 + 4 * lanes - 1) / (4 * lanes);
    const int64_t rows_per_group = tune->norm_wg_size / lanes;

    ggml_hrx_kernel_desc desc;
    desc.file   = "norm/rms_norm_f32.loom";
    desc.symbol = "ggml_rms_norm_f32_rows";
    desc.libs   = { "motifs/reduce.loom" };
    desc.config = {
        { "ggml.norm.wg_size", std::to_string(tune->norm_wg_size) },
        { "ggml.rms_norm.lanes", std::to_string(lanes) },
        { "ggml.rms_norm.vecs", std::to_string(vecs) },
    };
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    return ggml_hrx_dispatch(ctx, kernel, { (rows + rows_per_group - 1) / rows_per_group },
        {
            rows, src->ne[0], src->ne[1], src->ne[2],
            ss[1], ss[2], ss[3], (int64_t) (ggml_nbytes(src) / 4),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / 4),
            (int64_t) eps_bits,
        },
        { src, node });
}

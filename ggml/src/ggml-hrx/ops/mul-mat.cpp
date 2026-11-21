#include "ggml-hrx-ops.h"

#include <climits>
#include <string>

// The matrix-op kernel (kernels/matmul/mul_mat_wmma.loom) gets wave32 providers for gfx11 and gfx12 from motifs/wmma.loom.
// Other archs use the tiled kernel without matrix ops until their layout is validated.
static bool ggml_hrx_mul_mat_wmma_arch(const ggml_hrx_device_context * dev) {
    const ggml_hrx_matrix_op op = dev->arch->matrix_op;
    return (op == GGML_HRX_MATRIX_OP_WMMA_GFX11 || op == GGML_HRX_MATRIX_OP_WMMA_GFX12) && dev->arch->wave32;
}

// Weight type config value of kernels/matmul/lib/mm_core.loom. The value is -1 if the type is not supported.
static int ggml_hrx_mul_mat_weight_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return 2;
        default:             return -1;
    }
}

// Weight strides and spans are in halves. Q4_0 blocks have only 2-byte alignment.
static bool ggml_hrx_mul_mat_weight_strides(const ggml_tensor * w, int64_t s[GGML_MAX_DIMS]) {
    if (w->view_offs % 2 != 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (w->nb[i] % 2 != 0) {
            return false;
        }
        s[i] = (int64_t) (w->nb[i] / 2);
    }
    return true;
}

bool ggml_hrx_supports_mul_mat(const ggml_hrx_device_context * dev, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (ggml_hrx_mul_mat_weight_type(w->type) < 0 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // graph_compute skips an empty dst (for example no output rows).
    if (ggml_is_empty(op)) {
        return true;
    }
    int64_t s[GGML_MAX_DIMS];
    if (!ggml_hrx_mul_mat_weight_strides(w, s) || !ggml_hrx_elem_strides(x, s) || !ggml_hrx_elem_strides(op, s)) {
        return false;
    }
    if (x->view_offs % sizeof(float) != 0 || op->nb[0] != sizeof(float)) {
        return false;
    }
    // An empty tensor cannot be a buffer binding.
    if (ggml_is_empty(w) || ggml_is_empty(x)) {
        return false;
    }
    // The tiled kernels use one workgroup in y for each 32 rows or columns.
    const int64_t max_yz = dev->arch->max_workgroups_yz;
    if ((w->ne[1] + 31) / 32 > max_yz || (x->ne[1] + 31) / 32 > max_yz || x->ne[2] * x->ne[3] > max_yz) {
        return false;
    }
    // Indices, strides and spans are 32-bit launch scalars.
    return ggml_nbytes(w) <= INT_MAX && ggml_nbytes(x) <= INT_MAX && ggml_nbytes(op) <= INT_MAX;
}

// The matrix-op kernel needs contiguous input rows.
// It converts src1 to F16. The src1 precision request must allow this.
static bool ggml_hrx_mul_mat_use_wmma(const ggml_tensor * node, const int64_t xs[GGML_MAX_DIMS]) {
    const ggml_prec prec_src1 = ggml_prec_get_src(node, 1);
    if (prec_src1 != GGML_PREC_UNDEFINED && prec_src1 < GGML_PREC_F16) {
        return false;
    }
    return xs[0] == 1;
}

bool ggml_hrx_op_mul_mat(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    const ggml_tensor * w = node->src[0];
    const ggml_tensor * x = node->src[1];

    int64_t ws[GGML_MAX_DIMS];
    int64_t xs[GGML_MAX_DIMS];
    int64_t ds[GGML_MAX_DIMS];
    ggml_hrx_mul_mat_weight_strides(w, ws);
    ggml_hrx_elem_strides(x, xs);
    ggml_hrx_elem_strides(node, ds);

    const ggml_hrx_mul_mat_vec_tune & mv = ctx->device->tune->mul_mat_vec[w->type];
    // The vector kernel reads K in 256-value superblocks with 4-wide activation loads.
    // The choice does not depend on the column count. Thus a column gets the same result for each batch size.
    const bool k_tail = w->ne[0] % 256 != 0;
    const bool vec    = x->ne[1] <= mv.max_cols && w->ne[0] <= mv.xs_max_k && xs[0] == 1;
    const bool wmma   = x->ne[1] > mv.max_cols && ggml_hrx_mul_mat_wmma_arch(ctx->device) &&
                        ggml_hrx_mul_mat_use_wmma(node, xs);

    const int64_t n_batch = x->ne[2] * x->ne[3];

    ggml_hrx_kernel_desc desc;
    desc.file   = wmma ? "matmul/mul_mat_wmma.loom" : "matmul/mul_mat.loom";
    desc.symbol = wmma ? "ggml_mul_mat_wmma" : vec ? "ggml_mul_mat_vec_xs" : "ggml_mul_mat_tiled";
    desc.libs   = { "motifs/common.loom", "motifs/q4_0.loom", "matmul/lib/mm_core.loom", "matmul/lib/mm_vec.loom" };
    if (wmma) {
        desc.libs.push_back("matmul/lib/mm_wmma.loom");
        desc.libs.push_back("motifs/wmma.loom");
    }
    desc.config = { { "ggml.mul_mat.weight_type", std::to_string(ggml_hrx_mul_mat_weight_type(w->type)) } };
    if (vec) {
        desc.config.push_back({ "ggml.mul_mat.vec_cols", std::to_string(x->ne[1]) });
        desc.config.push_back({ "ggml.mul_mat.vec_xs_workgroups", std::to_string(mv.xs_wg_per_cu * ctx->device->n_cu) });
        desc.config.push_back({ "ggml.mul_mat.vec_xs_depth", std::to_string(mv.xs_depth) });
        desc.config.push_back({ "ggml.mul_mat.vec_k_tail", k_tail ? "1" : "0" });
    }
    ggml_hrx_kernel * kernel = ggml_hrx_get_kernel(ctx->device, desc);

    std::vector<int64_t> workload;
    if (vec) {
        workload = { w->ne[1], w->ne[0], n_batch };
    } else {
        workload = { w->ne[1], x->ne[1], n_batch };
    }

    return ggml_hrx_dispatch(ctx, kernel, workload,
        {
            w->ne[0], w->ne[1], x->ne[1], x->ne[2], x->ne[2] / w->ne[2], x->ne[3] / w->ne[3],
            ws[1], ws[2], ws[3], (int64_t) (ggml_nbytes(w) / 2),
            xs[0], xs[1], xs[2], xs[3], (int64_t) (ggml_nbytes(x) / sizeof(float)),
            ds[1], ds[2], ds[3], (int64_t) (ggml_nbytes(node) / sizeof(float)),
        },
        { w, x, node });
}

#include "ggml-metal-fuse.h"

#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"

#include <algorithm>

// ---- helpers -------------------------------------------------------------

// the pattern outputs (absolute graph node indices); the default is the last node
static const int * ggml_metal_fuse_outputs(const struct ggml_metal_fuse * fuse, int * buf) {
    if (fuse->outputs) {
        return fuse->outputs;
    }

    return buf;
}

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fuse_same_buffer(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fuse_check_norm(const struct ggml_tensor * const * nodes,
                                       const struct ggml_metal_fuse *    fuse,
                                       enum ggml_metal_fuse_mode         mode) {
    GGML_ASSERT(fuse->n_ops >= 2);

    for (int j = 1; j < fuse->n_ops; j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fuse_check_add_chain(const struct ggml_tensor * const * nodes,
                                            const struct ggml_metal_fuse *    fuse,
                                            enum ggml_metal_fuse_mode         mode) {
    GGML_ASSERT(fuse->n_ops >= 2);

    for (int j = 1; j < fuse->n_ops; j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSE_FULL) {
            if (!ggml_metal_fuse_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fuse_check_snake(const struct ggml_tensor * const * nodes,
                                        const struct ggml_metal_fuse *    fuse,
                                        enum ggml_metal_fuse_mode         mode) {
    const struct ggml_tensor * mul0     = nodes[0];
    const struct ggml_tensor * sin_node = nodes[1];
    const struct ggml_tensor * sqr      = nodes[2];
    const struct ggml_tensor * mul1     = nodes[3];
    const struct ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const struct ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const struct ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const struct ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const struct ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);
    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);
    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

// ---- patterns ------------------------------------------------------------

static const enum ggml_op ops_norm_mul[]     = { GGML_OP_NORM,     GGML_OP_MUL };
static const enum ggml_op ops_norm_mul_add[] = { GGML_OP_NORM,     GGML_OP_MUL, GGML_OP_ADD };
static const enum ggml_op ops_rms_norm_mul[]     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const enum ggml_op ops_rms_norm_mul_add[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const enum ggml_op ops_add_2[] = { GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_add_3[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_add_4[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_add_5[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_add_6[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_add_7[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const enum ggml_op ops_snake[] = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const struct ggml_metal_fuse ggml_metal_fuses[] = {
    { GGML_METAL_FUSE_NORM_MUL,     ops_norm_mul,         2, nullptr, 0, ggml_metal_fuse_check_norm },
    { GGML_METAL_FUSE_NORM_MUL_ADD, ops_norm_mul_add,     3, nullptr, 0, ggml_metal_fuse_check_norm },
    { GGML_METAL_FUSE_NORM_MUL,     ops_rms_norm_mul,     2, nullptr, 0, ggml_metal_fuse_check_norm },
    { GGML_METAL_FUSE_NORM_MUL_ADD, ops_rms_norm_mul_add, 3, nullptr, 0, ggml_metal_fuse_check_norm },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_2, 2, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_3, 3, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_4, 4, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_5, 5, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_6, 6, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_ADD_CHAIN, ops_add_7, 7, nullptr, 0, ggml_metal_fuse_check_add_chain },
    { GGML_METAL_FUSE_SNAKE,     ops_snake, 5, nullptr, 0, ggml_metal_fuse_check_snake },
};

const struct ggml_metal_fuse * ggml_metal_fuse_all(int * n) {
    *n = (int) sizeof(ggml_metal_fuses) / sizeof(ggml_metal_fuses[0]);

    return ggml_metal_fuses;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const struct ggml_metal_fuse * ggml_metal_fuse_next(
        const struct ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        enum ggml_metal_fuse_mode mode,
        int * n_out) {
    int n = 0;
    const struct ggml_metal_fuse * all = ggml_metal_fuse_all(&n);

    const struct ggml_metal_fuse * res = nullptr;
    int best = 1;

    for (int i = 0; i < n; i++) {
        const struct ggml_metal_fuse * fuse = &all[i];

        // only look for a longer match than the current best
        if (fuse->n_ops <= best) {
            continue;
        }
        if (idx + fuse->n_ops > n_idxs) {
            continue;
        }

        // the first node must match the pattern start
        if (gf->nodes[node_idxs[idx]]->op != fuse->ops[0]) {
            continue;
        }

        const struct ggml_tensor * nodes[GGML_METAL_FUSE_MAX];

        // common element-wise chain constraints: each node reads the previous one,
        // and all nodes have the same shape
        bool ok = true;
        for (int j = 0; j < fuse->n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];

            if (j > 0) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            continue;
        }

        // ggml_can_fuse_subgraph_ext expects outputs as absolute graph node indices
        int outputs_buf[GGML_MAX_SRC];
        outputs_buf[0] = node_idxs[idx + fuse->n_ops - 1];
        const int * outputs = ggml_metal_fuse_outputs(fuse, outputs_buf);
        const int n_outputs = fuse->n_outputs ? fuse->n_outputs : 1;

        // structural subgraph checks (op sequence, elidable uses, view containment)
        if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fuse->n_ops, fuse->ops, outputs, n_outputs)) {
            continue;
        }

        // pattern-specific checks
        if (fuse->check && !fuse->check(nodes, fuse, mode)) {
            continue;
        }

        best = fuse->n_ops;
        res = fuse;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back
int ggml_metal_fuse_max(const struct ggml_cgraph * gf, int idx) {
    int idxs[GGML_METAL_FUSE_MAX];

    int total = 0;
    int i = idx;

    while (i < gf->n_nodes && total < GGML_METAL_FUSE_MAX) {
        const int n_idxs = std::min(GGML_METAL_FUSE_MAX, (int) gf->n_nodes - i);
        for (int j = 0; j < n_idxs; j++) {
            idxs[j] = i + j;
        }

        int len = 1;
        const struct ggml_metal_fuse * fuse = ggml_metal_fuse_next(gf, idxs, n_idxs, 0, GGML_METAL_FUSE_STRUCTURAL, &len);
        if (!fuse || total + len > GGML_METAL_FUSE_MAX) {
            break;
        }

        total += len;
        i += len;
    }

    return std::max(total, 1);
}

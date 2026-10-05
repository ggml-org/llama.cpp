// checks the MUL_MAT+ADD packs of the Metal graph reorder (devices, pack readers, src1 rows), and that the encoder fuses
// no MUL_MAT+ADD the reorder leaves unpacked
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"

#include <cstdio>
#include <vector>

static constexpr int     n_tensors = 64;
static constexpr float   scale     = 2.0f;
static constexpr float   norm_eps  = 1e-6f;
// a mat-mul the few-row MMA kernels take: K a multiple of their 64-weight step, 2..16 src1 rows
static constexpr int64_t n_k       = 64;
static constexpr int64_t n_m       = 16;
static constexpr int64_t n_rows    = 8;
// src1 row counts below, inside and above the 2..16 rows of the few-row MMA kernels
static const std::vector<int64_t> batch_rows = { 1, 4, 9, 64, 512 };

struct device_case {
    const char * name;

    bool has_native_simdgroup_mm;
    bool packed;
};

// a device with probed simdgroup matrices that are native (MTLGPUFamilyApple7+) only if has_native_simdgroup_mm
static ggml_metal_device_props device_props(bool has_native_simdgroup_mm) {
    ggml_metal_device_props props = {};
    props.has_simdgroup_mm = has_native_simdgroup_mm;
    return props;
}

// the index of t in the nodes of graph, -1 if absent
static int node_index(ggml_cgraph * graph, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == t) {
            return i;
        }
    }
    return -1;
}

// the reordered positions of the tracked nodes
static std::vector<int> node_positions(ggml_cgraph * graph, const std::vector<ggml_tensor *> & tracked) {
    std::vector<int> res;
    for (const ggml_tensor * t : tracked) {
        res.push_back(node_index(graph, t));
    }
    return res;
}

// allocates the tensors of ctx in a CPU buffer marked as weights, as the model loader does
static ggml_backend_buffer_t alloc_weights(ggml_context * ctx) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

// what the add sums with the mat-mul
enum add_operand_kind {
    ADD_RESIDUAL,  // a same-shape activation
    ADD_BIAS,      // a one-row bias weight
    ADD_BIAS_VIEW, // a view of a bias weight
};

static const char * add_operand_name(add_operand_kind kind) {
    switch (kind) {
        case ADD_RESIDUAL:  return "residual";
        case ADD_BIAS:      return "bias";
        case ADD_BIAS_VIEW: return "bias view";
    }
    return "?";
}

static ggml_tensor * new_add_operand(ggml_context * ctx, ggml_context * ctx_w, add_operand_kind kind, int64_t rows) {
    switch (kind) {
        case ADD_RESIDUAL:  return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_m, rows);
        case ADD_BIAS:      return ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_m, 1);
        case ADD_BIAS_VIEW: return ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n_m), n_m, 1);
    }
    return nullptr;
}

// the reordered positions of a mat-mul with rows src1 rows, the add of an operand of kind to it and an independent
// mat-mul, which the reorder runs between the first mat-mul and the add unless the pair is packed
static std::vector<int> reorder_mul_mat_add(int64_t rows, add_operand_kind kind) {
    ggml_init_params params = { n_tensors*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx   = ggml_init(params);
    ggml_context * ctx_w = ggml_init(params);
    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, rows);
    ggml_tensor * mm    = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * add   = ggml_add(ctx, mm, new_add_operand(ctx, ctx_w, kind, rows));
    ggml_tensor * other = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_backend_buffer_t weights = alloc_weights(ctx_w);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, add);
    ggml_build_forward_expand(graph, other);
    ggml_graph_optimize(graph);
    const std::vector<int> res = node_positions(graph, { mm, add, other });

    ggml_backend_buffer_free(weights);
    ggml_free(ctx_w);
    ggml_free(ctx);
    return res;
}

// true if the independent mat-mul of reorder_mul_mat_add does not run between the mat-mul and the add
static bool add_packed(const std::vector<int> & pos) {
    return !(pos[0] < pos[2] && pos[2] < pos[1]);
}

static bool check_pack(const device_case & c) {
    const std::vector<int> pos = reorder_mul_mat_add(n_rows, ADD_RESIDUAL);
    const bool packed = add_packed(pos);

    const bool ok = packed == c.packed;
    std::printf("%s: mat-mul and add packed %d (expected %d): %s\n", c.name, packed, c.packed, ok ? "OK" : "FAIL");
    return ok;
}

static int run_pack_cases(const std::vector<device_case> & cases) {
    int failures = 0;
    for (const device_case & c : cases) {
        failures += check_pack(c) ? 0 : 1;
    }
    return failures;
}

// true if every row count in rows gives the reorder of the first one, on a device that fuses MUL_MAT+ADD
static bool same_reorder_for_rows(const std::vector<int64_t> & rows, add_operand_kind kind) {
    const std::vector<int> first = reorder_mul_mat_add(rows.front(), kind);
    for (auto r = rows.begin() + 1; r != rows.end(); ++r) {
        if (reorder_mul_mat_add(*r, kind) != first) {
            return false;
        }
    }
    return true;
}

// the pack must not depend on the batch size, or graphs with the same nodes get another allocation per ubatch size,
// and a bias add, which the encoder never fuses, must stay free to run next to independent nodes
static bool check_row_independent_pack(add_operand_kind kind) {
    const bool same = same_reorder_for_rows(batch_rows, kind);
    const bool packed = add_packed(reorder_mul_mat_add(batch_rows.front(), kind));
    const bool expected = kind == ADD_RESIDUAL;
    const bool ok = same && packed == expected;
    std::printf("MUL_MAT+ADD of a %s, reorder independent of src1 rows %d, packed %d (expected %d): %s\n",
        add_operand_name(kind), same, packed, expected, ok ? "OK" : "FAIL");
    return ok;
}

static int run_row_independent_pack_cases() {
    int failures = 0;
    for (add_operand_kind kind : { ADD_RESIDUAL, ADD_BIAS, ADD_BIAS_VIEW }) {
        failures += check_row_independent_pack(kind) ? 0 : 1;
    }
    return failures;
}

// true if the encoder's check fuses a few-row MUL_MAT with a same-shape residual, which is a weight if weight_res;
// the check compares Metal buffer ranges, so the tensors live in Metal buffers
static bool encoder_fuses_mul_mat_add(bool weight_res) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU));
    ggml_init_params params = { n_tensors*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx   = ggml_init(params);
    ggml_context * ctx_w = ggml_init(params);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_rows);
    ggml_tensor * mm  = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * add = ggml_add(ctx, mm, ggml_new_tensor_2d(weight_res ? ctx_w : ctx, GGML_TYPE_F32, n_m, n_rows));
    ggml_backend_buffer_t buffer  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_init_params params_gf = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_gf = ggml_init(params_gf);
    ggml_cgraph * gf = ggml_new_graph(ctx_gf);
    ggml_build_forward_expand(gf, add);

    const int idxs[] = { 0, 1 };
    int n_fused = 1;
    const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, 2, 0, GGML_METAL_FUSION_FULL, &n_fused);
    const bool fused = fusion != nullptr && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_MAT_ADD;
    ggml_free(ctx_gf);

    ggml_backend_buffer_free(weights);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx_w);
    ggml_free(ctx);
    return fused;
}

// the encoder may fuse only what the reorder packs, so it must leave a same-shape residual alone when it is a weight
static bool check_encoder_skips_weight_residual() {
    const bool fuses_residual = encoder_fuses_mul_mat_add(false);
    const bool fuses_weight   = encoder_fuses_mul_mat_add(true);
    const bool ok = fuses_residual && !fuses_weight;
    std::printf("encoder fuses MUL_MAT+ADD of a residual %d (expected 1), of a weight %d (expected 0): %s\n",
        fuses_residual, fuses_weight, ok ? "OK" : "FAIL");
    return ok;
}

static ggml_context * graph_ctx() {
    ggml_init_params params = { n_tensors*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

static void expand_all(ggml_cgraph * graph, const std::vector<ggml_tensor *> & outputs) {
    for (ggml_tensor * t : outputs) {
        ggml_build_forward_expand(graph, t);
    }
}

// the graph of outputs in build order, reordered for a device that fuses MUL_MAT+ADD
static ggml_cgraph * optimized_graph(ggml_context * ctx, const std::vector<ggml_tensor *> & outputs) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    expand_all(graph, outputs);
    ggml_graph_optimize(graph);
    return graph;
}

// the position of t among the nodes the encoder runs (views are skipped), -1 if absent
static int encoded_index(ggml_cgraph * graph, const ggml_tensor * t) {
    int res = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == t) {
            return res;
        }
        res += ggml_op_is_empty(node->op) ? 0 : 1;
    }
    return -1;
}

// true if the encoder runs the nodes back to back in this order, so they fuse
static bool encoded_in_a_row(ggml_cgraph * graph, const std::vector<ggml_tensor *> & nodes) {
    const int first = encoded_index(graph, nodes[0]);
    for (size_t j = 1; j < nodes.size(); ++j) {
        if (encoded_index(graph, nodes[j]) != first + (int) j) {
            return false;
        }
    }
    return true;
}

static bool report_reader(const char * name, bool packed, bool reader_after) {
    const bool ok = packed && reader_after;
    std::printf("%s: packed %d, reader after the write %d (expected 1, 1): %s\n", name, packed, reader_after, ok ? "OK" : "FAIL");
    return ok;
}

// x feeds a MUL_MAT+ADD pack chained with RMS_NORM+MUL, and c reads the residual sum h, an output inside the pack
static bool check_chained_pack_reader() {
    ggml_context * ctx = graph_ctx();
    ggml_tensor * x  = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_rows), scale);
    ggml_tensor * mm = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * h  = ggml_add(ctx, mm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_m, n_rows));
    ggml_tensor * n  = ggml_rms_norm(ctx, h, norm_eps);
    ggml_tensor * m  = ggml_mul(ctx, n, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_m));
    ggml_tensor * c  = ggml_scale(ctx, h, scale);

    ggml_cgraph * graph = optimized_graph(ctx, { m, c });
    const bool packed = encoded_in_a_row(graph, { mm, h, n, m });
    const bool after  = encoded_index(graph, c) > encoded_index(graph, h);
    ggml_free(ctx);

    return report_reader("reader of a chained MUL_MAT+ADD sum", packed, after);
}

int main() {
    const std::vector<device_case> devices = {
        { "native simdgroup matrices",       true,  true },
        { "probed or no simdgroup matrices", false, true },
    };
    const int failures = run_pack_cases(devices) + (check_chained_pack_reader() ? 0 : 1) + run_row_independent_pack_cases() +
        (check_encoder_skips_weight_residual() ? 0 : 1);

    return failures == 0 ? 0 : 1;
}

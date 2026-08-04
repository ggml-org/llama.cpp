// ggml-backend-auto.cpp
//
// Heuristic-based per-node backend auto-selection (Project B of
// the heterogeneous-backend scout). The companion to
// ``ggml-backend-residency.cpp`` (Project C). See
// ``ggml/include/ggml-backend.h`` for the API contract.
//
// What this does
// --------------
//
// ``ggml_backend_sched_auto_select(sched, graph, candidates,
// n_candidates)`` walks ``graph->nodes`` and, for each node,
// picks the best candidate backend from ``candidates`` (filtered
// by ``ggml_backend_supports_op``) and applies the choice via
// ``ggml_backend_sched_set_tensor_backend``. The heuristic is
// shape-aware:
//
//   * Layout ops (RESHAPE, VIEW, TRANSPOSE, PERMUTE, CONT) prefer
//     CPU. These are metadata-only ops that don't move data, so
//     the cost of pinning them to a "fancy" backend is pure
//     overhead.
//
//   * Elementwise ops (ADD, MUL, SCALE, CLAMP, REPEAT,
//     LEAKY_RELU, SQR, SQRT, LOG, SIN, COS, UNARY subset) prefer
//     the first ACCEL backend that supports them. On Apple
//     Silicon this is the ANE backend, which handles these ops at
//     lower power than GPU without a kernel-launch round trip.
//
//   * MUL_MAT prefers the first ACCEL backend that supports the
//     op AND whose shape is within the ANE-friendly limits
//     (M*N*K <= 1<<16 for the v1 heuristic; ANE rejects
//     misaligned or oversized matmuls). Larger matmuls fall
//     through to GPU. When Project A lands (ANE matmul backend,
//     separate work), the ACCEL branch starts firing and the
//     small-matmul workload moves off the GPU.
//
//   * Default: the first candidate that supports the op, in
//     ``candidates`` order. This is the safe fallback that
//     produces a working assignment even for ops the heuristic
//     doesn't have a specific rule for.
//
// Phase 1 / Phase 2
// -----------------
//
// Phase 1 is the heuristic (this file). Phase 2 (a follow-up
// commit) will add an optional microbenchmark pass: the selector
// runs each op on each candidate, picks the empirical winner,
// and caches the result keyed by (op_type, ne0, ne1, ne2,
// type_signature). The heuristic remains the default for fast
// graph construction (no warm-up pass); the benchmark is opt-in
// via a separate function.
//
// Constraints
// -----------
//
// * ``n_candidates`` must be >= 1. The first candidate is the
//   preferred default when no heuristic rule applies.
//
// * ``candidates`` may include the same backend type multiple
//   times (e.g. two GPU devices); the heuristic just picks the
//   first matching one.
//
// * The result is applied via ``ggml_backend_sched_set_tensor_backend``;
//   nodes for which no candidate supports the op are left at
//   the scheduler's default assignment and reported via the
//   return value (assigned_count < n_nodes).
//
// * The function does not allocate or free any tensor data; it
//   only updates the scheduler's per-tensor assignment table.
//
// See ``ggml/include/ggml-backend.h`` for the API contract and
// the doc-comment for the higher-level design rationale.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstddef>
#include <cstdint>

namespace {

// Heuristic threshold: matmuls with M*N*K elements below this
// are ANE-friendly (per the ggml-ane backend's documented shape
// limits in the comment block at ggml/src/ggml-ane/ggml-ane.mm:22-35
// and the 64 KB minimum IOSurface alloc). When the ANE matmul
// backend lands (Project A, separate work), matmuls below this
// threshold get the ACCEL backend; larger matmuls go to GPU.
constexpr int64_t GGML_ANE_MATMUL_ELEMENT_LIMIT = 1 << 16; // 65 536

// Layout / metadata-only ops. These don't move data; pinning
// them to a "fancy" backend is pure overhead. CPU is always the
// right answer.
bool ggml_op_is_layout(enum ggml_op op) {
    switch (op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

// Elementwise ops. These are the ANE-friendly class: ANE has
// hardware paths for ADD, MUL, SCALE, CLAMP, REPEAT,
// LEAKY_RELU, SQR, SQRT, LOG, SIN, COS, and a UNARY subset (per
// ggml/src/ggml-ane/ggml-ane.mm:1141-1208 supports_op). When an
// ACCEL backend supports the op, the heuristic prefers it over
// the GPU; this is the "free the GPU for matmul" win.
bool ggml_op_is_elementwise(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_REPEAT:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;
        case GGML_OP_UNARY:
            // The ANE supports a UNARY subset; ggml_backend_supports_op
            // will gate the actual assignment, so the heuristic
            // can short-circuit on the broader class and let
            // supports_op do the final filter.
            return true;
        default:
            return false;
    }
}

// MUL_MAT: matmul-shaped. Project A (ANE matmul backend) will
// add ACCEL support; for now the heuristic still asks for an
// ACCEL backend when the shape is small, but the supports_op
// filter on the ANE backend will return false and the fallback
// (GPU) will be selected. This means the heuristic is correct
// forward: as soon as Project A lands, the small-matmul workload
// starts landing on ANE with no further code change.
bool ggml_op_is_matmul(enum ggml_op op) {
    return op == GGML_OP_MUL_MAT || op == GGML_OP_MUL_MAT_ID
        || op == GGML_OP_TILE640_MATMUL;
}

// Number of "compute elements" in a tensor (a rough cost
// proxy: how many scalar slots the op touches). For a MUL_MAT
// (M, K) @ (K, N) the cost is M*N*K. For an elementwise op
// it's just the tensor's element count. We use this only for
// the ANE-friendly threshold; the GPU path doesn't care.
int64_t ggml_tensor_cost(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_MUL_MAT) {
        // src0 is (M, K), src1 is (K, N). The cost is M*N*K.
        const int64_t M = t->src[0] ? t->src[0]->ne[1] : 0;
        const int64_t K = t->src[0] ? t->src[0]->ne[0] : 0;
        const int64_t N = t->src[1] ? t->src[1]->ne[1] : 0;
        return M * N * K;
    }
    if (t->op == GGML_OP_MUL_MAT_ID) {
        // src0 (M, K, ?) x src1 (K, N, ids) — K is ne[0], N is
        // ne[1], ids is ne[2]. Cost is M*N*K*ids.
        const int64_t M = t->src[0] ? t->src[0]->ne[1] : 0;
        const int64_t K = t->src[0] ? t->src[0]->ne[0] : 0;
        const int64_t N = t->src[1] ? t->src[1]->ne[1] : 0;
        const int64_t ids = t->src[1] ? t->src[1]->ne[2] : 1;
        return M * N * K * ids;
    }
    if (t->op == GGML_OP_TILE640_MATMUL) {
        // Tile640 layout: ne[0] is the K dim (per the in-tree
        // tessera-tile640 conventions), ne[1] is M, ne[2] is N.
        const int64_t M = t->ne[1];
        const int64_t K = t->ne[0];
        const int64_t N = t->ne[2];
        return M * N * K;
    }
    // Fallback: element count for elementwise / scalar ops.
    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        n *= t->ne[i] > 0 ? t->ne[i] : 1;
    }
    return n;
}

// Pick the best backend for one node. ``candidates`` is the
// user's priority list; the heuristic walks it in order, with
// special-case rules for layout / elementwise / matmul ops.
ggml_backend_t ggml_pick_backend_for_node(
        const struct ggml_tensor * node,
        ggml_backend_t * candidates,
        size_t n_candidates) {
    if (n_candidates == 0) {
        return nullptr;
    }
    if (ggml_op_is_layout(node->op)) {
        // Layout: first CPU candidate wins. If there's no CPU
        // candidate, fall through to the first candidate that
        // supports the op (any backend can handle a no-op
        // metadata pass; CPU is just the cheapest).
        for (size_t i = 0; i < n_candidates; ++i) {
            ggml_backend_t b = candidates[i];
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
                ggml_backend_supports_op(b, node)) {
                return b;
            }
        }
        // No CPU candidate: fall through to the default picker.
    } else if (ggml_op_is_elementwise(node->op)) {
        // Elementwise: prefer the first ACCEL candidate that
        // supports the op. The ANE backend is the typical ACCEL
        // on Apple Silicon.
        for (size_t i = 0; i < n_candidates; ++i) {
            ggml_backend_t b = candidates[i];
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL &&
                ggml_backend_supports_op(b, node)) {
                return b;
            }
        }
        // No ACCEL that supports: fall through to GPU then CPU.
    } else if (ggml_op_is_matmul(node->op) &&
               ggml_tensor_cost(node) <= GGML_ANE_MATMUL_ELEMENT_LIMIT) {
        // Small MUL_MAT: prefer ACCEL if it supports the op and
        // the shape is within the ANE-friendly limit. This is the
        // branch that will start firing once Project A lands.
        for (size_t i = 0; i < n_candidates; ++i) {
            ggml_backend_t b = candidates[i];
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL &&
                ggml_backend_supports_op(b, node)) {
                return b;
            }
        }
        // No ACCEL that supports MUL_MAT yet: fall through to
        // GPU for the actual work. The user will get a normal
        // Metal matmul; the heuristic is "right" the moment
        // Project A enables the ACCEL path.
    }
    // Default: first candidate that supports the op. GPU first
    // (the typical candidate order), then CPU. This is the safe
    // fallback that produces a working assignment for any op the
    // heuristic doesn't have a specific rule for.
    for (size_t i = 0; i < n_candidates; ++i) {
        ggml_backend_t b = candidates[i];
        if (ggml_backend_supports_op(b, node)) {
            return b;
        }
    }
    // Nothing supports the op. The scheduler's default
    // assignment will fire; we report this by returning nullptr
    // and the caller counts it as "not assigned".
    return nullptr;
}

} // namespace

extern "C" size_t ggml_backend_sched_auto_select(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        ggml_backend_t * candidates,
        size_t n_candidates) {
    if (sched == nullptr || graph == nullptr || candidates == nullptr ||
        n_candidates == 0) {
        return 0;
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    size_t assigned = 0;
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }
        ggml_backend_t chosen = ggml_pick_backend_for_node(
            node, candidates, n_candidates);
        if (chosen != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, node, chosen);
            ++assigned;
        }
        // If chosen == nullptr, the node is left at the
        // scheduler's default assignment. We don't fail
        // (returning 0); the caller gets assigned_count and can
        // decide whether to handle the gap.
    }
    return assigned;
}

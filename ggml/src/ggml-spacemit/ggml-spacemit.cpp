#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-spacemit.h"

#include "spacemit-session.h"
#include "spacemit-kernels.h"
#include "spacemit-opnode.h"
#include "spacemit-context.h"

#include "ime.h"
#include "repack.h"
#include "spacemit-env.h"
#include "spine_mem_pool.h"
#include "rvv_kernels.h"

#include "ggml-cpu.h"
#include "ggml-profile.h"

// spine-runtime C++ API (hard dependency)
#include <spert.hpp>
#define SPACEMIT_HAS_SPERT 1

using namespace ggml::cpu::riscv64_spacemit;

// global_spine_env_info is initialized from spert::backend_info(),
// GGML_SPACEMIT_WORKERS and SPACEMIT_PERFER_CORE_ID instead of parsing
// /proc/cpuinfo.

namespace ggml::cpu::riscv64_spacemit {

spine_env_info::spine_env_info() {
    // Query spine-runtime for hardware info
    spert::BackendInfo info = spert::backend_info();

    // Determine IME support from the CC core architecture id
    uint16_t arch = (uint16_t) info.core_arch_id;

    // A100 = 0xA064, A200 = 0xA0C8, X100 = 0x5064
    perfer_core_arch_id = spine_core_arch_id{ arch };

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;
    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    // num_cores from env, fallback to spert backend_info
    const char * workers_str = getenv("GGML_SPACEMIT_WORKERS");
    if (workers_str) {
        num_cores = atoi(workers_str);
        if (num_cores <= 0) num_cores = 1;
    } else {
        num_cores = (int) info.num_cores;
        if (num_cores <= 0) num_cores = 1;
    }

    // Optional core pinning: SPACEMIT_PERFER_CORE_ID lists preferred physical
    // CC-core ids (comma-separated, same semantics as the old ggml-cpu/spacemit
    // backend). The graph-compute stream is then bound to these cores and the
    // worker count is capped by the pool size.
    if (const char * perfer_str = getenv("SPACEMIT_PERFER_CORE_ID"); perfer_str && *perfer_str) {
        const int n_cpus = (int) sysconf(_SC_NPROCESSORS_CONF);
        std::vector<int> ids;
        std::string       list(perfer_str);
        size_t            start = 0;
        while (start <= list.size()) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) {
                end = list.size();
            }
            std::string tok = list.substr(start, end - start);
            if (!tok.empty()) {
                int core_id = atoi(tok.c_str());
                if (core_id < 0 || (n_cpus > 0 && core_id >= n_cpus)) {
                    GGML_ABORT("invalid core id %d in SPACEMIT_PERFER_CORE_ID, should be between 0 and %d\n",
                               core_id, n_cpus - 1);
                }
                ids.push_back(core_id);
            }
            start = end + 1;
        }
        if (!ids.empty()) {
            perfer_core_ids = std::move(ids);
            if (num_cores > (int) perfer_core_ids.size()) {
                num_cores = (int) perfer_core_ids.size();
            }
        }
    }

    vlen = info.vlen;
    mem_backend = spine_mem_pool_backend::transparent_hugepage;
    const char * mem_backend_str = getenv("SPACEMIT_MEM_BACKEND");
    if (mem_backend_str) {
        if (strcmp(mem_backend_str, "hugepage") == 0) {
            mem_backend = spine_mem_pool_backend::transparent_hugepage;
        } else if (strcmp(mem_backend_str, "posix") == 0) {
            mem_backend = spine_mem_pool_backend::posix_memalign;
        } else if (strcmp(mem_backend_str, "hugetlb") == 0) {
            mem_backend = spine_mem_pool_backend::hugetlb_1g;
        }
    }

    // Allocate init_barrier (needed by ime.cpp kernel barriers)
    init_barrier = new spine_barrier_t[spine_init_barrier_count];
    spine_barrier_init(init_barrier, spine_init_barrier_count, 2);

    std::string core_ids_str = "auto";
    if (!perfer_core_ids.empty()) {
        core_ids_str.clear();
        for (int core_id : perfer_core_ids) {
            core_ids_str += std::to_string(core_id) + ",";
        }
        core_ids_str.pop_back();
    }
    GGML_LOG_INFO("ggml-spacemit: num_cores=%d, core_ids=%s, arch_id=0x%x, vlen=%zu, shared_mem=%zu, use_ime1=%d, use_ime2=%d\n",
                  num_cores, core_ids_str.c_str(), (unsigned) arch, info.vlen, info.shared_mem_size, use_ime1, use_ime2);
}

spine_env_info::~spine_env_info() {
    delete[] init_barrier;
    init_barrier = nullptr;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit

//** static config

//** helpers

static inline bool op_is_compute(ggml_tensor * node) {
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static spacemit_op_code op_remap_to_spacemit(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:    return SPACEMIT_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID: return SPACEMIT_OP_MUL_MAT_ID;
        case GGML_OP_ADD:        return SPACEMIT_OP_ADD;
        case GGML_OP_RMS_NORM:   return SPACEMIT_OP_RMS_NORM;
        case GGML_OP_ROPE:       return SPACEMIT_OP_ROPE;
        case GGML_OP_SOFT_MAX:   return SPACEMIT_OP_SOFTMAX;
        case GGML_OP_RESHAPE:    return SPACEMIT_OP_RESHAPE;
        case GGML_OP_VIEW:       return SPACEMIT_OP_VIEW;
        case GGML_OP_PERMUTE:    return SPACEMIT_OP_PERMUTE;
        case GGML_OP_TRANSPOSE:  return SPACEMIT_OP_TRANSPOSE;
        case GGML_OP_NONE:       return SPACEMIT_OP_NONE;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU: return SPACEMIT_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU: return SPACEMIT_OP_UNARY_GELU;
                default:                 break;
            }
            break;
        default:
            break;
    }
    return SPACEMIT_OP_INVALID;
}

//** op fusion helpers
//
// Fusion patterns, matched on the graph when the opnode list is built:
//   1. RMS_NORM + MUL    -> SPACEMIT_OP_RMS_NORM_MUL (one opnode at the rms position)
//   2. CONCAT + SSM_CONV -> SPACEMIT_OP_CONCAT_STATE at the concat position (partial write)
//                           + SPACEMIT_OP_SSM_CONV_CONCAT at the ssm_conv position (reads the
//                           concat inputs directly; the concat only materialized its trailing
//                           columns). Both halves must keep their original node positions: the
//                           graph allocator plans buffer reuse on the node order.
//
// Fused opnodes are dispatched by ggml_spacemit_compute_opnode through the
// tensor traits of their primary node (compute_forward_fused).
//
// try_fuse_node contains ALL fusion matching and emission: Pattern 2 consumes
// two nodes at different positions (the concat now, the ssm_conv when the loop
// reaches it — opnode list order is execution order, so the halves keep their
// original node positions); `emitted` carries the pairing across the gap.

// RMS_NORM followed by MUL (rms, weight) that can run as one fused RVV kernel.
static bool rms_norm_mul_fusable(const ggml_tensor * rms, const ggml_tensor * mul) {
    if (mul->op != GGML_OP_MUL || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return x && w && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
           mul->type == GGML_TYPE_F32 && mul->nb[0] == sizeof(float) &&
           w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
           w->ne[0] == rms->ne[0] && ggml_are_same_shape(x, mul);
}

// concat(conv_states, transpose(qkv), dim 0) as built by build_conv_state: contiguous F32 rows, the
// second input laid out transposed (column-major) so the fused kernel reads it column by column.
static bool is_transposed_concat_f32(const ggml_tensor * t) {
    if (t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(t, 0) != 0) {
        return false;
    }
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    return t->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) && s0->nb[0] == sizeof(float) && s1->nb[1] == sizeof(float) &&
           s1->nb[0] != sizeof(float) && s0->ne[1] == t->ne[1] && s1->ne[1] == t->ne[1] && s0->ne[2] == t->ne[2] &&
           s1->ne[2] == t->ne[2] && t->ne[3] == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

// First column of concat c covered by view v, or -1 if v is not a plain window of whole rows' trailing
// columns (same row/plane strides, starting in row 0).
static int64_t concat_view_first_col(const ggml_tensor * c, const ggml_tensor * v) {
    if (v->type != c->type || v->nb[0] != c->nb[0] || (v->ne[1] > 1 && v->nb[1] != c->nb[1]) ||
        (v->ne[2] > 1 && v->nb[2] != c->nb[2]) || v->ne[3] != 1 || v->view_offs / c->nb[1] != 0) {
        return -1;
    }
    const int64_t col = (int64_t) ((v->view_offs % c->nb[1]) / c->nb[0]);
    if (col + v->ne[0] > c->ne[0] || v->ne[1] > c->ne[1] || v->ne[2] > c->ne[2]) {
        return -1;
    }
    return col;
}

// concat -> ssm_conv fusion analysis: SSM_CONV reads the two concat inputs directly and CONCAT only
// writes the trailing columns that are read back through views (the conv state update). Returns the
// single ssm_conv consumer (with its graph index) and the first column any view reads back. Fails if
// the concat has any other consumer (or a second ssm_conv) that needs the full concat output.
static bool ssm_conv_concat_fusable(const ggml_cgraph * graph, int ci,
                                    ggml_tensor ** ssm_out, int * ssm_idx_out, int64_t * col0_out) {
    ggml_tensor * c = graph->nodes[ci];
    if (!is_transposed_concat_f32(c) || c->src[0]->ne[0] != 3) {
        return false;
    }

    ggml_tensor * ssm    = nullptr;
    int           ssm_idx = -1;
    int64_t       col0    = c->ne[0];  // min column read back through views

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * t = graph->nodes[i];

        if (t->view_src == c) {
            const int64_t col = concat_view_first_col(c, t);
            if (col < 0) {
                return false;
            }
            col0 = std::min(col0, col);
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * s = t->src[j];
            if (s == nullptr) {
                continue;
            }
            if (s == c) {
                if (t->op == GGML_OP_SSM_CONV && j == 0 && t->src[1] && t->src[1]->ne[0] == 4 &&
                    t->src[2] == c->src[1] && t->src[3] == c->src[0]) {
                    // the graph keeps the concat inputs alive until this node (build_ssm_conv)
                    if (ssm != nullptr) {
                        return false;  // more than one ssm_conv consumer
                    }
                    ssm     = t;
                    ssm_idx = i;
                } else if (t->view_src != c) {
                    return false;  // some other consumer needs the full concat output
                }
            } else if (s->view_src == c) {
                const int64_t col = concat_view_first_col(c, s);
                if (col < 0) {
                    return false;
                }
                col0 = std::min(col0, col);
            }
        }
    }

    if (ssm == nullptr || ssm_idx <= ci) {
        return false;
    }

    // The fused kernels re-check their layouts at compute time and abort on a mismatch (there is no
    // fallback once the concat half has materialized only its trailing columns), so the gate must be
    // a superset of forward_concat_dim0_transposed_f32 + forward_ssm_conv_concat_f32: types, element
    // strides and the n_t / d_inner shape agreement between the concat and the ssm_conv output.
    const ggml_tensor * w = ssm->src[1];
    if (ssm->type != GGML_TYPE_F32 || ssm->nb[0] != sizeof(float) ||
        w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float) ||
        c->src[1]->ne[0] != ssm->ne[1] || c->ne[1] != ssm->ne[0]) {
        return false;
    }

    *ssm_out     = ssm;
    *ssm_idx_out = ssm_idx;
    *col0_out    = col0;
    return true;
}

static bool try_fuse_node(const ggml_cgraph * graph, int & i, std::vector<spacemit_opnode> & nodes,
                          std::unordered_set<ggml_tensor *> & emitted) {
    ggml_tensor * n = graph->nodes[i];

    // Second half of Pattern 2, arriving here: an ssm_conv paired with an earlier concat.
    // Emitted at its own position — opnode list order is execution order, and the graph
    // allocator planned the ssm_conv output buffer on the original node order (see Pattern 2).
    if (emitted.count(n)) {
        nodes.emplace_back(n, std::vector<ggml_tensor *>{}, SPACEMIT_OP_SSM_CONV_CONCAT);
        return true;
    }

    // Find the next compute node, skipping empty (NONE) nodes. test-backend-ops
    // inserts NONE sentinels between ops for out-of-bounds detection, which must
    // not break fusion adjacency. Production graphs have no such sentinels.
    int next_idx = -1;
    for (int j = i + 1; j < graph->n_nodes; ++j) {
        if (op_is_compute(graph->nodes[j])) {
            next_idx = j;
            break;
        }
    }
    ggml_tensor * next_node = (next_idx >= 0) ? graph->nodes[next_idx] : nullptr;

    // Pattern 1: RMS_NORM + MUL
    if (n->op == GGML_OP_RMS_NORM && next_node && next_node->op == GGML_OP_MUL) {
        const int          idxs[2] = { i, next_idx };
        const enum ggml_op ops[2]  = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_ext(graph, idxs, ops, 2) && rms_norm_mul_fusable(n, next_node)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_RMS_NORM_MUL);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i = next_idx;  // skip past the fused MUL node
            return true;
        }
    }

    // Pattern 2: CONCAT + SSM_CONV. Runs as TWO opnodes at the original node positions: the
    // concat writes only the trailing columns that views read back (the conv state update), and
    // the ssm_conv, at its own position after the state update CPY, reads the concat inputs
    // directly. The ssm_conv must NOT be pulled to the concat position: the graph allocator
    // plans buffer reuse on the original node order, so writing the ssm_conv output early can
    // overwrite tensors that are still live between the two positions (bisected on Qwen3.5 GDN,
    // 2026-10-20: fused-at-concat corrupted both 2B and 4B on K1/K3).
    if (n->op == GGML_OP_CONCAT) {
        ggml_tensor * ssm     = nullptr;
        int           ssm_idx = -1;
        int64_t       col0    = 0;
        if (ssm_conv_concat_fusable(graph, i, &ssm, &ssm_idx, &col0)) {
            spacemit_opnode cat(n, {}, SPACEMIT_OP_CONCAT_STATE);
            cat.kernel_params[0] = (int32_t) col0;
            nodes.push_back(std::move(cat));
            // the ssm_conv half is emitted by try_fuse_node when the loop reaches its own
            // node position (see the top of this function) — never here at the concat position
            emitted.insert(ssm);
            return true;
        }
    }

    return false;
}

//** buffer interface
//
// Reuses spine_mem_pool_alloc/free for allocation and repack for init_tensor/set_tensor.
// Pattern adapted from ggml-cpu/spacemit/ime.cpp buffer management.

static void ggml_backend_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    if (base == nullptr) {
        return;
    }

    spine_mem_pool_free(base);
}

static void * ggml_backend_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    return base;
}

static enum ggml_status ggml_backend_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                   ggml_tensor *         tensor) {
    tensor->extra =
        (void *) const_cast<ggml::spacemit::tensor_traits_base *>(ggml_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                         ggml_tensor *         tensor,
                                                         uint8_t               value,
                                                         size_t                offset,
                                                         size_t                size) {
    GGML_ASSERT(tensor);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

static void ggml_backend_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                      ggml_tensor *         tensor,
                                                      const void *          data,
                                                      size_t                offset,
                                                      size_t                size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    // Quantized tensors with tensor traits are physically repacked, so they
    // must be uploaded atomically. Plain tensors preserve the ggml layout and
    // support the partial writes used by test-backend-ops and graph inputs.
    if (tensor->extra) {
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(size == ggml_nbytes(tensor));
        auto ok = ggml_riscv64_spacemit_repack_tensor(tensor, data, size);
        GGML_ASSERT(ok == 0);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const ggml_tensor *   tensor,
                                                     void *                data,
                                                     size_t                offset,
                                                     size_t                size) {
    const size_t logical_size = ggml_nbytes(tensor);
    GGML_ASSERT(offset <= logical_size && size <= logical_size - offset);
    if (tensor->extra) {
        if (offset == 0 && size == logical_size) {
            ggml_riscv64_spacemit_unpack_tensor(tensor, data, size);
        } else {
            std::vector<uint8_t> logical_data(logical_size);
            ggml_riscv64_spacemit_unpack_tensor(tensor, logical_data.data(), logical_size);
            memcpy(data, logical_data.data() + offset, size);
        }
    } else {
        memcpy(data, (const char *) tensor->data + offset, size);
    }
    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_spacemit_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

//** buffer type interface

static const char * ggml_backend_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "SPACEMIT";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                              size_t size) {
    void * base = spine_mem_pool_alloc(size, 64);
    if (base == nullptr) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_spacemit_buffer_i, base, size);
}

static size_t ggml_backend_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_spacemit_nbytes(buft, tensor);
}

static bool ggml_backend_spacemit_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_spacemit_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_spacemit_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_spacemit_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr, // default: alloc_buffer + linear allocator
    /* .get_alignment    = */ ggml_backend_spacemit_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_spacemit_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_spacemit_buffer_type_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr, // default: per-tensor get_alloc_size
    /* .is_host          = */ ggml_backend_spacemit_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_spacemit_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type buft_s = {
        /* .iface   = */ ggml_backend_spacemit_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    if (buft_s.device == nullptr) {
        buft_s.device = dev;
    }
    return &buft_s;
}

static bool ggml_backend_buffer_is_spacemit(const struct ggml_backend_buffer * b) {
    return b && b->buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
}

//** backend interface

spacemit_session::~spacemit_session() {
    free(workspace);
}

static const char * ggml_backend_spacemit_name(ggml_backend_t backend) {
    auto sess = static_cast<spacemit_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_spacemit_free(ggml_backend_t backend) {
    // The device registry owns the shared session.
    delete backend;
}

static ggml_status ggml_backend_spacemit_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<spacemit_session *>(backend->context);
    // Serialize hardware use across backend handles. The Stream itself remains
    // graph-scoped so idle contexts do not retain CC hardware resources.
    std::lock_guard<std::mutex> stream_lock(sess->stream_mutex);

    GGML_LOG_DEBUG("ggml-spacemit: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    // Build (or reuse) the opnode list for this graph. The graph uid identifies a
    // stable shape-class, so the list is only rebuilt when the topology changes.
    // try_fuse_node folds matching patterns into fused opnodes and consumes their
    // nodes; non-matching nodes are appended as single ops.
    if (!(graph->uid != 0 && sess->cached_graph.uid == graph->uid)) {
        sess->cached_graph.uid = graph->uid;
        sess->cached_graph.nodes.clear();
        sess->cached_graph.nodes.reserve(graph->n_nodes);
        std::unordered_set<ggml_tensor *> emitted;  // nodes already emitted by try_fuse_node
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }
            // On a match, try_fuse_node pushes the fused opnodes and advances i past
            // the nodes it consumed; otherwise the node is appended as a single op.
            if (try_fuse_node(graph, i, sess->cached_graph.nodes, emitted)) {
                continue;
            }
            sess->cached_graph.nodes.emplace_back(n);
        }

        int n_fused = 0;
        for (const auto & node : sess->cached_graph.nodes) {
            if (!node.fused.empty() || node.opcode != SPACEMIT_OP_INVALID) {
                n_fused++;
            }
        }
        GGML_LOG_DEBUG("ggml-spacemit: %s graph %llu: %zu opnodes, %d fused\n", sess->c_name(),
                       (unsigned long long) graph->uid, sess->cached_graph.nodes.size(), n_fused);
    }
    const auto & opnodes = sess->cached_graph.nodes;

    int n_threads = sess->num_cores > 0 ? sess->num_cores : 1;
    size_t workspace_size = 0;
    for (const auto & node : opnodes) {
        size_t node_workspace_size = 0;
        if (ggml_spacemit_get_work_size(n_threads, node.node, &node_workspace_size)) {
            workspace_size = std::max(workspace_size, node_workspace_size);
        }
    }

    if (sess->workspace_size < workspace_size) {
        void * new_workspace = realloc(sess->workspace, workspace_size);
        if (new_workspace == nullptr) {
            GGML_LOG_ERROR("ggml-spacemit: failed to allocate work buffer (%zu bytes)\n", workspace_size);
            return GGML_STATUS_ALLOC_FAILED;
        }
        sess->workspace      = new_workspace;
        sess->workspace_size = workspace_size;
    }
    auto * workspace = static_cast<uint8_t *>(sess->workspace);

    spert::StreamConfig stream_cfg;
    stream_cfg.n_cores = (uint32_t) sess->num_cores;
    {
        const auto & pool = global_spine_env_info.perfer_core_ids;
        stream_cfg.core_ids.assign(pool.begin(),
                                   pool.begin() + std::min<size_t>(pool.size(), stream_cfg.n_cores));
    }
    spert::Stream stream(stream_cfg);
    if (!stream.valid()) {
        GGML_LOG_ERROR("ggml-spacemit: failed to create spert stream\n");
        return GGML_STATUS_FAILED;
    }

    auto * contexts = sess->contexts.data();
    auto fut = stream.launch(
        spert::Grid{(uint32_t)n_threads},
        [&opnodes, workspace_size = sess->workspace_size, workspace, contexts,
         shared_mem_size = spert::backend_info().shared_mem_size](spert::Context * runtime) {
            // Match the old ggml-cpu/spacemit path: use the complete
            // worker-local TCM buffer directly.  alloc_shared() is a
            // dynamic sub-allocation API and is not equivalent here; it
            // adds allocator bookkeeping and can select a different shared
            // pool block layout than the original RVV kernels expect.
            auto shared = shared_mem_size > 0 ? runtime->shared_buffer() : spert::SharedBufferView{};
            if (shared_mem_size > 0 && (!shared || shared.size < shared_mem_size)) {
                throw std::runtime_error("ggml-spacemit: worker shared buffer unavailable");
            }

            const uint32_t ith = runtime->program_id(0);
            auto & ctx = contexts[ith];
            ctx.reset(*runtime, ith, runtime->grid_dim(0), workspace, workspace_size, shared);

            for (const auto & node : opnodes) {
                ggml_tensor * op = node.node;

                ggml_profile_log_spacemit_op_begin(op, (int) ith, (int) runtime->grid_dim(0));

                if (!ggml_spacemit_compute_opnode(ctx, node)) {
                    throw std::runtime_error(std::string("ggml-spacemit: failed to dispatch op ") +
                                             ggml_op_desc(op) + " (" + ggml_type_name(op->type) + ")");
                }
                ggml_profile_log_spacemit_op_end(op, (int) ith, (int) runtime->grid_dim(0));

                // Every core must finish the current node before any core starts
                // the next one. A bounded pointer look-ahead is not sufficient:
                // dependencies can be farther away or hidden behind aliases/views.
                ctx.sync();
            }

            ctx.clear();
        }
    );

    const spert::Status status = fut.sync();

    if (status != spert::Status::Ok) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_synchronize(ggml_backend_t backend) {
    // graph_compute waits on its launch future before returning, so there is
    // no outstanding asynchronous work to synchronize here.
    GGML_UNUSED(backend);
}

static struct ggml_backend_i spacemit_backend_i = {
    /* .get_name                = */ ggml_backend_spacemit_name,
    /* .free                    = */ ggml_backend_spacemit_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_spacemit_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_spacemit_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_spacemit_guid() {
    static ggml_guid guid = { 0x8b, 0x68, 0xed, 0xbf, 0xef, 0x23, 0x2e, 0x4a,
                              0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22 };
    return &guid;
}

bool ggml_backend_is_spacemit(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_spacemit_name;
}

//** device interface

static ggml_backend_t ggml_backend_spacemit_device_init(ggml_backend_dev_t dev, const char * params) {
    auto sess = static_cast<spacemit_session *>(dev->context);

    return new ggml_backend{
        /* .guid      = */ ggml_backend_spacemit_guid(),
        /* .interface = */ spacemit_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_spacemit_device_get_name(ggml_backend_dev_t dev) {
    auto sess = static_cast<spacemit_session *>(dev->context);
    return sess->c_name();
}

static const char * ggml_backend_spacemit_device_get_description(ggml_backend_dev_t dev) {
    return "Spacemit K3/X200 AI Engine";
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // SPACEMIT backend shares host memory (weights are in DRAM accessible by CC cores).
    // Report available system memory so the scheduler assigns tensors to us.
    size_t vfree = 0, vtotal = 0;
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    if (cpu_dev && cpu_dev->iface.get_memory) {
        cpu_dev->iface.get_memory(cpu_dev, &vfree, &vtotal);
    }
    *free  = vfree;
    *total = vtotal;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_spacemit_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_spacemit_device_get_name(dev);
    props->description = ggml_backend_spacemit_device_get_description(dev);
    props->type        = ggml_backend_spacemit_device_get_type(dev);
    ggml_backend_spacemit_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        // the model is mmapped in host memory, then copied into spine pool buffers
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_buffer_type_t ggml_backend_spacemit_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_spacemit_buffer_type(dev);
}

static bool ggml_backend_spacemit_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // all srcs and dst must be on our buffer type
    auto check_buf = [&](const ggml_tensor * t) -> bool {
        if (!t || !t->buffer) return true;  // unallocated is OK
        return ggml_backend_buffer_is_spacemit(t->buffer);
    };

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !check_buf(op->src[i])) {
            return false;
        }
    }
    if (!check_buf(op)) {
        return false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_SCALE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ROPE:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONCAT:
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CONT:
        case GGML_OP_REPEAT:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_L2_NORM:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_PAD:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SET:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
            supp = ggml_spacemit_get_tensor_traits(op) != nullptr;
            break;

        default:
            break;
    }

    return supp;

    GGML_UNUSED(dev);
}

static bool ggml_backend_spacemit_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_spacemit_device_i = {
    /* .get_name             = */ ggml_backend_spacemit_device_get_name,
    /* .get_description      = */ ggml_backend_spacemit_device_get_description,
    /* .get_memory           = */ ggml_backend_spacemit_device_get_memory,
    /* .get_type             = */ ggml_backend_spacemit_device_get_type,
    /* .get_props            = */ ggml_backend_spacemit_device_get_props,
    /* .init_backend         = */ ggml_backend_spacemit_device_init,
    /* .get_buffer_type      = */ ggml_backend_spacemit_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_spacemit_device_supports_op,
    /* .supports_buft        = */ ggml_backend_spacemit_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//** backend registry

#define GGML_SPACEMIT_MAX_DEVICES 1

struct ggml_spacemit_registry {
    ggml_spacemit_registry(ggml_backend_reg_t reg);
    ~ggml_spacemit_registry();

    ggml_backend_device devices[GGML_SPACEMIT_MAX_DEVICES];
};

ggml_spacemit_registry::ggml_spacemit_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-spacemit: Spacemit backend (experimental) : allocating new registry\n");

    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        devices[i].iface = ggml_backend_spacemit_device_i;
        devices[i].reg   = reg;

        auto * sess = new spacemit_session();

        // populate session from spert backend_info and env
        spert::BackendInfo info = spert::backend_info();
        sess->num_cores = global_spine_env_info.num_cores;
        sess->arch_id   = static_cast<int64_t>(global_spine_env_info.perfer_core_arch_id);
        sess->vlen      = (int64_t) info.vlen;
        sess->use_ime1  = global_spine_env_info.use_ime1;
        sess->use_ime2  = global_spine_env_info.use_ime2;
        sess->contexts.resize(sess->num_cores);

        sess->name = "SPACEMIT" + std::to_string(i);

        devices[i].context = sess;
    }
}

ggml_spacemit_registry::~ggml_spacemit_registry() {
    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        auto sess = static_cast<spacemit_session *>(devices[i].context);
        delete sess;
    }
}

static const char * ggml_backend_spacemit_reg_get_name(ggml_backend_reg_t reg) {
    return "SPACEMIT";
    GGML_UNUSED(reg);
}

static bool ggml_spacemit_disabled(void) {
    const char * str = getenv("GGML_SPACEMIT_BACKEND");
    return str == nullptr || strcmp(str, "1") != 0;
}

static size_t ggml_backend_spacemit_reg_get_device_count(ggml_backend_reg_t reg) {
    if (ggml_spacemit_disabled()) {
        return 0;
    }
    return GGML_SPACEMIT_MAX_DEVICES;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_spacemit_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_spacemit_registry *>(reg->context);

    if (hreg == nullptr) {
        return nullptr;
    }

    if (index >= GGML_SPACEMIT_MAX_DEVICES || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

static void * ggml_backend_spacemit_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static void ggml_spacemit_init(ggml_backend_reg * reg) {
    if (ggml_spacemit_disabled()) {
        return;
    }

    reg->context = new ggml_spacemit_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_spacemit_reg_i = {
    /* .get_name         = */ ggml_backend_spacemit_reg_get_name,
    /* .get_device_count = */ ggml_backend_spacemit_reg_get_device_count,
    /* .get_device       = */ ggml_backend_spacemit_reg_get_device,
    /* .get_proc_address = */ ggml_backend_spacemit_get_proc_address,
};

ggml_backend_reg_t ggml_backend_spacemit_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_spacemit_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_spacemit_init(&reg);
        }
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_spacemit_init(void) {
    ggml_backend_reg_t reg = ggml_backend_spacemit_reg();
    if (!reg) {
        return nullptr;
    }

    if (ggml_spacemit_disabled()) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_spacemit_reg_get_device(reg, 0);
    if (!dev) {
        return nullptr;
    }

    return ggml_backend_spacemit_device_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_spacemit_reg)

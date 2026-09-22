#include "qsa-score.hpp"

#include "ggml-impl.h"

#include "common.hpp"
#include "gemm.hpp"

#include <algorithm>

// The QSA indexer rectifies every head dot product before it sums the heads:
//   score[b, t] = sum over h of relu(dot(pooled[:, b], q[:, h, t]))
// The model graph writes the un-reduced product in full, [n_blocks, n_head*n_tps], which is
// 512 MiB at 128k context, relu's it in place, then collapses it to [n_blocks, n_tps].
// This runs the same GEMM in column tiles into a small pool buffer and reduces each tile
// straight into the final destination, avoiding reads and writes of the wide product.
//
// The chain is MUL_MAT -> RESHAPE -> RELU -> n_head x (VIEW, CONT or ADD).

static constexpr int SYCL_QSA_SCORE_MAX_HEADS = 8;

struct qsa_score_chain {
    int     i_mm;
    int     i_out;  // the last combiner, the only node of the chain that keeps a buffer
    int64_t n_heads;
};

static bool ggml_sycl_qsa_score_shape(const ggml_cgraph * cgraph, int i, qsa_score_chain * out) {
    if (i + 4 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * mm = cgraph->nodes[i];
    ggml_tensor * rs = cgraph->nodes[i + 1];
    ggml_tensor * rl = cgraph->nodes[i + 2];

    if (mm->op != GGML_OP_MUL_MAT || rs->op != GGML_OP_RESHAPE || rl->op != GGML_OP_UNARY) {
        return false;
    }
    if (ggml_get_unary_op(rl) != GGML_UNARY_OP_RELU) {
        return false;
    }

    const int64_t n_heads = rs->ne[1];
    if (n_heads < 1 || n_heads > SYCL_QSA_SCORE_MAX_HEADS) {
        return false;
    }

    const int i_out = i + 2 * (int) n_heads + 2;
    if (i_out >= cgraph->n_nodes) {
        return false;
    }

    // the GEMM and the relu are fused away, so nothing outside the chain may read them
    for (const ggml_tensor * n : { mm, rl }) {
        if (n->type != GGML_TYPE_F32 || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->view_src) {
            return false;
        }
        if (n->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        if (ggml_is_empty(n)) {
            return false;
        }
    }
    if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, i + 2) != n_heads) {
        return false;
    }

    // plain f32 GEMM of two contiguous operands, batched over ne2
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (!src0 || !src1 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(mm)) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[2] != src1->ne[2] || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (mm->ne[0] != src0->ne[1] || mm->ne[1] != src1->ne[1] || mm->ne[2] != src0->ne[2] || mm->ne[3] != 1) {
        return false;
    }
    // the precision request and the hadamard hint both change what ggml_sycl_mul_mat runs
    if (ggml_get_op_params_i32(mm, 0) != GGML_PREC_DEFAULT || ggml_get_op_params_i32(mm, 1) != GGML_HINT_NONE) {
        return false;
    }
    if (src0->ne[0] > INT32_MAX || src0->ne[1] > INT32_MAX || mm->ne[1] > INT32_MAX) {
        return false;
    }

    // the reshape only splits the GEMM's ne1 into heads and tokens
    if (rs->view_src != mm || rs->src[0] != mm || rs->view_offs != 0 || !ggml_is_contiguous(rs)) {
        return false;
    }
    if (rs->ne[0] != mm->ne[0] || rs->ne[1] * rs->ne[2] != mm->ne[1] || rs->ne[3] != mm->ne[2]) {
        return false;
    }
    if (rl->src[0] != rs || !ggml_are_same_shape(rl, rs) || !ggml_is_contiguous(rl)) {
        return false;
    }

    const int64_t n_blocks = rs->ne[0];
    const int64_t n_tps    = rs->ne[2];
    const int64_t n_stream = rs->ne[3];

    const ggml_tensor * acc = nullptr;
    for (int64_t h = 0; h < n_heads; ++h) {
        ggml_tensor * vw = cgraph->nodes[i + 3 + 2 * h];
        ggml_tensor * cb = cgraph->nodes[i + 4 + 2 * h];

        if (vw->op != GGML_OP_VIEW || vw->view_src != rl || vw->src[0] != rl) {
            return false;
        }
        if (vw->view_offs != (size_t) h * rl->nb[1]) {
            return false;
        }
        if (vw->ne[0] != n_blocks || vw->ne[1] != n_tps || vw->ne[2] != n_stream || vw->ne[3] != 1) {
            return false;
        }
        if (vw->nb[0] != rl->nb[0] || vw->nb[1] != rl->nb[2] || vw->nb[2] != rl->nb[3]) {
            return false;
        }
        if (ggml_node_get_use_count(cgraph, i + 3 + 2 * h) != 1) {
            return false;
        }

        if (h == 0) {
            if (cb->op != GGML_OP_CONT || cb->src[0] != vw) {
                return false;
            }
        } else {
            if (cb->op != GGML_OP_ADD || cb->src[0] != acc || cb->src[1] != vw) {
                return false;
            }
        }
        if (cb->type != GGML_TYPE_F32 || cb->view_src || !ggml_is_contiguous(cb) || ggml_is_empty(cb)) {
            return false;
        }
        if ((cb->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (cb->ne[0] != n_blocks || cb->ne[1] != n_tps || cb->ne[2] != n_stream || cb->ne[3] != 1) {
            return false;
        }
        // only the last combiner survives the fusion, so every earlier one must be private
        if (h + 1 < n_heads) {
            if (cb->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
                return false;
            }
            if (ggml_node_get_use_count(cgraph, i + 4 + 2 * h) != 1) {
                return false;
            }
        }
        acc = cb;
    }

    if (out) {
        *out = { i, i_out, n_heads };
    }
    return true;
}

bool ggml_sycl_can_fuse_qsa_score(const ggml_cgraph * cgraph, int i) {
    return g_ggml_sycl_enable_fusion && ggml_sycl_qsa_score_shape(cgraph, i, nullptr);
}

int ggml_sycl_qsa_score_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    qsa_score_chain c;
    if (!ggml_sycl_qsa_score_shape(cgraph, node_idx, &c)) {
        return 0;
    }
    return c.i_out - c.i_mm;
}

// dst[b, t0 + t, s] = sum over h of relu(tile[b, h, t, s])
static void k_qsa_score_reduce(const float * tile, float * dst, int64_t n_blocks, int64_t n_heads,
                               int64_t nt, int64_t n_tps, int64_t t0, const sycl::nd_item<2> & item) {
    const int64_t b = item.get_global_id(1);
    if (b >= n_blocks) {
        return;
    }

    const int64_t r = item.get_global_id(0);
    const float * col = tile + n_blocks*n_heads*r + b;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_heads; ++h) {
        acc += sycl::fmax(col[n_blocks*h], 0.0f);
    }

    const int64_t t = r % nt;
    const int64_t s = r / nt;
    dst[n_blocks*((t0 + t) + n_tps*s) + b] = acc;
}

// how much pool memory one column tile may take; the tile replaces a 512 MiB graph tensor
static int64_t qsa_score_tile_bytes() {
    static int64_t mib = -1;
    if (mib < 0) {
        const char * e = getenv("GGML_SYCL_QSA_SCORE_TILE_MIB");
        mib = e ? std::max(1, atoi(e)) : 32;
    }
    return mib << 20;
}

// Runs the chain matched by ggml_sycl_can_fuse_qsa_score(); returns the extra nodes consumed.
int ggml_sycl_fuse_qsa_score(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    qsa_score_chain c;
    if (!g_ggml_sycl_enable_fusion || !ggml_sycl_qsa_score_shape(cgraph, i, &c)) {
        return 0;
    }

    const ggml_tensor * src0 = cgraph->nodes[c.i_mm]->src[0];
    const ggml_tensor * src1 = cgraph->nodes[c.i_mm]->src[1];
    ggml_tensor *       dst  = cgraph->nodes[c.i_out];

    const int64_t k        = src0->ne[0];
    const int64_t n_blocks = src0->ne[1];
    const int64_t n_heads  = c.n_heads;
    const int64_t n_tps    = dst->ne[1];
    const int64_t n_stream = dst->ne[2];

    const int64_t row_bytes = n_blocks * n_heads * n_stream * (int64_t) sizeof(float);
    int64_t       t_tile    = std::min(n_tps, std::max<int64_t>(1, qsa_score_tile_bytes() / row_bytes));

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<float> tile(ctx.pool(), (size_t) (n_blocks * n_heads * t_tile * n_stream));

    const char * src0_dd = (const char *) src0->data;
    const char * src1_dd = (const char *) src1->data;
    float *      dst_dd  = (float *) dst->data;

    GGML_ASSERT(src0_dd && src1_dd && dst_dd);

    for (int64_t t0 = 0; t0 < n_tps; t0 += t_tile) {
        const int64_t nt = std::min(t_tile, n_tps - t0);
        const int64_t n  = n_heads * nt;

        for (int64_t s = 0; s < n_stream; ++s) {
            const char * a = src0_dd + s*src0->nb[2];
            const char * b = src1_dd + s*src1->nb[2] + t0*n_heads*src1->nb[1];
            float *      d = tile.get() + s*n_blocks*n;

#if GGML_SYCL_DNNL
            if (g_ggml_sycl_enable_dnn) {
                DnnlGemmWrapper::row_gemm(ctx, (int) n_blocks, (int) n, (int) k, a, DnnlGemmWrapper::to_dt<float>(),
                                          b, DnnlGemmWrapper::to_dt<float>(), d, DnnlGemmWrapper::to_dt<float>(),
                                          stream);
            } else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n_blocks, n, k,
                    dpct::get_value(&alpha, *stream), (const float *) a, k, (const float *) b, k,
                    dpct::get_value(&beta, *stream), d, n_blocks)));
            }
        }

        constexpr int block = 256;
        const sycl::range<2> local(1, block);
        const sycl::range<2> global(nt * n_stream, ((n_blocks + block - 1) / block) * block);

        const float * tile_dd = tile.get();
        stream->parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            k_qsa_score_reduce(tile_dd, dst_dd, n_blocks, n_heads, nt, n_tps, t0, item);
        });
    }

    return c.i_out - i;
}

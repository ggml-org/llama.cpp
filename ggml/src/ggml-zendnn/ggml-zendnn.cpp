#include "ggml-zendnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "zendnnl.hpp"

#include <cstring>


struct ggml_backend_zendnn_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    if constexpr (std::is_same_v<T, float>) {
        return zendnnl::common::data_type_t::f32;
    } else if constexpr (std::is_same_v<T, ggml_bf16_t>) {
        return zendnnl::common::data_type_t::bf16;
    } else if constexpr (std::is_same_v<T, block_q8_0>) {
        return zendnnl::common::data_type_t::s8;
    } else {
        return zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
static zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.num_threads = ctx->n_threads;

    if constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = true;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    return params;
}

/**
 * ZenDNN matmul: computes C = B * A.
 *
 * - A: weights, shape (k, m), column-major (each column is a weight vector for one output).
 * - B: input, shape (n, k), row-major (each row is an input sample).
 * - C: output, shape (n, m), row-major.
 *
 * Dimensions:
 *   m = output features (columns of C, columns of A)
 *   n = batch size      (rows of C, rows of B)
 *   k = inner dimension (columns of B, rows of A)
 */
template <typename TA, typename TB, typename TC>
static bool ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               const TA * A, int64_t lda, const TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    if constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', false, true,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        true,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    if (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<int>(status));
        return false;
    }
    return true;
}

static bool ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              const void * A, int64_t lda, const void * B, int64_t ldb, void * C,
                              int64_t ldc, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);

    // categorize types
    switch (Atype) {
        case GGML_TYPE_F32:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                return false;
            return ggml_zendnn_matmul<float, float, float>(
                ctx, m, n, k,
                (const float *)A, lda,
                (const float *)B, ldb,
                (float *)C, ldc);
        case GGML_TYPE_BF16:
            if (Btype != GGML_TYPE_BF16)
                return false;
            if (Ctype == GGML_TYPE_BF16)
                return ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (const ggml_bf16_t *)A, lda,
                    (const ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            if (Ctype == GGML_TYPE_F32)
                return ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, float>(
                    ctx, m, n, k,
                    (const ggml_bf16_t *)A, lda,
                    (const ggml_bf16_t *)B, ldb,
                    (float *)C, ldc);
            return false;
        case GGML_TYPE_Q8_0:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                return false;
            return ggml_zendnn_matmul<block_q8_0, float, float>(
                ctx, m, n, k,
                (const block_q8_0 *)A, lda,
                (const float *)B, ldb,
                (float *)C, ldc);
        default:
            return false; // unsupported type
    }
}

static void ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_type         const vec_dot_type = src0->type;
    ggml_from_float_t const from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    void * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    if (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t desired_wsize = ne13 * nbw3;
        if (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(new char[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel for collapse(3) num_threads(ctx->n_threads) schedule(static)
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    const float * src1_f32 = (float *)((char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    void * src1_conv = (char *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const void* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            const size_t row_size = ggml_row_size(vec_dot_type, ne10);
            if (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<const char *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<const char *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<char *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
template <typename TA, typename TB, typename TC>
static bool ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     const std::vector<int64_t> & n,
                                     const std::vector<const void *> & A, int64_t lda,
                                     const std::vector<const void *> & B, int64_t ldb,
                                     const std::vector<void *> & C, int64_t ldc) {

    const int n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<char>         layout(n_experts, 'r');
    std::vector<bool>         trans_a(n_experts, false);
    std::vector<bool>         trans_b(n_experts, true);
    std::vector<int>          batch_m(n_experts);
    std::vector<int>          batch_n(n_experts, m);
    std::vector<int>          batch_k(n_experts, k);
    std::vector<float>        alpha(n_experts, 1.0f);
    std::vector<float>        beta(n_experts, 0.0f);
    std::vector<const void *> bias(n_experts, nullptr);
    std::vector<int>          lda_v(n_experts, lda);
    std::vector<int>          ldb_v(n_experts, ldb);
    std::vector<int>          ldc_v(n_experts, ldc);
    std::vector<bool>         is_wei_const(n_experts, true);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    for (int i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        if constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    if (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<int>(status));
        return false;
    }
    return true;
}

static bool ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    const std::vector<int64_t> & n,
                                    const std::vector<const void *> & A, int64_t lda,
                                    const std::vector<const void *> & B, int64_t ldb,
                                    const std::vector<void *> & C, int64_t ldc,
                                    int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    for (size_t i = 0; i < n.size(); i++) {
        assert(n[i] >= 0);
    }
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);

    // categorize types
    switch (Atype) {
        case GGML_TYPE_F32:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                return false;
            return ggml_zendnn_group_matmul<float, float, float>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        case GGML_TYPE_BF16:
            if (Btype != GGML_TYPE_BF16)
                return false;
            if (Ctype == GGML_TYPE_BF16)
                return ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            if (Ctype == GGML_TYPE_F32)
                return ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, float>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            return false;
        case GGML_TYPE_Q8_0:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                return false;
            return ggml_zendnn_group_matmul<block_q8_0, float, float>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        default:
            return false; // unsupported type
    }
}

static void ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // expert weights
    const ggml_tensor * src1 = dst->src[1];  // inputs
    const ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_BINARY_OP_LOCALS

    // exit for no tokens to process
    if (ne2 == 0 || ne11 == 0) {
        return;
    }

    ggml_type         const vec_dot_type = src0->type;
    ggml_from_float_t const from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    int     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *)((const char *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            if (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    if (total_rows == 0) {
        return; // no rows to process
    }

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    const size_t f32_row_size = (size_t)ne10 * sizeof(float);
    const size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    if (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    const size_t wdata_cur_size = total_rows * gather_row_size;
    const size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    const size_t total_size = wdata_cur_size + dst_cur_size;
    if (ctx->work_size < total_size) {
        ctx->work_data.reset(new char[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    char * wdata_cur = ctx->work_data.get();
    char * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_count(n_active_experts);
    std::vector<const void *> batch_src(n_active_experts);
    std::vector<const void *> batch_wei(n_active_experts);
    std::vector<void *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<int>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        int     batch_idx = 0;
        for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            if (matrix_row_counts[cur_a] == 0) {
                continue;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += matrix_row_counts[cur_a] * gather_row_size;
            d_off += matrix_row_counts[cur_a] * ggml_row_size(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const int64_t w_off     = expert_wdata_off[cur_a];
        const int64_t d_off     = expert_dst_off[cur_a];
        const int     batch_idx = expert_batch_idx[cur_a];

        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            const mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            const int64_t id  = row_mapping.i1;
            const int64_t i11 = id % ne11;
            const int64_t i12 = row_mapping.i2;

            const char * src_row = (const char *) src1->data + i11*nb11 + i12*nb12;
            void *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            if (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((const float *) src_row, dst_row, ne10);
            } else {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_count[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (const char *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    if (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_count, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const int64_t d_off = expert_dst_off[cur_a];

        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            const mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            const int64_t id = row_mapping.i1;
            const int64_t i1 = id;
            const int64_t i2 = row_mapping.i2;

            std::memcpy(
                (char *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

static const char * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    return "ZenDNN";

    GGML_UNUSED(backend);
}

static void ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, node);
                break;
            case GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_zendnn_guid(void) {
    static const char * guid_str = "AMD-ZENDNN-ACCEL";
    return reinterpret_cast<ggml_guid_t>(const_cast<char*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(void) {
    ggml_backend_zendnn_context * ctx = new ggml_backend_zendnn_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_zendnn(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

void ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, int n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
static const char * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    return "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
static const char * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    return "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

static void ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false
    };
}

static ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    if (backend == NULL) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        return NULL;
    }

    return backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_zendnn_adaptive_fallback_enabled() {
    static const bool enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    return enabled;
}

static bool ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        {
            const ggml_tensor * weights = op->src[0];
            const ggml_tensor * inputs = op->src[1];

            const int64_t ne10 = inputs->ne[0];
            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];
            const int64_t min_batch = 1;

            if(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                return false;
            }

            if (ggml_zendnn_adaptive_fallback_enabled()) {
                const int64_t K = inputs->ne[0];
                const int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                const int64_t M = weights->ne[1];
                if(K <= 256 || N <= 128 || M <= 96) {
                    return false;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    const int64_t n_experts = weights->ne[2];
                    const int64_t max_experts = 32;
                    if (n_experts > max_experts) {
                        return false;
                    }
                }
            }
            else if (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                return false;
            }

            switch (weights->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_BF16:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ NULL,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ NULL,
    /* .event_new              = */ NULL,
    /* .event_free             = */ NULL,
    /* .event_synchronize      = */ NULL,
};

// backend reg interface
static const char * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    return "ZenDNN";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zendnn_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_zendnn_device;
}

static void * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_zendnn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(void) {
    static struct ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)

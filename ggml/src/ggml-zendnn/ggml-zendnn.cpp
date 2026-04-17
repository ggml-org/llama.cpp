#include "ggml-zendnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
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
    } else {
        return zendnnl::common::data_type_t::none;
    }
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

    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.num_threads = ctx->n_threads;

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
        {},                 // batch_params
        params              // params
    );

    if (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<int>(status));
        return false;
    }
    return true;
}

static bool ggml_zendnn_sgemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
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
    if (src1->type != vec_dot_type) {
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
            const void* wdata = src1->type == vec_dot_type ? src1->data : work_data;
            const size_t row_size = ggml_row_size(vec_dot_type, ne10);
            if (!ggml_zendnn_sgemm(ctx,
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
                                  vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN sgemm failed\n", __func__);
        }
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // [hidden_K, out_N, n_experts]
    const ggml_tensor * src1 = dst->src[1];  // [hidden_K, n_ids, n_tokens]
    const ggml_tensor * ids  = dst->src[2];  // [n_ids, n_tokens]

    const int n_as     = (int)src0->ne[2];
    const int n_ids    = (int)ids->ne[0];
    const int n_tokens = (int)ids->ne[1];

    const int hidden_K = (int)src0->ne[0];
    const int out_N    = (int)src0->ne[1];

    const size_t src0_elem_bytes = src0->nb[0];
    const size_t src1_elem_bytes = src1->nb[0];
    const size_t dst_elem_bytes  = dst->nb[0];

    const size_t src1_slot_stride  = src1->nb[1];
    const size_t src1_token_stride = src1->nb[2];

    const size_t dst_slot_stride   = dst->nb[1];
    const size_t dst_token_stride  = dst->nb[2];

    using dt = zendnnl::lowoha::matmul::data_type_t;
    auto to_dt = [](ggml_type t)->dt {
        switch(t) {
            case GGML_TYPE_F32:  return dt::f32;
            case GGML_TYPE_F16:  return dt::f16;
            case GGML_TYPE_BF16: return dt::bf16;
            default: GGML_ABORT("unsupported dtype");
        }
    };

    const dt src_dt = to_dt(src1->type);
    const dt wei_dt = to_dt(src0->type);
    const dt dst_dt = to_dt(dst->type);

    std::vector<int> M_all(n_as, 0);
    std::vector<std::vector<std::pair<int,int>>> assignments(n_as);

    for (int tok = 0; tok < n_tokens; tok++) {
        for (int slot = 0; slot < n_ids; slot++) {

            int expert =
                *(int32_t *)((char*)ids->data +
                slot * ids->nb[0] +
                tok  * ids->nb[1]);

            GGML_ASSERT(expert >= 0 && expert < n_as);

            M_all[expert]++;
            assignments[expert].emplace_back(tok, slot);
        }
    }

    

    const size_t packed_src_row_bytes = hidden_K * src1_elem_bytes;
    const size_t packed_dst_row_bytes = out_N    * dst_elem_bytes;
    const size_t packed_wei_bytes     = hidden_K * out_N * src0_elem_bytes;

    std::vector<char> layout;
    std::vector<bool> transA, transB, wconst;
    std::vector<int>  M_vec, N_vec, K_vec;
    std::vector<float> alpha, beta;
    std::vector<const void*> src_vec, wei_vec, bias_vec;
    std::vector<void*> dst_vec;
    std::vector<int> lda, ldb, ldc;

    std::vector<void*> src_allocs;
    std::vector<void*> wei_allocs;
    std::vector<void*> dst_allocs;

    std::vector<int> expert_map(n_as, -1);

    std::vector<zendnnl::lowoha::matmul::matmul_params> params;

    for (int e = 0; e < n_as; e++) {

        if (M_all[e] == 0) continue;

        expert_map[e] = layout.size();

        layout.push_back('r');
        transA.push_back(false);
        transB.push_back(true);
        wconst.push_back(true);

        M_vec.push_back(M_all[e]);
        N_vec.push_back(out_N);
        K_vec.push_back(hidden_K);

        alpha.push_back(1.0f);
        beta.push_back(0.0f);

        lda.push_back(hidden_K);
        ldb.push_back(hidden_K);
        ldc.push_back(out_N);

        bias_vec.push_back(nullptr);

        void * psrc = ::operator new(M_all[e] * packed_src_row_bytes);
        void * pwei = ::operator new(packed_wei_bytes);
        void * pdst = ::operator new(M_all[e] * packed_dst_row_bytes);

        src_allocs.push_back(psrc);
        wei_allocs.push_back(pwei);
        dst_allocs.push_back(pdst);

        src_vec.push_back(psrc);
        wei_vec.push_back(pwei);
        dst_vec.push_back(pdst);

        zendnnl::lowoha::matmul::matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = wei_dt;
        p.dtypes.dst = dst_dt;
        p.dtypes.bias = dt::f32;
        p.mem_format_a = 'n';
        p.mem_format_b = 'n';

        params.push_back(p);

        char * wei_dst = (char*)pwei;
        char * wei_src = (char*)src0->data + e * src0->nb[2];

        for (int n = 0; n < out_N; n++) {
            for (int k = 0; k < hidden_K; k++) {

                char * src_elem =
                    wei_src +
                    k * src0->nb[0] +
                    n * src0->nb[1];

                char * dst_elem =
                    wei_dst +
                    (n * hidden_K + k) * src0_elem_bytes;

                memcpy(dst_elem, src_elem, src0_elem_bytes);
            }
        }
    }

    for (int e = 0; e < n_as; e++) {

        if (M_all[e] == 0) continue;

        int gi = expert_map[e];

        char * dstp = (char*)src_allocs[gi];

        for (int row = 0; row < M_all[e]; row++) {

            int tok  = assignments[e][row].first;
            int slot = assignments[e][row].second;

            char * src_row =
                (char*)src1->data +
                slot * src1_slot_stride +
                tok  * src1_token_stride;

            char * dst_row =
                dstp + row * packed_src_row_bytes;

            for (int k = 0; k < hidden_K; k++) {

                memcpy(
                    dst_row + k * src1_elem_bytes,
                    src_row + k * src1->nb[0],
                    src1_elem_bytes);
            }
        }
    }

    if (src_vec.empty()) return;

    status_t result =
        zendnnl::lowoha::matmul::group_gemm_direct(
            layout, transA, transB,
            M_vec, N_vec, K_vec,
            alpha,
            src_vec, lda,
            wei_vec, ldb,
            bias_vec,
            beta,
            dst_vec, ldc,
            wconst,
            params);

    GGML_ASSERT(result == status_t::success);

    char * dst_base = (char*)dst->data;

    for (int e = 0; e < n_as; e++) {

        if (M_all[e] == 0) continue;

        int gi = expert_map[e];
        char * srcp = (char*)dst_vec[gi];

        for (int row = 0; row < M_all[e]; row++) {

            int tok  = assignments[e][row].first;
            int slot = assignments[e][row].second;

            char * src_row =
                srcp + row * packed_dst_row_bytes;

            char * dst_row =
                dst_base +
                slot * dst_slot_stride +
                tok  * dst_token_stride;

            for (int n = 0; n < out_N; n++) {

                memcpy(
                    dst_row + n * dst->nb[0],
                    src_row + n * dst_elem_bytes,
                    dst_elem_bytes);
            }
        }
    }

    for (void * p : src_allocs) operator delete(p);
    for (void * p : wei_allocs) operator delete(p);
    for (void * p : dst_allocs) operator delete(p);
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
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
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
            if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                    return false;
            }
            // MUL_MAT_ID performs best with a moderate number of experts due to its
            // gather + batched matmul + scatter approach. Future versions will leverage
            // ZenDNN's grouped_gemm for better scalability with larger expert counts:
            // https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_gemm_operator.md
            if (op->op == GGML_OP_MUL_MAT_ID) {
                const int64_t n_experts = weights->ne[2];
                const int64_t max_experts = 32;
                if (n_experts > max_experts) {
                    return false;
                }
            }
            switch (weights->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_BF16:
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

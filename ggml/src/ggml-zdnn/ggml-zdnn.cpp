#include "zdnn.h"
#include "ggml-zdnn.h"
#include "ggml-zdnn-impl.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <memory>
#include <csignal>
#include <unistd.h>

inline zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return FP32;
        case GGML_TYPE_F16:
            return FP16;
        case GGML_TYPE_BF16:
            return BFLOAT;
        case GGML_TYPE_I8:
            return INT8;
        case GGML_TYPE_I32:
            return INT32;
        case GGML_TYPE_Q8_0:
            return INT8;
        default:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            break;
    }
}

inline void ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                                    zdnn_tensor_desc  & tfm_desc,
                                    zdnn_ztensor      & ztensor,
                              const ggml_tensor       * src,
                              const int64_t           * ne,
                              const zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

inline void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor,
                                          void * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
}

inline void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &buffer->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } break;

        default:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &buffer->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } break;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));
}

static void ggml_zdnn_mul_mat_op(ggml_backend_zdnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const ggml_tensor * weights = src0;
    const ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * inputs_extra  = (ggml_backend_zdnn_buffer *)inputs->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;

    zdnn_tensor_desc ptd_bias, td_bias;
    zdnn_ztensor zt_bias;

    const int64_t weights_rows = ne01;
    const int64_t weights_cols = ne00;
    const int64_t inputs_rows  = ne11;
    const int64_t inputs_cols  = ne10;

    assert(inputs_cols == weights_cols);

    const int64_t output_rows = ne1;
    const int64_t output_cols = ne0;

    const int64_t bias_dim  [GGML_MAX_DIMS]  = { 1, 1, 1, output_cols };
    ggml_zdnn_create_tensor(ptd_bias, td_bias, zt_bias, output, bias_dim, ZDNN_1D);

    void * bias_data = (void *)calloc(ne0, ggml_element_size(output));
    if (weights_extra->ztensor.is_transformed == false) ggml_zdnn_load_tensor(weights_extra->ztensor, weights->data);
    if (inputs_extra->ztensor.is_transformed == false) ggml_zdnn_load_tensor(inputs_extra->ztensor, inputs->data);
    ggml_zdnn_load_tensor(zt_bias, bias_data);

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, weights_extra->name,
    //               weights->ne[3], weights->ne[2], weights->ne[1], weights->ne[0],
    //               weights_extra->pre_tfm_desc.dim1,
    //               weights_extra->pre_tfm_desc.dim2,
    //               weights_extra->pre_tfm_desc.dim3,
    //               weights_extra->pre_tfm_desc.dim4);

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, inputs_extra->name,
    //               inputs->ne[3], inputs->ne[2], inputs->ne[1], inputs->ne[0],
    //               inputs_extra->pre_tfm_desc.dim1,
    //               inputs_extra->pre_tfm_desc.dim2,
    //               inputs_extra->pre_tfm_desc.dim3,
    //               inputs_extra->pre_tfm_desc.dim4);

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[1]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim1  == inputs->ne[0]  && "inputs_extra->pre_tfm_desc.dim1 must match inputs->ne[0]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim2  == inputs->ne[1]  && "inputs_extra->pre_tfm_desc.dim2 must match inputs->ne[1]");

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &zt_bias,
                                        false, true, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->data));

    ZDNN_CHECK(zdnn_free_ztensor_buffer(&zt_bias));
    free(bias_data);
}

static void ggml_zdnn_mul_mat_dispatch(ggml_backend_zdnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    bool use_mul_mat_vec =
        (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % 2 == 0 && src1->ne[1] == 1;

    bool use_mul_mat_vec_q =
        ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool use_mul_mat_q =
        ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // debug helpers
    // GGML_LOG_INFO("%s: use_mul_mat_vec   = %d\n", __func__, use_mul_mat_vec);
    // GGML_LOG_INFO("%s: use_mul_mat_vec_q = %d\n", __func__, use_mul_mat_vec_q);
    // GGML_LOG_INFO("%s: use_mul_mat_q     = %d\n", __func__, use_mul_mat_q);
    // GGML_LOG_INFO("%s: src0: %8d %8d %8d %8d\n", __func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    // GGML_LOG_INFO("%s:       %8d %8d %8d %8d\n", __func__, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    // GGML_LOG_INFO("%s: src1: %8d %8d %8d %8d\n", __func__, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    // GGML_LOG_INFO("%s:       %8d %8d %8d %8d\n", __func__, src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    // GGML_LOG_INFO("%s: src0 is contiguous %d, transposed %d, type = %s, name = %s\n", __func__, ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    // GGML_LOG_INFO("%s: src1 is contiguous %d, transposed %d, type = %s, name = %s\n", __func__, ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1)
        && src1->ne[2] * src1->ne[3] > 1) {
        // general KQ + KQV multi-batch
        GGML_LOG_INFO("%s: using zdnn_mul_mat_batched for KQ + KQV multi-batch\n", __func__);
        // ggml_zdnn_mul_mat_batched(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_vec for vector multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_vec_q for quantized vector multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_vec_q, ggml_zdnn_quantize_row_q8_1);
    } else if (use_mul_mat_q) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_q for quantized matrix multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_q, ggml_zdnn_quantize_mmq_q8_1);
    } else {
        // GGML_LOG_INFO("%s: using zdnn_op_mul_mat for general matrix multiplication\n", __func__);
        ggml_zdnn_mul_mat_op(ctx, src0, src1, dst);
    }
}

static bool ggml_zdnn_compute_forward(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            ggml_zdnn_mul_mat_dispatch(ctx, dst->src[0], dst->src[1], dst);
            break;

        default:
            return false;
    }

    return true;
}

static enum ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_is_empty(node)
            || node->op == GGML_OP_NONE
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        bool ok = ggml_zdnn_compute_forward(ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, node->name, ggml_op_name(node->op));
        }

        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_zdnn_supports_op(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;

        case GGML_OP_MUL_MAT:
            {
                const ggml_tensor * src0 = op->src[0];
                const ggml_tensor * src1 = op->src[1];

                const int64_t ne10 = src1->ne[0];
                const int64_t ne0 = op->ne[0];
                const int64_t ne1 = op->ne[1];

                const int64_t max_batch = ctx_dev->max_size;

                return ggml_is_matrix(src0) &&
                       ggml_is_matrix(src1) &&
                       ggml_is_contiguous(src0) &&
                       ggml_is_contiguous(src1) &&
                       src0->view_src == nullptr && src1->view_src == nullptr &&
                       src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                       (ne0 <= max_batch && ne1 <= max_batch && ne10 <= max_batch);
            } break;

        default:
            return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
static ggml_backend_reg    g_ggml_backend_zdnn_reg;
static ggml_backend_device g_ggml_backend_zdnn_device;

static ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ false,
    /* .has_parmblkformat_1   = */ false,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

static int ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);

    if (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    if (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, sizeof(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_count++;
    return ctx->zdnn_device;
}

static void ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);
    assert(ctx->zdnn_device_ref_count > 0);

    ctx->zdnn_device_ref_count--;
    if (ctx->zdnn_device_ref_count == 0) {
        if (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

static ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = new ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    int device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->gf = nullptr;

    return ctx;
}

static void ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);
    delete ctx;
}

//
// backend interface
//

static void ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    for (int i = 0; i < ctx->n_buffers; i++) {
        if (ctx->buffers[i]->ztensor.buffer != NULL && ctx->buffers[i]->ztensor.is_transformed) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&ctx->buffers[i]->ztensor));
        }
    }

    delete ctx;
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    return ctx->all_data;
}

static enum ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    const int64_t tsize = ggml_nbytes(tensor);
    int buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    strncpy(zdnn_buffer->name, tensor->name, GGML_MAX_NAME - 1);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

static ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ NULL,
};

//
// default buffer type
//

static const char * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = new ggml_backend_zdnn_buffer_context();

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    int device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = true;
    ctx->n_buffers = 1;

    if (ctx->all_data != NULL) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    if (size_aligned > 0 && (ctx->all_data == NULL)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

static size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 256;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name       = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,  // defaults to ggml_nbytes
            /* .is_host        = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_zdnn;
}

static const char * ggml_backend_zdnn_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_NAME "_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_from_ptr_type(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_from_ptr_type_zdnn = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_zdnn_buffer_from_ptr_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,  // defaults to ggml_nbytes
            /* .is_host        = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_from_ptr_type_zdnn;
}

//
// backend
//

static const char * ggml_backend_zdnn_name(ggml_backend_t backend) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    free(backend);
}

static enum ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ggml_zdnn_graph_compute(backend, cgraph);
}

static ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name           = */ ggml_backend_zdnn_name,
    /* .free               = */ ggml_backend_zdnn_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_zdnn_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
};

static ggml_guid_t ggml_backend_zdnn_guid(void) {
    static const char * guid_str = "IBM-ZDNN-ACCELER";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

// TODO: remove in the future
ggml_backend_t ggml_backend_zdnn_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_zdnn_reg(), 0);

    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(ggml_backend));
    *backend = (ggml_backend) {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());

    GGML_UNUSED(backend);
}

//
// backend device
//

static const char * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    return "IBM Z Neural Network Processing Assist (NNPA)";
}

static void ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = (ggml_backend_dev_caps) {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(sizeof(ggml_backend));
    *backend = (ggml_backend) {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zdnn_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_backend_zdnn_buffer_context * ctx = new ggml_backend_zdnn_buffer_context();

    ctx->all_data  = ptr;
    ctx->all_size  = size;
    ctx->owned     = false;
    ctx->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *)((char *)ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    int device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = ptr;
    zdnn_buffer->size = size;
    ctx->buffers.push_back(std::move(zdnn_buffer));

    GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB\n",
                  __func__, size_aligned / 1024.0 / 1024.0);

    ++ctx->n_buffers;

    return ggml_backend_buffer_init(ggml_backend_zdnn_buffer_from_ptr_type(), ggml_backend_zdnn_buffer_i, ctx, size);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    return ggml_zdnn_supports_op(ctx_dev, op);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name ||
        buft->iface.get_name == ggml_backend_zdnn_buffer_from_ptr_type_get_name;

    GGML_UNUSED(dev);
}

static ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_zdnn_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend registry
//

static const char * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zdnn_reg_device_count(ggml_backend_reg_t reg) {
    if (!zdnn_is_nnpa_installed()) {
        return 0;
    }
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    return &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *) ggml_backend_zdnn_get_features;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address,
};

static void ggml_zdnn_cleanup(void) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = (ggml_backend_reg) {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ NULL,
        };

        g_ggml_backend_zdnn_device = (ggml_backend_device) {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main,
        };

        return &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)

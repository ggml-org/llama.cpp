#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-blas.h"

#include "ggml-blas/common.hpp"
#include "ggml-blas/mmf.hpp"
#include "ggml-blas/out-prod.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif


// BLAS backend - graph compute

static void ggml_blas_compute_forward_mul_mat(
        const ggml_backend_blas_context * ctx,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // inputs

    ggml_blas_mul_mat_f(ctx, src0, src1, dst);
}

static void ggml_blas_compute_forward_out_prod(
        const ggml_backend_blas_context * ctx,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // inputs
    const ggml_tensor * src1 = dst->src[1];  // weights

    ggml_blas_out_prod_f(ctx, src0, src1, dst);
}

// BLAS backend - buffer

static void ggml_backend_blas_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ggml_backend_blas_buffer_context * ctx = (ggml_backend_blas_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_blas_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ggml_backend_blas_buffer_context * ctx = (ggml_backend_blas_buffer_context *)buffer->context;
    uintptr_t data = (uintptr_t)ctx->data;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static enum ggml_status ggml_backend_blas_buffer_init_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor) {

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_backend_blas_buffer_context * ctx = (ggml_backend_blas_buffer_context *)buffer->context;
        ggml_backend_blas_buffer * extra = new ggml_backend_blas_buffer;

        extra->data = ggml_aligned_malloc(ggml_nelements(tensor) * sizeof(float)); // sizeof(float) because dequantized
        extra->size = ggml_nelements(tensor) * sizeof(float);

        tensor->extra = extra;
        ctx->buffers.push_back(extra);
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_blas_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor         * tensor,
        uint8_t               value,
        size_t                offset,
        size_t                size) {

    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_blas_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor         * tensor,
        const void          * data,
        size_t                offset,
        size_t                size) {

    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    ggml_backend_blas_buffer_type_context * buft_ctx = (ggml_backend_blas_buffer_type_context *)buffer->buft->context;
    ggml_backend_blas_buffer * extra = (ggml_backend_blas_buffer *)tensor->extra;

    const int64_t ne00 = tensor->ne[0];
    const int64_t ne01 = tensor->ne[1];
    const int64_t ne02 = tensor->ne[2];
    const int64_t ne03 = tensor->ne[3];

    const int64_t nb00 = tensor->nb[0];
    const int64_t nb01 = tensor->nb[1];
    const int64_t nb02 = tensor->nb[2];
    const int64_t nb03 = tensor->nb[3];

    const int64_t ne_plane = ne01*ne00;

    if (tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS
        && tensor->type != GGML_TYPE_F32
        && ggml_get_type_traits(tensor->type)->to_float != NULL) {

        const auto * type_traits = ggml_get_type_traits(tensor->type);
        ggml_to_float_t const to_float = type_traits->to_float;
        GGML_ASSERT(to_float != nullptr);

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void  *       x      = (char *)tensor->data + i02*nb02     + i03*nb03;
                      float * const wplane = (float *)extra->data + i02*ne_plane + i03*ne02*ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread / ne00), 1);
                const int n_threads = std::max(std::min(buft_ctx->n_threads, (int)(ne01 / min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel for num_threads(n_threads)
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((const char *)x + i01*nb01, wplane + i01*ne00, ne00);
                }
#else
                for (int i = 1; i < n_threads; i++) {
                    const int64_t start = (i + 0) * ne01/n_threads;
                    const int64_t end   = (i + 1) * ne01/n_threads;
                    if (start < end) {
                        buft_ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *)x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end   = ne01/n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *)x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        for (auto & task : buft_ctx->tasks) {
            task.get();
        }
        buft_ctx->tasks.clear();
#endif
    }

    GGML_UNUSED(nb00);
}

static void ggml_backend_blas_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor   * tensor,
        void                * data,
        size_t                offset,
        size_t                size) {

    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_blas_buffer_clear(
        ggml_backend_buffer_t buffer,
        uint8_t value) {

    GGML_ASSERT(buffer);

    ggml_backend_blas_buffer_context * ctx = (ggml_backend_blas_buffer_context *)buffer->context;
    memset(ctx->data, value, ctx->size);
}

static void ggml_backend_blas_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ggml_backend_blas_buffer_context * ctx = (ggml_backend_blas_buffer_context *)buffer->context;
    for (auto * extra : ctx->buffers) {
        ggml_aligned_free(extra->data, extra->size);
        delete extra;
    }
    ctx->buffers.clear();
}

static const ggml_backend_buffer_i ggml_backend_blas_buffer_i = {
    /* .free_buffer     = */ ggml_backend_blas_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_blas_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_blas_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_blas_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_blas_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_blas_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_blas_buffer_clear,
    /* .reset           = */ ggml_backend_blas_buffer_reset,
};

// BLAS backend buffer type

static const char * ggml_backend_blas_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_BLAS_NAME;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_blas_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {

    void * data = ggml_aligned_malloc(size);
    if (data == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    ggml_backend_blas_buffer_context * ctx = new ggml_backend_blas_buffer_context;
    ctx->data = data;
    ctx->size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_blas_buffer_i, ctx, size);
}

static size_t ggml_backend_blas_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {

    return TENSOR_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool ggml_backend_blas_buffer_type_is_host(ggml_backend_buffer_type_t buft) {

    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_blas_buffer_type(void) {
    static ggml_backend_blas_buffer_type_context buft_ctx = {
        /* .n_threads = */ (int)std::thread::hardware_concurrency(),
#ifndef GGML_USE_OPENMP
        /* .tasks     = */ std::vector<std::future<void>>(),
#endif
    };

    static ggml_backend_buffer_type ggml_backend_blas_buffer_type = {
        /* .iface = */ {
            /* .get_name             = */ ggml_backend_blas_buffer_type_get_name,
            /* .alloc_buffer         = */ ggml_backend_blas_buffer_type_alloc_buffer,
            /* .get_alignment        = */ ggml_backend_blas_buffer_type_get_alignment,
            /* .get_max_size         = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size       = */ NULL, // defaults to ggml_nbytes
            /* .is_host              = */ ggml_backend_blas_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ &buft_ctx,
    };

    return &ggml_backend_blas_buffer_type;
}

static const char * ggml_backend_blas_get_name(ggml_backend_t backend) {
    return GGML_BLAS_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_blas_graph_compute(
        ggml_backend_t backend,
        ggml_cgraph * cgraph) {

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_op_is_empty(node->op)) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                {
                    ggml_blas_compute_forward_mul_mat(ctx, node);
                } break;
            case GGML_OP_OUT_PROD:
                {
                    ggml_blas_compute_forward_out_prod(ctx, node);
                } break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_blas_i = {
    /* .get_name             = */ ggml_backend_blas_get_name,
    /* .free                 = */ ggml_backend_blas_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_blas_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_blas_init(void) {
    ggml_backend_blas_context * ctx = new ggml_backend_blas_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads = GGML_DEFAULT_N_THREADS;

    ggml_backend_t blas_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ ggml_backend_blas_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#if defined(OPENBLAS_VERSION) && defined(GGML_USE_OPENMP)
    if (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#if defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    if (blas_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return blas_backend;
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

void ggml_backend_blas_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    ctx->n_threads = n_threads;

#if defined(OPENBLAS_VERSION)
    openblas_set_num_threads(ctx->n_threads);
#endif

#if defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_num_threads(ctx->n_threads);
#endif

#if defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_num_threads(ctx->n_threads);
#endif
}

static const char * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    return GGML_BLAS_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return GGML_BLAS_NAME;
    #endif

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_blas_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (ggml_op_is_empty(dst->op)) {
        return true;
    }

    switch (dst->op) {
        case GGML_OP_MUL_MAT:
        {
            const int64_t ne10 = src1->ne[0];
            const int64_t ne0  = dst->ne[0];
            const int64_t ne1  = dst->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 1024;

            return ggml_is_contiguous(src0)
                   && ggml_is_contiguous(src1)
                   && src1->type == GGML_TYPE_F32
                   // NOTE: llama-bench creates views that somehow does not go through init_tensor
                   //       this prevents the uninitialized views from being used in BLAS
                   && src0->view_src == nullptr && src1->view_src == nullptr
                   && (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch)
                   && (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        case GGML_OP_OUT_PROD:
        {
            return src0->type == GGML_TYPE_F32
                   && src1->type == GGML_TYPE_F32
                   && ggml_is_matrix(src0)
                   && ggml_is_matrix(src1)
                   && ggml_is_contiguous(src0)
                   && (ggml_is_contiguous(src1) || ggml_is_transposed(src1));
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_blas_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// BLAS backend - backend (reg)

static const char * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_BLAS_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_blas_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_blas_device_context ctx;
    static ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_blas_device;
}

static void * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_blas_set_n_threads;
    }

    return nullptr;

    GGML_UNUSED(reg);
}

static const ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_count,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(void) {
    static ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BLAS_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)

#include "ggml-et.h"
#include "ggml-et-common.h"
#include "ggml-et-kernels.h"
#include "ggml-et-ops.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <cstring>

/*
  ET Driver.

  `ggml_et_driver()` handles both the device layer and the runtime,
  for doing actual operations on devices.
*/

static struct ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer> device_layer;
    std::shared_ptr<rt::IRuntime> runtime;
} _drv;

static bool ggml_et_driver_init() {
    if (_drv.runtime != nullptr) {
	assert(_drv.device_layer != nullptr);
    } else {
	try {
	    _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
	    _drv.runtime = rt::IRuntime::create(_drv.device_layer);
	    GGML_LOG_INFO("ET: FOUND %d devices!\n", _drv.device_layer->getDevicesCount());
	} catch (const std::exception& e) {
	    GGML_LOG_ERROR("ggml_et: %s", e.what());
	    if (_drv.device_layer != nullptr)
		_drv.device_layer.reset();
	    if (_drv.runtime != nullptr)
		_drv.runtime.reset();
	    return false;
	}
    }
    return true;
}

static std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    return _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    return _drv.runtime;
}

static ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

static void ggml_backend_et_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    if (ctx->data != nullptr) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        if (runtime) {
            GGML_LOG_DEBUG("ET: Freeing %zu bytes on device %d (ptr=%p)\n", ctx->size, ctx->devidx, ctx->data);
            runtime->freeDevice(ctx->rtid, static_cast<std::byte*>(ctx->data));
        }
    }
    delete ctx;
}

static void * ggml_backend_et_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    return ctx->data;
}

static enum ggml_status ggml_backend_et_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_et_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    // Create short-lived stream for this transfer
    rt::StreamId stream = runtime->createStream(ctx->rtid);

    std::byte * dst_ptr = static_cast<std::byte*>(tensor->data) + offset;
    const std::byte * src_ptr = static_cast<const std::byte*>(data);

    GGML_LOG_DEBUG("ET: Host->Device transfer %zu bytes (offset=%zu, tensor=%p, device=%d)\n", size, offset, (void*)tensor, ctx->devidx);
    rt::EventId event = runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

static void ggml_backend_et_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    rt::StreamId stream = runtime->createStream(ctx->rtid);

    const std::byte * src_ptr = static_cast<const std::byte*>(tensor->data) + offset;
    std::byte * dst_ptr = static_cast<std::byte*>(data);

    GGML_LOG_DEBUG("ET: Device->Host transfer %zu bytes (offset=%zu, tensor=%p, device=%d)\n", size, offset, static_cast<const void*>(tensor), ctx->devidx);
    rt::EventId event = runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

static bool ggml_backend_et_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_et_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const struct ggml_backend_buffer_i ggml_backend_et_buffer_i = {
    /* .free_buffer     = */ ggml_backend_et_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_et_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_et_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_et_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_et_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_et_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_et_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_ET_NAME;
}

static ggml_backend_buffer_t ggml_backend_et_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_et_buffer_type_context * btctx = (ggml_backend_et_buffer_type_context *)buft->context;

    ggml_backend_et_buffer_context * ctx = new ggml_backend_et_buffer_context;
    ctx->devidx = btctx->devidx;
    ctx->size = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        delete ctx;
        return nullptr;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    if (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        delete ctx;
        return nullptr;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->data = runtime->mallocDevice(ctx->rtid, size);
    if (ctx->data == nullptr) {
        delete ctx;
        return nullptr;
    }

    GGML_LOG_DEBUG("ET: Allocated %zu bytes on device %d (ptr=%p)\n", size, btctx->devidx, ctx->data);
    return ggml_backend_buffer_init(buft, ggml_backend_et_buffer_i, ctx, size);
}

static size_t ggml_backend_et_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_MEM_ALIGN;
}

static size_t ggml_backend_et_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_et_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes_pad(tensor);
}

static bool ggml_backend_et_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const struct ggml_backend_buffer_type_i ggml_backend_et_buffer_type_i = {
    /* .get_name         = */ ggml_backend_et_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_et_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_et_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_et_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_et_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_et_buffer_type_is_host,
};

static const char * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_ET_NAME;
}

static void ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *)backend->context;

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    if (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
        ggml_et_unload_all_kernels(dev_ctx);
    }

    delete et_ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_et_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *)backend->context;

    return ggml_backend_et_buffer_type(et_ctx->devidx);
}

static void ggml_backend_et_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_et_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_et_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static enum ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)backend->device->context;

    GGML_LOG_DEBUG("ET: Computing graph with %d nodes\n", cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) {
            continue;
        }

        GGML_LOG_DEBUG("ET: Processing node %d: %s (%s)\n", i, node->name, ggml_op_name(node->op));

        switch (node->op) {
            case GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, node);
                break;

            case GGML_OP_ADD:
                ggml_et_op_add(dev_ctx, node);
                break;

            case GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, node);
                break;

            case GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, node);
                break;

            case GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, node);
                break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are metadata-only operations that require no computation
                GGML_LOG_DEBUG("ET: No-op metadata operation: %s\n", ggml_op_name(node->op));
                break;

            default:
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s\n", ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }

    GGML_LOG_DEBUG("ET: Graph computation completed successfully\n");
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    // Log what operations are being queried for support (device level)
    const char * op_name = ggml_op_name(op->op);
    const char * type_name = ggml_type_name(op->type);

    // Get tensor dimensions and sizes
    char shape_str[256];
    snprintf(shape_str, sizeof(shape_str), "[%lld,%lld,%lld,%lld]",
             (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3]);

    // Get source tensor types, shapes, strides, and contiguity info
    char src_info[2048] = "";
    if (op->src[0]) {
        char src_str[512];
        snprintf(src_str, sizeof(src_str), " src0=%s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu]%s",
                ggml_type_name(op->src[0]->type),
                (long long)op->src[0]->ne[0], (long long)op->src[0]->ne[1],
                (long long)op->src[0]->ne[2], (long long)op->src[0]->ne[3],
                op->src[0]->nb[0], op->src[0]->nb[1], op->src[0]->nb[2], op->src[0]->nb[3],
                ggml_is_contiguous(op->src[0]) ? "C" : "NC");
        strcat(src_info, src_str);
    }
    if (op->src[1]) {
        char src_str[512];
        snprintf(src_str, sizeof(src_str), " src1=%s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu]%s",
                ggml_type_name(op->src[1]->type),
                (long long)op->src[1]->ne[0], (long long)op->src[1]->ne[1],
                (long long)op->src[1]->ne[2], (long long)op->src[1]->ne[3],
                op->src[1]->nb[0], op->src[1]->nb[1], op->src[1]->nb[2], op->src[1]->nb[3],
                ggml_is_contiguous(op->src[1]) ? "C" : "NC");
        strcat(src_info, src_str);
    }

    // Add output tensor contiguity info
    char output_contiguity[32];
    snprintf(output_contiguity, sizeof(output_contiguity), " out_nb[%zu,%zu,%zu,%zu]%s",
            op->nb[0], op->nb[1], op->nb[2], op->nb[3],
            ggml_is_contiguous(op) ? "C" : "NC");

    bool supported = false;
    switch (op->op) {
        case GGML_OP_MUL:
        case GGML_OP_ADD:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]) &&
                       ggml_is_contiguous(op->src[1]);
            break;
        case GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32 and F16 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32) {

                // Check first dimension contiguity requirements
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_size(op->src[1]->type));
                bool dst_first_dim_contiguous = (op->nb[0] == sizeof(float));

                // Check destination stride ordering
                bool dst_properly_ordered = (op->nb[0] <= op->nb[1] &&
                                            op->nb[1] <= op->nb[2] &&
                                            op->nb[2] <= op->nb[3]);

                supported = src0_first_dim_contiguous &&
                           src1_first_dim_contiguous &&
                           dst_first_dim_contiguous &&
                           dst_properly_ordered;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE (standard and NEOX modes only)
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0])) {
                // Check ROPE mode - only support standard (0x0) and NEOX (0x2)
                const int mode = ((const int32_t *) op->op_params)[2];
                supported = (mode == 0x0) || (mode & GGML_ROPE_TYPE_NEOX);
            } else {
                supported = false;
            }
            break;
        case GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_GLU:
            // Support F32 GLU operations (split tensor mode only, SWIGLU only for now)
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && // Require split mode
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Check GLU variant - only support SWIGLU for now
                ggml_glu_op glu_type = ggml_get_glu_op(op);
                supported = (glu_type == GGML_GLU_OP_SWIGLU);
            } else {
                supported = false;
            }
            break;
        default:
            supported = false;
            break;
    }

    GGML_LOG_DEBUG("ET: Device query support for %s (type=%s, shape=%s, bytes=%zu%s%s) -> %s\n",
                   op_name, type_name, shape_str, ggml_nbytes(op), src_info, output_contiguity,
                   supported ? "SUPPORTED" : "unsupported");

    return supported;
}

static bool ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_et_buffer_type_get_name;
}

static bool ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    // This should only include ops that are worth offloading
    // (i.e. large tensors)
    // We are offloading all ops for testing.
    switch (op->op) {
        case GGML_OP_MUL:
        case GGML_OP_ADD:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op) &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        case GGML_OP_MUL_MAT:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32) {

                // Check first dimension contiguity requirements
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_size(op->src[1]->type));
                bool dst_first_dim_contiguous = (op->nb[0] == sizeof(float));

                // Check destination stride ordering
                bool dst_properly_ordered = (op->nb[0] <= op->nb[1] &&
                                            op->nb[1] <= op->nb[2] &&
                                            op->nb[2] <= op->nb[3]);

                return src0_first_dim_contiguous &&
                       src1_first_dim_contiguous &&
                       dst_first_dim_contiguous &&
                       dst_properly_ordered;
            }
            return false;
        case GGML_OP_ROPE:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0])) {
                // Check ROPE mode - only support standard (0x0) and NEOX (0x2)
                const int mode = ((const int32_t *) op->op_params)[2];
                return (mode == 0x0) || (mode & GGML_ROPE_TYPE_NEOX);
            }
            return false;
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op) &&
                   ggml_is_contiguous(op->src[0]);
            return false;
        case GGML_OP_GLU:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && // Require split mode
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Check GLU variant - only support SWIGLU for now
                ggml_glu_op glu_type = ggml_get_glu_op(op);
                return (glu_type == GGML_GLU_OP_SWIGLU);
            }
            return false;
        default:
            return false;
    }
}

static const struct ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ NULL, // ggml checks for presence of these
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static const char * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_ET_NAME;
}

static const char * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->desc.c_str();
}

static void ggml_backend_et_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *free = dev_ctx->total_mem;
    *total = dev_ctx->total_mem;
}

static enum ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_et_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return ggml_backend_et_init(dev_ctx->devidx);
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->buftype;
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static const struct ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ NULL,
    /* .event_free        = */ NULL,
    /* .event_synchronize = */ NULL,
};


/*
  Backend Registry.
*/

static const char * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_ET_NAME;
}

static size_t ggml_backend_et_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    if (devidx >= ctx->devices.size()) {
        return nullptr;
    }
    return ctx->devices[devidx];
}

static void * ggml_backend_et_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_count,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_address,
};

ggml_backend_reg_t ggml_backend_et_reg(void) {
    static ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
	ggml_backend_et_reg_ctx * ctx = new ggml_backend_et_reg_ctx;

	if (!ggml_et_driver_init())
	    return nullptr;

	ggml_backend_reg_t r = new ggml_backend_reg {
	    /* .api_version = */ GGML_BACKEND_API_VERSION,
	    /* .iface       = */ ggml_backend_et_reg_i,
	    /* .context     = */ nullptr, // Set later
	};

	std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        for (int i = 0; i < ggml_et_devicelayer()->getDevicesCount(); i++) {
	    ggml_backend_dev_t dev = new ggml_backend_device {
		/* .iface   = */ ggml_backend_et_device_i,
		/* .reg     = */ r,
		/* .context = */ nullptr // Set later
	    };

	    rt::DeviceId rtid = rtids[i];
	    rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

	    // Create device context.
	    ggml_backend_et_device_context * dev_ctx = new ggml_backend_et_device_context;
	    dev_ctx->devidx = i;
	    dev_ctx->rtid = rtid;
	    dev_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->desc = "ET device " + std::to_string(i);
	    dev_ctx->total_mem = static_cast<size_t>(prop.memorySize_);
	    // Add buffer type for device to device context.
	    ggml_backend_et_buffer_type_context * bufty_ctx = new ggml_backend_et_buffer_type_context;
	    bufty_ctx->devidx = i;
	    bufty_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->buftype = new ggml_backend_buffer_type {
		/* .iface   = */ ggml_backend_et_buffer_type_i,
		/* .device  = */ dev,
		/* .context = */ bufty_ctx
	    };

	    // Create default stream for ordered execution on this device
	    dev_ctx->default_stream = ggml_et_runtime()->createStream(rtid);
	    GGML_LOG_DEBUG("ET: Created default stream for device %d\n", i);

	    dev->context = dev_ctx;

	    ctx->devices.push_back(dev);
	}

	r->context = ctx;
	return r;
    }();

    return _reg;
}

ggml_guid_t ggml_backend_et_guid(void) {
    static ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4, 0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    return &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    if (!ggml_et_driver_init())
	return nullptr;

    if (devidx >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_et_context * ctx = new ggml_backend_et_context;
    ctx->devidx = (int)devidx;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_et(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

int ggml_backend_et_get_device_count(void) {
    return ggml_backend_et_reg_get_device_count(ggml_backend_et_reg());
}

void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        snprintf(description, description_size, "ET Device %d (invalid)", devidx);
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    snprintf(description, description_size, "%s", dev_ctx->desc.c_str());
}

void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        *free = 0;
        *total = 0;
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_memory(dev, free, total);
}

ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num) {
    if (dev_num >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_num);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->buftype;
}

ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void) {
    static ggml_backend_buffer_type host_buffer_type = {
        /* .iface   = */ ggml_backend_et_buffer_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &host_buffer_type;
}

#include "ggml-openvino.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino/utils.h"
#include "ggml-quants.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/allocator.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/tensor.hpp>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

// =====================================================
// OpenVINO Buffer Implementation using ov::Tensor
// =====================================================
//
// Design: This implementation uses a hybrid approach:
// 1. For weight tensors: Store a pre-built ov::op::v0::Constant in tensor->extra
//    - This avoids the memcpy during graph construction
//    - For quantized weights, the constant is already converted to OpenVINO format
// 2. For KV cache / compute tensors: Store an ov::Tensor in tensor->extra
//    - This can be directly passed to infer_request
//    - Future: can be changed to ov::RemoteTensor for GPU/NPU
//
// This design is similar to:
// - CUDA split buffer: tensor->extra stores device pointers
// - CPU repack buffer: tensor->extra stores tensor_traits with repacked data
// =====================================================

// Buffer context that manages per-tensor allocations (no contiguous buffer for weights)
struct ggml_backend_openvino_buffer_context {
    int device;
    std::string name;

    // For non-weight buffers (KV cache, compute), we still use contiguous allocation
    void * data;
    size_t size;
    bool is_remote;

    // Wrapping of the buffer
    std::shared_ptr<ov::Tensor> ov_buffer;

    // Track all extras for cleanup
    std::map<ggml_tensor *, ggml_openvino_extra_base *> tensor_extras;

    // Used for re-allocation on device for kvcache
    void * data_prev;

    ggml_backend_openvino_buffer_context(int device, size_t size, bool is_remote = false) :
        device(device),
        name(std::string(GGML_OPENVINO_NAME) + std::to_string(device)),
        data(nullptr),
        size(size),
        is_remote(is_remote) {
        if (size == 0) {
            return;
        }

        const auto & device_name = ggml_openvino_get_device_name();

        if (is_remote) {
            GGML_ASSERT(device_name == "GPU");
            auto remote_context = ggml_openvino_get_remote_context();
            auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
            ov::intel_gpu::ocl::USMTensor usm_tensor =
                gpu_context.create_usm_device_tensor(ov::element::u8, ov::Shape{size});
            data = usm_tensor.get();
            ov_buffer = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
        } else {
            data = ggml_aligned_malloc(size);
            ov_buffer = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size}, data);
        }

        if (data == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
            return;
        }

        if (reinterpret_cast<uintptr_t>(data) % TENSOR_ALIGNMENT != 0) {
            GGML_LOG_ERROR("%s: %s buffer is not aligned to %d bytes\n", __func__, device_name.c_str(),
                           TENSOR_ALIGNMENT);
            GGML_ABORT("fatal error");
        }
    }

    ~ggml_backend_openvino_buffer_context() {
        // Clean up all tensor extras
        for (auto & pair : tensor_extras) {
            delete pair.second;
        }
        tensor_extras.clear();
        if (!is_remote && data != nullptr) {
            ggml_aligned_free(data, size);
        }
    }
};

// Buffer type context (per-device)
struct ggml_backend_openvino_buffer_type_context {
    int device;
    std::string name;
};

// Buffer interface functions
static void ggml_backend_openvino_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_openvino_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    return ctx->data;
}

static enum ggml_status ggml_backend_openvino_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    // Put kvcache on device memory for GPU (NPU memory is too small even for kvcache)
    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY && strncmp(tensor->name, "cache_", 6) == 0 && !ctx->is_remote &&
        ggml_openvino_get_device_name() == "GPU" && !getenv("GGML_OPENVINO_STATEFUL_EXECUTION")) {
        GGML_ASSERT(ctx->tensor_extras.empty());
        auto device = ctx->device;
        auto size = ctx->size;
        auto * data_prev = ctx->data;
        delete ctx;
        ctx = new ggml_backend_openvino_buffer_context(device, size, true);
        buffer->context = ctx;
        tensor->data = (char *) ctx->data + ((char *) tensor->data - (char *) data_prev);
    }

    // Views share the extra from view_src
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        if (tensor->view_src->extra != nullptr) {
            tensor->extra = tensor->view_src->extra;
        }
        return GGML_STATUS_SUCCESS;
    }

    ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (tensor->data != nullptr && !ggml_is_quantized(tensor->type)) {
        ggml_openvino_tensor_extra * extra = ggml_openvino_create_tensor_extra(tensor, ctx->is_remote);
        if (extra != nullptr) {
            auto it = ctx->tensor_extras.find(tensor);
            if (it != ctx->tensor_extras.end()) {
                delete it->second;
            }
            ctx->tensor_extras[tensor] = extra;
            tensor->extra = extra;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_openvino_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       ggml_tensor * tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memfill
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, (char *) tensor->data + offset, &pattern, sizeof(pattern), size, 0, nullptr,
                                     nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memset((char *) tensor->data + offset, value, size);
    }
}

static void ggml_backend_openvino_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor * tensor,
                                                    const void * data,
                                                    size_t offset,
                                                    size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    // Check if this is a weight buffer (usage is set BEFORE set_tensor is called)
    bool is_weight_buffer = (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Full tensor set: offset=0, full size, not a view
    bool is_full_tensor_set = (offset == 0 && size == ggml_nbytes(tensor) && tensor->view_src == nullptr);
    // 2D tensor (typical weight shape)
    bool is_2d = (tensor->ne[2] == 1 && tensor->ne[3] == 1);

    // Check if this is a quantized weight tensor that needs extraction/requantization
    ggml_openvino_extracted_layout layout = {};
    if (is_weight_buffer && is_full_tensor_set && is_2d && ggml_is_quantized(tensor->type)) {
        layout = ggml_openvino_get_extracted_layout(tensor);
    }

    if (layout.total_size > 0) {
        // Quantized weight tensor with extraction/requantization
        uint8_t * buf_base = (uint8_t *) tensor->data;

        try {
            std::shared_ptr<ov::Node> constant = process_weight_tensor(tensor, data, buf_base);
            constant->set_friendly_name(tensor->name);

            // Store in tensor->extra
            if (layout.is_requant && layout.requant_type.has_value() &&
                layout.requant_type.value() == ExtraQuantType::F16) {
                // F16 requant case - use weight_extra
                auto * extra = new ggml_openvino_weight_extra(constant);
                ctx->tensor_extras[tensor] = extra;
                tensor->extra = extra;
                GGML_LOG_DEBUG("%s: requantized %s to F16\n", __func__, tensor->name);
            } else {
                // Quantized case - use quantized_weight_extra
                // Create tensors with external memory (already filled by process_weight_tensor)
                ov::element::Type weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
                ov::Shape weight_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
                ov::Shape scale_shape = {static_cast<size_t>(tensor->ne[1]),
                                         static_cast<size_t>(tensor->ne[0] / layout.weights_per_block)};

                ov::Tensor weights(weight_type, weight_shape, buf_base + layout.weights_offset);
                ov::Tensor scales(ov::element::f16, scale_shape, buf_base + layout.scales_offset);
                ov::Tensor biases(ov::element::f16, scale_shape, buf_base + layout.biases_offset);

                auto * extra = new ggml_openvino_quantized_weight_extra(std::move(weights), std::move(scales),
                                                                        std::move(biases), constant);
                ctx->tensor_extras[tensor] = extra;
                tensor->extra = extra;

                if (layout.is_requant) {
                    GGML_LOG_DEBUG("%s: requantized %s to %s (u%d, block_size=%ld)\n", __func__, tensor->name,
                                   layout.requant_type.value() == ExtraQuantType::Q4_0_128 ? "Q4_0_128" : "Q8_0_32",
                                   layout.is_u4 ? 4 : 8, layout.weights_per_block);
                } else {
                    int64_t n_blocks = ggml_nelements(tensor) / layout.weights_per_block;
                    GGML_LOG_DEBUG("%s: extracted quantized constant for %s (u%d, %zu weights, %ld blocks)\n", __func__,
                                   tensor->name, layout.is_u4 ? 4 : 8, layout.weights_size, n_blocks);
                }
            }

        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: failed to process quantized data for %s: %s\n", __func__, tensor->name, e.what());
            // Fall back to storing raw data
            memcpy((char *) tensor->data + offset, data, size);
        }
    } else if (is_weight_buffer && is_full_tensor_set && is_2d &&
               (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16)) {
        // F16/F32/BF16 weight tensor
        try {
            std::shared_ptr<ov::Node> constant = process_weight_tensor(tensor, data, tensor->data);
            constant->set_friendly_name(tensor->name);

            // Store in tensor->extra
            ggml_openvino_weight_extra * extra = new ggml_openvino_weight_extra(constant);
            ctx->tensor_extras[tensor] = extra;
            tensor->extra = extra;

            GGML_LOG_DEBUG("%s: created shared-memory constant for %s\n", __func__, tensor->name);

        } catch (const std::exception & e) {
            GGML_LOG_DEBUG("%s: failed to create shared-memory constant for %s: %s\n", __func__, tensor->name,
                           e.what());
        }
    } else {
        // Non-weight tensor (KV cache, activations, etc.) - copy data
        if (ctx->is_remote) {
            cl_command_queue queue = ggml_openvino_get_cl_queue();
            auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
            if (queue != nullptr && mem_cpy_fn != nullptr) {
                cl_int err =
                    mem_cpy_fn(queue, CL_TRUE, (char *) tensor->data + offset, data, size, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
                }
            } else {
                GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            }
        } else {
            memcpy((char *) tensor->data + offset, data, size);
        }

        ggml_openvino_tensor_extra * extra = ggml_openvino_create_tensor_extra(tensor, ctx->is_remote);
        if (extra == nullptr) {
            GGML_LOG_ERROR("%s: failed to create tensor extra for %s\n", __func__, tensor->name);
            return;
        }

        auto it = ctx->tensor_extras.find(tensor);
        if (it != ctx->tensor_extras.end()) {
            delete it->second;
        }
        ctx->tensor_extras[tensor] = extra;
        tensor->extra = extra;
    }
}

static void ggml_backend_openvino_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * tensor,
                                                    void * data,
                                                    size_t offset,
                                                    size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memcpy (device-to-host)
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue != nullptr && mem_cpy_fn != nullptr) {
            cl_int err =
                mem_cpy_fn(queue, CL_TRUE, data, (const char *) tensor->data + offset, size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
            }
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memcpy(data, (const char *) tensor->data + offset, size);
    }
}

static bool ggml_backend_openvino_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * src,
                                                    ggml_tensor * dst) {
    // GGML_LOG_DEBUG("%s: src tensor name=%s, dst tensor name=%s\n", __func__, src->name, dst->name);
    GGML_ASSERT(src != nullptr && dst != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memcpy
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue == nullptr || mem_cpy_fn == nullptr) {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            return false;
        }
        // Can copy from host to device
        if (ggml_backend_buffer_is_host(src->buffer)) {
            cl_int err = mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (host-to-device) failed with error %d\n", __func__, err);
                return false;
            }
            return true;
        }
        // Can also copy from device to device if both are OpenVINO remote buffers
        if (ggml_backend_buffer_is_openvino(src->buffer)) {
            ggml_backend_openvino_buffer_context * src_ctx =
                (ggml_backend_openvino_buffer_context *) src->buffer->context;
            if (src_ctx->is_remote) {
                cl_int err =
                    mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (device-to-device) failed with error %d\n", __func__,
                                   err);
                    return false;
                }
                return true;
            }
        }
        return false;
    }

    // Host buffer - can copy from any host buffer
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_openvino_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    GGML_ASSERT(ctx->data != nullptr);
    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, ctx->data, &pattern, sizeof(pattern), ctx->size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_WARN("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_WARN("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer clear\n",
                          __func__);
        }
    } else {
        memset(ctx->data, value, ctx->size);
    }
}

static const ggml_backend_buffer_i ggml_backend_openvino_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_openvino_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_openvino_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_openvino_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_openvino_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_openvino_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_openvino_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_openvino_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_openvino_buffer_clear,
    /* .reset           = */ NULL,
};

// Buffer type interface functions
static const char * ggml_backend_openvino_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_openvino_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t size) {
    ggml_backend_openvino_buffer_type_context * buft_ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;

    // Create buffer context with contiguous memory allocation
    ggml_backend_openvino_buffer_context * ctx = new ggml_backend_openvino_buffer_context(buft_ctx->device, size);

    if (ctx->data == nullptr && size > 0) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_openvino_buffer_interface, ctx, size);
}

static size_t ggml_backend_openvino_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static size_t ggml_backend_openvino_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_openvino_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    // For quantized 2D tensors (weights), we need extra space for extracted data
    if (ggml_is_quantized(tensor->type) && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        ggml_openvino_extracted_layout layout = ggml_openvino_get_extracted_layout(tensor);
        if (layout.total_size > 0) {
            GGML_LOG_DEBUG(
                "%s: tensor %s needs %zu bytes (original %zu, extracted: weights=%zu scales=%zu biases=%zu)\n",
                __func__, tensor->name, layout.total_size, ggml_nbytes(tensor), layout.weights_size, layout.scales_size,
                layout.biases_size);
            return layout.total_size;
        }
    }

    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

// Get buffer type for a specific device
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device) {
    GGML_ASSERT(device >= 0 && device < ggml_backend_openvino_get_device_count());

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> buffer_types;
    static std::vector<ggml_backend_openvino_buffer_type_context> buffer_type_contexts;

    if (buffer_types.empty()) {
        int device_count = ggml_backend_openvino_get_device_count();
        buffer_types.resize(device_count);
        buffer_type_contexts.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            buffer_type_contexts[i].device = i;
            buffer_type_contexts[i].name = std::string(GGML_OPENVINO_NAME) + std::to_string(i);

            buffer_types[i] = ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_openvino_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), i),
                /* .context = */ &buffer_type_contexts[i],
            };
        }
    }

    return &buffer_types[device];
}

// =====================================================
// OpenVINO Host Buffer Implementation
// =====================================================

static const char * ggml_backend_openvino_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    static std::string name;
    name = ctx->name + "_HOST";
    return name.c_str();
}

static bool ggml_backend_openvino_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_openvino_host_buffer_type_is_host,
};

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device) {
    GGML_ASSERT(device >= 0 && device < ggml_backend_openvino_get_device_count());

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> buffer_types;
    static std::vector<ggml_backend_openvino_buffer_type_context> buffer_type_contexts;

    if (buffer_types.empty()) {
        int device_count = ggml_backend_openvino_get_device_count();
        buffer_types.resize(device_count);
        buffer_type_contexts.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            buffer_type_contexts[i].device = i;
            buffer_type_contexts[i].name = std::string(GGML_OPENVINO_NAME) + std::to_string(i);

            buffer_types[i] = ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_openvino_host_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), i),
                /* .context = */ &buffer_type_contexts[i],
            };
        }
    }

    return &buffer_types[device];
}

bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_openvino_buffer_free_buffer;
}

bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_buffer_type_get_name;
}

bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_host_buffer_type_get_name;
}

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================

struct ggml_backend_openvino_context {
    int device;               // the device ID currently in use
    std::string name;         // context Name
    std::string description;  // context description

    // OpenVINO core components
    ov::Core core;                             // OpenVINO core interface
    std::shared_ptr<ov::CompiledModel> model;  // compiled Model
    ov::InferRequest infer_request;            // inference Request

    // OpenVINO Multi-stream support
    static const int MAX_STREAMS = 8;       // define the maximum number of flows
    std::vector<ov::InferRequest> streams;  // used to support multi-stream reasoning
    int current_stream;                     // the currently active stream index

    // state Management
    bool is_initialized;  // initialize

    ggml_backend_openvino_context() :
        device(0),
        name("OpenVINO"),
        description("OpenVINO Backend Context"),
        current_stream(0),
        is_initialized(false) {}
};

static void ggml_backend_openvino_free(ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    delete ctx;
    delete backend;
}

static const char * ggml_backend_openvino_get_name(ggml_backend_t backend) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ov_graph_compute(cgraph);
    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_openvino_interface = {
    /* .get_name                = */ ggml_backend_openvino_get_name,
    /* .free                    = */ ggml_backend_openvino_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_openvino_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

int ggml_backend_openvino_get_device_count() {
    return ggml_openvino_info().device_count;
}

static ggml_guid_t ggml_backend_openvino_guid(void) {
    static ggml_guid guid = {0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97,
                             0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d};
    return &guid;
}

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device) {
    if (device < 0 || device >= ggml_backend_openvino_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_openvino_context * ctx = new ggml_backend_openvino_context;
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t openvino_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_openvino_guid(),
        /* .interface = */ ggml_backend_openvino_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), device),
        /* .context   = */ ctx,
    };

    return openvino_backend;
}

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_openvino_guid());
}

struct ggml_backend_openvino_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_openvino_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_openvino_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_openvino_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif  // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_openvino_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_openvino_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_openvino_device_get_name(dev);
    props->description = ggml_backend_openvino_device_get_description(dev);
    props->type = ggml_backend_openvino_device_get_type(dev);
    ggml_backend_openvino_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_openvino_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_host_buffer_type(ctx->device);
}

static bool is_op_unsupported_case(const ggml_tensor * op) {
    switch (op->op) {
    case GGML_OP_SOFT_MAX: {
        if (op->src[2] != nullptr) {
            GGML_LOG_WARN("OpenVINO backend does not support SOFT_MAX with sinks\n");
            return true;
        }
        float scale = 1.0f;
        float max_bias = 0.0f;
        const auto * op_params = op->op_params;
        memcpy(&scale, (const float *) op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) op_params + 1, sizeof(float));
        if (max_bias > 0) {
            GGML_LOG_WARN("OpenVINO backend does not support SOFT_MAX with max_bias > 0\n");
            return true;
        }
        break;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
        if (op->src[4] != nullptr) {
            GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with sinks\n");
            return true;
        }
        float scale = 1.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        const auto * op_params = op->op_params;
        memcpy(&scale, (const float *) op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) op_params + 2, sizeof(float));
        if (max_bias > 0) {
            GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with max_bias > 0\n");
            return true;
        }
        if (logit_softcap != 0) {
            GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with logit_softcap != 0\n");
            return true;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (op->type == GGML_TYPE_BF16) {
            // err msg: [GPU] Could not find a suitable kernel for transpose
            GGML_LOG_WARN("OpenVINO backend does not support PERMUTE with BF16 type\n");
            return true;
        }
        break;
    }
    case GGML_OP_CPY: {
        if (op->src[1] != op) {
            GGML_LOG_WARN("OpenVINO backend only supports CPY that is a cast\n");
            return true;
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16) {
            // Has accuracy issue, try enabling this and see `test-backend-ops -o "MUL_MAT"`
            GGML_LOG_WARN("OpenVINO backend does not support MUL_MAT with two F16 tensors\n");
            return true;
        }
        break;
    }
    case GGML_OP_ROPE: {
        const int32_t * op_params = op->op_params;
        const int n_dims = op_params[1];
        const int mode = op_params[2];
        if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_VISION) {
            GGML_LOG_WARN("OpenVINO backend does not support ROPE with mode %d\n", mode);
            return true;
        }
        if (n_dims != 0.0f && n_dims != op->src[0]->ne[0]) {
            GGML_LOG_WARN("OpenVINO backend does not support ROPE with n_dims %d != src[0]->ne[0] %ld\n", n_dims,
                          op->src[0]->ne[0]);
            return true;
        }
        if (op->type != GGML_TYPE_F32) {
            GGML_LOG_WARN("OpenVINO backend does not support ROPE with type %s\n", ggml_type_name(op->type));
            return true;
        }
        float freq_scale;
        float ext_factor;
        memcpy(&freq_scale, op_params + 6, sizeof(float));
        memcpy(&ext_factor, op_params + 7, sizeof(float));
        if (ext_factor != 0.0f) {
            GGML_LOG_WARN("OpenVINO backend does not support ROPE with ext_factor %f != 0.0f\n", ext_factor);
            return true;
        }
        if (op->src[0]->op == GGML_OP_VIEW) {
            if (op->src[0]->view_src->ne[1] != op->src[0]->ne[2]) {
                GGML_LOG_WARN(
                    "OpenVINO backend does not support ROPE with src[0]->view_src->ne[1] %ld != src[0]->ne[2] "
                    "%ld\n",
                    op->src[0]->view_src->ne[1], op->src[0]->ne[2]);
                return true;
            }
        }
        break;
    }
    default:
        break;
    }
    return false;
}

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

    static std::set<ggml_type> supported_types{GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_I64,
                                               GGML_TYPE_I32,  GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K,
                                               GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K};

    static const std::set<ggml_op> supported_ops{GGML_OP_NONE, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_VIEW,
                                                 GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_TRANSPOSE,
                                                 GGML_OP_GET_ROWS, GGML_OP_ROPE, GGML_OP_RMS_NORM, GGML_OP_SCALE,
                                                 // softmax is not updated due to replaced by flash_attn_ext
                                                 // GGML_OP_SOFT_MAX,
                                                 GGML_OP_SET_ROWS, GGML_OP_FLASH_ATTN_EXT, GGML_OP_CPY};
    static const std::set<ggml_unary_op> supported_unary_ops{
        GGML_UNARY_OP_SILU,
    };
    static const std::set<ggml_glu_op> supported_glu_ops{
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_GEGLU,
    };

    switch (op->op) {
    case GGML_OP_UNARY: {
        auto supported = supported_unary_ops.find(ggml_get_unary_op(op)) != supported_unary_ops.end();
        if (!supported) {
            GGML_LOG_WARN("OpenVINO backend does not support unary op %s\n", ggml_unary_op_name(ggml_get_unary_op(op)));
            return false;
        }
        break;
    }
    case GGML_OP_GLU: {
        auto supported = supported_glu_ops.find(ggml_get_glu_op(op)) != supported_glu_ops.end();
        if (!supported) {
            GGML_LOG_WARN("OpenVINO backend does not support GLU op %s\n", ggml_glu_op_name(ggml_get_glu_op(op)));
            return false;
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();
        if (!supported) {
            GGML_LOG_WARN("OpenVINO backend does not support op %s\n", ggml_op_name(op->op));
            return false;
        }
    }
    }

    if (supported_types.find(op->type) == supported_types.end()) {
        GGML_LOG_WARN("OpenVINO backend does not support tensor type %s\n", ggml_type_name(op->type));
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (supported_types.find(src->type) == supported_types.end()) {
            GGML_LOG_WARN("OpenVINO backend does not support tensor type %s\n", ggml_type_name(src->type));
            return false;
        }
        if (ggml_is_quantized(src->type) && src->ne[2] != 1) {
            GGML_LOG_WARN("OpenVINO backend does not support 3D quantized tensors\n");
            return false;
        }
    }

    if (is_op_unsupported_case(op)) {
        return false;
    }
    return true;
}

static bool ggml_backend_openvino_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_openvino(buft) || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_openvino_device_interface = {
    /* .get_name             = */ ggml_backend_openvino_device_get_name,
    /* .get_description      = */ ggml_backend_openvino_device_get_description,
    /* .get_memory           = */ ggml_backend_openvino_device_get_memory,
    /* .get_type             = */ ggml_backend_openvino_device_get_type,
    /* .get_props            = */ ggml_backend_openvino_device_get_props,
    /* .init_backend         = */ ggml_backend_openvino_device_init,
    /* .get_buffer_type      = */ ggml_backend_openvino_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_openvino_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_openvino_device_supports_op,
    /* .supports_buft        = */ ggml_backend_openvino_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct ggml_backend_openvino_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_openvino_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_openvino_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return ggml_openvino_info().device_count;
}

static ggml_backend_dev_t ggml_backend_openvino_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_openvino_reg_context * ctx = (ggml_backend_openvino_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_openvino_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_openvino_reg_interface = {
    /* .get_name         = */ ggml_backend_openvino_reg_get_name,
    /* .get_device_count = */ ggml_backend_openvino_reg_get_device_count,
    /* .get_device       = */ ggml_backend_openvino_reg_get_device,
    /* .get_proc_address = */ ggml_backend_openvino_get_proc_address,
};

static int get_openvino_device_count() {
    return 1;
}

static ggml_openvino_device_info ggml_openvino_init() {
    // Initialize device config singleton from env var
    ggml_openvino_init_device_config();
    GGML_LOG_INFO("OpenVINO: using device %s\n", ggml_openvino_get_device_name().c_str());

    ggml_openvino_device_info info = {};
    info.device_count = get_openvino_device_count();
    return info;
}

const ggml_openvino_device_info & ggml_openvino_info() {
    static ggml_openvino_device_info info = ggml_openvino_init();
    return info;
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void) {
    static ggml_backend_reg reg;

    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_openvino_reg_context * ctx = new ggml_backend_openvino_reg_context;

            for (int i = 0; i < ggml_openvino_info().device_count; i++) {
                ggml_backend_openvino_device_context * dev_ctx = new ggml_backend_openvino_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_OPENVINO_NAME + std::to_string(i);

                dev_ctx->description = ov::get_openvino_version().description;

                ggml_backend_dev_t dev =
                    new ggml_backend_device{/* .interface = */ ggml_backend_openvino_device_interface,
                                            /* .reg       = */ &reg,
                                            /* .context   = */ dev_ctx};
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{/* .api_version = */ GGML_BACKEND_API_VERSION,
                                   /* .iface       = */ ggml_backend_openvino_reg_interface,
                                   /* .context     = */ ctx};
        }

        initialized = true;
    }

    return &reg;
}

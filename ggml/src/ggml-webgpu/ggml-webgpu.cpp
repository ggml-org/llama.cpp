#include "ggml-webgpu.h"

#include <webgpu/webgpu_cpp.h>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-wgsl-shaders.hpp"

#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

#ifdef GGML_WEBGPU_DEBUG
#define WEBGPU_LOG_DEBUG(msg) std::cout << msg << std::endl
#else
#define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif // GGML_WEBGPU_DEBUG

/* Constants */

#define WEBGPU_MUL_MAT_WG_SIZE 64
#define WEBGPU_MUL_MAT_PARAMS_SIZE (13 * sizeof(uint32_t)) // M, N, K, batch sizes, broadcasts
#define WEBGPU_CPY_PARAMS_SIZE (15 * sizeof(uint32_t)) // strides and offsets
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4 // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to their locations.
static void * const webgpu_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

// Always returns the base offset of a tensor, regardless of views.
static uint64_t webgpu_tensor_offset(const ggml_tensor * tensor) {
    if (tensor->view_src) {
        return (uint8_t *) tensor->view_src->data - (uint8_t *) webgpu_ptr_base;
    }
    return (uint8_t *) tensor->data - (uint8_t *) webgpu_ptr_base;
}

/* Struct definitions */

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Limits limits;
    wgpu::SupportedFeatures features;

    std::mutex mutex;
    bool device_initialized = false;

    // pipelines and parameter buffers
    // TODO: reuse params buffers for different pipelines when possible
    wgpu::ComputePipeline memset_pipeline;
    wgpu::Buffer memset_params_dev_buf;
    wgpu::Buffer memset_params_host_buf;
    wgpu::ComputePipeline mul_mat_pipeline;
    wgpu::Buffer mul_mat_params_dev_buf;
    wgpu::Buffer mul_mat_params_host_buf;
    wgpu::ComputePipeline cpy_pipeline;
    wgpu::Buffer cpy_params_dev_buf;
    wgpu::Buffer cpy_params_host_buf;

    size_t memset_bytes_per_thread;

    // Staging buffer for reading data from the GPU
    wgpu::Buffer get_tensor_staging_buf;
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

struct ggml_backend_webgpu_reg_context {
    webgpu_context webgpu_ctx;

    size_t device_count;
    const char * name;
};

struct ggml_backend_webgpu_device_context {
    webgpu_context webgpu_ctx;

    std::string device_name;
    std::string device_desc;
};

struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;

    std::string name;
};

struct ggml_backend_webgpu_buffer_context {
    webgpu_context webgpu_ctx;

    wgpu::Buffer buffer;

    ggml_backend_webgpu_buffer_context(webgpu_context ctx, wgpu::Buffer buf) :
        webgpu_ctx(ctx), buffer(buf) {
    }
};

/* End struct definitions */

/* WebGPU object initializations */

static void ggml_webgpu_create_pipeline(wgpu::Device &device, wgpu::ComputePipeline &pipeline, const char * shader_code, const char * label, const std::vector<wgpu::ConstantEntry> &constants = {}) {
    WEBGPU_LOG_DEBUG("ggml_webgpu_create_pipeline()");
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;
    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;
    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label = label;
    pipeline_desc.compute.module = shader_module;
    pipeline_desc.compute.entryPoint = "main"; // Entry point in the WGSL code
    pipeline_desc.layout = nullptr; // nullptr means auto layout
    if (constants.size() > 0) {
        pipeline_desc.compute.constants = constants.data();
        pipeline_desc.compute.constantCount = constants.size();
    }
    pipeline = device.CreateComputePipeline(&pipeline_desc);
}

static void ggml_webgpu_create_buffer(wgpu::Device &device, wgpu::Buffer &buffer, size_t size, wgpu::BufferUsage usage, const char* label) {
    WEBGPU_LOG_DEBUG("ggml_webgpu_create_buffer()");

    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size = size;
    buffer_desc.usage = usage;
    buffer_desc.label = label;
    buffer_desc.mappedAtCreation = false;
    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

/** End WebGPU object initializations */

/** WebGPU Actions */

static void ggml_backend_webgpu_map_buffer(webgpu_context ctx, wgpu::Buffer buffer, wgpu::MapMode mode, size_t offset, size_t size) {
    ctx->instance.WaitAny(buffer.MapAsync(
        mode, offset, size, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to map buffer: %s\n", message.data);
            }
        }),
        UINT64_MAX
    );
}

static void ggml_backend_webgpu_buffer_memset(webgpu_context ctx, wgpu::Buffer buf, uint32_t value, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    wgpu::Device device = ctx->device;

    // map the host parameters buffer
    ggml_backend_webgpu_map_buffer(ctx, ctx->memset_params_host_buf, wgpu::MapMode::Write, 0, ctx->memset_params_host_buf.GetSize());
    uint32_t * params = (uint32_t *) ctx->memset_params_host_buf.GetMappedRange();

    params[0] = (uint32_t)offset;
    params[1] = (uint32_t)size;
    params[2] = value;
    ctx->memset_params_host_buf.Unmap();

    wgpu::BindGroupEntry entries[2];
    entries[0].binding = 0; // binding for the buffer to memset
    entries[0].buffer = buf;
    entries[0].offset = 0;
    entries[0].size = buf.GetSize();
    entries[1].binding = 1; // binding for the parameters
    entries[1].buffer = ctx->memset_params_dev_buf;
    entries[1].offset = 0;
    entries[1].size = ctx->memset_params_dev_buf.GetSize();

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout = ctx->memset_pipeline.GetBindGroupLayout(0);
    bind_group_desc.entryCount = 2;
    bind_group_desc.label = "ggml_memset";
    bind_group_desc.entries = entries;
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(
        ctx->memset_params_host_buf, 0,
        ctx->memset_params_dev_buf, 0,
        ctx->memset_params_dev_buf.GetSize()
    );
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline);
    pass.SetBindGroup(0, bind_group);
    size_t bytes_per_wg = ctx->limits.maxComputeWorkgroupSizeX * ctx->memset_bytes_per_thread;
    pass.DispatchWorkgroups(((size + 3) + bytes_per_wg - 1) / bytes_per_wg, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();

    ctx->queue.Submit(1, &commands);
}

static void ggml_backend_webgpu_wait_on_submission(webgpu_context ctx) {
    // Wait for the queue to finish processing all commands
    ctx->instance.WaitAny(ctx->queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
            if (status != wgpu::QueueWorkDoneStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to wait on queue: %s\n", message.data);
            }
        }),
        UINT64_MAX
    );
}

/** End WebGPU Actions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *)backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *)backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

    // TODO: cleanup
    GGML_UNUSED(ctx);
}

// Returns true if node has enqueued work into the queue, false otherwise
static bool ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node){
    if (ggml_is_empty(node)) {
        return false;
    }

    WEBGPU_LOG_DEBUG("ggml_webgpu_encode_node(" << node << ", " << ggml_op_name(node->op) << ")");


    switch (node->op) {
        // no-ops
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return false;

        case GGML_OP_CPY: {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            const ggml_tensor * src = node->src[0];
            ggml_backend_webgpu_buffer_context * src_ctx = (ggml_backend_webgpu_buffer_context *) src->buffer->context;
            size_t src_offset = webgpu_tensor_offset(src) + src->view_offs;
            // assumes power of 2 offset alignment
            size_t src_misalignment = src_offset & (ctx->limits.minStorageBufferOffsetAlignment - 1);
            // align to minimum offset alignment
            src_offset &= ~(ctx->limits.minStorageBufferOffsetAlignment - 1);
            ggml_backend_webgpu_buffer_context * dst_ctx = (ggml_backend_webgpu_buffer_context *) node->buffer->context;
            size_t dst_offset = webgpu_tensor_offset(node) + node->view_offs;
            size_t dst_misalignment = dst_offset & (ctx->limits.minStorageBufferOffsetAlignment - 1);
            dst_offset &= ~(ctx->limits.minStorageBufferOffsetAlignment - 1);

            wgpu::Device device = ctx->device;
            ggml_backend_webgpu_map_buffer(ctx, ctx->cpy_params_host_buf,
                wgpu::MapMode::Write, 0, ctx->cpy_params_host_buf.GetSize());
            uint32_t * params = (uint32_t *) ctx->cpy_params_host_buf.GetMappedRange();
            uint32_t ne = (uint32_t)ggml_nelements(node);
            params[0] = ne;
            params[1] = src_misalignment/ggml_type_size(src->type);
            params[2] = dst_misalignment/ggml_type_size(node->type);

            // Convert byte-strides to element-strides
            params[3] = (uint32_t)src->nb[0]/ggml_type_size(src->type);
            params[4] = (uint32_t)src->nb[1]/ggml_type_size(src->type);
            params[5] = (uint32_t)src->nb[2]/ggml_type_size(src->type);
            params[6] = (uint32_t)src->nb[3]/ggml_type_size(src->type);
            params[7] = (uint32_t)node->nb[0]/ggml_type_size(node->type);
            params[8] = (uint32_t)node->nb[1]/ggml_type_size(node->type);
            params[9] = (uint32_t)node->nb[2]/ggml_type_size(node->type);
            params[10] = (uint32_t)node->nb[3]/ggml_type_size(node->type);
            // Logical shape — same for both tensors even if permuted
            params[11] = (uint32_t)(src->ne[0]);
            params[12] = (uint32_t)(src->ne[1]);
            params[13] = (uint32_t)(src->ne[2]);
            params[14] = (uint32_t)(src->ne[3]);

            ctx->cpy_params_host_buf.Unmap();

            wgpu::BindGroupEntry entries[3];
            entries[0].binding = 0;
            entries[0].buffer = src_ctx->buffer;
            entries[0].offset = src_offset;
            entries[0].size = (ggml_nbytes(src) + src_misalignment + WEBGPU_STORAGE_BUF_BINDING_MULT - 1) & ~(WEBGPU_STORAGE_BUF_BINDING_MULT - 1);

            entries[1].binding = 1;
            entries[1].buffer = dst_ctx->buffer;
            entries[1].offset = dst_offset;
            entries[1].size = (ggml_nbytes(node) + dst_misalignment + WEBGPU_STORAGE_BUF_BINDING_MULT - 1) & ~(WEBGPU_STORAGE_BUF_BINDING_MULT - 1);

            entries[2].binding = 2;
            entries[2].buffer = ctx->cpy_params_dev_buf;
            entries[2].offset = 0;
            entries[2].size = ctx->cpy_params_dev_buf.GetSize();

            wgpu::BindGroupDescriptor bind_group_desc;
            bind_group_desc.layout = ctx->cpy_pipeline.GetBindGroupLayout(0);
            bind_group_desc.label = "ggml_op_cpy";
            bind_group_desc.entryCount = 3;
            bind_group_desc.entries = entries;
            wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(
                ctx->cpy_params_host_buf, 0,
                ctx->cpy_params_dev_buf, 0,
                ctx->cpy_params_dev_buf.GetSize()
            );
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(ctx->cpy_pipeline);
            pass.SetBindGroup(0, bind_group);
            size_t max_wg_size = ctx->limits.maxComputeWorkgroupSizeX;
            pass.DispatchWorkgroups((ne + max_wg_size - 1) / max_wg_size);
            pass.End();
            wgpu::CommandBuffer commands = encoder.Finish();

            // TODO, don't submit here, batch submissions
            ctx->queue.Submit(1, &commands);
            // TODO, don't wait on submission here
            ggml_backend_webgpu_wait_on_submission(ctx);
            return true;
        }

        case GGML_OP_MUL_MAT:
         {
            const ggml_tensor * src0 = node->src[0];
            ggml_backend_webgpu_buffer_context * src0_ctx = (ggml_backend_webgpu_buffer_context *) src0->buffer->context;
            size_t src0_offset = webgpu_tensor_offset(src0) + src0->view_offs;
            const ggml_tensor * src1 = node->src[1];
            ggml_backend_webgpu_buffer_context * src1_ctx = (ggml_backend_webgpu_buffer_context *) src1->buffer->context;
            size_t src1_offset = webgpu_tensor_offset(src1) + src1->view_offs;
            ggml_backend_webgpu_buffer_context * dst_ctx = (ggml_backend_webgpu_buffer_context *) node->buffer->context;

            size_t dst_offset = webgpu_tensor_offset(node) + node->view_offs;

            wgpu::Device device = ctx->device;

            // map the host parameters buffer
            ggml_backend_webgpu_map_buffer(ctx, ctx->mul_mat_params_host_buf,
                wgpu::MapMode::Write, 0, ctx->mul_mat_params_host_buf.GetSize());
            uint32_t * params = (uint32_t *) ctx->mul_mat_params_host_buf.GetMappedRange();

            params[0] = (uint32_t)node->ne[1]; // number of rows in result (M)
            params[1] = (uint32_t)node->ne[0]; // number of columns in result (N)
            params[2] = (uint32_t)src0->ne[0]; // number of columns in src0/src1 (K)

            params[3] = (uint32_t)src0->nb[1]/ggml_type_size(src0->type); // stride (elements) of src0 in dimension 1
            params[4] = (uint32_t)src1->nb[1]/ggml_type_size(src1->type); // stride (elements) of src1 in dimension 1
            params[5] = (uint32_t)src0->nb[2]/ggml_type_size(src0->type); // stride (elements) of src0 in dimension 2
            params[6] = (uint32_t)src1->nb[2]/ggml_type_size(src1->type); // stride (elements) of src1 in dimension 2
            params[7] = (uint32_t)src0->nb[3]/ggml_type_size(src0->type); // stride (elements) of src0 in dimension 3
            params[8] = (uint32_t)src1->nb[3]/ggml_type_size(src1->type); // stride (elements) of src1 in dimension 3

            params[9] = (uint32_t)src0->ne[2]; // batch size in dimension 2
            params[10] = (uint32_t)src0->ne[3]; // batch size in dimension 3
            params[11] = (uint32_t)(src1->ne[2]/src0->ne[2]); // broadcast in dimension 2
            params[12] = (uint32_t)(src1->ne[3]/src0->ne[3]); // broadcast in dimension 3

            ctx->mul_mat_params_host_buf.Unmap();

            wgpu::BindGroupEntry entries[4];
            entries[0].binding = 0;
            entries[0].buffer = src0_ctx->buffer;
            entries[0].offset = src0_offset;
            entries[0].size = ggml_nbytes(src0);

            entries[1].binding = 1;
            entries[1].buffer = src1_ctx->buffer;
            entries[1].offset = src1_offset;
            entries[1].size = ggml_nbytes(src1);

            entries[2].binding = 2;
            entries[2].buffer = dst_ctx->buffer;
            entries[2].offset = dst_offset;
            entries[2].size = ggml_nbytes(node);

            entries[3].binding = 3;
            entries[3].buffer = ctx->mul_mat_params_dev_buf;
            entries[3].offset = 0;
            entries[3].size = ctx->mul_mat_params_dev_buf.GetSize();

            wgpu::BindGroupDescriptor bind_group_desc;
            bind_group_desc.layout = ctx->mul_mat_pipeline.GetBindGroupLayout(0);
            bind_group_desc.entryCount = 4;
            bind_group_desc.label = "ggml_op_mul_mat";
            bind_group_desc.entries = entries;
            wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(
                ctx->mul_mat_params_host_buf, 0,
                ctx->mul_mat_params_dev_buf, 0,
                ctx->mul_mat_params_dev_buf.GetSize()
            );
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(ctx->mul_mat_pipeline);
            pass.SetBindGroup(0, bind_group);
            pass.DispatchWorkgroups((node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3] + WEBGPU_MUL_MAT_WG_SIZE - 1) / WEBGPU_MUL_MAT_WG_SIZE);
            pass.End();
            wgpu::CommandBuffer commands = encoder.Finish();

            // TODO, don't submit here, batch submissions
            ctx->queue.Submit(1, &commands);
            // TODO, don't wait on submission here
            ggml_backend_webgpu_wait_on_submission(ctx);
            return true;
        }

        default:
            return false;
    }
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = static_cast<ggml_backend_webgpu_context *>(backend->context);
    webgpu_context ctx = backend_ctx->webgpu_ctx;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_webgpu_encode_node(ctx, cgraph->nodes[i]);
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

static void ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_free_buffer()");
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    ctx->buffer.Destroy();
}

// Returns the "fake" base pointer.
static void * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return webgpu_ptr_base;
}

static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (size == 0) {
        WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor: size is zero, nothing to do.");
        return;
    }

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;
    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->webgpu_ctx, buf_ctx->buffer, val32, total_offset, size);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_context webgpu_ctx = buf_ctx->webgpu_ctx;

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    webgpu_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size/4)*4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;
        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;
        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *)&val32)[i] = ((const uint8_t *)data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(webgpu_ctx, buf_ctx->buffer, val32, total_offset + (size - remaining_size), remaining_size);
    }
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_context webgpu_ctx = buf_ctx->webgpu_ctx;
    wgpu::Device device = webgpu_ctx->device;

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    size_t final_size = size;
    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next multiple of 4
        final_size = size + (4 - (size % 4));
    }

    std::lock_guard<std::mutex> lock(webgpu_ctx->mutex);

    if (webgpu_ctx->get_tensor_staging_buf == nullptr ||
        webgpu_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        if (webgpu_ctx->get_tensor_staging_buf) {
            webgpu_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device, webgpu_ctx->get_tensor_staging_buf, final_size,
            wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buf_ctx->buffer, total_offset, webgpu_ctx->get_tensor_staging_buf, 0, final_size);
    wgpu::CommandBuffer commands = encoder.Finish();
    // Submit the command buffer to the queue
    webgpu_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_buffer(webgpu_ctx, webgpu_ctx->get_tensor_staging_buf, wgpu::MapMode::Read, 0, final_size);
    // Must specify size here since the staging buffer might be larger than the tensor size
    const void * mapped_range = webgpu_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, mapped_range, size);
    webgpu_ctx->get_tensor_staging_buf.Unmap();
}

static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->webgpu_ctx, buf_ctx->buffer, value, 0, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL, // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL, // TODO: optional, think it coordinates with .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);

    wgpu::Buffer buf;
    ggml_webgpu_create_buffer(ctx->webgpu_ctx->device, buf, size,
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst, "allocated_buffer");

    ggml_backend_webgpu_buffer_context * buf_ctx = new ggml_backend_webgpu_buffer_context(ctx->webgpu_ctx, buf);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->webgpu_ctx->limits.minStorageBufferOffsetAlignment;
}

// maxBufferSize might be larger, but you can't bind more than maxStorageBufferBindingSize to a single binding.
static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->webgpu_ctx->limits.maxStorageBufferBindingSize;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

static const char * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_name.c_str();
}

static const char * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_desc.c_str();
}

static void ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: what do we actually want to return here? maxBufferSize might not be the full available memory.
    *free = ctx->webgpu_ctx->limits.maxBufferSize;
    *total = ctx->webgpu_ctx->limits.maxBufferSize;
}

static enum ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_guid_t ggml_backend_webgpu_guid(void) {
    static const char * guid_str = "__ggml_webgpu :)";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

static void ggml_webgpu_init_memset_pipeline(webgpu_context webgpu_ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_wg_size = webgpu_ctx->limits.maxComputeWorkgroupSizeX;
    size_t max_threads = max_wg_size * webgpu_ctx->limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    webgpu_ctx->memset_bytes_per_thread = (webgpu_ctx->limits.maxStorageBufferBindingSize + max_threads - 1) / max_threads;
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].key = "wg_size";
    constants[0].value = max_wg_size;
    constants[1].key = "bytes_per_thread";
    constants[1].value = webgpu_ctx->memset_bytes_per_thread;
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->memset_pipeline, wgsl_memset, "memset", constants);
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->memset_params_dev_buf,
        3 * sizeof(uint32_t), // 3 parameters: buffer size, offset, value
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "memset_params_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->memset_params_host_buf,
        3 * sizeof(uint32_t), wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, "memset_params_host_buf");
}

static void ggml_webgpu_init_mul_mat_pipeline(webgpu_context webgpu_ctx) {
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline, wgsl_mul_mat, "mul_mat");
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->mul_mat_params_dev_buf, WEBGPU_MUL_MAT_PARAMS_SIZE,
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "mul_mat_params_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->mul_mat_params_host_buf, WEBGPU_MUL_MAT_PARAMS_SIZE,
        wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, "mul_mat_params_host_buf");
}

static void ggml_webgpu_init_cpy_pipeline(webgpu_context webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key = "wg_size";
    constants[0].value = webgpu_ctx->limits.maxComputeWorkgroupSizeX;

    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline, wgsl_cpy, "cpy", constants);
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->cpy_params_dev_buf, WEBGPU_CPY_PARAMS_SIZE,
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "cpy_params_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->device, webgpu_ctx->cpy_params_host_buf, WEBGPU_CPY_PARAMS_SIZE,
        wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, "cpy_params_host_buf");
}

// TODO: Make thread safe if multiple devices are used
static ggml_backend_t ggml_backend_webgpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_device_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    webgpu_context webgpu_ctx = dev_ctx->webgpu_ctx;

    std::lock_guard<std::mutex> lock(webgpu_ctx->mutex);

    if (!webgpu_ctx->device_initialized) {
        // Initialize device
        wgpu::DeviceDescriptor dev_desc;
        dev_desc.requiredLimits = &webgpu_ctx->limits;
        dev_desc.requiredFeatures = webgpu_ctx->features.features;
        dev_desc.requiredFeatureCount = webgpu_ctx->features.featureCount;
        dev_desc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous,
            [](const wgpu::Device& device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
                GGML_UNUSED(device);
                GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason), message.data);
        });
        dev_desc.SetUncapturedErrorCallback(
            [](const wgpu::Device& device, wgpu::ErrorType reason, wgpu::StringView message) {
                GGML_UNUSED(device);
                GGML_LOG_ERROR("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason), message.data);
        });
        webgpu_ctx->instance.WaitAny(webgpu_ctx->adapter.RequestDevice(&dev_desc, wgpu::CallbackMode::WaitAnyOnly,
            [webgpu_ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", message.data);
                    return;
                }
                webgpu_ctx->device = device;
            }),
            UINT64_MAX
        );
        GGML_ASSERT(webgpu_ctx->device != nullptr);

        // Initialize (compute) queue
        webgpu_ctx->queue = webgpu_ctx->device.GetQueue();

        ggml_webgpu_init_memset_pipeline(webgpu_ctx);
        ggml_webgpu_init_mul_mat_pipeline(webgpu_ctx);
        ggml_webgpu_init_cpy_pipeline(webgpu_ctx);
        webgpu_ctx->device_initialized = true;
    }

    static ggml_backend_webgpu_context backend_ctx;
    backend_ctx.name = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx.webgpu_ctx = webgpu_ctx;

    // See GGML Backend Interface section
    static ggml_backend backend = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ &backend_ctx,
    };

    return &backend;
}

static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section
    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_webgpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_webgpu_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_webgpu_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ NULL, // defaults to false
        },
        /* .device  = */ dev,
        /* .context = */ NULL,
    };

    return &ggml_backend_webgpu_buffer_type;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return  buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CPY:
            return op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_MUL_MAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        default:
            return false;
    }
}

static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_device_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->name;
}

static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->device_count;
}

// TODO: Does this need to be thread safe? Is it only called once?
// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    webgpu_context ctx = reg_ctx->webgpu_ctx;

    wgpu::RequestAdapterOptions options = {};
    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
            return;
        }
        *static_cast<wgpu::Adapter *>(userdata) = adapter;
    };
    void *userdata = &ctx->adapter;
    ctx->instance.WaitAny(ctx->instance.RequestAdapter(&options, wgpu::CallbackMode::WaitAnyOnly, callback, userdata), UINT64_MAX);
    GGML_ASSERT(ctx->adapter != nullptr);

    ctx->adapter.GetLimits(&ctx->limits);
    ctx->adapter.GetFeatures(&ctx->features);

    wgpu::AdapterInfo info{};
    ctx->adapter.GetInfo(&info);

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.webgpu_ctx = ctx;
    device_ctx.device_name = GGML_WEBGPU_NAME;
    device_ctx.device_desc = std::string(info.description.data);

    GGML_LOG_INFO("ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | device_desc: %s\n",
        info.vendorID, info.vendor.data, info.architecture.data, info.deviceID, info.device.data, info.description.data);

    // See GGML Backend Device Interface section
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };
    return &device;
}


static const struct ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

/* End GGML Backend Registration Interface */

// TODO: Does this need to be thread safe? Is it only called once?
ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    webgpu_context webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->device_initialized = false;

    static ggml_backend_webgpu_reg_context ctx;
    ctx.webgpu_ctx = webgpu_ctx;
    ctx.name = GGML_WEBGPU_NAME;
    ctx.device_count = 1;

    wgpu::InstanceDescriptor instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = {wgpu::InstanceFeatureName::TimedWaitAny};
    instance_descriptor.requiredFeatures = instance_features.data();
    instance_descriptor.requiredFeatureCount = instance_features.size();
    webgpu_ctx->instance = wgpu::CreateInstance(&instance_descriptor);
    GGML_ASSERT(webgpu_ctx->instance != nullptr);

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ &ctx,
    };
    return &reg;
}

ggml_backend_t ggml_backend_webgpu_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_webgpu_reg(), 0);

    return ggml_backend_webgpu_device_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)

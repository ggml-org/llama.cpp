/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#include "ggml-webgpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-wgsl-shaders.hpp"

#include <webgpu/webgpu_cpp.h>

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(msg)  std::cout << msg << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 32
#else
#    define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif  // GGML_WEBGPU_DEBUG

/* Constants */

#define WEBGPU_COMMAND_SUBMIT_BATCH_SIZE     16
#define WEBGPU_MUL_MAT_WG_SIZE               64
#define WEBGPU_NUM_PARAM_BUFS                100
#define WEBGPU_PARAMS_BUF_SIZE_BYTES         128  // enough for 32 parameters
#define WEBGPU_NUM_SET_ROWS_ERROR_BUFS       32
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES 4
#define WEBGPU_STORAGE_BUF_BINDING_MULT      4  // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to their locations.
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

// Always returns the base offset of a tensor, regardless of views.
static uint64_t webgpu_tensor_offset(const ggml_tensor * tensor) {
    if (tensor->view_src) {
        return (uint8_t *) tensor->view_src->data - (uint8_t *) webgpu_ptr_base;
    }
    return (uint8_t *) tensor->data - (uint8_t *) webgpu_ptr_base;
}

/* Struct definitions */

// Forward reference
static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label);

struct webgpu_pool_bufs {
    wgpu::Buffer host_buf;
    wgpu::Buffer dev_buf;
};

// Holds a pool of parameter buffers for WebGPU operations
struct webgpu_buf_pool {
    std::vector<webgpu_pool_bufs> free;

    std::mutex mutex;

    std::condition_variable cv;

    void init(wgpu::Device      device,
              int               num_bufs,
              size_t            buf_size,
              wgpu::BufferUsage dev_buf_usage,
              wgpu::BufferUsage host_buf_usage) {
        for (int i = 0; i < num_bufs; i++) {
            wgpu::Buffer host_buf;
            wgpu::Buffer dev_buf;
            ggml_webgpu_create_buffer(device, host_buf, buf_size, host_buf_usage, "ggml_webgpu_host_pool_buf");
            ggml_webgpu_create_buffer(device, dev_buf, buf_size, dev_buf_usage, "ggml_webgpu_dev_pool_buf");
            free.push_back({ host_buf, dev_buf });
        }
    }

    webgpu_pool_bufs alloc_bufs() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !free.empty(); });
        webgpu_pool_bufs bufs = free.back();
        free.pop_back();
        return bufs;
    }

    void free_bufs(std::vector<webgpu_pool_bufs> bufs) {
        std::lock_guard<std::mutex> lock(mutex);
        free.insert(free.end(), bufs.begin(), bufs.end());
        cv.notify_all();
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & bufs : free) {
            bufs.host_buf.Destroy();
            bufs.dev_buf.Destroy();
        }
        free.clear();
    }
};

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    wgpu::Limits   limits;

    std::recursive_mutex mutex;

    webgpu_buf_pool param_buf_pool;
    webgpu_buf_pool set_rows_error_buf_pool;

    wgpu::ComputePipeline memset_pipeline;
    wgpu::ComputePipeline mul_mat_pipeline[30][2];
    wgpu::ComputePipeline set_rows_pipeline;
    wgpu::ComputePipeline cpy_pipeline;

    size_t memset_bytes_per_thread;

    // Staging buffer for reading data from the GPU
    wgpu::Buffer get_tensor_staging_buf;

    // Command buffers which need to be submitted
    std::vector<wgpu::CommandBuffer> staged_command_bufs;

    // Parameter buffers associated with the staged command buffers
    std::vector<webgpu_pool_bufs> staged_param_bufs;
    // Buffers associated with set_rows operations, used to store potential errors
    std::vector<webgpu_pool_bufs> staged_set_row_error_bufs;

    std::vector<wgpu::FutureWaitInfo> callback_futures;

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

struct ggml_backend_webgpu_reg_context {
    webgpu_context webgpu_ctx;
    size_t         device_count;
    const char *   name;
};

struct ggml_backend_webgpu_device_context {
    webgpu_context webgpu_ctx;
    std::string    device_name;
    std::string    device_desc;
};

struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

struct ggml_backend_webgpu_buffer_context {
    webgpu_context webgpu_ctx;
    wgpu::Buffer   buffer;

    ggml_backend_webgpu_buffer_context(webgpu_context ctx, wgpu::Buffer buf) :
        webgpu_ctx(std::move(ctx)),
        buffer(std::move(buf)) {}
};

/* End struct definitions */

/* WebGPU object initializations */

static void ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                        wgpu::ComputePipeline &                  pipeline,
                                        const char *                             shader_code,
                                        const char *                             label,
                                        const std::vector<wgpu::ConstantEntry> & constants = {}) {
    WEBGPU_LOG_DEBUG("ggml_webgpu_create_pipeline()");

    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    if (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constantCount = constants.size();
    }
    pipeline = device.CreateComputePipeline(&pipeline_desc);
}

static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label) {
    WEBGPU_LOG_DEBUG("ggml_webgpu_create_buffer()");

    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = false;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

/** End WebGPU object initializations */

/** WebGPU Actions */

// Wait for the queue to finish processing all submitted work
static void ggml_backend_webgpu_wait_on_submission(webgpu_context & ctx) {
    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    if (ctx->callback_futures.empty()) {
        // no existing callbacks, wait on queue submission
        ctx->instance.WaitAny(ctx->queue.OnSubmittedWorkDone(
                                  wgpu::CallbackMode::AllowSpontaneous,
                                  [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                          GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n", std::string(message).c_str());
                                      }
                                  }),
                              UINT64_MAX);
    } else {
        // existing callbacks, wait on them
        ctx->instance.WaitAny(ctx->callback_futures.size(), ctx->callback_futures.data(), UINT64_MAX);
        ctx->callback_futures.clear();
    }
}

static void ggml_backend_webgpu_submit_queue(webgpu_context & ctx) {
    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_submit_queue()");
    if (ctx->staged_command_bufs.empty()) {
        // Nothing to submit
        return;
    }
    ctx->queue.Submit(ctx->staged_command_bufs.size(), ctx->staged_command_bufs.data());

    // If there are SET_ROWS operations in this submission, copy their error buffers to the host.
    if (ctx->staged_set_row_error_bufs.size() > 0) {
        wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
        for (auto & error_bufs : ctx->staged_set_row_error_bufs) {
            // Copy the error buffer to the host buffer
            encoder.CopyBufferToBuffer(error_bufs.dev_buf, 0, error_bufs.host_buf, 0, error_bufs.host_buf.GetSize());
        }
        wgpu::CommandBuffer commands = encoder.Finish();
        ctx->queue.Submit(1, &commands);
    }

    ctx->staged_command_bufs.clear();
    std::vector<webgpu_pool_bufs> staged_param_bufs         = std::move(ctx->staged_param_bufs);
    std::vector<webgpu_pool_bufs> staged_set_row_error_bufs = std::move(ctx->staged_set_row_error_bufs);

    // Free the staged parameter buffers once the submission completes
    wgpu::Future p_f = ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous,
        [ctx, staged_param_bufs](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
            if (status != wgpu::QueueWorkDoneStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n", std::string(message).c_str());
            }
            // Free the staged buffers
            ctx->param_buf_pool.free_bufs(staged_param_bufs);
        });
    ctx->callback_futures.push_back({ p_f });

    // Check for errrors in SET_ROWS operations
    for (auto & error_bufs : staged_set_row_error_bufs) {
        wgpu::Future f = error_bufs.host_buf.MapAsync(
            wgpu::MapMode::Read,
            0,
            error_bufs.host_buf.GetSize(),
            wgpu::CallbackMode::AllowSpontaneous,
            [ctx, error_bufs](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to map error buffer: %s\n", std::string(message).c_str());
                } else {
                    const uint32_t * error_data = (const uint32_t *) error_bufs.host_buf.GetConstMappedRange();
                    if (*error_data) {
                        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
                    }
                    // We can't unmap in here due to WebGPU reentrancy limitations.
                    ctx->set_rows_error_buf_pool.free_bufs({ error_bufs });
                }
            });
        ctx->callback_futures.push_back({ f });
    }
}

static void ggml_backend_webgpu_map_buffer(webgpu_context & ctx,
                                           wgpu::Buffer &   buffer,
                                           wgpu::MapMode    mode,
                                           size_t           offset,
                                           size_t           size) {
    ctx->instance.WaitAny(buffer.MapAsync(mode,
                                          offset,
                                          size,
                                          wgpu::CallbackMode::AllowSpontaneous,
                                          [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                                              if (status != wgpu::MapAsyncStatus::Success) {
                                                  GGML_LOG_ERROR("ggml_webgpu: Failed to map buffer: %s\n",
                                                                 message.data);
                                              }
                                          }),
                          UINT64_MAX);
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
static void ggml_backend_webgpu_debug(webgpu_context & ctx) {
    ggml_backend_webgpu_submit_queue(ctx);
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);

    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    const uint32_t * debug_data = (const uint32_t *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug data:";
    for (size_t i = 0; i < WEBGPU_DEBUG_BUF_ELEMS; i++) {
        std::cout << "  " << i << ": " << debug_data[i];
    }
    std::cout << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

static void ggml_backend_webgpu_build_and_enqueue(webgpu_context &                  ctx,
                                                  wgpu::ComputePipeline &           pipeline,
                                                  std::vector<uint32_t>             params,
                                                  std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                  uint32_t                          wg_x,
                                                  bool                              submit_and_wait = false) {
    webgpu_pool_bufs params_bufs = ctx->param_buf_pool.alloc_bufs();

    ggml_backend_webgpu_map_buffer(ctx, params_bufs.host_buf, wgpu::MapMode::Write, 0, params_bufs.host_buf.GetSize());
    uint32_t * _params = (uint32_t *) params_bufs.host_buf.GetMappedRange();
    for (size_t i = 0; i < params.size(); i++) {
        _params[i] = params[i];
    };

    params_bufs.host_buf.Unmap();

    uint32_t params_bufs_binding_num = bind_group_entries.size();
    bind_group_entries.push_back({ .binding = params_bufs_binding_num,
                                   .buffer  = params_bufs.dev_buf,
                                   .offset  = 0,
                                   .size    = params_bufs.dev_buf.GetSize() });

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = pipeline.GetBindGroupLayout(0);
    bind_group_desc.entryCount = bind_group_entries.size();
    bind_group_desc.entries    = bind_group_entries.data();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(params_bufs.host_buf, 0, params_bufs.dev_buf, 0, params_bufs.dev_buf.GetSize());
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    if (submit_and_wait) {
        // Submit and wait immediately
        ctx->queue.Submit(1, &commands);
        ctx->instance.WaitAny(ctx->queue.OnSubmittedWorkDone(
                                  wgpu::CallbackMode::AllowSpontaneous,
                                  [ctx, params_bufs](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                          GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n", message.data);
                                      }
                                      ctx->param_buf_pool.free_bufs({ params_bufs });
                                  }),
                              UINT64_MAX);
    } else {
        // Lock the context mutex when pushing to the staging vectors.
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        // Enqueue commands and only submit if we have enough staged commands
        ctx->staged_command_bufs.push_back(commands);
        ctx->staged_param_bufs.push_back(params_bufs);
        if (ctx->staged_command_bufs.size() == WEBGPU_COMMAND_SUBMIT_BATCH_SIZE) {
            ggml_backend_webgpu_submit_queue(ctx);
        }
    }
}

static void ggml_backend_webgpu_buffer_memset(webgpu_context & ctx,
                                              wgpu::Buffer &   buf,
                                              uint32_t         value,
                                              size_t           offset,
                                              size_t           size) {
    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0, .buffer = buf, .offset = 0, .size = buf.GetSize() }
    };
    size_t   bytes_per_wg = ctx->limits.maxComputeWorkgroupSizeX * ctx->memset_bytes_per_thread;
    uint32_t wg_x         = ((size + 3) + bytes_per_wg - 1) / bytes_per_wg;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->memset_pipeline, params, entries, wg_x, true);
}

/** End WebGPU Actions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

    // TODO: cleanup
    GGML_UNUSED(ctx);
}

static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    return webgpu_tensor_offset(tensor) + tensor->view_offs;
}

static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    return ctx->buffer;
}

static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & (ctx->limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & ~(ctx->limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ggml_tensor * t) {
    return (ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(ctx, t) + WEBGPU_STORAGE_BUF_BINDING_MULT - 1) &
           ~(WEBGPU_STORAGE_BUF_BINDING_MULT - 1);
}

static void ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     // Convert byte-strides to element-strides
                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
                                     // Logical shape — same for both tensors even if permuted
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3] };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(dst),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) }
    };

    size_t   max_wg_size = ctx->limits.maxComputeWorkgroupSizeX;
    uint32_t wg_x        = (ne + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->cpy_pipeline, params, entries, wg_x);
}

static void ggml_webgpu_set_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    // For set rows specifically, we need to check if src and idx are empty tensors.
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return;
    }

    webgpu_pool_bufs error_bufs = ctx->set_rows_error_buf_pool.alloc_bufs();
    if (error_bufs.host_buf.GetMapState() == wgpu::BufferMapState::Mapped) {
        error_bufs.host_buf.Unmap();
    }

    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     // Convert byte-strides to element-strides
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
                                     // Shape of src
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],
                                     // Shape of idx
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(idx),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, idx),
         .size    = ggml_webgpu_tensor_binding_size(ctx, idx) },
        { .binding = 2,
         .buffer  = ggml_webgpu_tensor_buf(dst),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) },
        { .binding = 3, .buffer = error_bufs.dev_buf, .offset = 0, .size = error_bufs.dev_buf.GetSize() }
    };

    size_t   max_wg_size = ctx->limits.maxComputeWorkgroupSizeX;
    uint32_t wg_x        = (src->ne[1] * src->ne[2] * src->ne[3] + max_wg_size - 1) / max_wg_size;

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    ctx->staged_set_row_error_bufs.push_back(error_bufs);

    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->set_rows_pipeline, params, entries, wg_x);
}

static void ggml_webgpu_mul_mat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) dst->ne[1],                                  // number of rows in result (M)
        (uint32_t) dst->ne[0],                                  // number of columns in result (N)
        (uint32_t) src0->ne[0],                                 // number of columns in src0/src1 (K)
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),  // stride (elements/blocks) of src0 in dimension 1
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),  // stride (elements/blocks) of src1 in dimension 1
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),  // stride (elements/blocks) of src0 in dimension 2
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),  // stride (elements/blocks) of src1 in dimension 2
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),  // stride (elements/blocks) of src0 in dimension 3
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),  // stride (elements/blocks) of src1 in dimension 3
        (uint32_t) src0->ne[2],                                 // batch size in dimension 2
        (uint32_t) src0->ne[3],                                 // batch size in dimension 3
        (uint32_t) (src1->ne[2] / src0->ne[2]),                 // broadcast in dimension 2
        (uint32_t) (src1->ne[3] / src0->ne[3])                  // broadcast in dimension 3
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src0),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src0),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src0) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(src1),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src1),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src1) },
        { .binding = 2,
         .buffer  = ggml_webgpu_tensor_buf(dst),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst)  },
    };

    uint32_t wg_x =
        (dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] + WEBGPU_MUL_MAT_WG_SIZE - 1) / WEBGPU_MUL_MAT_WG_SIZE;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->mul_mat_pipeline[src0->type][src1->type], params, entries, wg_x);
}

// Returns true if node has enqueued work into the queue, false otherwise
static bool ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return false;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode_node(" << node << ", " << ggml_op_name(node->op) << ")");

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    switch (node->op) {
            // no-ops
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return false;
        case GGML_OP_CPY:
            {
                ggml_webgpu_cpy(ctx, src0, node);
                break;
            }
        case GGML_OP_SET_ROWS:
            {
                ggml_webgpu_set_rows(ctx, src0, src1, node);
                break;
            }
        case GGML_OP_MUL_MAT:
            {
                ggml_webgpu_mul_mat(ctx, src0, src1, node);
                break;
            }
        default:
            return false;
    }
    return true;
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = static_cast<ggml_backend_webgpu_context *>(backend->context);
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_webgpu_encode_node(ctx, cgraph->nodes[i]);
    }

    ggml_backend_webgpu_submit_queue(ctx);
    ggml_backend_webgpu_wait_on_submission(ctx);

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

static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    if (size == 0) {
        WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor: size is zero, nothing to do.");
        return;
    }

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", "
                                                                 << offset << ", " << size << ")");

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->webgpu_ctx, buf_ctx->buffer, val32, total_offset, size);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", "
                                                              << offset << ", " << size << ")");
    ggml_backend_webgpu_buffer_context * buf_ctx    = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_context                       webgpu_ctx = buf_ctx->webgpu_ctx;

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    webgpu_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(
            webgpu_ctx, buf_ctx->buffer, val32, total_offset + (size - remaining_size), remaining_size);
    } else {
        // wait for WriteBuffer to complete
        ggml_backend_webgpu_wait_on_submission(webgpu_ctx);
    }
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", "
                                                              << offset << ", " << size << ")");

    ggml_backend_webgpu_buffer_context * buf_ctx    = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_context                       webgpu_ctx = buf_ctx->webgpu_ctx;
    wgpu::Device                         device     = webgpu_ctx->device;

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    size_t final_size = size;
    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next multiple of 4
        final_size = size + (4 - (size % 4));
    }

    std::lock_guard<std::recursive_mutex> lock(webgpu_ctx->mutex);

    if (webgpu_ctx->get_tensor_staging_buf == nullptr || webgpu_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        if (webgpu_ctx->get_tensor_staging_buf) {
            webgpu_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device,
                                  webgpu_ctx->get_tensor_staging_buf,
                                  final_size,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                                  "get_tensor_staging_buf");
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
    /* .init_tensor     = */ NULL,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL,  // TODO: optional, think it coordinates with .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);

    wgpu::Buffer buf;
    ggml_webgpu_create_buffer(ctx->webgpu_ctx->device,
                              buf,
                              (size + WEBGPU_STORAGE_BUF_BINDING_MULT - 1) & ~(WEBGPU_STORAGE_BUF_BINDING_MULT - 1),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              "allocated_buffer");

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
    *free                                    = ctx->webgpu_ctx->limits.maxBufferSize;
    *total                                   = ctx->webgpu_ctx->limits.maxBufferSize;
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
    return reinterpret_cast<ggml_guid_t>((void *) guid_str);
}

static void ggml_webgpu_init_memset_pipeline(webgpu_context & webgpu_ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_wg_size = webgpu_ctx->limits.maxComputeWorkgroupSizeX;
    size_t max_threads = max_wg_size * webgpu_ctx->limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    webgpu_ctx->memset_bytes_per_thread =
        (webgpu_ctx->limits.maxStorageBufferBindingSize + max_threads - 1) / max_threads;
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].key   = "wg_size";
    constants[0].value = max_wg_size;
    constants[1].key   = "bytes_per_thread";
    constants[1].value = webgpu_ctx->memset_bytes_per_thread;
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->memset_pipeline, wgsl_memset, "memset", constants);
}

static void ggml_webgpu_init_mul_mat_pipeline(webgpu_context & webgpu_ctx) {
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F32][GGML_TYPE_F32],
                                wgsl_mul_mat_f32_f32,
                                "mul_mat_f32_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F16][GGML_TYPE_F16],
                                wgsl_mul_mat_f16_f16,
                                "mul_mat_f16_f16");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F16][GGML_TYPE_F32],
                                wgsl_mul_mat_f16_f32,
                                "mul_mat_f16_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_0_f32,
                                "mul_mat_q4_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_1][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_1_f32,
                                "mul_mat_q4_1_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_0_f32,
                                "mul_mat_q5_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_1][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_1_f32,
                                "mul_mat_q5_1_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q8_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q8_0_f32,
                                "mul_mat_q8_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q2_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q2_k_f32,
                                "mul_mat_q2_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q3_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q3_k_f32,
                                "mul_mat_q3_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_k_f32,
                                "mul_mat_q4_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_k_f32,
                                "mul_mat_q5_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q6_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q6_k_f32,
                                "mul_mat_q6_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_XXS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_xxs_f32,
                                "mul_mat_iq2_xxs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_XS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_xs_f32,
                                "mul_mat_iq2_xs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_s_f32,
                                "mul_mat_iq2_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ3_XXS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq3_xxs_f32,
                                "mul_mat_iq3_xxs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ3_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq3_s_f32,
                                "mul_mat_iq3_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ1_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq1_s_f32,
                                "mul_mat_iq1_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ1_M][GGML_TYPE_F32],
                                wgsl_mul_mat_iq1_m_f32,
                                "mul_mat_iq1_m_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ4_NL][GGML_TYPE_F32],
                                wgsl_mul_mat_iq4_nl_f32,
                                "mul_mat_iq4_nl_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device,
                                webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ4_XS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq4_xs_f32,
                                "mul_mat_iq4_xs_f32");
}

static void ggml_webgpu_init_set_rows_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = webgpu_ctx->limits.maxComputeWorkgroupSizeX;
    ggml_webgpu_create_pipeline(
        webgpu_ctx->device, webgpu_ctx->set_rows_pipeline, wgsl_set_rows, "set_rows", constants);
}

static void ggml_webgpu_init_cpy_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = webgpu_ctx->limits.maxComputeWorkgroupSizeX;
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline, wgsl_cpy, "cpy", constants);
}

static ggml_backend_t ggml_backend_webgpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_device_init()");

    ggml_backend_webgpu_device_context * dev_ctx    = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    webgpu_context                       webgpu_ctx = dev_ctx->webgpu_ctx;

    static ggml_backend_webgpu_context backend_ctx;
    backend_ctx.name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
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
                        /* .get_alloc_size   = */ NULL,  // defaults to ggml_nbytes
            /* .is_host          = */ NULL,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ NULL,
    };

    return &ggml_backend_webgpu_buffer_type;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_MUL_MAT:
            {
                switch (op->src[1]->type) {
                    case GGML_TYPE_F16:
                        return op->src[0]->type == GGML_TYPE_F16;
                    case GGML_TYPE_F32:
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                            case GGML_TYPE_Q4_0:
                            case GGML_TYPE_Q4_1:
                            case GGML_TYPE_Q5_0:
                            case GGML_TYPE_Q5_1:
                            case GGML_TYPE_Q8_0:
                            case GGML_TYPE_Q2_K:
                            case GGML_TYPE_Q3_K:
                            case GGML_TYPE_Q4_K:
                            case GGML_TYPE_Q5_K:
                            case GGML_TYPE_Q6_K:
                            case GGML_TYPE_IQ2_XXS:
                            case GGML_TYPE_IQ2_XS:
                            case GGML_TYPE_IQ2_S:
                            case GGML_TYPE_IQ3_XXS:
                            case GGML_TYPE_IQ3_S:
                            case GGML_TYPE_IQ1_S:
                            case GGML_TYPE_IQ1_M:
                            case GGML_TYPE_IQ4_NL:
                            case GGML_TYPE_IQ4_XS:
                                return true;
                            default:
                                return false;
                        }
                    default:
                        return false;
                }
            }
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
    auto                        callback =
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char * message, void * userdata) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                return;
            }
            *static_cast<wgpu::Adapter *>(userdata) = std::move(adapter);
        };
    void * userdata = &ctx->adapter;
    ctx->instance.WaitAny(
        ctx->instance.RequestAdapter(&options, wgpu::CallbackMode::AllowSpontaneous, callback, userdata), UINT64_MAX);
    GGML_ASSERT(ctx->adapter != nullptr);

    ctx->adapter.GetLimits(&ctx->limits);

    wgpu::AdapterInfo info{};
    ctx->adapter.GetInfo(&info);

    // Initialize device
    std::vector<wgpu::FeatureName> required_features = { wgpu::FeatureName::ShaderF16,
                                                         wgpu::FeatureName::ImplicitDeviceSynchronization };
    wgpu::DeviceDescriptor         dev_desc;
    dev_desc.requiredLimits       = &ctx->limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR(
                "ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason), std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR(
                "ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason), std::string(message).c_str());
        });
    ctx->instance.WaitAny(ctx->adapter.RequestDevice(
                              &dev_desc,
                              wgpu::CallbackMode::AllowSpontaneous,
                              [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                                  if (status != wgpu::RequestDeviceStatus::Success) {
                                      GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                                      return;
                                  }
                                  ctx->device = std::move(device);
                              }),
                          UINT64_MAX);
    GGML_ASSERT(ctx->device != nullptr);

    // Initialize (compute) queue
    ctx->queue = ctx->device.GetQueue();

    // Create buffer pool for shader parameters
    ctx->param_buf_pool.init(ctx->device,
                             WEBGPU_NUM_PARAM_BUFS,
                             WEBGPU_PARAMS_BUF_SIZE_BYTES,
                             wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                             wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite);
    ctx->set_rows_error_buf_pool.init(ctx->device,
                                      WEBGPU_NUM_SET_ROWS_ERROR_BUFS,
                                      WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                                      wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage,
                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);

    ggml_webgpu_init_memset_pipeline(ctx);
    ggml_webgpu_init_mul_mat_pipeline(ctx);
    ggml_webgpu_init_set_rows_pipeline(ctx);
    ggml_webgpu_init_cpy_pipeline(ctx);

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(ctx->device,
                              ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                              "debug_host_buf");
    ggml_webgpu_create_buffer(ctx->device,
                              ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                              "debug_dev_buf");
#endif

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.webgpu_ctx  = ctx;
    device_ctx.device_name = GGML_WEBGPU_NAME;
    device_ctx.device_desc = info.description;

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID,
        std::string(info.vendor).c_str(),
        std::string(info.architecture).c_str(),
        info.deviceID,
        std::string(info.device).c_str(),
        std::string(info.description).c_str());

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

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    webgpu_context webgpu_ctx = std::make_shared<webgpu_context_struct>();

    static ggml_backend_webgpu_reg_context ctx;
    ctx.webgpu_ctx   = webgpu_ctx;
    ctx.name         = GGML_WEBGPU_NAME;
    ctx.device_count = 1;

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.requiredFeatureCount                 = instance_features.size();
    webgpu_ctx->instance                                     = wgpu::CreateInstance(&instance_descriptor);
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

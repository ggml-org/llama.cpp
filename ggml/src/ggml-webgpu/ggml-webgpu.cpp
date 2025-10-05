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
#define WEBGPU_WAIT_ANY_BATCH_SIZE           64
#define WEBGPU_MUL_MAT_WG_SIZE               64
#define WEBGPU_NUM_PARAM_BUFS                100
#define WEBGPU_PARAMS_BUF_SIZE_BYTES         128  // enough for 32 parameters
#define WEBGPU_NUM_SET_ROWS_ERROR_BUFS       32
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES 4
#define WEBGPU_STORAGE_BUF_BINDING_MULT      4  // a storage buffer binding size must be a multiple of 4

// For operations which process a row in parallel, this seems like a reasonable default
#define WEBGPU_ROW_SPLIT_WG_SIZE 64

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

    // Separate this out from limits since on some Metal systems, the limit returned by
    // querying the limits is higher than the actual allowed maximum.
    uint32_t max_wg_size_x;

    std::recursive_mutex mutex;

    webgpu_buf_pool param_buf_pool;
    webgpu_buf_pool set_rows_error_buf_pool;

    wgpu::ComputePipeline memset_pipeline;
    wgpu::ComputePipeline mul_mat_pipeline[30][2];
    wgpu::ComputePipeline set_rows_pipeline;
    wgpu::ComputePipeline get_rows_pipeline[30];
    wgpu::ComputePipeline get_rows_f32_no_vec_pipeline;
    wgpu::ComputePipeline cpy_pipeline[2][2];          // src type, dst type
    wgpu::ComputePipeline add_pipeline[2][2];          // type, inplace
    wgpu::ComputePipeline sub_pipeline[2][2];          // type, inplace
    wgpu::ComputePipeline mul_pipeline[2][2];          // type, inplace
    wgpu::ComputePipeline div_pipeline[2][2];          // type, inplace
    wgpu::ComputePipeline rms_norm_pipeline[2];        // inplace
    wgpu::ComputePipeline rope_pipeline[2][2][2];      // type, ff, inplace
    wgpu::ComputePipeline glu_pipeline[7][2][2];       // glu-op, type, split
    wgpu::ComputePipeline scale_pipeline[2];           // inplace
    wgpu::ComputePipeline soft_max_pipeline[3][2][2];  // (no_mask, f32_mask, f16_mask), has_sink, inplace

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
        ctx->instance.WaitAny(
            ctx->queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                           [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                                               if (status != wgpu::QueueWorkDoneStatus::Success) {
                                                   GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n",
                                                                  std::string(message).c_str());
                                               }
                                           }),
            UINT64_MAX);
    } else {
        // WebGPU implementations may limit the number of futures that can be waited on at once,
        // so wait in batches (64 is what Dawn supports).
        for (size_t i = 0; i < ctx->callback_futures.size(); i += WEBGPU_WAIT_ANY_BATCH_SIZE) {
            size_t end = std::min(i + WEBGPU_WAIT_ANY_BATCH_SIZE, ctx->callback_futures.size());
            ctx->instance.WaitAny(end - i, ctx->callback_futures.data() + i, UINT64_MAX);
        }
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
            wgpu::MapMode::Read, 0, error_bufs.host_buf.GetSize(), wgpu::CallbackMode::AllowSpontaneous,
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
    ctx->instance.WaitAny(buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
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
                                                  const char *                      bind_group_label = nullptr,
                                                  bool                              submit_and_wait  = false) {
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
    if (bind_group_label) {
        bind_group_desc.label = bind_group_label;
    }
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
            ggml_backend_webgpu_wait_on_submission(ctx);
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
    size_t   bytes_per_wg = ctx->max_wg_size_x * ctx->memset_bytes_per_thread;
    uint32_t wg_x         = ((size + 3) + bytes_per_wg - 1) / bytes_per_wg;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->memset_pipeline, params, entries, wg_x, "MEMSET", true);
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

// Used to determine if two tensors are the same for in-place operations
static bool ggml_webgpu_tensor_equal(ggml_tensor * a, ggml_tensor * b) {
    return (ggml_webgpu_tensor_buf(a).Get() == ggml_webgpu_tensor_buf(b).Get()) &&
           (ggml_webgpu_tensor_offset(a) == ggml_webgpu_tensor_offset(b));
}

static void ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)), (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)), (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

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

    size_t   max_wg_size = ctx->max_wg_size_x;
    uint32_t wg_x        = (ne + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->cpy_pipeline[src->type][dst->type], params, entries, wg_x,
                                          ggml_op_name(dst->op));
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

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)), (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)), (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

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

    size_t   max_wg_size = ctx->max_wg_size_x;
    uint32_t wg_x        = (src->ne[1] * src->ne[2] * src->ne[3] + max_wg_size - 1) / max_wg_size;

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    ctx->staged_set_row_error_bufs.push_back(error_bufs);

    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->set_rows_pipeline, params, entries, wg_x, ggml_op_name(dst->op));
}

static void ggml_webgpu_get_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)), (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)), (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Shape of dst
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

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
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) }
    };

    size_t   max_wg_size = ctx->max_wg_size_x;
    uint32_t wg_x        = (dst->ne[1] * dst->ne[2] * dst->ne[3] + max_wg_size - 1) / max_wg_size;

    wgpu::ComputePipeline pipeline = ctx->get_rows_pipeline[src->type];
    if (src->type == GGML_TYPE_F32 && dst->ne[0] % 4 != 0) {
        pipeline = ctx->get_rows_f32_no_vec_pipeline;
    }
    ggml_backend_webgpu_build_and_enqueue(ctx, pipeline, params, entries, wg_x, ggml_op_name(dst->op));
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
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->mul_mat_pipeline[src0->type][src1->type], params, entries, wg_x,
                                          ggml_op_name(dst->op));
}

static void ggml_webgpu_binary_op(webgpu_context &        ctx,
                                  ggml_tensor *           src0,
                                  ggml_tensor *           src1,
                                  ggml_tensor *           dst,
                                  wgpu::ComputePipeline & pipeline,
                                  bool                    inplace) {
    std::vector<uint32_t> params = {
        (uint32_t) ggml_nelements(dst),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src0),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src0),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src0) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(src1),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src1),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src1) }
    };
    if (!inplace) {
        entries.push_back({ .binding = 2,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    size_t   max_wg_size = ctx->max_wg_size_x;
    uint32_t wg_x        = (ggml_nelements(dst) + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, pipeline, params, entries, wg_x, ggml_op_name(dst->op));
}

static void ggml_webgpu_rms_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    int inplace = ggml_webgpu_tensor_equal(src, dst);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        *(uint32_t *) dst->op_params  // epsilon, treated as f32 in the shader
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) }
    };
    if (!inplace) {
        entries.push_back({ .binding = 1,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->rms_norm_pipeline[inplace], params, entries, ggml_nrows(src),
                                          ggml_op_name(dst->op));
}

static void ggml_webgpu_rope(webgpu_context & ctx,
                             ggml_tensor *    src0,
                             ggml_tensor *    src1,
                             ggml_tensor *    src2,
                             ggml_tensor *    dst) {
    const int inplace         = ggml_webgpu_tensor_equal(src0, dst);
    const int has_freq_factor = (src2 != nullptr);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));

    float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        *(uint32_t *) &theta_scale,
        *(uint32_t *) &attn_factor,
        *(uint32_t *) &freq_scale,
        *(uint32_t *) &ext_factor,
        *(uint32_t *) &corr_dims[0],
        *(uint32_t *) &corr_dims[1],
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3]
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src0),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src0),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src0) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(src1),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src1),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src1) }
    };
    uint32_t dst_binding = 2;
    if (has_freq_factor) {
        dst_binding = 3;
        entries.push_back({ .binding = 2,
                            .buffer  = ggml_webgpu_tensor_buf(src2),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, src2),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, src2) });
    }
    if (!inplace) {
        entries.push_back({ .binding = dst_binding,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    wgpu::ComputePipeline pipeline    = ctx->rope_pipeline[dst->type][has_freq_factor][inplace];
    size_t                max_wg_size = ctx->max_wg_size_x;
    uint32_t              wg_x        = (ggml_nelements(src0) / 2 + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, pipeline, params, entries, wg_x, ggml_op_name(dst->op));
}

static void ggml_webgpu_glu(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const int split = (src1 != nullptr);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],  // swapped
        *(uint32_t *) &dst->op_params[2],            // alpha, for swiglu_oai
        *(uint32_t *) &dst->op_params[3],            // limit, for swiglu_oai
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src0),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src0),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src0) },
    };
    uint32_t dst_binding = 1;
    if (split) {
        dst_binding = 2;
        entries.push_back({ .binding = 1,
                            .buffer  = ggml_webgpu_tensor_buf(src1),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, src1),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, src1) });
    }
    entries.push_back({ .binding = dst_binding,
                        .buffer  = ggml_webgpu_tensor_buf(dst),
                        .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                        .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });

    wgpu::ComputePipeline pipeline    = ctx->glu_pipeline[ggml_get_glu_op(dst)][dst->type][split];
    size_t                max_wg_size = ctx->max_wg_size_x;
    uint32_t              wg_x        = (ggml_nelements(dst) + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, pipeline, params, entries, wg_x, ggml_op_name(dst->op));
}

static void ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    int inplace = ggml_webgpu_tensor_equal(src, dst);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        *(uint32_t *) dst->op_params,     // scale
        *(uint32_t *) &dst->op_params[1]  // bias
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) }
    };
    if (!inplace) {
        entries.push_back({ .binding = 1,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    size_t   max_wg_size = ctx->max_wg_size_x;
    uint32_t wg_x        = (ggml_nelements(dst) + max_wg_size - 1) / max_wg_size;
    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->scale_pipeline[inplace], params, entries, wg_x,
                                          ggml_op_name(dst->op));
}

static void ggml_webgpu_soft_max(webgpu_context & ctx,
                                 ggml_tensor *    src0,
                                 ggml_tensor *    src1,
                                 ggml_tensor *    src2,
                                 ggml_tensor *    dst) {
    const int inplace   = ggml_webgpu_tensor_equal(src0, dst);
    const int mask_type = (src1 != nullptr) ? src1->type : 2;  // use 2 for no mask here
    const int has_sink  = (src2 != nullptr);
    float     max_bias;
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    float n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
    float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        mask_type < 2 ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        mask_type < 2 ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0,
        mask_type < 2 ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) : 0,
        mask_type < 2 ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        mask_type < 2 ? (uint32_t) src1->ne[2] : 0,
        mask_type < 2 ? (uint32_t) src1->ne[3] : 0,
        *(uint32_t *) dst->op_params,  // scale
        *(uint32_t *) &max_bias,
        *(uint32_t *) &n_head_log2,
        *(uint32_t *) &m0,
        *(uint32_t *) &m1
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src0),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src0),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src0) }
    };
    uint32_t binding_num = 1;
    if (mask_type < 2) {
        entries.push_back({ .binding = binding_num,
                            .buffer  = ggml_webgpu_tensor_buf(src1),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, src1),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, src1) });
        binding_num++;
    }
    if (has_sink) {
        entries.push_back({ .binding = binding_num,
                            .buffer  = ggml_webgpu_tensor_buf(src2),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, src2),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, src2) });
        binding_num++;
    }
    if (!inplace) {
        entries.push_back({ .binding = binding_num,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    ggml_backend_webgpu_build_and_enqueue(ctx, ctx->soft_max_pipeline[mask_type][has_sink][inplace], params, entries,
                                          ggml_nrows(dst), ggml_op_name(dst->op));
}

// Returns true if node has enqueued work into the queue, false otherwise
static bool ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return false;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode_node(" << node << ", " << ggml_op_name(node->op) << ")");

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];

    switch (node->op) {
            // no-ops
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return false;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            ggml_webgpu_cpy(ctx, src0, node);
            break;
        case GGML_OP_SET_ROWS:
            ggml_webgpu_set_rows(ctx, src0, src1, node);
            break;
        case GGML_OP_GET_ROWS:
            ggml_webgpu_get_rows(ctx, src0, src1, node);
            break;
        case GGML_OP_MUL_MAT:
            ggml_webgpu_mul_mat(ctx, src0, src1, node);
            break;
        case GGML_OP_ADD:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->add_pipeline[node->type][inplace], inplace);
                break;
            }
        case GGML_OP_SUB:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->sub_pipeline[node->type][inplace], inplace);
                break;
            }
        case GGML_OP_MUL:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->mul_pipeline[node->type][inplace], inplace);
                break;
            }
        case GGML_OP_DIV:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->div_pipeline[node->type][inplace], inplace);
                break;
            }
        case GGML_OP_RMS_NORM:
            ggml_webgpu_rms_norm(ctx, src0, node);
            break;
        case GGML_OP_ROPE:
            ggml_webgpu_rope(ctx, src0, src1, src2, node);
            break;
        case GGML_OP_GLU:
            ggml_webgpu_glu(ctx, src0, src1, node);
            break;
        case GGML_OP_SCALE:
            ggml_webgpu_scale(ctx, src0, node);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
            break;
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
    /* .graph_optimize          = */ NULL,
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
        ggml_backend_webgpu_buffer_memset(webgpu_ctx, buf_ctx->buffer, val32, total_offset + (size - remaining_size),
                                          remaining_size);
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
    ggml_webgpu_create_buffer(ctx->webgpu_ctx->device, buf,
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

// Workgroup size is a common constant
static std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = wg_size;
    return constants;
}

static void ggml_webgpu_init_memset_pipeline(webgpu_context & webgpu_ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_wg_size = webgpu_ctx->max_wg_size_x;
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
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F32][GGML_TYPE_F32],
                                wgsl_mul_mat_f32_f32, "mul_mat_f32_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F16][GGML_TYPE_F16],
                                wgsl_mul_mat_f16_f16, "mul_mat_f16_f16");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_F16][GGML_TYPE_F32],
                                wgsl_mul_mat_f16_f32, "mul_mat_f16_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_0_f32, "mul_mat_q4_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_1][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_1_f32, "mul_mat_q4_1_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_0_f32, "mul_mat_q5_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_1][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_1_f32, "mul_mat_q5_1_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q8_0][GGML_TYPE_F32],
                                wgsl_mul_mat_q8_0_f32, "mul_mat_q8_0_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q2_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q2_k_f32, "mul_mat_q2_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q3_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q3_k_f32, "mul_mat_q3_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q4_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q4_k_f32, "mul_mat_q4_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q5_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q5_k_f32, "mul_mat_q5_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_Q6_K][GGML_TYPE_F32],
                                wgsl_mul_mat_q6_k_f32, "mul_mat_q6_k_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_XXS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_xxs_f32, "mul_mat_iq2_xxs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_XS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_xs_f32, "mul_mat_iq2_xs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ2_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq2_s_f32, "mul_mat_iq2_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ3_XXS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq3_xxs_f32, "mul_mat_iq3_xxs_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ3_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq3_s_f32, "mul_mat_iq3_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ1_S][GGML_TYPE_F32],
                                wgsl_mul_mat_iq1_s_f32, "mul_mat_iq1_s_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ1_M][GGML_TYPE_F32],
                                wgsl_mul_mat_iq1_m_f32, "mul_mat_iq1_m_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ4_NL][GGML_TYPE_F32],
                                wgsl_mul_mat_iq4_nl_f32, "mul_mat_iq4_nl_f32");
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_mat_pipeline[GGML_TYPE_IQ4_XS][GGML_TYPE_F32],
                                wgsl_mul_mat_iq4_xs_f32, "mul_mat_iq4_xs_f32");
}

static void ggml_webgpu_init_set_rows_pipeline(webgpu_context & webgpu_ctx) {
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->set_rows_pipeline, wgsl_set_rows, "set_rows",
                                ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x));
}

static void ggml_webgpu_init_get_rows_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_F32], wgsl_get_rows_f32_vec,
                                "get_rows_f32_vec", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_f32_no_vec_pipeline, wgsl_get_rows_f32,
                                "get_rows_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_F16], wgsl_get_rows_f16,
                                "get_rows_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_I32], wgsl_get_rows_i32,
                                "get_rows_i32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q4_0], wgsl_get_rows_q4_0,
                                "get_rows_q4_0", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q4_1], wgsl_get_rows_q4_1,
                                "get_rows_q4_1", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q5_0], wgsl_get_rows_q5_0,
                                "get_rows_q5_0", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q5_1], wgsl_get_rows_q5_1,
                                "get_rows_q5_1", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q8_0], wgsl_get_rows_q8_0,
                                "get_rows_q8_0", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q2_K], wgsl_get_rows_q2_k,
                                "get_rows_q2_k", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q3_K], wgsl_get_rows_q3_k,
                                "get_rows_q3_k", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q4_K], wgsl_get_rows_q4_k,
                                "get_rows_q4_k", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q5_K], wgsl_get_rows_q5_k,
                                "get_rows_q5_k", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_Q6_K], wgsl_get_rows_q6_k,
                                "get_rows_q6_k", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ2_XXS],
                                wgsl_get_rows_iq2_xxs, "get_rows_iq2_xxs", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ2_XS],
                                wgsl_get_rows_iq2_xs, "get_rows_iq2_xs", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ2_S], wgsl_get_rows_iq2_s,
                                "get_rows_iq2_s", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ3_XXS],
                                wgsl_get_rows_iq3_xxs, "get_rows_iq3_xxs", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ3_S], wgsl_get_rows_iq3_s,
                                "get_rows_iq3_s", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ1_S], wgsl_get_rows_iq1_s,
                                "get_rows_iq1_s", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ1_M], wgsl_get_rows_iq1_m,
                                "get_rows_iq1_m", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ4_NL],
                                wgsl_get_rows_iq4_nl, "get_rows_iq4_nl", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->get_rows_pipeline[GGML_TYPE_IQ4_XS],
                                wgsl_get_rows_iq4_xs, "get_rows_iq4_xs", constants);
}

static void ggml_webgpu_init_cpy_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline[GGML_TYPE_F32][GGML_TYPE_F32],
                                wgsl_cpy_f32_f32, "cpy_f32_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline[GGML_TYPE_F32][GGML_TYPE_F16],
                                wgsl_cpy_f32_f16, "cpy_f32_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline[GGML_TYPE_F16][GGML_TYPE_F32],
                                wgsl_cpy_f16_f32, "cpy_f16_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->cpy_pipeline[GGML_TYPE_F16][GGML_TYPE_F16],
                                wgsl_cpy_f16_f16, "cpy_f16_f16", constants);
}

static void ggml_webgpu_init_add_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->add_pipeline[GGML_TYPE_F32][0], wgsl_add_f32, "add_f32",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->add_pipeline[GGML_TYPE_F16][0], wgsl_add_f16, "add_f16",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->add_pipeline[GGML_TYPE_F32][1], wgsl_add_f32_inplace,
                                "add_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->add_pipeline[GGML_TYPE_F16][1], wgsl_add_f16_inplace,
                                "add_f16_inplace", constants);
}

static void ggml_webgpu_init_sub_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->sub_pipeline[GGML_TYPE_F32][0], wgsl_sub_f32, "sub_f32",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->sub_pipeline[GGML_TYPE_F16][0], wgsl_sub_f16, "sub_f16",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->sub_pipeline[GGML_TYPE_F32][1], wgsl_sub_f32_inplace,
                                "sub_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->sub_pipeline[GGML_TYPE_F16][1], wgsl_sub_f16_inplace,
                                "sub_f16_inplace", constants);
}

static void ggml_webgpu_init_mul_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_pipeline[GGML_TYPE_F32][0], wgsl_mul_f32, "mul_f32",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_pipeline[GGML_TYPE_F16][0], wgsl_mul_f16, "mul_f16",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_pipeline[GGML_TYPE_F32][1], wgsl_mul_f32_inplace,
                                "mul_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->mul_pipeline[GGML_TYPE_F16][1], wgsl_mul_f16_inplace,
                                "mul_f16_inplace", constants);
}

static void ggml_webgpu_init_div_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->div_pipeline[GGML_TYPE_F32][0], wgsl_div_f32, "div_f32",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->div_pipeline[GGML_TYPE_F16][0], wgsl_div_f16, "div_f16",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->div_pipeline[GGML_TYPE_F32][1], wgsl_div_f32_inplace,
                                "div_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->div_pipeline[GGML_TYPE_F16][1], wgsl_div_f16_inplace,
                                "div_f16_inplace", constants);
}

static void ggml_webgpu_init_rms_norm_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_ROW_SPLIT_WG_SIZE);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rms_norm_pipeline[0], wgsl_rms_norm, "rms_norm",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rms_norm_pipeline[1], wgsl_rms_norm_inplace,
                                "rms_norm_inplace", constants);
}

static void ggml_webgpu_init_rope_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F32][0][0], wgsl_rope_f32,
                                "rope_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F32][0][1],
                                wgsl_rope_f32_inplace, "rope_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F32][1][0], wgsl_rope_f32_ff,
                                "rope_f32_ff", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F32][1][1],
                                wgsl_rope_f32_ff_inplace, "rope_f32_ff_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F16][0][0], wgsl_rope_f16,
                                "rope_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F16][0][1],
                                wgsl_rope_f16_inplace, "rope_f16_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F16][1][0], wgsl_rope_f16_ff,
                                "rope_f16_ff", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->rope_pipeline[GGML_TYPE_F16][1][1],
                                wgsl_rope_f16_ff_inplace, "rope_f16_ff_inplace", constants);
}

static void ggml_webgpu_init_glu_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    // reglu
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_REGLU][GGML_TYPE_F32][0],
                                wgsl_reglu_f32, "reglu_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_REGLU][GGML_TYPE_F16][0],
                                wgsl_reglu_f16, "reglu_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_REGLU][GGML_TYPE_F32][1],
                                wgsl_reglu_f32_split, "reglu_f32_split", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_REGLU][GGML_TYPE_F16][1],
                                wgsl_reglu_f16_split, "reglu_f16_split", constants);
    // geglu
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU][GGML_TYPE_F32][0],
                                wgsl_geglu_f32, "geglu_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU][GGML_TYPE_F16][0],
                                wgsl_geglu_f16, "geglu_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU][GGML_TYPE_F32][1],
                                wgsl_geglu_f32_split, "geglu_f32_split", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU][GGML_TYPE_F16][1],
                                wgsl_geglu_f16_split, "geglu_f16_split", constants);
    // swiglu
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU][GGML_TYPE_F32][0],
                                wgsl_swiglu_f32, "swiglu_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU][GGML_TYPE_F16][0],
                                wgsl_swiglu_f16, "swiglu_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU][GGML_TYPE_F32][1],
                                wgsl_swiglu_f32_split, "swiglu_f32_split", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU][GGML_TYPE_F16][1],
                                wgsl_swiglu_f16_split, "swiglu_f16_split", constants);
    // swiglu_oai
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU_OAI][GGML_TYPE_F32][0],
                                wgsl_swiglu_oai_f32, "swiglu_oai_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_SWIGLU_OAI][GGML_TYPE_F32][1],
                                wgsl_swiglu_oai_f32_split, "swiglu_oai_f32_split", constants);
    // geglu_erf
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F32][0],
                                wgsl_geglu_erf_f32, "geglu_erf_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F16][0],
                                wgsl_geglu_erf_f16, "geglu_erf_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F32][1],
                                wgsl_geglu_erf_f32_split, "geglu_erf_f32_split", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F16][1],
                                wgsl_geglu_erf_f16_split, "geglu_erf_f16_split", constants);
    // geglu_quick
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F32][0],
                                wgsl_geglu_quick_f32, "geglu_quick_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F16][0],
                                wgsl_geglu_quick_f16, "geglu_quick_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F32][1],
                                wgsl_geglu_quick_f32_split, "geglu_quick_f32_split", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->glu_pipeline[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F16][1],
                                wgsl_geglu_quick_f16_split, "geglu_quick_f16_split", constants);
}

static void ggml_webgpu_init_scale_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(webgpu_ctx->max_wg_size_x);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->scale_pipeline[0], wgsl_scale_f32, "scale_f32",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->scale_pipeline[1], wgsl_scale_f32_inplace,
                                "scale_f32_inplace", constants);
}

static void ggml_webgpu_init_soft_max_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_ROW_SPLIT_WG_SIZE);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[2][0][0], wgsl_soft_max_f32,
                                "soft_max_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[2][0][1], wgsl_soft_max_f32_inplace,
                                "soft_max_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[2][1][0], wgsl_soft_max_f32_sink,
                                "soft_max_f32_sink", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[2][1][1],
                                wgsl_soft_max_f32_sink_inplace, "soft_max_f32_sink_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[0][0][0], wgsl_soft_max_f32_mask_f32,
                                "soft_max_f32_mask_f32", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[0][0][1],
                                wgsl_soft_max_f32_mask_f32_inplace, "soft_max_f32_mask_f32_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[1][0][0], wgsl_soft_max_f32_mask_f16,
                                "soft_max_f32_mask_f16", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[1][0][1],
                                wgsl_soft_max_f32_mask_f16_inplace, "soft_max_f32_mask_f16_inplace", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[0][1][0],
                                wgsl_soft_max_f32_mask_f32_sink, "soft_max_f32_mask_f32_sink", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[0][1][1],
                                wgsl_soft_max_f32_mask_f32_sink_inplace, "soft_max_f32_mask_f32_sink_inplace",
                                constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[1][1][0],
                                wgsl_soft_max_f32_mask_f16_sink, "soft_max_f32_mask_f16_sink", constants);
    ggml_webgpu_create_pipeline(webgpu_ctx->device, webgpu_ctx->soft_max_pipeline[1][1][1],
                                wgsl_soft_max_f32_mask_f16_sink_inplace, "soft_max_f32_mask_f16_sink_inplace",
                                constants);
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

static bool ggml_webgpu_supported_qtype(ggml_type type) {
    switch (type) {
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
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    webgpu_context webgpu_ctx = ctx->webgpu_ctx;

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    if (ggml_nbytes(op) > webgpu_ctx->limits.maxStorageBufferBindingSize ||
        (src0 != nullptr && ggml_nbytes(src0) > webgpu_ctx->limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr && ggml_nbytes(src1) > webgpu_ctx->limits.maxStorageBufferBindingSize)) {
        return false;
    }

    bool supports_op = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            supports_op = true;
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        case GGML_OP_SET_ROWS:
            supports_op = (op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I64);
            break;
        case GGML_OP_GET_ROWS:
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                switch (src1->type) {
                    case GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        break;
                    case GGML_TYPE_F32:
                        switch (src0->type) {
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
                                supports_op = true;
                                break;
                            default:
                                break;
                        }
                    default:
                        break;
                }
                break;
            }
        case GGML_OP_RMS_NORM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    break;
                default:
                    break;
            }
            break;
        case GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        default:
            break;
    }
    if (ggml_nbytes(op) > webgpu_ctx->limits.maxStorageBufferBindingSize ||
        (src0 != nullptr && ggml_nbytes(src0) > webgpu_ctx->limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr && ggml_nbytes(src1) > webgpu_ctx->limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr && ggml_nbytes(src2) > webgpu_ctx->limits.maxStorageBufferBindingSize)) {
        supports_op = false;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    if (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } else {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    return supports_op;
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
    ctx->instance.WaitAny(ctx->instance.RequestAdapter(
                              &options, wgpu::CallbackMode::AllowSpontaneous,
                              [&ctx](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char * message) {
                                  if (status != wgpu::RequestAdapterStatus::Success) {
                                      GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                                      return;
                                  }
                                  ctx->adapter = std::move(adapter);
                              }),
                          UINT64_MAX);
    GGML_ASSERT(ctx->adapter != nullptr);

    ctx->adapter.GetLimits(&ctx->limits);
    ctx->max_wg_size_x = 288;  // default value

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
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });
    ctx->instance.WaitAny(ctx->adapter.RequestDevice(
                              &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
                              [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                                  if (status != wgpu::RequestDeviceStatus::Success) {
                                      GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n",
                                                     std::string(message).c_str());
                                      return;
                                  }
                                  ctx->device = std::move(device);
                              }),
                          UINT64_MAX);
    GGML_ASSERT(ctx->device != nullptr);

    // Initialize (compute) queue
    ctx->queue = ctx->device.GetQueue();

    // Create buffer pool for shader parameters
    ctx->param_buf_pool.init(ctx->device, WEBGPU_NUM_PARAM_BUFS, WEBGPU_PARAMS_BUF_SIZE_BYTES,
                             wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                             wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite);
    ctx->set_rows_error_buf_pool.init(ctx->device, WEBGPU_NUM_SET_ROWS_ERROR_BUFS, WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                                      wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage,
                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);

    ggml_webgpu_init_memset_pipeline(ctx);
    ggml_webgpu_init_mul_mat_pipeline(ctx);
    ggml_webgpu_init_set_rows_pipeline(ctx);
    ggml_webgpu_init_get_rows_pipeline(ctx);
    ggml_webgpu_init_cpy_pipeline(ctx);
    ggml_webgpu_init_add_pipeline(ctx);
    ggml_webgpu_init_sub_pipeline(ctx);
    ggml_webgpu_init_mul_pipeline(ctx);
    ggml_webgpu_init_div_pipeline(ctx);
    ggml_webgpu_init_rms_norm_pipeline(ctx);
    ggml_webgpu_init_rope_pipeline(ctx);
    ggml_webgpu_init_glu_pipeline(ctx);
    ggml_webgpu_init_scale_pipeline(ctx);
    ggml_webgpu_init_soft_max_pipeline(ctx);

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(ctx->device, ctx->debug_host_buf, WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(ctx->device, ctx->debug_dev_buf, WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.webgpu_ctx  = ctx;
    device_ctx.device_name = GGML_WEBGPU_NAME;
    device_ctx.device_desc = info.description;

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());

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

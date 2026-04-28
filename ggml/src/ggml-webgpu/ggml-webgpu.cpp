/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#include "ggml-webgpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-webgpu-shader-lib.hpp"
#include "ggml.h"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#if defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
static inline void compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, max_per_dim));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

static inline uint32_t ggml_webgpu_u32_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(msg)  std::cout << msg << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#else
#    define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id) auto cpu_total_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)                                                         \
        auto   cpu_total_end_##id = std::chrono::high_resolution_clock::now();                            \
        double cpu_total_time_##id =                                                                      \
            std::chrono::duration<double, std::milli>(cpu_total_end_##id - cpu_total_start_##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_total_time_##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        double cpu_detail_time_##id =                                                                       \
            std::chrono::duration<double, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#else
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_COUNT        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES (WEBGPU_MAX_PROFILE_QUERY_COUNT * sizeof(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64u
#define WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_SIZE_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

// For operations which process a row in parallel, this seems like a reasonable
// default
#define WEBGPU_ROW_SPLIT_WG_SIZE 64

// Track https://github.com/gpuweb/gpuweb/issues/5315 for fixes to
// implementations so this can be removed, necessary only for get_rows right now
#define WEBGPU_MAX_WG_SIZE 288

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
struct webgpu_param_arena {
    wgpu::Buffer buffer;
    size_t       slot_stride = 0;
    size_t       slot_size   = 0;
    uint32_t     slot_count  = 0;
    uint32_t     next_slot   = 0;

    void init(wgpu::Device device, size_t slot_size, uint32_t slot_count, size_t alignment) {
        this->slot_stride = ROUNDUP_POW2(slot_size, alignment);
        this->slot_size   = slot_size;
        this->slot_count  = slot_count;
        this->next_slot   = 0;

        ggml_webgpu_create_buffer(device, buffer, this->slot_stride * slot_count,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_size);
        if (next_slot >= slot_count) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        return slot_stride * next_slot++;
    }

    void reset() { next_slot = 0; }

    void cleanup() {
        if (buffer) {
            buffer.Destroy();
            buffer = nullptr;
        }
    }

    ~webgpu_param_arena() { this->cleanup(); }
};

struct webgpu_encoded_op {
    uint32_t num_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

struct webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

struct webgpu_capabilities {
    wgpu::Limits limits;
    bool         supports_subgroups       = false;
    bool         supports_subgroup_matrix = false;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_size     = 0;
    uint32_t max_subgroup_size = 0;
    size_t   memset_bytes_per_thread;
};

// Stores global webgpu members
struct webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_size = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::Buffer         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::Buffer    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, double> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, double> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        if (this->get_tensor_staging_buf) {
            this->get_tensor_staging_buf.Destroy();
            this->get_tensor_staging_buf = nullptr;
        }
        if (this->memset_params_buf) {
            this->memset_params_buf.Destroy();
            this->memset_params_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        if (this->debug_host_buf) {
            this->debug_host_buf.Destroy();
            this->debug_host_buf = nullptr;
        }
        if (this->debug_dev_buf) {
            this->debug_dev_buf.Destroy();
            this->debug_dev_buf = nullptr;
        }
#endif
    }
};

typedef std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::Buffer             set_rows_dev_error_buf;
    wgpu::Buffer             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;

    size_t memset_bytes_per_thread;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, double> shader_gpu_time_ms;
    wgpu::Buffer                            profile_timestamp_dev_buf;
    wgpu::Buffer                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_count = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        if (this->profile_timestamp_host_buf) {
            this->profile_timestamp_host_buf.Destroy();
            this->profile_timestamp_host_buf = nullptr;
        }
        if (this->profile_timestamp_dev_buf) {
            this->profile_timestamp_dev_buf.Destroy();
            this->profile_timestamp_dev_buf = nullptr;
        }
        if (this->profile_timestamp_query_set) {
            this->profile_timestamp_query_set.Destroy();
            this->profile_timestamp_query_set = nullptr;
        }
#endif
        if (this->set_rows_host_error_buf) {
            this->set_rows_host_error_buf.Destroy();
            this->set_rows_host_error_buf = nullptr;
        }
        if (this->set_rows_dev_error_buf) {
            this->set_rows_dev_error_buf.Destroy();
            this->set_rows_dev_error_buf = nullptr;
        }
    }
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
struct ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_count;
    const char *          name;
};

// Per-device struct for the global logical device interface
struct ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
struct ggml_backend_webgpu_buffer_context {
    wgpu::Buffer          buffer;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_buffer_context(wgpu::Buffer buf, std::string lbl, webgpu_global_context global_ctx_) :
        buffer(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   const char *                             shader_code,
                                                   const char *                             label,
                                                   const std::vector<wgpu::ConstantEntry> & constants = {}) {
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
    return { device.CreateComputePipeline(&pipeline_desc), label };
}

static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      const char *      label) {
    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = false;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    return ctx->buffer;
}

static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & (ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & ~(ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ggml_tensor * t) {
    return ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(ctx, t), WEBGPU_STORAGE_BUF_BINDING_MULT);
}

struct ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    for (ggml_tensor * tensor : tensors) {
        const size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        const size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    return { merged_offset, merged_end - merged_offset };
}

static uint32_t ggml_webgpu_tensor_merged_element_offset(const ggml_tensor *                      tensor,
                                                         const ggml_webgpu_merged_binding_range & merged_range) {
    return (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
}

static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::Buffer buffer,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.buffer               = std::move(buffer);
    entry.offset               = offset;
    entry.size                 = size;
    return entry;
}

static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    return ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_size(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
static void ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  const char *     wait_name,
                                                  const char *     failure_name,
                                                  const char *     callback_message) {
    if (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    if (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    if (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<int>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
static uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    return UINT32_MAX;
}

static uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    return WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

static void ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [&callback_status, &callback_message](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                callback_status  = status;
                callback_message = std::string(message);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::QueueWorkDoneStatus::Success,
                                          "Queue wait", "Queue work", callback_message.c_str());
}

static void ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer &          buffer,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

static void ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                const wgpu::CommandBuffer commands,
                                                uint32_t &                num_inflight_batches) {
    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        num_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    num_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
static void ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    const float * debug_data = (const float *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_data[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

static webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         const std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            result = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    result.num_kernels = dispatches.size();

    for (size_t i = 0; i < dispatches.size(); i++) {
        const webgpu_dispatch_desc & dispatch     = dispatches[i];
        const size_t                 param_size   = dispatch.params.size() * sizeof(uint32_t);
        const size_t                 param_offset = ctx->param_arena.alloc_slot(param_size);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_num = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_num, ctx->param_arena.buffer, param_offset,
                                                            ctx->param_arena.slot_size));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    for (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer, param_offsets[i], dispatches[i].params.data(),
                                           dispatches[i].params.size() * sizeof(uint32_t));
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    for (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_count + 2 <= WEBGPU_MAX_PROFILE_QUERY_COUNT);
        const uint32_t query_begin = ctx->profile_timestamp_query_count++;
        const uint32_t query_end   = ctx->profile_timestamp_query_count++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginningOfPassWriteIndex   = query_begin;
        ts_writes.endOfPassWriteIndex         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        result.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#else
    for (size_t i = 0; i < dispatches.size(); i++) {
        ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
        ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
        ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
    }
#endif

    return result;
}

static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    return ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

static void ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                              wgpu::Buffer &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::vector<uint32_t>             params       = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries      = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.GetSize()) };
    size_t                            bytes_per_wg = WEBGPU_MAX_WG_SIZE * ctx->capabilities.memset_bytes_per_thread;
    uint32_t                          wg_x         = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.buffer               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entryCount = entries.size();
    bind_group_desc.entries    = entries.data();
    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();

    wgpu::CommandBuffer              command  = encoder.Finish();
    std::vector<wgpu::CommandBuffer> commands = { command };
    ctx->queue.Submit(commands.size(), commands.data());
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

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    double total_cpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    if (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    for (const auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    double total_gpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        total_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        double pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#if defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

static webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    const uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    const uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (((const int32_t *) dst->op_params)[3] / dst_type_size),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        1u,
        (uint32_t) (((const int32_t *) dst->op_params)[0] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[1] / dst_type_size),
        (uint32_t) (((const int32_t *) dst->op_params)[2] / dst_type_size),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_index = 0;
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_index++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index + 1, dst));

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        // Shapes
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        // Pad sizes
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 3),
        (uint32_t) ggml_get_op_params_i32(dst, 4),
        (uint32_t) ggml_get_op_params_i32(dst, 5),
        (uint32_t) ggml_get_op_params_i32(dst, 6),
        (uint32_t) ggml_get_op_params_i32(dst, 7),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
    const uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],

        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    const int32_t s0    = ggml_get_op_params_i32(dst, 0);
    const int32_t s1    = ggml_get_op_params_i32(dst, 1);
    const int32_t p0    = ggml_get_op_params_i32(dst, 2);
    const int32_t p1    = ggml_get_op_params_i32(dst, 3);
    const int32_t d0    = ggml_get_op_params_i32(dst, 4);
    const int32_t d1    = ggml_get_op_params_i32(dst, 5);
    const bool    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    const uint32_t KW = src0->ne[0];
    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    const uint32_t IW = src1->ne[0];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    const uint32_t OW = dst->ne[1];
    const uint32_t OH = is_2D ? dst->ne[2] : 1;

    const uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
    const uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
    const uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    const uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));

    const uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
    const uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
    const uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    const uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_size(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        si0,
        si1,
        si2,
        si3,
        so0,
        so1,
        so2,
        so3,

        KW,
        KH,
        IC,

        IW,
        IH,
        N,

        OW,
        OH,

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    const uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    const uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
    const uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    src3,
                                              ggml_tensor *    src4,
                                              ggml_tensor *    src5,
                                              ggml_tensor *    src6,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.src5                           = src5;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline pipeline    = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx);
    auto *          decisions   = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
    const bool      xbc_overlap = decisions->xbc_overlap;

    uint32_t offset_x        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    uint32_t offset_B        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_size(src4->type));
    uint32_t offset_C        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_size(src5->type));
    size_t   xbc_bind_offset = 0;
    size_t   xbc_bind_size   = 0;
    if (xbc_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        xbc_bind_offset = merged_range.offset;
        xbc_bind_size   = merged_range.size;
        offset_x        = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        offset_B        = ggml_webgpu_tensor_merged_element_offset(src4, merged_range);
        offset_C        = ggml_webgpu_tensor_merged_element_offset(src5, merged_range);
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        offset_x,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_size(src3->type)),
        offset_B,
        offset_C,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_size(src6->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_size(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_size(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
        (uint32_t) ggml_nelements(src1),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
    };
    if (xbc_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), xbc_bind_offset, xbc_bind_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src6));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, src6));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst));
    }

    const uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
                                                     ggml_tensor *    src0,
                                                     ggml_tensor *    src1,
                                                     ggml_tensor *    src2,
                                                     ggml_tensor *    src3,
                                                     ggml_tensor *    src4,
                                                     ggml_tensor *    src5,
                                                     ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    const uint32_t s_v      = (uint32_t) src2->ne[0];
    const uint32_t h        = (uint32_t) src2->ne[1];
    const uint32_t n_tokens = (uint32_t) src2->ne[2];
    const uint32_t n_seqs   = (uint32_t) src2->ne[3];
    const float    scale    = 1.0f / sqrtf((float) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, sizeof(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_size(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        scale_u32,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.get());

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
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    if (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.GetSize()));
    }

    uint32_t threads;
    if (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } else {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x = CEIL_DIV(threads, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, 1);
}

// Workgroup size is a common constant
static std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = wg_size;
    return constants;
}

static webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    const bool float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size                    = WEBGPU_MAX_WG_SIZE;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst) };

    uint32_t blocks_per_row = (uint32_t) (dst->ne[0] / (src->type == GGML_TYPE_F32 && dst->ne[0] % 4 == 0 ? 4 : 1));
    uint32_t total_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t total_threads  = float_parallel ? blocks_per_row * total_rows : total_rows;
    uint32_t wg_x           = CEIL_DIV(total_threads, decisions->wg_size);

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    bool is_vec = (dst->ne[1] == 1);

    // Determine if we should use fast path
    bool use_fast = false;
    switch (src1->type) {
        case GGML_TYPE_F16:
            use_fast = (src0->type == GGML_TYPE_F16);
            break;
        case GGML_TYPE_F32:
            // TODO: implement better mat-mat for k-quants, mat-vec for all k-quants except q6_K
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q1_0:
                    use_fast = true;
                    break;
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    use_fast = is_vec;
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;

    // Get or create pipeline
    webgpu_pipeline pipeline;

    if (use_fast && is_vec) {
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx);
    } else if (use_fast) {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx);
    } else {
        pipeline = ctx->shader_lib->get_mul_mat_legacy_pipeline(shader_lib_ctx);
    }

    // Build params
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) src0->ne[0],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[2],
        (uint32_t) src0->ne[3],
        (uint32_t) (src1->ne[2] / src0->ne[2]),
        (uint32_t) (src1->ne[3] / src0->ne[3])
    };

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    if (use_fast && is_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t total_wg      = output_groups * batches;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    } else if (use_fast) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        if (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } else {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_size_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_size_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t total_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    } else {  // legacy
        auto *   decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
        uint32_t wg_size   = decisions->wg_size;
        uint32_t total_wg  = CEIL_DIV(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3], wg_size);
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    const uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    const uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    const uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
    };

    const size_t dst_offset          = ggml_webgpu_tensor_offset(dst);
    const size_t gathered_buf_nbytes = src0->ne[2] * src1->ne[2] * sizeof(uint32_t);

    const size_t gathered_expert_used_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t gathered_tokens_align_offset =
        ROUNDUP_POW2(gathered_expert_used_align_offset + gathered_buf_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t gathered_count_ids_align_offset =
        ROUNDUP_POW2(gathered_tokens_align_offset + gathered_buf_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

    const size_t gathered_binding_size = ROUNDUP_POW2(gathered_buf_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t gathered_count_ids_binding_size =
        ROUNDUP_POW2(src0->ne[2] * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), gathered_expert_used_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), gathered_tokens_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), gathered_count_ids_align_offset,
                                          gathered_count_ids_binding_size),
    };

    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    const uint32_t gather_total_wg = param_n_expert;
    const uint32_t gather_wg_x     = std::min(gather_total_wg, max_wg_per_dim);
    const uint32_t gather_wg_y     = CEIL_DIV(gather_total_wg, gather_wg_x);

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { gather_wg_x, gather_wg_y }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), gathered_expert_used_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), gathered_tokens_align_offset,
                                          gathered_binding_size),
        ggml_webgpu_make_bind_group_entry(5, ggml_webgpu_tensor_buf(dst), gathered_count_ids_align_offset,
                                          gathered_count_ids_binding_size),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    uint32_t wg_m;

    uint32_t tile_m_s           = main_decisions->tile_m * main_decisions->wg_size_m;
    uint32_t tile_n_s           = main_decisions->tile_n * main_decisions->wg_size_n;
    wg_m                        = CEIL_DIV(dst->ne[0], tile_m_s);
    uint32_t total_gathered     = dst->ne[1] * dst->ne[2];
    uint32_t max_active_experts = std::min((uint32_t) src0->ne[2], total_gathered);
    uint32_t max_wg_n           = CEIL_DIV(total_gathered, tile_n_s) + max_active_experts;
    uint32_t total_wg           = wg_m * max_wg_n;

    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    float scale         = ggml_get_op_params_f32(dst, 0);
    float max_bias      = ggml_get_op_params_f32(dst, 1);
    float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    float n_head_log2 = float(1u << (uint32_t) floor(log2(Q->ne[2])));
    float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = Q;
    shader_lib_ctx.src1                           = K;
    shader_lib_ctx.src2                           = V;
    shader_lib_ctx.src3                           = mask;
    shader_lib_ctx.src4                           = sinks;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix       = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
    shader_lib_ctx.sg_mat_m           = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n           = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k           = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.max_subgroup_size  = ctx->global_ctx->capabilities.max_subgroup_size;
    webgpu_pipeline pipeline          = ctx->shader_lib->get_flash_attn_pipeline(
        shader_lib_ctx, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    auto *     decisions  = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    const int  has_mask   = (mask != nullptr);
    const int  has_sinks  = (sinks != nullptr);
    const bool kv_overlap = decisions->kv_overlap;

    uint32_t offset_k       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
    uint32_t offset_v       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
    size_t   kv_bind_offset = 0;
    size_t   kv_bind_size   = 0;
    if (kv_overlap) {
        const ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        kv_bind_offset                                      = merged_range.offset;
        kv_bind_size                                        = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        offset_k,
        offset_v,
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0,
        has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_size(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 3
        has_mask ? (uint32_t) (mask->nb[3] / ggml_type_size(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)

    };
    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    if (kv_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_index = kv_overlap ? 2u : 3u;
    if (has_mask) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
    }
    if (has_sinks) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, sinks));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));

    if (decisions->path != GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
        uint32_t wg_per_head = CEIL_DIV(Q->ne[1], decisions->q_tile);
        uint32_t wg_x        = wg_per_head * Q->ne[2] * Q->ne[3];  // wg per head * number of heads * number of batches
        return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
    }

    wgpu::Buffer blk_buf         = {};
    uint64_t     blk_size_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_count = 0;

    const uint32_t vec_nwg_cap = std::max(1u, std::min<uint32_t>(32u, ctx->global_ctx->capabilities.max_subgroup_size));
    uint32_t       nwg         = 1u;
    const uint64_t kv_span     = (uint64_t) std::max(1u, decisions->kv_tile);
    while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    nwg                           = std::min(nwg, vec_nwg_cap);
    const uint64_t nrows          = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    const bool     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_size_bytes  = 0;
    wgpu::Buffer tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_size   = 0;
    const size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    const size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    if (use_vec_reduce) {
        const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        const uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_data_elems;
        tmp_size_bytes =
            ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    } else {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_size_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    if (has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        const uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
        blk_batch_count             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        const uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
        blk_size_bytes              = ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        const ggml_webgpu_shader_lib_context blk_shader_ctx = shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_size(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_size_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_size_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = params;
    if (has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_size(ctx, Q)),
    };
    if (kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
    } else {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
    }
    uint32_t split_binding_index = kv_overlap ? 2u : 3u;
    if (has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_size(ctx, mask)));
    }
    if (has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_size(ctx, sinks)));
    }
    if (has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_index++, blk_buf, blk_entries[1].offset, blk_size_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_index++, tmp_buf, tmp_bind_offset, tmp_bind_size));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_size(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    if (use_vec_reduce) {
        const uint32_t reduce_wg_size = std::max(
            32u, std::min<uint32_t>(nwg * 32u, ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = shader_lib_ctx;
        reduce_shader_ctx.max_wg_size                    = reduce_wg_size;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_size_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_size(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    const uint64_t split_wg_total = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_total <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    if (has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_count }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_total, 1u }
    });
    if (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    const bool inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    if (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        switch (unary_op) {
            case GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    float alpha_n = ggml_get_op_params_f32(dst, 1);
                    float alpha_p = ggml_get_op_params_f32(dst, 2);
                    float beta    = ggml_get_op_params_f32(dst, 3);
                    float eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    break;
                }
            default:
                break;
        }
    } else if (dst->op == GGML_OP_CLAMP) {
        float clamp_min = ggml_get_op_params_f32(dst, 0);
        float clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } else if (dst->op == GGML_OP_FILL) {
        float fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_binary_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_binary_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
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

    std::vector<wgpu::BindGroupEntry> entries;

    if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        if (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        dim,
        (uint32_t) src0->ne[dim]
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_size(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    if (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } else if (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } else {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_size(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_size(mul_src->type));
    size_t merged_offset = 0;
    size_t merged_size   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_size(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_size(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) mul_src->ne[0],
        (uint32_t) mul_src->ne[1],
        (uint32_t) mul_src->ne[2],
        (uint32_t) mul_src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(rn_dst, 0))  // epsilon, treated as f32 in the shader
    };

    std::vector<wgpu::BindGroupEntry> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    if (decisions->src_overlap) {
        const ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_size    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    if (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } else if (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } else {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(dst));
}

static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(src));
}

static webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace         = decisions->inplace;
    const int  has_freq_factor = (src2 != nullptr);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
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
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_factor),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_factor),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3]
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    if (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  // limit, for swiglu_oai
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
    };
    uint32_t dst_binding = 1;
    if (split) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
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
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    if (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    const bool inplace     = decisions->inplace;
    const int  has_mask    = (src1 != nullptr);
    const int  has_sink    = (src2 != nullptr);
    float      max_bias    = ggml_get_op_params_f32(dst, 1);
    float      n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
    float      m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_size(ctx, src0)) };
    uint32_t                          binding_num = 1;
    if (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_num, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        binding_num++;
    }
    if (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, src2));
        binding_num++;
    }
    if (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_num, dst));
    }

    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(dst));
}

static webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_top_k = dst->op == GGML_OP_TOP_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    const uint32_t src_ne0 = (uint32_t) src->ne[0];
    const uint32_t nrows   = (uint32_t) ggml_nrows(src);
    const uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_size);
    const uint32_t block_size =
        is_top_k ? std::min(argsort_decisions->wg_size, (uint32_t) dst->ne[0]) : argsort_decisions->wg_size;
    uint32_t out_ne0 = src_ne0;
    if (is_top_k) {
        if (npr > 1) {
            const uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_size;
            out_ne0                  = (npr - 1) * block_size + std::min(last_tile, block_size);
        } else {
            out_ne0 = block_size;
        }
    }

    uint32_t merge_len    = block_size;
    uint32_t merge_passes = 0;
    while (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    const bool start_in_tmp = (merge_passes % 2) == 1;

    const size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    const size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * sizeof(int32_t);
    const size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    const size_t tmp_binding_size = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t dst_binding_size =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    const uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    const uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    const uint32_t offset_tmp  = 0;
    const uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    const uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    const uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));
    const uint32_t stride_idx1 = out_ne0;
    const uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    const uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    const uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    const size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    const size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    const uint32_t                    total_wg_init = npr * nrows;
    const uint32_t                    max_wg    = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    const uint32_t                    wg_x_init = std::min(total_wg_init, max_wg);
    const uint32_t                    wg_y_init = CEIL_DIV(total_wg_init, wg_x_init);
    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_size)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    if (merge_passes == 0) {
        return ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    bool     in_is_tmp = start_in_tmp;
    uint32_t len       = block_size;
    while (len < out_ne0) {
        const uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        const bool     out_is_tmp  = !in_is_tmp;
        const uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        const uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        const size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        const size_t   size_in     = in_is_tmp ? tmp_binding_size : dst_binding_size;
        const size_t   size_out    = out_is_tmp ? tmp_binding_size : dst_binding_size;
        const uint32_t top_k_out   = (is_top_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        const uint32_t stride_out1 = top_k_out;
        const uint32_t stride_out2 = top_k_out * (uint32_t) dst->ne[1];
        const uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> merge_params = { offset_src,
                                               offset_in,
                                               offset_out,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               out_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               top_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, size_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, size_out)
        };

        const uint32_t total_wg_merge = nm * nrows;
        const uint32_t wg_x_merge     = std::min(total_wg_merge, max_wg);
        const uint32_t wg_y_merge     = CEIL_DIV(total_wg_merge, wg_x_merge);
        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    return ggml_backend_webgpu_build_multi(ctx, dispatches);
}

static webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool                  total_sum = dst->op == GGML_OP_SUM;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     total_sum ? 1 : (uint32_t) src->ne[1],
                                     total_sum ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_sum_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x = total_sum ? 1 : ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static bool ggml_webgpu_can_fuse_rms_norm_mul(const struct ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        return false;
    }

    // additional constraints specific to this fusion
    const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }
    // rms_norm shader assumes contiguous rows
    if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    return true;
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           int            node_idx,
                                                           int &          num_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  node  = nodes[node_idx];

    if (ggml_is_empty(node)) {
        return std::nullopt;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << node << ", " << ggml_op_name(node->op) << ")");

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
            return std::nullopt;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_webgpu_cpy(ctx, src0, node);
        case GGML_OP_SET:
            return ggml_webgpu_set(ctx, src0, src1, node);
        case GGML_OP_SET_ROWS:
            return ggml_webgpu_set_rows(ctx, src0, src1, node);
        case GGML_OP_GET_ROWS:
            return ggml_webgpu_get_rows(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT:
            return ggml_webgpu_mul_mat(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT_ID:
            return ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return ggml_webgpu_binary_op(ctx, src0, src1, node);
        case GGML_OP_CONCAT:
            return ggml_webgpu_concat(ctx, src0, src1, node);
        case GGML_OP_REPEAT:
            return ggml_webgpu_repeat(ctx, src0, node);
        case GGML_OP_RMS_NORM:
            if (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, node_idx)) {
                num_encoded_ops        = 2;
                ggml_tensor * mul_node = nodes[node_idx + 1];
                return ggml_webgpu_rms_norm_mul(ctx, src0, node, mul_node->src[0], mul_node->src[1], mul_node);
            } else {
                return ggml_webgpu_row_norm(ctx, src0, node);
            }
        case GGML_OP_L2_NORM:
            return ggml_webgpu_row_norm(ctx, src0, node);
        case GGML_OP_ROPE:
            return ggml_webgpu_rope(ctx, src0, src1, src2, node);
        case GGML_OP_GLU:
            return ggml_webgpu_glu(ctx, src0, src1, node);
        case GGML_OP_SCALE:
            return ggml_webgpu_scale(ctx, src0, node);
        case GGML_OP_SOFT_MAX:
            return ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_LOG:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_SOLVE_TRI:
            return ggml_webgpu_solve_tri(ctx, src0, src1, node);
        case GGML_OP_SSM_CONV:
            return ggml_webgpu_ssm_conv(ctx, src0, src1, node);
        case GGML_OP_SSM_SCAN:
            return ggml_webgpu_ssm_scan(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node->src[6],
                                        node);
        case GGML_OP_GATED_DELTA_NET:
            return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node);
        case GGML_OP_PAD:
            return ggml_webgpu_pad(ctx, src0, node);
        case GGML_OP_ARGMAX:
            return ggml_webgpu_argmax(ctx, src0, node);
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // we reuse the same argsort implementation for top_k
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_CUMSUM:
            return ggml_webgpu_cumsum(ctx, src0, node);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            return ggml_webgpu_sum_rows(ctx, src0, node);
        case GGML_OP_CONV_2D:
            return ggml_webgpu_conv_2d(ctx, src0, src1, node);
        case GGML_OP_IM2COL:
            return ggml_webgpu_im2col(ctx, src0, src1, node);
        default:
            return std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
static void ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        const std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       num_inflight_batches) {
    if (pipeline_names.empty()) {
        return;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_count,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.CopyBufferToBuffer(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_count * sizeof(uint64_t));

    wgpu::CommandBuffer profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, num_inflight_batches);

    const size_t mapped_size = ctx->profile_timestamp_query_count * sizeof(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_count == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_size);
    const uint64_t * ts_data = (const uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_size);

    for (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        const double elapsed_ms = double(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
static void ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & num_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, num_inflight_batches);
    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.GetSize());
    const uint32_t * error_data = (const uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    if (*error_data) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(num_inflight_batches);
#endif
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t num_batched_kernels  = 0;
    uint32_t num_inflight_batches = 0;
    bool     contains_set_rows    = false;
    bool     batch_compute_passes = true;
    int      num_encoded_ops      = 1;
    int      node_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_count = 0;
    batch_compute_passes               = false;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    if (batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    while (node_idx < cgraph->n_nodes) {
        if (cgraph->nodes[node_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = true;
        }
        if (auto cmd = ggml_webgpu_encode(ctx, cgraph, node_idx, num_encoded_ops)) {
            commands.push_back(*cmd);
            num_batched_kernels += cmd.value().num_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        if (num_batched_kernels >= ctx->global_ctx->command_submit_batch_size) {
            if (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            num_batched_kernels                = 0;
            wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            if (batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
        }

        node_idx += num_encoded_ops;
        num_encoded_ops = 1;
    }

    if (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    if (num_batched_kernels > 0) {
        wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, num_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, num_inflight_batches);
#endif

    if (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, num_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    bool                  recorded = false;
};

static ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = new ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = new ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    return event;
}

static void ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

static void ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    if (!event_ctx->recorded) {
        return;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    if (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = false;
}

static void ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = true;
}

static void ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

static void ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 const void *   data,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
}

static void ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ NULL,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

static void ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->buffer != nullptr) {
        ctx->buffer.Destroy();
        delete ctx;
    }
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
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        return;
    }

    WEBGPU_CPU_PROFILE_TOTAL_START(memset_tensor);

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32, total_offset, size);
    WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        for (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *   tensor,
                                                  void *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t final_size = size;
    if (size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_size = size + (4 - (size % 4));
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    if (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        if (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_size,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buf_ctx->buffer, total_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_size);
    // Must specify size here since the staging buffer might be larger than the tensor size
    const void * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, mapped_range, size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
}

static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->global_ctx);
}

static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ NULL,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    static std::atomic<int> buffer_count;
    int                     buffer_id = buffer_count++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
}

static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             const ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    switch (tensor->op) {
        case GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            break;
        case GGML_OP_TOP_K:
            {
                const ggml_tensor * src0 = tensor->src[0];
                if (src0) {
                    const size_t full = sizeof(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const ggml_tensor * Q     = tensor->src[0];
                const ggml_tensor * K     = tensor->src[1];
                const ggml_tensor * V     = tensor->src[2];
                const ggml_tensor * mask  = tensor->src[3];
                const ggml_tensor * sinks = tensor->src[4];
                if (Q && K && V) {
                    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
                    shader_lib_ctx.src0                           = const_cast<ggml_tensor *>(Q);
                    shader_lib_ctx.src1                           = const_cast<ggml_tensor *>(K);
                    shader_lib_ctx.src2                           = const_cast<ggml_tensor *>(V);
                    shader_lib_ctx.src3                           = const_cast<ggml_tensor *>(mask);
                    shader_lib_ctx.src4                           = const_cast<ggml_tensor *>(sinks);
                    shader_lib_ctx.dst                            = const_cast<ggml_tensor *>(tensor);
                    shader_lib_ctx.max_wg_size =
                        ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
                    shader_lib_ctx.wg_mem_limit_bytes =
                        ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                    shader_lib_ctx.supports_subgroups = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
                    shader_lib_ctx.supports_subgroup_matrix =
                        ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
                    shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
                    shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
                    shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
                    shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;

                    const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
                        shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

                    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
                        const uint32_t kv_tile = decisions.kv_tile;

                        const uint32_t vec_nwg_cap = std::max(
                            1u, std::min<uint32_t>(32u, ctx->webgpu_global_ctx->capabilities.max_subgroup_size));
                        uint32_t       nwg     = 1u;
                        const uint64_t kv_span = (uint64_t) std::max(1u, kv_tile);
                        while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
                            nwg <<= 1;
                        }
                        nwg = std::min(nwg, vec_nwg_cap);

                        const size_t align =
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
                        const uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                        if (nwg > 1u) {
                            const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                            const uint64_t tmp_stats_elems = nrows * 2u * nwg;
                            const size_t   tmp_size_bytes  = ROUNDUP_POW2(
                                (tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
                            res += tmp_size_bytes + align;
                        } else {
                            res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                        }
                        if (mask != nullptr) {
                            const uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                            const uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                            const uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
                            const uint32_t blk_batch_count = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                            const uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
                            const size_t   blk_size_bytes =
                                ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                            res += blk_size_bytes + align;
                        }
                        res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                    }
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            {
                const ggml_tensor * src0 = tensor->src[0];
                const ggml_tensor * src1 = tensor->src[1];
                if (src0 && src1) {
                    const size_t gathered_size = sizeof(uint32_t) * tensor->src[0]->ne[2] * tensor->src[1]->ne[2];
                    const size_t gathered_count_ids_size = sizeof(uint32_t) * tensor->src[0]->ne[2];
                    res                                  = ROUNDUP_POW2(
                        res + gathered_size * 2 + gathered_count_ids_size +
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment * 3,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            break;
        default:
            break;
    }
    return res;
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
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#if UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_size = static_cast<uint64_t>(UINTPTR_MAX);
    if (max_buffer_size > max_ptr_size) {
        max_buffer_size = max_ptr_size;
    }
#endif
    *free  = static_cast<size_t>(max_buffer_size);
    *total = static_cast<size_t>(max_buffer_size);
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
    static ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    return &guid;
}

static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_threads = WEBGPU_MAX_WG_SIZE * ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].key     = "wg_size";
    constants[0].value   = WEBGPU_MAX_WG_SIZE;
    constants[1].key     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    wgpu::RequestAdapterOptions options = {};

#ifndef __EMSCRIPTEN__
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    const char * const          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabledToggleCount = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&ctx](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char * message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                    return;
                }
                ctx->webgpu_global_ctx->adapter = std::move(adapter);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#ifndef __EMSCRIPTEN__
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.nextInChain = &subgroup_matrix_configs;
    }
#endif
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_size = ggml_backend_webgpu_get_command_submit_batch_size();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    wgpu::SupportedFeatures features;
    ctx->webgpu_global_ctx->adapter.GetFeatures(&features);
    // we require f16 support
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ShaderF16));
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);

    bool valid_subgroup_matrix_config = false;
#ifndef __EMSCRIPTEN__
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
            const wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            if (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = true;
                break;
            }
        }
    }
#endif
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // For subgroup matrix code to be the most efficient, we would like the subgroup size to be consistent and accurate.
    // Unfortunately, that is not possible, so we use the maximum subgroup size reported by the adapter.
    ctx->webgpu_global_ctx->capabilities.max_subgroup_size = info.subgroupMaxSize;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#ifndef __EMSCRIPTEN__
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    if (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    if (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason),
                       std::string(message).c_str());
        });

#ifndef __EMSCRIPTEN__
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    const char * const          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    const char * const          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabledToggleCount  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabledToggleCount = 1;

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    return;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
    return true;
}

static webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_size + WEBGPU_NUM_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "set_rows_host_error_buf");

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_webgpu_create_buffer(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_COUNT;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * sizeof(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif
    return webgpu_ctx;
}

static ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = new ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
                        /* .get_name       = */ ggml_backend_webgpu_buffer_type_get_name,
                        /* .alloc_buffer   = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
                        /* .get_alignment  = */ ggml_backend_webgpu_buffer_type_get_alignment,
                        /* .get_max_size   = */ ggml_backend_webgpu_buffer_type_get_max_size,
                        /* .get_alloc_size = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
                        /* .is_host        = */ NULL,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ NULL
    };

    return &ggml_backend_webgpu_buffer_type;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_webgpu_supported_qtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
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

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
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
        case GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            break;
        case GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32 || src0->type == GGML_TYPE_I16);
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            supports_op = ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) ||
                          (op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32);
            break;
        case GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            break;
        case GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) && src0->type == GGML_TYPE_F32 &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            break;
        case GGML_OP_GET_ROWS:
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } else if (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
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
                            case GGML_TYPE_Q1_0:
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
        case GGML_OP_MUL_MAT_ID:
            switch (src1->type) {
                case GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    break;
                case GGML_TYPE_F32:
                    switch (src0->type) {
                        case GGML_TYPE_F32:
                        case GGML_TYPE_F16:
                        case GGML_TYPE_Q1_0:
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
                            supports_op = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              src2->type == src1->type && op->type == GGML_TYPE_F32;
                if (!supports_op) {
                    break;
                }
                ggml_webgpu_shader_lib_context shader_lib_ctx = {};
                shader_lib_ctx.src0                           = src0;
                shader_lib_ctx.src1                           = src1;
                shader_lib_ctx.src2                           = src2;
                shader_lib_ctx.src3                           = op->src[3];
                shader_lib_ctx.src4                           = op->src[4];
                shader_lib_ctx.dst                            = const_cast<ggml_tensor *>(op);
                shader_lib_ctx.supports_subgroups             = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
                shader_lib_ctx.supports_subgroup_matrix = ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
                shader_lib_ctx.wg_mem_limit_bytes =
                    ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
                shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
                shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
                shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;

                const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
                    shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
                const size_t limit_bytes = ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
                const bool   has_mask    = op->src[3] != nullptr;
                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
                    const size_t min_bytes =
                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                    if (min_bytes > limit_bytes) {
                        supports_op = false;
                    }
                    break;
                }

                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
                    const size_t min_bytes =
                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                    if (min_bytes > limit_bytes) {
                        supports_op = false;
                    }
                    break;
                }

                if (!ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
                    supports_op = false;
                    break;
                }
                const size_t min_bytes =
                    ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
                                                        (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
                if (min_bytes > limit_bytes) {
                    supports_op = false;
                }
                break;
            }
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
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
        case GGML_OP_UNARY:
            {
                const ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                switch (UNARY_OP) {
                    case GGML_UNARY_OP_ABS:
                    case GGML_UNARY_OP_SGN:
                    case GGML_UNARY_OP_NEG:
                    case GGML_UNARY_OP_STEP:
                    case GGML_UNARY_OP_TANH:
                    case GGML_UNARY_OP_ELU:
                    case GGML_UNARY_OP_RELU:
                    case GGML_UNARY_OP_SIGMOID:
                    case GGML_UNARY_OP_GELU:
                    case GGML_UNARY_OP_GELU_QUICK:
                    case GGML_UNARY_OP_SILU:
                    case GGML_UNARY_OP_HARDSWISH:
                    case GGML_UNARY_OP_HARDSIGMOID:
                    case GGML_UNARY_OP_EXP:
                    case GGML_UNARY_OP_GELU_ERF:
                    case GGML_UNARY_OP_SOFTPLUS:
                    case GGML_UNARY_OP_EXPM1:
                    case GGML_UNARY_OP_FLOOR:
                    case GGML_UNARY_OP_CEIL:
                    case GGML_UNARY_OP_ROUND:
                    case GGML_UNARY_OP_TRUNC:
                    case GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        break;
                    default:
                        break;
                }
            }
            break;
        case GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            break;
        case GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            break;
        case GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            break;
        case GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            break;
        case GGML_OP_GATED_DELTA_NET:
            {
                const uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            break;
        case GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_FILL:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_TOP_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            break;
        case GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            break;
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            break;
        default:
            break;
    }
    if (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
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
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
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

// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, reg_ctx->webgpu_global_ctx);
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

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    static ggml_backend_webgpu_reg_context * ctx = new ggml_backend_webgpu_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_count = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    if (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        return &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.requiredFeatureCount                 = instance_features.size();

#ifndef __EMSCRIPTEN__
    const char * const          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabledToggleCount = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(new webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    if (ctx->webgpu_global_ctx->instance != nullptr) {
        wgpu::RequestAdapterOptions options = {};

        // probe for adapter support
        ctx->webgpu_global_ctx->instance.WaitAny(
            ctx->webgpu_global_ctx->instance.RequestAdapter(
                &options, wgpu::CallbackMode::AllowSpontaneous,
                [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, const char * message) {
                    if (status != wgpu::RequestAdapterStatus::Success) {
                        GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                        return;
                    }
                    adapter = std::move(_adapter);
                }),
            UINT64_MAX);
    }

    if (adapter != nullptr) {
        ctx->device_count = 1;
    }

    return &reg;
}

ggml_backend_t ggml_backend_webgpu_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_webgpu_reg(), 0);

    return ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)

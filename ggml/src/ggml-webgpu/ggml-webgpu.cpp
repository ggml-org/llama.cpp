/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#include "ggml-webgpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-webgpu-shader-lib.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "pre_wgsl.hpp"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

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
#    define WEBGPU_NUM_TIMESTAMP_QUERY_BUFS       24
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES 16  // e.g. enough for two timestamps
#endif

/* Constants */

// Track https://github.com/gpuweb/gpuweb/issues/5315 for fixes to implementations so this can be removed.
#define WEBGPU_MAX_WG_SIZE 288

#define WEBGPU_MUL_MAT_WG_SIZE               256
#define WEBGPU_NUM_PARAM_BUFS                32u
#define WEBGPU_COMMAND_SUBMIT_BATCH_SIZE     8u
#define WEBGPU_WAIT_ANY_TIMEOUT_MS           0
// Maximum number of in-flight submissions per-thread, to avoid exhausting the parameter buffer pool
#define WEBGPU_MAX_INFLIGHT_SUBS_PER_THREAD  WEBGPU_NUM_PARAM_BUFS / WEBGPU_COMMAND_SUBMIT_BATCH_SIZE
#define WEBGPU_PARAMS_BUF_SIZE_BYTES         128  // enough for 32 parameters
#define WEBGPU_NUM_SET_ROWS_ERROR_BUFS       32
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES 4
#define WEBGPU_STORAGE_BUF_BINDING_MULT      4  // a storage buffer binding size must be a multiple of 4

// For operations which process a row in parallel, this seems like a reasonable default
#define WEBGPU_ROW_SPLIT_WG_SIZE 64

// Matrix multiplication parameters

// Register tiling parameters
#define WEBGPU_MUL_MAT_TILE_M    8
#define WEBGPU_MUL_MAT_TILE_N    8
#define WEBGPU_MUL_MAT_WG_SIZE_M 8
#define WEBGPU_MUL_MAT_WG_SIZE_N 8
#define WEBGPU_MUL_MAT_TILE_K    32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#define WEBGPU_MUL_MAT_SUBGROUP_M        2
// The number of subgroups in the N dimension
#define WEBGPU_MUL_MAT_SUBGROUP_N        2
// The number of subgroup matrices each subgroup accumulates over
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M 4
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N 2

// Matrix-vector multiplication parameters
#define WEBGPU_MUL_MAT_VEC_WG_SIZE        256
// Must be multiple of 4 to work with vectorized paths, and must divide mul_mat_vec wg size
#define WEBGPU_MUL_MAT_VEC_OUTPUTS_PER_WG 64
#define WEBGPU_MUL_MAT_VEC_TILE_K         256

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

// The futures to wait on for a single queue submission
struct webgpu_submission_futures {
    std::vector<wgpu::FutureWaitInfo> futures;
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

#ifdef GGML_WEBGPU_GPU_PROFILE
struct webgpu_gpu_profile_bufs {
    wgpu::Buffer   host_buf;
    wgpu::Buffer   dev_buf;
    wgpu::QuerySet query_set;
};

// Holds a pool of parameter buffers for WebGPU operations
struct webgpu_gpu_profile_buf_pool {
    std::vector<webgpu_gpu_profile_bufs> free;

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
            ggml_webgpu_create_buffer(device, host_buf, buf_size, host_buf_usage, "ggml_webgpu_host_profile_buf");
            ggml_webgpu_create_buffer(device, dev_buf, buf_size, dev_buf_usage, "ggml_webgpu_dev_profile_buf");
            // Create a query set for 2 timestamps
            wgpu::QuerySetDescriptor ts_query_set_desc = {};

            ts_query_set_desc.type      = wgpu::QueryType::Timestamp;
            ts_query_set_desc.count     = 2;
            wgpu::QuerySet ts_query_set = device.CreateQuerySet(&ts_query_set_desc);

            free.push_back({ host_buf, dev_buf, ts_query_set });
        }
    }

    webgpu_gpu_profile_bufs alloc_bufs() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !free.empty(); });
        webgpu_gpu_profile_bufs bufs = free.back();
        free.pop_back();
        return bufs;
    }

    void free_bufs(std::vector<webgpu_gpu_profile_bufs> bufs) {
        std::lock_guard<std::mutex> lock(mutex);
        free.insert(free.end(), bufs.begin(), bufs.end());
        cv.notify_all();
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & bufs : free) {
            bufs.host_buf.Destroy();
            bufs.dev_buf.Destroy();
            bufs.query_set.Destroy();
        }
        free.clear();
    }
};
#endif

struct webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    void *                context = nullptr;
};

struct webgpu_command {
    wgpu::CommandBuffer             commands;
    std::vector<webgpu_pool_bufs>   params_bufs;
    std::optional<webgpu_pool_bufs> set_rows_error_bufs;
#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_gpu_profile_bufs timestamp_query_bufs;
    std::string             pipeline_name;
#endif
};

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    wgpu::Limits   limits;

    uint32_t max_subgroup_size;

    bool     supports_subgroup_matrix = false;
    uint32_t sg_mat_m;
    uint32_t sg_mat_n;
    uint32_t sg_mat_k;

    std::recursive_mutex mutex;
    std::atomic_uint     inflight_threads = 0;

    webgpu_buf_pool param_buf_pool;
    webgpu_buf_pool set_rows_error_buf_pool;

    pre_wgsl::Preprocessor p;

    std::map<int, webgpu_pipeline> memset_pipelines;                                 // variant or type index

    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> mul_mat_pipelines;  // src0_type, src1_type, vectorized
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>>
        mul_mat_vec_pipelines;                                                       // src0_type, src1_type, vectorized

    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
        flash_attn_pipelines;

    std::unordered_map<int, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<int, webgpu_pipeline> argsort_pipelines;        // key is order (asc/desc)
    std::unordered_map<int, webgpu_pipeline> argsort_merge_pipelines;  // key is order (asc/desc)
    std::unordered_map<int, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet

    std::unordered_map<ggml_webgpu_set_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_key_hash>
                                                  set_rows_pipelines;
    std::map<int, std::map<int, webgpu_pipeline>> get_rows_pipelines;                 // src_type, vectorized

    std::map<int, std::map<int, webgpu_pipeline>> cpy_pipelines;                      // src_type, dst_type
    std::map<int, std::map<int, webgpu_pipeline>> add_pipelines;                      // type, inplace
    std::map<int, std::map<int, webgpu_pipeline>> sub_pipelines;                      // type, inplace
    std::map<int, std::map<int, webgpu_pipeline>> mul_pipelines;                      // type, inplace
    std::map<int, std::map<int, webgpu_pipeline>> div_pipelines;                      // type, inplace

    std::map<int, webgpu_pipeline>                               rms_norm_pipelines;  // inplace
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> rope_pipelines;      // type, ff, inplace
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> glu_pipelines;       // glu_op, type, split
    std::map<int, webgpu_pipeline>                               scale_pipelines;     // inplace
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> soft_max_pipelines;  // mask_type, has_sink, inplace
    std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash>
        unary_pipelines;
    std::unordered_map<ggml_webgpu_pad_pipeline_key, webgpu_pipeline, ggml_webgpu_pad_pipeline_key_hash> pad_pipelines;

    size_t memset_bytes_per_thread;

    // Staging buffer for reading data from the GPU
    wgpu::Buffer get_tensor_staging_buf;

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, double> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, double> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, double> shader_gpu_time_ms;
    // Profiling: pool of timestamp query buffers (one per operation)
    webgpu_gpu_profile_buf_pool             timestamp_query_buf_pool;
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
    std::string    label;

    ggml_backend_webgpu_buffer_context(webgpu_context ctx, wgpu::Buffer buf, std::string lbl) :
        webgpu_ctx(std::move(ctx)),
        buffer(std::move(buf)),
        label(std::move(lbl)) {}
};

/* WebGPU object initializations */

// Process a WGSL shader string, replacing tokens of the form {{KEY}} with
// the corresponding values provided in `repls`.
static std::string ggml_webgpu_process_shader_repls(const char *                               src,
                                                    const std::map<std::string, std::string> & repls) {
    if (!src) {
        return std::string();
    }
    std::string s = src;
    for (const auto & kv : repls) {
        std::string token = "{{" + kv.first + "}}";
        size_t      pos   = 0;
        while ((pos = s.find(token, pos)) != std::string::npos) {
            s.replace(pos, token.length(), kv.second);
            pos += kv.second.length();
        }
    }
    return s;
}

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

/** End WebGPU object initializations */

/** WebGPU Actions */

// Wait for the queue to finish processing all submitted work
static void ggml_backend_webgpu_wait(webgpu_context &                         ctx,
                                     std::vector<webgpu_submission_futures> & futures,
                                     bool                                     block = true) {
    // If we have too many in-flight submissions, wait on the oldest one first. If there are many threads,
    // inflight_max may be 0, meaning that we must wait on all futures.
    uint64_t timeout_ms       = block ? UINT64_MAX : 0;
    uint32_t inflight_threads = ctx->inflight_threads;
    uint32_t inflight_max     = WEBGPU_MAX_INFLIGHT_SUBS_PER_THREAD / std::max(inflight_threads, 1u);
    while (futures.size() >= inflight_max && futures.size() > 0) {
        ctx->instance.WaitAny(futures[0].futures.size(), futures[0].futures.data(), UINT64_MAX);
        futures.erase(futures.begin());
    }
    size_t i = 0;
    while (i < futures.size()) {
        auto waitStatus = ctx->instance.WaitAny(futures[i].futures.size(), futures[i].futures.data(), timeout_ms);
        switch (waitStatus) {
            case wgpu::WaitStatus::Success:
                futures.erase(futures.begin() + i);
                break;
            case wgpu::WaitStatus::TimedOut:
                i++;
                break;
            case wgpu::WaitStatus::Error:
                GGML_LOG_ERROR("ggml_webgpu: WaitAny returned an error\n");
                break;
            default:
                GGML_LOG_ERROR("ggml_webgpu: WaitAny returned an unknown status\n");
                break;
        }
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

static webgpu_submission_futures ggml_backend_webgpu_submit(webgpu_context ctx, std::vector<webgpu_command> commands) {
    std::vector<wgpu::CommandBuffer> command_buffers;
    std::vector<webgpu_pool_bufs>    params_bufs;
    std::vector<webgpu_pool_bufs>    set_rows_error_bufs;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::pair<std::string, webgpu_gpu_profile_bufs>> pipeline_name_and_ts_bufs;
#endif

    for (const auto & command : commands) {
        command_buffers.push_back(command.commands);
        params_bufs.insert(params_bufs.end(), command.params_bufs.begin(), command.params_bufs.end());
        if (command.set_rows_error_bufs) {
            set_rows_error_bufs.push_back(command.set_rows_error_bufs.value());
        }
    }
    ctx->queue.Submit(command_buffers.size(), command_buffers.data());

    std::vector<wgpu::FutureWaitInfo> futures;

    wgpu::Future p_f = ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous,
        [ctx, params_bufs](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
            if (status != wgpu::QueueWorkDoneStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n", std::string(message).c_str());
            }
            // Free the staged buffers
            ctx->param_buf_pool.free_bufs(params_bufs);
        });
    futures.push_back({ p_f });

    for (const auto & bufs : set_rows_error_bufs) {
        wgpu::Future f = bufs.host_buf.MapAsync(
            wgpu::MapMode::Read, 0, bufs.host_buf.GetSize(), wgpu::CallbackMode::AllowSpontaneous,
            [ctx, bufs](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to map error buffer: %s\n", std::string(message).c_str());
                } else {
                    const uint32_t * error_data = (const uint32_t *) bufs.host_buf.GetConstMappedRange();
                    if (*error_data) {
                        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
                    }
                    // We can't unmap in here due to WebGPU reentrancy limitations.
                    ctx->set_rows_error_buf_pool.free_bufs({ bufs });
                }
            });
        futures.push_back({ f });
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    for (const auto & command : commands) {
        auto label   = command.pipeline_name;
        auto ts_bufs = command.timestamp_query_bufs;

        wgpu::Future f = ts_bufs.host_buf.MapAsync(
            wgpu::MapMode::Read, 0, ts_bufs.host_buf.GetSize(), wgpu::CallbackMode::AllowSpontaneous,
            [ctx, ts_bufs, label](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                if (status != wgpu::MapAsyncStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to map timestamp buffer: %s\n", std::string(message).c_str());
                } else {
                    const uint64_t * ts_data    = (const uint64_t *) ts_bufs.host_buf.GetConstMappedRange();
                    // WebGPU timestamps are in ns; convert to ms
                    double           elapsed_ms = double(ts_data[1] - ts_data[0]) * 1e-6;
                    ctx->shader_gpu_time_ms[label] += elapsed_ms;
                    // We can't unmap in here due to WebGPU reentrancy limitations.
                    ctx->timestamp_query_buf_pool.free_bufs({ ts_bufs });
                }
            });
        futures.push_back({ f });
    }
#endif
    return { futures };
}

static webgpu_command ggml_backend_webgpu_build_multi(
    webgpu_context &                                       ctx,
    const std::vector<webgpu_pipeline> &                   pipelines,
    const std::vector<std::vector<uint32_t>> &             params_list,
    const std::vector<std::vector<wgpu::BindGroupEntry>> & bind_group_entries_list,
    const std::vector<std::pair<uint32_t, uint32_t>> &     workgroups_list,
    const std::optional<webgpu_pool_bufs> &                set_rows_error_bufs = std::nullopt) {
    GGML_ASSERT(pipelines.size() == params_list.size());
    GGML_ASSERT(pipelines.size() == bind_group_entries_list.size());
    GGML_ASSERT(pipelines.size() == workgroups_list.size());

    std::vector<webgpu_pool_bufs> params_bufs_list;
    std::vector<wgpu::BindGroup>  bind_groups;

    for (size_t i = 0; i < pipelines.size(); i++) {
        webgpu_pool_bufs params_bufs = ctx->param_buf_pool.alloc_bufs();

        ggml_backend_webgpu_map_buffer(ctx, params_bufs.host_buf, wgpu::MapMode::Write, 0,
                                       params_bufs.host_buf.GetSize());
        uint32_t * _params = (uint32_t *) params_bufs.host_buf.GetMappedRange();
        for (size_t j = 0; j < params_list[i].size(); j++) {
            _params[j] = params_list[i][j];
        }
        params_bufs.host_buf.Unmap();

        std::vector<wgpu::BindGroupEntry> entries            = bind_group_entries_list[i];
        uint32_t                          params_binding_num = entries.size();
        entries.push_back({ .binding = params_binding_num,
                            .buffer  = params_bufs.dev_buf,
                            .offset  = 0,
                            .size    = params_bufs.dev_buf.GetSize() });

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = pipelines[i].pipeline.GetBindGroupLayout(0);
        bind_group_desc.entryCount = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = pipelines[i].name.c_str();
        bind_groups.push_back(ctx->device.CreateBindGroup(&bind_group_desc));

        params_bufs_list.push_back(params_bufs);
    }

    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    for (const auto & params_bufs : params_bufs_list) {
        encoder.CopyBufferToBuffer(params_bufs.host_buf, 0, params_bufs.dev_buf, 0, params_bufs.dev_buf.GetSize());
    }

    // If there are SET_ROWS operations in this submission, copy their error buffers to the host.
    if (set_rows_error_bufs) {
        encoder.CopyBufferToBuffer(set_rows_error_bufs->dev_buf, 0, set_rows_error_bufs->host_buf, 0,
                                   set_rows_error_bufs->host_buf.GetSize());
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_gpu_profile_bufs ts_bufs = ctx->timestamp_query_buf_pool.alloc_bufs();
    if (ts_bufs.host_buf.GetMapState() == wgpu::BufferMapState::Mapped) {
        ts_bufs.host_buf.Unmap();
    }

    wgpu::PassTimestampWrites   ts_writes = { .querySet                  = ts_bufs.query_set,
                                              .beginningOfPassWriteIndex = 0,
                                              .endOfPassWriteIndex       = 1 };
    wgpu::ComputePassDescriptor pass_desc = { .timestampWrites = &ts_writes };
    wgpu::ComputePassEncoder    pass      = encoder.BeginComputePass(&pass_desc);
#else
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
#endif
    for (size_t i = 0; i < pipelines.size(); i++) {
        pass.SetPipeline(pipelines[i].pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(workgroups_list[i].first, workgroups_list[i].second, 1);
    }
    pass.End();

#ifdef GGML_WEBGPU_GPU_PROFILE
    encoder.ResolveQuerySet(ts_bufs.query_set, 0, 2, ts_bufs.dev_buf, 0);
    encoder.CopyBufferToBuffer(ts_bufs.dev_buf, 0, ts_bufs.host_buf, 0, ts_bufs.host_buf.GetSize());
#endif

    wgpu::CommandBuffer commands = encoder.Finish();
    webgpu_command      result   = {};
    result.commands              = commands;
    result.params_bufs           = params_bufs_list;
    result.set_rows_error_bufs   = set_rows_error_bufs;
#ifdef GGML_WEBGPU_GPU_PROFILE
    result.timestamp_query_bufs = ts_bufs;
    // TODO: handle multiple pipeline names
    result.pipeline_name        = pipelines.front().name;
#endif
    return result;
}

static webgpu_command ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                webgpu_pipeline &                 pipeline,
                                                std::vector<uint32_t>             params,
                                                std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                uint32_t                          wg_x,
                                                uint32_t                          wg_y                = 1,
                                                std::optional<webgpu_pool_bufs>   set_rows_error_bufs = std::nullopt) {
    return ggml_backend_webgpu_build_multi(ctx,
                                           {
                                               pipeline
    },
                                           { params }, { bind_group_entries }, { { wg_x, wg_y } }, set_rows_error_bufs);
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
    size_t   bytes_per_wg = WEBGPU_MAX_WG_SIZE * ctx->memset_bytes_per_thread;
    uint32_t wg_x         = CEIL_DIV(size + 3, bytes_per_wg);

    webgpu_command command = ggml_backend_webgpu_build(ctx, ctx->memset_pipelines[0], params, entries, wg_x);
    std::vector<webgpu_submission_futures> futures = { ggml_backend_webgpu_submit(ctx, { command }) };
    ggml_backend_webgpu_wait(ctx, futures);
}

/** End WebGPU Actions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    return ctx->name.c_str();
}

// TODO: implement proper cleanup
static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    double total_cpu = 0.0;
    for (const auto & kv : ctx->webgpu_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    for (const auto & kv : ctx->webgpu_ctx->cpu_time_ms) {
        double pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    if (ctx->webgpu_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    for (const auto & kv : ctx->webgpu_ctx->cpu_detail_ms) {
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
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#if defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

#if !defined(GGML_WEBGPU_CPU_PROFILE) && !defined(GGML_WEBGPU_GPU_PROFILE)
    GGML_UNUSED(ctx);
#endif
}

static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
    return webgpu_tensor_offset(tensor) + tensor->view_offs;
}

static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    return ctx->buffer;
}

static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & (ctx->limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, const ggml_tensor * t) {
    size_t offset = ggml_webgpu_tensor_offset(t);
    return offset & ~(ctx->limits.minStorageBufferOffsetAlignment - 1);
}

static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ggml_tensor * t) {
    return ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(ctx, t), WEBGPU_STORAGE_BUF_BINDING_MULT);
}

// Used to determine if two tensors are the same for in-place operations
static bool ggml_webgpu_tensor_equal(ggml_tensor * a, ggml_tensor * b) {
    return (ggml_webgpu_tensor_buf(a).Get() == ggml_webgpu_tensor_buf(b).Get()) &&
           (ggml_webgpu_tensor_offset(a) == ggml_webgpu_tensor_offset(b));
}

static webgpu_command ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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

    uint32_t wg_x = CEIL_DIV(ne, WEBGPU_MAX_WG_SIZE);
    return ggml_backend_webgpu_build(ctx, ctx->cpy_pipelines[src->type][dst->type], params, entries, wg_x);
}

static webgpu_command ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    const bool circular = ggml_get_op_params_i32(dst, 8) != 0;

    ggml_webgpu_pad_pipeline_key       pipeline_key   = { .circular = circular };
    ggml_webgpu_pad_shader_lib_context shader_lib_ctx = { .key = pipeline_key,
                                                          .max_wg_size =
                                                              ctx->limits.maxComputeInvocationsPerWorkgroup };

    webgpu_pipeline pipeline;
    {
        // TODO: remove guard once pipeline caches are per-thread
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->pad_pipelines.find(pipeline_key);
        if (it != ctx->pad_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_pad_shader(ctx->p, wgsl_pad, shader_lib_ctx);
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            pipeline.context = processed.decisions;
            ctx->pad_pipelines.emplace(pipeline_key, pipeline);
        }
    }

    ggml_webgpu_generic_shader_decisions decisions =
        *static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context);

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
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(dst),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) }
    };

    uint32_t wg_x = CEIL_DIV(ne, decisions.wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static std::optional<webgpu_command> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                          ggml_tensor *    src,
                                                          ggml_tensor *    idx,
                                                          ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty tensors.
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return std::nullopt;
    }

    ggml_webgpu_set_rows_pipeline_key key = { .dst_type = dst->type,
                                              .vec4     = src->ne[0] % 4 == 0,
                                              .i64_idx  = idx->type == GGML_TYPE_I64 };

    ggml_webgpu_set_rows_shader_lib_context shader_lib_ctx = { .key = key,
                                                               .max_wg_size =
                                                                   ctx->limits.maxComputeInvocationsPerWorkgroup };

    webgpu_pipeline pipeline;
    // TODO: remove guard once pipeline caches are per-thread
    {
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->set_rows_pipelines.find(key);
        if (it != ctx->set_rows_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_set_rows_shader(ctx->p, wgsl_set_rows, shader_lib_ctx);
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            pipeline.context = processed.decisions;
            ctx->set_rows_pipelines.emplace(key, pipeline);
        }
    }

    ggml_webgpu_generic_shader_decisions decisions =
        *static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context);

    std::optional<webgpu_pool_bufs> error_bufs = std::nullopt;
    if (key.i64_idx) {
        error_bufs = ctx->set_rows_error_buf_pool.alloc_bufs();
        if (error_bufs->host_buf.GetMapState() == wgpu::BufferMapState::Mapped) {
            error_bufs->host_buf.Unmap();
        }
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
         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) }
    };

    if (key.i64_idx) {
        entries.push_back(
            { .binding = 3, .buffer = error_bufs->dev_buf, .offset = 0, .size = error_bufs->dev_buf.GetSize() });
    }

    uint32_t threads;
    if (key.vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } else {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x = CEIL_DIV(threads, decisions.wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, 1, error_bufs);
}

static webgpu_command ggml_webgpu_get_rows(webgpu_context & ctx,
                                           ggml_tensor *    src,
                                           ggml_tensor *    idx,
                                           ggml_tensor *    dst) {
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

    uint32_t wg_x = CEIL_DIV(dst->ne[1] * dst->ne[2] * dst->ne[3], WEBGPU_MAX_WG_SIZE);

    uint32_t        vectorized = src->type == GGML_TYPE_F32 && dst->ne[0] % 4 == 0;
    webgpu_pipeline pipeline   = ctx->get_rows_pipelines[src->type][vectorized];
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_mul_mat(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) dst->ne[0],                                  // number of rows in result (M, transposed)
        (uint32_t) dst->ne[1],                                  // number of columns in result (N)
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

    webgpu_pipeline pipeline = ctx->mul_mat_pipelines[src0->type][src1->type][0];

    uint32_t wg_x = CEIL_DIV(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3], WEBGPU_MUL_MAT_WG_SIZE);
    uint32_t wg_y = 1;

    bool use_fast = false;
    switch (src1->type) {
        case GGML_TYPE_F16:
            use_fast = (src0->type == GGML_TYPE_F16);
            break;
        case GGML_TYPE_F32:
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                    use_fast = true;
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    if (use_fast) {
        int vectorized = src0->ne[0] % 4 == 0 && dst->ne[0] % 4 == 0 && dst->ne[1] % 4 == 0;
        if (dst->ne[1] == 1) {
            // We don't support vectorized mul_mat_vec for quantized types
            vectorized             = vectorized && (src0->type < 2);
            pipeline               = ctx->mul_mat_vec_pipelines[src0->type][src1->type][vectorized];
            uint32_t batches       = dst->ne[2] * dst->ne[3];
            uint32_t output_groups = CEIL_DIV(dst->ne[0], WEBGPU_MUL_MAT_VEC_OUTPUTS_PER_WG);
            uint32_t total_wg      = output_groups * batches;
            wg_x                   = total_wg % ctx->limits.maxComputeWorkgroupsPerDimension;
            wg_y                   = CEIL_DIV(total_wg, ctx->limits.maxComputeWorkgroupsPerDimension);
        } else {
            pipeline = ctx->mul_mat_pipelines[src0->type][src1->type][vectorized];
            uint32_t wg_m;
            uint32_t wg_n;
#ifndef __EMSCRIPTEN__
            if (ctx->supports_subgroup_matrix) {
                // The total number of subgroups/workgroups needed per matrix.
                uint32_t wg_m_sg_tile = WEBGPU_MUL_MAT_SUBGROUP_M * WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M * ctx->sg_mat_m;
                wg_m                  = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
                uint32_t wg_n_sg_tile = WEBGPU_MUL_MAT_SUBGROUP_N * WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N * ctx->sg_mat_n;
                wg_n                  = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
            } else {
#endif
                uint32_t tile_m_s = WEBGPU_MUL_MAT_TILE_M * WEBGPU_MUL_MAT_WG_SIZE_M;
                uint32_t tile_n_s = WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N;
                wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
                wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
#ifndef __EMSCRIPTEN__
            }
#endif

            wg_x = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        }
    }
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

static webgpu_command ggml_webgpu_flash_attn(webgpu_context & ctx,
                                             ggml_tensor *    Q,
                                             ggml_tensor *    K,
                                             ggml_tensor *    V,
                                             ggml_tensor *    mask,
                                             ggml_tensor *    sinks,
                                             ggml_tensor *    dst) {
    float scale = *(float *) dst->op_params;
    float max_bias;
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    float logit_softcap;
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    float n_head_log2 = float(1u << (uint32_t) floor(log2(Q->ne[2])));
    float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int has_mask  = (mask != nullptr);
    const int has_sinks = (sinks != nullptr);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type)),
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
        *(uint32_t *) &scale,              // scale (possibly adjusted for logit softcap)
        *(uint32_t *) &max_bias,
        *(uint32_t *) &logit_softcap,
        *(uint32_t *) &n_head_log2,
        *(uint32_t *) &m0,
        *(uint32_t *) &m1

    };
    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(Q),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, Q),
         .size    = ggml_webgpu_tensor_binding_size(ctx, Q) },
        { .binding = 1,
         .buffer  = ggml_webgpu_tensor_buf(K),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, K),
         .size    = ggml_webgpu_tensor_binding_size(ctx, K) },
        { .binding = 2,
         .buffer  = ggml_webgpu_tensor_buf(V),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, V),
         .size    = ggml_webgpu_tensor_binding_size(ctx, V) }
    };
    uint32_t binding_index = 3;
    if (has_mask) {
        entries.push_back({ .binding = binding_index++,
                            .buffer  = ggml_webgpu_tensor_buf(mask),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, mask),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, mask) });
    }
    if (has_sinks) {
        entries.push_back({ .binding = binding_index++,
                            .buffer  = ggml_webgpu_tensor_buf(sinks),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, sinks),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, sinks) });
    }
    entries.push_back({ .binding = binding_index++,
                        .buffer  = ggml_webgpu_tensor_buf(dst),
                        .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                        .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });

    bool kv_direct =
        (K->type == GGML_TYPE_F16) && (Q->ne[0] % ctx->sg_mat_k == 0) && (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);

    ggml_webgpu_flash_attn_pipeline_key key = {
        .kv_type            = K->type,
        .head_dim_qk        = (uint32_t) Q->ne[0],
        .head_dim_v         = (uint32_t) V->ne[0],
        .kv_direct          = kv_direct,
        .has_mask           = static_cast<bool>(has_mask),
        .has_sinks          = static_cast<bool>(has_sinks),
        .uses_logit_softcap = logit_softcap != 0.0f,
    };

    webgpu_pipeline                         pipeline;
    // TODO: remove guard once pipeline caches are per-thread
    {
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->flash_attn_pipelines.find(key);
        if (it != ctx->flash_attn_pipelines.end()) {
            pipeline  = it->second;
        } else {
            ggml_webgpu_flash_attn_shader_lib_context shader_lib_ctx = { .key      = key,
                                                                         .sg_mat_m = ctx->sg_mat_m,
                                                                         .sg_mat_n = ctx->sg_mat_n,
                                                                         .sg_mat_k = ctx->sg_mat_k,
                                                                         .wg_mem_limit_bytes =
                                                                             ctx->limits.maxComputeWorkgroupStorageSize,
                                                                         .max_subgroup_size = ctx->max_subgroup_size };

            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_flash_attn_shader(ctx->p, wgsl_flash_attn, shader_lib_ctx);
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            pipeline.context = processed.decisions;
            ctx->flash_attn_pipelines.emplace(key, pipeline);
        }
    }

    ggml_webgpu_flash_attn_shader_decisions decisions =
        *static_cast<ggml_webgpu_flash_attn_shader_decisions *>(pipeline.context);


    uint32_t wg_per_head = CEIL_DIV(Q->ne[1], decisions.q_tile);
    uint32_t wg_x        = wg_per_head * Q->ne[2] * Q->ne[3];  // wg per head * number of heads * number of batches
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_unary = dst->op == GGML_OP_UNARY;
    bool inplace  = ggml_webgpu_tensor_equal(src, dst) || (dst->op == GGML_OP_FILL);
    int  op       = is_unary ? (int) ggml_get_unary_op(dst) : dst->op;

    ggml_webgpu_unary_pipeline_key pipeline_key = {
        .type = dst->type, .op = op, .is_unary = is_unary, .inplace = inplace
    };
    ggml_webgpu_unary_shader_lib_context shader_lib_ctx = { .key = pipeline_key,
                                                            .max_wg_size =
                                                                ctx->limits.maxComputeInvocationsPerWorkgroup };

    webgpu_pipeline pipeline;
    {
        // TODO: remove guard once pipeline caches are per-thread
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->unary_pipelines.find(pipeline_key);
        if (it != ctx->unary_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_unary_shader(ctx->p, wgsl_unary, shader_lib_ctx);
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            pipeline.context = processed.decisions;
            ctx->unary_pipelines.emplace(pipeline_key, pipeline);
        }
    }

    ggml_webgpu_generic_shader_decisions decisions =
        *static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context);

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
                    params.push_back(*reinterpret_cast<const uint32_t *>(&alpha_n));
                    params.push_back(*reinterpret_cast<const uint32_t *>(&alpha_p));
                    params.push_back(*reinterpret_cast<const uint32_t *>(&beta));
                    params.push_back(*reinterpret_cast<const uint32_t *>(&eps));
                    break;
                }
            default:
                break;
        }
    } else if (dst->op == GGML_OP_CLAMP) {
        float clamp_min = ggml_get_op_params_f32(dst, 0);
        float clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(*reinterpret_cast<const uint32_t *>(&clamp_min));
        params.push_back(*reinterpret_cast<const uint32_t *>(&clamp_max));
    } else if (dst->op == GGML_OP_FILL) {
        float fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(*reinterpret_cast<const uint32_t *>(&fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(effective_src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, effective_src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, effective_src) },
    };
    if (!inplace) {
        entries.push_back({ .binding = 1,
                            .buffer  = ggml_webgpu_tensor_buf(dst),
                            .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                            .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
    }

    uint32_t wg_x = CEIL_DIV(ne, decisions.wg_size);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_binary_op(webgpu_context &  ctx,
                                            ggml_tensor *     src0,
                                            ggml_tensor *     src1,
                                            ggml_tensor *     dst,
                                            webgpu_pipeline & pipeline,
                                            bool              inplace) {
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

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), WEBGPU_MAX_WG_SIZE);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_rms_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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

    return ggml_backend_webgpu_build(ctx, ctx->rms_norm_pipelines[inplace], params, entries, ggml_nrows(src));
}

static webgpu_command ggml_webgpu_rope(webgpu_context & ctx,
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

    webgpu_pipeline pipeline = ctx->rope_pipelines[dst->type][has_freq_factor][inplace];
    uint32_t        wg_x     = CEIL_DIV(ggml_nelements(dst), WEBGPU_MAX_WG_SIZE);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_glu(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
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

    webgpu_pipeline pipeline = ctx->glu_pipelines[ggml_get_glu_op(dst)][dst->type][split];
    uint32_t        wg_x     = CEIL_DIV(ggml_nelements(dst), WEBGPU_MAX_WG_SIZE);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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

    uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), WEBGPU_MAX_WG_SIZE);
    return ggml_backend_webgpu_build(ctx, ctx->scale_pipelines[inplace], params, entries, wg_x);
}

static webgpu_command ggml_webgpu_soft_max(webgpu_context & ctx,
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

    return ggml_backend_webgpu_build(ctx, ctx->soft_max_pipelines[mask_type][has_sink][inplace], params, entries,
                                     ggml_nrows(dst));
}

static webgpu_command ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

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

    ggml_webgpu_generic_shader_lib_context shader_lib_ctx = {
        .vec4        = src->ne[0] % 4 == 0,
        .max_wg_size = ctx->limits.maxComputeInvocationsPerWorkgroup,
    };

    webgpu_pipeline pipeline;
    {
        // TODO: remove guard once pipeline caches are per-thread
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->argmax_pipelines.find(shader_lib_ctx.vec4);
        if (it != ctx->argmax_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_generic_shader(ctx->p, wgsl_argmax, shader_lib_ctx, "argmax");
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            ctx->argmax_pipelines.emplace(shader_lib_ctx.vec4, pipeline);
        }
    }
    uint32_t wg_x = ggml_nelements(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool          is_top_k = dst->op == GGML_OP_TOP_K;
    // ascending order is 0, descending order is 1
    const int32_t order    = is_top_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(dst, 0);

    ggml_webgpu_argsort_shader_lib_context shader_lib_ctx = { .max_wg_size =
                                                                  ctx->limits.maxComputeInvocationsPerWorkgroup,
                                                              .wg_mem_limit_bytes =
                                                                  ctx->limits.maxComputeWorkgroupStorageSize,
                                                              .order = order };

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    webgpu_pipeline                       argsort_pipeline;
    auto                                  it = ctx->argsort_pipelines.find(order);
    if (it != ctx->argsort_pipelines.end()) {
        argsort_pipeline = it->second;
    } else {
        ggml_webgpu_processed_shader processed =
            ggml_webgpu_preprocess_argsort_shader(ctx->p, wgsl_argsort, shader_lib_ctx);
        argsort_pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
        argsort_pipeline.context = processed.decisions;
        ctx->argsort_pipelines.emplace(order, argsort_pipeline);
    }
    ggml_webgpu_argsort_shader_decisions argsort_decisions =
        *static_cast<ggml_webgpu_argsort_shader_decisions *>(argsort_pipeline.context);

    webgpu_pipeline argsort_merge_pipeline;
    it = ctx->argsort_merge_pipelines.find(order);
    if (it != ctx->argsort_merge_pipelines.end()) {
        argsort_merge_pipeline = it->second;
    } else {
        ggml_webgpu_processed_shader processed =
            ggml_webgpu_preprocess_argsort_merge_shader(ctx->p, wgsl_argsort_merge, shader_lib_ctx);
        argsort_merge_pipeline =
            ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
        argsort_merge_pipeline.context = processed.decisions;
        ctx->argsort_merge_pipelines.emplace(order, argsort_merge_pipeline);
    }

    const uint32_t src_ne0 = (uint32_t) src->ne[0];
    const uint32_t nrows   = (uint32_t) ggml_nrows(src);
    const uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions.wg_size);
    const uint32_t block_size =
        is_top_k ? std::min(argsort_decisions.wg_size, (uint32_t) dst->ne[0]) : argsort_decisions.wg_size;
    uint32_t out_ne0 = src_ne0;
    if (is_top_k) {
        if (npr > 1) {
            const uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions.wg_size;
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

    const size_t dst_offset       = ggml_webgpu_tensor_offset(dst);
    const size_t idx_nbytes       = out_ne0 * ggml_nrows(dst) * sizeof(int32_t);
    const size_t tmp_offset       = ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->limits.minStorageBufferOffsetAlignment);
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

    std::vector<webgpu_pipeline>                   pipelines;
    std::vector<std::vector<uint32_t>>             params_list;
    std::vector<std::vector<wgpu::BindGroupEntry>> entries_list;
    std::vector<std::pair<uint32_t, uint32_t>>     workgroups_list;

    const uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    const size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    const size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    const uint32_t                    total_wg_init = npr * nrows;
    const uint32_t                    max_wg        = ctx->limits.maxComputeWorkgroupsPerDimension;
    const uint32_t                    wg_x_init     = std::min(total_wg_init, max_wg);
    const uint32_t                    wg_y_init     = CEIL_DIV(total_wg_init, wg_x_init);
    std::vector<wgpu::BindGroupEntry> init_entries  = {
        { .binding = 0,
         .buffer  = ggml_webgpu_tensor_buf(src),
         .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
         .size    = ggml_webgpu_tensor_binding_size(ctx, src) },
        { .binding = 1, .buffer = ggml_webgpu_tensor_buf(dst), .offset = init_align_offset, .size = init_binding_size }
    };

    pipelines.push_back(argsort_pipeline);
    params_list.push_back(std::move(init_params));
    entries_list.push_back(std::move(init_entries));
    workgroups_list.push_back({ wg_x_init, wg_y_init });

    if (merge_passes == 0) {
        return ggml_backend_webgpu_build_multi(ctx, pipelines, params_list, entries_list, workgroups_list);
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
            { .binding = 0,
             .buffer  = ggml_webgpu_tensor_buf(src),
             .offset  = ggml_webgpu_tensor_align_offset(ctx, src),
             .size    = ggml_webgpu_tensor_binding_size(ctx, src) },
            { .binding = 1, .buffer = ggml_webgpu_tensor_buf(dst), .offset = align_in, .size = size_in },
            { .binding = 2, .buffer = ggml_webgpu_tensor_buf(dst), .offset = align_out, .size = size_out }
        };

        const uint32_t total_wg_merge = nm * nrows;
        const uint32_t wg_x_merge     = std::min(total_wg_merge, max_wg);
        const uint32_t wg_y_merge     = CEIL_DIV(total_wg_merge, wg_x_merge);
        workgroups_list.push_back({ wg_x_merge, wg_y_merge });
        pipelines.push_back(argsort_merge_pipeline);
        params_list.push_back(std::move(merge_params));
        entries_list.push_back(std::move(merge_entries));

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    return ggml_backend_webgpu_build_multi(ctx, pipelines, params_list, entries_list, workgroups_list);
}

static webgpu_command ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

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

    ggml_webgpu_generic_shader_lib_context shader_lib_ctx = {
        .vec4        = false,
        .max_wg_size = ctx->limits.maxComputeInvocationsPerWorkgroup,
    };
    webgpu_pipeline pipeline;
    // TODO: remove guard once pipeline caches are per-thread
    {
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->cumsum_pipelines.find(1);
        if (it != ctx->cumsum_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_generic_shader(ctx->p, wgsl_cumsum, shader_lib_ctx, "cumsum");
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            ctx->cumsum_pipelines.emplace(1, pipeline);
        }
    }
    uint32_t wg_x = ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

static webgpu_command ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool                  total_sum = dst->op == GGML_OP_SUM;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     total_sum ? 1 : (uint32_t) src->ne[1],
                                     total_sum ? 1 : (uint32_t) src->ne[2] };

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

    ggml_webgpu_generic_shader_lib_context shader_lib_ctx = {
        .vec4        = false,
        .max_wg_size = ctx->limits.maxComputeInvocationsPerWorkgroup,
    };

    webgpu_pipeline pipeline;
    {
        // TODO: remove guard once pipeline caches are per-thread
        std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
        auto                                  it = ctx->sum_rows_pipelines.find(1);
        if (it != ctx->sum_rows_pipelines.end()) {
            pipeline = it->second;
        } else {
            ggml_webgpu_processed_shader processed =
                ggml_webgpu_preprocess_generic_shader(ctx->p, wgsl_sum_rows, shader_lib_ctx, "sum_rows");
            pipeline = ggml_webgpu_create_pipeline(ctx->device, processed.wgsl.c_str(), processed.variant.c_str());
            ctx->sum_rows_pipelines.emplace(1, pipeline);
        }
    }
    uint32_t wg_x = total_sum ? 1 : ggml_nrows(dst);
    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
static std::optional<webgpu_command> ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return std::nullopt;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return std::nullopt;
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
            return std::nullopt;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_webgpu_cpy(ctx, src0, node);
        case GGML_OP_SET_ROWS:
            return ggml_webgpu_set_rows(ctx, src0, src1, node);
        case GGML_OP_GET_ROWS:
            return ggml_webgpu_get_rows(ctx, src0, src1, node);
        case GGML_OP_MUL_MAT:
            return ggml_webgpu_mul_mat(ctx, src0, src1, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
        case GGML_OP_ADD:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                return ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->add_pipelines[node->type][inplace], inplace);
            }
        case GGML_OP_SUB:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                return ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->sub_pipelines[node->type][inplace], inplace);
            }
        case GGML_OP_MUL:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                return ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->mul_pipelines[node->type][inplace], inplace);
            }
        case GGML_OP_DIV:
            {
                int inplace = ggml_webgpu_tensor_equal(src0, node);
                return ggml_webgpu_binary_op(ctx, src0, src1, node, ctx->div_pipelines[node->type][inplace], inplace);
            }
        case GGML_OP_RMS_NORM:
            return ggml_webgpu_rms_norm(ctx, src0, node);
        case GGML_OP_ROPE:
            return ggml_webgpu_rope(ctx, src0, src1, src2, node);
        case GGML_OP_GLU:
            return ggml_webgpu_glu(ctx, src0, src1, node);
        case GGML_OP_SCALE:
            return ggml_webgpu_scale(ctx, src0, node);
        case GGML_OP_SOFT_MAX:
            return ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
        case GGML_OP_UNARY:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_CLAMP:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_FILL:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_LOG:
            return ggml_webgpu_unary_op(ctx, src0, node);
        case GGML_OP_PAD:
            return ggml_webgpu_pad(ctx, src0, node);
        case GGML_OP_ARGMAX:
            return ggml_webgpu_argmax(ctx, src0, node);
        case GGML_OP_ARGSORT:
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_TOP_K:
            // we reuse the same argsort implementation for top_k
            return ggml_webgpu_argsort(ctx, src0, node);
        case GGML_OP_CUMSUM:
            return ggml_webgpu_cumsum(ctx, src0, node);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
            return ggml_webgpu_sum_rows(ctx, src0, node);
        default:
            return std::nullopt;
    }
}

static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = static_cast<ggml_backend_webgpu_context *>(backend->context);
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    ctx->inflight_threads++;

    std::vector<webgpu_command>            commands;
    std::vector<webgpu_submission_futures> futures;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (auto cmd = ggml_webgpu_encode_node(ctx, cgraph->nodes[i])) {
            commands.push_back(*cmd);
        }
        // compute the batch size based on the number of inflight threads
        uint32_t inflight_threads = ctx->inflight_threads;
        uint32_t batch_size       = std::min(std::max(1u, WEBGPU_NUM_PARAM_BUFS / std::max(inflight_threads, 1u)),
                                             WEBGPU_COMMAND_SUBMIT_BATCH_SIZE);
        if (commands.size() >= batch_size) {
            futures.push_back(ggml_backend_webgpu_submit(ctx, commands));
            // Process events and check for completed submissions
            ctx->instance.ProcessEvents();
            ggml_backend_webgpu_wait(ctx, futures, false);
            commands.clear();
        }
    }
    if (!commands.empty()) {
        webgpu_submission_futures new_futures = ggml_backend_webgpu_submit(ctx, commands);
        futures.push_back(new_futures);
    }

    ggml_backend_webgpu_wait(ctx, futures);
    ctx->inflight_threads--;
    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx);
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

    WEBGPU_CPU_PROFILE_TOTAL_START(memset_tensor);

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->webgpu_ctx, buf_ctx->buffer, val32, total_offset, size);
    WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->webgpu_ctx);
}

static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx    = (ggml_backend_webgpu_buffer_context *) buffer->context;
    webgpu_context                       webgpu_ctx = buf_ctx->webgpu_ctx;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");

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
        webgpu_ctx->instance.WaitAny(
            webgpu_ctx->queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                                  [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                                          GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n",
                                                                         std::string(message).c_str());
                                                      }
                                                  }),
            UINT64_MAX);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, webgpu_ctx);
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
    webgpu_context webgpu_ctx = buf_ctx->webgpu_ctx;
    wgpu::Device   device     = webgpu_ctx->device;

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
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, webgpu_ctx);
}

static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->webgpu_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->webgpu_ctx);
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
    static std::atomic<int> buffer_count;
    int                     buffer_id = buffer_count++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);

    wgpu::Buffer buf;
    ggml_webgpu_create_buffer(ctx->webgpu_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new ggml_backend_webgpu_buffer_context(ctx->webgpu_ctx, buf, buf_name);

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

static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             const ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    switch (tensor->op) {
        case GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_ctx->limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            break;
        case GGML_OP_TOP_K:
            {
                const ggml_tensor * src0 = tensor->src[0];
                if (src0) {
                    const size_t full = sizeof(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(full * 2 + ctx->webgpu_ctx->limits.minStorageBufferOffsetAlignment,
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
    uint64_t                             max_buffer_size = ctx->webgpu_ctx->limits.maxBufferSize;
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
    size_t max_threads                  = WEBGPU_MAX_WG_SIZE * webgpu_ctx->limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    webgpu_ctx->memset_bytes_per_thread = CEIL_DIV(webgpu_ctx->limits.maxStorageBufferBindingSize, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].key                = "wg_size";
    constants[0].value              = WEBGPU_MAX_WG_SIZE;
    constants[1].key                = "bytes_per_thread";
    constants[1].value              = webgpu_ctx->memset_bytes_per_thread;
    webgpu_ctx->memset_pipelines[0] = ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_memset, "memset", constants);
}

static void ggml_webgpu_init_mul_mat_pipeline(webgpu_context & webgpu_ctx) {
    // Q4/Q5/Q8 classic quantizations
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q4_0][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q4_0_f32, "mul_mat_q4_0_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q4_1][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q4_1_f32, "mul_mat_q4_1_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q5_0][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q5_0_f32, "mul_mat_q5_0_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q5_1][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q5_1_f32, "mul_mat_q5_1_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q8_0][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q8_0_f32, "mul_mat_q8_0_f32");

    // K-quantizations
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q2_K][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q2_k_f32, "mul_mat_q2_k_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q3_K][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q3_k_f32, "mul_mat_q3_k_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q4_K][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q4_k_f32, "mul_mat_q4_k_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q5_K][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q5_k_f32, "mul_mat_q5_k_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q6_K][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_q6_k_f32, "mul_mat_q6_k_f32");

    // IQ quantizations (2-, 3-, 4-bit variants)
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ2_XXS][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq2_xxs_f32, "mul_mat_iq2_xxs_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ2_XS][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq2_xs_f32, "mul_mat_iq2_xs_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ2_S][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq2_s_f32, "mul_mat_iq2_s_f32");

    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ3_XXS][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq3_xxs_f32, "mul_mat_iq3_xxs_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ3_S][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq3_s_f32, "mul_mat_iq3_s_f32");

    // 1-bit and 4-bit IQ variants
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ1_S][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq1_s_f32, "mul_mat_iq1_s_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ1_M][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq1_m_f32, "mul_mat_iq1_m_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ4_NL][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq4_nl_f32, "mul_mat_iq4_nl_f32");
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_IQ4_XS][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_mat_iq4_xs_f32, "mul_mat_iq4_xs_f32");

    std::string proc_mul_mat_f32_f32;
    std::string proc_mul_mat_f32_f32_vec;
    std::string proc_mul_mat_f16_f32;
    std::string proc_mul_mat_f16_f32_vec;
    std::string proc_mul_mat_f16_f16;
    std::string proc_mul_mat_f16_f16_vec;
    std::string proc_mul_mat_q4_0_f32;
    std::string proc_mul_mat_q4_0_f32_vec;

    std::vector<wgpu::ConstantEntry> mul_mat_constants;
#ifndef __EMSCRIPTEN__
    if (webgpu_ctx->supports_subgroup_matrix) {
        std::map<std::string, std::string> sg_matrix_repls;
        sg_matrix_repls["WEBGPU_MAX_SUBGROUP_SIZE"] = std::to_string(webgpu_ctx->max_subgroup_size);
        sg_matrix_repls["WEBGPU_TILE_K"]            = std::to_string(WEBGPU_MUL_MAT_TILE_K);
        sg_matrix_repls["WEBGPU_SUBGROUP_M"]        = std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M);
        sg_matrix_repls["WEBGPU_SUBGROUP_N"]        = std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N);
        sg_matrix_repls["WEBGPU_SUBGROUP_MATRIX_M"] = std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M);
        sg_matrix_repls["WEBGPU_SUBGROUP_MATRIX_N"] = std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N);
        sg_matrix_repls["WEBGPU_SG_MAT_M_SIZE"]     = std::to_string(webgpu_ctx->sg_mat_m);
        sg_matrix_repls["WEBGPU_SG_MAT_N_SIZE"]     = std::to_string(webgpu_ctx->sg_mat_n);
        sg_matrix_repls["WEBGPU_SG_MAT_K_SIZE"]     = std::to_string(webgpu_ctx->sg_mat_k);

        proc_mul_mat_f32_f32 = ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f32_f32, sg_matrix_repls);
        proc_mul_mat_f32_f32_vec =
            ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f32_f32_vec, sg_matrix_repls);
        proc_mul_mat_f16_f32 = ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f16_f32, sg_matrix_repls);
        proc_mul_mat_f16_f32_vec =
            ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f16_f32_vec, sg_matrix_repls);
        proc_mul_mat_f16_f16 = ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f16_f16, sg_matrix_repls);
        proc_mul_mat_f16_f16_vec =
            ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_f16_f16_vec, sg_matrix_repls);
        proc_mul_mat_q4_0_f32 =
            ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_q4_0_f32, sg_matrix_repls);
        proc_mul_mat_q4_0_f32_vec =
            ggml_webgpu_process_shader_repls(wgsl_mul_mat_subgroup_matrix_q4_0_f32_vec, sg_matrix_repls);
    } else {
#endif
        mul_mat_constants.push_back({ .key = "TILE_K", .value = WEBGPU_MUL_MAT_TILE_K });
        mul_mat_constants.push_back({ .key = "WORKGROUP_SIZE_M", .value = WEBGPU_MUL_MAT_WG_SIZE_M });
        mul_mat_constants.push_back({ .key = "WORKGROUP_SIZE_N", .value = WEBGPU_MUL_MAT_WG_SIZE_N });

        std::map<std::string, std::string> reg_repls;
        reg_repls["WEBGPU_TILE_M"] = std::to_string(WEBGPU_MUL_MAT_TILE_M);
        reg_repls["WEBGPU_TILE_N"] = std::to_string(WEBGPU_MUL_MAT_TILE_N);

        proc_mul_mat_f32_f32      = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f32_f32, reg_repls);
        proc_mul_mat_f32_f32_vec  = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f32_f32_vec, reg_repls);
        proc_mul_mat_f16_f32      = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f16_f32, reg_repls);
        proc_mul_mat_f16_f32_vec  = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f16_f32_vec, reg_repls);
        proc_mul_mat_f16_f16      = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f16_f16, reg_repls);
        proc_mul_mat_f16_f16_vec  = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_f16_f16_vec, reg_repls);
        proc_mul_mat_q4_0_f32     = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_q4_0_f32, reg_repls);
        proc_mul_mat_q4_0_f32_vec = ggml_webgpu_process_shader_repls(wgsl_mul_mat_reg_tile_q4_0_f32_vec, reg_repls);
#ifndef __EMSCRIPTEN__
    }
#endif

    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F32][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f32_f32.c_str(), "mul_mat_f32_f32", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F32][GGML_TYPE_F32][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f32_f32_vec.c_str(), "mul_mat_f32_f32_vec", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F16][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f16_f32.c_str(), "mul_mat_f16_f32", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F16][GGML_TYPE_F32][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f16_f32_vec.c_str(), "mul_mat_f16_f32_vec", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F16][GGML_TYPE_F16][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f16_f16.c_str(), "mul_mat_f16_f16", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_F16][GGML_TYPE_F16][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_f16_f16_vec.c_str(), "mul_mat_f16_f16_vec", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q4_0][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_q4_0_f32.c_str(), "mul_mat_q4_0_f32", mul_mat_constants);
    webgpu_ctx->mul_mat_pipelines[GGML_TYPE_Q4_0][GGML_TYPE_F32][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, proc_mul_mat_q4_0_f32_vec.c_str(), "mul_mat_q4_0_f32_vec", mul_mat_constants);

    std::vector<wgpu::ConstantEntry> mul_mat_vec_constants(3);
    mul_mat_vec_constants[0].key   = "WORKGROUP_SIZE";
    mul_mat_vec_constants[0].value = WEBGPU_MUL_MAT_VEC_WG_SIZE;
    mul_mat_vec_constants[1].key   = "TILE_K";
    mul_mat_vec_constants[1].value = WEBGPU_MUL_MAT_VEC_TILE_K;
    mul_mat_vec_constants[2].key   = "OUTPUTS_PER_WG";
    mul_mat_vec_constants[2].value = WEBGPU_MUL_MAT_VEC_OUTPUTS_PER_WG;

    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F32][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f32_f32, "mul_mat_vec_f32_f32", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F32][GGML_TYPE_F32][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f32_f32_vec, "mul_mat_vec_f32_f32_vec", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F16][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f16_f32, "mul_mat_vec_f16_f32", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F16][GGML_TYPE_F32][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f16_f32_vec, "mul_mat_vec_f16_f32_vec", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F16][GGML_TYPE_F16][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f16_f16, "mul_mat_vec_f16_f16", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_F16][GGML_TYPE_F16][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_f16_f16_vec, "mul_mat_vec_f16_f16_vec", mul_mat_vec_constants);
    webgpu_ctx->mul_mat_vec_pipelines[GGML_TYPE_Q4_0][GGML_TYPE_F32][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_mul_mat_vec_q4_0_f32, "mul_mat_vec_q4_0_f32", mul_mat_vec_constants);
}

static void ggml_webgpu_init_get_rows_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->get_rows_pipelines[GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_f32, "get_rows_f32", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_f32_vec, "get_rows_f32_vec", constants);

    webgpu_ctx->get_rows_pipelines[GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_f16, "get_rows_f16", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_I32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_i32, "get_rows_i32", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q4_0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q4_0, "get_rows_q4_0", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q4_1][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q4_1, "get_rows_q4_1", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q5_0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q5_0, "get_rows_q5_0", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q5_1][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q5_1, "get_rows_q5_1", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q8_0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q8_0, "get_rows_q8_0", constants);

    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q2_K][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q2_k, "get_rows_q2_k", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q3_K][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q3_k, "get_rows_q3_k", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q4_K][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q4_k, "get_rows_q4_k", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q5_K][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q5_k, "get_rows_q5_k", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_Q6_K][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_q6_k, "get_rows_q6_k", constants);

    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ2_XXS][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq2_xxs, "get_rows_iq2_xxs", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ2_XS][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq2_xs, "get_rows_iq2_xs", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ2_S][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq2_s, "get_rows_iq2_s", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ3_XXS][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq3_xxs, "get_rows_iq3_xxs", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ3_S][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq3_s, "get_rows_iq3_s", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ1_S][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq1_s, "get_rows_iq1_s", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ1_M][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq1_m, "get_rows_iq1_m", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ4_NL][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq4_nl, "get_rows_iq4_nl", constants);
    webgpu_ctx->get_rows_pipelines[GGML_TYPE_IQ4_XS][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_get_rows_iq4_xs, "get_rows_iq4_xs", constants);
}

static void ggml_webgpu_init_cpy_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->cpy_pipelines[GGML_TYPE_F32][GGML_TYPE_F32] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_cpy_f32_f32, "cpy_f32_f32", constants);
    webgpu_ctx->cpy_pipelines[GGML_TYPE_F32][GGML_TYPE_I32] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_cpy_f32_i32, "cpy_f32_i32", constants);
    webgpu_ctx->cpy_pipelines[GGML_TYPE_F32][GGML_TYPE_F16] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_cpy_f32_f16, "cpy_f32_f16", constants);
    webgpu_ctx->cpy_pipelines[GGML_TYPE_F16][GGML_TYPE_F32] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_cpy_f16_f32, "cpy_f16_f32", constants);
    webgpu_ctx->cpy_pipelines[GGML_TYPE_F16][GGML_TYPE_F16] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_cpy_f16_f16, "cpy_f16_f16", constants);
}

static void ggml_webgpu_init_add_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->add_pipelines[GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_add_f32, "add_f32", constants);
    webgpu_ctx->add_pipelines[GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_add_f16, "add_f16", constants);
    webgpu_ctx->add_pipelines[GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_add_f32_inplace, "add_f32_inplace", constants);
    webgpu_ctx->add_pipelines[GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_add_f16_inplace, "add_f16_inplace", constants);
}

static void ggml_webgpu_init_sub_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->sub_pipelines[GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_sub_f32, "sub_f32", constants);
    webgpu_ctx->sub_pipelines[GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_sub_f16, "sub_f16", constants);
    webgpu_ctx->sub_pipelines[GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_sub_f32_inplace, "sub_f32_inplace", constants);
    webgpu_ctx->sub_pipelines[GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_sub_f16_inplace, "sub_f16_inplace", constants);
}

static void ggml_webgpu_init_mul_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->mul_pipelines[GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_f32, "mul_f32", constants);
    webgpu_ctx->mul_pipelines[GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_f16, "mul_f16", constants);
    webgpu_ctx->mul_pipelines[GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_f32_inplace, "mul_f32_inplace", constants);
    webgpu_ctx->mul_pipelines[GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_mul_f16_inplace, "mul_f16_inplace", constants);
}

static void ggml_webgpu_init_div_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->div_pipelines[GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_div_f32, "div_f32", constants);
    webgpu_ctx->div_pipelines[GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_div_f16, "div_f16", constants);
    webgpu_ctx->div_pipelines[GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_div_f32_inplace, "div_f32_inplace", constants);
    webgpu_ctx->div_pipelines[GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_div_f16_inplace, "div_f16_inplace", constants);
}

static void ggml_webgpu_init_rms_norm_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_ROW_SPLIT_WG_SIZE);

    webgpu_ctx->rms_norm_pipelines[0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rms_norm, "rms_norm", constants);
    webgpu_ctx->rms_norm_pipelines[1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rms_norm_inplace, "rms_norm_inplace", constants);
}

static void ggml_webgpu_init_rope_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->rope_pipelines[GGML_TYPE_F32][0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f32, "rope_f32", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F32][0][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f32_inplace, "rope_f32_inplace", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F32][1][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f32_ff, "rope_f32_ff", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F32][1][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f32_ff_inplace, "rope_f32_ff_inplace", constants);

    webgpu_ctx->rope_pipelines[GGML_TYPE_F16][0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f16, "rope_f16", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F16][0][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f16_inplace, "rope_f16_inplace", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F16][1][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f16_ff, "rope_f16_ff", constants);
    webgpu_ctx->rope_pipelines[GGML_TYPE_F16][1][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_rope_f16_ff_inplace, "rope_f16_ff_inplace", constants);
}

static void ggml_webgpu_init_glu_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    // REGLU
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_REGLU][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_reglu_f32, "reglu_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_REGLU][GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_reglu_f16, "reglu_f16", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_REGLU][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_reglu_f32_split, "reglu_f32_split", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_REGLU][GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_reglu_f16_split, "reglu_f16_split", constants);

    // GEGLU
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_f32, "geglu_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU][GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_f16, "geglu_f16", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_f32_split, "geglu_f32_split", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU][GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_f16_split, "geglu_f16_split", constants);

    // SWIGLU
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_f32, "swiglu_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU][GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_f16, "swiglu_f16", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_f32_split, "swiglu_f32_split", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU][GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_f16_split, "swiglu_f16_split", constants);

    // SWIGLU_OAI
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU_OAI][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_oai_f32, "swiglu_oai_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_SWIGLU_OAI][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_swiglu_oai_f32_split, "swiglu_oai_f32_split", constants);

    // GEGLU_ERF
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_erf_f32, "geglu_erf_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_erf_f16, "geglu_erf_f16", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_erf_f32_split, "geglu_erf_f32_split", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_ERF][GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_erf_f16_split, "geglu_erf_f16_split", constants);

    // GEGLU_QUICK
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F32][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_quick_f32, "geglu_quick_f32", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F16][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_quick_f16, "geglu_quick_f16", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F32][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_quick_f32_split, "geglu_quick_f32_split", constants);
    webgpu_ctx->glu_pipelines[GGML_GLU_OP_GEGLU_QUICK][GGML_TYPE_F16][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_geglu_quick_f16_split, "geglu_quick_f16_split", constants);
}

static void ggml_webgpu_init_scale_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_MAX_WG_SIZE);

    webgpu_ctx->scale_pipelines[0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_scale_f32, "scale_f32", constants);
    webgpu_ctx->scale_pipelines[1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_scale_f32_inplace, "scale_f32_inplace", constants);
}

static void ggml_webgpu_init_soft_max_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants = ggml_webgpu_wg_size_entry(WEBGPU_ROW_SPLIT_WG_SIZE);

    // f32 (no mask)
    webgpu_ctx->soft_max_pipelines[2][0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_soft_max_f32, "soft_max_f32", constants);
    webgpu_ctx->soft_max_pipelines[2][0][1] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_soft_max_f32_inplace, "soft_max_f32_inplace", constants);
    webgpu_ctx->soft_max_pipelines[2][1][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_soft_max_f32_sink, "soft_max_f32_sink", constants);
    webgpu_ctx->soft_max_pipelines[2][1][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_sink_inplace, "soft_max_f32_sink_inplace", constants);

    // f32 mask (mask_type = 0)
    webgpu_ctx->soft_max_pipelines[0][0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_soft_max_f32_mask_f32, "soft_max_f32_mask_f32", constants);
    webgpu_ctx->soft_max_pipelines[0][0][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f32_inplace, "soft_max_f32_mask_f32_inplace", constants);
    webgpu_ctx->soft_max_pipelines[0][1][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f32_sink, "soft_max_f32_mask_f32_sink", constants);
    webgpu_ctx->soft_max_pipelines[0][1][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f32_sink_inplace, "soft_max_f32_mask_f32_sink_inplace", constants);

    // f16 mask (mask_type = 1)
    webgpu_ctx->soft_max_pipelines[1][0][0] =
        ggml_webgpu_create_pipeline(webgpu_ctx->device, wgsl_soft_max_f32_mask_f16, "soft_max_f32_mask_f16", constants);
    webgpu_ctx->soft_max_pipelines[1][0][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f16_inplace, "soft_max_f32_mask_f16_inplace", constants);
    webgpu_ctx->soft_max_pipelines[1][1][0] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f16_sink, "soft_max_f32_mask_f16_sink", constants);
    webgpu_ctx->soft_max_pipelines[1][1][1] = ggml_webgpu_create_pipeline(
        webgpu_ctx->device, wgsl_soft_max_f32_mask_f16_sink_inplace, "soft_max_f32_mask_f16_sink_inplace", constants);
}

// TODO: move most initialization logic here
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
                        /* .get_alloc_size   = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
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
            // TODO: support non-contiguous tensors, e.g. for MOE_EXPERT_REDUCE
            // see https://github.com/ggml-org/llama.cpp/pull/16857
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type) && ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            supports_op = ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) ||
                          (op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32);
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
        case GGML_OP_FLASH_ATTN_EXT:
            {
                if (!webgpu_ctx->supports_subgroup_matrix) {
                    break;
                }
                // Head dimensions must fit in workgroup memory with minimum tile sizes
                size_t     limit_bytes = webgpu_ctx->limits.maxComputeWorkgroupStorageSize;
                const bool has_mask    = op->src[3] != nullptr;
                const bool kv_direct   = src1->type == GGML_TYPE_F16 && (src0->ne[0] % webgpu_ctx->sg_mat_k) == 0 &&
                                       (src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD) == 0;
                const size_t min_bytes = ggml_webgpu_flash_attn_wg_mem_bytes(
                    webgpu_ctx->sg_mat_m, webgpu_ctx->sg_mat_n, (uint32_t) src0->ne[0], (uint32_t) src2->ne[0],
                    has_mask, kv_direct);
                if (min_bytes > limit_bytes) {
                    break;
                }

                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              src2->type == src1->type && op->type == GGML_TYPE_F32;
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
        case GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            break;
        case GGML_OP_FILL:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            break;
        case GGML_OP_LOG:
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
// TODO: move most logic to device_init function so backend can be freed/initialized properly
// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    webgpu_context ctx = reg_ctx->webgpu_ctx;

    wgpu::RequestAdapterOptions options = {};

#ifndef __EMSCRIPTEN__
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    const char * const          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabledToggleCount = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

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

    wgpu::AdapterInfo info{};
#ifndef __EMSCRIPTEN__
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    if (ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.nextInChain = &subgroup_matrix_configs;
    }
#endif
    ctx->adapter.GetInfo(&info);

    wgpu::SupportedFeatures features;
    ctx->adapter.GetFeatures(&features);
    // we require f16 support
    GGML_ASSERT(ctx->adapter.HasFeature(wgpu::FeatureName::ShaderF16));

#ifndef __EMSCRIPTEN__
    // Only support square f16 matrices of size 8 or 16 for now
    bool valid_subgroup_matrix_config = false;
    if (ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
            const wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            if (config.M == config.N && config.N == config.K && (config.K == 8 || config.K == 16) &&
                config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->sg_mat_m                = config.M;
                ctx->sg_mat_n                = config.N;
                ctx->sg_mat_k                = config.K;
                valid_subgroup_matrix_config = true;
                break;
            }
        }
    }

    ctx->supports_subgroup_matrix = valid_subgroup_matrix_config;
#endif
    // For subgroup matrix code to be the most efficient, we would like the subgroup size to be consistent and accurate.
    // Unfortunately, that is not possible, so we use the maximum subgroup size reported by the adapter.
    ctx->max_subgroup_size = info.subgroupMaxSize;

    // Initialize device
    std::vector<wgpu::FeatureName> required_features = { wgpu::FeatureName::ShaderF16 };

#ifndef __EMSCRIPTEN__
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    if (ctx->supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_UNUSED(reason);
            GGML_UNUSED(message);
            //TODO: uncomment once proper free logic is in place
            //GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
            //std::string(message).c_str());
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
    const char * const deviceEnabledToggles[]  = { "skip_validation", "disable_robustness", "disable_workgroup_init",
                                                   "disable_polyfills_on_integer_div_and_mod" };
    const char * const deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabledToggleCount  = 4;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabledToggleCount = 1;

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

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

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Initialize buffer pool for timestamp queries (profiling)
    ctx->timestamp_query_buf_pool.init(ctx->device, WEBGPU_NUM_TIMESTAMP_QUERY_BUFS,
                                       WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                                       wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
                                       wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
#endif

    ctx->set_rows_error_buf_pool.init(ctx->device, WEBGPU_NUM_SET_ROWS_ERROR_BUFS, WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                                      wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::Storage,
                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);

    ggml_webgpu_init_memset_pipeline(ctx);
    ggml_webgpu_init_mul_mat_pipeline(ctx);
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

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, ctx);
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

#ifndef __EMSCRIPTEN__
    const char * const          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabledToggleCount = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    webgpu_ctx->instance = wgpu::CreateInstance(&instance_descriptor);

#ifdef __EMSCRIPTEN__
    if (webgpu_ctx->instance == nullptr) {
        GGML_LOG_ERROR("ggml_webgpu: Failed to create WebGPU instance. Make sure either -sASYNCIFY or -sJSPI is set\n");
        return nullptr;
    }
#endif
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

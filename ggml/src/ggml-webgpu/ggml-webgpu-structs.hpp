#ifndef GGML_WEBGPU_STRUCTS_HPP
#define GGML_WEBGPU_STRUCTS_HPP

#include "ggml.h"
#include "pre_wgsl.hpp"

#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

void ggml_webgpu_create_buffer(wgpu::Device &    device,
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
};

struct webgpu_command {
    wgpu::CommandBuffer             commands;
    webgpu_pool_bufs                params_bufs;
    std::optional<webgpu_pool_bufs> set_rows_error_bufs;
#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_gpu_profile_bufs timestamp_query_bufs;
    std::string             pipeline_name;
#endif
};

struct flash_attn_pipeline_key {
    int      q_type;
    int      kv_type;
    int      mask_type;
    int      sinks_type;
    int      dst_type;
    uint32_t head_dim_q;
    uint32_t head_dim_v;
    uint32_t n_heads;
    bool     has_mask;
    bool     has_sinks;
    bool     uses_logit_softcap;

    bool operator==(const flash_attn_pipeline_key & other) const {
        return q_type == other.q_type && kv_type == other.kv_type && mask_type == other.mask_type &&
               sinks_type == other.sinks_type && dst_type == other.dst_type && head_dim_q == other.head_dim_q &&
               head_dim_v == other.head_dim_v && n_heads == other.n_heads && has_mask == other.has_mask &&
               has_sinks == other.has_sinks && uses_logit_softcap == other.uses_logit_softcap;
    }
};

struct flash_attn_pipeline_key_hash {
    size_t operator()(const flash_attn_pipeline_key & key) const {
        size_t seed = 0;
        auto   mix  = [&seed](size_t value) {
            seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        mix(std::hash<int>{}(key.q_type));
        mix(std::hash<int>{}(key.kv_type));
        mix(std::hash<int>{}(key.mask_type));
        mix(std::hash<int>{}(key.sinks_type));
        mix(std::hash<int>{}(key.dst_type));
        mix(std::hash<uint32_t>{}(key.head_dim_q));
        mix(std::hash<uint32_t>{}(key.head_dim_v));
        mix(std::hash<uint32_t>{}(key.n_heads));
        mix(std::hash<bool>{}(key.has_mask));
        mix(std::hash<bool>{}(key.has_sinks));
        mix(std::hash<bool>{}(key.uses_logit_softcap));
        return seed;
    }
};

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    wgpu::Limits   limits;

    uint32_t subgroup_size;

#ifndef __EMSCRIPTEN__
    bool                       supports_subgroup_matrix = false;
    wgpu::SubgroupMatrixConfig subgroup_matrix_config;
#endif

    std::recursive_mutex mutex;
    std::atomic_uint     inflight_threads = 0;

    webgpu_buf_pool param_buf_pool;
    webgpu_buf_pool set_rows_error_buf_pool;

    pre_wgsl::Preprocessor p;

    std::map<int, webgpu_pipeline> memset_pipelines;                                 // variant or type index

    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> mul_mat_pipelines;  // src0_type, src1_type, vectorized
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>>
        mul_mat_vec_pipelines;                                                       // src0_type, src1_type, vectorized

    std::unordered_map<flash_attn_pipeline_key, webgpu_pipeline, flash_attn_pipeline_key_hash> flash_attn_pipelines;

    std::map<int, std::map<int, webgpu_pipeline>> set_rows_pipelines;                 // dst_type, vectorized
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
    std::map<int, std::map<int, std::map<int, webgpu_pipeline>>> unary_pipelines;     // unary_op, type, inplace

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

using webgpu_context = std::shared_ptr<webgpu_context_struct>;

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

#endif  // GGML_WEBGPU_STRUCTS_HPP

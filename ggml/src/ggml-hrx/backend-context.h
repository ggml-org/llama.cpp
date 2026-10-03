#pragma once

#include "ggml-backend-impl.h"
#include "graph/value-map.h"
#include "runtime/graph-program-cache.h"
#include "runtime/host-buffer-registry.h"
#include "runtime/host-memory.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;

struct ggml_backend_hrx_device_context;

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device;
    std::string                       name;
    bool                              host_visible = false;
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device;
    hrx_buffer_t                      buffer;
    uint8_t *                         base;
    uint64_t                          identity;
    uint64_t                          generation;
    bool                              direct_host_binding;
};

struct ggml_backend_hrx_device_context {
    hrx_device_t                                   device = nullptr;
    std::string                                    name;
    std::string                                    description;
    std::string                                    architecture;
    size_t                                         memory_total = 0;
    bool                                           use_direct_host_bindings = false;
    ggml_backend_buffer_type                       buft               = {};
    ggml_backend_hrx_buffer_type_context           buft_context       = {};
    ggml_backend_buffer_type                       host_buft          = {};
    ggml_backend_hrx_buffer_type_context           host_buft_context  = {};
    ggml::hrx::HostBufferRegistry                  host_buffers;
    std::atomic<uint64_t>                          synchronous_upload_fallbacks{ 0 };
    std::atomic<uint64_t>                          synchronous_download_fallbacks{ 0 };
    std::mutex                                     buffer_stream_mutex;
    hrx_stream_t                                   buffer_stream = nullptr;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context *      device;
    hrx_stream_t                           stream;
    ggml::hrx::KernelExecutableCache       kernel_executables;
    ggml::hrx::GraphProgramCache           graph_programs;
    ggml::hrx::PreparedCommandProgramCache prepared_programs;
    ggml::hrx::TransientArena              transient_arena;
    ggml::hrx::HostTransferManager         host_transfers;
    ggml::hrx::HostWeightCache             host_weights;
    std::string                            name;
};

struct ggml_backend_hrx_reg_context {
    bool                                                          initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device>                              devices;

    ~ggml_backend_hrx_reg_context();
};

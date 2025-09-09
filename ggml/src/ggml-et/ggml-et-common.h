#pragma once

#include <runtime/IRuntime.h>
#include <device-layer/IDeviceLayer.h>
#include <string>
#include <unordered_map>
#include "ggml-backend-impl.h"

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

struct ggml_backend_et_buffer_type_context {
    int devidx;
    std::string name;
};

struct ggml_backend_et_buffer_context {
    int devidx;
    void * data;                    // Device memory pointer
    size_t size;
    rt::DeviceId rtid;
};

struct ggml_backend_et_context {
    int devidx;
};

struct ggml_backend_et_device_context {
    int devidx;
    rt::DeviceId rtid;
    std::string name;
    std::string desc;
    size_t total_mem;
    ggml_backend_buffer_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;
};

struct ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
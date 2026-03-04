#pragma once

// Internal shared definitions for the CXL backend

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "cxl-device.h"

#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#define GGML_CXL_NAME "CXL"
#define GGML_CXL_LOG  "ggml-cxl: "

// Device context shared across device interface, buffer type, and backend
struct ggml_backend_cxl_device_context {
    int              index;      // device index
    std::string      name;       // "CXL0", "CXL1", ...
    std::string      description;
    struct cxl_device cxl_dev;   // low-level CXL device handle
    std::mutex       cmd_mutex;  // protect concurrent command execution
};

// Buffer context
struct ggml_backend_cxl_buffer_context {
    ggml_backend_cxl_device_context * dev_ctx;
    uint64_t dev_ptr;  // device memory pointer
    size_t   size;
};

// Helper macros
#define CXL_DEV_CTX(dev) ((ggml_backend_cxl_device_context *)(dev)->context)
#define CXL_BUF_CTX(buf) ((ggml_backend_cxl_buffer_context *)(buf)->context)

// Extern declarations for interfaces defined in separate files
extern const ggml_backend_buffer_type_i ggml_backend_cxl_buffer_type_interface;
extern const ggml_backend_buffer_i      ggml_backend_cxl_buffer_interface;
extern const ggml_backend_device_i      ggml_backend_cxl_device_interface;

// Functions used across files
ggml_backend_t             ggml_backend_cxl_device_init(ggml_backend_dev_t dev, const char * params);
ggml_backend_buffer_type_t ggml_backend_cxl_device_get_buffer_type(ggml_backend_dev_t dev);

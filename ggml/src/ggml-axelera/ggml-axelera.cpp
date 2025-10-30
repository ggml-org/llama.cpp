/*
 * Copyright (c) 2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-axelera.h"
#include "ggml-axelera-compiler.h"
#include "ggml-impl.h"

// Configuration options (can be set via environment variables)
static size_t opt_ndev = 1;  // Number of Axelera devices to use

#define AXELERA_LOG_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AXELERA_LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define AXELERA_LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

//
// Axelera Device Context
//

struct ggml_axelera_device_context {
    int32_t device_id;
    std::string name;
    std::string description;
    size_t free_memory;
    size_t total_memory;

    ggml_axelera_device_context(int32_t id) : device_id(id) {
        name = "Axelera" + std::to_string(id);
        description = "Axelera AI Accelerator Device " + std::to_string(id);

        // TODO: Query actual device memory from Axelera driver
        // For now, use placeholder values
        total_memory = 4ULL * 1024 * 1024 * 1024; // 4GB
        free_memory = total_memory;
    }

    ~ggml_axelera_device_context() {
        AXELERA_LOG_DEBUG("Releasing Axelera device %d\n", device_id);
    }
};

//
// Buffer Type Implementation
//

struct ggml_backend_axelera_buffer_context {
    int32_t device_id;
    void* data;
    size_t size;

    ggml_backend_axelera_buffer_context(int32_t device, size_t size)
        : device_id(device), size(size) {
        // TODO: Allocate memory on Axelera device
        // For now, use host memory as placeholder
        data = malloc(size);
        if (data == nullptr) {
            throw std::runtime_error("Failed to allocate Axelera buffer");
        }
    }

    ~ggml_backend_axelera_buffer_context() {
        // TODO: Free memory on Axelera device
        free(data);
    }
};

// Buffer interface functions
static void ggml_backend_axelera_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto* ctx = static_cast<ggml_backend_axelera_buffer_context*>(buffer->context);
    delete ctx;
}

static void* ggml_backend_axelera_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto* ctx = static_cast<ggml_backend_axelera_buffer_context*>(buffer->context);
    return ctx->data;
}

static enum ggml_status ggml_backend_axelera_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_axelera_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor,
    const void* data, size_t offset, size_t size) {
    auto* ctx = static_cast<ggml_backend_axelera_buffer_context*>(buffer->context);
    // Calculate tensor offset in buffer
    size_t tensor_offset = (char*)tensor->data - (char*)ctx->data;
    memcpy((char*)ctx->data + tensor_offset + offset, data, size);
}

static void ggml_backend_axelera_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor,
    void* data, size_t offset, size_t size) {
    auto* ctx = static_cast<ggml_backend_axelera_buffer_context*>(buffer->context);
    // Calculate tensor offset in buffer
    size_t tensor_offset = (const char*)tensor->data - (const char*)ctx->data;
    memcpy(data, (const char*)ctx->data + tensor_offset + offset, size);
}

static bool ggml_backend_axelera_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void ggml_backend_axelera_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* ctx = static_cast<ggml_backend_axelera_buffer_context*>(buffer->context);
    memset(ctx->data, value, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_axelera_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_axelera_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_axelera_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_axelera_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_axelera_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_axelera_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_axelera_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_axelera_buffer_clear,
    /* .reset           = */ nullptr,
};

// Buffer type interface functions
static const char* ggml_backend_axelera_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Axelera";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_axelera_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    auto* buft_ctx = static_cast<int32_t*>(buft->context);
    int32_t device_id = *buft_ctx;

    try {
        auto* ctx = new ggml_backend_axelera_buffer_context(device_id, size);
        return new ggml_backend_buffer{
            /* .iface   = */ ggml_backend_axelera_buffer_interface,
            /* .buft    = */ buft,
            /* .context = */ ctx,
            /* .size    = */ size,
            /* .usage   = */ GGML_BACKEND_BUFFER_USAGE_ANY,
        };
    } catch (...) {
        return nullptr;
    }
}

static size_t ggml_backend_axelera_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32; // 32-byte alignment for Axelera
    GGML_UNUSED(buft);
}

static size_t ggml_backend_axelera_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_axelera_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static bool ggml_backend_axelera_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_axelera_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_axelera_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_axelera_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_axelera_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_axelera_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_axelera_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_axelera_buffer_type_is_host,
};

// Device buffer types (one per device)
static ggml_backend_buffer_type ggml_backend_axelera_buffer_types[GGML_AXELERA_MAX_DEVICES];
static int32_t ggml_backend_axelera_buffer_type_devices[GGML_AXELERA_MAX_DEVICES];

ggml_backend_buffer_type_t ggml_backend_axelera_buffer_type(int32_t device) {
    if (device < 0 || device >= GGML_AXELERA_MAX_DEVICES) {
        return nullptr;
    }

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (ggml_backend_axelera_buffer_types[device].iface.get_name == nullptr) {
        ggml_backend_axelera_buffer_type_devices[device] = device;
        ggml_backend_axelera_buffer_types[device] = {
            /* .iface   = */ ggml_backend_axelera_buffer_type_interface,
            /* .device  = */ nullptr,  // Will be set by device
            /* .context = */ &ggml_backend_axelera_buffer_type_devices[device],
        };
    }

    return &ggml_backend_axelera_buffer_types[device];
}

//
// Backend Implementation
//

static const char* ggml_backend_axelera_name(ggml_backend_t backend) {
    AXELERA_LOG_DEBUG("[TRACE] backend.get_name(backend=%p)\n", (void*)backend);
    auto* ctx = static_cast<ggml_axelera_device_context*>(backend->context);
    AXELERA_LOG_DEBUG("[TRACE]   -> returning: %s\n", ctx->name.c_str());
    return ctx->name.c_str();
}

static void ggml_backend_axelera_free(ggml_backend_t backend) {
    AXELERA_LOG_DEBUG("[TRACE] backend.free(backend=%p)\n", (void*)backend);
    // Context is managed by device, so we don't delete it here
    delete backend;
    AXELERA_LOG_DEBUG("[TRACE]   -> freed\n");
}

static ggml_status ggml_backend_axelera_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {

    AXELERA_LOG_DEBUG("[TRACE] backend.graph_compute(backend=%p, cgraph=%p, n_nodes=%d)\n",
                      (void*)backend, (void*)cgraph, cgraph->n_nodes);

    // Print detailed graph information if debug logging is enabled
    if (getenv("GGML_LOG_LEVEL") && strcmp(getenv("GGML_LOG_LEVEL"), "debug") == 0) {
        AXELERA_LOG_DEBUG("=== Axelera Graph Compute (Debug) ===\n");

        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor* node = cgraph->nodes[i];

            AXELERA_LOG_DEBUG("Node %d: %s\n", i, node->name);
            AXELERA_LOG_DEBUG("  Operation: %s\n", ggml_op_name(node->op));
            AXELERA_LOG_DEBUG("  Type: %s\n", ggml_type_name(node->type));
            AXELERA_LOG_DEBUG("  Shape: [%lld, %lld, %lld, %lld]\n",
                            node->ne[0], node->ne[1], node->ne[2], node->ne[3]);

            // Print source tensors
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s] != nullptr) {
                    ggml_tensor* src = node->src[s];
                    AXELERA_LOG_DEBUG("  Src[%d]: %s, type=%s, shape=[%lld, %lld, %lld, %lld]\n",
                                    s, src->name, ggml_type_name(src->type),
                                    src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
                }
            }
        }

        AXELERA_LOG_DEBUG("=== End Graph Info ===\n");
    }

    // Use graph planning internally for compilation and execution
    AXELERA_LOG_DEBUG("Creating graph plan...\n");
    ggml_backend_graph_plan_t plan = ggml_axelera_graph_plan_create(backend, cgraph);

    if (!plan) {
        AXELERA_LOG_ERROR("Failed to create graph plan, falling back to CPU\n");
        return GGML_STATUS_FAILED;
    }

    AXELERA_LOG_DEBUG("Executing graph plan...\n");
    ggml_status status = ggml_axelera_graph_plan_compute(backend, plan);

    AXELERA_LOG_DEBUG("Freeing graph plan...\n");
    ggml_axelera_graph_plan_free(backend, plan);

    if (status != GGML_STATUS_SUCCESS) {
        AXELERA_LOG_ERROR("Graph execution failed\n");
    }

    AXELERA_LOG_DEBUG("[TRACE]   -> returning status=%d (%s)\n",
                      status, status == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");
    return status;
}

static bool ggml_backend_axelera_supports_op(ggml_backend_t backend, const ggml_tensor* op) {
    AXELERA_LOG_DEBUG("[TRACE] backend.supports_op(backend=%p, op=%p, op_type=%s, op_name=%s)\n",
                      (void*)backend, (void*)op, ggml_op_name(op->op), op->name);

    // TODO: Check which operations are supported on Axelera
    // For now, return false for all operations
    bool result = false;

    AXELERA_LOG_DEBUG("[TRACE]   -> returning: %s\n", result ? "true" : "false");
    return result;

    GGML_UNUSED(backend);
}

static bool ggml_backend_axelera_supports_buft(
    ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    const char* buft_name = buft->iface.get_name ? buft->iface.get_name(buft) : "unknown";
    AXELERA_LOG_DEBUG("[TRACE] backend.supports_buft(backend=%p, buft=%p, buft_name=%s)\n",
                      (void*)backend, (void*)buft, buft_name);

    // Support Axelera buffers and host buffers
    bool result = buft->iface.get_name == ggml_backend_axelera_buffer_type_get_name ||
                  ggml_backend_buft_is_host(buft);

    AXELERA_LOG_DEBUG("[TRACE]   -> returning: %s\n", result ? "true" : "false");
    return result;

    GGML_UNUSED(backend);
}

static bool ggml_backend_axelera_offload_op(ggml_backend_t backend, const ggml_tensor* op) {
    AXELERA_LOG_DEBUG("[TRACE] backend.offload_op(backend=%p, op=%p, op_type=%s, op_name=%s)\n",
                      (void*)backend, (void*)op, ggml_op_name(op->op), op->name);

    // TODO: Implement operation offload decision logic
    bool result = ggml_backend_axelera_supports_op(backend, op);

    AXELERA_LOG_DEBUG("[TRACE]   -> returning: %s\n", result ? "true" : "false");
    return result;
}

static struct ggml_backend_i ggml_backend_axelera_interface = {
    /* .get_name             = */ ggml_backend_axelera_name,
    /* .free                 = */ ggml_backend_axelera_free,
    /* .set_tensor_async     = */ nullptr,
    /* .get_tensor_async     = */ nullptr,
    /* .cpy_tensor_async     = */ nullptr,
    /* .synchronize          = */ nullptr,
    /* .graph_plan_create    = */ ggml_axelera_graph_plan_create,
    /* .graph_plan_free      = */ ggml_axelera_graph_plan_free,
    /* .graph_plan_update    = */ nullptr,
    /* .graph_plan_compute   = */ ggml_axelera_graph_plan_compute,
    /* .graph_compute        = */ ggml_backend_axelera_graph_compute,
    /* .event_record         = */ nullptr,
    /* .event_wait           = */ nullptr,
    /* .graph_optimize       = */ nullptr,
};

static ggml_guid_t ggml_backend_axelera_guid() {
    static ggml_guid guid = { 0xa1, 0xe1, 0xe7, 0xa0, 0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

//
// Device Interface Implementation
//

static ggml_backend_t ggml_backend_axelera_device_init(
    ggml_backend_dev_t dev, const char* params) {
    auto* dev_ctx = static_cast<ggml_axelera_device_context*>(dev->context);

    AXELERA_LOG_INFO("Initializing Axelera backend for device %d\n", dev_ctx->device_id);

    return new ggml_backend{
        /* .guid     = */ ggml_backend_axelera_guid(),
        /* .iface    = */ ggml_backend_axelera_interface,
        /* .device   = */ dev,
        /* .context  = */ dev_ctx,
    };

    GGML_UNUSED(params);
}

static const char* ggml_backend_axelera_device_get_name(ggml_backend_dev_t dev) {
    auto* ctx = static_cast<ggml_axelera_device_context*>(dev->context);
    return ctx->name.c_str();
}

static const char* ggml_backend_axelera_device_get_description(ggml_backend_dev_t dev) {
    auto* ctx = static_cast<ggml_axelera_device_context*>(dev->context);
    return ctx->description.c_str();
}

static void ggml_backend_axelera_device_get_memory(
    ggml_backend_dev_t dev, size_t* free, size_t* total) {
    auto* ctx = static_cast<ggml_axelera_device_context*>(dev->context);

    // TODO: Query actual memory from Axelera driver
    *free = ctx->free_memory;
    *total = ctx->total_memory;
}

static enum ggml_backend_dev_type ggml_backend_axelera_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_axelera_device_get_props(
    ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = ggml_backend_axelera_device_get_name(dev);
    props->description = ggml_backend_axelera_device_get_description(dev);
    props->type = ggml_backend_axelera_device_get_type(dev);
    ggml_backend_axelera_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_axelera_device_get_buffer_type(
    ggml_backend_dev_t dev) {
    auto* ctx = static_cast<ggml_axelera_device_context*>(dev->context);
    return ggml_backend_axelera_buffer_type(ctx->device_id);
}

static ggml_backend_buffer_type_t ggml_backend_axelera_device_get_host_buffer_type(
    ggml_backend_dev_t dev) {
    // TODO: Implement pinned host memory buffer type
    return nullptr;
    GGML_UNUSED(dev);
}

static bool ggml_backend_axelera_device_supports_op(
    ggml_backend_dev_t dev, const ggml_tensor* op) {
    // TODO: Implement operation support check
    // For now, return false for all operations
    return false;

    GGML_UNUSED(dev);
    GGML_UNUSED(op);
}

static bool ggml_backend_axelera_device_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_axelera_buffer_type_get_name ||
           ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static bool ggml_backend_axelera_device_offload_op(
    ggml_backend_dev_t dev, const ggml_tensor* op) {
    return ggml_backend_axelera_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_axelera_device_interface = {
    /* .get_name             = */ ggml_backend_axelera_device_get_name,
    /* .get_description      = */ ggml_backend_axelera_device_get_description,
    /* .get_memory           = */ ggml_backend_axelera_device_get_memory,
    /* .get_type             = */ ggml_backend_axelera_device_get_type,
    /* .get_props            = */ ggml_backend_axelera_device_get_props,
    /* .init_backend         = */ ggml_backend_axelera_device_init,
    /* .get_buffer_type      = */ ggml_backend_axelera_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_axelera_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_axelera_device_supports_op,
    /* .supports_buft        = */ ggml_backend_axelera_device_supports_buft,
    /* .offload_op           = */ ggml_backend_axelera_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// Backend Registry
//

struct ggml_axelera_registry {
    ggml_backend_device devices[GGML_AXELERA_MAX_DEVICES];

    ggml_axelera_registry(ggml_backend_reg_t reg);
    ~ggml_axelera_registry();
};

ggml_axelera_registry::ggml_axelera_registry(ggml_backend_reg_t reg) {
    AXELERA_LOG_INFO("Initializing Axelera backend registry with %zu devices\n", opt_ndev);

    // TODO: Detect actual Axelera devices
    // For now, create mock devices

    for (size_t i = 0; i < opt_ndev && i < GGML_AXELERA_MAX_DEVICES; i++) {
        devices[i].iface = ggml_backend_axelera_device_interface;
        devices[i].reg = reg;
        try {
            devices[i].context = new ggml_axelera_device_context(i);
        } catch (...) {
            AXELERA_LOG_ERROR("Failed to create Axelera device context for device %zu\n", i);
            devices[i].context = nullptr;
        }
    }
}

ggml_axelera_registry::~ggml_axelera_registry() {
    AXELERA_LOG_INFO("Releasing Axelera backend registry\n");

    for (size_t i = 0; i < opt_ndev && i < GGML_AXELERA_MAX_DEVICES; i++) {
        auto* ctx = static_cast<ggml_axelera_device_context*>(devices[i].context);
        delete ctx;
    }
}

static const char* ggml_backend_axelera_reg_get_name(ggml_backend_reg_t reg) {
    return "Axelera";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_axelera_reg_get_device_count(ggml_backend_reg_t reg) {
    return opt_ndev;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_axelera_reg_get_device(
    ggml_backend_reg_t reg, size_t index) {
    auto* registry = static_cast<ggml_axelera_registry*>(reg->context);

    if (index >= opt_ndev || index >= GGML_AXELERA_MAX_DEVICES) {
        return nullptr;
    }

    if (registry->devices[index].context == nullptr) {
        return nullptr;
    }

    return &registry->devices[index];
}

static void* ggml_backend_axelera_get_proc_address(ggml_backend_reg_t reg, const char* name) {
    // No special proc addresses for now
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static void ggml_backend_axelera_reg_init(ggml_backend_reg* reg) {
    // Parse environment variables
    const char* str_ndev = getenv("GGML_AXELERA_NDEV");
    if (str_ndev) {
        opt_ndev = strtoul(str_ndev, nullptr, 0);
        if (opt_ndev > GGML_AXELERA_MAX_DEVICES) {
            opt_ndev = GGML_AXELERA_MAX_DEVICES;
        }
    }

    // TODO: Detect actual number of Axelera devices
    // For now, use environment variable or default

    reg->context = new ggml_axelera_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_axelera_reg_interface = {
    /* .get_name         = */ ggml_backend_axelera_reg_get_name,
    /* .get_device_count = */ ggml_backend_axelera_reg_get_device_count,
    /* .get_device       = */ ggml_backend_axelera_reg_get_device,
    /* .get_proc_address = */ ggml_backend_axelera_get_proc_address,
};

//
// Public API Implementation
//

ggml_backend_reg_t ggml_backend_axelera_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_axelera_reg_interface,
        /* .context     = */ nullptr
    };

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (!initialized) {
        ggml_backend_axelera_reg_init(&reg);
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_axelera_init(int32_t device) {
    ggml_backend_reg_t reg = ggml_backend_axelera_reg();

    if (device < 0 || device >= (int32_t)ggml_backend_axelera_reg_get_device_count(reg)) {
        AXELERA_LOG_ERROR("Invalid device index: %d\n", device);
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_axelera_reg_get_device(reg, device);
    if (dev == nullptr) {
        AXELERA_LOG_ERROR("Device %d is not available\n", device);
        return nullptr;
    }

    return ggml_backend_axelera_device_init(dev, nullptr);
}

bool ggml_backend_is_axelera(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_axelera_name;
}

int32_t ggml_backend_axelera_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_axelera_reg();
    return (int32_t)ggml_backend_axelera_reg_get_device_count(reg);
}

ggml_backend_buffer_type_t ggml_backend_axelera_host_buffer_type(void) {
    // TODO: Implement pinned host memory buffer type
    return nullptr;
}

void ggml_backend_axelera_get_device_description(
    int32_t device, char* description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_axelera_reg();
    ggml_backend_dev_t dev = ggml_backend_axelera_reg_get_device(reg, device);

    if (dev == nullptr) {
        snprintf(description, description_size, "Invalid device");
        return;
    }

    auto* ctx = static_cast<ggml_axelera_device_context*>(dev->context);
    snprintf(description, description_size, "%s", ctx->description.c_str());
}

void ggml_backend_axelera_get_device_memory(
    int32_t device, size_t* free, size_t* total) {
    ggml_backend_reg_t reg = ggml_backend_axelera_reg();
    ggml_backend_dev_t dev = ggml_backend_axelera_reg_get_device(reg, device);

    if (dev == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }

    ggml_backend_axelera_device_get_memory(dev, free, total);
}

// Dynamic loading support
GGML_BACKEND_DL_IMPL(ggml_backend_axelera_reg)

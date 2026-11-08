#pragma once

#include "ggml-et-common.h"

#include <string>
#include <utility>
#include <vector>

#define ET_TRACE_BUFFER_SIZE (1024 * 1024 * 8UL)

// Load kernel from file or embedded data and store handle in device context
// Returns true on success, false on failure
//
// Loading strategy:
// - If GGML_ET_KERNELS_PATH env var is set: tries to load from ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf
// - If file not found or env var not set: falls back to embedded kernel data
// - Returns false if kernel cannot be loaded from either source
//
// Kernel is loaded using the device's default stream
bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name);

// Launch kernel with parameters on device's default stream
// Performs lazy loading: automatically loads kernel if not already loaded
// Kernel path: ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf (default: /opt/et/ggml/kernels/)
// Returns true on success, false on failure
// Execution is synchronous - waits for completion
bool ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           const std::string &              kernel_name,
                           void *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask       = 0xFFFFFFFF,
                           bool                             enable_print     = false,
                           bool                             sync_error_check = false);

void ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx);
bool ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx);
void ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx);
bool ggml_et_uberkernel_failed(const ggml_backend_et_uberkernel_context * uk_ctx);

// Unload kernel from device and free resources
// Safe to call even if kernel not loaded
void ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name);

// Unload all kernels from device context
// Called during device cleanup
void ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx);

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx);

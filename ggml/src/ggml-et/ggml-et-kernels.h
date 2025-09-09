#pragma once

#include "ggml-et-common.h"
#include <string>
#include <vector>

// Read kernel ELF file into memory buffer
// Returns empty vector on failure
std::vector<std::byte> ggml_et_read_kernel_file(const std::string& kernel_path);

// Load kernel from ELF file and store handle in device context
// Returns true on success, false on failure
// Kernel is loaded using the device's default stream
bool ggml_et_load_kernel(ggml_backend_et_device_context* dev_ctx,
                         const std::string& kernel_name,
                         const std::string& kernel_path);

// Launch kernel with parameters on device's default stream
// Performs lazy loading: automatically loads kernel if not already loaded
// Kernel path: ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf (default: /opt/et/ggml/kernels/)
// Returns true on success, false on failure
// Execution is synchronous - waits for completion
bool ggml_et_launch_kernel(ggml_backend_et_device_context* dev_ctx,
                           const std::string& kernel_name,
                           void* params,
                           size_t params_size,
                           uint64_t shire_mask = 0xFFFFFFFF);

// Unload kernel from device and free resources
// Safe to call even if kernel not loaded
void ggml_et_unload_kernel(ggml_backend_et_device_context* dev_ctx,
                           const std::string& kernel_name);

// Unload all kernels from device context
// Called during device cleanup
void ggml_et_unload_all_kernels(ggml_backend_et_device_context* dev_ctx);

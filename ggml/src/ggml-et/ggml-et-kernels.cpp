#include "ggml-et-kernels.h"
#include "ggml-impl.h"
#include "ggml-et-kernels-embed.hpp"
#include <fstream>
#include <cstdlib>
#include <cstring>

// Get embedded kernel data by name
static std::vector<std::byte> ggml_et_get_embedded_kernel(const std::string& kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    if (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        return {};
    }

    const unsigned char* data = it->second.first;
    uint64_t size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    GGML_LOG_DEBUG("ET: Retrieved embedded kernel %s (%zu bytes)\n", kernel_name.c_str(), buffer.size());
    return buffer;
}

// Read kernel from file (for development/override)
static std::vector<std::byte> ggml_et_read_kernel_file(const std::string& kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    GGML_LOG_DEBUG("ET: Read kernel file %s (%zu bytes)\n", kernel_path.c_str(), buffer.size());
    return buffer;
}

// Load kernel from file or embedded data
bool ggml_et_load_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        return false;
    }

    // Check if kernel already loaded
    if (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        return true;
    }

    std::vector<std::byte> kernel_data;
    const char* kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    if (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data = ggml_et_read_kernel_file(kernel_file);

        if (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } else {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    if (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        if (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            return false;
        }
        GGML_LOG_INFO("ET: Loading embedded kernel %s\n", kernel_name.c_str());
    }

    try {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;

        GGML_LOG_INFO("ET: Loaded kernel %s on device %d (KernelId=%d, LoadAddr=%p)\n",
                      kernel_name.c_str(), dev_ctx->devidx,
                      static_cast<int>(load_result.kernel_),
                      (void*)load_result.loadAddress_);
        return true;

    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

bool ggml_et_launch_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name,
                          void* params, size_t params_size, uint64_t shire_mask) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        return false;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        if (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            return false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        if (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            return false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    try {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(true);          // Wait for completion
        k_opts.setFlushL3(false);         // No L3 flush needed

        GGML_LOG_DEBUG("ET: Launching kernel %s (KernelId=%d) with %zu bytes params on device %d\n",
                       kernel_name.c_str(), static_cast<int>(kernel_id), params_size, dev_ctx->devidx);

        runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                             reinterpret_cast<std::byte*>(params), params_size, k_opts);

        // Wait for completion (synchronous execution)
        runtime->waitForStream(dev_ctx->default_stream);

        GGML_LOG_DEBUG("ET: Kernel %s completed successfully\n", kernel_name.c_str());
        return true;

    } catch (const std::exception& e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

void ggml_et_unload_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it != dev_ctx->loaded_kernels.end()) {
        try {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
            GGML_LOG_DEBUG("ET: Unloaded kernel %s from device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        } catch (const std::exception& e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

void ggml_et_unload_all_kernels(ggml_backend_et_device_context* dev_ctx) {
    if (!dev_ctx) {
        return;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    for (const auto& kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    for (const auto& kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

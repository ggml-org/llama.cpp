#include "ggml-et-kernels.h"
#include "ggml-impl.h"
#include <fstream>
#include <cstdlib>

std::vector<std::byte> ggml_et_read_kernel_file(const std::string& kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        GGML_LOG_ERROR("ET: Cannot open kernel file: %s\n", kernel_path.c_str());
        return {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    GGML_LOG_DEBUG("ET: Read kernel file %s (%zu bytes)\n", kernel_path.c_str(), buffer.size());
    return buffer;
}

bool ggml_et_load_kernel(ggml_backend_et_device_context* dev_ctx, const std::string& kernel_name, const std::string& kernel_path) {
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

    // Read kernel file
    auto kernel_data = ggml_et_read_kernel_file(kernel_path);
    if (kernel_data.empty()) {
        return false;
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
        // Kernel not loaded - attempt to load it
        const char* kernels_base_path = getenv("GGML_ET_KERNELS_PATH");
        if (!kernels_base_path) {
            kernels_base_path = "/opt/et/ggml/kernels";
        }

        std::string kernel_path = std::string(kernels_base_path) + "/" + kernel_name + ".elf";

        GGML_LOG_INFO("ET: Lazy loading kernel %s from %s\n", kernel_name.c_str(), kernel_path.c_str());

        if (!ggml_et_load_kernel(dev_ctx, kernel_name, kernel_path)) {
            GGML_LOG_ERROR("ET: Failed to load kernel %s from %s\n", kernel_name.c_str(), kernel_path.c_str());
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

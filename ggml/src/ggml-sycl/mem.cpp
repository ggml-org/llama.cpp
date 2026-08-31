#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif

#include "base.hpp"
#include "mem.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

const char * mem_api_int2str(int mem_api) {
    if (mem_api == MEMORY_API_TYPE_SYCL) {
        return "SYCL API";
    } else if (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        return "Level Zero API";
    } else {
        return "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
bool query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_count = 0;

#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#else
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    try {
        ze_result_t zes_init = zesInit(0);
        if (zes_init != ZE_RESULT_SUCCESS) {
            std::cerr << "Warning: zesInit failed with code " << static_cast<int>(zes_init)
                      << ". Sysman free-memory query may be unavailable.\n";
        }

        if (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero; falling back to SYCL memory query.\n");
            total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
            free_bytes  = total_bytes;
            return false;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        if (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null; falling back to SYCL memory query.\n");
            total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
            free_bytes  = total_bytes;
            return false;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_count, nullptr);
        if (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules. Falling back to SYCL memory query.\n");
            total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
            free_bytes  = total_bytes;
            return false;
        }

        std::vector<zes_mem_handle_t> modules(module_count);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_count, modules.data());
        if (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules. Falling back to SYCL memory query.\n");
            total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
            free_bytes  = total_bytes;
            return false;
        }

        for (uint32_t i = 0; i < module_count; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            if (r != ZE_RESULT_SUCCESS) {
                continue;
            }

            free_bytes += state.free;
            total_bytes += state.size;
        }

        if (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes. Falling back to SYCL memory query.\n");
            total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
            free_bytes  = total_bytes;
            return false;
        }
        return true;
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
        free_bytes  = total_bytes;
        return false;
    }
}
#endif

bool get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[%s]Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    if (dev.has(sycl::aspect::ext_intel_free_memory)) {
        try {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            return true;
        } catch (const sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory. Using total memory as free "
                "memory.");
            free_bytes = total_bytes;
            return false;
        }
    } else {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory. Using total memory as free memory.");
        free_bytes = total_bytes;
    }
#else
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105. Using total memory as free memory.");
    free_bytes = total_bytes;
#endif
    return true;
}

bool get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {
    const auto name       = dev.get_info<sycl::info::device::name>();
    const auto vendor     = dev.get_info<sycl::info::device::vendor>();
    const auto global_mem = dev.get_info<sycl::info::device::global_mem_size>();

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__, name.c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__, vendor.c_str());
    GGML_SYCL_DEBUG("[%s]GPU Global Memory: %zu bytes\n", __func__, static_cast<size_t>(global_mem));

    if (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s]Querying free memory using Level Zero API.\n", __func__);
        if (!query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            //fallback to SYCL API if Level Zero API fails
            GGML_SYCL_DEBUG("[%s]Falling back to SYCL API for memory query.\n", __func__);
            return get_memory_size_by_sycl_api(dev, free_bytes, total_bytes);
        }
        return true;
#else
        GGML_SYCL_DEBUG("[%s]Level Zero API support is not enabled. Please enable it to use this feature.\n", __func__);
        return false;
#endif
    } else {  //MEMORY_API_TYPE_SYCL
        return get_memory_size_by_sycl_api(dev, free_bytes, total_bytes);
    }
}

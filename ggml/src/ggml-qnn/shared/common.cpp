
#include "common.hpp"

#include <memory>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-qnn.h"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/sysinfo.h>
#    include <unistd.h>
#endif

namespace {

struct ggml_backend_qnn_reg_impl : ggml_backend_reg {
    std::vector<backend_device_proxy_ptr> device_proxies;
    std::vector<ggml_backend_device>      devices;

    explicit ggml_backend_qnn_reg_impl(ggml_backend_reg_i backend_iface) {
        context = this;
        iface   = backend_iface;

        LOG_INFO("backend registry init\n");
        for (size_t i = 0; i < TOTAL_BACKEND_COUNT; i++) {
            const auto device_enum =
                (backend_index_type) (TOTAL_BACKEND_COUNT - 1 - i);  // init from the last device, i.e. NPU

            backend_device_proxy_ptr device_proxy;
            if (device_enum < QNN_BACKEND_COUNT) {
#ifndef GGML_HEXAGON_NPU_ONLY
                device_proxy = create_qnn_backend_context(device_enum);
#else
                LOG_DEBUG("skip qnn device %d\n", (int) device_enum);
                continue;
#endif
            } else {
#ifdef GGML_QNN_ENABLE_HEXAGON_BACKEND
                device_proxy = create_hexagon_backend_context(device_enum);
#else
                LOG_DEBUG("skip hexagon device %d\n", (int) device_enum);
                continue;
#endif
            }

            if (!device_proxy) {
                LOG_DEBUG("skip device %d\n", (int) device_enum);
                continue;
            }

            devices.emplace_back(ggml_backend_device{
                /* iface = */ device_proxy->get_iface(),
                /* reg = */ this,
                /* context = */ device_proxy->get_context(),
            });

            device_proxies.emplace_back(device_proxy);
        }
    }
};

const char * ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    // TODO: should we use a different name?
    return "qualcomm";
}

size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_qnn_reg_impl *) reg->context;
    return ctx->devices.size();
}

ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_qnn_reg_impl *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &(ctx->devices[index]);
}

const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
    /* .get_name         = */ ggml_backend_qnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_qnn_reg_get_device_count,
    /* .get_device_get   = */ ggml_backend_qnn_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

}  // namespace

ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_qnn_reg_impl reg{ ggml_backend_qnn_reg_interface };
    return &reg;
}

namespace common {

#ifdef _WIN32

size_t get_system_total_memory_in_bytes() {
    MEMORYSTATUSEX mem = {};
    mem.dwLength       = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        return mem.ullTotalPhys;
    }

    return 0;
}

size_t get_system_free_memory_in_bytes() {
    MEMORYSTATUSEX mem = {};
    mem.dwLength       = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        return mem.ullAvailPhys;
    }

    return 0;
}

#else

size_t get_system_total_memory_in_bytes() {
    struct sysinfo info = {};
    if (sysinfo(&info) == 0) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }

    auto pages     = (size_t) sysconf(_SC_PHYS_PAGES);
    auto page_size = (size_t) sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

size_t get_system_free_memory_in_bytes() {
    struct sysinfo info = {};
    if (sysinfo(&info) == 0) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }

    auto avail_pages = (size_t) sysconf(_SC_AVPHYS_PAGES);
    auto page_size   = (size_t) sysconf(_SC_PAGE_SIZE);
    return avail_pages * page_size;
}

#endif

}  // namespace common

#include "ggml-cxl-impl.h"
#include "ggml-cxl.h"

#include <vector>

static struct cxl_device raw_devices[CXL_MAX_DEVICES];
static int n_cxl_devices = -1;  // -1 = not yet discovered
static std::vector<ggml_backend_cxl_device_context *> device_contexts;
static std::vector<ggml_backend_dev_t> devices;

static void cxl_discover_devices() {
    if (n_cxl_devices >= 0) {
        return; // already discovered
    }

    int found = cxl_device_discover_all(raw_devices, CXL_MAX_DEVICES);

    GGML_LOG_INFO(GGML_CXL_LOG "discovered %d CXL Type 2 device(s)\n", found);

    // Map devices, compact array to only include successfully mapped devices
    int mapped = 0;
    for (int i = 0; i < found; i++) {
        if (cxl_device_map(&raw_devices[i]) != 0) {
            GGML_LOG_ERROR(GGML_CXL_LOG "failed to map device %d (%s)\n",
                           i, raw_devices[i].pci_addr);
            continue;
        }

        GGML_LOG_INFO(GGML_CXL_LOG "  device %d: %s at %s (%.0f MiB)\n",
                       mapped, raw_devices[i].name, raw_devices[i].pci_addr,
                       (double)raw_devices[i].total_memory / (1024.0 * 1024.0));

        if (mapped != i) {
            raw_devices[mapped] = raw_devices[i];
        }
        raw_devices[mapped].index = mapped;
        mapped++;
    }
    n_cxl_devices = mapped;
}

static size_t ggml_backend_cxl_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    cxl_discover_devices();
    return (size_t)n_cxl_devices;
}

static void ggml_backend_cxl_reg_init_devices(ggml_backend_reg_t reg) {
    if (!devices.empty()) {
        return;
    }

    cxl_discover_devices();

    for (int i = 0; i < n_cxl_devices; i++) {
        auto * ctx = new ggml_backend_cxl_device_context;
        ctx->index = i;
        ctx->name = "CXL" + std::to_string(i);
        ctx->description = "CXL Type 2 GPU proxy (" + std::string(raw_devices[i].pci_addr) + ")";
        ctx->cxl_dev = raw_devices[i];

        device_contexts.push_back(ctx);

        auto * dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_cxl_device_interface,
            /* .reg     = */ reg,
            /* .context = */ ctx,
        };
        devices.push_back(dev);
    }
}

static ggml_backend_dev_t ggml_backend_cxl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    if (index >= devices.size()) {
        return nullptr;
    }
    return devices[index];
}

static const char * ggml_backend_cxl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CXL_NAME;
}

static const ggml_backend_reg_i ggml_backend_cxl_reg_interface = {
    /* .get_name         = */ ggml_backend_cxl_reg_get_name,
    /* .get_device_count = */ ggml_backend_cxl_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cxl_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_cxl_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cxl_reg_interface,
        /* .context     = */ nullptr,
    };

    static bool initialized = false;
    if (!initialized) {
        ggml_backend_cxl_reg_init_devices(&reg);
        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cxl_reg)

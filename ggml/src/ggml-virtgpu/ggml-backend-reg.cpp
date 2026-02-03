#include "ggml-remoting.h"
#include "ggml-virtgpu.h"

#include <iostream>
#include <mutex>

static virtgpu * apir_initialize() {
    static virtgpu * apir_gpu_instance = NULL;
    static bool      apir_initialized  = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        if (apir_initialized) {
            return apir_gpu_instance;
        }

        apir_gpu_instance = create_virtgpu();
        if (!apir_gpu_instance) {
            GGML_ABORT("failed to initialize the virtgpu");
        }

        apir_initialized = true;
    }

    return apir_gpu_instance;
}

static int ggml_backend_remoting_get_device_count() {
    virtgpu * gpu = apir_initialize();
    if (!gpu) {
        GGML_LOG_WARN("apir_initialize failed\n");
        return 0;
    }

    return apir_device_get_count(gpu);
}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);

    return ggml_backend_remoting_get_device_count();
}

static std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    return devices[device];
}

static void ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    if (devices.size() > 0) {
        GGML_LOG_INFO("%s: already initialized\n", __func__);
        return;
    }

    virtgpu * gpu = apir_initialize();
    if (!gpu) {
        GGML_LOG_ERROR("apir_initialize failed\n");
        return;
    }

    static bool initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int i = 0; i < ggml_backend_remoting_get_device_count(); i++) {
                ggml_backend_remoting_device_context * ctx       = new ggml_backend_remoting_device_context;
                char                                   desc[256] = "API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_REMOTING_FRONTEND_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = true;
        }
    }
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    return ggml_backend_remoting_get_device(device);
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    return GGML_REMOTING_FRONTEND_NAME;
}

static const ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    if (!gpu) {
        GGML_LOG_ERROR("virtgpu_apir_initialize failed\n");
        return NULL;
    }

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    static bool initialized = false;
    if (initialized) {
        return &reg;
    }
    initialized = true;

    ggml_backend_remoting_reg_init_devices(&reg);

    GGML_LOG_INFO("%s: initialized\n", __func__);

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)

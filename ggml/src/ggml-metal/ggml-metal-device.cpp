#include "ggml-metal-device.h"

#include <memory>

struct ggml_backend_metal_device_deleter {
    void operator()(ggml_backend_metal_device_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_metal_device, ggml_backend_metal_device_deleter> ggml_backend_metal_device_ptr;

ggml_backend_metal_device_t ggml_backend_metal_device_get(void) {
    static ggml_backend_metal_device_ptr ctx { ggml_backend_metal_device_init() };

    return ctx.get();
}

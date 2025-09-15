#include "ggml-metal-device.h"

#include <memory>

struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(void) {
    static ggml_metal_device_ptr ctx { ggml_metal_device_init() };

    return ctx.get();
}

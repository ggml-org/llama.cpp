#include "ggml-vulkan.h"
#include <cstdio>

int main(int argc, char ** argv) {
    int device_count = ggml_backend_vk_get_device_count();
    printf("Found %d Vulkan devices\\n", device_count);

    for (int i = 0; i < device_count; i++) {
        ggml_vk_device_info info = ggml_backend_vk_get_device_info(i);
        printf("\\nDevice %d: %s\\n", i, info.device_name);
        printf("  Vendor ID: %04x\\n", info.vendor_id);
        printf(" Device ID: %04x\\n", info.device_id);
        printf("  API Version: 0x%08x\\n", info.api_version);
        printf("  Total Device Local Memory: %llu MB\\n", info.total_device_local_memory / (1024 * 1024));
        printf("  Has Memory Budget Ext: %s\\n", info.has_memory_budget_ext ? "Yes" : "No");
        printf("  Supports Float16: %s\\n", info.supports_float16 ? "Yes" : "No");
        printf("  Supports 16-bit Storage: %s\\n", info.supports_16bit_storage ? "Yes" : "No");
        
        int default_layers = ggml_backend_vk_get_default_gpu_layers(i, -1);
        printf("  Default GPU Layers (heuristic): %d\\n", default_layers);
    }

    return 0;
}

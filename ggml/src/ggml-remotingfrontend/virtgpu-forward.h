int apir_get_device_count(struct virtgpu *gpu);
const char *apir_get_device_name(struct virtgpu *gpu);
const char *apir_get_device_description(struct virtgpu *gpu);
uint32_t apir_get_device_type(struct virtgpu *gpu);
void apir_get_device_memory(struct virtgpu *gpu, size_t *free, size_t *total);

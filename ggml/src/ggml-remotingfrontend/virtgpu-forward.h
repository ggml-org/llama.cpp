struct ggml_tensor;

int apir_device_get_count(struct virtgpu *gpu);
const char *apir_device_get_name(struct virtgpu *gpu);
const char *apir_device_get_description(struct virtgpu *gpu);
uint32_t apir_device_get_type(struct virtgpu *gpu);
void apir_device_get_memory(struct virtgpu *gpu, size_t *free, size_t *total);
bool apir_device_supports_op(struct virtgpu *gpu, const ggml_tensor *op);

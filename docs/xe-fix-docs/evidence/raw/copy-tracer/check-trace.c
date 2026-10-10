#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <assert.h>

int main(void) {
    struct ggml_init_params params = { .mem_size = 4096, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, "tracer_cpu_check");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buf);
    float input[4] = {1, 2, 3, 4}, output[4] = {0};
    ggml_backend_tensor_set_async(cpu, t, input, 0, sizeof(input));
    ggml_backend_synchronize(cpu);
    ggml_backend_tensor_get(t, output, 0, sizeof(output));
    for (int i = 0; i < 4; ++i) assert(input[i] == output[i]);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

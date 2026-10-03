#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

// Regression test for https://github.com/ggml-org/llama.cpp/issues/26367
//
// A host-side get_split_state callback that returns `axis == GGML_MAX_DIMS`
// (an invalid value) used to be accepted by the bounds check in
// `ggml_backend_meta_get_split_state` because it compared with `<=` instead
// of `<`. The following `tensor->ne[ret.axis]` then read one element past the
// end of the ne[] array (which only has GGML_MAX_DIMS entries), hitting
// tensor->nb[0] and failing the `ne_sum == tensor->ne[ret.axis]` assertion.
static ggml_backend_meta_get_split_state_t invalid_split_state = [](const ggml_tensor * tensor, void * ud) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(ud);

    struct ggml_backend_meta_split_state st = {};
    st.axis       = (ggml_backend_meta_split_axis) GGML_MAX_DIMS;
    st.n_segments = 1;
    st.ne[0]      = 0;
    st.nr[0]      = 1;
    return st;
};

int main() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        fprintf(stderr, "no CPU backend available\n");
        return 1;
    }

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(&cpu_dev, 1, invalid_split_state, nullptr);
    if (meta_dev == nullptr) {
        fprintf(stderr, "failed to create meta device\n");
        return 1;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_dev);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    if (buffer == nullptr) {
        fprintf(stderr, "failed to allocate meta buffer\n");
        return 1;
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to init ggml context\n");
        return 1;
    }

    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);

    printf("ok\n");
    return 0;
}

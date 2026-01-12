// Validate that ggml_backend_tensor_alloc sets a usable tensor->data pointer.

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdio>
#include <vector>

static void test_tensor_data_pointer() {
    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    GGML_ASSERT(t != nullptr);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    GGML_ASSERT(buft != nullptr);

    const size_t          alloc_size = ggml_backend_buft_get_alloc_size(buft, t);
    ggml_backend_buffer_t buffer     = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    GGML_ASSERT(buffer != nullptr);

    void * base = ggml_backend_buffer_get_base(buffer);
    GGML_ASSERT(base != nullptr);

    const ggml_status alloc_status = ggml_backend_tensor_alloc(buffer, t, base);
    GGML_ASSERT(alloc_status == GGML_STATUS_SUCCESS);
    GGML_ASSERT(t->data != nullptr);

    const char * buf_start = static_cast<const char *>(ggml_backend_buffer_get_base(buffer));
    const char * buf_end   = buf_start + ggml_backend_buffer_get_size(buffer);
    const char * data_ptr  = static_cast<const char *>(t->data);
    GGML_ASSERT(data_ptr >= buf_start);
    GGML_ASSERT(data_ptr + ggml_nbytes(t) <= buf_end);

    const std::vector<float> input{ 1.0f, 2.0f, 3.5f, 4.5f, 5.0f, 6.25f, 7.0f, 8.75f };
    ggml_backend_tensor_set(t, input.data(), 0, input.size() * sizeof(float));

    std::vector<float> output(input.size(), 0.0f);
    ggml_backend_tensor_get(t, output.data(), 0, output.size() * sizeof(float));

    for (size_t i = 0; i < input.size(); ++i) {
        GGML_ASSERT(output[i] == input[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    test_tensor_data_pointer();
    printf("test-tensor-data: PASS\n");
    return 0;
}

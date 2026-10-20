#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool run_tensor_n_case(
        std::vector<ggml_tensor *> & tensors,
        const std::vector<size_t> & offsets,
        const std::vector<size_t> & sizes,
        const std::vector<size_t> & data_offsets) {
    const size_t n = tensors.size();

    std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
    std::vector<uint8_t> data(256, 0);
    std::vector<uint8_t> out(256, 0);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(data.data() + data_offsets[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    ggml_backend_tensor_get_n(const_tensors.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        if (memcmp(data.data() + data_offsets[i], out.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    std::vector<uint8_t> slice(256, 0);
    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(slice.data(), 0, slice.size());
        ggml_backend_tensor_get(tensors[i], slice.data(), offsets[i], sizes[i]);
        if (memcmp(slice.data(), data.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    return true;
}

static bool test_backend_tensor_n(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 1024;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * t_big = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 6);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return false;
    }

    {
        std::vector<ggml_tensor *> tensors = { t0, t1, t2, t0 };
        std::vector<size_t> offsets = { 0, 0, 0, 0 };
        std::vector<size_t> sizes = { 3*4, 2*4, 4*4, 0 };
        std::vector<size_t> data_offsets = { 0, 3*4, 5*4, 64 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> tensors = { t_big, t_big, t_big };
        std::vector<size_t> offsets = { 0, 8, 16 };
        std::vector<size_t> sizes = { 8, 8, 8 };
        std::vector<size_t> data_offsets = { 0, 8, 16 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    return true;
}

int main() {
    ggml_backend_load_all();

    const size_t n_devs = ggml_backend_dev_count();
    size_t n_ok = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);

        printf("%s ", name);
        fflush(stdout);

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, NULL));
        if (!backend) {
            printf("FAILED\n");
            continue;
        }

        const bool ok = test_backend_tensor_n(backend.get());
        printf("%s\n", ok ? "PASSED" : "FAILED");
        if (ok) {
            n_ok++;
        }
    }

    printf("%zu/%zu backends passed\n", n_ok, n_devs);

    return n_ok == n_devs ? 0 : 1;
}

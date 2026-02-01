// Validate GGUF matmul shape extraction for sycl-kernel-bench.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "ggml.h"
#include "gguf.h"
#include "../tools/sycl-kernel-bench/model_shapes.hpp"

int main() {
    char path[] = "/tmp/llama-sycl-shapes-XXXXXX.gguf";
    const int suffix_len = 5; // ".gguf"
    const int fd = mkstemps(path, suffix_len);
    if (fd < 0) {
        std::perror("mkstemps");
        return 1;
    }
    close(fd);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "Failed to init ggml context\n");
        std::remove(path);
        return 1;
    }

    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
    ggml_set_name(tensor, "test.weight");
    void * data = ggml_get_data(tensor);
    if (data) {
        std::memset(data, 0, ggml_nbytes(tensor));
    }

    gguf_context * gguf = gguf_init_empty();
    if (!gguf) {
        std::fprintf(stderr, "Failed to init gguf context\n");
        ggml_free(ctx);
        std::remove(path);
        return 1;
    }
    gguf_add_tensor(gguf, tensor);
    gguf_write_to_file(gguf, path, false);
    gguf_free(gguf);
    ggml_free(ctx);

    std::vector<sycl_bench::ModelMatmulShape> shapes;
    std::string error;
    if (!sycl_bench::load_model_matmul_shapes(path, shapes, error)) {
        std::fprintf(stderr, "Failed to load shapes: %s\n", error.c_str());
        std::remove(path);
        return 1;
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "No shapes returned\n");
        std::remove(path);
        return 1;
    }

    const auto & shape = shapes.front();
    if (shape.name != "test.weight") {
        std::fprintf(stderr, "Unexpected tensor name: %s\n", shape.name.c_str());
        std::remove(path);
        return 1;
    }
    if (shape.type != GGML_TYPE_Q4_0) {
        std::fprintf(stderr, "Unexpected tensor type\n");
        std::remove(path);
        return 1;
    }
    if (shape.dim_k != 128 || shape.dim_m != 64) {
        std::fprintf(stderr, "Unexpected dims: M=%lld K=%lld\n",
                     static_cast<long long>(shape.dim_m),
                     static_cast<long long>(shape.dim_k));
        std::remove(path);
        return 1;
    }

    std::remove(path);
    return 0;
}

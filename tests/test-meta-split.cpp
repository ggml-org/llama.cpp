#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct split_ud {
    size_t ndev;
};

static ggml_backend_meta_split_state split_state_callback(const ggml_tensor * tensor, void * userdata) {
    const auto * ud = static_cast<const split_ud *>(userdata);
    ggml_backend_meta_split_state state{};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.nr[0] = 1;
    state.n_segments = 1;

    if (std::strcmp(tensor->name, "root") == 0) {
        if (ud->ndev == 0 || tensor->ne[2] % (int64_t) ud->ndev != 0) {
            std::fprintf(stderr, "invalid test split: ne2=%lld ndev=%zu\n", (long long) tensor->ne[2], ud->ndev);
            std::abort();
        }
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        for (size_t j = 0; j < ud->ndev; ++j) {
            state.ne[j] = tensor->ne[2] / (int64_t) ud->ndev;
        }
    }
    return state;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> simple_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
            simple_devs.push_back(dev);
        }
    }
    if (simple_devs.size() < 2) {
        std::puts("meta split readback test skipped: fewer than two non-CPU devices");
        return 0;
    }

    split_ud ud{simple_devs.size()};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(simple_devs.data(), simple_devs.size(), split_state_callback, &ud);
    ggml_backend_ptr backend(ggml_backend_dev_init(meta_dev, nullptr));
    if (!backend) {
        std::fprintf(stderr, "failed to initialize meta backend\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    ggml_tensor * root = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 8, 1);
    ggml_set_name(root, "root");
    // Swap dimensions 0 and 1 while preserving dimension 2.  The result is
    // deliberately non-contiguous but remains split along axis 2.
    ggml_tensor * permuted = ggml_permute(ctx.get(), root, 1, 0, 2, 3);
    ggml_set_name(permuted, "root-permuted");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate meta tensors\n");
        return 1;
    }

    const size_t nbytes = ggml_nbytes(root);
    std::vector<float> expected(nbytes / sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = std::sin((float) i * 0.125f);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, nbytes);

    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(permuted, actual.data(), 0, nbytes);

    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "permuted axis-2 readback mismatch at %zu: %.9g != %.9g\n", i, expected[i], actual[i]);
            return 1;
        }
    }

    std::puts("meta split axis-2 permuted readback passed");
    return 0;
}

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdlib>
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

    if (std::strcmp(tensor->name, "root") == 0 || std::strcmp(tensor->name, "axis3") == 0) {
        const int axis = std::strcmp(tensor->name, "axis3") == 0 ? 3 : 2;
        if (ud->ndev == 0 || tensor->ne[axis] % (int64_t) ud->ndev != 0) {
            std::fprintf(stderr, "invalid test split: axis=%d ne=%lld ndev=%zu\n", axis, (long long) tensor->ne[axis], ud->ndev);
            std::abort();
        }
        state.axis = axis == 3 ? GGML_BACKEND_SPLIT_AXIS_3 : GGML_BACKEND_SPLIT_AXIS_2;
        for (size_t j = 0; j < ud->ndev; ++j) {
            state.ne[j] = tensor->ne[axis] / (int64_t) ud->ndev;
        }
    } else if (std::strcmp(tensor->name, "partial") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    } else if (std::strcmp(tensor->name, "segments") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.n_segments = 2;
        state.nr[0] = 1;
        state.nr[1] = 1;
        for (size_t j = 0; j < ud->ndev; ++j) {
            state.ne[j] = tensor->ne[0] / 2 / (int64_t) ud->ndev;
            state.ne[ud->ndev + j] = tensor->ne[0] / 2 / (int64_t) ud->ndev;
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
    ggml_tensor * mirror = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 8, 1);
    ggml_set_name(mirror, "mirror");
    ggml_tensor * axis3 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 3, 4, 4, 8);
    ggml_set_name(axis3, "axis3");
    // Keep the split on axis 3 while permuting dimensions inside each
    // physical row.  This is non-contiguous metadata but each shard remains
    // safe for the axis-3 row-wise transfer path.
    ggml_tensor * axis3_permuted = ggml_permute(ctx.get(), axis3, 1, 0, 2, 3);
    ggml_set_name(axis3_permuted, "axis3-permuted");
    ggml_tensor * partial = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 4, 1);
    ggml_set_name(partial, "partial");
    ggml_tensor * segments = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 8, 4, 1, 1);
    ggml_set_name(segments, "segments");
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
    ggml_backend_tensor_get(root, actual.data(), 0, nbytes);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "contiguous axis-2 readback mismatch at %zu: %.9g != %.9g\n", i, expected[i], actual[i]);
            return 1;
        }
    }

    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_backend_tensor_get(permuted, actual.data(), 0, nbytes);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "permuted axis-2 readback mismatch at %zu: %.9g != %.9g\n", i, expected[i], actual[i]);
            return 1;
        }
    }

    std::vector<float> mirrored(expected.size());
    for (size_t i = 0; i < mirrored.size(); ++i) {
        mirrored[i] = std::cos((float) i * 0.0625f);
    }
    ggml_backend_tensor_set(mirror, mirrored.data(), 0, nbytes);
    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_backend_tensor_get(mirror, actual.data(), 0, nbytes);
    for (size_t i = 0; i < mirrored.size(); ++i) {
        if (mirrored[i] != actual[i]) {
            std::fprintf(stderr, "mirrored readback mismatch at %zu: %.9g != %.9g\n", i, mirrored[i], actual[i]);
            return 1;
        }
    }

    const size_t axis3_nbytes = ggml_nbytes(axis3);
    std::vector<float> axis3_expected(axis3_nbytes / sizeof(float));
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        axis3_expected[i] = (float) (i * 3 + 1);
    }
    ggml_backend_tensor_set(axis3, axis3_expected.data(), 0, axis3_nbytes);
    std::vector<float> axis3_actual(axis3_expected.size(), 0.0f);
    ggml_backend_tensor_get(axis3, axis3_actual.data(), 0, axis3_nbytes);
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        if (axis3_expected[i] != axis3_actual[i]) {
            std::fprintf(stderr, "axis-3 readback mismatch at %zu: %.9g != %.9g\n", i, axis3_expected[i], axis3_actual[i]);
            return 1;
        }
    }

    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_get(axis3_permuted, axis3_actual.data(), 0, axis3_nbytes);
    if (axis3_actual != axis3_expected) {
        std::fprintf(stderr, "permuted axis-3 readback mismatch\n");
        return 1;
    }

    const size_t axis3_row_bytes = axis3->nb[3];
    const size_t axis3_row_elems = axis3_row_bytes / sizeof(float);
    const size_t patch_row_start = 2;
    const size_t patch_row_count = 3;
    std::vector<float> patch(patch_row_count * axis3_row_elems, -7.0f);
    ggml_backend_tensor_set(axis3, patch.data(), patch_row_start * axis3_row_bytes, patch.size() * sizeof(float));
    std::vector<float> patched_expected = axis3_expected;
    std::copy(patch.begin(), patch.end(), patched_expected.begin() + patch_row_start * axis3_row_elems);
    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_get(axis3, axis3_actual.data(), 0, axis3_nbytes);
    if (axis3_actual != patched_expected) {
        std::fprintf(stderr, "axis-3 partial set/readback mismatch\n");
        return 1;
    }

    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_set_async(backend.get(), axis3, axis3_expected.data(), 0, axis3_nbytes);
    ggml_backend_synchronize(backend.get());
    ggml_backend_tensor_get_async(backend.get(), axis3, axis3_actual.data(), 0, axis3_nbytes);
    ggml_backend_synchronize(backend.get());
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        if (axis3_expected[i] != axis3_actual[i]) {
            std::fprintf(stderr, "axis-3 async readback mismatch at %zu: %.9g != %.9g\n", i, axis3_expected[i], axis3_actual[i]);
            return 1;
        }
    }

    const size_t partial_nbytes = ggml_nbytes(partial);
    std::vector<float> partial_expected(partial_nbytes / sizeof(float));
    for (size_t i = 0; i < partial_expected.size(); ++i) {
        partial_expected[i] = (float) (i + 0.25f);
    }
    ggml_backend_tensor_set(partial, partial_expected.data(), 0, partial_nbytes);
    std::vector<float> partial_actual(partial_expected.size(), 0.0f);
    ggml_backend_tensor_get(partial, partial_actual.data(), 0, partial_nbytes);
    if (partial_actual != partial_expected) {
        std::fprintf(stderr, "partial set/readback mismatch\n");
        return 1;
    }
    std::fill(partial_actual.begin(), partial_actual.end(), 0.0f);
    ggml_backend_tensor_set_async(backend.get(), partial, partial_expected.data(), 0, partial_nbytes);
    ggml_backend_synchronize(backend.get());
    ggml_backend_tensor_get_async(backend.get(), partial, partial_actual.data(), 0, partial_nbytes);
    ggml_backend_synchronize(backend.get());
    if (partial_actual != partial_expected) {
        std::fprintf(stderr, "partial async set/readback mismatch\n");
        return 1;
    }

    const size_t segments_nbytes = ggml_nbytes(segments);
    std::vector<float> segments_expected(segments_nbytes / sizeof(float));
    for (size_t i = 0; i < segments_expected.size(); ++i) {
        segments_expected[i] = (float) (1000 + i);
    }
    ggml_backend_tensor_set(segments, segments_expected.data(), 0, segments_nbytes);
    std::vector<float> segments_actual(segments_expected.size(), 0.0f);
    ggml_backend_tensor_get(segments, segments_actual.data(), 0, segments_nbytes);
    if (segments_actual != segments_expected) {
        std::fprintf(stderr, "multi-segment readback mismatch\n");
        return 1;
    }

    std::puts("meta split axis-2, axis-3, mirrored, partial, and multi-segment readback passed");
    return 0;
}

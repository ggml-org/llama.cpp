#include "fattn-mkl.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

static ggml_tensor make_tensor(void * data) {
    ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_F16;
    tensor.data = data;

    const int64_t shape[GGML_MAX_DIMS] = { 64, 128, 8, 3 };
    const size_t stride[GGML_MAX_DIMS] = { 2, 128, 16384, 131072 };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        tensor.ne[i] = shape[i];
        tensor.nb[i] = stride[i];
    }

    return tensor;
}

int main() {
    expect(ggml_sycl_fattn_mkl_batch_offset(1, 4096, 3) == 0,
           "broadcast K/V must keep the batch-zero offset");
    expect(ggml_sycl_fattn_mkl_batch_offset(4, 4096, 3) == 12288,
           "batched K/V must use ib * nb3");

    char storage_a[1] = {};
    char storage_b[1] = {};
    ggml_tensor k = make_tensor(storage_a);
    ggml_tensor v = make_tensor(storage_a);
    v.view_src = &k;

    expect(ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &v),
           "exact zero-offset K/V alias must be reusable");

    {
        ggml_tensor changed_k = k;
        changed_k.view_offs = 16;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&changed_k, &v),
               "nonzero K view offset must reject alias reuse");
    }
    {
        ggml_tensor changed_v = v;
        changed_v.view_offs = 16;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &changed_v),
               "nonzero V view offset must reject alias reuse");
    }
    {
        ggml_tensor changed_k = k;
        ggml_tensor changed_v = v;
        changed_k.view_offs = 16;
        changed_v.view_offs = 16;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&changed_k, &changed_v),
               "matching nonzero K/V view offsets must reject alias reuse");
    }
    {
        ggml_tensor changed_k = k;
        ggml_tensor changed_v = v;
        changed_k.view_offs = 16;
        changed_v.view_offs = 32;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&changed_k, &changed_v),
               "different K/V view offsets must reject alias reuse");
    }
    {
        ggml_tensor changed_v = v;
        changed_v.type = GGML_TYPE_F32;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &changed_v),
               "different K/V types must reject alias reuse");
    }
    {
        ggml_tensor changed_v = v;
        changed_v.data = storage_b;
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &changed_v),
               "different K/V data pointers must reject alias reuse");
    }

    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        ggml_tensor changed_v = v;
        ++changed_v.ne[dim];
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &changed_v),
               "different K/V shape metadata must reject alias reuse");
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        ggml_tensor changed_v = v;
        ++changed_v.nb[dim];
        expect(!ggml_sycl_fattn_mkl_can_reuse_k_for_v(&k, &changed_v),
               "different K/V stride metadata must reject alias reuse");
    }

    return 0;
}

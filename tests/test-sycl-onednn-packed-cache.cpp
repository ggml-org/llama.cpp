#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

static bool run_onednn_packed_cache_test() {
    setenv("GGML_SYCL_ONEDNN_PACK_M", "16", 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: SYCL backend unavailable\n");
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        std::printf("SKIP: buffer types unavailable\n");
        ggml_backend_free(backend);
        return true;
    }

    ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init failed\n");
        ggml_backend_free(backend);
        return false;
    }

    const int ncols   = 256;
    const int nrows   = 128;
    const int ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ncols, nrows);
    ggml_set_name(weight, "onednn_packed_weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ncols, ntokens);
    ggml_set_name(input, "onednn_packed_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "onednn_packed_output");

    const size_t weight_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    const size_t input_size  = ggml_backend_buft_get_alloc_size(dev_buft, input);
    const size_t output_size = ggml_backend_buft_get_alloc_size(dev_buft, output);

    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    ggml_backend_buffer_t input_buf  = ggml_backend_buft_alloc_buffer(dev_buft, input_size);
    ggml_backend_buffer_t output_buf = ggml_backend_buft_alloc_buffer(dev_buft, output_size);

    if (!weight_buf || !input_buf || !output_buf) {
        std::printf("FAIL: buffer allocation failed\n");
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (output_buf) ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(input_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(output_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));
    ggml_backend_tensor_alloc(input_buf, input, ggml_backend_buffer_get_base(input_buf));
    ggml_backend_tensor_alloc(output_buf, output, ggml_backend_buffer_get_base(output_buf));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_sycl_set_onednn_pack_m(weight, 32);

    std::vector<ggml_fp16_t> weight_data(ncols * nrows);
    std::vector<ggml_fp16_t> input_data(ncols * ntokens);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = ggml_fp32_to_fp16(0.01f * static_cast<float>(i % 97));
    }
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = ggml_fp32_to_fp16(0.001f * static_cast<float>(i % 31));
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(ggml_fp16_t));

    void * aos_ptr = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_AOS);
    if (!aos_ptr) {
        std::printf("SKIP: failed to cache AoS layout\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    void * packed_ptr = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_ONEDNN_PACKED);
    if (!packed_ptr || !weight->layout || weight->layout->mode != GGML_LAYOUT_ONEDNN_PACKED) {
        std::printf("SKIP: onednn packed layout unavailable\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }
    if (weight->layout->onednn_pack_m != 32) {
        std::printf("FAIL: expected onednn pack_m=32, got=%lld\n", (long long) weight->layout->onednn_pack_m);
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_sycl_set_onednn_pack_m(weight, 64);
    void * repacked_ptr = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_ONEDNN_PACKED);
    if (!repacked_ptr || !weight->layout || weight->layout->mode != GGML_LAYOUT_ONEDNN_PACKED) {
        std::printf("FAIL: expected repacked onednn layout\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    if (weight->layout->onednn_pack_m != 64) {
        std::printf("FAIL: expected onednn pack_m=64 after repack, got=%lld\n",
                    (long long) weight->layout->onednn_pack_m);
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    sycl::queue & q = dpct::dev_mgr::instance().get_device(0).default_queue();
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
    if (!cache) {
        std::printf("SKIP: unified cache unavailable\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    const bool packed_cached = key.valid && cache->is_cached(key, GGML_LAYOUT_ONEDNN_PACKED);
    const bool aos_cached    = key.valid && cache->is_cached(key, GGML_LAYOUT_AOS);

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(output_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!packed_cached) {
        std::printf("FAIL: expected onednn packed layout cached\n");
        return false;
    }
    if (aos_cached) {
        std::printf("FAIL: expected AoS cache entry to be evicted\n");
        return false;
    }

    std::printf("PASS: onednn packed layout cached and AoS evicted\n");
    return true;
}

int main() {
    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_ONEDNN_PACKED);
    return run_onednn_packed_cache_test() ? 0 : 1;
}

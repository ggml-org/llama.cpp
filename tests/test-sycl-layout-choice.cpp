// SYCL unified-cache layout choice test.
// Verifies that pre-finalize layouts are purged after finalize_layouts selects a single layout.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool run_layout_choice_test() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP: SYCL backend unavailable\n");
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        printf("SKIP: buffer types unavailable\n");
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
        printf("FAIL: ggml_init failed\n");
        ggml_backend_free(backend);
        return false;
    }

    const int ncols   = 1024;
    const int nrows   = 128;
    const int ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "layout_choice_weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, ntokens);
    ggml_set_name(input, "layout_choice_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "layout_choice_output");

    const size_t weight_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    const size_t input_size  = ggml_backend_buft_get_alloc_size(dev_buft, input);
    const size_t output_size = ggml_backend_buft_get_alloc_size(dev_buft, output);

    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    ggml_backend_buffer_t input_buf  = ggml_backend_buft_alloc_buffer(dev_buft, input_size);
    ggml_backend_buffer_t output_buf = ggml_backend_buft_alloc_buffer(dev_buft, output_size);

    if (!weight_buf || !input_buf || !output_buf) {
        printf("FAIL: buffer allocation failed\n");
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

    std::vector<uint8_t> weight_data(ggml_nbytes(weight), 0);
    std::vector<float>   input_data(ncols * ntokens, 0.1f);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    void * pre_cached = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_COALESCED);
    if (!pre_cached) {
        printf("SKIP: coalesced layout unavailable, cannot validate purge\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute failed\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    layout_mode chosen_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl_get_layout_choice_for_tensor(weight, 0, &chosen_layout)) {
        printf("FAIL: missing layout choice for weight after finalize\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    if (chosen_layout != GGML_LAYOUT_SOA) {
        printf("FAIL: expected SoA layout choice, got %d\n", (int) chosen_layout);
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
        printf("SKIP: unified cache unavailable\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    const void * key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    const bool   soa_cached = cache->is_cached(key, GGML_LAYOUT_SOA);
    const bool   coa_cached = cache->is_cached(key, GGML_LAYOUT_COALESCED);

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(output_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!soa_cached) {
        printf("FAIL: expected SoA layout cached after finalize\n");
        return false;
    }
    if (coa_cached) {
        printf("FAIL: coalesced layout should be purged after finalize\n");
        return false;
    }

    printf("PASS: layout choice enforced (SoA only)\n");
    return true;
}

int main() {
    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_SOA);
    bool ok = run_layout_choice_test();
    return ok ? 0 : 1;
}

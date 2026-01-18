// Unit tests for SYCL layout selection and unified cache behavior
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-layout-cache

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"

static const char * layout_name(ggml_layout_mode mode) {
    switch (mode) {
        case GGML_LAYOUT_AOS:       return "AOS";
        case GGML_LAYOUT_SOA:       return "SOA";
        case GGML_LAYOUT_COALESCED: return "COALESCED";
        case GGML_LAYOUT_XMX_TILED: return "XMX_TILED";
        case GGML_LAYOUT_XMX_GEMM_TILED: return "XMX_GEMM_TILED";
        default:                    return "UNKNOWN";
    }
}

static const char * usage_name(tensor_usage usage) {
    switch (usage) {
        case tensor_usage::UNKNOWN:           return "UNKNOWN";
        case tensor_usage::ATTENTION_WEIGHT:  return "ATTENTION_WEIGHT";
        case tensor_usage::FFN_WEIGHT:        return "FFN_WEIGHT";
        case tensor_usage::MOE_EXPERT_WEIGHT: return "MOE_EXPERT_WEIGHT";
        case tensor_usage::MOE_GATE:          return "MOE_GATE";
        case tensor_usage::EMBEDDING:         return "EMBEDDING";
        case tensor_usage::NORM:              return "NORM";
        default:                              return "UNKNOWN";
    }
}

static void reset_layout_choices() {
    ggml_sycl::test_clear_host_weight_registry();
    ggml_backend_sycl_set_model_loading(true);
    ggml_backend_sycl_set_model_loading(false);
}

static bool expect_usage(const char * label, tensor_usage got, tensor_usage expected) {
    if (got != expected) {
        fprintf(stderr, "%s: expected usage %s, got %s\n",
                label, usage_name(expected), usage_name(got));
        return false;
    }
    return true;
}

static bool expect_layout(const char * label, ggml_layout_mode got, ggml_layout_mode expected) {
    if (got != expected) {
        fprintf(stderr, "%s: expected layout %s, got %s\n",
                label, layout_name(expected), layout_name(got));
        return false;
    }
    return true;
}

static void fill_pattern(std::vector<uint8_t> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131) ^ 0x5a);
    }
}

static bool test_layout_selection(int device_id, bool xmx_supported) {
    bool ok = true;

    tensor_usage usage = infer_tensor_usage("attn_q.weight");
    ok &= expect_usage("infer_tensor_usage(attn_q.weight)", usage, tensor_usage::ATTENTION_WEIGHT);
    ok &= expect_layout("layout_policy(attn_q, Q8_0)",
                        layout_policy::get_with_override(GGML_TYPE_Q8_0, usage, device_id),
                        GGML_LAYOUT_COALESCED);

    usage = infer_tensor_usage("ffn_gate_exps.weight");
    ok &= expect_usage("infer_tensor_usage(ffn_gate_exps.weight)", usage, tensor_usage::MOE_EXPERT_WEIGHT);
    const ggml_layout_mode expected_xmx = xmx_supported ? GGML_LAYOUT_XMX_TILED : GGML_LAYOUT_SOA;
    ok &= expect_layout("layout_policy(ffn_gate_exps, MXFP4)",
                        layout_policy::get_with_override(GGML_TYPE_MXFP4, usage, device_id),
                        expected_xmx);

    usage = infer_tensor_usage("ffn_up.weight");
    ok &= expect_usage("infer_tensor_usage(ffn_up.weight)", usage, tensor_usage::FFN_WEIGHT);
    ok &= expect_layout("layout_policy(ffn_up, Q4_0)",
                        layout_policy::get_with_override(GGML_TYPE_Q4_0, usage, device_id),
                        GGML_LAYOUT_COALESCED);

    usage = infer_tensor_usage("tok_norm.weight");
    ok &= expect_usage("infer_tensor_usage(tok_norm.weight)", usage, tensor_usage::NORM);
    ok &= expect_layout("layout_policy(tok_norm, F32)",
                        layout_policy::get_with_override(GGML_TYPE_F32, usage, device_id),
                        GGML_LAYOUT_AOS);

    return ok;
}

static bool test_aos_drop(int device_id) {
    reset_layout_choices();

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to allocate CPU weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    const size_t weight_bytes = ggml_nbytes(weight);
    std::vector<uint8_t> host_data(weight_bytes);
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get cache key\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get unified cache\n");
        return false;
    }

    void * aos_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_AOS);
    if (!aos_ptr || !cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to cache AOS layout\n");
        return false;
    }

    void * coalesced_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_COALESCED);
    if (!coalesced_ptr || !cache->is_cached(key, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to cache COALESCED layout\n");
        return false;
    }

    if (cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "AOS cache entry not dropped after COALESCED cache\n");
        return false;
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    return true;
}

static bool test_mul_mat_layout_choice_coalesced(int device_id) {
    reset_layout_choices();

    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_COALESCED);

    const char * prev_disable_graph = std::getenv("GGML_SYCL_DISABLE_GRAPH");
    setenv("GGML_SYCL_DISABLE_GRAPH", "1", 1);

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        printf("SKIP: SYCL backend unavailable\n");
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        printf("SKIP: buffer types unavailable\n");
        ggml_backend_free(backend);
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init failed\n");
        ggml_backend_free(backend);
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return false;
    }

    const int64_t ncols   = QK4_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows   = 4;
    const int64_t ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");
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
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
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
    std::vector<float>   input_data(ncols * ntokens, 0.25f);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        printf("FAIL: mul_mat graph compute failed\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return false;
    }

    layout_mode chosen_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl_get_layout_choice_for_tensor(weight, device_id, &chosen_layout)) {
        printf("FAIL: missing layout choice for weight after mul_mat\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return false;
    }
    if (chosen_layout != GGML_LAYOUT_COALESCED) {
        printf("FAIL: expected coalesced layout choice, got %d\n", (int) chosen_layout);
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (prev_disable_graph) {
            setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
        } else {
            unsetenv("GGML_SYCL_DISABLE_GRAPH");
        }
        return false;
    }

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(output_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (prev_disable_graph) {
        setenv("GGML_SYCL_DISABLE_GRAPH", prev_disable_graph, 1);
    } else {
        unsetenv("GGML_SYCL_DISABLE_GRAPH");
    }
    return true;
}

static bool test_device_weight_layout_cache(int device_id) {
    reset_layout_choices();

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
    if (!buft) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to allocate SYCL weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    const size_t weight_bytes = ggml_nbytes(weight);
    std::vector<uint8_t> host_data(weight_bytes);
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache key for device layout test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get unified cache for device layout test\n");
        return false;
    }

    void * aos_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_AOS);
    if (!aos_ptr || !cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to cache AOS layout for device weight\n");
        return false;
    }

    void * coalesced_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_COALESCED);
    if (!coalesced_ptr || !cache->is_cached(key, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to cache COALESCED layout for device weight\n");
        return false;
    }

    if (cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "AOS cache entry not dropped after COALESCED cache for device weight\n");
        return false;
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool test_host_reorder_uses_host_cache(int device_id) {
    reset_layout_choices();

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols  = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows  = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to allocate CPU weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    std::vector<uint8_t> host_data(ggml_nbytes(weight));
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get cache key for host reorder test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    ggml_sycl::host_cache *    host_cache = ggml_sycl::get_host_cache_for_device(device_id);
    if (!cache || !host_cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get cache(s) for host reorder test\n");
        return false;
    }

    void * coalesced_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_COALESCED);
    if (!coalesced_ptr) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to cache coalesced layout for host weight\n");
        return false;
    }

    if (!cache->is_cached(key, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Unified cache missing coalesced entry for host weight\n");
        return false;
    }

    if (!host_cache->is_cached(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Host cache missing coalesced entry for host weight\n");
        return false;
    }

    cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED);
    host_cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED);

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    return true;
}

static bool test_device_reorder_skips_host_cache(int device_id) {
    reset_layout_choices();

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
    if (!buft) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols  = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows  = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to allocate SYCL weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    std::vector<uint8_t> host_data(ggml_nbytes(weight));
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache key for device reorder test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    ggml_sycl::host_cache *    host_cache = ggml_sycl::get_host_cache_for_device(device_id);
    if (!cache || !host_cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache(s) for device reorder test\n");
        return false;
    }

    void * coalesced_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_COALESCED);
    if (!coalesced_ptr) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to cache coalesced layout for device weight\n");
        return false;
    }

    if (!cache->is_cached(key, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Unified cache missing coalesced entry for device weight\n");
        return false;
    }

    if (host_cache->is_cached(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Host cache unexpectedly used for device weight\n");
        return false;
    }

    cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED);

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool test_layout_ptr_eviction_guard(int device_id) {
    reset_layout_choices();

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
    if (!buft) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context for eviction guard test\n");
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    ggml_set_name(weight, "attn_q.bias");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to allocate SYCL weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    auto * extra = new ggml_tensor_extra_gpu();
    weight->extra = extra;

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache key for eviction guard test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get unified cache for eviction guard test\n");
        return false;
    }

    std::vector<float> host_data(ggml_nelements(weight), 1.0f);
    bool               needs_fill = false;
    void *             layout_ptr = cache->ensure_cached_alloc(
        key, host_data.data(), ggml_nbytes(weight), ggml_nbytes(weight),
        ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);
    if (!layout_ptr || !cache->is_cached(key, GGML_LAYOUT_AOS)) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to cache AOS layout for eviction guard test\n");
        return false;
    }

    extra->layout.mode       = GGML_LAYOUT_AOS;
    extra->layout.data_ptr   = layout_ptr;
    extra->layout.size       = ggml_nbytes(weight);
    extra->layout.owns_memory = false;
    extra->layout.device_id  = device_id;
    extra->layout.qtype      = weight->type;
    extra->layout.n_elements = ggml_nelements(weight);
    extra->layout.n_experts  = 1;

    cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);

    void * resolved_ptr = ggml_sycl_get_layout_ptr(weight, device_id);
    if (resolved_ptr != weight->data) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Evicted layout pointer was reused unexpectedly\n");
        return false;
    }

    weight->extra = nullptr;
    delete extra;
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return true;
}

static bool test_model_load_host_buffer_avoids_pinned(int device_id) {
    GGML_UNUSED(device_id);

    ggml_backend_sycl_set_model_loading(true);
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    const size_t               size      = 128ULL * 1024ULL * 1024ULL;  // 128MB
    ggml_backend_buffer_t      buffer    = ggml_backend_buft_alloc_buffer(host_buft, size);
    ggml_backend_sycl_set_model_loading(false);

    if (buffer == nullptr) {
        fprintf(stderr, "test_model_load_host_buffer_avoids_pinned: allocation failed\n");
        return false;
    }

    void *             ptr   = ggml_backend_buffer_get_base(buffer);
    const sycl::usm::alloc typ = sycl::get_pointer_type(ptr, dpct::get_in_order_queue().get_context());
    ggml_backend_buffer_free(buffer);

    if (typ == sycl::usm::alloc::host || typ == sycl::usm::alloc::shared) {
        fprintf(stderr, "test_model_load_host_buffer_avoids_pinned: got USM alloc type %d\n", (int) typ);
        return false;
    }

    return true;
}

static bool test_model_load_preload_caches_weight(int device_id) {
    ggml_sycl::test_clear_host_weight_registry();
    ggml_backend_sycl_set_model_loading(true);

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "test_model_load_preload_caches_weight: backend init failed\n");
        ggml_backend_sycl_set_model_loading(false);
        return false;
    }

    ggml_init_params params{};
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_sycl_set_model_loading(false);
        ggml_backend_free(backend);
        fprintf(stderr, "test_model_load_preload_caches_weight: ctx init failed\n");
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 64);
    ggml_set_name(weight, "attn_q.weight");

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    const size_t weight_size             = ggml_backend_buft_get_alloc_size(host_buft, weight);
    ggml_backend_buffer_t weight_buf     = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    if (!weight_buf) {
        ggml_backend_sycl_set_model_loading(false);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "test_model_load_preload_caches_weight: buffer alloc failed\n");
        return false;
    }

    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_sycl_register_weight_usage(ggml_get_name(weight), GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT);

    std::vector<uint8_t> weight_data(ggml_nbytes(weight), 0);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());

    ggml_backend_sycl_set_model_loading(false);

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!cache || !key.valid || !cache->is_cached_any(key)) {
        fprintf(stderr, "test_model_load_preload_caches_weight: cache miss\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    ggml_sycl::test_clear_layout_override();
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE_TILED", "1", 1);
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "256", 1);

    const auto & info = ggml_sycl_info();
    if (info.device_count <= 0) {
        fprintf(stderr, "No SYCL devices available; skipping test.\n");
        return 0;
    }

    const int device_id = 0;
    const bool xmx_supported = info.devices[device_id].xmx_caps.supported &&
                               info.devices[device_id].xmx_caps.supports_int8;

    bool ok = true;
    ok &= test_layout_selection(device_id, xmx_supported);
    ok &= test_aos_drop(device_id);
    ok &= test_mul_mat_layout_choice_coalesced(device_id);
    ok &= test_host_reorder_uses_host_cache(device_id);
    ok &= test_device_reorder_skips_host_cache(device_id);
    ok &= test_device_weight_layout_cache(device_id);
    ok &= test_layout_ptr_eviction_guard(device_id);
    ok &= test_model_load_host_buffer_avoids_pinned(device_id);
    ok &= test_model_load_preload_caches_weight(device_id);

    if (ok) {
        fprintf(stderr, "layout-cache tests passed.\n");
    }
    return ok ? 0 : 1;
}
#endif

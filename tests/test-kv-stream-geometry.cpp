#include "../ggml/src/ggml-kv-stream.h"
#include "../src/llama-kv-stream-config.h"
#include "testing.h"

#include <algorithm>
#include <limits>

using status = ggml_kv_stream_status;
using operand = ggml_kv_stream_operand;

// Expected CUDA contract from d873e5db9, not an advertisement of kernels in this new branch.
static ggml_kv_stream_type_capabilities reference_type(int32_t type) {
    ggml_kv_stream_type_capabilities cap{type};
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_IQ4_NL:
            cap = {type, true, true, false, true}; break;
        case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
            cap = {type, true, true, true, true}; break;
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ1_M:
        case GGML_TYPE_MXFP4: case GGML_TYPE_NVFP4:
            cap = {type, true, false, false, true}; break;
        case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0:
            cap = {type, true, false, false, false}; break;
        default: break;
    }
    return cap;
}

static ggml_kv_stream_capabilities reference_pair(int32_t k, int32_t v, bool all_quants = true) {
    return {reference_type(k), reference_type(v), all_quants, true};
}

static ggml_kv_stream_shape shape(int32_t k = GGML_TYPE_Q8_0, int32_t v = GGML_TYPE_Q4_0) {
    return {k, v, 256, 256, 4, 256, 128};
}

static llama_kv_stream_config model() {
    llama_kv_stream_config config;
    config.enabled = true;
    config.arch = LLM_ARCH_QWEN35;
    config.target_context = config.flash_attention = config.kv_offload = config.cuda_backend = true;
    config.sequences = config.devices = 1;
    config.context_tokens = 262144;
    for (int i = 0; i < 16; ++i) config.layers.push_back({i*4 + 3, shape(), reference_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0), true});
    return config;
}

// Construct metadata without allocating buffers. Q is token-major; the output uses GGML's permuted axes.
struct attention {
    ggml_tensor q{}, k{}, v{}, mask{}, dst{};
    ggml_kv_stream_attention_limits limits{256, 256, 256, 256, 128};

    attention() {
        q.type = dst.type = GGML_TYPE_F32;
        k.type = GGML_TYPE_Q8_0;
        v.type = GGML_TYPE_Q4_0;
        mask.type = GGML_TYPE_F16;
        const int64_t nq[] = {256, 513, 16, 1}, nk[] = {256, 1024, 4, 1}, nd[] = {256, 16, 513, 1};
        std::copy(nq, nq+4, q.ne);
        std::copy(nk, nk+4, k.ne);
        std::copy(nk, nk+4, v.ne);
        std::copy(nd, nd+4, dst.ne);
        q.nb[0] = 4; q.nb[2] = 1024; q.nb[1] = 16384; q.nb[3] = 16384*513;
        k.nb[0] = 34; k.nb[2] = 272; k.nb[1] = 1088; k.nb[3] = 1088*1024;
        v.nb[0] = 18; v.nb[2] = 144; v.nb[1] = 576; v.nb[3] = 576*1024;
        dst.nb[0] = 4; dst.nb[1] = 1024; dst.nb[2] = 16384; dst.nb[3] = 16384*513;
        mask.ne[0] = 1024; mask.ne[1] = 544; mask.ne[2] = mask.ne[3] = 1;
        mask.nb[0] = 2; mask.nb[1] = 2048; mask.nb[2] = mask.nb[3] = 2048*544;
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &q; dst.src[1] = &k; dst.src[2] = &v; dst.src[3] = &mask;
    }
    ggml_kv_stream_result validate(ggml_kv_stream_execution & output) {
        return ggml_kv_stream_attention_validate(&dst, limits, reference_pair(k.type, v.type), 256, output);
    }
};

int main() {
    testing t;
    t.test("q8_q4_reference_page_and_full_context_bytes", [](testing & t) {
        ggml_kv_stream_layout layout;
        if (!t.assert_true(ggml_kv_stream_layout_make(shape(), 256, layout).status == status::success)) return;
        t.assert_equal(size_t(272), layout.k_row_bytes);
        t.assert_equal(size_t(144), layout.v_row_bytes);
        t.assert_equal(size_t(278528), layout.k_bytes);
        t.assert_equal(size_t(278528), layout.v_offset);
        t.assert_equal(size_t(147456), layout.v_bytes);
        t.assert_equal(size_t(425984), layout.bytes);
        t.assert_equal(size_t(1), layout.pages);
        t.assert_equal(size_t(256), layout.tail_tokens);
        t.assert_true(ggml_kv_stream_layout_make(shape(), 262144, layout).status == status::success);
        t.assert_equal(size_t(436207616), layout.bytes);
    });

    t.test("nine_online_types_all_pairs_use_independent_rows", [](testing & t) {
        struct row { ggml_type type; size_t bytes; };
        const row rows[] = {{GGML_TYPE_F32,1024}, {GGML_TYPE_F16,512}, {GGML_TYPE_BF16,512},
            {GGML_TYPE_Q8_0,272}, {GGML_TYPE_Q5_0,176}, {GGML_TYPE_Q5_1,192},
            {GGML_TYPE_Q4_0,144}, {GGML_TYPE_Q4_1,160}, {GGML_TYPE_IQ4_NL,144}};
        for (const auto & k : rows) for (const auto & v : rows) {
            for (bool all : {false, true}) {
                ggml_kv_stream_execution plan;
                auto caps = reference_pair(k.type, v.type, all);
                if (!t.assert_true(ggml_kv_stream_resolve(shape(k.type, v.type), caps, 256, plan).status == status::success)) return;
                t.assert_equal(k.bytes, plan.storage.k_row_bytes);
                t.assert_equal(v.bytes, plan.storage.v_row_bytes);
                t.assert_equal((k.bytes + v.bytes)*4*256, plan.storage.bytes);
                const bool direct = all && caps.k.direct_attention && caps.v.direct_attention;
                t.assert_true((plan.attention == ggml_kv_stream_attention::direct) == direct);
                t.assert_equal(direct ? size_t(0) : size_t(1048576), plan.conversion.bytes);
            }
        }
    });

    t.test("all_enum_entries_classify_storage_without_asserting", [](testing & t) {
        for (int type = -1; type <= GGML_TYPE_COUNT; ++type) {
            const int32_t dtype = type;
            ggml_kv_stream_layout layout;
            const auto result = ggml_kv_stream_layout_make(shape(dtype, GGML_TYPE_F16), 256, layout);
            t.assert_true((result.status == status::success) == reference_type(dtype).storage);
        }
    });

    t.test("capabilities_fail_closed_independently_for_k_and_v", [](testing & t) {
        for (bool key : {false, true}) {
            auto caps = reference_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
            auto & selected = key ? caps.k : caps.v;
            const auto side = key ? operand::k : operand::v;
            ggml_kv_stream_execution result;
            result.storage.bytes = 77;
            selected.storage = false;
            auto failed = ggml_kv_stream_resolve(shape(), caps, 256, result);
            t.assert_true(failed.status == status::unsupported_storage && failed.operand == side);
            selected.storage = true; selected.online_write = false;
            failed = ggml_kv_stream_resolve(shape(), caps, 256, result);
            t.assert_true(failed.status == status::unsupported_write && failed.operand == side);
            selected.online_write = true; selected.type = GGML_TYPE_F32;
            failed = ggml_kv_stream_resolve(shape(), caps, 256, result);
            t.assert_true(failed.status == status::capability_mismatch && failed.operand == side);
            t.assert_equal(size_t(77), result.storage.bytes);
        }
    });

    t.test("storage_or_conversion_does_not_imply_online_kv_support", [](testing & t) {
        for (auto type : {GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XS, GGML_TYPE_NVFP4, GGML_TYPE_TQ1_0}) {
            ggml_kv_stream_execution plan;
            t.assert_true(ggml_kv_stream_resolve(shape(type, GGML_TYPE_Q4_0),
                reference_pair(type, GGML_TYPE_Q4_0), 256, plan).status == status::unsupported_write);
        }
        auto caps = reference_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        caps.direct_pair = false; caps.v.convert_f16 = false;
        ggml_kv_stream_execution plan;
        t.assert_true(ggml_kv_stream_resolve(shape(), caps, 256, plan).status == status::unsupported_attention);
        caps.v.convert_f16 = true; caps.f16_attention = false;
        t.assert_true(ggml_kv_stream_resolve(shape(), caps, 256, plan).status == status::unsupported_attention);
    });

    t.test("separate_planes_are_not_interleaved_page_records", [](testing & t) {
        auto spec = shape(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
        spec.head_dim_k = spec.head_dim_v = 32; spec.heads = 1; spec.page_tokens = 1;
        ggml_kv_stream_layout page, span;
        t.assert_true(ggml_kv_stream_layout_make(spec, 1, page).status == status::success);
        t.assert_true(ggml_kv_stream_layout_make(spec, 3, span).status == status::success);
        t.assert_equal(size_t(162), page.bytes);
        t.assert_equal(size_t(54), span.k_bytes);
        t.assert_equal(size_t(128), span.v_offset);
        t.assert_equal(size_t(230), span.bytes);
        t.assert_true(span.bytes != 3*page.bytes);
    });

    t.test("tail_pages_and_empty_cache_have_exact_offsets", [](testing & t) {
        ggml_kv_stream_layout layout;
        t.assert_true(ggml_kv_stream_layout_make(shape(), 0, layout).status == status::success);
        t.assert_equal(size_t(0), layout.bytes);
        t.assert_equal(size_t(0), layout.pages);
        ggml_kv_stream_page page;
        t.assert_true(ggml_kv_stream_page_make(shape(), 273, 1, page).status == status::success);
        t.assert_equal(size_t(256), page.token_begin);
        t.assert_equal(size_t(17), page.tokens);
        t.assert_equal(size_t(278528), page.k_offset);
        t.assert_equal(size_t(147456), page.v_offset);
        t.assert_equal(size_t(18496), page.k_bytes);
        t.assert_equal(size_t(9792), page.v_bytes);
        t.assert_true(ggml_kv_stream_page_make(shape(), 273, 2, page).status == status::page_out_of_range);
        t.assert_equal(size_t(17), page.tokens);
        t.assert_true(ggml_kv_stream_page_make(shape(), 0, SIZE_MAX, page).status == status::page_out_of_range);
    });

    t.test("checked_sizes_and_alignment_preserve_output_on_failure", [](testing & t) {
        ggml_kv_stream_layout result; result.bytes = 77;
        auto spec = shape();
        spec.head_dim_k = 255;
        t.assert_true(ggml_kv_stream_layout_make(spec, 1, result).status == status::invalid_shape);
        spec = shape(); spec.heads = INT64_MAX;
        t.assert_true(ggml_kv_stream_layout_make(spec, 256, result).status == status::overflow);
        spec = shape(); spec.alignment = 3;
        t.assert_true(ggml_kv_stream_layout_make(spec, 1, result).status == status::invalid_alignment);
        spec = shape(); spec.page_tokens = -1;
        t.assert_true(ggml_kv_stream_layout_make(spec, 1, result).status == status::invalid_shape);
        spec = {GGML_TYPE_F16, GGML_TYPE_F16, 1, 1, 1, 1, 128};
        t.assert_true(ggml_kv_stream_layout_make(spec, SIZE_MAX/2, result).status == status::overflow);
        t.assert_equal(size_t(77), result.bytes);
    });

    t.test("divide_quant_block_count_before_multiplication", [](testing & t) {
        if (sizeof(size_t) < 8) return;
        auto spec = shape(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
        spec.head_dim_k = (INT64_MAX/32)*32; spec.head_dim_v = 32; spec.heads = 1; spec.page_tokens = 1;
        ggml_kv_stream_layout result;
        t.assert_true(ggml_kv_stream_layout_make(spec, 1, result).status == status::success);
        t.assert_equal(size_t(spec.head_dim_k/32)*18, result.k_row_bytes);
    });

    t.test("attention_metadata_accepts_wide_queries_without_buffers", [](testing & t) {
        attention graph;
        ggml_kv_stream_execution plan;
        t.assert_true(graph.validate(plan).status == status::success);
        t.assert_equal(size_t(425984), plan.storage.bytes);
        graph.dst.src[3] = nullptr;
        t.assert_true(graph.validate(plan).status == status::success);
        graph.q.nb[1] = 1024; graph.q.nb[2] = 1024*513;
        t.assert_true(graph.validate(plan).status == status::success);
    });

    t.test("attention_rejects_invalid_heads_strides_mask_and_sinks", [](testing & t) {
        for (int bad = 0; bad < 10; ++bad) {
            attention graph;
            if (bad == 0) graph.k.ne[0] = 128;
            if (bad == 1) graph.q.ne[2] = 15;
            if (bad == 2) graph.k.nb[1] += 128;
            if (bad == 3) graph.v.nb[2] += 16;
            if (bad == 4) graph.mask.ne[0] = 512;
            if (bad == 5) graph.mask.type = GGML_TYPE_F32;
            if (bad == 6) graph.dst.src[4] = &graph.mask;
            if (bad == 7) graph.k.ne[1] = 1023;
            if (bad == 8) graph.dst.ne[1] = graph.dst.ne[2];
            if (bad == 9) graph.q.ne[3] = 2;
            ggml_kv_stream_execution output; output.storage.bytes = 99;
            t.assert_true(graph.validate(output).status != status::success);
            t.assert_equal(size_t(99), output.storage.bytes);
        }
    });

    t.test("initial_model_contract_and_uniform_layer_payload", [](testing & t) {
        auto config = model();
        llama_kv_stream_config_plan plan;
        if (sizeof(size_t) < 8) {
            t.assert_true(llama_kv_stream_config_validate(config, plan).status == llama_kv_stream_config_status::geometry_error);
            return;
        }
        t.assert_true(llama_kv_stream_config_validate(config, plan).status == llama_kv_stream_config_status::success);
        t.assert_true(plan.enabled);
        t.assert_equal(size_t(16), plan.layers);
        t.assert_equal(size_t(262144), plan.padded_context_tokens);
        t.assert_equal(uint64_t(6979321856), uint64_t(plan.all_layers_bytes));
        config.context_tokens = 257;
        t.assert_true(llama_kv_stream_config_validate(config, plan).status == llama_kv_stream_config_status::success);
        t.assert_equal(size_t(512), plan.padded_context_tokens);
    });

    t.test("model_limits_fail_closed_and_disabled_is_noop", [](testing & t) {
        for (int bad = 0; bad < 13; ++bad) {
            auto config = model();
            if (bad == 0) config.arch = LLM_ARCH_LLAMA;
            if (bad == 1) config.target_context = false;
            if (bad == 2) config.sequences = 2;
            if (bad == 3) config.devices = 2;
            if (bad == 4) config.cuda_backend = false;
            if (bad == 5) config.flash_attention = false;
            if (bad == 6) config.kv_offload = false;
            if (bad == 7) config.layers.clear();
            if (bad == 8) config.layers[1].id = config.layers[0].id;
            if (bad == 9) config.context_tokens = SIZE_MAX;
            if (bad == 10) config.layers[0].shape.page_tokens = 128;
            if (bad == 11) config.layers[0].shape.head_dim_k = 128;
            if (bad == 12) config.layers[0].offloaded = false;
            llama_kv_stream_config_plan plan; plan.all_layers_bytes = 77;
            t.assert_true(llama_kv_stream_config_validate(config, plan).status != llama_kv_stream_config_status::success);
            t.assert_equal(size_t(77), plan.all_layers_bytes);
        }
        llama_kv_stream_config disabled;
        llama_kv_stream_config_plan plan; plan.enabled = true;
        t.assert_true(llama_kv_stream_config_validate(disabled, plan).status == llama_kv_stream_config_status::success);
        t.assert_true(!plan.enabled);
    });

    t.test("equal_total_page_bytes_do_not_imply_uniform_kv_planes", [](testing & t) {
        auto config = model();
        config.layers[1].shape = shape(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
        config.layers[1].capabilities = reference_pair(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
        llama_kv_stream_config_plan output;
        t.assert_true(llama_kv_stream_config_validate(config, output).status == llama_kv_stream_config_status::nonuniform_geometry);
        config = model();
        config.layers[1].capabilities.direct_pair = false;
        t.assert_true(llama_kv_stream_config_validate(config, output).status == llama_kv_stream_config_status::nonuniform_geometry);
    });

    t.test("model_head_count_is_bounded_before_runtime_narrowing", [](testing & t) {
        auto config = model();
        for (auto & layer : config.layers) layer.shape.heads = int64_t(INT32_MAX) + 1;
        llama_kv_stream_config_plan output;
        t.assert_true(llama_kv_stream_config_validate(config, output).status == llama_kv_stream_config_status::unsupported_geometry);
    });

    t.test("all_layer_storage_overflow_is_not_hidden_by_valid_pages", [](testing & t) {
        if (sizeof(size_t) < 8) return;
        auto config = model();
        config.layers.clear();
        for (int i = 0; i < 128; ++i) {
            auto spec = shape(); spec.heads = INT32_MAX;
            config.layers.push_back({i, spec, reference_pair(spec.type_k, spec.type_v), true});
        }
        llama_kv_stream_config_plan output; output.all_layers_bytes = 77;
        const auto result = llama_kv_stream_config_validate(config, output);
        t.assert_true(result.status == llama_kv_stream_config_status::geometry_error);
        t.assert_true(result.geometry.status == status::overflow);
        t.assert_equal(size_t(77), output.all_layers_bytes);
    });

    t.test("page_boundaries_cover_each_plane_once", [](testing & t) {
        for (size_t tokens : {0, 1, 255, 256, 257, 511, 512, 513}) {
            ggml_kv_stream_layout whole;
            t.assert_true(ggml_kv_stream_layout_make(shape(), tokens, whole).status == status::success);
            size_t k_sum = 0, v_sum = 0, covered = 0;
            for (size_t i = 0; i < whole.pages; ++i) {
                ggml_kv_stream_page page;
                t.assert_true(ggml_kv_stream_page_make(shape(), tokens, i, page).status == status::success);
                t.assert_equal(k_sum, page.k_offset);
                t.assert_equal(v_sum, page.v_offset);
                k_sum += page.k_bytes; v_sum += page.v_bytes; covered += page.tokens;
            }
            t.assert_equal(tokens, covered);
            t.assert_equal(whole.k_bytes, k_sum);
            t.assert_equal(whole.v_bytes, v_sum);
        }
    });

    t.test("attention_grid_overflow_and_zero_limits_are_rejected", [](testing & t) {
        attention graph;
        ggml_kv_stream_execution output;
        graph.q.ne[1] = graph.dst.ne[2] = INT32_MAX;
        t.assert_true(graph.validate(output).status == status::overflow);
        graph.limits.key_token_multiple = 0;
        t.assert_true(graph.validate(output).status == status::invalid_shape);
    });

    return t.summary();
}

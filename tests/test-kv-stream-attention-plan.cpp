#include "../ggml/src/ggml-cuda/kv-stream-attention-plan.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-dispatch.h"
#include "testing.h"

#include <limits>
#include <utility>

using family = ggml_cuda_kv_stream_kernel_family;
using style = ggml_cuda_kv_stream_execution_style;
using status = ggml_cuda_kv_stream_plan_status;

struct fixture {
    ggml_cuda_kv_stream_attention_metadata metadata{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 24, 4, 2, 1023, 1024};
    ggml_cuda_kv_stream_plan_support support{true, true, true, true, true};
    ggml_cuda_kv_stream_workspace_requirements requirements{true, 0, 4096, 128, 32768};

    // Failure must not publish a partial replacement of a previously usable descriptor.
    void reject(testing & t, status expected, family selected = family::vector, style execution = style::resumable) const {
        ggml_cuda_kv_stream_attention_plan output{};
        output.family = family::tile;
        output.style = style::native;
        output.metadata.active_tokens = 17;
        output.output_allocation_bytes = 512;
        output.requirements.scratch_bytes = 128;
        t.assert_true(ggml_cuda_kv_stream_attention_plan_make(metadata, selected, execution, support, requirements, output) == expected);
        t.assert_true(output.family == family::tile && output.style == style::native);
        t.assert_equal(int64_t(17), output.metadata.active_tokens);
        t.assert_equal(size_t(512), output.output_allocation_bytes);
        t.assert_equal(size_t(128), output.requirements.scratch_bytes);
    }
};

int main() {
    testing t;
    t.test("stock_family_is_supplied_not_selected_by_a_second_policy", [](testing & t) {
        fixture f;
        for (auto selected : {family::vector, family::tile, family::mma}) {
            ggml_cuda_kv_stream_attention_plan output;
            t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, selected, style::spanned, f.support, f.requirements, output) == status::success);
            t.assert_true(output.family == selected && output.style == style::spanned);
            t.assert_equal(f.metadata.active_tokens, output.metadata.active_tokens);
            t.assert_equal(f.metadata.type_k, output.metadata.type_k);
                t.assert_equal(size_t(49152), output.output_allocation_bytes);
            t.assert_equal(size_t(4096), output.requirements.scratch_bytes);
            t.assert_equal(size_t(32768), output.requirements.shared_bytes);
        }
    });
    t.test("existing_modern_dispatch_observations_remain_unchanged", [](testing & t) {
        using path = ggml_cuda_kv_stream_attention_path;
        for (int cc : {860, 890, 1200}) {
            fixture f;
            for (uint32_t queries : {1u, 2u}) {
                f.metadata.queries = queries;
                const auto observed = ggml_cuda_kv_stream_attention_select(cc, queries, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
                const auto expected = cc == 860 && queries == 2 ? path::mma : path::vector;
                t.assert_true(observed == expected);
                const auto supplied = observed == path::mma ? family::mma : family::vector;
                ggml_cuda_kv_stream_attention_plan output;
                t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, supplied, style::resumable, f.support, f.requirements, output) == status::success);
                t.assert_true(output.family == supplied);
            }
        }
        t.assert_true(ggml_cuda_kv_stream_attention_select(860, 2, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0) == path::none);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860, 2, GGML_TYPE_F16, GGML_TYPE_F16) == path::none);
    });
    t.test("pascal_tg1_quantized_admission_does_not_enable_later_query_widths", [](testing & t) {
        using path = ggml_cuda_kv_stream_attention_path;
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q4_1},
                std::pair{GGML_TYPE_Q4_0,GGML_TYPE_F16}, std::pair{GGML_TYPE_BF16,GGML_TYPE_Q8_0}}) {
            t.assert_true(ggml_cuda_kv_stream_attention_select(610,1,pair.first,pair.second) == path::vector);
            for (uint32_t queries : {0u,2u,3u,4u})
                t.assert_true(ggml_cuda_kv_stream_attention_select(610,queries,pair.first,pair.second) == path::none);
            for (int cc : {500,600,609})
                t.assert_true(ggml_cuda_kv_stream_attention_select(cc,1,pair.first,pair.second) == path::none);
        }
        for (auto pair : {std::pair{GGML_TYPE_F16,GGML_TYPE_F16}, std::pair{GGML_TYPE_BF16,GGML_TYPE_F16}})
            t.assert_true(ggml_cuda_kv_stream_attention_select(610,1,pair.first,pair.second) == path::none);
    });
    t.test("vector_resume_values_follow_compiled_copy_width_not_physical_gpu", [](testing & t) {
        for (auto type : {GGML_TYPE_F16,GGML_TYPE_BF16}) {
            t.assert_equal(16u,ggml_cuda_kv_stream_vector_values_per_thread(type,610));
            t.assert_equal(32u,ggml_cuda_kv_stream_vector_values_per_thread(type,700));
            t.assert_equal(32u,ggml_cuda_kv_stream_vector_values_per_thread(type,1200));
        }
        for (auto type : {GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0}) {
            for (int cc : {610,700,860,1200})
                t.assert_equal(8u,ggml_cuda_kv_stream_vector_values_per_thread(type,cc));
        }
    });
    t.test("metadata_contract_has_no_new_query_or_head_size_policy", [](testing & t) {
        fixture f;
        f.metadata.head_dim_k = f.metadata.head_dim_v = 96;
        f.metadata.query_heads = 12;
        f.metadata.active_tokens = f.metadata.padded_tokens = 512;
        f.support.streamed_compiled = false;
        f.requirements.output_extra_bytes = 8192;
        for (int64_t queries : {1, 2, 3, 4, 16, 256}) {
            f.metadata.queries = queries;
            ggml_cuda_kv_stream_attention_plan output;
            t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, family::tile, style::native, f.support, f.requirements, output) == status::success);
            t.assert_equal(size_t(96*12*queries*sizeof(float)) + 8192, output.output_allocation_bytes);
        }
    });
    t.test("native_output_extras_and_global_scratch_are_separate_from_shared_memory", [](testing & t) {
        fixture f;
        f.support.streamed_compiled = false;
        f.requirements.output_extra_bytes = 2*256*1024*4*sizeof(uint16_t);
        ggml_cuda_kv_stream_attention_plan output;
        t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, family::mma, style::native, f.support, f.requirements, output) == status::success);
        t.assert_equal(size_t(49152) + f.requirements.output_extra_bytes, output.output_allocation_bytes);
        t.assert_equal(size_t(4096), output.requirements.scratch_bytes);
        t.assert_equal(size_t(32768), output.requirements.shared_bytes);
        f.reject(t, status::missing_streamed_implementation);
        f.support.streamed_compiled = true;
        f.reject(t, status::invalid_requirements);
        f.requirements.output_extra_bytes = 0;
        f.requirements.scratch_bytes = 0;
        t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, family::vector, style::spanned, f.support, f.requirements, output) == status::success);
        t.assert_equal(size_t(49152), output.output_allocation_bytes);
    });
    t.test("availability_failures_have_distinct_reasons", [](testing & t) {
        fixture f;
        f.reject(t, status::stock_unavailable, family::none);
        f.support.stock_compiled = false;
        f.reject(t, status::missing_stock_code);
        f.support = {true, false, true, true, true};
        f.reject(t, status::missing_streamed_implementation);
        f.support = {true, true, false, true, true};
        f.reject(t, status::unsupported_device_features);
        f.support = {true, true, true, false, true};
        f.reject(t, status::unsupported_geometry);
        f.support = {true, true, true, true, false};
        f.reject(t, status::unsupported_launch_resources);
        f.support = {true, true, true, true, true};
        f.requirements.known = false;
        f.reject(t, status::missing_requirements);
    });
    t.test("invalid_metadata_is_rejected_before_type_queries", [](testing & t) {
        for (int type : {-1, int(GGML_TYPE_COUNT), std::numeric_limits<int>::max()}) {
            fixture f; f.metadata.type_k = type; f.reject(t, status::invalid_metadata);
            f = {}; f.metadata.type_v = type; f.reject(t, status::invalid_metadata);
        }
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            if (ggml_blck_size(ggml_type(type)) > 0) continue;
            fixture f; f.metadata.type_k = type; f.reject(t, status::invalid_metadata);
        }
        for (auto member : {&ggml_cuda_kv_stream_attention_metadata::head_dim_k,
                &ggml_cuda_kv_stream_attention_metadata::head_dim_v,
                &ggml_cuda_kv_stream_attention_metadata::query_heads,
                &ggml_cuda_kv_stream_attention_metadata::kv_heads,
                &ggml_cuda_kv_stream_attention_metadata::queries,
                &ggml_cuda_kv_stream_attention_metadata::active_tokens,
                &ggml_cuda_kv_stream_attention_metadata::padded_tokens}) {
            for (int64_t value : {int64_t(0), int64_t(-1)}) {
                fixture f; f.metadata.*member = value; f.reject(t, status::invalid_metadata);
            }
        }
        fixture f;
        f.metadata.query_heads = 23; f.reject(t, status::invalid_metadata);
        f = {}; f.metadata.head_dim_k = 255; f.reject(t, status::invalid_metadata);
        f = {}; f.metadata.head_dim_v = 255; f.reject(t, status::invalid_metadata);
        f = {}; f.metadata.padded_tokens = 1022; f.reject(t, status::invalid_metadata);
        f = {}; f.metadata.queries = 1024; f.reject(t, status::invalid_metadata);
        f = {}; f.reject(t, status::invalid_metadata, static_cast<family>(255));
        f.reject(t, status::invalid_metadata, family::vector, static_cast<style>(255));
    });
    t.test("unsupported_pairs_are_backend_evidence_not_a_q8_q4_allocation_rule", [](testing & t) {
        for (auto key : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1}) {
            for (auto value : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0}) {
                fixture f; f.metadata.type_k = key; f.metadata.type_v = value;
                ggml_cuda_kv_stream_attention_plan output;
                t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, family::tile, style::spanned, f.support, f.requirements, output) == status::success);
                f.support.streamed_compiled = false;
                f.reject(t, status::missing_streamed_implementation, family::tile, style::spanned);
                f.support.streamed_compiled = true;
                f.support.geometry = false;
                f.reject(t, status::unsupported_geometry, family::tile, style::spanned);
            }
        }
    });
    t.test("unrepresentable_sizes_and_invalid_alignment_do_not_publish", [](testing & t) {
        fixture f;
        f.requirements.output_extra_bytes = SIZE_MAX;
        f.reject(t, status::overflow, family::mma, style::native);
        f = {}; f.requirements.scratch_bytes = SIZE_MAX; f.reject(t, status::overflow);
        f = {}; f.metadata.type_v = GGML_TYPE_F16;
        f.metadata.head_dim_v = int64_t(1) << 32;
        f.metadata.query_heads = int64_t(1) << 32;
        f.reject(t, status::overflow);
        for (size_t alignment : {size_t(0), size_t(3), size_t(127)}) {
            f = {}; f.requirements.scratch_alignment = alignment; f.reject(t, status::invalid_requirements);
        }
    });
    t.test("descriptors_snapshot_metadata_and_preserve_exact_scratch_counts", [](testing & t) {
        fixture f;
        f.requirements.scratch_bytes = 129;
        ggml_cuda_kv_stream_attention_plan output;
        t.assert_true(ggml_cuda_kv_stream_attention_plan_make(f.metadata, family::vector, style::resumable, f.support, f.requirements, output) == status::success);
        f.metadata.active_tokens = 7;
        f.requirements.scratch_bytes = 1;
        t.assert_equal(int64_t(1023), output.metadata.active_tokens);
        t.assert_equal(size_t(129), output.requirements.scratch_bytes);
        t.assert_equal(size_t(128), output.requirements.scratch_alignment);
        t.assert_equal(size_t(49152), output.output_allocation_bytes);
    });
    return t.summary();
}

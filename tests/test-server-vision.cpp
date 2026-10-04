#include "../tools/server/server-common.h"
#include "../tools/server/server-task.h"
#include "testing.h"
#include <cstring>

static common_params qualified() {
    common_params p;
    p.mmproj.path = "projector.gguf";
    p.shared_device_memory_bytes = 1024*1048576;
    p.fit_params = false; p.n_parallel = 1; p.mmproj_use_gpu = true;
    p.cache_type_k = GGML_TYPE_Q8_0; p.cache_type_v = GGML_TYPE_Q4_0;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return p;
}

int main() {
    testing t;
    t.test("serial_no_mtp_vision_arena_is_admitted_without_changing_params", [](testing & t) {
        auto p = qualified();
        t.assert_true(server_uses_vision_arena(p));
        t.assert_true(server_vision_arena_config_error(p) == nullptr);
        t.assert_true(p.shared_device_memory_bytes == 1024*1048576 && !p.fit_params);
    });
    t.test("unsupported_vision_configurations_stay_rejected", [](testing & t) {
        auto check = [&](auto change) { auto p = qualified(); change(p); t.assert_true(server_vision_arena_config_error(p) != nullptr); };
        check([](auto & p) { p.n_parallel = 2; });
        check([](auto & p) { p.fit_params = true; });
        check([](auto & p) { p.shared_device_memory_bytes = 0; p.kv_stream_pool_bytes = 1048576; });
        check([](auto & p) { p.kv_stream_pool_bytes = 1048576; });
        check([](auto & p) { p.mmproj_use_gpu = false; });
        check([](auto & p) { p.cache_type_k = GGML_TYPE_F16; });
        check([](auto & p) { p.cache_type_v = GGML_TYPE_Q8_0; });
        check([](auto & p) { p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; });
        check([](auto & p) { p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP}; p.speculative.draft.n_max = 4; });
        check([](auto & p) { p.speculative.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}; });
        check([](auto & p) { p.kv_stream_auxiliary_layers = 1; });
        check([](auto & p) { p.embedding = true; });
        check([](auto & p) { p.no_kv_offload = true; });
    });
    t.test("media_prefix_and_serialization_preserve_whole_chunks_and_model_positions", [](testing & t) {
        mtmd::input_chunks_ptr input(mtmd_test_create_input_chunks());
        auto * image = mtmd_input_chunks_get(input.get(),1);
        const size_t rows = mtmd_input_chunk_get_n_tokens(image);
        const auto positions = mtmd_input_chunk_get_n_pos(image);
        server_tokens a(llama_tokens{},true),b(llama_tokens{},true);
        a.push_back(1); a.push_back(image); a.push_back(2);
        b.push_back(1); b.push_back(image); b.push_back(3);
        t.assert_equal(size_t(1)+rows,a.get_common_prefix(b));
        t.assert_equal(llama_pos(2+positions),a.pos_next());
        t.assert_equal(llama_pos(1+positions),a.pos_next(1+rows));
        const auto serialized = a.serialize();
        llama_tokens packed(serialized.size()/sizeof(llama_token));
        std::memcpy(packed.data(),serialized.data(),serialized.size());
        auto restored = server_tokens::deserialize(packed,true);
        t.assert_equal(a.size(),restored.get_common_prefix(a));
        t.assert_equal(a.pos_next(),restored.pos_next());
        restored.keep_first(1+rows);
        t.assert_equal(size_t(1)+rows,restored.size());
        t.assert_equal(llama_pos(1+positions),restored.pos_next());
    });
    t.test("eager_multimodal_and_text_only_paths_keep_existing_policy", [](testing & t) {
        auto p = qualified(); p.shared_device_memory_bytes = 0; p.n_parallel = 4;
        t.assert_true(!server_uses_vision_arena(p) && !server_vision_arena_config_error(p));
        p = qualified(); p.mmproj.path.clear(); p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        t.assert_true(!server_uses_vision_arena(p) && !server_vision_arena_config_error(p));
    });
    t.test("serial_embedded_mtp_vision_is_admitted_only_with_bounded_drafting", [](testing & t) {
        auto p = qualified();
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        for (int length = 1; length <= 3; ++length) {
            p.speculative.draft.n_max = length;
            t.assert_true(server_vision_arena_config_error(p) == nullptr);
        }
        p.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        t.assert_true(server_vision_arena_config_error(p) != nullptr);
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE}; p.kv_stream_auxiliary_layers = 1;
        t.assert_true(server_vision_arena_config_error(p) != nullptr);
    });
    t.test("request_cannot_override_the_qualified_execution_mode", [](testing & t) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.tokens = server_tokens(llama_tokens{1,2,3},true);
        t.assert_true(!server_vision_arena_request_error(task));
        task.params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.params.speculative.draft.n_max = 3;
        t.assert_true(server_vision_arena_request_error(task, true) == nullptr);
        task.params.speculative.draft.n_max = 4;
        t.assert_true(server_vision_arena_request_error(task, true) != nullptr);
        task.params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE}; task.params.lora[0] = 1.0f;
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.params.lora.clear(); task.type = SERVER_TASK_TYPE_EMBEDDING;
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.type = SERVER_TASK_TYPE_COMPLETION; task.tokens = server_tokens(llama_tokens{LLAMA_TOKEN_NULL},true);
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
    });
    return t.summary();
}

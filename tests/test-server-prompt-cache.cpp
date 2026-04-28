#include "server-task.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static server_prompt make_prompt(std::initializer_list<llama_token> tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens(tokens), false);
    return prompt;
}

static void add_checkpoint(server_prompt & prompt, int64_t n_tokens) {
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = n_tokens;
    checkpoint.n_tokens = n_tokens;
    checkpoint.data     = { 1, 2, 3, 4 };
    prompt.checkpoints.push_back(std::move(checkpoint));
}

static void add_checkpoint_with_bounds(server_prompt & prompt, llama_pos pos_max, int64_t n_tokens, uint8_t marker) {
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = pos_max;
    checkpoint.n_tokens = n_tokens;
    checkpoint.data     = { marker };
    prompt.checkpoints.push_back(std::move(checkpoint));
}

static void test_find_checkpoint_before_tail_truncation_pos() {
    server_prompt prompt = make_prompt({ 1, 2, 3, 4, 5, 6, 7, 8 });

    add_checkpoint_with_bounds(prompt, 3, 4, 4);
    add_checkpoint_with_bounds(prompt, 5, 6, 6);
    add_checkpoint_with_bounds(prompt, 7, 12, 12); // invalid: more tokens than prompt

    const server_prompt_checkpoint * latest = server_prompt_find_checkpoint_before_pos(prompt, 7);
    require(latest != nullptr, "expected a checkpoint before tail truncation position");
    require(latest->n_tokens == 6, "expected latest compatible checkpoint before p0");
    require(latest->data == std::vector<uint8_t>{ 6 }, "expected latest compatible checkpoint data");

    const server_prompt_checkpoint * earlier = server_prompt_find_checkpoint_before_pos(prompt, 5);
    require(earlier != nullptr, "expected an earlier checkpoint before p0");
    require(earlier->n_tokens == 4, "expected checkpoint with pos_max strictly before p0");

    require(server_prompt_find_checkpoint_before_pos(prompt, 3) == nullptr,
            "checkpoint at pos_max >= p0 must not be used for tail truncation restore");
}

static void test_oaicompat_chat_streams_reasoning_delta() {
    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    parser_params.generation_prompt = "<think>";

    task_result_state state(parser_params);

    server_task_result_cmpl_partial partial = {};
    partial.content = "I am thinking";
    partial.n_decoded = 1;
    partial.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    partial.oaicompat_model = "test-model";
    partial.oaicompat_cmpl_id = "chatcmpl-test";
    partial.update(state);

    json chunks = partial.to_json_oaicompat_chat();
    bool found_reasoning = false;
    for (const auto & chunk : chunks) {
        if (!chunk.contains("choices") || chunk.at("choices").empty()) {
            continue;
        }
        const auto & delta = chunk.at("choices").at(0).at("delta");
        if (delta.contains("reasoning_content") && delta.at("reasoning_content") == "I am thinking") {
            found_reasoning = true;
        }
    }

    require(found_reasoning, "streaming chat response should broadcast reasoning_content deltas");
}

static void test_oaicompat_chat_final_contains_reasoning() {
    server_task_result_cmpl_final final = {};
    final.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    final.oaicompat_model = "test-model";
    final.oaicompat_cmpl_id = "chatcmpl-test";
    final.include_usage = true;
    final.oaicompat_msg.role = "assistant";
    final.oaicompat_msg.reasoning_content = "I am thinking";
    final.oaicompat_msg.content = "The answer is 4.";

    json body = final.to_json_oaicompat_chat();
    const auto & message = body.at("choices").at(0).at("message");
    require(message.contains("reasoning_content"), "final chat response should contain reasoning_content");
    require(message.at("reasoning_content") == "I am thinking",
            "final chat response should extract reasoning before </think>");
    require(message.at("content") == "The answer is 4.",
            "final chat response should keep post-thinking content separate");
}

static void test_full_removal_keeps_exact_shorter_without_checkpoint() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });

    require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate initial long prompt");
    require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "short exact prompt should be cached when the longer state has no restorable prefix checkpoint");
    require(cache.states.size() == 2,
            "cache should retain both long and short prompts without a restorable prefix checkpoint");
}

static void test_full_removal_reuses_longer_checkpoint_for_shorter_prompt() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
    add_checkpoint(long_prompt, short_prompt.n_tokens());

    require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate checkpointed long prompt");
    require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) == nullptr,
            "short exact prompt should be skipped when a longer checkpoint can restore it");
    require(cache.states.size() == 1,
            "checkpointed long prompt should make the exact shorter prompt redundant");
}

static void test_full_removal_only_removes_obsolete_shorter_with_checkpoint() {
    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });

        require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "long prompt without checkpoint should still be cached");
        require(cache.states.size() == 2,
                "short prompt must not be removed when long prompt cannot restore that prefix");
    }

    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
        add_checkpoint(long_prompt, short_prompt.n_tokens());

        require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate checkpointed long prompt");
        require(cache.states.size() == 1,
                "short prompt should be removed once long prompt has a restorable prefix checkpoint");
        require(cache.states.front().n_tokens() == long_prompt.n_tokens(),
                "remaining cache entry should be the checkpointed long prompt");
    }
}

int main() {
    test_find_checkpoint_before_tail_truncation_pos();
    test_oaicompat_chat_streams_reasoning_delta();
    test_oaicompat_chat_final_contains_reasoning();
    test_full_removal_keeps_exact_shorter_without_checkpoint();
    test_full_removal_reuses_longer_checkpoint_for_shorter_prompt();
    test_full_removal_only_removes_obsolete_shorter_with_checkpoint();

    return 0;
}

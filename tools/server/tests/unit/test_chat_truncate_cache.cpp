// TODO AI generated tests, to be revised

#include "server-common.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

// Helpers

static common_chat_msg make_msg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role    = role;
    msg.content = content;
    return msg;
}

static common_chat_msg make_msg_with_image(const std::string & role, const std::string & text) {
    common_chat_msg msg;
    msg.role = role;
    msg.content_parts.push_back({"text", text});
    msg.content_parts.push_back({"media_marker", ""});
    return msg;
}

static std::vector<common_chat_msg> build_conversation(int n_turns, bool include_final_user = true) {
    std::vector<common_chat_msg> msgs;
    msgs.push_back(make_msg("system", "You are helpful."));
    for (int i = 1; i <= n_turns; ++i) {
        msgs.push_back(make_msg("user",      "User message " + std::to_string(i)));
        msgs.push_back(make_msg("assistant", "Asst message " + std::to_string(i)));
    }
    if (include_final_user) {
        msgs.push_back(make_msg("user", "Final user message"));
    }
    return msgs;
}

// ---- Tests ----

static void test_hash_deterministic() {
    auto msgs = build_conversation(3);
    std::vector<raw_buffer> files;

    auto h1 = chat_compute_hashes(msgs, files);
    auto h2 = chat_compute_hashes(msgs, files);

    assert(h1.full_hash   == h2.full_hash);
    assert(h1.prefix_hash == h2.prefix_hash);
    assert(h1.n_messages   == h2.n_messages);
    printf("  PASS test_hash_deterministic\n");
}

static void test_hash_changes_with_content() {
    auto msgs1 = build_conversation(3);
    auto msgs2 = build_conversation(3);
    msgs2.back().content = "Different final message";
    std::vector<raw_buffer> files;

    auto h1 = chat_compute_hashes(msgs1, files);
    auto h2 = chat_compute_hashes(msgs2, files);

    assert(h1.full_hash != h2.full_hash);
    printf("  PASS test_hash_changes_with_content\n");
}

static void test_prefix_hash_matches_previous_full() {
    // Simulate: request 1 = [sys, u1], request 2 = [sys, u1, a1, u2]
    // prefix of request 2 (up to second-to-last user = u1) should match full hash of request 1
    std::vector<raw_buffer> files;

    std::vector<common_chat_msg> req1;
    req1.push_back(make_msg("system", "You are helpful."));
    req1.push_back(make_msg("user", "First question"));

    std::vector<common_chat_msg> req2;
    req2.push_back(make_msg("system", "You are helpful."));
    req2.push_back(make_msg("user",      "First question"));
    req2.push_back(make_msg("assistant", "First answer"));
    req2.push_back(make_msg("user",      "Second question"));

    auto h1 = chat_compute_hashes(req1, files);
    auto h2 = chat_compute_hashes(req2, files);

    assert(h2.prefix_hash == h1.full_hash);
    assert(h2.full_hash   != h1.full_hash);
    printf("  PASS test_prefix_hash_matches_previous_full\n");
}

static void test_prefix_hash_zero_with_single_user() {
    // Only one user message -> no prefix possible
    std::vector<common_chat_msg> msgs;
    msgs.push_back(make_msg("system", "Hello"));
    msgs.push_back(make_msg("user", "Only user msg"));
    std::vector<raw_buffer> files;

    auto h = chat_compute_hashes(msgs, files);
    assert(h.prefix_hash == 0);
    printf("  PASS test_prefix_hash_zero_with_single_user\n");
}

static void test_prefix_hash_with_multimodal() {
    // Verify that media file sizes affect the hash
    std::vector<common_chat_msg> msgs;
    msgs.push_back(make_msg("system", "Hello"));
    msgs.push_back(make_msg_with_image("user", "Look at this"));
    msgs.push_back(make_msg("assistant", "I see it"));
    msgs.push_back(make_msg_with_image("user", "Another image"));

    raw_buffer file1(100, 0xFF);
    raw_buffer file2(200, 0xAA);

    std::vector<raw_buffer> files1 = {file1, file2};
    std::vector<raw_buffer> files2 = {file1, file2};
    // Same files -> same hash
    auto h1 = chat_compute_hashes(msgs, files1);
    auto h2 = chat_compute_hashes(msgs, files2);
    assert(h1.full_hash == h2.full_hash);

    // Different file size -> different hash
    raw_buffer file2_diff(300, 0xBB);
    std::vector<raw_buffer> files3 = {file1, file2_diff};
    auto h3 = chat_compute_hashes(msgs, files3);
    assert(h1.full_hash != h3.full_hash);
    printf("  PASS test_prefix_hash_with_multimodal\n");
}

static void test_apply_cached_drop_text() {
    // Build inputs with 4 droppable turns + final user
    common_chat_templates_inputs inputs;
    inputs.messages = build_conversation(4, true);
    // Messages: sys, u1, a1, u2, a2, u3, a3, u4, a4, u_final = 10 messages

    size_t original_size = inputs.messages.size();
    assert(original_size == 10);

    // Drop 2 turns (u1+a1, u2+a2 = 4 messages)
    apply_cached_drop(inputs, nullptr, 2);
    assert(inputs.messages.size() == original_size - 4);
    // System should still be first
    assert(inputs.messages[0].role == "system");
    // u3 should now be the first user message
    assert(inputs.messages[1].content == "User message 3");
    printf("  PASS test_apply_cached_drop_text\n");
}

static void test_apply_cached_drop_clamps() {
    // Dropping more than available turns should clamp
    common_chat_templates_inputs inputs;
    inputs.messages = build_conversation(2, true);
    // Droppable turns: u1+a1, u2+a2 (2 turns). Final user is protected.

    apply_cached_drop(inputs, nullptr, 100); // way more than available
    // Should have: sys + final_user = 2 messages
    assert(inputs.messages.size() == 2);
    assert(inputs.messages[0].role == "system");
    assert(inputs.messages[1].content == "Final user message");
    printf("  PASS test_apply_cached_drop_clamps\n");
}

static void test_apply_cached_drop_zero_noop() {
    common_chat_templates_inputs inputs;
    inputs.messages = build_conversation(3, true);
    size_t original_size = inputs.messages.size();

    apply_cached_drop(inputs, nullptr, 0);
    assert(inputs.messages.size() == original_size);
    printf("  PASS test_apply_cached_drop_zero_noop\n");
}

static void test_apply_cached_drop_with_files() {
    common_chat_templates_inputs inputs;
    inputs.messages.push_back(make_msg("system", "Hi"));
    inputs.messages.push_back(make_msg_with_image("user", "Image 1"));
    inputs.messages.push_back(make_msg("assistant", "Got it"));
    inputs.messages.push_back(make_msg_with_image("user", "Image 2"));
    inputs.messages.push_back(make_msg("assistant", "Seen"));
    inputs.messages.push_back(make_msg_with_image("user", "Image 3")); // last user, protected

    raw_buffer f1(10, 0), f2(20, 0), f3(30, 0);
    std::vector<raw_buffer> files = {f1, f2, f3};

    // Drop 1 turn (u1 + a1) which has 1 file
    apply_cached_drop(inputs, &files, 1);
    assert(files.size() == 2);
    assert(inputs.messages.size() == 4); // sys, u2, a2, u3
    assert(inputs.messages[1].content_parts[0].text == "Image 2");
    printf("  PASS test_apply_cached_drop_with_files\n");
}

static void test_cache_invalidation() {
    chat_truncate_memory cache;
    cache.cache[42] = {3, 10};
    cache.cached_n_ctx = 512;
    cache.cached_max_keep = 0.5f;
    cache.cached_n_predict = 64;

    // Same config -> no invalidation
    cache.invalidate_if_config_changed(512, 0.5f, 64);
    assert(cache.cache.size() == 1);

    // Different config -> cache cleared
    cache.invalidate_if_config_changed(1024, 0.5f, 64);
    assert(cache.cache.empty());

    // Config values are set even when cache is empty, so next call with same config doesn't invalidate
    cache.cache[99] = {1, 5};
    cache.invalidate_if_config_changed(1024, 0.5f, 64);
    assert(cache.cache.size() == 1); // entry preserved, config unchanged
    printf("  PASS test_cache_invalidation\n");
}

static void test_cache_exact_hit_flow() {
    // Simulate the 3-path logic: exact hit should reuse n_drop
    chat_truncate_memory cache;
    std::vector<raw_buffer> files;

    auto msgs = build_conversation(5, true);
    auto h = chat_compute_hashes(msgs, files);

    // Populate cache as if previous request computed n_drop=2
    cache.cache[h.full_hash] = {2, h.n_messages};

    // Look up: should be an exact hit
    auto it = cache.cache.find(h.full_hash);
    assert(it != cache.cache.end());
    assert(it->second.n_messages == h.n_messages);
    assert(it->second.n_drop == 2);
    printf("  PASS test_cache_exact_hit_flow\n");
}

static void test_cache_prefix_hit_flow() {
    // Simulate: cache has entry for req1, req2's prefix matches req1's full_hash
    chat_truncate_memory cache;
    std::vector<raw_buffer> files;

    std::vector<common_chat_msg> req1;
    req1.push_back(make_msg("system", "Hi"));
    req1.push_back(make_msg("user", "Q1"));

    std::vector<common_chat_msg> req2;
    req2.push_back(make_msg("system", "Hi"));
    req2.push_back(make_msg("user",      "Q1"));
    req2.push_back(make_msg("assistant", "A1"));
    req2.push_back(make_msg("user",      "Q2"));

    auto h1 = chat_compute_hashes(req1, files);
    auto h2 = chat_compute_hashes(req2, files);

    // Cache req1's result
    cache.cache[h1.full_hash] = {0, h1.n_messages};

    // Lookup req2: no exact hit
    assert(cache.cache.find(h2.full_hash) == cache.cache.end());
    // But prefix hit: h2.prefix_hash == h1.full_hash
    auto it = cache.cache.find(h2.prefix_hash);
    assert(it != cache.cache.end());
    assert(it->second.n_drop == 0);
    printf("  PASS test_cache_prefix_hit_flow\n");
}

static void test_cache_miss_flow() {
    chat_truncate_memory cache;
    std::vector<raw_buffer> files;

    auto msgs = build_conversation(3, true);
    auto h = chat_compute_hashes(msgs, files);

    // Empty cache -> miss
    assert(cache.cache.find(h.full_hash) == cache.cache.end());
    assert(h.prefix_hash == 0 || cache.cache.find(h.prefix_hash) == cache.cache.end());
    printf("  PASS test_cache_miss_flow\n");
}

// E2E test using real model + templates + chat_truncate_cached().
// Requires a GGUF model path via argv[1] or LLAMACPP_TEST_MODELFILE env var.
// Skipped gracefully if no model is available.
static void test_e2e_multi_turn_with_model(const char * model_path) {
    //
    // Loads vocab-only from the model, reads a chatml jinja template,
    // and runs chat_truncate_cached() for a 4-request session with real
    // tokenization. n_ctx=256, max_keep=0.5 → target=128.
    //
    // Session with many turns to force truncation:
    //   req1: miss   → truncation fires, n_drop stored
    //   req2: prefix → base_drop reused + extra if needed
    //   req3: prefix → base_drop reused + extra if needed
    //   req4: exact  → no recomputation
    //
    const int32_t n_ctx     = 512;
    const float   max_keep  = 0.3f;
    const int32_t n_predict = -1;
    const int32_t target    = chat_truncate_target_tokens(n_ctx, max_keep, n_predict);
    const int32_t threshold = chat_truncate_threshold(n_ctx, n_predict, target);

    // Load model vocab-only
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "    SKIP: failed to load model from %s\n", model_path);
        llama_backend_free();
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Load chatml template (the model's built-in, or fall back to a simple one)
    common_chat_templates_ptr tmpls(common_chat_templates_init(model, ""));
    assert(tmpls.get() != nullptr);

    chat_truncate_memory cache;

    std::vector<raw_buffer> files;

    // Build initial conversation with enough turns to force truncation in 256 tokens.
    // Each subsequent request extends this by appending [assistant, user] — mirroring
    // how a real chat client sends messages. This ensures prefix_hash of reqN+1
    // matches full_hash of reqN.
    const int n_turns = 20;
    std::vector<common_chat_msg> msgs;
    msgs.push_back(make_msg("system", "You are a helpful assistant."));
    for (int i = 1; i <= n_turns; ++i) {
        msgs.push_back(make_msg("user",      "Please explain topic " + std::to_string(i) + " in detail."));
        msgs.push_back(make_msg("assistant", "Here is my explanation of topic " + std::to_string(i) + "."));
    }
    msgs.push_back(make_msg("user", "Final question."));

    // -- Request 1: miss --
    auto msgs1 = msgs;
    {
        common_chat_templates_inputs inputs;
        inputs.messages = msgs1;
        auto result = chat_truncate_cached(
            inputs, tmpls.get(), vocab, files, cache,
            n_ctx, max_keep, n_predict, target, threshold);
        assert(result == CHAT_TRUNCATE_CACHE_MISS);
        int32_t tokens_after = chat_n_tokens(inputs, tmpls.get(), vocab);
        assert(tokens_after < target);
        printf("    req1: miss, tokens_after=%d (target=%d)\n", tokens_after, target);
    }
    auto h1 = chat_compute_hashes(msgs1, files);
    size_t n_drop_1 = cache.cache[h1.full_hash].n_drop;
    assert(n_drop_1 > 0);
    printf("    req1: n_drop=%zu stored\n", n_drop_1);

    // -- Request 2: prefix hit (append assistant reply + new user question) --
    auto msgs2 = msgs1;
    msgs2.push_back(make_msg("assistant", "Here is my answer to the final question."));
    msgs2.push_back(make_msg("user",      "Please explain topic 21 in detail."));
    {
        common_chat_templates_inputs inputs;
        inputs.messages = msgs2;
        auto result = chat_truncate_cached(
            inputs, tmpls.get(), vocab, files, cache,
            n_ctx, max_keep, n_predict, target, threshold);
        assert(result == CHAT_TRUNCATE_CACHE_PREFIX);
        int32_t tokens_after = chat_n_tokens(inputs, tmpls.get(), vocab);
        assert(tokens_after < target);
        printf("    req2: prefix hit, tokens_after=%d\n", tokens_after);
    }
    auto h2 = chat_compute_hashes(msgs2, files);
    size_t n_drop_2 = cache.cache[h2.full_hash].n_drop;
    assert(n_drop_2 >= n_drop_1);
    printf("    req2: n_drop=%zu (>= req1's %zu)\n", n_drop_2, n_drop_1);

    // -- Request 3: prefix hit (grew again) --
    auto msgs3 = msgs2;
    msgs3.push_back(make_msg("assistant", "Here is my explanation of topic 21."));
    msgs3.push_back(make_msg("user",      "Please explain topic 22 in detail."));
    {
        common_chat_templates_inputs inputs;
        inputs.messages = msgs3;
        auto result = chat_truncate_cached(
            inputs, tmpls.get(), vocab, files, cache,
            n_ctx, max_keep, n_predict, target, threshold);
        assert(result == CHAT_TRUNCATE_CACHE_PREFIX);
        int32_t tokens_after = chat_n_tokens(inputs, tmpls.get(), vocab);
        assert(tokens_after < target);
        printf("    req3: prefix hit, tokens_after=%d\n", tokens_after);
    }
    auto h3 = chat_compute_hashes(msgs3, files);
    size_t n_drop_3 = cache.cache[h3.full_hash].n_drop;
    assert(n_drop_3 >= n_drop_2);
    printf("    req3: n_drop=%zu (>= req2's %zu)\n", n_drop_3, n_drop_2);

    // -- Request 4: prefix hit with extra=0 --
    // Extend with a very short turn. After applying base_drop from msgs3,
    // the tiny extra should still fit under target without more drops.
    auto msgs4 = msgs3;
    msgs4.push_back(make_msg("assistant", "Ok."));
    msgs4.push_back(make_msg("user", "Hi"));
    {
        common_chat_templates_inputs inputs;
        inputs.messages = msgs4;
        auto result = chat_truncate_cached(
            inputs, tmpls.get(), vocab, files, cache,
            n_ctx, max_keep, n_predict, target, threshold);
        assert(result == CHAT_TRUNCATE_CACHE_PREFIX);
        int32_t tokens_after = chat_n_tokens(inputs, tmpls.get(), vocab);
        assert(tokens_after < target);
        printf("    req4: prefix hit, tokens_after=%d\n", tokens_after);
    }
    auto h4 = chat_compute_hashes(msgs4, files);
    size_t n_drop_4 = cache.cache[h4.full_hash].n_drop;
    assert(n_drop_4 == n_drop_3); // extra=0: same n_drop as base
    printf("    req4: n_drop=%zu == req3's %zu (extra=0)\n", n_drop_4, n_drop_3);

    // -- Request 5: exact hit (resend req4) --
    {
        common_chat_templates_inputs inputs;
        inputs.messages = msgs4;
        auto result = chat_truncate_cached(
            inputs, tmpls.get(), vocab, files, cache,
            n_ctx, max_keep, n_predict, target, threshold);
        assert(result == CHAT_TRUNCATE_CACHE_EXACT);
        int32_t tokens_after = chat_n_tokens(inputs, tmpls.get(), vocab);
        assert(tokens_after < target);
        printf("    req5: exact hit, tokens_after=%d\n", tokens_after);
    }
    assert(cache.cache[h4.full_hash].n_drop == n_drop_4);

    llama_model_free(model);
    llama_backend_free();
    printf("  PASS test_e2e_multi_turn_with_model\n");
}

int main(int argc, char * argv[]) {
    printf("Running chat truncation cache tests...\n\n");

    printf("Hash tests:\n");
    test_hash_deterministic();
    test_hash_changes_with_content();
    test_prefix_hash_matches_previous_full();
    test_prefix_hash_zero_with_single_user();
    test_prefix_hash_with_multimodal();

    printf("\napply_cached_drop tests:\n");
    test_apply_cached_drop_text();
    test_apply_cached_drop_clamps();
    test_apply_cached_drop_zero_noop();
    test_apply_cached_drop_with_files();

    printf("\nCache logic tests:\n");
    test_cache_invalidation();
    test_cache_exact_hit_flow();
    test_cache_prefix_hit_flow();
    test_cache_miss_flow();

    printf("\nE2E multi-turn test (with model):\n");
    const char * model_path = (argc > 1) ? argv[1] : getenv("LLAMACPP_TEST_MODELFILE");
    if (model_path && strlen(model_path) > 0) {
        test_e2e_multi_turn_with_model(model_path);
    } else {
        printf("  SKIP: no model provided (pass path as arg or set LLAMACPP_TEST_MODELFILE)\n");
    }

    printf("\nAll tests passed!\n");
    return 0;
}

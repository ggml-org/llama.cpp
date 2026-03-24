#include "matcher-sampler.h"
#include "unicode.h"

#include "llama.h"
#include "ggml.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

// Reasoning budget sampler test helper
// These tests use nullptr vocab which safely falls back to treating all tokens as complete
// (The UTF-8 boundary detection logic is tested separately in test_utf8_boundary_detection)
static void test_reasoning_budget(
    const char * test_name,
    const std::vector<llama_token> & sequence,
    const std::vector<llama_token> & start_tokens,
    const std::vector<llama_token> & end_tokens,
    const std::vector<llama_token> & forced_tokens,
    int32_t budget,
    matcher_ssm_rb_state initial_state,
    size_t expected_force_start,   // token index where forcing should start (SIZE_MAX = never)
    size_t expected_force_end      // token index where forcing should end (after this, no more forcing)
) {
    // Find the maximum token ID to ensure our vocab covers all tokens
    llama_token max_token = 0;
    for (auto t : sequence) max_token = std::max(max_token, t);
    for (auto t : start_tokens) max_token = std::max(max_token, t);
    for (auto t : end_tokens) max_token = std::max(max_token, t);
    for (auto t : forced_tokens) max_token = std::max(max_token, t);

    // Create a minimal sampler with mock vocabulary
    // For this test, we use nullptr as vocab since we're testing state transitions
    // The UTF-8 boundary check will treat all tokens as complete (safe fallback)
    auto * sampler = common_matcher_sampler_init_reasoning_budget(
        nullptr,  // vocab - not used for basic state machine tests
        start_tokens,
        end_tokens,
        forced_tokens,
        budget,
        initial_state
    );

    // Create a test token data array for checking forcing behavior
    // Vocab size must be large enough to include all tokens (start, end, forced, sequence)
    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t)max_token + 1;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token)i, logf((float)(i+1)), 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    size_t actual_force_start = SIZE_MAX;
    size_t actual_force_end = SIZE_MAX;

    // Feed the sequence and track when forcing occurs
    for (size_t i = 0; i < sequence.size(); i++) {
        llama_sampler_accept(sampler, sequence[i]);

        // Check if we're in forcing state by applying and seeing if logits are modified
        cur_p.selected = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            cur[j].logit = logf((float)(j+1));  // reset logits
        }

        llama_sampler_apply(sampler, &cur_p);

        // Check if forcing is active (all logits except one should be -INFINITY)
        size_t finite_count = 0;
        llama_token finite_token = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            if (std::isfinite(cur[j].logit)) {
                finite_count++;
                finite_token = cur[j].id;
            }
        }

        fprintf(stderr, "    i=%zu: token=%d, finite_count=%zu, finite_token=%d\n", i, (int)sequence[i], finite_count, (int)finite_token);

        if (finite_count == 1) {
            if (actual_force_start == SIZE_MAX) {
                actual_force_start = i;
            }
            actual_force_end = i;
        } else if (actual_force_start != SIZE_MAX && actual_force_end != SIZE_MAX) {
            // Forcing stopped
            break;
        }
    }

    llama_sampler_free(sampler);

    // Verify forcing occurred at expected positions
    if (expected_force_start == SIZE_MAX) {
        if (actual_force_start != SIZE_MAX) {
            fprintf(stderr, "Test '%s' FAILED: Expected no forcing, but forcing occurred at %zu\n", test_name, actual_force_start);
            GGML_ASSERT(false && "Expected no forcing, but forcing occurred");
        }
    } else {
        if (actual_force_start == SIZE_MAX) {
            fprintf(stderr, "Test '%s' FAILED: Expected forcing but none occurred\n", test_name);
            GGML_ASSERT(false && "Expected forcing but none occurred");
        }
        if (actual_force_start != expected_force_start) {
            fprintf(stderr, "Test '%s' FAILED: Forcing started at %zu, expected %zu\n", test_name, actual_force_start, expected_force_start);
            GGML_ASSERT(false && "Forcing started at wrong position");
        }
    }

    if (expected_force_end != SIZE_MAX) {
        if (actual_force_end < expected_force_end) {
            fprintf(stderr, "Test '%s' FAILED: Forcing ended at %zu, expected >= %zu\n", test_name, actual_force_end, expected_force_end);
            GGML_ASSERT(false && "Forcing ended too early");
        }
    }

    fprintf(stderr, "  Test '%s' passed (force_start=%zu, force_end=%zu)\n", test_name, actual_force_start, actual_force_end);
    (void)sequence;
}

// UTF-8 boundary detection unit test
// Tests common_utf8_is_complete() from unicode.h
static void test_utf8_boundary_detection() {
    // Complete sequences
    GGML_ASSERT(common_utf8_is_complete("hello"));
    GGML_ASSERT(common_utf8_is_complete(""));
    GGML_ASSERT(common_utf8_is_complete("\xC2\xA0"));            // complete 2-byte UTF-8 (U+00A0)
    GGML_ASSERT(common_utf8_is_complete("\xE2\x80\x9C"));        // complete 3-byte UTF-8 (left double quote)
    GGML_ASSERT(common_utf8_is_complete("\xF0\x9F\x98\x80"));    // complete 4-byte UTF-8 (emoji)
    GGML_ASSERT(common_utf8_is_complete("abc\xC3\xA9"));         // ASCII + complete 2-byte

    // Incomplete sequences
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xC2", 1)));            // 2-byte start, missing continuation
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2\x80", 2)));        // 3-byte start + 1 cont, missing 1
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2", 1)));            // 3-byte start, missing 2
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F\x98", 3)));    // 4-byte start + 2 cont, missing 1
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F", 2)));        // 4-byte start + 1 cont, missing 2
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0", 1)));            // 4-byte start, missing 3
    GGML_ASSERT(!common_utf8_is_complete(std::string("\x80", 1)));            // orphan continuation byte

    // Mixed: ASCII followed by start of multi-byte
    GGML_ASSERT(!common_utf8_is_complete(std::string("hello\xC3", 6)));       // ASCII + incomplete 2-byte
    GGML_ASSERT(common_utf8_is_complete(std::string("hello\xC3\xA9", 7)));    // ASCII + complete 2-byte
}

// Tool call grammar SSM test helper
// Tests that the tool call grammar SSM correctly transitions between states
// and delegates apply to the grammar sampler when in GRAMMAR_SAMPLING state.
//
// Uses a mock "grammar sampler" that sets all logits to 0.0 (distinguishable
// from the default positive logits) to detect when grammar apply is active.
static const char * mock_grammar_name(const struct llama_sampler * /*smpl*/) {
    return "mock-grammar";
}

static void mock_grammar_accept(struct llama_sampler * smpl, llama_token token) {
    auto * accepted = (std::vector<llama_token> *) smpl->ctx;
    accepted->push_back(token);
}

static void mock_grammar_apply(struct llama_sampler * /*smpl*/, llama_token_data_array * cur_p) {
    // Mark all logits as 0.0 so the test can detect grammar application
    for (size_t i = 0; i < cur_p->size; i++) {
        cur_p->data[i].logit = 0.0f;
    }
}

static void mock_grammar_reset(struct llama_sampler * smpl) {
    auto * accepted = (std::vector<llama_token> *) smpl->ctx;
    accepted->clear();
}

static struct llama_sampler * mock_grammar_clone(const struct llama_sampler * smpl) {
    auto * accepted = (const std::vector<llama_token> *) smpl->ctx;
    static struct llama_sampler_i mock_grammar_i;  // forward ref
    return llama_sampler_init(&mock_grammar_i, new std::vector<llama_token>(*accepted));
}

static void mock_grammar_free(struct llama_sampler * smpl) {
    delete (std::vector<llama_token> *) smpl->ctx;
}

static struct llama_sampler_i mock_grammar_i = {
    /* .name              = */ mock_grammar_name,
    /* .accept            = */ mock_grammar_accept,
    /* .apply             = */ mock_grammar_apply,
    /* .reset             = */ mock_grammar_reset,
    /* .clone             = */ mock_grammar_clone,
    /* .free              = */ mock_grammar_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * create_mock_grammar() {
    return llama_sampler_init(&mock_grammar_i, new std::vector<llama_token>());
}

static const std::vector<llama_token> & mock_grammar_get_accepted(struct llama_sampler * smpl) {
    return *(const std::vector<llama_token> *) smpl->ctx;
}

static bool is_grammar_applied(struct llama_sampler * sampler, size_t n_vocab) {
    std::vector<llama_token_data> cur;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token)i, 1.0f, 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);

    // Mock grammar sets all logits to 0.0
    return cur_p.data[0].logit == 0.0f;
}

static void test_tool_call_grammar() {
    const std::vector<llama_token> think_start = {200};
    const std::vector<llama_token> think_end   = {201};
    const std::vector<std::vector<llama_token>> tool_starts = {{300, 301}};
    const size_t n_vocab = 400;

    // Test 1: Tool call trigger outside thinking activates grammar
    {
        auto * grammar = create_mock_grammar();
        auto * sampler = common_matcher_sampler_init_tool_call_grammar(
            think_start, think_end, tool_starts, grammar);

        // Outside thinking, send tool call start tokens
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        llama_sampler_accept(sampler, 50);  // random token
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        llama_sampler_accept(sampler, 300); // first token of tool_start
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        llama_sampler_accept(sampler, 301); // completes tool_start -> grammar active
        GGML_ASSERT(is_grammar_applied(sampler, n_vocab));

        // Check that trigger tokens were replayed into grammar
        // The mock grammar's accepted list should contain: 300, 301
        // (replayed on trigger) + whatever we accepted after
        // We need to get the grammar from inside the sampler... but we can't easily.
        // Instead, verify grammar stays active on subsequent tokens
        llama_sampler_accept(sampler, 60);
        GGML_ASSERT(is_grammar_applied(sampler, n_vocab));

        llama_sampler_free(sampler);
        fprintf(stderr, "  Test 'tool call trigger activates grammar' passed\n");
    }

    // Test 2: Tool call trigger inside thinking is suppressed
    {
        auto * grammar = create_mock_grammar();
        auto * sampler = common_matcher_sampler_init_tool_call_grammar(
            think_start, think_end, tool_starts, grammar);

        // Enter thinking
        llama_sampler_accept(sampler, 200);  // think_start
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        // Try tool call start inside thinking - should be ignored
        llama_sampler_accept(sampler, 300);
        llama_sampler_accept(sampler, 301);
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        // Exit thinking
        llama_sampler_accept(sampler, 201);  // think_end
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        // Now tool call should work
        llama_sampler_accept(sampler, 300);
        llama_sampler_accept(sampler, 301);
        GGML_ASSERT(is_grammar_applied(sampler, n_vocab));

        llama_sampler_free(sampler);
        fprintf(stderr, "  Test 'tool call suppressed inside thinking' passed\n");
    }

    // Test 3: Partial tool call match that resets
    {
        auto * grammar = create_mock_grammar();
        auto * sampler = common_matcher_sampler_init_tool_call_grammar(
            think_start, think_end, tool_starts, grammar);

        // Partial match then break
        llama_sampler_accept(sampler, 300);  // first token of tool_start
        llama_sampler_accept(sampler, 50);   // not 301, breaks the match
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        // Full match after reset
        llama_sampler_accept(sampler, 300);
        llama_sampler_accept(sampler, 301);
        GGML_ASSERT(is_grammar_applied(sampler, n_vocab));

        llama_sampler_free(sampler);
        fprintf(stderr, "  Test 'partial tool call match resets' passed\n");
    }

    // Test 4: Combined reasoning budget + tool call grammar
    {
        const std::vector<llama_token> rb_start  = {200};
        const std::vector<llama_token> rb_end    = {201};
        const std::vector<llama_token> rb_forced = {999, 201};

        auto * matcher = common_matcher_sampler_init_reasoning_budget(
            nullptr, rb_start, rb_end, rb_forced, 3, MATCHER_SSM_RB_IDLE);

        auto * grammar = create_mock_grammar();
        common_matcher_sampler_add_tool_call_grammar(
            matcher, think_start, think_end, tool_starts, grammar);

        // Tool call outside thinking should activate grammar
        llama_sampler_accept(matcher, 300);
        llama_sampler_accept(matcher, 301);
        GGML_ASSERT(is_grammar_applied(matcher, n_vocab));

        llama_sampler_free(matcher);
        fprintf(stderr, "  Test 'combined reasoning budget + tool call grammar' passed\n");
    }

    // Test 5: Multiple trigger sequences — second trigger fires
    {
        const std::vector<std::vector<llama_token>> multi_triggers = {{300, 301}, {400, 401}};

        auto * grammar = create_mock_grammar();
        auto * sampler = common_matcher_sampler_init_tool_call_grammar(
            think_start, think_end, multi_triggers, grammar);

        // First trigger doesn't match (wrong second token)
        llama_sampler_accept(sampler, 300);
        llama_sampler_accept(sampler, 50);   // breaks first trigger
        GGML_ASSERT(!is_grammar_applied(sampler, n_vocab));

        // Second trigger fires
        llama_sampler_accept(sampler, 400);
        llama_sampler_accept(sampler, 401);
        GGML_ASSERT(is_grammar_applied(sampler, n_vocab));

        llama_sampler_free(sampler);
        fprintf(stderr, "  Test 'multiple triggers - second fires' passed\n");
    }
}

int main(void) {
    // Reasoning budget sampler tests
    printf("Testing reasoning budget sampler... ");

    // Test 1: Basic budget with start/end tokens - no forcing (natural end before budget exhausted)
    {
        const std::vector<llama_token> start = {100};  // start token
        const std::vector<llama_token> end = {101};    // end token
        const std::vector<llama_token> forced = {102}; // forced token (not used in this test)
        const std::vector<llama_token> sequence = {100, 50, 51, 101, 52}; // start, two tokens, end, one more

        test_reasoning_budget("natural end before budget exhausted", sequence, start, end, forced,
            5,      // budget of 5 tokens
            MATCHER_SSM_RB_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    }

    // Test 2: Budget exhausted, forcing should occur
    // Flow: i=0 accept(100)->COUNTING, i=1 accept(50)->remaining=1, i=2 accept(51)->remaining=0->FORCING
    // Forcing is active at i=2 and i=3 (when apply() is called while in FORCING state)
    // At i=4, force_pos becomes 2 which equals forced_tokens.size(), so state becomes DONE
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101}; // forced message + end
        const std::vector<llama_token> sequence = {100, 50, 51, 52, 53}; // start + 4 tokens (budget=2)

        test_reasoning_budget("budget exhausted forcing", sequence, start, end, forced,
            2,      // budget of 2 tokens
            MATCHER_SSM_RB_IDLE,
            2,      // forcing starts at i=2 (after accept(51) depletes budget, apply() forces)
            3);     // forcing continues through i=3 (at i=4 state becomes DONE)
    }

    // Test 3: Activate immediately with budget=0, forcing should start right away
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 51, 52}; // start token first, then 3 tokens

        test_reasoning_budget("activate immediately budget=0", sequence, start, end, forced,
            0,      // budget of 0 tokens
            MATCHER_SSM_RB_COUNTING, // starts counting, promoted to FORCING since budget=0
            0,      // forcing starts at i=0 (after accept(100), budget=0 goes straight to FORCING)
            1);     // forcing continues through i=1 (at i=2 state becomes DONE)
    }

    // Test 4: No start/end tokens configured - passthrough (no forcing)
    {
        const std::vector<llama_token> start = {};
        const std::vector<llama_token> end = {};
        const std::vector<llama_token> forced = {102};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("no start/end configured", sequence, start, end, forced,
            2,      // budget
            MATCHER_SSM_RB_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing (no start/end configured)
    }

    // Test 5: Activate immediately with budget > 0, count down then force
    // Flow: i=0 accept(50)->remaining=1, i=1 accept(51)->remaining=0->FORCING
    // So forcing starts at i=1 (apply after accept sees FORCING with force_pos=0)
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("activate immediately with budget", sequence, start, end, forced,
            2,      // budget of 2 tokens
            MATCHER_SSM_RB_COUNTING,
            1,      // forcing starts at i=1 (after 2 accepts deplete budget)
            2);     // forcing continues through i=2
    }

    printf("OK (5 tests passed)\n");

    printf("Testing UTF-8 boundary detection... ");
    test_utf8_boundary_detection();
    printf("OK\n");

    printf("Testing tool call grammar SSM... ");
    test_tool_call_grammar();
    printf("OK (5 tests passed)\n");

    return 0;
}

#include "reasoning-budget.h"
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
    common_reasoning_budget_state initial_state,
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
    auto * sampler = common_reasoning_budget_init(
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
        // Check if we're in forcing state by applying and seeing if logits are modified
        cur_p.selected = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            cur[j].logit = logf((float)(j+1));  // reset logits
        }

        llama_sampler_apply(sampler, &cur_p);

        // Check if forcing is active (all logits except one should be -INFINITY)
        size_t not_neg_inf = 0;
        llama_token not_neg_inf_token = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            if (std::isfinite(cur[j].logit) || cur[j].logit > 0) { // +INFINITY
                not_neg_inf++;
                not_neg_inf_token = cur[j].id;
            }
        }

        llama_sampler_accept(sampler, sequence[i]);

        fprintf(stderr, "    i=%zu: token=%d, not_neg_inf_count=%zu, not_neg_inf_token=%d\n", i, (int)sequence[i], not_neg_inf, (int)not_neg_inf_token);

        if (not_neg_inf == 1) {
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
// Tests common_utf8_is_complete() from reasoning-budget.h
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
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    }

    // Test 2: Budget exhausted, forcing should occur
    // Flow: i=0 apply()->passthrough, accept(100)->COUNTING; i=1 accept(50)->remaining=1
    // i=2 accept(51)->remaining=0->FORCING; i=3 apply() forces token[0]; i=4 apply() forces token[1]
    // At i=4, accept() advances force_pos to 2 which equals forced_tokens.size(), so state becomes DONE
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101}; // forced message + end
        const std::vector<llama_token> sequence = {100, 50, 51, 52, 53}; // start + 4 tokens (budget=2)

        test_reasoning_budget("budget exhausted forcing", sequence, start, end, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_IDLE,
            3,      // forcing starts at i=3 (accept at i=2 depletes budget, apply at i=3 forces)
            4);     // forcing continues through i=4 (accept at i=4 transitions to DONE)
    }

    // Test 3: Activate immediately with budget=0, forcing should start right away
    // Flow: init promotes COUNTING+budget=0 to FORCING, so apply() sees FORCING at i=0
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 51, 52}; // start token first, then 3 tokens

        test_reasoning_budget("activate immediately budget=0", sequence, start, end, forced,
            0,      // budget of 0 tokens
            REASONING_BUDGET_COUNTING, // starts counting, promoted to FORCING since budget=0
            0,      // forcing starts at i=0 (initialized in FORCING, apply forces immediately)
            1);     // forcing continues through i=1 (accept at i=1 transitions to DONE)
    }

    // Test 4: No start/end tokens configured - passthrough (no forcing)
    {
        const std::vector<llama_token> start = {};
        const std::vector<llama_token> end = {};
        const std::vector<llama_token> forced = {102};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("no start/end configured", sequence, start, end, forced,
            2,      // budget
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing (no start/end configured)
    }

    // Test 5: Activate immediately with budget > 0, count down then force
    // Flow: i=0 accept(50)->remaining=1, i=1 accept(51)->remaining=0->FORCING
    // Forcing starts at i=2 (apply sees FORCING after accept at i=1 transitioned)
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("activate immediately with budget", sequence, start, end, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_COUNTING,
            2,      // forcing starts at i=2 (after 2 accepts deplete budget, apply at i=2 forces)
            3);     // forcing continues through i=3
    }

    // Test 6: Multi-block thinking. First block ends naturally at i=2, second
    // start tag at i=3 re-arms the budget, which then exhausts at i=5.
    // Regression: before this fix, DONE absorbed all subsequent tokens and a
    // second <think> block ran unbudgeted.
    // Flow: i=0 accept(100)->COUNTING rem=2; i=1 accept(50)->rem=1;
    //       i=2 accept(101)->end_matcher matches, DONE;
    //       i=3 accept(100)->re-arm, COUNTING rem=2;
    //       i=4 accept(60)->rem=1; i=5 accept(61)->rem=0->FORCING;
    //       i=6 apply()->forces token[0]=102, accept(62)->force_pos=1, stay FORCING;
    //       i=7 apply()->forces token[1]=101, accept(63)->force_pos=2->DONE.
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 101, 100, 60, 61, 62, 63};

        test_reasoning_budget("multi-block re-arms budget after DONE", sequence, start, end, forced,
            2,      // budget of 2 tokens (per block)
            REASONING_BUDGET_IDLE,
            6,      // forcing starts at i=6 (after second block exhausts at i=5)
            7);     // forcing continues through i=7
    }

    printf("OK (6 tests passed)\n");

    printf("Testing UTF-8 boundary detection... ");
    test_utf8_boundary_detection();
    printf("OK\n");

    return 0;
}

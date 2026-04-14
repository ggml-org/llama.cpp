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

static void test_reasoning_budget(
    const char * test_name,
    const std::vector<llama_token> & sequence,
    const std::vector<llama_token> & start_tokens,
    const std::vector<llama_token> & end_tokens,
    const std::vector<llama_token> & forced_tokens,
    const std::vector<llama_token> & message_tokens,
    int32_t budget,
    int32_t conclusion_budget,
    common_reasoning_budget_state initial_state,
    size_t expected_force_start,
    size_t expected_force_end
) {
    llama_token max_token = 0;
    for (size_t k = 0; k < sequence.size();      k++) { if (sequence[k]      > max_token) max_token = sequence[k];      }
    for (size_t k = 0; k < start_tokens.size();  k++) { if (start_tokens[k]  > max_token) max_token = start_tokens[k];  }
    for (size_t k = 0; k < end_tokens.size();    k++) { if (end_tokens[k]    > max_token) max_token = end_tokens[k];    }
    for (size_t k = 0; k < forced_tokens.size(); k++) { if (forced_tokens[k] > max_token) max_token = forced_tokens[k]; }
    for (size_t k = 0; k < message_tokens.size();k++) { if (message_tokens[k]> max_token) max_token = message_tokens[k];}

    auto * sampler = common_reasoning_budget_init(
        nullptr,
        start_tokens, end_tokens, forced_tokens, message_tokens,
        budget, conclusion_budget, initial_state
    );

    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t)(max_token + 1);
    for (size_t i = 0; i < n_vocab; i++) {
        llama_token_data d;
        d.id = (llama_token)i; d.logit = logf((float)(i+1)); d.p = 0.0f;
        cur.push_back(d);
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    size_t actual_force_start = SIZE_MAX;
    size_t actual_force_end   = SIZE_MAX;

    for (size_t i = 0; i < sequence.size(); i++) {
        cur_p.selected = -1;
        for (size_t j = 0; j < cur.size(); j++) { cur[j].logit = logf((float)(j+1)); }

        llama_sampler_apply(sampler, &cur_p);

        size_t finite_count = 0;
        llama_token finite_token = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            if (std::isfinite(cur[j].logit)) { finite_count++; finite_token = cur[j].id; }
        }

        llama_sampler_accept(sampler, sequence[i]);

        fprintf(stderr, "    i=%zu: token=%d, finite_count=%zu, finite_token=%d\n",
                i, (int)sequence[i], finite_count, (int)finite_token);

        if (finite_count == 1) {
            if (actual_force_start == SIZE_MAX) { actual_force_start = i; }
            actual_force_end = i;
        }
    }

    llama_sampler_free(sampler);

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
    if (expected_force_end != SIZE_MAX && actual_force_end < expected_force_end) {
        fprintf(stderr, "Test '%s' FAILED: Forcing ended at %zu, expected >= %zu\n", test_name, actual_force_end, expected_force_end);
        GGML_ASSERT(false && "Forcing ended too early");
    }

    fprintf(stderr, "  Test '%s' passed (force_start=%zu, force_end=%zu)\n", test_name, actual_force_start, actual_force_end);
}

static void test_utf8_boundary_detection() {
    GGML_ASSERT(common_utf8_is_complete("hello"));
    GGML_ASSERT(common_utf8_is_complete(""));
    GGML_ASSERT(common_utf8_is_complete("\xC2\xA0"));
    GGML_ASSERT(common_utf8_is_complete("\xE2\x80\x9C"));
    GGML_ASSERT(common_utf8_is_complete("\xF0\x9F\x98\x80"));
    GGML_ASSERT(common_utf8_is_complete("abc\xC3\xA9"));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xC2", 1)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2\x80", 2)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2", 1)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F\x98", 3)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F", 2)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0", 1)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("\x80", 1)));
    GGML_ASSERT(!common_utf8_is_complete(std::string("hello\xC3", 6)));
    GGML_ASSERT(common_utf8_is_complete(std::string("hello\xC3\xA9", 7)));
}

int main(void) {
    printf("Testing reasoning budget sampler... ");

    // Test 1: Natural end before budget exhausted
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {102}, msg = {};
        std::vector<llama_token> seq = {100, 50, 51, 101, 52};
        test_reasoning_budget("natural end before budget exhausted", seq, start, end, forced, msg, 5, 0, REASONING_BUDGET_IDLE, SIZE_MAX, SIZE_MAX);
    }

    // Test 2: Budget exhausted, forcing occurs
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {102, 101}, msg = {};
        std::vector<llama_token> seq = {100, 50, 51, 52, 53};
        test_reasoning_budget("budget exhausted forcing", seq, start, end, forced, msg, 2, 0, REASONING_BUDGET_IDLE, 3, 4);
    }

    // Test 3: Budget=0 forces immediately
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {102, 101}, msg = {};
        std::vector<llama_token> seq = {100, 50, 51, 52};
        test_reasoning_budget("activate immediately budget=0", seq, start, end, forced, msg, 0, 0, REASONING_BUDGET_COUNTING, 0, 1);
    }

    // Test 4: No start/end — passthrough
    {
        std::vector<llama_token> start = {}, end = {}, forced = {102}, msg = {};
        std::vector<llama_token> seq = {50, 51, 52, 53};
        test_reasoning_budget("no start/end configured", seq, start, end, forced, msg, 2, 0, REASONING_BUDGET_IDLE, SIZE_MAX, SIZE_MAX);
    }

    // Test 5: Start in COUNTING state, count down then force
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {102, 101}, msg = {};
        std::vector<llama_token> seq = {50, 51, 52, 53};
        test_reasoning_budget("activate immediately with budget", seq, start, end, forced, msg, 2, 0, REASONING_BUDGET_COUNTING, 2, 3);
    }

    // Test 6: Two-phase — model concludes naturally in conclusion window
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {101}, msg = {200};
        std::vector<llama_token> seq = {100, 50, 51, 200, 101, 52};
        test_reasoning_budget("two-phase natural end in conclusion window", seq, start, end, forced, msg, 2, 3, REASONING_BUDGET_IDLE, 3, 3);
    }

    // Test 7: Two-phase — conclusion budget exhausted, safety net fires
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {101}, msg = {200};
        std::vector<llama_token> seq = {100, 50, 51, 200, 52, 101};
        test_reasoning_budget("two-phase conclusion budget exhausted safety net fires", seq, start, end, forced, msg, 2, 1, REASONING_BUDGET_IDLE, 3, 5);
    }

    // Test 8: Two-phase — no message tokens, conclusion only (skips INJECTING)
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {101}, msg = {};
        std::vector<llama_token> seq = {100, 50, 51, 101, 52};
        test_reasoning_budget("two-phase no message tokens conclusion only", seq, start, end, forced, msg, 2, 5, REASONING_BUDGET_IDLE, SIZE_MAX, SIZE_MAX);
    }

    // Test 9: Backward compat — conclusion_budget=0 identical to original
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {102, 101}, msg = {};
        std::vector<llama_token> seq = {100, 50, 51, 52, 53};
        test_reasoning_budget("backward compat conclusion_budget=0", seq, start, end, forced, msg, 2, 0, REASONING_BUDGET_IDLE, 3, 4);
    }

    // Test 10: Two-phase — multi-token message (3 tokens all forced before CONCLUDING)
    {
        std::vector<llama_token> start = {100}, end = {101}, forced = {101}, msg = {200, 201, 202};
        std::vector<llama_token> seq = {100, 50, 51, 200, 201, 202, 101, 52};
        test_reasoning_budget("two-phase multi-token message injection", seq, start, end, forced, msg, 2, 5, REASONING_BUDGET_IDLE, 3, 5);
    }

    printf("OK (10 tests passed)\n");

    printf("Testing UTF-8 boundary detection... ");
    test_utf8_boundary_detection();
    printf("OK\n");

    return 0;
}
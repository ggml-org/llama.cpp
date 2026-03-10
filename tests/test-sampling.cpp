#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties(last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(1024, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

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
    bool activate_immediately,
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
    auto * sampler = llama_sampler_init_reasoning_budget(
        nullptr,  // vocab - not used for basic state machine tests
        start_tokens.data(), start_tokens.size(),
        end_tokens.data(), end_tokens.size(),
        forced_tokens.data(), forced_tokens.size(),
        budget,
        activate_immediately
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
// Tests the core logic used by the reasoning budget sampler to detect incomplete UTF-8 sequences
// This mirrors llama_utf8_is_incomplete() from llama-sampler.cpp
static void test_utf8_boundary_detection() {
    // Reimplement the same logic as llama_utf8_is_incomplete for testing
    auto is_incomplete = [](const std::string & s) -> bool {
        if (s.empty()) {
            return false;
        }

        int i = (int)s.size() - 1;
        int n_cont = 0;
        while (i >= 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) {
            n_cont++;
            i--;
        }

        if (i < 0) {
            return true;  // only continuation bytes, no leading byte
        }

        const unsigned char lead = static_cast<unsigned char>(s[i]);

        if ((lead & 0x80) == 0x00) {
            return n_cont > 0;  // ASCII followed by continuation bytes = malformed
        }

        int expected;
        if      ((lead & 0xE0) == 0xC0) { expected = 1; }
        else if ((lead & 0xF0) == 0xE0) { expected = 2; }
        else if ((lead & 0xF8) == 0xF0) { expected = 3; }
        else { return true; }  // invalid leading byte

        return n_cont < expected;
    };

    // Complete sequences — should NOT wait
    GGML_ASSERT(!is_incomplete("hello"));
    GGML_ASSERT(!is_incomplete(""));
    GGML_ASSERT(!is_incomplete("\xC2\xA0"));               // complete 2-byte UTF-8 (U+00A0)
    GGML_ASSERT(!is_incomplete("\xE2\x80\x9C"));           // complete 3-byte UTF-8 (left double quote)
    GGML_ASSERT(!is_incomplete("\xF0\x9F\x98\x80"));       // complete 4-byte UTF-8 (emoji)
    GGML_ASSERT(!is_incomplete("abc\xC3\xA9"));            // ASCII + complete 2-byte

    // Incomplete sequences — SHOULD wait
    GGML_ASSERT(is_incomplete(std::string("\xC2", 1)));              // 2-byte start, missing continuation
    GGML_ASSERT(is_incomplete(std::string("\xE2\x80", 2)));          // 3-byte start + 1 cont, missing 1
    GGML_ASSERT(is_incomplete(std::string("\xE2", 1)));              // 3-byte start, missing 2
    GGML_ASSERT(is_incomplete(std::string("\xF0\x9F\x98", 3)));      // 4-byte start + 2 cont, missing 1
    GGML_ASSERT(is_incomplete(std::string("\xF0\x9F", 2)));          // 4-byte start + 1 cont, missing 2
    GGML_ASSERT(is_incomplete(std::string("\xF0", 1)));              // 4-byte start, missing 3
    GGML_ASSERT(is_incomplete(std::string("\x80", 1)));              // orphan continuation byte

    // Mixed: ASCII followed by start of multi-byte
    GGML_ASSERT(is_incomplete(std::string("hello\xC3", 6)));        // ASCII + incomplete 2-byte
    GGML_ASSERT(!is_incomplete(std::string("hello\xC3\xA9", 7)));   // ASCII + complete 2-byte
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(void) {
    ggml_time_init();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f, 0.0f, 0.0f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

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
            false,  // don't activate immediately
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
            false,  // don't activate immediately
            2,      // forcing starts at i=2 (after accept(51) depletes budget, apply() forces)
            3);     // forcing continues through i=3 (at i=4 state becomes DONE)
    }

    // Test 3: Activate immediately with budget=0, forcing should start right away
    // Flow: Since no start token in sequence, state stays IDLE (no start/end configured means passthrough)
    // This test needs start token to be in the sequence or use activate_immediately with start token present
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 51, 52}; // start token first, then 3 tokens

        test_reasoning_budget("activate immediately budget=0", sequence, start, end, forced,
            0,      // budget of 0 tokens
            true,   // activate immediately when start token seen
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
            false,  // don't activate immediately
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
            true,   // activate immediately
            1,      // forcing starts at i=1 (after 2 accepts deplete budget)
            2);     // forcing continues through i=2
    }

    printf("OK (5 tests passed)\n");

    printf("Testing UTF-8 boundary detection... ");
    test_utf8_boundary_detection();
    printf("OK\n");

    test_perf();

    return 0;
}

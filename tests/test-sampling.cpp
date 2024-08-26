#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

static void dump(const llama_token_data_array * candidates) {
    for (size_t i = 0; i < candidates->size; i++) {
        printf("%d: %f (%f)\n", candidates->data[i].id, candidates->data[i].p, candidates->data[i].logit);
    }
}

#define DUMP(__candidates) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__candidates)); printf("-\n"); } while(0)

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & expected_probs, int k) {
    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_top_k(nullptr, &candidates_p, k, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-5);
    }
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & expected_probs, float p) {
    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_top_p(nullptr, &candidates_p, p, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_tfs(const std::vector<float> & probs, const std::vector<float> & expected_probs, float z) {
    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    DUMP(&candidates_p);
    llama_sample_tail_free(nullptr, &candidates_p, z, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & expected_probs, float p) {
    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    DUMP(&candidates_p);
    llama_sample_min_p(nullptr, &candidates_p, p, 1);
    DUMP(&candidates_p);
    llama_sample_softmax(nullptr, &candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & expected_probs, float p) {
    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    DUMP(&candidates_p);
    llama_sample_typical(nullptr, &candidates_p, p, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_repetition_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    const size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_repetition_penalties(nullptr, &candidates_p, (const llama_token *) last_tokens.data(), last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_sampler_queue(
    const size_t n_vocab, const std::string samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        const float logit = logf(token_id);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab-1;

    for (auto s : samplers_sequence) {
        switch (s){
            case 'k': llama_sample_top_k    (nullptr, &candidates_p, top_k, 1); break;
            case 'f': GGML_ABORT("tail_free test not implemented");
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': llama_sample_top_p    (nullptr, &candidates_p, top_p, 1); break;
            case 'm': llama_sample_min_p    (nullptr, &candidates_p, min_p, 1); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        llama_sample_softmax(nullptr, &candidates_p); // make sure tokens are sorted for tests

        const int size = candidates_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(candidates_p.data[0].id == max_token_id);
            GGML_ASSERT(candidates_p.data[expected_size-1].id == min_token_id);
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
            GGML_ASSERT(candidates_p.data[0].id == max_token_id);
            GGML_ASSERT(candidates_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f-min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(candidates_p.data[0].id == max_token_id);
            GGML_ASSERT(candidates_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05ld top_k=%05d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

int main(void) {
    ggml_time_init();

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 1);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/1.0f, 0.3f/1.0f, 0.2f/1.0f, 0.1f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/1.0f, 0.3f/1.0f, 0.2f/1.0f, 0.1f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.9f, 0.3f/0.9f, 0.2f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.9f, 0.3f/0.9f, 0.2f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.7f, 0.3f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.7f, 0.3f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);

    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f}, 0.25f);
    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f, 0.25f}, 0.75f);
    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f, 0.25f}, 0.99f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f}, 0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f}, {0.2f, 0.2f, 0.2f}, 0.5f);

    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0.25f, 0.25f, 0.25f, 0.25f, 0},   50.0f, 0.0f, 0.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0.5f, 0.5f, 0, 0, 0},       50.0f, 0.0f, 0.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.5f, 0.5f, 0, 0, 0}, 50.0f, 0.0f, 0.0f);

    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.249997f, 0.249997f, 0.249997f, 0.249997f, 0.000011f}, 1.0f, 5.0f, 5.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.499966f, 0.499966f, 0.000023f, 0.000023f, 0.000023f}, 1.0f, 5.0f, 5.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.499977f, 0.499977f, 0.000023f, 0.000023f, 0.000000f}, 1.0f, 5.0f, 5.0f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0002f, 1.0f);
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

    printf("OK\n");

    return 0;
}

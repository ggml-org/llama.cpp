#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <unordered_map>

// Tests that common_sampler_sample_and_accept_n_rejection preserves the target
// distribution regardless of what the draft distribution Q looks like.

static void test_probabilistic_sampling(llama_model * model, llama_context * ctx) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(n_vocab >= 3);
    std::vector<float> logits(n_vocab, -INFINITY);
    logits[0] = std::log(0.1f);
    logits[1] = std::log(0.3f);
    logits[2] = std::log(0.6f);

    common_params_sampling params;
    params.seed = 42;
    params.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    for (float temp : {0.5f, 1.0f, 2.0f}) {
        params.temp = temp;
        common_sampler_ptr sampler(common_sampler_init(model, params));
        common_sampler_sample_logits(sampler.get(), logits.data(), n_vocab);
        const auto * candidates = common_sampler_get_candidates(sampler.get(), false);
        double sum = 0.0;
        double norm = 0.0;
        for (int32_t i = 0; i < 3; ++i) {
            norm += std::exp((double) logits[i] / temp);
        }
        for (size_t i = 0; i < candidates->size; ++i) {
            const auto & token = candidates->data[i];
            const double expected = std::exp((double) logits[token.id] / temp) / norm;
            GGML_ASSERT(std::abs(token.p - expected) < 1e-6);
            sum += token.p;
        }
        GGML_ASSERT(std::abs(sum - 1.0) < 1e-6);
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    batch.add(0, 0, 0, true);
    batch.add(1, 1, 0, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

    const float target_probs[] = {0.75f, 0.2f, 0.05f};
    for (int32_t row = 0; row < 2; ++row) {
        float * target_logits = llama_get_logits_ith(ctx, row);
        std::fill(target_logits, target_logits + n_vocab, -INFINITY);
        for (int32_t i = 0; i < 3; ++i) {
            target_logits[i] = std::log(target_probs[i]);
        }
    }

    params.temp = 1.0f;
    common_sampler_ptr draft(common_sampler_init(model, params));
    params.seed = 123;
    common_sampler_ptr target(common_sampler_init(model, params));
    int counts[3] = {};
    const int n_samples = 12000;
    for (int i = 0; i < n_samples; ++i) {
        const llama_token token = common_sampler_sample_logits(draft.get(), logits.data(), n_vocab);
        const auto * candidates = common_sampler_get_candidates(draft.get(), true);
        const std::vector<std::vector<llama_token_data>> q = {
            {candidates->data, candidates->data + candidates->size}
        };
        const auto accepted = common_sampler_sample_and_accept_n_rejection(target.get(), ctx, {0, 1}, {token}, q);
        GGML_ASSERT(accepted[0] >= 0 && accepted[0] < 3);
        ++counts[accepted[0]];
    }
    for (int i = 0; i < 3; ++i) {
        GGML_ASSERT(std::abs((double) counts[i] / n_samples - target_probs[i]) < 0.025);
    }
    const std::vector<std::vector<llama_token_data>> outside_vocab = {{{n_vocab, 0.0f, 1.0f}}};
    for (int i = 0; i < 100; ++i) {
        const auto accepted = common_sampler_sample_and_accept_n_rejection(target.get(), ctx, {0, 1}, {n_vocab}, outside_vocab);
        GGML_ASSERT(accepted.size() == 1 && accepted[0] >= 0 && accepted[0] < 3);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
}

static int test_speculative_rejection_sampling(common_params & params, int n_trials) {
    params.sampling.backend_sampling = false;
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    test_probabilistic_sampling(model_tgt, ctx_tgt);

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed     = 18;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.temp     = 1.0f;

    // Initialize the sampler for the target model.
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize target sampler\n");
        return 1;
    }

    // Sample once so the sampler chain runs and populates cur_p. We don't need
    // the returned token id itself so the return value is ignored.
    common_sampler_sample(smpl.get(), ctx_tgt, -1);

    // Get the target candidates distribution in sorted order.
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl.get(), true);

    // Clone smpl before the trial loop starts. We need the dist_tgt pointer to stay valid.
    common_sampler_ptr smpl_verify(common_sampler_clone(smpl.get()));
    if (!smpl_verify) {
        LOG_ERR("failed to clone target sampler\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::unordered_map<llama_token, int> token_counts;

    std::vector<llama_token_data> q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        q[i].id = i;
        q[i].logit = 0.0f;
    }

    for (int trial = 0; trial < n_trials; ++trial) {
        float q_sum = 0.0f;

        for (auto & td : q) {
            td.p = uni(rng);
            q_sum += td.p;
        }

        const float inv_sum = 1.0f / q_sum;
        for (auto & td : q) {
            td.p *= inv_sum;
        }

        float u = uni(rng);

        // Fallback to the last bucket to absorb residual probability from FP precision loss
        llama_token draft_tok = q.back().id;
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0f) {
                draft_tok = e.id;
                break;
            }
        }

        // Both indices reuse the prompt's last output logits. The second entry
        // is the bonus draw slot; it is always stale and its result is ignored.
        const llama_tokens draft = { draft_tok };
        const std::vector<std::vector<llama_token_data>> draft_q = { q };
        const std::vector<int> idxs = { -1, -1 };

        const std::vector<llama_token> result =
            common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, false);

        token_counts[result[0]]++;
    }

    LOG("\n");
    LOG("Empirical vs target distribution after %d trials:\n", n_trials);

    bool   ok      = true;
    size_t covered = 0;
    const double log_confidence = std::log(2.0 * n_vocab / 1e-6);

    auto get_count_for = [&](llama_token id) -> int {
        const auto it = token_counts.find(id);
        return it != token_counts.end() ? it->second : 0;
    };

    // Iterate over the target distribution
    for (size_t i = 0; i < dist_tgt->size; ++i) {
        const llama_token id  = dist_tgt->data[i].id;
        const float       p_t = dist_tgt->data[i].p;

        const int   count = get_count_for(id);
        const float p_e   = (float) count / n_trials;
        const float diff  = std::fabs(p_t - p_e);

        // We only deal with ids in dist_tgt, and if each trial result was one
        // of dist_tgt's candidates then covered ends up equal to the number of
        // trials. If not this inconsistency tells us how many trials produced
        // a token outside of the target's distribution.
        covered += count;

        // Bernstein's bound also covers rare buckets; account for all vocabulary comparisons.
        const float tolerance = std::sqrt(2.0 * p_t * (1.0f - p_t) * log_confidence / n_trials) +
            2.0 * log_confidence / (3.0 * n_trials);

        LOG("token %3d : target = %f, empirical = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);

        if (diff > tolerance) {
            LOG_ERR("token %d: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                    id, p_e, p_t, diff, tolerance);
            ok = false;
        }
    }

    if (covered != (size_t) n_trials) {
        LOG_ERR("%zu of %d trials produced a token outside dist_tgt's candidates\n",
                (size_t) n_trials - covered, n_trials);
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    return 0;
}

struct parsed_args {
    int n_trials;
    std::vector<char *> argv;
};

static std::optional<parsed_args> parse_n_trials_arg(int argc, char ** argv, int default_n_trials) {
    parsed_args result;
    result.n_trials = default_n_trials;
    result.argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-trials") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --n-trials requires a number argument\n", __func__);
                return std::nullopt;
            }
            int value = std::atoi(argv[i + 1]);
            if (value <= 0) {
                LOG_ERR("%s: --n-trials requires a positive number argument\n", __func__);
                return std::nullopt;
            }
            result.n_trials = value;
            i++;
        } else {
            result.argv.push_back(argv[i]);
        }
    }
    result.argv.push_back(nullptr);

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    auto parsed = parse_n_trials_arg(argc, argv, 5000);
    if (!parsed) {
        return 1;
    }
    const int fargc = (int) parsed->argv.size() - 1;

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());

    llama_backend_init();

    const int result = test_speculative_rejection_sampling(params, parsed->n_trials);
    if (result == 0) {
        LOG("\nAll tests passed.\n");
    } else {
        LOG("\nSome tests failed.\n");
    }

    llama_backend_free();
    return result;
}

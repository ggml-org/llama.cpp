#include "speculative.h"
#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>

/**
 * Tests where the focus is verify that the speculative draft models emitted
 * tokens match the target models distribution. So when rejection sampling
 * occurs it should still produce the same probability distribution as if we
 * directly sampled from the target model.
 */

static int test_speculative_rejection_sampling(common_params & params, int n_trials) {
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    // Initialize the sampler for the target model.
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize target sampler\n");
        return 1;
    }

    // Accept all the tokens that were processes above, so that samplers that
    // require keeping track of state like repeat/frequency/presence sampler can
    // update their state so that when we run the sampler chain they know which
    // tokens have already appeared and the can suppress/boost on the next
    // sampling run.
    for (llama_token id : prompt) {
        // passing false for is_generated as these tokens were not generated via
        // sampling. Grammar should only start tracking once the model begins
        // emitting tokens that were themselves sampled under the grammars
        // constrait which is not the case here.
        common_sampler_accept(smpl.get(), id, false);
    }

    // Sample once so the sampler chain runs and populates cur_p. We don't need
    // the returned token id itself so the return value is ignored.
    common_sampler_sample(smpl.get(), ctx_tgt, -1);

    // Get the target candidates distribution in sorted order.
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl.get(), true);

    // Clone smpl before the trial loop starts. We need the dist_tgt pointer to
    // stay valid.
    common_sampler_ptr smpl_verify(common_sampler_clone(smpl.get()));
    if (!smpl_verify) {
        LOG_ERR("failed to clone target sampler\n");
        return 1;
    }

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    params.speculative.draft.backend_sampling = false;
    common_params params_dft = common_base_params_to_speculative(params);

    auto spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    auto * model_dft = spec_init->model();
    auto * ctx_dft   = spec_init->context();
    if (!model_dft || !ctx_dft) {
        LOG_ERR("failed to initialize draft model and draft context\n");
        return 1;
    }

    // set parameters for speculative decoding implementation.
    params.speculative.draft.ctx_tgt       = ctx_tgt;
    params.speculative.draft.ctx_dft       = ctx_dft;
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.n_max         = 1;
    params.speculative.draft.n_min         = 0;
    params.speculative.draft.p_min         = 0.0f;

    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));

    common_speculative_begin(spec.get(), 0, prompt);

    // Draft model processing of the batch to populate the drafts kv-cache.
    if (!common_speculative_process(spec.get(), batch)) {
        LOG_ERR("failed to process the draft prompt\n");
        return 1;
    }

    auto * mem_dft = llama_get_memory(ctx_dft);

    const llama_pos pos_last = (llama_pos) prompt.size() - 1;
    const llama_seq_id seq_id = 0;

    std::unordered_map<llama_token, int> token_counts;

    for (int trial = 0; trial < n_trials; ++trial) {
        // Remove whatever the previous trial drafted.
        llama_memory_seq_rm(mem_dft, seq_id, pos_last, -1);

        llama_tokens draft;
        // candidate distribution
        std::vector<std::vector<llama_token_data>> draft_q;

        auto & d_params   = common_speculative_get_draft_params(spec.get(), seq_id);
        d_params.drafting = true;
        d_params.n_max    = 1;
        d_params.pos0     = pos_last;
        d_params.id_last  = prompt.back();
        d_params.prompt   = &prompt;
        d_params.result   = &draft;
        d_params.result_q = &draft_q;
        d_params.temp     = params.sampling.temp;
        d_params.seed     = (uint32_t) trial;

        common_speculative_draft(spec.get());
        if (draft.size() != 1) {
            LOG_ERR("trial %d: expected 1 draft token, got %zu\n", trial, draft.size());
            return 1;
        }

        common_sampler_reset(smpl_verify.get());
        for (llama_token id : prompt) {
            common_sampler_accept(smpl_verify.get(), id, false);
        }

        // Set the target output logits to use. Both indices reuse the prompts
        // last output, but only result[0] is tested. If accepted, the proposal
        // triggers a bonus draw from stale logits, which is ignored.
        const std::vector<int> idxs = { -1, -1 };
        const std::vector<llama_token> result =
            common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, false);

        // update the count for this sampled token id.
        token_counts[result[0]]++;
    }

    LOG("\n");
    LOG("Draft vs target distribution after %d trials:\n", n_trials);

    bool   ok      = true;
    size_t covered = 0;

    auto get_count_for = [&](llama_token id) -> int {
        const auto it = token_counts.find(id);
        return it != token_counts.end() ? it->second : 0;
    };

    // Iterate over the target distribution
    for (size_t i = 0; i < dist_tgt->size; ++i) {
        const llama_token id  = dist_tgt->data[i].id;
        const float       p_t = dist_tgt->data[i].p;

        const int        count = get_count_for(id);
        const float      p_e = (float) count / n_trials;
        const float      diff = std::fabs(p_t - p_e);

        // We only deal with ids in dist_tgt, and if each trial result was one
        // of dist_tgs candidates then covered ends up equal to the number of
        // trials. If not this inconsistency tells us how many trials produced
        // a token outside of the targets distribution.
        covered += count;

        const float standard_error = std::sqrt(p_t * (1.0f - p_t) / n_trials);
        const float tolerance = 6.0f * standard_error;

        LOG("token %3d : target = %f, draft = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);

        if (diff > tolerance) {
            LOG_ERR("token %d: draft probability %f too far from target %f (diff %f > tol %f)\n",
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
            result.n_trials = std::atoi(argv[i + 1]);
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

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());
    LOG("Using draft model '%s'\n", params.speculative.draft.mparams.path.c_str());

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

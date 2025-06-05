// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

#include <thread>
#include <vector>
#include <atomic>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            common_log_add(common_log_main(), level, "%s", text);
        }
    }, NULL);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    int dev_count = ggml_backend_dev_count();
    int gpu_dev_count = 0;
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_dev_count++;
        }
    }
    const int num_models = gpu_dev_count + 1; // GPUs + 1 CPU model
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    struct model_context {
        llama_model_ptr model;
        std::vector<llama_context_ptr> contexts;
        std::vector<std::unique_ptr<common_sampler, decltype(&common_sampler_free)>> samplers;
    };

    std::vector<model_context> models;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        model_context this_model;

        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        mparams.main_gpu = m < gpu_dev_count ? m : -1;

        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == NULL) {
            LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        this_model.model.reset(model);

        for (int c = 0; c < num_contexts; ++c) {
            LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);
            llama_context * ctx = llama_init_from_model(model, cparams);
            if (ctx == NULL) {
                LOG_ERR("%s: failed to create context\n", __func__);
                return 1;
            }
            this_model.contexts.emplace_back(ctx);

            common_sampler * sampler = common_sampler_init(model, params.sampling);
            if (sampler == NULL) {
                LOG_ERR("%s: failed to create sampler\n", __func__);
                return 1;
            }
            this_model.samplers.emplace_back(sampler, common_sampler_free);

            threads.emplace_back([model, ctx, sampler, &params, &failed, m, c, num_models, num_contexts]() {
                llama_batch batch = {};
                {
                    auto prompt = common_tokenize(ctx, params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    batch = llama_batch_get_one(prompt.data(), prompt.size());
                    if (llama_decode(ctx, batch)) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token;
                    if (batch.n_tokens > 0) {
                        token = common_sampler_sample(sampler, ctx, batch.n_tokens - 1);
                    } else {
                        token = llama_vocab_bos(vocab);
                    }

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }
                    result += common_token_to_piece(ctx, token);

                    batch = llama_batch_get_one(&token, 1);
                    if (llama_decode(ctx, batch)) {
                        LOG_ERR("failed to decode\n");
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: Result: '%s'\n", m + 1, num_models, c + 1, num_contexts, result.c_str());
            });

        }

        models.emplace_back(std::move(this_model));
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads completed successfully.\n");
    return 0;
}

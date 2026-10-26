#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <limits>

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 1;
    }

    llama_backend_init();

    auto * model = llama_model_load_from_file(
        argv[1], llama_model_default_params());

    if (!model) {
        llama_backend_free();
        return 1;
    }

    for (int version = 1; version <= 2; ++version) {
        common_params_sampling params;
        params.mirostat = version;
        params.mirostat_tau = -2.0f;
        params.mirostat_eta = -0.1f;
        params.penalty_last_n = -5;

        auto * sampler = common_sampler_init(model, params);

        GGML_ASSERT(params.mirostat_tau == 5.0f);
        GGML_ASSERT(params.mirostat_eta == 0.1f);
        GGML_ASSERT(params.penalty_last_n == 0);

        common_sampler_free(sampler);

        params.mirostat_tau = std::numeric_limits<float>::quiet_NaN();
        params.mirostat_eta = std::numeric_limits<float>::infinity();

        sampler = common_sampler_init(model, params);

        GGML_ASSERT(params.mirostat_tau == 5.0f);
        GGML_ASSERT(params.mirostat_eta == 0.1f);

        common_sampler_free(sampler);

        params.mirostat_tau = 0.0f;
        params.mirostat_eta = 0.0f;
        params.penalty_last_n = -1;

        sampler = common_sampler_init(model, params);

        GGML_ASSERT(params.mirostat_tau == 0.0f);
        GGML_ASSERT(params.mirostat_eta == 0.0f);
        GGML_ASSERT(params.penalty_last_n == -1);

        common_sampler_free(sampler);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}

#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <chrono>
#include <cinttypes>
#include <thread>

using namespace std::chrono_literals;

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    const llama_params_fit_status status = llama_params_fit(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
        params.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    if (status != LLAMA_PARAMS_FIT_STATUS_SUCCESS) {
        LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
        exit(1);
    }

    LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
    common_log_flush(common_log_main());
    printf("-c %" PRIu32 " -ngl %" PRIu32, cparams.n_ctx, mparams.n_gpu_layers);

    size_t nd = llama_max_devices();
    while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
        nd--;
    }
    if (nd > 1) {
        for (size_t id = 0; id < nd; id++) {
            if (id == 0) {
                printf(" -ts ");
            }
            printf("%s%" PRIu32, id > 0 ? "," : "", uint32_t(mparams.tensor_split[id]));
        }
    }

    const size_t ntbo = llama_max_tensor_buft_overrides();
    bool any_tbo = false;
    for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
        if (itbo == 0) {
            printf(" -ot \"");
        }
        printf("%s%s=%s", itbo > 0 ? "," : "", mparams.tensor_buft_overrides[itbo].pattern, ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft));
        any_tbo = true;
    }
    printf("%s\n", any_tbo ? "\"" : "");

    return 0;
}

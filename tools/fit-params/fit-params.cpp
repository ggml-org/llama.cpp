#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <iostream>

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
    llama_params_fit(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_margin, params.fit_params_min_ctx,
        params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);

    LOG_INF("Printing fitted CLI arguments to stdout...\n");
    std::cout << "-c "    << cparams.n_ctx;
    std::cout << " -ngl " << mparams.n_gpu_layers;

    size_t nd = llama_max_devices();
    while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
        nd--;
    }
    if (nd > 1) {
        for (size_t id = 0; id < nd; id++) {
            if (id == 0) {
                std::cout << " -ts ";
            }
            if (id > 0) {
                std::cout << ",";
            }
            std::cout << mparams.tensor_split[id];
        }
    }

    const size_t ntbo = llama_max_tensor_buft_overrides();
    for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
        if (itbo == 0) {
            std::cout << " -ot ";
        }
        if (itbo > 0) {
            std::cout << ",";
        }
        std::cout << mparams.tensor_buft_overrides[itbo].pattern << "=" << ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft);
    }
    std::cout << "\n";

    return 0;
}

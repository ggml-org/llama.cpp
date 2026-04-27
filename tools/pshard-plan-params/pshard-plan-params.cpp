#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <cinttypes>

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

    params.tensor_buft_overrides.resize(4096);

    const uint32_t tier_max_auto = std::min(std::max(cparams.n_batch, (uint32_t) 16384), cparams.n_ctx);
    const uint32_t tier_max      = std::min(params.pshard_tier_max > 0 ? params.pshard_tier_max : tier_max_auto, cparams.n_ctx);

    mparams.pshard_registry        = llama_pshard_registry_create(tier_max, cparams.n_seq_max);
    mparams.pshard_cache_skip_load = true;

    LOG_INF("%s: planning pshard tensor overrides...\n", __func__);
    llama_params_fit_pshard(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_buft_overrides.data(), params.max_vram_alloc);

    LOG_INF("%s: planning complete, registry written next to model file\n", __func__);
    llama_pshard_registry_free(mparams.pshard_registry);

    return 0;
}

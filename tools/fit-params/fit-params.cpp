#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (!params.fit_params_print) {
        // #30424 — a speculative decoding config changes what has to fit: the draft model is
        // measured alongside the target, and an MTP context runs on the weights of the main
        // model. Fit them together, same as the server does for its own config estimate.
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

        common_params_fit_status status;
        if (has_draft || spec_mtp) {
            common_params params_dft = common_base_params_to_speculative(params);

            auto mparams_dft = common_model_params_to_llama(params_dft);
            auto cparams_dft = common_context_params_to_llama(params_dft);
            if (spec_mtp) {
                cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }
            cparams_dft.n_rs_seq = 0;

            const common_fit_extra_model extra = {
                /*.path_model   =*/ params_dft.model.path.c_str(),
                /*.mparams      =*/ &mparams_dft,
                /*.cparams      =*/ &cparams_dft,
                /*.shares_model =*/ !has_draft, // an MTP context runs on the weights of the main model
            };

            status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                    params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                    &extra,
                    params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        } else {
            status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                    params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                    nullptr,
                    params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        }
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("-c %" PRIu32 " -ngl %" PRIi32, cparams.n_ctx, mparams.n_gpu_layers);

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

        // #30424 — echo the speculative config so the printed line is a complete server config
        if (has_draft && !params.speculative.draft.mparams.path.empty()) {
            const std::string & draft_path = params.speculative.draft.mparams.path;
            if (draft_path.find(' ') != std::string::npos) {
                printf(" -md \"%s\"", draft_path.c_str());
            } else {
                printf(" -md %s", draft_path.c_str());
            }
            if (params.speculative.draft.n_gpu_layers >= 0) {
                printf(" -ngld %d", params.speculative.draft.n_gpu_layers);
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
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.path.c_str(), &mparams, &cparams);
    }

    return 0;
}

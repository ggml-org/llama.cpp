#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#if defined(_MSC_VER)
#    pragma warning(disable : 4244 4267)  // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;
    struct ggml_opt_optimizer_params & optimizer_params = params.optimize;

    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n",
                __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    // load the model and apply lora adapter, if any
    common_init_result  llama_init = common_init_from_params(params);
    llama_model_ptr &   model      = llama_init.model;
    llama_context_ptr & ctx        = llama_init.context;
    auto pctx = ctx.get();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<llama_token> tokens  = common_tokenize(pctx, params.prompt, true);
    ggml_opt_dataset_t       dataset = common_opt_dataset_init(pctx, tokens, llama_n_ctx(pctx) / 2);

    float alpha                = optimizer_params.adamw.alpha;
    optimizer_params.sgd.alpha = alpha;
    float wd                   = optimizer_params.adamw.wd;
    optimizer_params.sgd.wd    = wd;
    LOG_INF("-optimizer %s -lr %.2g -wd %.2g -epochs %d -val %.2g\n", ggml_opt_optimizer_name(params.optimizer),
            (double) alpha, (double) wd, params.epochs, (double) params.val_split);

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/&optimizer_params,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(pctx, model.get(), lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (unsigned epoch = 0; epoch < params.epochs; ++epoch) {
        llama_opt_epoch(pctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    llama_model_save_to_file(model.get(), "finetuned-model.gguf");

    llama_backend_free();

    return 0;
}

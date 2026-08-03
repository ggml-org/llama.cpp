// tessera-train-lk.cpp - native LK (acceptance-rate) drafter training driver.
//
// Trains an autoregressive speculative-decoding drafter to directly maximize
// the verifier acceptance rate, using GGML_OPT_LOSS_TYPE_LK. Reads
// llama.tessera.spec.v1 traces (llama-imatrix --telemetry-out --telemetry-topk),
// builds the dense-label dataset the llama-layer LK path already expects, runs
// the epoch loop, and saves the trained drafter GGUF. See
// docs/tessera-lk-training-design.md for the design.
//
// Mimics examples/training/finetune.cpp; the only structural differences are
// the data loader (traces -> [prime, draft...] tokens + dense verifier labels)
// and loss_type = LK.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "tessera-lk-train-data.h"

#include "ggml-opt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    printf("usage: %s -m drafter.gguf --traces spec.jsonl -o trained.gguf [options]\n", prog);
    printf("\n");
    printf("LK drafter training: minimize the total-variation distance between the\n");
    printf("drafter and verifier distributions, i.e. maximize the acceptance rate.\n");
    printf("\n");
    printf("Tessera options:\n");
    printf("  --traces PATH        llama.tessera.spec.v1 JSONL (from llama-imatrix --telemetry-out)\n");
    printf("  --block-size B       drafted tokens per step; n_ctx = B+1 (default: auto-detect modal)\n");
    printf("  --max-examples N     dataset cap, bounds dense-label memory (default 512)\n");
    printf("  --dry-run            build the dataset and print stats; do not train or save\n");
    printf("\n");
    printf("Standard training options (shared with llama-finetune):\n");
    printf("  -m, --model PATH     drafter model to train (weights are modified in place)\n");
    printf("  -o, --out-file PATH  where to save the trained model\n");
    printf("  -epochs, --epochs N        number of epochs\n");
    printf("  -lr, --learning-rate F     adamw | sgd optimizer alpha\n");
    printf("  -opt, --optimizer NAME     adamw | sgd (default adamw)\n");
    printf("  -val-split, --val-split F  validation fraction\n");
    printf("  -ngl, --n-gpu-layers N     layers on the GPU\n");
    printf("\n");
    printf("The context size is forced to block_size+1 (one spec step per datapoint).\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ---- pull tessera-specific flags out of argv; pass the rest to common ----
    std::string traces_path;
    int  block_size   = 0;     // 0 -> auto-detect modal drafted count
    int  max_examples = 512;
    bool dry_run      = false;

    std::vector<std::string> pass;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--traces"       && i + 1 < argc) { traces_path  = argv[++i]; }
        else if (a == "--block-size"   && i + 1 < argc) { block_size   = std::atoi(argv[++i]); }
        else if (a == "--max-examples" && i + 1 < argc) { max_examples = std::atoi(argv[++i]); }
        else if (a == "--dry-run")                       { dry_run      = true; }
        else { pass.push_back(a); }
    }

    if (traces_path.empty()) {
        print_usage(argv[0]);
        LOG_ERR("--traces is required\n");
        return 1;
    }
    if (max_examples <= 0) {
        LOG_ERR("--max-examples must be > 0\n");
        return 1;
    }

    std::vector<char *> cargv;
    cargv.push_back(argv[0]);
    for (auto & s : pass) {
        cargv.push_back(const_cast<char *>(s.c_str()));
    }
    const int cargc = (int) cargv.size();

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(cargc, cargv.data(), params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // ---- resolve block size (auto-detect the modal drafted count) ----
    if (block_size <= 0) {
        block_size = ts_lk_train_detect_block_size(traces_path.c_str());
        if (block_size <= 0) {
            LOG_ERR("could not auto-detect block size from %s (no llama.tessera.spec.v1 records?); pass --block-size\n",
                    traces_path.c_str());
            return 1;
        }
        LOG_INF("auto-detected block_size = %d (modal drafted count)\n", block_size);
    }
    const int n_ctx_dp = block_size + 1;

    // ---- force the training context to one spec step per datapoint ----
    // Mirrors finetune: writable weights + f32 KV cache (OUT_PROD has no f16)
    // and flash attention disabled (FLASH_ATTN_EXT has no backward pass, so the
    // training graph must use the differentiable non-flash attention path).
    params.load_mode        = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k     = GGML_TYPE_F32;
    params.cache_type_v     = GGML_TYPE_F32;
    params.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.n_ctx            = n_ctx_dp;
    params.n_batch          = n_ctx_dp;
    params.n_ubatch         = n_ctx_dp;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL || ctx == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    LOG_INF("drafter: n_vocab = %d, n_ctx = %d (block_size = %d)\n", n_vocab, n_ctx_dp, block_size);

    // ---- collect usable trace records (capped), then build the dataset ----
    std::vector<std::string> usable;
    {
        std::ifstream fin(traces_path);
        if (!fin) {
            LOG_ERR("cannot open traces: %s\n", traces_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(fin, line) && (int) usable.size() < max_examples) {
            if (!line.empty() && ts_lk_train_line_usable(line.c_str(), block_size) == 1) {
                usable.push_back(line);
            }
        }
    }

    const int ndata = (int) usable.size();
    if (ndata == 0) {
        LOG_ERR("no usable llama.tessera.spec.v1 records with drafted == %d in %s\n",
                block_size, traces_path.c_str());
        return 1;
    }

    const int64_t ne_label  = (int64_t) n_ctx_dp * n_vocab;
    const double  label_mib = (double) ndata * ne_label * sizeof(float) / (1024.0 * 1024.0);
    LOG_INF("dataset: %d examples, dense-label memory ~%.1f MiB\n", ndata, label_mib);

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_F32, n_ctx_dp, ne_label, ndata, /*ndata_shard =*/ 1);

    llama_token * data_ptr   = (llama_token *) ggml_opt_dataset_data(dataset)->data;
    float       * labels_ptr = (float *)       ggml_opt_dataset_labels(dataset)->data;

    for (int idata = 0; idata < ndata; ++idata) {
        const int rc = ts_lk_train_example_from_line(
                usable[idata].c_str(), block_size, n_vocab,
                data_ptr   + (size_t) idata * n_ctx_dp,
                labels_ptr + (size_t) idata * ne_label);
        if (rc != 1) {
            LOG_ERR("failed to build training example %d (rc = %d)\n", idata, rc);
            return 1;
        }
    }
    LOG_INF("built %d training examples\n", ndata);
    usable.clear();
    usable.shrink_to_fit();

    if (dry_run) {
        LOG_INF("dry-run: dataset built OK; skipping training and save\n");
        ggml_opt_dataset_free(dataset);
        llama_backend_free();
        return 0;
    }

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    // Pin the training context to one spec step. The context capacity is padded
    // up to a multiple of 256, but each datapoint is exactly block_size+1 tokens
    // and must be KV-independent; n_ctx_train tells the epoch loop to process
    // exactly that many positions per datapoint.
    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/(uint32_t) n_ctx_dp,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
        /*loss_type       =*/GGML_OPT_LOSS_TYPE_LK,
        /*use_weighted_ce =*/false,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        double loss_train = 0.0, acc_train = 0.0;
        ggml_opt_result_loss(result_train, &loss_train, nullptr);
        ggml_opt_result_accuracy(result_train, &acc_train, nullptr);
        LOG_INF("epoch %u: train LK loss %.6f, top-1 agreement %.4f\n",
                lr.epoch, loss_train, acc_train);

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);

    if (params.out_file.empty()) {
        LOG_ERR("no --out-file specified; trained model not saved\n");
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
        LOG_INF("saved trained drafter to %s\n", params.out_file.c_str());
    }

    llama_backend_free();

    return 0;
}

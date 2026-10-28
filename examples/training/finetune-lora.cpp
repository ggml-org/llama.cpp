#include "arg.h"
#include "common.h"
#include "finetune.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static uint32_t parse_lora_modules(const std::string & modules_str) {
    static const std::map<std::string, uint32_t> module_map = {
        {"attn_q",   LLAMA_ADAPTER_LORA_TARGET_ATTN_Q},
        {"attn_k",   LLAMA_ADAPTER_LORA_TARGET_ATTN_K},
        {"attn_v",   LLAMA_ADAPTER_LORA_TARGET_ATTN_V},
        {"attn_o",   LLAMA_ADAPTER_LORA_TARGET_ATTN_O},
        {"ffn_gate", LLAMA_ADAPTER_LORA_TARGET_FFN_GATE},
        {"ffn_up",   LLAMA_ADAPTER_LORA_TARGET_FFN_UP},
        {"ffn_down", LLAMA_ADAPTER_LORA_TARGET_FFN_DOWN},
        {"output",   LLAMA_ADAPTER_LORA_TARGET_OUTPUT},
        {"all",      LLAMA_ADAPTER_LORA_TARGET_ALL},
    };

    uint32_t target_modules = 0;
    std::stringstream ss(modules_str);
    std::string module;

    while (std::getline(ss, module, ',')) {
        module.erase(0, module.find_first_not_of(" \t"));
        module.erase(module.find_last_not_of(" \t") + 1);

        auto it = module_map.find(module);
        if (it == module_map.end()) {
            LOG_ERR("Unknown LoRA target module: %s\n", module.c_str());
            LOG_ERR("Available modules: attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, output, all\n");
            return 0;
        }
        target_modules |= it->second;
    }

    return target_modules;
}

struct checkpoint_metadata {
    int32_t  epoch;
    int32_t  lora_rank;
    float    lora_alpha;
    uint32_t target_modules;
};

static std::string get_checkpoint_filename(const std::string & checkpoint_dir, int64_t step) {
    std::ostringstream oss;
    oss << checkpoint_dir << "/checkpoint_step_" << std::setfill('0') << std::setw(8) << step;
    return oss.str();
}

static std::string find_latest_checkpoint(const std::string & checkpoint_dir) {
    if (!std::filesystem::exists(checkpoint_dir)) {
        return "";
    }

    std::string latest_checkpoint;
    int64_t latest_step = -1;

    for (const auto & entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string dirname = entry.path().filename().string();
        if (dirname.find("checkpoint_step_") != 0 || dirname.size() < 16) {
            continue;
        }
        try {
            const int64_t step = std::stoll(dirname.substr(16, 8));
            if (step > latest_step) {
                latest_step = step;
                latest_checkpoint = entry.path().string();
            }
        } catch (const std::exception &) {
            continue;
        }
    }

    return latest_checkpoint;
}

static bool save_checkpoint(llama_context * ctx, llama_adapter_lora * adapter, const checkpoint_metadata & metadata, const std::string & checkpoint_dir) {
    if (!std::filesystem::exists(checkpoint_dir)) {
        if (!std::filesystem::create_directories(checkpoint_dir)) {
            LOG_ERR("Failed to create checkpoint directory: %s\n", checkpoint_dir.c_str());
            return false;
        }
    }

    if (!llama_adapter_lora_save_checkpoint(adapter, ctx, checkpoint_dir.c_str())) {
        LOG_ERR("Failed to save LoRA checkpoint\n");
        return false;
    }

    const std::string meta_path = checkpoint_dir + "/metadata.txt";
    std::ofstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        LOG_ERR("Failed to save checkpoint metadata\n");
        return false;
    }
    meta_file << "epoch="          << metadata.epoch          << "\n";
    meta_file << "lora_rank="      << metadata.lora_rank      << "\n";
    meta_file << "lora_alpha="     << metadata.lora_alpha     << "\n";
    meta_file << "target_modules=" << metadata.target_modules << "\n";
    meta_file.close();

    LOG_INF("Checkpoint saved to %s\n", checkpoint_dir.c_str());
    return true;
}

static bool load_checkpoint_metadata(const std::string & checkpoint_dir, checkpoint_metadata & metadata) {
    if (!std::filesystem::exists(checkpoint_dir)) {
        LOG_ERR("Checkpoint directory does not exist: %s\n", checkpoint_dir.c_str());
        return false;
    }

    LOG_INF("Loading checkpoint from: %s\n", checkpoint_dir.c_str());

    const std::string meta_path = checkpoint_dir + "/metadata.txt";
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        LOG_ERR("Failed to open checkpoint metadata file: %s\n", meta_path.c_str());
        return false;
    }

    try {
        std::string line;
        while (std::getline(meta_file, line)) {
            const size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                continue;
            }
            const std::string key   = line.substr(0, eq_pos);
            const std::string value = line.substr(eq_pos + 1);

            if (key == "epoch") {
                metadata.epoch = std::stoi(value);
            } else if (key == "lora_rank") {
                metadata.lora_rank = std::stoi(value);
            } else if (key == "lora_alpha") {
                metadata.lora_alpha = std::stof(value);
            } else if (key == "target_modules") {
                metadata.target_modules = std::stoul(value);
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("Invalid checkpoint metadata in %s: %s\n", meta_path.c_str(), e.what());
        return false;
    }

    return true;
}

struct checkpoint_callback_data {
    llama_context      * ctx;
    llama_adapter_lora * adapter;
    int32_t              checkpoint_save_steps;
    std::string          checkpoint_save_dir;
    int64_t              global_step;
    int32_t              current_epoch;
    int32_t              lora_rank;
    float                lora_alpha;
    uint32_t             target_modules;
};

static checkpoint_callback_data * g_checkpoint_data = nullptr;

static void checkpoint_progress_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (!train) {
        return;
    }

    checkpoint_callback_data * cb_data = g_checkpoint_data;

    cb_data->global_step++;

    if (cb_data->global_step % cb_data->checkpoint_save_steps == 0) {
        const checkpoint_metadata meta = {
            /*epoch          =*/ cb_data->current_epoch,
            /*lora_rank      =*/ cb_data->lora_rank,
            /*lora_alpha     =*/ cb_data->lora_alpha,
            /*target_modules =*/ cb_data->target_modules,
        };

        const std::string checkpoint_path = get_checkpoint_filename(cb_data->checkpoint_save_dir, cb_data->global_step);

        if (!save_checkpoint(cb_data->ctx, cb_data->adapter, meta, checkpoint_path)) {
            LOG_ERR("Failed to save checkpoint at step %" PRId64 "\n", cb_data->global_step);
        }
    }
}

// learning rate per optimizer step: a linear warmup to lr0 over the first n_warmup steps, then
// the lr_opt value of the epoch, or a cosine or linear decay from lr0 to lr_min over the rest of the run
struct lr_schedule {
    const lr_opt * lr;
    std::string    type;     // constant, cosine or linear
    int64_t        step;     // optimizer steps done
    int64_t        n_steps;  // optimizer steps of the whole run
    int64_t        n_warmup; // warmup steps
    float          lr_epoch; // lr_opt value of the current epoch
};

static ggml_opt_optimizer_params lr_schedule_opt_pars(void * userdata) {
    lr_schedule * s = (lr_schedule *) userdata;

    float lr = s->lr_epoch;
    if (s->step < s->n_warmup) {
        lr = s->lr->lr0 * (float) (s->step + 1) / (float) s->n_warmup;
    } else if (s->type != "constant") {
        const float   lr_min   = std::max(s->lr->lr_min, 0.0f);
        const int64_t n_decay  = std::max<int64_t>(s->n_steps - s->n_warmup, 1);
        const float   progress = std::min((float) (s->step - s->n_warmup) / (float) n_decay, 1.0f);
        if (s->type == "cosine") {
            lr = lr_min + 0.5f*(s->lr->lr0 - lr_min)*(1.0f + std::cos(3.14159265358979323846f*progress));
        } else {
            lr = s->lr->lr0 + (lr_min - s->lr->lr0)*progress;
        }
    }
    s->step++;

    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    result.adamw.alpha = result.sgd.alpha = lr;
    result.adamw.wd    = result.sgd.wd    = s->lr->wd;
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    const uint32_t target_modules = parse_lora_modules(params.lora_modules);
    if (target_modules == 0) {
        return 1;
    }

    if (params.auto_resume && params.resume_from.empty()) {
        params.resume_from = find_latest_checkpoint(params.checkpoint_save_dir);
        if (!params.resume_from.empty()) {
            LOG_INF("Auto-resume: found checkpoint %s\n", params.resume_from.c_str());
        }
    }

    // a checkpoint directory holds the adapter (model.gguf) and the optimizer state (optimizer.gguf)
    if (!params.resume_from.empty()) {
        params.warmup = false;

        const std::filesystem::path checkpoint_dir(params.resume_from);
        const std::string adapter_path = (checkpoint_dir / "model.gguf").string();

        LOG_INF("Loading checkpoint LoRA adapter: %s\n", adapter_path.c_str());
        params.lora_adapters.clear();
        params.lora_adapters.push_back({ adapter_path, 1.0f, "", "", nullptr });
    }

    if (params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n", __func__);
        params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? LLAMA_LOAD_MODE_MLOCK
                                                                          : LLAMA_LOAD_MODE_NONE;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    params.training = true;

    // load the model and apply the lora adapter, if any
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const bool has_existing_lora = !params.lora_adapters.empty();
    llama_adapter_lora * adapter = nullptr;

    if (has_existing_lora) {
        LOG_INF("Finetuning existing LoRA adapter %s\n", params.lora_adapters[0].path.c_str());
        adapter = params.lora_adapters[0].ptr;
        if (!adapter) {
            LOG_ERR("Existing LoRA adapter is null\n");
            return 1;
        }
    } else {
        const llama_adapter_lora_train_params lora_params = {
            /*target_modules =*/ target_modules,
            /*rank           =*/ params.lora_rank,
            /*alpha          =*/ params.lora_alpha,
            /*init_std       =*/ 0.02f,
            /*seed           =*/ params.lora_seed,
        };

        LOG_INF("LoRA configuration: rank=%d, alpha=%.1f (scaling=%.3f), modules=%s, seed=%u\n",
                lora_params.rank, lora_params.alpha, lora_params.alpha / lora_params.rank,
                params.lora_modules.c_str(), lora_params.seed);

        adapter = llama_adapter_lora_init_train(model, &lora_params);
        if (!adapter) {
            LOG_ERR("%s: LoRA training initialization failed\n", __func__);
            return 1;
        }

        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) {
            LOG_ERR("%s: failed to apply the LoRA adapter\n", __func__);
            return 1;
        }
    }

    ggml_opt_dataset_t dataset;

    if (params.assistant_loss_only) {
        LOG_INF("Using JSON dataset with chat template and assistant-only loss\n");
        dataset = common_opt_sft_dataset_init(ctx, params.prompt, params.chat_template);
    } else {
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
        LOG_INF("Using standard next-token prediction mode\n");
        dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);
    }

    if (dataset == nullptr) {
        LOG_ERR("Failed to create dataset. Please check your input file and parameters.\n");
        return 1;
    }

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);
    const int64_t batches_per_epoch = idata_split;

    if (batches_per_epoch <= 0) {
        LOG_ERR("Training split is empty. Adjust --val-split or dataset size.\n");
        return 1;
    }

    lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -lr-scheduler %s -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, params.lr_schedule.c_str(), (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    const int64_t n_steps  = (int64_t) lr.epochs * batches_per_epoch;
    const int64_t n_warmup = std::min<int64_t>(params.lr_warmup_steps > 0 ? params.lr_warmup_steps : params.lr_warmup_ratio * n_steps, n_steps);
    if (n_warmup > 0) {
        LOG_INF("learning rate warmup over %" PRId64 " of %" PRId64 " steps\n", n_warmup, n_steps);
    }

    lr_schedule schedule = {
        /*lr       =*/ &lr,
        /*type     =*/ params.lr_schedule,
        /*step     =*/ 0,
        /*n_steps  =*/ n_steps,
        /*n_warmup =*/ n_warmup,
        /*lr_epoch =*/ lr.lr0,
    };

    bool checkpoint_loaded = false;
    std::string optimizer_path;

    if (!params.resume_from.empty()) {
        checkpoint_metadata meta = {};
        if (!load_checkpoint_metadata(params.resume_from, meta)) {
            return 1;
        }
        if (meta.lora_rank != params.lora_rank) {
            LOG_ERR("Checkpoint LoRA rank (%d) doesn't match current rank (%d)\n", meta.lora_rank, params.lora_rank);
            return 1;
        }
        if (meta.lora_alpha != params.lora_alpha) {
            LOG_ERR("Checkpoint LoRA alpha (%.3f) doesn't match current alpha (%.3f)\n", meta.lora_alpha, params.lora_alpha);
            return 1;
        }
        if (meta.target_modules != target_modules) {
            LOG_ERR("Checkpoint target_modules doesn't match current target_modules\n");
            return 1;
        }
        checkpoint_loaded = true;
        optimizer_path = (std::filesystem::path(params.resume_from) / "optimizer.gguf").string();
    }

    llama_opt_params lopt_params = llama_opt_default_params();
    lopt_params.param_filter         = llama_opt_param_filter_lora;
    lopt_params.get_opt_pars         = lr_schedule_opt_pars;
    lopt_params.get_opt_pars_ud      = &schedule;
    lopt_params.optimizer_type       = params.optimizer;
    lopt_params.checkpoint_path      = checkpoint_loaded ? optimizer_path.c_str() : nullptr;
    lopt_params.load_optimizer_state = checkpoint_loaded;
    lopt_params.assistant_loss_only  = params.assistant_loss_only;

    llama_opt_init(ctx, model, lopt_params);

    // the number of training steps already done, continue from the matching epoch and batch
    int64_t start_step = 0;
    if (checkpoint_loaded) {
        start_step = llama_opt_get_iter(ctx) - 1; // the optimizer iteration is 1-based
        LOG_INF("Resuming training from global step %" PRId64 "\n", start_step);
    }
    const unsigned start_epoch = start_step / batches_per_epoch;
    schedule.step = start_step;

    checkpoint_callback_data cb_data = {
        /*ctx                   =*/ ctx,
        /*adapter               =*/ adapter,
        /*checkpoint_save_steps =*/ params.checkpoint_save_steps,
        /*checkpoint_save_dir   =*/ params.checkpoint_save_dir,
        /*global_step           =*/ start_step,
        /*current_epoch         =*/ (int32_t) start_epoch,
        /*lora_rank             =*/ params.lora_rank,
        /*lora_alpha            =*/ params.lora_alpha,
        /*target_modules        =*/ target_modules,
    };
    g_checkpoint_data = &cb_data;

    const ggml_opt_epoch_callback callback = params.checkpoint_save_steps > 0 ?
        checkpoint_progress_callback : ggml_opt_epoch_callback_progress_bar;

    if (params.checkpoint_save_steps > 0) {
        LOG_INF("Checkpointing enabled, saving every %d steps to %s\n", params.checkpoint_save_steps, params.checkpoint_save_dir.c_str());
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = start_epoch; lr.epoch < lr.epochs; ++lr.epoch) {
        cb_data.current_epoch = lr.epoch;
        schedule.lr_epoch     = lr.get_lr(lr.epoch);

        const int64_t resume_batch = lr.epoch == start_epoch ? start_step % batches_per_epoch : 0;
        if (resume_batch > 0) {
            LOG_INF("Resuming epoch %u at batch %" PRId64 "\n", lr.epoch, resume_batch);
        }

        llama_opt_epoch_resume(ctx, dataset, result_train, result_eval, idata_split,
                               callback, ggml_opt_epoch_callback_progress_bar, resume_batch);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    g_checkpoint_data = nullptr;
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    std::string adapter_filename = params.out_file;
    if (adapter_filename.empty()) {
        adapter_filename = has_existing_lora ? "finetuned-lora-adapter.gguf" : "trained-lora-adapter.gguf";
    }

    if (llama_adapter_lora_save(adapter, adapter_filename.c_str())) {
        LOG_INF("LoRA adapter saved: %s\n", adapter_filename.c_str());
    } else {
        LOG_ERR("Failed to save LoRA adapter: %s\n", adapter_filename.c_str());
    }

    llama_backend_free();

    return 0;
}

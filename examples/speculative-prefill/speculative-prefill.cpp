#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-prefill.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE 128

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    common_params_model dft_model = params.speculative.prefill.model;
    if (dft_model.empty()) {
        dft_model = params.speculative.draft.mparams;
    }

    if (dft_model.empty()) {
        LOG_ERR("%s: draft model is required for speculative prefill (specify with -mpd, --spec-prefill-model, or -md)\n", __func__);
        return 1;
    }

    // force speculative prefill enabled
    params.speculative.prefill.enabled = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    // load target model
    LOG_INF("%s: loading target model...\n", __func__);
    auto init_tgt = common_init_from_params(params);
    if (!init_tgt) {
        LOG_ERR("%s: failed to load target model\n", __func__);
        return 1;
    }

    llama_model * model_tgt = init_tgt->model();
    llama_context * ctx_tgt = init_tgt->context();

    if (llama_model_is_recurrent(model_tgt)) {
        LOG_ERR("%s: speculative prefill is not supported for recurrent models\n", __func__);
        return 1;
    }

    // load draft model with standard attention to allow attention extraction
    LOG_INF("%s: loading draft model...\n", __func__);
    common_params params_dft = common_base_params_to_speculative(params);
    params_dft.model = dft_model;
    if (params.speculative.prefill.n_ctx > 0) {
        params_dft.n_ctx = params.speculative.prefill.n_ctx;
    }
    if (params.speculative.prefill.n_gpu_layers != -1) {
        params_dft.n_gpu_layers = params.speculative.prefill.n_gpu_layers;
    }
    if (!params.speculative.prefill.devices.empty()) {
        params_dft.devices = params.speculative.prefill.devices;
    } else if (!params.speculative.draft.devices.empty()) {
        params_dft.devices = params.speculative.draft.devices;
    }
    params_dft.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // standard attention needed to capture kq_soft_max weights

    auto init_dft = common_init_from_params(params_dft, /*model_only=*/true);
    if (!init_dft) {
        LOG_ERR("%s: failed to load draft model\n", __func__);
        return 1;
    }

    llama_model * model_dft = init_dft->model();
    if (!model_dft) {
        LOG_ERR("%s: failed to load draft model\n", __func__);
        return 1;
    }

    if (llama_model_is_recurrent(model_dft)) {
        LOG_ERR("%s: draft model is recurrent and not supported for speculative prefill\n", __func__);
        return 1;
    }

    if (llama_model_target_layer_ids_n(model_dft) > 0) {
        LOG_ERR("%s: draft model '%s' is target-dependent and cannot be used for speculative prefill\n", __func__, dft_model.path.c_str());
        return 1;
    }

    if (params.speculative.prefill.n_ctx <= 0 && params_dft.n_ctx > (int32_t) llama_model_n_ctx_train(model_dft)) {
        params_dft.n_ctx = llama_model_n_ctx_train(model_dft);
        LOG_INF("%s: capping speculative prefill draft context to training limit (%d tokens)\n", __func__, params_dft.n_ctx);
    }

    llama_context_params cparams_dft = common_context_params_to_llama(params_dft);
    llama_context_ptr ctx_dft_own(llama_init_from_model(model_dft, cparams_dft));
    llama_context * ctx_dft = ctx_dft_own.get();
    if (!ctx_dft) {
        LOG_ERR("%s: failed to create draft context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_ERR("%s: draft model bos tokens must match target model. add: %d - %d, id: %d - %d\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft ? n_vocab_tgt - n_vocab_dft : n_vocab_dft - n_vocab_tgt;
        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    __func__, n_vocab_tgt, n_vocab_dft, vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }
    }

    // tokenize prompt
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx_tgt, params.prompt, true, true);
    const int32_t n_prompt = (int32_t) prompt_tokens.size();

    if (n_prompt == 0) {
        LOG_ERR("%s: empty prompt provided\n", __func__);
        return 1;
    }

    LOG_INF("%s: prompt tokens = %d\n", __func__, n_prompt);
    LOG_INF("%s: speculative prefill config: percentage = %.2f, chunk_size = %d, lookahead = %d, pool_kernel = %d\n",
            __func__,
            (double) params.speculative.prefill.percentage,
            params.speculative.prefill.chunk_size,
            params.speculative.prefill.look_ahead_cnt,
            params.speculative.prefill.pool_kernel_size);

    llama_seq_id seq_id = 0;

    // initialize draft sampler for lookahead steps
    common_params_sampling sparams_dft = params.sampling;
    sparams_dft.temp = 0.0f;
    common_sampler_ptr smpl_dft(common_sampler_init(model_dft, sparams_dft));

    // step 1: run speculative prefill on draft model
    const auto t_spec_prefill_start = ggml_time_us();

    common_speculative_prefill_result spec_res = common_speculative_prefill_execute(
        ctx_dft,
        smpl_dft.get(),
        prompt_tokens,
        seq_id,
        params.speculative.prefill);

    LOG_INF("%s: speculative prefill kept %d / %d tokens (%.1f%%)\n",
            __func__,
            spec_res.n_prompt_kept,
            spec_res.n_prompt_orig,
            100.0f * (float) spec_res.n_prompt_kept / (float) spec_res.n_prompt_orig);
    LOG_INF("%s: draft eval time: %.2f ms, importance estimation time: %.2f ms\n",
            __func__,
            spec_res.t_draft_eval_us / 1000.0f,
            spec_res.t_estimate_us / 1000.0f);

    // step 2: sparse target model prefill
    const auto t_tgt_prefill_start = ggml_time_us();

    const int32_t n_batch_tgt = llama_n_batch(ctx_tgt);
    const int32_t n_kept_total = (int32_t) spec_res.kept_indices.size();
    llama_batch batch_tgt = llama_batch_init(std::min(n_kept_total, n_batch_tgt), 0, 1);

    int ret = 0;
    for (int32_t i = 0; i < n_kept_total; i += n_batch_tgt) {
        const int32_t n_eval = std::min(n_kept_total - i, n_batch_tgt);
        common_batch_clear(batch_tgt);

        for (int32_t j = 0; j < n_eval; ++j) {
            const int32_t k = i + j;
            const int32_t orig_idx = spec_res.kept_indices[k];
            const bool is_last = (k == n_kept_total - 1);
            common_batch_add(batch_tgt, prompt_tokens[orig_idx], (llama_pos) k, { seq_id }, is_last);
        }

        ret = llama_decode(ctx_tgt, batch_tgt);
        if (ret != 0) {
            LOG_ERR("%s: failed to decode sparse prompt on target model, ret = %d\n", __func__, ret);
            llama_batch_free(batch_tgt);
            return 1;
        }
    }
    llama_batch_free(batch_tgt);

    llama_synchronize(ctx_tgt);
    llama_synchronize(ctx_dft);

    const auto t_tgt_prefill_end = ggml_time_us();
    const double ttft_ms = (t_tgt_prefill_end - t_spec_prefill_start) / 1000.0;
    const double tgt_prefill_ms = (t_tgt_prefill_end - t_tgt_prefill_start) / 1000.0;

    LOG_INF("%s: sparse target prefill time = %.2f ms\n", __func__, tgt_prefill_ms);
    LOG_INF("%s: total Time-To-First-Token (TTFT) = %.2f ms (%.2f effective prompt tokens/s)\n",
            __func__, ttft_ms, ttft_ms > 0.0 ? (1000.0 * (double) n_prompt / ttft_ms) : 0.0);

    // step 3: autoregressive generation
    common_sampler_ptr smpl_tgt(common_sampler_init(model_tgt, params.sampling));

    LOG("\n--- Generation Start ---\n");

    int32_t n_predict = params.n_predict > 0 ? params.n_predict : 32;
    int32_t cur_pos = n_kept_total;
    int32_t n_generated = 0;

    llama_batch batch_gen = llama_batch_init(1, 0, 1);

    const auto t_gen_start = ggml_time_us();

    for (int32_t i = 0; i < n_predict; ++i) {
        const llama_token token_id = common_sampler_sample(smpl_tgt.get(), ctx_tgt, -1);
        common_sampler_accept(smpl_tgt.get(), token_id, true);

        if (llama_vocab_is_eog(vocab_tgt, token_id)) {
            break;
        }

        const std::string piece = common_token_to_piece(ctx_tgt, token_id);
        LOG("%s", piece.c_str());
        fflush(stdout);

        common_batch_clear(batch_gen);
        common_batch_add(batch_gen, token_id, (llama_pos) cur_pos++, { seq_id }, true);

        ret = llama_decode(ctx_tgt, batch_gen);
        if (ret != 0) {
            LOG_ERR("%s: failed to decode generated token %d, ret = %d\n", __func__, i, ret);
            break;
        }

        n_generated++;
    }

    const auto t_gen_end = ggml_time_us();
    const double gen_ms = (t_gen_end - t_gen_start) / 1000.0;

    LOG("\n--- Generation End ---\n\n");

    LOG_INF("generated %d tokens in %.2f ms (%.2f tokens/s)\n",
            n_generated, gen_ms, n_generated > 0 ? (n_generated / (gen_ms / 1000.0)) : 0.0);

    llama_batch_free(batch_gen);
    llama_backend_free();

    return 0;
}

#include "speculative-prefill.h"

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#define SPF_DBG(fmt, ...) LOG_DBG("spec-prefill: " fmt, __VA_ARGS__)
#define SPF_INF(fmt, ...) LOG_INF("spec-prefill: " fmt, __VA_ARGS__)
#define SPF_ERR(fmt, ...) LOG_ERR("spec-prefill: " fmt, __VA_ARGS__)

std::vector<float> common_speculative_prefill_compute_importance(
    const std::vector<std::vector<float>> & attn_layers,
    int32_t n_prompt,
    int32_t n_heads,
    int32_t pool_kernel_size) {
    if (attn_layers.empty() || n_prompt <= 0 || n_heads <= 0) {
        return std::vector<float>(std::max(0, n_prompt), 1.0f);
    }

    std::vector<float> max_importance(n_prompt, 0.0f);
    std::vector<float> smoothed(n_prompt, 0.0f);

    const int32_t pad = pool_kernel_size > 0 ? (pool_kernel_size / 2) : 0;

    for (const auto & layer_attn : attn_layers) {
        if ((int32_t) layer_attn.size() < n_heads * n_prompt) {
            continue;
        }

        for (int32_t h = 0; h < n_heads; ++h) {
            const float * head_attn = layer_attn.data() + (size_t) h * n_prompt;

            if (pool_kernel_size > 1) {
                for (int32_t i = 0; i < n_prompt; ++i) {
                    const int32_t start = std::max(0, i - pad);
                    const int32_t end   = std::min(n_prompt - 1, i + pad);

                    float sum = 0.0f;
                    for (int32_t j = start; j <= end; ++j) {
                        sum += head_attn[j];
                    }
                    smoothed[i] = sum / (float) (end - start + 1);
                }
            } else {
                std::copy(head_attn, head_attn + n_prompt, smoothed.begin());
            }

            for (int32_t i = 0; i < n_prompt; ++i) {
                if (smoothed[i] > max_importance[i]) {
                    max_importance[i] = smoothed[i];
                }
            }
        }
    }

    return max_importance;
}

std::vector<int32_t> common_speculative_prefill_select_indices(
    const std::vector<float> & importance_scores,
    const common_params_speculative_prefill & params) {
    const int32_t n_prompt = (int32_t) importance_scores.size();
    if (n_prompt <= 0) {
        return {};
    }

    const float percentage = std::max(0.01f, std::min(1.0f, params.percentage));
    std::vector<int32_t> selected;

    if (params.chunk_size > 0 && n_prompt > params.chunk_size) {
        const int32_t chunk_size = params.chunk_size;
        const int32_t n_chunks = (n_prompt + chunk_size - 1) / chunk_size;

        std::vector<std::pair<float, int32_t>> chunk_scores(n_chunks);

        for (int32_t c = 0; c < n_chunks; ++c) {
            const int32_t start = c * chunk_size;
            const int32_t end   = std::min(n_prompt, (c + 1) * chunk_size);

            float sum = 0.0f;
            for (int32_t i = start; i < end; ++i) {
                sum += importance_scores[i];
            }
            chunk_scores[c] = { sum / (float) (end - start), c };
        }

        const int32_t n_keep_chunks = std::max(1, (int32_t) std::ceil(n_chunks * percentage));

        std::sort(chunk_scores.begin(), chunk_scores.end(),
            [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                return a.first > b.first;
            });

        for (int32_t k = 0; k < n_keep_chunks && k < n_chunks; ++k) {
            const int32_t c = chunk_scores[k].second;
            const int32_t start = c * chunk_size;
            const int32_t end   = std::min(n_prompt, (c + 1) * chunk_size);

            for (int32_t i = start; i < end; ++i) {
                selected.push_back(i);
            }
        }
    } else {
        const int32_t n_keep = std::max(1, (int32_t) std::ceil(n_prompt * percentage));

        std::vector<std::pair<float, int32_t>> token_scores(n_prompt);
        for (int32_t i = 0; i < n_prompt; ++i) {
            token_scores[i] = { importance_scores[i], i };
        }

        std::sort(token_scores.begin(), token_scores.end(),
            [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                return a.first > b.first;
            });

        for (int32_t k = 0; k < n_keep; ++k) {
            selected.push_back(token_scores[k].second);
        }
    }

    if (params.keep_bos) {
        selected.push_back(0);
    }

    if (params.keep_last) {
        if (params.chunk_size > 0 && n_prompt > params.chunk_size) {
            const int32_t start = ((n_prompt - 1) / params.chunk_size) * params.chunk_size;
            for (int32_t i = start; i < n_prompt; ++i) {
                selected.push_back(i);
            }
        } else {
            selected.push_back(n_prompt - 1);
        }
    }

    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());

    return selected;
}

struct cb_attn_collector_data {
    int32_t n_prompt = 0;
    std::vector<std::vector<float>> layer_attns;
};

static bool cb_collect_attn(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return strncmp(t->name, "kq_soft_max", 11) == 0;
    }

    if (strncmp(t->name, "kq_soft_max", 11) != 0) {
        return true;
    }

    auto * data = (cb_attn_collector_data *) user_data;
    if (data == nullptr || data->n_prompt <= 0) {
        return true;
    }

    if (t->type != GGML_TYPE_F32) {
        return true;
    }

    const int32_t n_past_total = (int32_t) t->ne[0];
    const int32_t n_heads      = (int32_t) t->ne[2];
    const int32_t n_prompt     = data->n_prompt;

    if (t->nb[0] != sizeof(float) || n_heads <= 0 || n_past_total < n_prompt) {
        return true;
    }

    std::vector<float> layer_data((size_t) n_heads * n_prompt);

    for (int32_t h = 0; h < n_heads; ++h) {
        float * dst = layer_data.data() + (size_t) h * n_prompt;
        ggml_backend_tensor_get(t, dst, (size_t) h * t->nb[2], (size_t) n_prompt * sizeof(float));
    }

    data->layer_attns.push_back(std::move(layer_data));

    return true;
}

common_speculative_prefill_result common_speculative_prefill_execute(
    llama_context * ctx_dft,
    common_sampler * smpl_dft,
    const std::vector<llama_token> & prompt,
    llama_seq_id seq_id,
    const common_params_speculative_prefill & params) {
    common_speculative_prefill_result res;
    res.n_prompt_orig = (int32_t) prompt.size();

    if (prompt.empty()) {
        return res;
    }

    if (!params.enabled || params.percentage >= 1.0f || (int32_t) prompt.size() <= params.chunk_size) {
        res.kept_indices.resize(prompt.size());
        std::iota(res.kept_indices.begin(), res.kept_indices.end(), 0);
        res.n_prompt_kept = (int32_t) res.kept_indices.size();
        res.importance_scores.assign(prompt.size(), 1.0f);
        return res;
    }

    const int32_t n_ctx_dft = llama_n_ctx(ctx_dft);
    const int32_t lookahead = std::max(1, params.look_ahead_cnt);

    if ((int32_t) prompt.size() + lookahead > n_ctx_dft) {
        SPF_INF("prompt size (%d) + lookahead (%d) exceeds draft context (%d); skipping speculative prefill\n",
                (int32_t) prompt.size(), lookahead, n_ctx_dft);
        res.kept_indices.resize(prompt.size());
        std::iota(res.kept_indices.begin(), res.kept_indices.end(), 0);
        res.n_prompt_kept = (int32_t) res.kept_indices.size();
        res.importance_scores.assign(prompt.size(), 1.0f);
        return res;
    }

    const auto t_start = ggml_time_us();

    const llama_model * model_dft = llama_get_model(ctx_dft);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    const int32_t n_heads_dft     = llama_model_n_head(model_dft);

    // 1. evaluate full prompt on draft model
    {
        const int32_t n_batch_dft = llama_n_batch(ctx_dft);
        llama_batch batch_prompt = llama_batch_init(std::min((int32_t) prompt.size(), n_batch_dft), 0, 1);

        for (int32_t i = 0; i < (int32_t) prompt.size(); i += n_batch_dft) {
            const int32_t n_eval = std::min((int32_t) prompt.size() - i, n_batch_dft);
            common_batch_clear(batch_prompt);

            for (int32_t j = 0; j < n_eval; ++j) {
                const int32_t idx = i + j;
                const bool is_last = (idx == (int32_t) prompt.size() - 1);
                common_batch_add(batch_prompt, prompt[idx], (llama_pos) idx, { seq_id }, is_last);
            }

            const int ret = llama_decode(ctx_dft, batch_prompt);
            if (ret != 0) {
                SPF_ERR("failed to decode prompt on draft model, ret = %d\n", ret);
                llama_batch_free(batch_prompt);
                res.kept_indices.resize(prompt.size());
                std::iota(res.kept_indices.begin(), res.kept_indices.end(), 0);
                res.n_prompt_kept = (int32_t) res.kept_indices.size();
                return res;
            }
        }
        llama_batch_free(batch_prompt);
    }

    const auto t_prefill_end = ggml_time_us();
    res.t_draft_eval_us = t_prefill_end - t_start;

    // 2. lookahead decode steps with attention extraction
    std::vector<float> total_importance(prompt.size(), 0.0f);

    cb_attn_collector_data cb_data;
    cb_data.n_prompt = (int32_t) prompt.size();

    // attach callback
    llama_set_eval_callback(ctx_dft, cb_collect_attn, &cb_data);

    llama_batch batch_decode = llama_batch_init(1, 0, 1);

    int32_t actual_steps = 0;
    int32_t cur_pos = (int32_t) prompt.size();

    for (int32_t k = 0; k < lookahead; ++k) {
        cb_data.layer_attns.clear();

        const llama_token token_id = common_sampler_sample(smpl_dft, ctx_dft, -1);
        common_sampler_accept(smpl_dft, token_id, true);

        if (llama_vocab_is_eog(vocab_dft, token_id)) {
            break;
        }

        common_batch_clear(batch_decode);
        common_batch_add(batch_decode, token_id, cur_pos++, { seq_id }, true);

        const int ret = llama_decode(ctx_dft, batch_decode);
        if (ret != 0) {
            SPF_ERR("failed lookahead decode step %d, ret = %d\n", k, ret);
            break;
        }

        if (!cb_data.layer_attns.empty()) {
            const auto step_importance = common_speculative_prefill_compute_importance(
                cb_data.layer_attns,
                (int32_t) prompt.size(),
                n_heads_dft,
                params.pool_kernel_size);

            for (size_t i = 0; i < prompt.size(); ++i) {
                total_importance[i] += step_importance[i];
            }
            actual_steps++;
        }
    }

    llama_batch_free(batch_decode);

    // detach callback
    llama_set_eval_callback(ctx_dft, nullptr, nullptr);

    if (actual_steps == 0) {
        LOG_WRN("spec-prefill: attention was not captured (flash attn on?); keeping full prompt\n");
        res.kept_indices.resize(prompt.size());
        std::iota(res.kept_indices.begin(), res.kept_indices.end(), 0);
        res.n_prompt_kept = (int32_t) res.kept_indices.size();
        res.importance_scores.assign(prompt.size(), 1.0f);
        return res;
    }

    for (size_t i = 0; i < prompt.size(); ++i) {
        total_importance[i] /= (float) actual_steps;
    }

    const auto t_est_start = ggml_time_us();
    res.importance_scores = total_importance;
    res.kept_indices = common_speculative_prefill_select_indices(total_importance, params);
    res.n_prompt_kept = (int32_t) res.kept_indices.size();
    res.t_estimate_us = ggml_time_us() - t_est_start;

    return res;
}

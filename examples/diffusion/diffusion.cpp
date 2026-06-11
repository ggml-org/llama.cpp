#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Self-conditioning (DiffusionGemma): cache each step's canvas-row logits and feed them into the next
    // step (canvas = [n_input, max_length)); set_sc is a no-op for other models.
    llama_model *      sc_model = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t      sc_canvas = params.max_length - n_input;
    std::vector<float> sc_buffer;
    if (params.self_conditioning) {
        sc_buffer.assign((size_t) sc_canvas * n_vocab, 0.0f);
    }

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        GGML_ASSERT(params.max_length % params.block_length == 0);
        num_blocks = params.max_length / params.block_length;
        GGML_ASSERT(params.steps % num_blocks == 0);
        steps_per_block = params.steps / num_blocks;
    }

    std::vector<float> confidence(params.max_length);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    for (int block_num = 0; block_num < num_blocks; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;

            if (params.step_callback) {
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    break;
                }
            }

            // Setup batch
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = output_tokens[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            if (params.self_conditioning) {
                // step 0 has no previous prediction: keep the SC subgraph (stable graph shape) but gate it off
                llama_diffusion_set_sc(sc_model, sc_buffer.data(), global_step == 0 ? 0.0f : 1.0f, 1.0f, true);
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }
                logits = llama_get_logits(ctx);
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            if (params.self_conditioning) {
                std::memcpy(sc_buffer.data(), logits + (size_t) n_input * n_vocab,
                            (size_t) sc_canvas * n_vocab * sizeof(float));
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                if (params.shift_logits) {
                    return pos == 0 ? logits : logits + (pos - 1) * n_vocab;
                }
                return logits + pos * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }
                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;  // never reveal as mask
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    int32_t       pos        = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }
                    if (params.suppress_mask_token) {
                        candidates[params.mask_token_id].logit = -INFINITY;  // never reveal as mask
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    llama_token sampled_token = cur_p.data[cur_p.selected].id;

                    float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
        }
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / params.steps,
            total_sampling_time / 1000.0 / params.steps);

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = params.max_length;
}

// Entropy-bound denoiser for DiffusionGemma-style canvas models (see diffusion.h). The canvas is
// random-initialized; each step samples a candidate per position, accepts the lowest-entropy positions
// within a mutual-information bound, and renoises the rest under a linear temperature schedule. The output
// is the stable argmax canvas. Mirrors the reference transformers EntropyBoundSampler; set_sc is a no-op
// for non-DiffusionGemma models.
void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t C       = params.max_length - n_input;            // canvas length
    const int32_t S       = std::max(1, params.max_denoising_steps);

    // device-resident SC: source self-conditioning from a persistent device buffer (written in-graph from
    // the prev step's logits) instead of the 268 MB host upload each step. Exact: SC values/math unchanged.
    const bool dev_sc = params.gpu_sampling;
    const bool gpu_sample_reduce = params.gpu_sample_reduce && dev_sc;  // Stage-1: sample from sc_dev on-device
    llama_diffusion_set_device_sc(model, dev_sc);

    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(C);                    // working (renoised) canvas, fed to the forward
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);                      // random init (not mask)
    }

    // previous step's raw logits, for self-cond (host upload path only; device SC keeps them on-device)
    std::vector<float>       sc_buffer((size_t) (dev_sc ? 0 : C) * n_vocab, 0.0f);
    std::vector<llama_token> argmax_canvas(C, 0);                  // model's best prediction = the output
    std::vector<llama_token> prev_argmax(C, -1);                  // stability history (-1 -> step 0 is unstable)
    std::vector<float>       entropy(C);
    std::vector<llama_token> denoiser(C);
    std::vector<int32_t>     order(C);
    std::vector<float>       u(C);                                // pre-drawn multinomial draws (determinism)
    std::vector<llama_token> renoise(C);                         // pre-drawn renoise tokens

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);

    // Cached path: PREFILL the prompt once (writing the prefix K/V store), then each step DECODE only the
    // canvas, reading the cached prefix - instead of re-decoding [prompt|canvas] every step. The packed
    // canvas logits then start at row 0 (cached) instead of row n_input (unified).
    const int32_t logit_off = params.kv_cache ? 0 : n_input;
    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, n_input);
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;  // encode() forces all rows to output anyway; set them so it stays quiet
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: PREFILL decode failed\n", __func__);
            llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
            llama_batch_free(batch);
            return;
        }
    }

    float   prev_temp_inv = 1.0f;
    int     held          = 0;
    bool    finished      = false;
    bool    device_sample_ok = gpu_sample_reduce;   // latched off if a backend (e.g. Metal) can't device-sample

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;                    // 0-based
        const float   t        = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        if (params.kv_cache) {
            llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_input);
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = (i < n_input) ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        }

        // self-conditioning = softmax(previous step's logits / previous t); gated off on the first step.
        // device SC ignores the host pointer (reads sc_dev), so pass nullptr; the gate + temp are identical.
        llama_diffusion_set_sc(model, dev_sc ? nullptr : sc_buffer.data(),
                               step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }

        // Stage-1: when on, skip the 268 MB logits D2H + host reductions and sample on the GPU from sc_dev.
        // DG_DEVSAMPLE_CHECK forces both paths so we can diff them; it needs the host logits, so fetch them.
        const bool gpu_reduce  = dev_sc && device_sample_ok;
        const bool want_logits = !gpu_reduce || std::getenv("DG_DEVSAMPLE_CHECK") || std::getenv("DG_SC_CHECK");
        const float * logits = nullptr;                           // canvas rows packed: [C or max_length, n_vocab]
        if (want_logits) {
            logits = llama_get_logits(ctx);
        } else {
            llama_synchronize(ctx);                               // sc_dev write must complete before we read it
        }

        // debug: verify the device SC buffer captured exactly this step's canvas logits (== what the host
        // path uploads next step). Single-run check, independent of cross-run nondeterminism. DG_SC_CHECK=1.
        if (dev_sc && logits && std::getenv("DG_SC_CHECK")) {
            static std::vector<float> sc_dbg;
            sc_dbg.resize((size_t) C * n_vocab);
            const size_t got = llama_diffusion_debug_get_sc_dev(model, sc_dbg.data(), sc_dbg.size());
            double maxabs = 0.0; size_t nmiss = 0; double sumabs = 0.0;
            for (int32_t pos = 0; pos < C; pos++) {
                const float * hrow = logits + (size_t) (logit_off + pos) * n_vocab;
                const float * drow = sc_dbg.data() + (size_t) pos * n_vocab;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const double d = std::fabs((double) hrow[v] - (double) drow[v]);
                    sumabs += d;
                    if (d > maxabs) { maxabs = d; }
                    if (d != 0.0) { nmiss++; }
                }
            }
            LOG_INF("DG_SC_CHECK step %d: got=%zu maxabs=%.6g sumabs=%.6g nmiss=%zu/%zu sc_dev[0]=%.4f host[0]=%.4f\n",
                    step_idx, got, maxabs, sumabs, nmiss, (size_t) C * n_vocab,
                    sc_dbg.empty() ? 0.0f : sc_dbg[0], logits[(size_t) logit_off * n_vocab]);
        }

        // pre-draw the step's randomness single-threaded so the output is seed-reproducible
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // per position: argmax, entropy of softmax(raw/t), and a multinomial sample; stash raw row for SC
        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
                float m = -INFINITY; int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }
                const float target = u[pos] * Z;
                float   cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1; bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }
                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;
                // device SC keeps prev-step logits on-device (cpy in-graph), so no host stash needed
                if (!dev_sc) {
                    std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row, n_vocab * sizeof(float));
                }
            }
        };
        auto run_host_worker = [&]() {
            std::vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto & th : pool) { th.join(); }
        };

        if (gpu_reduce) {
            // Stage-1: argmax/entropy/sampled straight from sc_dev. argmax matches the host bit-for-bit; Z and
            // entropy differ only by the parallel-reduction order, so some sampled tokens may shift near ties.
            if (!llama_diffusion_device_sample(model, u.data(), argmax_canvas.data(), entropy.data(),
                                               denoiser.data(), C, temp_inv)) {
                // Some backends (e.g. Metal) cannot run the on-device sampler. Warn once and use the host
                // reduction for the rest of the run instead of retrying (and logging) on every step.
                if (device_sample_ok) {
                    LOG_WRN("%s: on-device sampling unsupported on this backend; using host sampling\n", __func__);
                    device_sample_ok = false;
                }
                if (!logits) { logits = llama_get_logits(ctx); }
                run_host_worker();
            } else if (std::getenv("DG_DEVSAMPLE_CHECK")) {
                // run the host worker into shadow buffers with the SAME u and diff: argmax must be exact.
                std::vector<llama_token> h_argmax = argmax_canvas, h_denoise = denoiser;
                std::vector<float>       h_entropy = entropy;
                std::vector<llama_token> d_argmax = argmax_canvas, d_denoise = denoiser;
                std::vector<float>       d_entropy = entropy;
                std::swap(argmax_canvas, h_argmax); std::swap(denoiser, h_denoise); std::swap(entropy, h_entropy);
                run_host_worker();  // fills argmax_canvas/denoiser/entropy from host
                int  amax_mismatch = 0, tok_diff = 0; double max_dH = 0.0, max_relZ = 0.0;
                for (int32_t pos = 0; pos < C; pos++) {
                    if (argmax_canvas[pos] != d_argmax[pos]) { amax_mismatch++; }
                    if (denoiser[pos]      != d_denoise[pos]) { tok_diff++; }
                    const double dH = std::fabs((double) entropy[pos] - (double) d_entropy[pos]);
                    if (dH > max_dH) { max_dH = dH; }
                    const double relZ = std::fabs((double) entropy[pos] - (double) d_entropy[pos]) /
                                        (std::fabs((double) entropy[pos]) + 1e-6);
                    if (relZ > max_relZ) { max_relZ = relZ; }
                }
                LOG_INF("DG_DEVSAMPLE_CHECK step %d: amax_mismatch=%d/%d tok_diff=%d/%d max|dH|=%.3e max_relH=%.3e\n",
                        step_idx, amax_mismatch, C, tok_diff, C, max_dH, max_relZ);
                // keep the DEVICE result for the actual run (host shadow was only for the diff)
                std::swap(argmax_canvas, d_argmax); std::swap(denoiser, d_denoise); std::swap(entropy, d_entropy);
            }
        } else {
            run_host_worker();
        }

        // accept the lowest-entropy positions within the MI bound (sum of strictly-earlier entropies <= bound)
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= params.entropy_bound) { accepted[pos] = 1; }
        }

        // renoise: accepted -> sampled token, rest -> fresh random; the displayed/output canvas is the argmax
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos]          = accepted[pos] ? denoiser[pos] : renoise[pos];
            output_tokens[n_input + pos] = argmax_canvas[pos];
            entropy_sum += entropy[pos];
        }

        // adaptive stop: argmax stable for stability_threshold steps AND confident (low mean entropy)
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
        if (held >= params.stability_threshold && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        if (params.step_callback &&
            !params.step_callback(step_idx, S, output_tokens, params.max_length, params.step_callback_user_data)) {
            break;
        }
    }

    if (params.kv_cache) {
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);  // restore default for later turns / masked path
    }
    if (dev_sc) {
        llama_diffusion_set_device_sc(model, false);             // restore host SC path for later turns
    }
    llama_batch_free(batch);
    n_generated = params.max_length;
}

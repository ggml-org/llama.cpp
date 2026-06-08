// Block-diffusion generation for diffusion_gemma.
//
// Implements the reference block-diffusion loop (EntropyBoundSampler + StableAndConfident
// stopping + linear temperature schedule) with KV-cache reuse:
//
//   * ENCODER phase (causal, no self-conditioning): the prompt is prefilled once into the
//     unified sliding-window KV cache. Its per-layer K/V become the read-only prefix.
//   * DECODER phase (bidirectional, self-conditioned): each denoising step decodes only the
//     canvas tokens at positions [n_past, n_past+canvas). They read the cached prefix and
//     attend the canvas bidirectionally. After reading the logits the canvas K/V is rolled
//     back (llama_memory_seq_rm) so the cache keeps only the committed prefix.
//
// This avoids re-encoding the prompt on every denoising step. Multi-block autoregressive
// generation (commit the finalized canvas via an encoder pass, then advance n_past) is
// layered on top of this single-block loop.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

// reference defaults from generation_config.json / DiffusionGemmaGenerationConfig
static constexpr int   DEF_CANVAS_LENGTH      = 256;
static constexpr int   DEF_MAX_DENOISE_STEPS  = 48;
static constexpr float ENTROPY_BOUND          = 0.1f;   // EntropyBoundSamplerConfig.entropy_bound
static constexpr float TEMP_MIN               = 0.4f;   // LinearTemperatureScheduleConfig.t_min
static constexpr float TEMP_MAX               = 0.8f;   // LinearTemperatureScheduleConfig.t_max
static constexpr float CONFIDENCE_THRESHOLD   = 0.005f; // StableAndConfident.confidence_threshold
static constexpr int   STABILITY_THRESHOLD    = 1;      // StableAndConfident.stability_threshold

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// apply the model's chat template to the user prompt (this is a chat-trained model)
static std::string format_chat(llama_model * model, const std::string & prompt) {
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user";
    user.content = prompt;
    inputs.messages.push_back(user);
    inputs.add_generation_prompt = true;
    return common_chat_templates_apply(tmpls.get(), inputs).prompt;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();

    // diffusion config (env-overridable for quick CPU testing)
    // canvas_length is fixed at the trained block size (256); overriding it is for experiments only.
    const int   canvas_length = env_int("DG_CANVAS", DEF_CANVAS_LENGTH);
    const int   n_steps       = env_int("DG_STEPS", DEF_MAX_DENOISE_STEPS);
    // number of autoregressive canvas blocks = ceil(-n / canvas_length), i.e. -n is the total
    // number of tokens to generate (e.g. -n 256 -> 1 canvas, -n 512 -> 2, -n 1024 -> 4). The
    // model may stop earlier on an EOG token. DG_MAX_CANVASES overrides; default (no -n) is 1.
    const int   blocks_from_n = params.n_predict > 0 ? (params.n_predict + canvas_length - 1) / canvas_length : 1;
    const int   max_canvases  = std::max(env_int("DG_MAX_CANVASES", blocks_from_n), 1); // autoregressive blocks
    const float entropy_bound = ENTROPY_BOUND;

    // top-k host sampling (CLI flags; default = full softmax over the whole vocab):
    //   --top-k k                 : top-k logits per position for softmax/sample/self-cond (0 = full).
    //   --top-k-start/--top-k-end : anneal k from START (first/high-entropy step) to END (last step).
    //   --top-k-tail-correction   : exact full-vocab entropy (logsumexp) for the accept/stop signal,
    //                               instead of the under-estimating top-k entropy.
    // --top-k uses its own "0 = disabled" convention and is applied only when explicitly passed.
    const int topk_fixed = (params.sampling.user_sampling_config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K)
                         ? params.sampling.top_k
                         : 0;
    const int topk_start = params.diffusion.top_k_start;
    const int topk_end   = params.diffusion.top_k_end;
    const int topk_tail  = params.diffusion.top_k_tail_correction ? 1 : 0;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    // Offload all layers to the GPU by default (when built with a GPU backend, e.g.
    // -DGGML_CUDA=ON). Pass -ngl N to limit offload, or -ngl 0 to force CPU. With a
    // CPU-only build this has no effect. (params.n_gpu_layers defaults to -1 = auto.)
    model_params.n_gpu_layers = params.n_gpu_layers >= 0 ? params.n_gpu_layers : 999;
    model_params.devices      = params.devices.data();
    model_params.use_mmap     = params.use_mmap;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: not a diffusion model\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // Build the prompt prefix. Text-only path tokenizes the chat-formatted prompt. Multimodal
    // path (--mmproj + --image) tokenizes via libmtmd: the image marker expands to the gemma
    // image tokens and the vision embeddings are produced by the GEMMA4V mmproj.
    const bool use_mm = !params.mmproj.path.empty() && !params.image.empty();

    std::vector<llama_token> prompt_tokens;               // text-only prefill
    mtmd::context_ptr        mctx_vision;                 // multimodal context
    mtmd::input_chunks       mm_chunks(mtmd_input_chunks_init());
    int prefix_len = 0;                                   // total positions in the prompt prefix

    if (use_mm) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu       = params.mmproj_use_gpu;
        mparams.print_timings = false;
        mparams.n_threads     = params.cpuparams.n_threads;
        mctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!mctx_vision) {
            LOG_ERR("error: failed to load mmproj '%s'\n", params.mmproj.path.c_str());
            llama_model_free(model);
            return 1;
        }

        // load image(s) and build one media marker per image
        mtmd::bitmaps bitmaps;
        std::string markers;
        for (const auto & img : params.image) {
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mctx_vision.get(), img.c_str()));
            if (!bmp.ptr) {
                LOG_ERR("error: failed to load image '%s'\n", img.c_str());
                llama_model_free(model);
                return 1;
            }
            bitmaps.entries.push_back(std::move(bmp));
            markers += mtmd_default_marker();
            markers += "\n";
        }

        // chat-format with the image marker(s) prepended to the user content
        const std::string formatted = format_chat(model, markers + params.prompt);
        LOG_INF("formatted prompt: %s\n", formatted.c_str());

        mtmd_input_text text;
        text.text          = formatted.c_str();
        text.add_special   = false;
        text.parse_special = true;
        auto bmp_c = bitmaps.c_ptr();
        if (mtmd_tokenize(mctx_vision.get(), mm_chunks.ptr.get(), &text, bmp_c.data(), bmp_c.size()) != 0) {
            LOG_ERR("error: mtmd_tokenize failed\n");
            llama_model_free(model);
            return 1;
        }
        prefix_len = (int) mtmd_helper_get_n_pos(mm_chunks.ptr.get());
    } else {
        // text-only: chat-format and tokenize (turn/channel special tokens)
        if (!params.prompt.empty()) {
            const std::string formatted = format_chat(model, params.prompt);
            LOG_INF("formatted prompt: %s\n", formatted.c_str());
            prompt_tokens = common_tokenize(vocab, formatted, /*add_special*/ false, /*parse_special*/ true);
        }
        prefix_len = (int) prompt_tokens.size();
    }

    // Context holds the committed prefix (prompt + finalized canvases) plus the canvas being
    // denoised, plus one extra canvas of headroom (the in-flight canvas's K/V is written then
    // rolled back each denoising step, so the ring buffer needs room before cells are reused).
    const int n_ctx_min = prefix_len + (max_canvases + 1) * canvas_length;
    const int n_ctx     = std::max<int>(n_ctx_min, (int) params.n_ctx);
    // Largest single decode is the prompt prefill (prefix_len) or a canvas pass (canvas_length).
    const int n_ub      = std::max(std::max(prefix_len, canvas_length), 1);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ub;
    ctx_params.n_ubatch = n_ub;
    ctx_params.no_perf  = params.no_perf;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    llama_set_diffusion_prompt_len(ctx, prefix_len);
    llama_memory_t mem = llama_get_memory(ctx);

    LOG_INF("diffusion-gemma: prefix=%d canvas=%d max_canvases=%d steps=%d entropy_bound=%.3f temp=[%.2f,%.2f] n_ctx=%d mm=%d\n",
            prefix_len, canvas_length, max_canvases, n_steps, entropy_bound, TEMP_MIN, TEMP_MAX, n_ctx, (int) use_mm);
    if (topk_fixed > 0 || (topk_start > 0 && topk_end > 0)) {
        LOG_INF("diffusion-gemma: top-k sampling: fixed=%d anneal=[%d->%d] tail_correction=%d (vocab=%d)\n",
                topk_fixed, topk_start, topk_end, topk_tail, n_vocab);
    }

    std::mt19937 rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed);
    std::uniform_int_distribution<int> rand_tok(0, n_vocab - 1);
    std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);

    llama_batch batch = llama_batch_init(n_ub, 0, 1);

    // ---- ENCODER phase: prefill the prompt prefix into the KV cache (no self-conditioning) ----
    int n_past = 0;
    llama_set_diffusion_decoder_phase(ctx, false);
    llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
    const auto t_prefill_start = std::chrono::steady_clock::now();
    if (use_mm) {
        // mtmd_helper_eval_chunks decodes text chunks (tokens, causal) and the image chunk
        // (vision embeddings, bidirectional for gemma) into the cache, managing causal_attn.
        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(mctx_vision.get(), ctx, mm_chunks.ptr.get(),
                /*n_past*/ 0, /*seq_id*/ 0, /*n_batch*/ n_ub, /*logits_last*/ true, &new_n_past)) {
            LOG_ERR("error: multimodal prefill failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        n_past = (int) new_n_past; // prompt + image K/V is now the committed read-only prefix
    } else if (prefix_len > 0) {
        llama_set_causal_attn(ctx, true);
        batch.n_tokens = prefix_len;
        for (int i = 0; i < prefix_len; ++i) {
            batch.token[i]     = prompt_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == prefix_len - 1) ? 1 : 0; // logits unused; keep n_outputs >= 1
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: prompt prefill (encoder) decode failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        n_past = prefix_len; // prompt K/V is now the committed read-only prefix
    }
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();
    if (prefix_len > 0) {
        // encoder-phase prefill: causal, NO self-conditioning -> same forward as the base
        // gemma4 model (the self-cond matmul only runs in the decoder/denoise phase).
        LOG_INF("prefill (encoder, no self-cond): %d tokens in %.3f s (%.1f tok/s)\n",
                prefix_len, prefill_s, prefill_s > 0.0 ? prefix_len / prefill_s : 0.0);
    }

    std::vector<llama_token> canvas(canvas_length);
    std::vector<llama_token> argmax_canvas(canvas_length, -1);
    std::vector<llama_token> prev_argmax(canvas_length, -1);
    std::vector<llama_token> accepted(canvas_length);
    // sparse self-conditioning over the canvas: the previous step's top-SC_K token ids + their
    // (renormalized) softmax probabilities per position. Fed to the next decode (Option-2 graph
    // gather: the decoder gathers just these SC_K embedding rows and blends them, instead of a
    // dense full-vocab probs @ token_embd matmul). Zero probs => no self-conditioning (step 1).
    // SC_K must match llama_model_diffusion_gemma::N_SC_TOPK (the graph's fixed gather width).
    const int SC_K = 256;
    std::vector<int32_t> sc_ids ((size_t) SC_K * canvas_length, 0);
    std::vector<float>   sc_probs((size_t) SC_K * canvas_length, 0.0f);

    // all generated tokens across the autoregressive canvas blocks
    std::vector<llama_token> generated;

    // ---- autoregressive block loop: each block denoises a canvas against the cached prefix,
    //      then (if continuing) commits its finalized tokens to the cache as the next prefix ----
    int n_blocks_run  = 0;
    int n_steps_total = 0;
    const auto t_gen_start = std::chrono::steady_clock::now();

    bool done = false;
    for (int block = 0; block < max_canvases && !done; ++block) {
    ++n_blocks_run;
    // 1. initialize canvas with random tokens
    for (auto & t : canvas) t = rand_tok(rng);
    std::fill(prev_argmax.begin(), prev_argmax.end(), -1);
    llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0); // first step: zero self-conditioning

    // 2. denoising loop (DECODER phase): cur_step = n_steps .. 1
    int step_k = 0; // top-k used for the current step (0 = full softmax), for logging
    for (int cur_step = n_steps; cur_step >= 1; --cur_step) {
        ++n_steps_total;
        // 2a. decode the canvas only, at positions [n_past, n_past+canvas). Bidirectional, with
        // self-conditioning; it reads the cached prefix read-only. All canvas tokens are outputs.
        llama_set_causal_attn(ctx, false);
        llama_set_diffusion_decoder_phase(ctx, true);
        batch.n_tokens = canvas_length;
        for (int j = 0; j < canvas_length; ++j) {
            batch.token[j]     = canvas[j];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: llama_decode failed at step %d\n", cur_step);
            break;
        }
        // canvas logits occupy rows [0, canvas_length) (canvas-only ubatch)
        const float * logits = llama_get_logits(ctx);

        // 2b. linear temperature schedule: t = t_min + (t_max - t_min) * (cur_step / n_steps)
        const float temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * ((float) cur_step / (float) n_steps);

        std::vector<float> entropy(canvas_length);
        std::vector<llama_token> sampled(canvas_length);

        // k for this step: 0 = full softmax. With annealing, k is high at the first (high-entropy)
        // step and low at the last, since early canvases are flat (need many tokens) and late ones
        // are peaked (a few suffice).
        int k_step = 0;
        if (topk_start > 0 && topk_end > 0) {
            const float frac = (n_steps > 1) ? (float) (cur_step - 1) / (float) (n_steps - 1) : 0.0f; // 1 at first step (cur_step=n_steps), 0 at last (cur_step=1)
            k_step = (int) lroundf(topk_end + (topk_start - topk_end) * frac);
        } else if (topk_fixed > 0) {
            k_step = topk_fixed;
        }
        if (k_step <= 0 || k_step >= n_vocab) k_step = 0;
        step_k = k_step;

        // sparse self-cond for the NEXT step: top-SC_K (id, prob) per position. Cleared each step;
        // unused slots stay (id 0, prob 0) so the graph gather contributes nothing for them.
        std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);
        std::fill(sc_ids.begin(),   sc_ids.end(),   0);

        if (k_step == 0) {
            // ---- full softmax over the whole vocabulary (reference behaviour) ----
            // self-cond still feeds only the top-SC_K tokens (full-normalized probs); the dropped
            // tail carries negligible embedding weight and the post RMS norm absorbs the scale.
            std::vector<float> probs(n_vocab);
            std::vector<std::pair<float,int>> scheap; scheap.reserve(SC_K); // min-heap of (x, idx), size SC_K
            const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
            for (int j = 0; j < canvas_length; ++j) {
                const float * lg = logits + (size_t) j * n_vocab;
                float maxl = -INFINITY;
                int   amax = 0;
                scheap.clear();
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v] / temp;
                    if (x > maxl) { maxl = x; amax = v; }
                    if ((int) scheap.size() < SC_K) {
                        scheap.push_back({x, v});
                        std::push_heap(scheap.begin(), scheap.end(), cmp);
                    } else if (x > scheap.front().first) {
                        std::pop_heap(scheap.begin(), scheap.end(), cmp);
                        scheap.back() = {x, v};
                        std::push_heap(scheap.begin(), scheap.end(), cmp);
                    }
                }
                float sum = 0.0f;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = expf(lg[v] / temp - maxl);
                    probs[v] = p;
                    sum += p;
                }
                float ent = 0.0f;
                const float r = rand_unif(rng) * sum;
                float cum = 0.0f;
                int   tok = amax;
                bool  picked = false;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = probs[v] / sum;
                    if (p > 0.0f) ent -= p * logf(p);
                    cum += probs[v];
                    if (!picked && cum >= r) { tok = v; picked = true; }
                }
                // store top-SC_K self-cond (full-normalized probability per selected token)
                int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                float   * spr = sc_probs.data() + (size_t) j * SC_K;
                int slot = 0;
                for (auto & h : scheap) { sid[slot] = h.second; spr[slot] = expf(h.first - maxl) / sum; ++slot; }
                entropy[j]       = ent;
                sampled[j]       = tok;
                argmax_canvas[j] = amax;
            }
        } else {
            // ---- top-k host sampling: softmax / entropy / sample / self-cond over the top-k
            // logits only. Self-cond feeds the top min(k,SC_K) tokens (renormalized over the
            // sampled top-k), gathered in-graph; the dropped tail carries negligible weight. ----
            const int heap_k = std::max(k_step, SC_K); // collect enough for both sampling and self-cond
            std::vector<std::pair<float,int>> heap; // min-heap of (logit/temp, idx), size heap_k
            heap.reserve(heap_k);
            const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
            for (int j = 0; j < canvas_length; ++j) {
                const float * lg = logits + (size_t) j * n_vocab;
                float maxl = -INFINITY;
                int   amax = 0;
                heap.clear();
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v] / temp;
                    if (x > maxl) { maxl = x; amax = v; }
                    if ((int) heap.size() < heap_k) {
                        heap.push_back({x, v});
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    } else if (x > heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end(), cmp);
                        heap.back() = {x, v};
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    }
                }
                // sort the collected entries by logit descending (exp is monotonic with x): the
                // first k_step drive sampling/entropy, the first SC_K drive self-cond.
                std::sort(heap.begin(), heap.end(), [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });

                // softmax over the sampled top-k (renormalized); reuse .first to hold exp value
                float Zk = 0.0f;
                for (int i = 0; i < k_step; ++i) { const float e = expf(heap[i].first - maxl); heap[i].first = e; Zk += e; }

                float ent;
                if (topk_tail) {
                    // exact full entropy via logsumexp over all logits (one expf pass, no per-token logf):
                    //   H = ln(Z) - (sum_i (z_i-max) e_i)/Z
                    double Zf = 0.0, T = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        const double d = (double) (lg[v] / temp) - (double) maxl;
                        const double e = exp(d);
                        Zf += e; T += d * e;
                    }
                    ent = (float) (log(Zf) - T / Zf);
                } else {
                    ent = 0.0f;
                    for (int i = 0; i < k_step; ++i) { const float q = heap[i].first / Zk; if (q > 0.0f) ent -= q * logf(q); }
                }

                // multinomial sample over the sampled top-k
                const float r = rand_unif(rng) * Zk;
                float cum = 0.0f;
                int   tok = amax;
                bool  picked = false;
                for (int i = 0; i < k_step; ++i) {
                    cum += heap[i].first;
                    if (!picked && cum >= r) { tok = heap[i].second; picked = true; }
                }

                // store top-SC_K self-cond (renormalized over the sampled top-k)
                int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                float   * spr = sc_probs.data() + (size_t) j * SC_K;
                const int n_sc = std::min(k_step, SC_K);
                for (int i = 0; i < n_sc; ++i) { sid[i] = heap[i].second; spr[i] = heap[i].first / Zk; }
                entropy[j]       = ent;
                sampled[j]       = tok;
                argmax_canvas[j] = amax;
            }
        }

        // roll back the canvas K/V written by this decode so the cache keeps only the committed
        // prefix [0, n_past); the next step re-decodes the canvas fresh against that prefix.
        llama_memory_seq_rm(mem, 0, n_past, -1);

        // self-conditioning for the NEXT denoising step: feed this step's top-SC_K (id, prob)
        llama_set_diffusion_self_cond_topk(ctx, sc_ids.data(), sc_probs.data(), SC_K, canvas_length);

        // 2c. entropy-bound accept: sort positions by entropy ascending, accept the prefix
        // where sum(entropy of all-but-last) <= entropy_bound (monotonic -> prefix selection)
        std::vector<int> order(canvas_length);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });

        std::vector<char> accept_mask(canvas_length, 0);
        float prefix = 0.0f;
        for (int k = 0; k < canvas_length; ++k) {
            if (prefix <= entropy_bound) {
                accept_mask[order[k]] = 1;
                prefix += entropy[order[k]];
            } else {
                break;
            }
        }

        // accepted canvas: accepted positions take the sampled token, others keep current
        int n_accept = 0;
        for (int i = 0; i < canvas_length; ++i) {
            if (accept_mask[i]) { accepted[i] = sampled[i]; ++n_accept; }
        }

        // mean entropy (confidence)
        const float mean_ent = std::accumulate(entropy.begin(), entropy.end(), 0.0f) / canvas_length;

        // 2d. stopping: stable (argmax canvas unchanged for STABILITY_THRESHOLD steps) AND confident
        bool stable = (STABILITY_THRESHOLD == 0) || (argmax_canvas == prev_argmax);
        bool confident = mean_ent < CONFIDENCE_THRESHOLD;
        LOG_INF("step %3d  temp=%.3f  k=%d  accepted=%4d/%d  mean_entropy=%.4f%s\n",
                cur_step, temp, step_k, n_accept, canvas_length, mean_ent,
                (stable && confident) ? "  [STOP]" : "");
        if (stable && confident) {
            break;
        }
        prev_argmax = argmax_canvas;

        // 2e. renoise non-accepted positions with fresh random tokens -> next canvas
        for (int i = 0; i < canvas_length; ++i) {
            canvas[i] = accept_mask[i] ? accepted[i] : rand_tok(rng);
        }
    }

    // 3. block output = the inline argmax of the last (stable) denoising step's logits.
    // This matches the reference (DiffusionGemma _denoising_step uses argmax(processed_logits)
    // taken during the denoising forward, read once the canvas is stable + confident). There is
    // no separate read-out: the never-accepted tail is the model's own prediction given the
    // settled context, rather than a stale-random scratch buffer.
    const std::vector<llama_token> & block_out = argmax_canvas;

    // accumulate this block's finalized tokens; stop after a block that contains an EOG token
    generated.insert(generated.end(), block_out.begin(), block_out.end());
    for (int j = 0; j < canvas_length; ++j) {
        if (llama_vocab_is_eog(vocab, block_out[j])) { done = true; break; }
    }

    // 4. COMMIT (ENCODER phase): if another block follows, write the finalized canvas's plain
    // (non-self-conditioned, causal) K/V into the cache and advance the prefix pointer, so the
    // next block's canvas cross-attends to it. Skipped on the last block / on EOG.
    if (!done && block + 1 < max_canvases) {
        llama_set_causal_attn(ctx, true);
        llama_set_diffusion_decoder_phase(ctx, false);
        llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
        batch.n_tokens = canvas_length;
        for (int j = 0; j < canvas_length; ++j) {
            batch.token[j]     = block_out[j];
            batch.pos[j]       = n_past + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (j == canvas_length - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: canvas commit (encoder) decode failed at block %d\n", block);
            break;
        }
        n_past += canvas_length; // finalized canvas is now part of the read-only prefix
        LOG_INF("committed block %d -> n_past=%d\n", block, n_past);
    }
    } // end autoregressive block loop

    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen_start).count();

    llama_batch_free(batch);

    // the full denoised canvas (thought channel + response), for reference
    LOG_INF("\n=== generated canvas ===\n%s\n", common_detokenize(vocab, generated, false).c_str());

    // the model answers in a "<|channel>thought ... <channel|>" block followed by the response;
    // extract the final response (after the last channel-close), truncated at the first
    // end-of-generation token, and drop trailing duplicate sentences.
    llama_token chan_close = LLAMA_TOKEN_NULL;
    {
        auto t = common_tokenize(vocab, "<channel|>", false, true);
        if (t.size() == 1) chan_close = t[0];
    }
    const int n_gen = (int) generated.size();
    int start = 0;
    if (chan_close != LLAMA_TOKEN_NULL) {
        for (int j = 0; j < n_gen; ++j) if (generated[j] == chan_close) start = j + 1;
    }
    std::vector<llama_token> answer;
    for (int j = start; j < n_gen; ++j) {
        if (llama_vocab_is_eog(vocab, generated[j])) break;
        answer.push_back(generated[j]);
    }
    std::string ans = common_detokenize(vocab, answer, false);
    // drop a trailing exact-duplicate of the answer if the canvas repeated it
    {
        std::string s = ans;
        size_t h = s.find_first_not_of(" \n\t"); if (h != std::string::npos) s = s.substr(h);
        const size_t half = s.size() / 2;
        if (half > 0 && s.compare(0, half, s, s.size() - half, half) == 0) {
            ans = s.substr(0, half); // "X X" -> "X"
        }
    }
    LOG_INF("=== answer ===\n%s\n", ans.c_str());

    // generation timing (excludes model load + prompt prefill): wall-clock of the denoising
    // block loop, the canvas tokens produced, and effective throughput.
    const int n_canvas_tok = n_blocks_run * canvas_length;
    LOG_INF("=== perf ===\n");
    LOG_INF("generation: %d block(s), %d denoising steps, %d canvas tokens in %.2f s "
            "(%.1f canvas tok/s, %.3f s/step); answer tokens=%d\n",
            n_blocks_run, n_steps_total, n_canvas_tok, gen_s,
            gen_s > 0.0 ? n_canvas_tok / gen_s : 0.0,
            n_steps_total > 0 ? gen_s / n_steps_total : 0.0,
            (int) answer.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

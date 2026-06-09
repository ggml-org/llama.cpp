// OpenAI-compatible HTTP server for the block-diffusion models (diffusion-gemma).
//
// This is the llama-server analogue for the diffusion family: it loads a block-diffusion model
// once and serves the same denoising generation loop as diffusion-gemma-cli over HTTP, exposing
// the OpenAI endpoints (/v1/chat/completions, /v1/completions, /v1/models) plus the llama-server
// observability surface (/health, /v1/health, /props, /metrics, /slots) with the same response
// `timings`/`usage` objects, the same per-request timing logs, the same access log, and a Prometheus
// /metrics endpoint -- all with the metric semantics re-mapped to block diffusion (canvas blocks,
// denoising steps, s/step) instead of autoregressive token-by-token decode.
//
// Generation is NOT autoregressive: each request denoises one or more 256-token canvases against a
// cached prompt prefix (see diffusion-gemma-cli.cpp for the full description). A single llama_context
// is reused across requests and is not thread-safe, so generation is serialized behind a mutex (one
// slot); the HTTP layer still accepts many connections concurrently.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

// llama-server-style log macros (server-common.h): "srv  <func>: ..." / "slot <func>: id N | ..."
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SLT_INF(id, fmt, ...) LOG_INF("slot %12.*s: id %2d | " fmt, 12, __func__, (id), __VA_ARGS__)

// reference defaults from generation_config.json / DiffusionGemmaGenerationConfig (mirror the CLI)
static constexpr int   DEF_CANVAS_LENGTH    = 256;
static constexpr int   DEF_MAX_DENOISE_STEPS = 48;
static constexpr float ENTROPY_BOUND        = 0.1f;
static constexpr float TEMP_MIN             = 0.4f;
static constexpr float TEMP_MAX             = 0.8f;
static constexpr float CONFIDENCE_THRESHOLD = 0.005f;
static constexpr int   STABILITY_THRESHOLD  = 1;
static constexpr int   SC_K                 = 256; // must match llama_model_diffusion_gemma::N_SC_TOPK
static constexpr int   GPU_SAMPLING_MAX_TOP_K = 1024; // CUDA diffusion sampler limit

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// ------------------------------------------------------------------------------------------------
// per-request generation parameters and result
// ------------------------------------------------------------------------------------------------
struct diffusion_request {
    int      canvas_length = DEF_CANVAS_LENGTH;
    int      n_steps       = DEF_MAX_DENOISE_STEPS;
    int      max_canvases  = 1;     // ceil(max_tokens / canvas_length)
    int      topk_fixed    = 0;     // 0 = full softmax
    int      topk_start    = 0;
    int      topk_end      = 0;
    bool     topk_tail     = false;
    bool     ignore_eos    = false; // run all max_canvases blocks (don't stop at end-of-text)
    uint32_t seed          = 1234;
};

struct diffusion_result {
    std::string answer;            // final response text (post channel-split / eog truncation)
    std::string full;             // full detokenized canvas (thought + response)
    int    prompt_tokens   = 0;    // prompt prefix tokens processed (encoder prefill)
    int    answer_tokens   = 0;    // tokens in the extracted answer
    int    n_blocks        = 0;    // canvas blocks run
    int    n_steps_total   = 0;    // denoising steps across all blocks
    int    canvas_tokens   = 0;    // n_blocks * canvas_length
    int    n_decode        = 0;    // total llama_decode() calls (prefill + denoise + commit)
    double prefill_ms      = 0.0;  // encoder-phase prompt prefill wall time
    double gen_ms          = 0.0;  // denoising block-loop wall time
    bool   ok              = false;
    std::string error;
};

// ------------------------------------------------------------------------------------------------
// server metrics (llama-server analogue; protected by its own mutex)
// ------------------------------------------------------------------------------------------------
struct server_metrics {
    std::mutex mu;
    int64_t  t_start = 0;                 // process start (unix seconds)

    // cumulative counters
    uint64_t n_requests_total           = 0;
    uint64_t n_prompt_tokens_total      = 0;
    double   t_prompt_ms_total          = 0.0;
    uint64_t n_tokens_predicted_total   = 0;   // answer tokens
    double   t_predicted_ms_total       = 0.0; // denoise time
    uint64_t n_decode_total             = 0;
    uint64_t n_blocks_total             = 0;
    uint64_t n_steps_total              = 0;
    uint64_t n_canvas_tokens_total      = 0;

    // last-request gauges
    double   last_prompt_tps            = 0.0;
    double   last_predicted_tps         = 0.0;
    double   last_steps_per_second      = 0.0;
    double   last_canvas_tps            = 0.0;
    double   last_ms_per_step           = 0.0;

    std::atomic<int> n_processing{0};

    void add(const diffusion_result & r) {
        std::lock_guard<std::mutex> lk(mu);
        n_requests_total          += 1;
        n_prompt_tokens_total     += r.prompt_tokens;
        t_prompt_ms_total         += r.prefill_ms;
        n_tokens_predicted_total  += r.answer_tokens;
        t_predicted_ms_total      += r.gen_ms;
        n_decode_total            += r.n_decode;
        n_blocks_total            += r.n_blocks;
        n_steps_total             += r.n_steps_total;
        n_canvas_tokens_total     += r.canvas_tokens;
        last_prompt_tps      = r.prefill_ms > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
        last_predicted_tps   = r.gen_ms     > 0 ? r.answer_tokens * 1e3 / r.gen_ms     : 0.0;
        last_steps_per_second= r.gen_ms     > 0 ? r.n_steps_total * 1e3 / r.gen_ms     : 0.0;
        last_canvas_tps      = r.gen_ms     > 0 ? r.canvas_tokens * 1e3 / r.gen_ms     : 0.0;
        last_ms_per_step     = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total        : 0.0;
    }
};

// ------------------------------------------------------------------------------------------------
// server state (one model + one reused context == one slot, guarded by a mutex)
// ------------------------------------------------------------------------------------------------
struct diffusion_server {
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_memory_t      mem   = nullptr;
    llama_batch         batch{};
    common_chat_templates_ptr templates;

    int    canvas_length = DEF_CANVAS_LENGTH;
    int    n_steps       = DEF_MAX_DENOISE_STEPS;
    int    n_ctx         = 0;
    int    n_ub          = 0;       // ubatch == canvas_length (keeps the decoder gather graph small)
    int    n_vocab       = 0;
    int    topk_fixed    = 0;
    int    topk_start    = 0;
    int    topk_end      = 0;
    bool   topk_tail     = false;
    bool   use_gpu_sampling = false;
    bool   use_device_self_cond = false;
    bool   use_device_loop = false;
    std::string model_id;
    std::string model_path;
    std::string build_info = "llama.cpp diffusion-gemma-server";

    server_metrics metrics;
    std::mutex gen_mutex;            // serialize generation (ctx is single-threaded == one slot)

    std::string format_messages(const json & messages) const {
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.add_generation_prompt = true;
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }

    // causal prefill of `toks` starting at position pos0, chunked to n_ub (encoder phase).
    // increments *n_dec with the number of llama_decode() calls made.
    bool prefill_causal(const std::vector<llama_token> & toks, int pos0, int * n_dec) {
        llama_set_causal_attn(ctx, true);
        llama_set_diffusion_decoder_phase(ctx, false);
        llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
        const int n = (int) toks.size();
        for (int off = 0; off < n; off += n_ub) {
            const int cnt = std::min(n_ub, n - off);
            batch.n_tokens = cnt;
            for (int i = 0; i < cnt; ++i) {
                batch.token[i]     = toks[off + i];
                batch.pos[i]       = pos0 + off + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (off + i == n - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) return false;
            ++(*n_dec);
        }
        return true;
    }

    // run the block-diffusion denoising loop for one request (mirrors diffusion-gemma-cli.cpp)
    diffusion_result generate(const std::vector<llama_token> & prompt_tokens, const diffusion_request & rq) {
        diffusion_result out;
        const int prefix_len = (int) prompt_tokens.size();
        int n_decode = 0;

        int max_canvases = rq.max_canvases;
        const int fit = (n_ctx - prefix_len) / canvas_length - 1;
        if (fit < 1) {
            out.error = "prompt too long for context (n_ctx=" + std::to_string(n_ctx) + ")";
            return out;
        }
        if (max_canvases > fit) max_canvases = fit;
        if (max_canvases < 1)   max_canvases = 1;

        llama_memory_clear(mem, true);

        std::mt19937 rng(rq.seed);
        std::uniform_int_distribution<int> rand_tok(0, n_vocab - 1);
        std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);

        // ---- ENCODER phase: prefill the prompt prefix (causal, no self-conditioning) ----
        const auto t_prefill0 = std::chrono::steady_clock::now();
        int n_past = 0;
        if (prefix_len > 0) {
            if (!prefill_causal(prompt_tokens, 0, &n_decode)) {
                out.error = "prompt prefill (encoder) decode failed";
                return out;
            }
            n_past = prefix_len;
        }
        out.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill0).count();

        std::vector<llama_token> canvas(canvas_length);
        std::vector<llama_token> argmax_canvas(canvas_length, -1);
        std::vector<llama_token> prev_argmax(canvas_length, -1);
        std::vector<llama_token> accepted(canvas_length);
        std::vector<int32_t>     sc_ids ((size_t) SC_K * canvas_length, 0);
        std::vector<float>       sc_probs((size_t) SC_K * canvas_length, 0.0f);
        std::vector<llama_token> generated;

        int n_blocks_run = 0, n_steps_total = 0;
        bool done = false;
        const int topk_max_requested =
            (rq.topk_start > 0 && rq.topk_end > 0) ? std::max(rq.topk_start, rq.topk_end) : rq.topk_fixed;
        const bool use_device_loop_request = use_device_loop &&
            (topk_max_requested <= 0 || topk_max_requested <= GPU_SAMPLING_MAX_TOP_K);

        const auto t_gen0 = std::chrono::steady_clock::now();
        for (int block = 0; block < max_canvases && !done; ++block) {
            ++n_blocks_run;
            for (auto & t : canvas) t = rand_tok(rng);
            std::fill(prev_argmax.begin(), prev_argmax.end(), -1);
            llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);

            for (int cur_step = n_steps; cur_step >= 1; --cur_step) {
                ++n_steps_total;
                const float temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * ((float) cur_step / (float) n_steps);

                int k_step = 0;
                if (rq.topk_start > 0 && rq.topk_end > 0) {
                    const float frac = (n_steps > 1) ? (float) (cur_step - 1) / (float) (n_steps - 1) : 0.0f;
                    k_step = (int) lroundf(rq.topk_end + (rq.topk_start - rq.topk_end) * frac);
                } else if (rq.topk_fixed > 0) {
                    k_step = rq.topk_fixed;
                }
                if (k_step <= 0 || k_step >= n_vocab) k_step = 0;
                const bool use_gpu_sampling_step = use_gpu_sampling &&
                    (k_step <= 0 || k_step <= GPU_SAMPLING_MAX_TOP_K);
                const bool use_device_self_cond_step = use_gpu_sampling_step && use_device_self_cond;
                const bool use_device_loop_step = use_device_loop_request && use_gpu_sampling_step;

                llama_set_causal_attn(ctx, false);
                llama_set_diffusion_decoder_phase(ctx, true);
                llama_set_diffusion_gpu_sampling(ctx, use_gpu_sampling_step);
                batch.n_tokens = canvas_length;
                for (int j = 0; j < canvas_length; ++j) {
                    batch.token[j]     = canvas[j];
                    batch.pos[j]       = n_past + j;
                    batch.n_seq_id[j]  = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]    = 1;
                }
                if (llama_decode(ctx, batch) != 0) {
                    out.error = "llama_decode failed at denoising step " + std::to_string(cur_step);
                    return out;
                }
                ++n_decode;

                std::vector<float> entropy(canvas_length);
                std::vector<llama_token> sampled(canvas_length);

                if (use_device_loop_step) {
                    llama_diffusion_sample_params sample_params = {
                        /* .n_tokens              = */ canvas_length,
                        /* .top_k                 = */ k_step,
                        /* .self_cond_top_k       = */ SC_K,
                        /* .temperature           = */ temp,
                        /* .seed                  = */ rq.seed,
                        /* .step                  = */ (uint32_t) n_steps_total,
                        /* .top_k_tail_correction = */ rq.topk_tail,
                    };
                    llama_diffusion_sample_result sample_result = {
                        /* .sampled         = */ nullptr,
                        /* .argmax          = */ nullptr,
                        /* .entropy         = */ nullptr,
                        /* .self_cond_ids   = */ nullptr,
                        /* .self_cond_probs = */ nullptr,
                        /* .final_tokens    = */ cur_step == 1 ? argmax_canvas.data() : nullptr,
                        /* .update_canvas_on_device = */ cur_step > 1,
                        /* .entropy_bound   = */ ENTROPY_BOUND,
                    };
                    if (!llama_diffusion_sample_topk(ctx, &sample_params, &sample_result)) {
                        llama_memory_seq_rm(mem, 0, n_past, -1);
                        out.error = "CUDA diffusion device loop failed at denoising step " +
                            std::to_string(cur_step) + " (k=" + std::to_string(k_step) + ")";
                        return out;
                    }
                    llama_memory_seq_rm(mem, 0, n_past, -1);
                    continue;
                }

                if (!use_device_self_cond_step) {
                    std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);
                    std::fill(sc_ids.begin(),   sc_ids.end(),   0);
                }

                if (use_gpu_sampling_step) {
                    llama_diffusion_sample_params sample_params = {
                        /* .n_tokens              = */ canvas_length,
                        /* .top_k                 = */ k_step,
                        /* .self_cond_top_k       = */ SC_K,
                        /* .temperature           = */ temp,
                        /* .seed                  = */ rq.seed,
                        /* .step                  = */ (uint32_t) n_steps_total,
                        /* .top_k_tail_correction = */ rq.topk_tail,
                    };
                    llama_diffusion_sample_result sample_result = {
                        /* .sampled         = */ sampled.data(),
                        /* .argmax          = */ argmax_canvas.data(),
                        /* .entropy         = */ entropy.data(),
                        /* .self_cond_ids   = */ use_device_self_cond_step ? nullptr : sc_ids.data(),
                        /* .self_cond_probs = */ use_device_self_cond_step ? nullptr : sc_probs.data(),
                    };
                    if (!llama_diffusion_sample_topk(ctx, &sample_params, &sample_result)) {
                        llama_memory_seq_rm(mem, 0, n_past, -1);
                        out.error = "CUDA diffusion sampling failed at denoising step " +
                            std::to_string(cur_step) + " (k=" + std::to_string(k_step) + ")";
                        return out;
                    }
                } else if (k_step == 0) {
                    const float * logits = llama_get_logits(ctx);
                    const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
                    std::vector<float> probs(n_vocab);
                    std::vector<std::pair<float,int>> scheap; scheap.reserve(SC_K);
                    for (int j = 0; j < canvas_length; ++j) {
                        const float * lg = logits + (size_t) j * n_vocab;
                        float maxl = -INFINITY; int amax = 0;
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
                        for (int v = 0; v < n_vocab; ++v) { const float p = expf(lg[v] / temp - maxl); probs[v] = p; sum += p; }
                        float ent = 0.0f;
                        const float r = rand_unif(rng) * sum;
                        float cum = 0.0f; int tok = amax; bool picked = false;
                        for (int v = 0; v < n_vocab; ++v) {
                            const float p = probs[v] / sum;
                            if (p > 0.0f) ent -= p * logf(p);
                            cum += probs[v];
                            if (!picked && cum >= r) { tok = v; picked = true; }
                        }
                        int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                        float   * spr = sc_probs.data() + (size_t) j * SC_K;
                        int slot = 0;
                        for (auto & h : scheap) { sid[slot] = h.second; spr[slot] = expf(h.first - maxl) / sum; ++slot; }
                        entropy[j] = ent; sampled[j] = tok; argmax_canvas[j] = amax;
                    }
                } else {
                    const float * logits = llama_get_logits(ctx);
                    const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
                    const int heap_k = std::max(k_step, SC_K);
                    std::vector<std::pair<float,int>> heap; heap.reserve(heap_k);
                    for (int j = 0; j < canvas_length; ++j) {
                        const float * lg = logits + (size_t) j * n_vocab;
                        float maxl = -INFINITY; int amax = 0;
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
                        std::sort(heap.begin(), heap.end(), [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });
                        float Zk = 0.0f;
                        for (int i = 0; i < k_step; ++i) { const float e = expf(heap[i].first - maxl); heap[i].first = e; Zk += e; }
                        float ent;
                        if (rq.topk_tail) {
                            double Zf = 0.0, T = 0.0;
                            for (int v = 0; v < n_vocab; ++v) {
                                const double d = (double) (lg[v] / temp) - (double) maxl;
                                const double e = exp(d); Zf += e; T += d * e;
                            }
                            ent = (float) (log(Zf) - T / Zf);
                        } else {
                            ent = 0.0f;
                            for (int i = 0; i < k_step; ++i) { const float q = heap[i].first / Zk; if (q > 0.0f) ent -= q * logf(q); }
                        }
                        const float r = rand_unif(rng) * Zk;
                        float cum = 0.0f; int tok = amax; bool picked = false;
                        for (int i = 0; i < k_step; ++i) { cum += heap[i].first; if (!picked && cum >= r) { tok = heap[i].second; picked = true; } }
                        int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                        float   * spr = sc_probs.data() + (size_t) j * SC_K;
                        const int n_sc = std::min(k_step, SC_K);
                        for (int i = 0; i < n_sc; ++i) { sid[i] = heap[i].second; spr[i] = heap[i].first / Zk; }
                        entropy[j] = ent; sampled[j] = tok; argmax_canvas[j] = amax;
                    }
                }

                llama_memory_seq_rm(mem, 0, n_past, -1);

                std::vector<int> order(canvas_length);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });
                std::vector<char> accept_mask(canvas_length, 0);
                float prefix = 0.0f;
                for (int k = 0; k < canvas_length; ++k) {
                    if (prefix <= ENTROPY_BOUND) { accept_mask[order[k]] = 1; prefix += entropy[order[k]]; }
                    else break;
                }
                for (int i = 0; i < canvas_length; ++i) if (accept_mask[i]) accepted[i] = sampled[i];

                const float mean_ent = std::accumulate(entropy.begin(), entropy.end(), 0.0f) / canvas_length;
                const bool stable    = (STABILITY_THRESHOLD == 0) || (argmax_canvas == prev_argmax);
                const bool confident = mean_ent < CONFIDENCE_THRESHOLD;
                if (stable && confident) break;
                prev_argmax = argmax_canvas;

                if (!use_device_self_cond_step) {
                    llama_set_diffusion_self_cond_topk(ctx, sc_ids.data(), sc_probs.data(), SC_K, canvas_length);
                }

                for (int i = 0; i < canvas_length; ++i) canvas[i] = accept_mask[i] ? accepted[i] : rand_tok(rng);
            }

            const std::vector<llama_token> & block_out = argmax_canvas;
            generated.insert(generated.end(), block_out.begin(), block_out.end());
            if (!rq.ignore_eos) {
                for (int j = 0; j < canvas_length; ++j) {
                    if (llama_vocab_is_eog(vocab, block_out[j])) { done = true; break; }
                }
            }
            if (!done && block + 1 < max_canvases) {
                if (!prefill_causal(block_out, n_past, &n_decode)) {
                    out.error = "canvas commit (encoder) decode failed";
                    return out;
                }
                n_past += canvas_length;
            }
        }
        out.gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen0).count();
        out.full   = common_detokenize(vocab, generated, false);

        // extract the final response: after the last "<channel|>" close, until the first eog token
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
        {
            std::string s = ans;
            size_t h = s.find_first_not_of(" \n\t"); if (h != std::string::npos) s = s.substr(h);
            const size_t half = s.size() / 2;
            if (half > 0 && s.compare(0, half, s, s.size() - half, half) == 0) ans = s.substr(0, half);
        }

        out.answer        = ans;
        out.prompt_tokens = prefix_len;
        out.answer_tokens = (int) answer.size();
        out.n_blocks      = n_blocks_run;
        out.n_steps_total = n_steps_total;
        out.canvas_tokens = n_blocks_run * canvas_length;
        out.n_decode      = n_decode;
        out.ok            = true;
        return out;
    }
};

// ------------------------------------------------------------------------------------------------
// JSON / log helpers
// ------------------------------------------------------------------------------------------------
static std::atomic<uint64_t> g_req_counter{0};
static std::string gen_id(const char * prefix) { return std::string(prefix) + "-" + std::to_string(g_req_counter.fetch_add(1)); }
static json error_json(const std::string & msg, const std::string & type, int code) {
    return json{{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
}

// llama-server `timings` object, with diffusion fields. prompt_* describes the encoder prefill;
// predicted_* describes the answer tokens; the nested `diffusion` object describes the denoising.
static json timings_json(const diffusion_result & r) {
    const double ppt = r.prompt_tokens > 0 ? r.prefill_ms / r.prompt_tokens : 0.0;
    const double pps = r.prefill_ms    > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
    const double dpt = r.answer_tokens > 0 ? r.gen_ms / r.answer_tokens : 0.0;
    const double dps = r.gen_ms        > 0 ? r.answer_tokens * 1e3 / r.gen_ms : 0.0;
    const double sps = r.gen_ms        > 0 ? r.n_steps_total * 1e3 / r.gen_ms : 0.0;
    const double cps = r.gen_ms        > 0 ? r.canvas_tokens * 1e3 / r.gen_ms : 0.0;
    const double mps = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total : 0.0;
    return json{
        {"cache_n", 0},
        {"prompt_n", r.prompt_tokens},
        {"prompt_ms", r.prefill_ms},
        {"prompt_per_token_ms", ppt},
        {"prompt_per_second", pps},
        {"predicted_n", r.answer_tokens},
        {"predicted_ms", r.gen_ms},
        {"predicted_per_token_ms", dpt},
        {"predicted_per_second", dps},
        {"diffusion", {
            {"n_blocks", r.n_blocks},
            {"n_steps", r.n_steps_total},
            {"canvas_tokens", r.canvas_tokens},
            {"ms_per_step", mps},
            {"steps_per_second", sps},
            {"canvas_tokens_per_second", cps},
            {"n_decode", r.n_decode},
        }},
    };
}

static json usage_json(const diffusion_result & r) {
    return json{
        {"prompt_tokens", r.prompt_tokens},
        {"completion_tokens", r.answer_tokens},
        {"total_tokens", r.prompt_tokens + r.answer_tokens},
        {"prompt_tokens_details", {{"cached_tokens", 0}}},
    };
}

// per-request timing log, llama_perf-style but with the denoise phase substituted for AR eval
static void log_timings(const diffusion_result & r) {
    const double ppt = r.prompt_tokens > 0 ? r.prefill_ms / r.prompt_tokens : 0.0;
    const double pps = r.prefill_ms    > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
    const double mps = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total : 0.0;
    const double sps = r.gen_ms        > 0 ? r.n_steps_total * 1e3 / r.gen_ms : 0.0;
    const double cps = r.gen_ms        > 0 ? r.canvas_tokens * 1e3 / r.gen_ms : 0.0;
    SLT_INF(0, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            r.prefill_ms, r.prompt_tokens, ppt, pps);
    SLT_INF(0, "    denoise time = %10.2f ms / %5d steps  (%8.2f ms per step,  %8.2f steps per second)\n",
            r.gen_ms, r.n_steps_total, mps, sps);
    SLT_INF(0, "      gen tokens = %5d answer | %5d canvas over %d block(s) (%8.2f canvas tok/s)\n",
            r.answer_tokens, r.canvas_tokens, r.n_blocks, cps);
    SLT_INF(0, "      total time = %10.2f ms / %5d decode call(s)\n",
            r.prefill_ms + r.gen_ms, r.n_decode);
}

static diffusion_request request_from_body(const json & body, const diffusion_server & srv, uint32_t default_seed) {
    diffusion_request rq;
    rq.canvas_length = srv.canvas_length;
    rq.n_steps       = srv.n_steps;
    rq.seed          = default_seed;
    rq.topk_fixed    = srv.topk_fixed;
    rq.topk_start    = srv.topk_start;
    rq.topk_end      = srv.topk_end;
    rq.topk_tail     = srv.topk_tail;
    int max_tokens = 0;
    if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())                       max_tokens = body["max_tokens"].get<int>();
    else if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer()) max_tokens = body["max_completion_tokens"].get<int>();
    rq.max_canvases = max_tokens > 0 ? (max_tokens + rq.canvas_length - 1) / rq.canvas_length : 1;
    if (body.contains("top_k") && body["top_k"].is_number_integer()) {
        rq.topk_fixed = body["top_k"].get<int>();
        rq.topk_start = 0;
        rq.topk_end   = 0;
    }
    if (body.contains("top_k_start") && body["top_k_start"].is_number_integer()) rq.topk_start = body["top_k_start"].get<int>();
    if (body.contains("top_k_end")   && body["top_k_end"].is_number_integer())   rq.topk_end   = body["top_k_end"].get<int>();
    if (body.contains("top_k_tail_correction") && body["top_k_tail_correction"].is_boolean()) {
        rq.topk_tail = body["top_k_tail_correction"].get<bool>();
    }
    if (body.contains("seed")  && body["seed"].is_number_integer()) rq.seed = (uint32_t) body["seed"].get<int64_t>();
    if (body.contains("ignore_eos") && body["ignore_eos"].is_boolean()) rq.ignore_eos = body["ignore_eos"].get<bool>();
    return rq;
}

// ------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string hostname = "127.0.0.1";
    int         port     = 8080;
    std::string api_key;
    bool        enable_metrics = false;
    bool        enable_slots   = false;

    // Pull server-only flags out of argv before common_params_parse (which validates against
    // LLAMA_EXAMPLE_DIFFUSION and would reject these). Everything else (-m, --mmproj, -ngl, -c,
    // --seed, --top-k*, -t, ...) is forwarded and parsed normally.
    std::vector<char *> fwd;
    fwd.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * def) -> std::string { return (i + 1 < argc) ? argv[++i] : def; };
        if (a == "--host")            { hostname = next("127.0.0.1"); }
        else if (a == "--port")       { port = atoi(next("8080").c_str()); }
        else if (a == "--api-key")    { api_key = next(""); }
        else if (a == "--metrics")    { enable_metrics = true; }
        else if (a == "--slots")      { enable_slots = true; }
        else if (a == "--no-metrics") { enable_metrics = false; }
        else if (a == "--no-slots")   { enable_slots = false; }
        else                          { fwd.push_back(argv[i]); }
    }

    common_params params;
    if (!common_params_parse((int) fwd.size(), fwd.data(), params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    SRV_INF("%s\n", "loading model");

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers >= 0 ? params.n_gpu_layers : 999;
    model_params.devices      = params.devices.data();
    model_params.use_mmap     = params.use_mmap;

    diffusion_server srv;
    srv.model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!srv.model) {
        SRV_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(srv.model)) {
        SRV_ERR("'%s' is not a diffusion model\n", params.model.path.c_str());
        llama_model_free(srv.model);
        return 1;
    }

    srv.vocab         = llama_model_get_vocab(srv.model);
    srv.n_vocab       = llama_vocab_n_tokens(srv.vocab);
    srv.canvas_length = DEF_CANVAS_LENGTH;
    srv.n_steps       = DEF_MAX_DENOISE_STEPS;
    srv.n_ub          = srv.canvas_length;
    srv.n_ctx         = params.n_ctx > 0 ? (int) params.n_ctx : 4096;
    if (srv.n_ctx < 2 * srv.canvas_length) srv.n_ctx = 2 * srv.canvas_length;
    srv.topk_fixed    = (params.sampling.user_sampling_config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K)
                            ? params.sampling.top_k
                            : 0;
    srv.topk_start    = params.diffusion.top_k_start;
    srv.topk_end      = params.diffusion.top_k_end;
    srv.topk_tail     = params.diffusion.top_k_tail_correction;
    srv.model_path    = params.model.path;
    srv.metrics.t_start = (int64_t) std::time(nullptr);

    {
        std::string p = params.model.path;
        size_t slash = p.find_last_of("/\\");
        srv.model_id = (slash == std::string::npos) ? p : p.substr(slash + 1);
        size_t dot = srv.model_id.rfind(".gguf");
        if (dot != std::string::npos) srv.model_id = srv.model_id.substr(0, dot);
        if (srv.model_id.empty()) srv.model_id = "diffusion-gemma";
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = srv.n_ctx;
    ctx_params.n_batch  = srv.n_ub;
    ctx_params.n_ubatch = srv.n_ub;
    ctx_params.no_perf  = params.no_perf;

    srv.ctx = llama_init_from_model(srv.model, ctx_params);
    if (!srv.ctx) {
        SRV_ERR("%s\n", "failed to create context");
        llama_model_free(srv.model);
        return 1;
    }
    llama_set_n_threads(srv.ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    srv.mem   = llama_get_memory(srv.ctx);
    srv.batch = llama_batch_init(srv.n_ub, 0, 1);
    srv.templates = common_chat_templates_init(srv.model, "");

    const int topk_max_requested =
        (srv.topk_start > 0 && srv.topk_end > 0) ? std::max(srv.topk_start, srv.topk_end) : srv.topk_fixed;
    const bool gpu_sampling_requested = env_int("DG_GPU_SAMPLING", 1) != 0;
    srv.use_gpu_sampling = gpu_sampling_requested &&
                           llama_diffusion_sample_topk_supported(srv.ctx);
    srv.use_device_self_cond = srv.use_gpu_sampling && env_int("DG_DEVICE_SELFCOND", 1) != 0;
    srv.use_device_loop = srv.use_device_self_cond && env_int("DG_DEVICE_LOOP", 1) != 0;
    llama_set_diffusion_gpu_sampling(srv.ctx, srv.use_gpu_sampling);
    const bool default_topk_gpu_ok = topk_max_requested <= 0 || topk_max_requested <= GPU_SAMPLING_MAX_TOP_K;

    const uint32_t default_seed = params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed;

    SRV_INF("%s\n", llama_print_system_info());
    SRV_INF("model loaded: '%s' | n_ctx = %d | canvas = %d | denoise steps = %d | 1 slot\n",
            srv.model_id.c_str(), srv.n_ctx, srv.canvas_length, srv.n_steps);
    SRV_INF("gpu sampling: %s%s%s%s | top-k fixed=%d anneal=[%d->%d] tail_correction=%d\n",
            srv.use_gpu_sampling ? "on" : "off",
            (!default_topk_gpu_ok ? " (default top-k will use CPU fallback until k <= CUDA limit)" :
             (!gpu_sampling_requested ? " (disabled by DG_GPU_SAMPLING=0)" : "")),
            srv.use_device_self_cond ? " | device self-cond: on" : "",
            srv.use_device_loop ? " | device loop: on" : "",
            srv.topk_fixed, srv.topk_start, srv.topk_end, srv.topk_tail ? 1 : 0);

    // ------------------------------------------------------------------------------------------
    // HTTP server
    // ------------------------------------------------------------------------------------------
    httplib::Server http;
    http.set_default_headers({
        {"Server", "llama.cpp-diffusion-gemma"},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });

    // access log (llama-server: SRV_TRC "done request: METHOD PATH ADDR STATUS"; skip noisy paths)
    http.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        if (req.path == "/health" || req.path == "/v1/health" || req.path == "/metrics" ||
            req.path == "/props"  || req.path == "/models"    || req.path == "/v1/models") {
            return;
        }
        SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
        SRV_DBG("request body: %s\n", req.body.c_str());
    });

    if (!api_key.empty()) {
        http.set_pre_routing_handler([api_key](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "OPTIONS" || req.path == "/health" || req.path == "/v1/health") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            if (req.get_header_value("Authorization") != "Bearer " + api_key) {
                res.status = 401;
                res.set_content(error_json("invalid api key", "authentication_error", 401).dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    http.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    auto health = [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    };
    http.Get("/health", health);
    http.Get("/v1/health", health);

    http.Get("/v1/models", [&srv](const httplib::Request &, httplib::Response & res) {
        json models = json::array();
        models.push_back({{"id", srv.model_id}, {"object", "model"}, {"created", (int64_t) std::time(nullptr)}, {"owned_by", "local"}});
        res.set_content(json{{"object", "list"}, {"data", models}}.dump(), "application/json");
    });
    http.Get("/models", [&srv](const httplib::Request &, httplib::Response & res) {
        json models = json::array();
        models.push_back({{"id", srv.model_id}, {"object", "model"}, {"created", (int64_t) std::time(nullptr)}, {"owned_by", "local"}});
        res.set_content(json{{"object", "list"}, {"data", models}}.dump(), "application/json");
    });

    // /props (llama-server analogue): server + model + default generation settings
    http.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        json props{
            {"default_generation_settings", {
                {"n_ctx", srv.n_ctx},
                {"canvas_length", srv.canvas_length},
                {"n_denoise_steps", srv.n_steps},
                {"temperature_min", TEMP_MIN},
                {"temperature_max", TEMP_MAX},
                {"entropy_bound", ENTROPY_BOUND},
                {"confidence_threshold", CONFIDENCE_THRESHOLD},
                {"stability_threshold", STABILITY_THRESHOLD},
                {"self_cond_topk", SC_K},
                {"gpu_sampling", srv.use_gpu_sampling},
                {"device_self_cond", srv.use_device_self_cond},
                {"device_loop", srv.use_device_loop},
                {"top_k", srv.topk_fixed},
                {"top_k_start", srv.topk_start},
                {"top_k_end", srv.topk_end},
                {"top_k_tail_correction", srv.topk_tail},
            }},
            {"total_slots", 1},
            {"model_path", srv.model_path},
            {"model_alias", srv.model_id},
            {"n_ctx", srv.n_ctx},
            {"build_info", srv.build_info},
            {"generation_type", "block-diffusion"},
            {"endpoint_slots", enable_slots},
            {"endpoint_metrics", enable_metrics},
        };
        res.set_content(props.dump(), "application/json");
    });

    // /metrics (Prometheus). Gated behind --metrics, like llama-server.
    if (enable_metrics) {
        http.Get("/metrics", [&srv](const httplib::Request &, httplib::Response & res) {
            server_metrics & m = srv.metrics;
            std::lock_guard<std::mutex> lk(m.mu);
            const int n_proc = m.n_processing.load();
            char buf[8192];
            int n = snprintf(buf, sizeof(buf),
                // --- counters (re-mapped to diffusion where the AR concept differs) ---
                "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
                "# TYPE llamacpp:prompt_tokens_total counter\n"
                "llamacpp:prompt_tokens_total %llu\n"
                "# HELP llamacpp:prompt_seconds_total Prompt (encoder prefill) process time.\n"
                "# TYPE llamacpp:prompt_seconds_total counter\n"
                "llamacpp:prompt_seconds_total %.3f\n"
                "# HELP llamacpp:tokens_predicted_total Number of answer tokens generated.\n"
                "# TYPE llamacpp:tokens_predicted_total counter\n"
                "llamacpp:tokens_predicted_total %llu\n"
                "# HELP llamacpp:tokens_predicted_seconds_total Denoising (generation) process time.\n"
                "# TYPE llamacpp:tokens_predicted_seconds_total counter\n"
                "llamacpp:tokens_predicted_seconds_total %.3f\n"
                "# HELP llamacpp:n_decode_total Total number of llama_decode() calls (prefill + denoise + commit).\n"
                "# TYPE llamacpp:n_decode_total counter\n"
                "llamacpp:n_decode_total %llu\n"
                "# HELP llamacpp:requests_total Total number of completed generation requests.\n"
                "# TYPE llamacpp:requests_total counter\n"
                "llamacpp:requests_total %llu\n"
                "# HELP llamacpp:diffusion_blocks_total Total number of canvas blocks denoised.\n"
                "# TYPE llamacpp:diffusion_blocks_total counter\n"
                "llamacpp:diffusion_blocks_total %llu\n"
                "# HELP llamacpp:diffusion_steps_total Total number of denoising steps.\n"
                "# TYPE llamacpp:diffusion_steps_total counter\n"
                "llamacpp:diffusion_steps_total %llu\n"
                "# HELP llamacpp:diffusion_canvas_tokens_total Total canvas tokens denoised (blocks * canvas_length).\n"
                "# TYPE llamacpp:diffusion_canvas_tokens_total counter\n"
                "llamacpp:diffusion_canvas_tokens_total %llu\n"
                // --- gauges (last request) ---
                "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.\n"
                "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
                "llamacpp:prompt_tokens_seconds %.3f\n"
                "# HELP llamacpp:predicted_tokens_seconds Average answer-token throughput in tokens/s.\n"
                "# TYPE llamacpp:predicted_tokens_seconds gauge\n"
                "llamacpp:predicted_tokens_seconds %.3f\n"
                "# HELP llamacpp:diffusion_steps_per_second Denoising steps per second (last request).\n"
                "# TYPE llamacpp:diffusion_steps_per_second gauge\n"
                "llamacpp:diffusion_steps_per_second %.3f\n"
                "# HELP llamacpp:diffusion_canvas_tokens_per_second Canvas tokens per second (last request).\n"
                "# TYPE llamacpp:diffusion_canvas_tokens_per_second gauge\n"
                "llamacpp:diffusion_canvas_tokens_per_second %.3f\n"
                "# HELP llamacpp:diffusion_ms_per_step Milliseconds per denoising step (last request).\n"
                "# TYPE llamacpp:diffusion_ms_per_step gauge\n"
                "llamacpp:diffusion_ms_per_step %.3f\n"
                "# HELP llamacpp:requests_processing Number of requests currently generating.\n"
                "# TYPE llamacpp:requests_processing gauge\n"
                "llamacpp:requests_processing %d\n",
                (unsigned long long) m.n_prompt_tokens_total,
                m.t_prompt_ms_total / 1000.0,
                (unsigned long long) m.n_tokens_predicted_total,
                m.t_predicted_ms_total / 1000.0,
                (unsigned long long) m.n_decode_total,
                (unsigned long long) m.n_requests_total,
                (unsigned long long) m.n_blocks_total,
                (unsigned long long) m.n_steps_total,
                (unsigned long long) m.n_canvas_tokens_total,
                m.last_prompt_tps,
                m.last_predicted_tps,
                m.last_steps_per_second,
                m.last_canvas_tps,
                m.last_ms_per_step,
                n_proc);
            res.set_header("Process-Start-Time-Unix", std::to_string(m.t_start));
            res.set_content(std::string(buf, n > 0 ? (size_t) n : 0), "text/plain; version=0.0.4");
        });
        SRV_INF("%s\n", "metrics endpoint enabled at /metrics");
    }

    // /slots (single slot). Gated behind --slots, like llama-server.
    if (enable_slots) {
        http.Get("/slots", [&srv](const httplib::Request &, httplib::Response & res) {
            json slot{
                {"id", 0},
                {"n_ctx", srv.n_ctx},
                {"is_processing", srv.metrics.n_processing.load() > 0},
                {"generation_type", "block-diffusion"},
                {"params", {
                    {"canvas_length", srv.canvas_length},
                    {"n_denoise_steps", srv.n_steps},
                    {"self_cond_topk", SC_K},
                    {"gpu_sampling", srv.use_gpu_sampling},
                    {"device_self_cond", srv.use_device_self_cond},
                    {"device_loop", srv.use_device_loop},
                    {"top_k", srv.topk_fixed},
                    {"top_k_start", srv.topk_start},
                    {"top_k_end", srv.topk_end},
                    {"top_k_tail_correction", srv.topk_tail},
                }},
            };
            res.set_content(json::array({slot}).dump(), "application/json");
        });
        SRV_INF("%s\n", "slots endpoint enabled at /slots");
    }

    // shared: run one generation (under the slot mutex), update metrics, print timing log
    auto run_for_body = [&srv, default_seed](const json & body, const std::vector<llama_token> & prompt_tokens) {
        diffusion_request rq = request_from_body(body, srv, default_seed);
        srv.metrics.n_processing.fetch_add(1);
        std::unique_lock<std::mutex> lock(srv.gen_mutex);
        diffusion_result r = srv.generate(prompt_tokens, rq);
        lock.unlock();
        srv.metrics.n_processing.fetch_sub(1);
        if (r.ok) { srv.metrics.add(r); log_timings(r); }
        return r;
    };

    // POST /v1/chat/completions
    http.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400; res.set_content(error_json("'messages' (array) is required", "invalid_request_error", 400).dump(), "application/json"); return;
        }
        std::string prompt;
        try { prompt = srv.format_messages(body["messages"]); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("failed to format messages: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        const std::vector<llama_token> prompt_tokens = common_tokenize(srv.vocab, prompt, false, true);
        const bool stream = body.value("stream", false);

        const diffusion_result r = run_for_body(body, prompt_tokens);
        if (!r.ok) { res.status = 500; res.set_content(error_json(r.error, "server_error", 500).dump(), "application/json"); return; }

        if (!stream) {
            json out{
                {"id", gen_id("chatcmpl")}, {"object", "chat.completion"}, {"created", (int64_t) std::time(nullptr)}, {"model", srv.model_id},
                {"choices", json::array({ json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", r.answer}}}, {"finish_reason", "stop"}} })},
                {"usage", usage_json(r)},
                {"timings", timings_json(r)},
            };
            res.set_content(out.dump(), "application/json");
            return;
        }

        const std::string id = gen_id("chatcmpl");
        const int64_t created = (int64_t) std::time(nullptr);
        const std::string model_id = srv.model_id;
        const std::string answer = r.answer;
        const json timings = timings_json(r);
        res.set_chunked_content_provider("text/event-stream",
            [id, created, model_id, answer, timings](size_t, httplib::DataSink & sink) {
                auto send = [&](const json & c) { std::string s = "data: " + c.dump() + "\n\n"; return sink.write(s.data(), s.size()); };
                auto chunk = [&](const json & delta, const char * finish, const json * extra) {
                    json c{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_id},
                           {"choices", json::array({ json{{"index", 0}, {"delta", delta}, {"finish_reason", finish ? json(finish) : json(nullptr)}} })}};
                    if (extra) c["timings"] = *extra;
                    return send(c);
                };
                if (!chunk(json{{"role", "assistant"}}, nullptr, nullptr)) return false;
                if (!chunk(json{{"content", answer}}, nullptr, nullptr))   return false;
                if (!chunk(json::object(), "stop", &timings))              return false;
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    // POST /v1/completions (plain prompt)
    http.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            res.status = 400; res.set_content(error_json("'prompt' (string) is required", "invalid_request_error", 400).dump(), "application/json"); return;
        }
        const std::string prompt = body["prompt"].get<std::string>();
        const std::vector<llama_token> prompt_tokens = common_tokenize(srv.vocab, prompt, true, true);

        const diffusion_result r = run_for_body(body, prompt_tokens);
        if (!r.ok) { res.status = 500; res.set_content(error_json(r.error, "server_error", 500).dump(), "application/json"); return; }
        json out{
            {"id", gen_id("cmpl")}, {"object", "text_completion"}, {"created", (int64_t) std::time(nullptr)}, {"model", srv.model_id},
            {"choices", json::array({ json{{"index", 0}, {"text", r.answer}, {"finish_reason", "stop"}} })},
            {"usage", usage_json(r)},
            {"timings", timings_json(r)},
        };
        res.set_content(out.dump(), "application/json");
    });

    SRV_INF("server is listening on http://%s:%d\n", hostname.c_str(), port);
    SRV_INF("%s\n", "all slots are idle");
    if (!http.listen(hostname, port)) {
        SRV_ERR("failed to bind to %s:%d\n", hostname.c_str(), port);
        llama_batch_free(srv.batch);
        llama_free(srv.ctx);
        llama_model_free(srv.model);
        llama_backend_free();
        return 1;
    }

    llama_batch_free(srv.batch);
    llama_free(srv.ctx);
    llama_model_free(srv.model);
    llama_backend_free();
    return 0;
}

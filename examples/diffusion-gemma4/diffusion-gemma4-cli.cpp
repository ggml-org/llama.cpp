// Block-diffusion generation for diffusion_gemma4.
//
// Implements the reference block-diffusion loop (EntropyBoundSampler + StableAndConfident
// stopping + linear temperature schedule) over a single canvas, driving the bidirectional
// no-KV-cache graph.
//
// SCOPE (first runnable version): unconditioned generation (no prompt context) with
// self-conditioning = 0. The graph currently feeds zero soft-embeddings, so this matches
// the verified first-denoising-step behaviour every step. The self-conditioning feedback
// (softmax(prev_logits) @ embed, via a soft-embeddings input channel) and prompt/encoder-KV
// conditioning are added on top of this loop next; until then output quality is limited and
// the prompt is not yet used.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// reference defaults from generation_config.json / DiffusionGemma4GenerationConfig
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
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    // diffusion config (env-overridable for quick CPU testing)
    const int   canvas_length = env_int("DG4_CANVAS", params.n_predict > 0 ? params.n_predict : DEF_CANVAS_LENGTH);
    const int   n_steps       = env_int("DG4_STEPS", DEF_MAX_DENOISE_STEPS);
    const float entropy_bound = ENTROPY_BOUND;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
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

    // tokenize the prompt (causal prefix). The canvas is appended after it.
    // this is a chat-trained model, so apply its chat template (turn/channel tokens).
    std::vector<llama_token> prompt_tokens;
    if (!params.prompt.empty()) {
        const std::string formatted = format_chat(model, params.prompt);
        LOG_INF("formatted prompt: %s\n", formatted.c_str());
        prompt_tokens = common_tokenize(vocab, formatted, /*add_special*/ false, /*parse_special*/ true);
    }
    const int n_prompt = (int) prompt_tokens.size();
    const int n_seq    = n_prompt + canvas_length; // [prompt ; canvas]

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_seq;
    ctx_params.n_batch  = n_seq;
    ctx_params.n_ubatch = n_seq; // whole sequence in one ubatch -> full (prefix) attention
    ctx_params.no_perf  = params.no_perf;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    llama_set_diffusion_prompt_len(ctx, n_prompt);

    LOG_INF("diffusion-gemma4: prompt=%d canvas=%d steps=%d entropy_bound=%.3f temp=[%.2f,%.2f]\n",
            n_prompt, canvas_length, n_steps, entropy_bound, TEMP_MIN, TEMP_MAX);

    std::mt19937 rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed);
    std::uniform_int_distribution<int> rand_tok(0, n_vocab - 1);
    std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);

    // 1. initialize canvas with random tokens
    std::vector<llama_token> canvas(canvas_length);
    for (auto & t : canvas) t = rand_tok(rng);

    llama_batch batch = llama_batch_init(n_seq, 0, 1);

    std::vector<llama_token> argmax_canvas(canvas_length, -1);
    std::vector<llama_token> prev_argmax(canvas_length, -1);
    std::vector<llama_token> accepted = canvas;

    // self-conditioning buffer over the full [prompt ; canvas] sequence: softmax(processed_logits)
    // of the previous step in the canvas columns (prompt columns stay zero). Fed to the next decode.
    std::vector<float> self_cond_buf((size_t) n_vocab * n_seq, 0.0f);
    llama_set_diffusion_self_cond(ctx, nullptr, 0, 0); // first step: zero self-conditioning

    // 2. denoising loop: cur_step = n_steps .. 1
    for (int cur_step = n_steps; cur_step >= 1; --cur_step) {
        // 2a. decode [prompt ; canvas]. Request logits for ALL tokens: in the no-KV-cache path
        // llama.cpp prunes tokens that are not outputs, which would drop the prompt from the
        // attention and stop the canvas from attending to it. We only read the canvas rows.
        batch.n_tokens = n_seq;
        for (int i = 0; i < n_seq; ++i) {
            batch.token[i]     = (i < n_prompt) ? prompt_tokens[i] : canvas[i - n_prompt];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: llama_decode failed at step %d\n", cur_step);
            break;
        }
        // canvas logits start at row n_prompt (prompt rows precede them)
        const float * logits = llama_get_logits(ctx) + (size_t) n_prompt * n_vocab;

        // 2b. linear temperature schedule: t = t_min + (t_max - t_min) * (cur_step / n_steps)
        const float temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * ((float) cur_step / (float) n_steps);

        std::vector<float> entropy(canvas_length);
        std::vector<llama_token> sampled(canvas_length);
        std::vector<float> probs(n_vocab);

        for (int j = 0; j < canvas_length; ++j) {
            const float * lg = logits + (size_t) j * n_vocab;
            // softmax of processed (temperature-scaled) logits
            float maxl = -INFINITY;
            int   amax = 0;
            for (int v = 0; v < n_vocab; ++v) {
                const float x = lg[v] / temp;
                if (x > maxl) { maxl = x; amax = v; }
            }
            float sum = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                const float p = expf(lg[v] / temp - maxl);
                probs[v] = p;
                sum += p;
            }
            // entropy + multinomial sample
            float ent = 0.0f;
            const float r = rand_unif(rng) * sum;
            float cum = 0.0f;
            int   tok = amax;
            bool  picked = false;
            float * sc = self_cond_buf.data() + (size_t) (n_prompt + j) * n_vocab; // canvas column
            for (int v = 0; v < n_vocab; ++v) {
                const float p = probs[v] / sum;
                sc[v] = p;                              // store normalized softmax for self-conditioning
                if (p > 0.0f) ent -= p * logf(p);
                cum += probs[v];
                if (!picked && cum >= r) { tok = v; picked = true; }
            }
            entropy[j]       = ent;
            sampled[j]       = tok;
            argmax_canvas[j] = amax;
        }

        // self-conditioning for the NEXT denoising step: feed this step's softmax(processed_logits)
        llama_set_diffusion_self_cond(ctx, self_cond_buf.data(), n_vocab, n_seq);

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
        LOG_INF("step %3d  temp=%.3f  accepted=%4d/%d  mean_entropy=%.4f%s\n",
                cur_step, temp, n_accept, canvas_length, mean_ent,
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

    llama_batch_free(batch);

    std::string out = common_detokenize(vocab, accepted, false);
    LOG_INF("\n=== generated canvas ===\n%s\n", out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

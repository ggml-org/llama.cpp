// Persistent "visual" generation server for DiffusionGemma: load the GGUF once, then run the OPTIMIZED
// entropy-bound decoder (the same diffusion_generate_entropy_bound the CLI's --diffusion-visual uses)
// and stream the per-step argmax canvas back so a UI can watch the denoise resolve in place. Unlike the
// raw-logits server, NO [C, n_vocab] logits are shipped to the client and no host-side sampling happens:
// argmax/entropy/multinomial and self-conditioning stay on the GPU (Stage 1 + Stage 2).
//
// Tokenization, chat templating and detokenization all happen here, from the GGUF's own embedded tokenizer
// + chat template (same path as llama-diffusion-cli), so the client needs no tokenizer files of its own.
//
// Protocol (synchronous, one request per line on stdin):
//   stdin  : a line containing a request-file path R
//   file R : UTF-8 JSON  {"seed": <int>, "n_blocks": <int>, "messages": [ {"role","content"}, ... ]}
//            (messages are OpenAI chat-completion format; the GGUF chat template is applied here)
//   stdout : a stream of newline records, then "DONE":
//              F <block> <step> <total> <json-string>   one per denoising step (current canvas, decoded)
//              C <block> <json-string>                  cumulative committed answer text after this block
//              DONE                                      end of this request
//              ERR <msg>                                request failed
//   "QUIT"/EOF -> exit.
//
// Usage: llama-diffusion-gemma-visual-server <model.gguf>   (env NGL for gpu layers, MAXTOK, FA for flash-attn)

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "../diffusion/diffusion.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string read_text_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s;
    if (sz > 0) {
        s.resize((size_t) sz);
        if (fread(&s[0], 1, (size_t) sz, f) != (size_t) sz) { fclose(f); return {}; }
    }
    fclose(f);
    return s;
}

// per-request callback state: which block we are on, where to stream frames, and how to decode them
struct vis_cb_data {
    int                  block   = 0;
    int                  n_input = 0;
    FILE *               out     = nullptr;
    const llama_vocab *  vocab   = nullptr;
};

// Stream the current argmax canvas (tokens[n_input .. n_tokens)) as one "F" record per denoising step,
// decoded to text and JSON-escaped so it survives the line protocol intact (spaces/newlines/unicode).
static bool vis_step_callback(int32_t step, int32_t total_steps, const llama_token * tokens,
                              int32_t n_tokens, void * user_data) {
    auto * d = (vis_cb_data *) user_data;
    const int n_canvas = n_tokens - d->n_input;
    if (n_canvas <= 0) return true;
    std::vector<llama_token> canvas(tokens + d->n_input, tokens + n_tokens);
    const std::string text = common_detokenize(d->vocab, canvas, /*special*/ false);
    fprintf(d->out, "F %d %d %d %s\n", d->block, step, total_steps, nlohmann::json(text).dump().c_str());
    fflush(d->out);
    return true;
}

// Trim a denoised canvas like the CLI: cut at the first end-of-generation token, else at the onset of a
// repetition loop (a token recurring at stride 1-2 for >= 6 steps).
static size_t trim_canvas(const llama_vocab * vocab, const llama_token * canvas, size_t n) {
    size_t cut = n;
    for (size_t i = 0; i < n; i++) {
        if (llama_vocab_is_eog(vocab, canvas[i])) { cut = i; break; }
    }
    for (size_t i = 0; i + 1 < cut; i++) {
        bool loop = false;
        for (size_t stride = 1; stride <= 2 && !loop; stride++) {
            size_t reps = 0;
            for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) { reps++; }
            loop = reps >= 6;
        }
        if (loop) { cut = i; break; }
    }
    return cut;
}

static float meta_f(llama_model * m, const char * key, float def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
}
static int32_t meta_i(llama_model * m, const char * key, int32_t def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? (int32_t) strtol(buf, nullptr, 10) : def;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const int MAXTOK = atoi(getenv("MAXTOK") ? getenv("MAXTOK") : "8192");

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = atoi(getenv("NGL") ? getenv("NGL") : "0");
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    if (!llama_model_is_diffusion(model)) { fprintf(stderr, "not a diffusion model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // chat template + tokenizer come from the GGUF itself (same as the CLI): no client-side tokenizer needed
    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, "");

    int64_t canvas_length = 0;
    {
        char canvas_str[32];
        if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
            canvas_length = strtol(canvas_str, nullptr, 10);
        }
    }
    if (canvas_length <= 0) { fprintf(stderr, "model has no diffusion.canvas_length\n"); return 1; }

    // Enable the self-conditioning graph before context creation so the reserve sizes the compute buffer
    // (matches the CLI). The entropy-bound decoder supplies the real SC state per step.
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = MAXTOK;
    cparams.n_batch  = MAXTOK;
    cparams.n_ubatch = MAXTOK;   // non-causal: the whole [prompt | canvas] must fit one ubatch
    cparams.no_perf  = true;
    cparams.flash_attn_type = getenv("FA") && atoi(getenv("FA"))
                                ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_set_causal_attn(ctx, false);

    // entropy-bound params from GGUF metadata + reference defaults (kept in sync with the CLI)
    diffusion_eb_params base;
    base.max_denoising_steps  = meta_i(model, "diffusion.eb_max_steps", 48);
    base.t_min                = meta_f(model, "diffusion.eb_t_min", 0.4f);
    base.t_max                = meta_f(model, "diffusion.eb_t_max", 0.8f);
    base.entropy_bound        = meta_f(model, "diffusion.eb_entropy_bound", 0.1f);
    base.stability_threshold  = meta_i(model, "diffusion.eb_stability_threshold", 1);
    base.confidence_threshold = meta_f(model, "diffusion.eb_confidence_threshold", 0.005f);

    // Stage 1 + Stage 2 are single-device features (sc_dev / prompt-KV store are single-GPU). Auto-enable
    // them for one CUDA device, exactly like the CLI's --diffusion-* auto resolution.
    int gpu_devs = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const auto dt = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) { gpu_devs++; }
    }
    const bool one_gpu = (gpu_devs <= 1);
    base.kv_cache          = one_gpu;
    base.gpu_sampling      = one_gpu;
    base.gpu_sample_reduce = one_gpu;

    std::vector<llama_token> output_tokens(MAXTOK);

    fprintf(stderr, "diffusion-gemma-visual-server ready (n_vocab=%d, canvas=%d, MAXTOK=%d, NGL=%d, "
            "gpu_sampling=%s sample_reduce=%s kv_cache=%s)\n",
            n_vocab, (int) canvas_length, MAXTOK, mparams.n_gpu_layers,
            base.gpu_sampling ? "on" : "off", base.gpu_sample_reduce ? "on" : "off",
            base.kv_cache ? "on" : "off");
    printf("READY %d\n", n_vocab); fflush(stdout);

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (strcmp(line, "QUIT") == 0) break;

        // parse the request file: {"seed", "n_blocks", "messages":[...]} -> chat template -> token prefix
        int seed = 0, n_blocks = 1;
        std::vector<llama_token> prefix;
        try {
            const std::string raw = read_text_file(line);
            if (raw.empty()) { printf("ERR badreq\n"); fflush(stdout); continue; }
            const nlohmann::ordered_json req = nlohmann::ordered_json::parse(raw);
            seed     = req.value("seed", 0);
            n_blocks = req.value("n_blocks", 1);
            std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(req.at("messages"));
            common_chat_templates_inputs inputs;
            inputs.messages              = messages;
            inputs.add_generation_prompt = true;
            const std::string prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            prefix = common_tokenize(vocab, prompt, /*add special*/ true, /*parse special*/ true);
        } catch (const std::exception & e) {
            printf("ERR parse %s\n", e.what()); fflush(stdout); continue;
        }
        if (prefix.empty()) { printf("ERR emptyprompt\n"); fflush(stdout); continue; }

        const int P = (int) prefix.size();          // original prompt length; the answer is what grows past it
        std::vector<llama_token> answer;             // cumulative committed canvas tokens (across blocks)

        for (int b = 0; b < std::max(1, n_blocks); b++) {
            const int32_t prefix_len = (int32_t) prefix.size();
            const int32_t max_length = prefix_len + (int32_t) canvas_length;
            if (max_length > MAXTOK) { printf("ERR toolong %d\n", (int) max_length); break; }

            diffusion_eb_params eb = base;
            eb.max_length              = max_length;
            eb.seed                    = seed + b;   // distinct per block, deterministic from the request seed
            eb.visual_mode             = true;
            vis_cb_data cb{ b, prefix_len, stdout, vocab };
            eb.step_callback           = vis_step_callback;
            eb.step_callback_user_data = &cb;

            int32_t n_generated = 0;
            diffusion_generate_entropy_bound(ctx, prefix.data(), output_tokens.data(), prefix_len, eb, n_generated);
            if (n_generated <= prefix_len) { if (b == 0) printf("ERR gen\n"); break; }

            const llama_token * canvas = output_tokens.data() + prefix_len;
            const size_t cut = trim_canvas(vocab, canvas, (size_t) canvas_length);

            answer.insert(answer.end(), canvas, canvas + cut);
            const std::string answer_text = common_detokenize(vocab, answer, /*special*/ false);
            printf("C %d %s\n", b, nlohmann::json(answer_text).dump().c_str()); fflush(stdout);

            if (cut < (size_t) canvas_length) break;                 // eog / repetition loop: answer complete
            prefix.insert(prefix.end(), canvas, canvas + cut);       // commit the block, denoise the next
        }
        (void) P;
        printf("DONE\n"); fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

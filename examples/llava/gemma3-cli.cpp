#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "stb_image.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"

#include <vector>
#include <limits.h>
#include <inttypes.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static bool g_is_generating = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying Gemma 3 vision capabilities.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for using Gemma 3 vision model\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  --image and -p are optional, if NOT provided, the CLI will run in chat mode\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            LOG("\nInterrupted by user\n");
            _exit(130);
        }
    }
}
#endif

struct gemma3_context {
    struct clip_ctx    * ctx_clip = NULL;
    common_init_result   llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    llama_batch         batch;

    int n_threads    = 1;
    llama_pos n_past = 0;

    gemma3_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(params.n_batch, 0, 1);
        init_clip_model(params);
    }

    void init_clip_model(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        ctx_clip = clip_model_load(clip_path, params.verbosity > 1);
    }

    ~gemma3_context() {
        clip_free(ctx_clip);
    }
};

struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

static int eval_text(gemma3_context & ctx, std::string input, bool logits_last = false) {
    llama_tokens tokens = common_tokenize(ctx.lctx, input, false, true);
    common_batch_clear(ctx.batch);
    for (llama_token & t : tokens) {
        common_batch_add(ctx.batch, t, ctx.n_past++, {0}, false);
    }
    if (logits_last) {
        ctx.batch.logits[ctx.batch.n_tokens - 1] = true;
    }
    // LOG("eval_text (n_tokens = %d): %s\n", (int)tokens.size(), input.c_str());
    if (llama_decode(ctx.lctx, ctx.batch)) {
        LOG_ERR("Failed to decode text\n");
        return 1;
    }
    return 0;
}

static int eval_image(gemma3_context & ctx, std::string & fname) {
    std::vector<float> image_embd_v;
    int n_embd = llama_model_n_embd(ctx.model);
    int n_tokens = 256;
    image_embd_v.resize(n_tokens * n_embd);

    bool ok;
    struct clip_image_u8 * img_u8 = clip_image_u8_init();
    ok = clip_image_load_from_file(fname.c_str(), img_u8);
    if (!ok) {
        LOG_ERR("Unable to load image %s\n", fname.c_str());
        clip_image_u8_free(img_u8);
        return 2; // non-fatal error
    }

    clip_image_f32_batch batch_f32;
    ok = clip_image_preprocess(ctx.ctx_clip, img_u8, &batch_f32);
    if (!ok) {
        LOG_ERR("Unable to preprocess image\n");
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return 1;
    }

    int64_t t0 = ggml_time_ms();
    LOG("Encoding image %s\n", fname.c_str());
    ok = clip_image_batch_encode(ctx.ctx_clip, ctx.n_threads, &batch_f32, image_embd_v.data());
    if (!ok) {
        LOG_ERR("Unable to encode image\n");
        clip_image_f32_batch_free(&batch_f32);
        clip_image_u8_free(img_u8);
        return 1;
    }
    LOG("Image encoded in %" PRId64 " ms\n", ggml_time_ms() - t0);

    clip_image_f32_batch_free(&batch_f32);
    clip_image_u8_free(img_u8);

    // decode image embeddings
    int64_t t1 = ggml_time_ms();
    eval_text(ctx, "<start_of_image>");
    llama_set_causal_attn(ctx.lctx, false);
    decode_embd_batch batch_img(image_embd_v.data(), n_tokens, ctx.n_past, 0);
    if (llama_decode(ctx.lctx, batch_img.batch)) {
        LOG_ERR("failed to decode image\n");
        return 1;
    }
    ctx.n_past += n_tokens;
    llama_set_causal_attn(ctx.lctx, true);
    eval_text(ctx, "<end_of_image>");
    LOG("Image decoded in %" PRId64 " ms\n", ggml_time_ms() - t1);
    return 0;
}

static int generate_response(gemma3_context & ctx, common_sampler * smpl, int n_predict) {
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating) {
            printf("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(smpl, ctx.lctx, -1);
        common_sampler_accept(smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            printf("\n");
            break; // end of generation
        }

        printf("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;
    params.sampling.temp = 0.2; // lower temp by default for better quality

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        return 1;
    }

    gemma3_context ctx(params);
    printf("%s: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    struct common_sampler * smpl = common_sampler_init(ctx.model, params.sampling);
    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (eval_text(ctx, "<bos>")) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (eval_text(ctx, "<start_of_turn>user\n")) {
            return 1;
        }
        for (auto & fname : params.image) {
            if (eval_image(ctx, fname)) {
                return 1;
            }
        }
        if (eval_text(ctx, params.prompt + "<end_of_turn><start_of_turn>model\n", true)) {
            return 1;
        }
        if (generate_response(ctx, smpl, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        LOG("\n   /image <path>    load an image");
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        if (eval_text(ctx, "<start_of_turn>user\n")) {
            return 1;
        }

        while (true) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(console::user_input);
            std::string line;
            console::readline(line, false);
            console::set_display(console::reset);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                llama_kv_self_seq_rm(ctx.lctx, 0, 1, -1); // keep BOS
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            if (line.find("/image") == 0) {
                std::string image = line.substr(7);
                int res = eval_image(ctx, image);
                if (res == 2) {
                    continue; // image not found
                }
                if (res) {
                    return 1;
                }
                continue;
            }
            if (eval_text(ctx, line + "<end_of_turn><start_of_turn>model\n", true)) {
                return 1;
            }
            if (generate_response(ctx, smpl, n_predict)) {
                return 1;
            }
            if (eval_text(ctx, "<end_of_turn><start_of_turn>user\n")) {
                return 1;
            }
        }
    }

    return 0;
}

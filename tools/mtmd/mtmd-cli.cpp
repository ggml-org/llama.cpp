#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-cli-audio.h"

#include <chrono>
#include <thread>
#include <vector>
#include <limits.h>
#include <cinttypes>
#include <clocale>

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

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready binary.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand. Do not add unnecessary complexity. The goal is to have a simple CLI for testing multimodal support.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  --realtime starts a live audio session with the microphone, for models that take an audio stream\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
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
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

// this is only used by tests.sh to capture the response ; it's not meant to be used in production
static void inject_test_response_marker() {
    const char * env = std::getenv("MTMD_TEST_RESPONSE_MARKER");
    if (env) {
        LOG("%s\n", env);
    }
}

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    mtmd_helper_cap   mtmd_cap = {};
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    common_batch        batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper::video_ptr> videos;

    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    std::string video_ffmpeg_bin_dir;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    common_debug_cb_user_data cb_data;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        if (!model || !lctx) {
            exit(1);
        }
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = common_batch(lctx); // batch for next token generation
        n_batch = params.n_batch;

        init_vision_context(params);

        mtmd_cap = mtmd_helper_get_capabilities(model, ctx_vision.get());
        // a realtime transcription model has no chat, it streams the audio below
        if (!mtmd_cap.chat && !mtmd_cap.audio_stream) {
            LOG_ERR("Model does not support chat mode\n");
            LOG_ERR("Hint: for TTS models, please use llama-tts\n");
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * mmproj_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.device           = params.mmproj_device;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        {
            // non-causal models need the whole image in one ubatch
            const int n_ubatch = llama_n_ubatch(lctx);
            auto mem = mtmd_get_memory_usage(mmproj_path, mparams);
            if (mem.use_non_causal && mem.image_max_tokens > n_ubatch) {
                LOG_WRN("%s: cap image_max_tokens (original=%d) to n_ubatch (%d) because model needs non-causal attention on image\n", __func__, mem.image_max_tokens, n_ubatch);
                LOG_WRN("%s: increase n_ubatch (-ub) to increase vision token budget\n", __func__);
                mparams.image_max_tokens = n_ubatch;
                mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_ubatch);
            }
        }
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval;
        }
        ctx_vision.reset(mtmd_init_from_file(mmproj_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", mmproj_path);
            exit(1);
        }

        video_ffmpeg_bin_dir = params.video_ffmpeg_bin_dir;
        init_opt.video_params.fps_target = params.video_fps;
        init_opt.video_params.timestamp_interval_ms = params.video_timestamp_interval_ms;
        init_opt.video_params.ffmpeg_bin_dir = video_ffmpeg_bin_dir.empty()
                            ? nullptr : video_ffmpeg_bin_dir.c_str();
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        auto res = mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str(), false, init_opt);
        if (!res.bitmap) {
            return false;
        }
        bitmaps.entries.emplace_back(res.bitmap);
        if (res.video_ctx) {
            videos.emplace_back(res.video_ctx);
        }
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break; // end of generation
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // eval the token
        ctx.batch.clear();
        ctx.batch.add(token_id, ctx.n_past++, 0, true);
        if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

// the audio of a realtime session: the microphone, or one audio file
struct mtmd_cli_audio_input {
    mtmd::bitmap      bmp; // owns the samples of a file
    mtmd_cli_audio_in in;
    int               rate = 0;

    bool init(mtmd_cli_context & ctx, const common_params & params) {
        rate = mtmd_get_audio_sample_rate(ctx.ctx_vision.get());
        if (params.image.empty() && params.realtime) {
            return in.init_mic(rate);
        }
        if (params.image.size() != 1) {
            LOG_ERR("This model takes an audio stream, pass exactly one --audio file%s\n", params.realtime ? "" : " or --realtime");
            return false;
        }
        const std::string & fname = params.image[0];
        auto res = mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), fname.c_str(), false, ctx.init_opt);
        if (!res.bitmap) {
            return false; // error is already printed by libmtmd
        }
        bmp.ptr.reset(res.bitmap);
        if (!mtmd_bitmap_is_audio(bmp.ptr.get())) {
            LOG_ERR("%s: %s is not an audio file\n", __func__, fname.c_str());
            return false;
        }
        const size_t n_chunk = (size_t) rate * mtmd_audio_stream_params_default().step_ms / 1000;
        in.init_samples((const float *) bmp.data(), bmp.n_bytes() / sizeof(float), rate, params.realtime, n_chunk);
        return true;
    }
};

static void wait_for_audio() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// decode the rows that the audio stream has ready, returns their count or -1 on error
static int32_t decode_audio_rows(mtmd_cli_context & ctx, mtmd_audio_stream * stream, llama_token last_token, bool logits_last) {
    const int32_t n = mtmd_audio_stream_step(stream, last_token);
    if (n == -2) {
        LOG_ERR("%s: stream error\n", __func__);
        return -1;
    }
    if (n <= 0) {
        return 0;
    }
    if (mtmd_audio_stream_is_seq_start(stream)) {
        // the model restarts its context, the text goes on in a new line
        llama_memory_clear(llama_get_memory(ctx.lctx), true);
        if (ctx.n_past > 0) {
            LOG("\n");
        }
        ctx.n_past = 0;
    }
    if (ctx.n_past + n > (llama_pos) llama_n_ctx(ctx.lctx)) {
        LOG_ERR("\n%s: the context is full, increase it with -c\n", __func__);
        return -1;
    }
    const int     n_embd = llama_model_n_embd_inp(ctx.model);
    const float * rows   = mtmd_audio_stream_get_output_embd(stream);
    ctx.batch.clear();
    for (int32_t i = 0; i < n; i++) {
        const llama_pos pos[GGML_MROPE_SECTIONS] = { ctx.n_past, ctx.n_past, ctx.n_past, 0 };
        ctx.n_past++;
        ctx.batch.add_embd({ rows + (size_t) i * n_embd, 1, (size_t) n_embd }, pos, 0, logits_last && i == n - 1);
    }
    if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
        LOG_ERR("%s: failed to decode\n", __func__);
        return -1;
    }
    return n;
}

// realtime transcription model: print the transcript as the audio comes in
static int run_transcription(mtmd_cli_context & ctx, mtmd_cli_audio_input & input, int n_predict) {
    mtmd::audio_stream_ptr stream(mtmd_audio_stream_init(ctx.ctx_vision.get(), mtmd_audio_stream_params_default()));
    if (!stream) {
        return 1;
    }

    size_t      n_fed     = 0;
    int         n_gen     = 0;
    int64_t     n_dropped = 0;
    llama_token tok       = LLAMA_TOKEN_NULL;
    std::vector<float> buf;
    const int64_t t_start = ggml_time_us();
    while (!g_is_interrupted) {
        const int32_t n = decode_audio_rows(ctx, stream.get(), tok, true);
        if (n < 0) {
            return 1;
        }
        if (n == 0) {
            buf.clear();
            if (input.in.read(buf) > 0) {
                mtmd_audio_stream_push(stream.get(), buf.data(), buf.size());
                n_fed += buf.size();
            } else if (!input.in.is_end()) {
                wait_for_audio();
            } else if (mtmd_audio_stream_flush(stream.get(), true) != 0) {
                break; // all the audio is transcribed
            }
            if (input.in.n_dropped() > n_dropped) {
                n_dropped = input.in.n_dropped();
                LOG_WRN("\n%s: the model is slower than the audio, %.1f s lost so far\n", __func__, (double) n_dropped / input.rate);
            }
            continue;
        }

        tok = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        common_sampler_accept(ctx.smpl, tok, true);
        if (llama_vocab_is_eog(ctx.vocab, tok)) {
            break;
        }
        // the model emits pad and word-boundary tokens between words
        if (!llama_vocab_is_control(ctx.vocab, tok)) {
            LOG("%s", common_token_to_piece(ctx.lctx, tok).c_str());
            fflush(stdout);
        }
        if (++n_gen >= n_predict) {
            break;
        }
    }

    const double t_total = (ggml_time_us() - t_start) / 1e6;
    const double t_audio = (double) n_fed / input.rate;
    LOG("\n\n%s: %d tokens, %.2f s audio in %.2f s (%.2fx realtime)\n", __func__, n_gen, t_audio, t_total, t_audio / t_total);
    llama_perf_context_print(ctx.lctx);

    return 0;
}

// the text around the turns of a conversation, taken from the chat template with placeholder messages
// (same method as the realtime API of llama-server)
struct mtmd_cli_turn_text {
    std::string first_header; // system prompt + first user header
    std::string user_header;
    std::string user_footer;
    std::string assistant_header;
    std::string assistant_footer;

    bool init(mtmd_cli_context & ctx, const common_params & params) {
        const std::string user  = "@@USER@@";
        const std::string reply = "@@REPLY@@";
        auto render = [&](std::vector<std::string> roles, bool add_generation_prompt) {
            common_chat_templates_inputs inputs;
            if (!params.system_prompt.empty()) {
                roles.insert(roles.begin(), "system");
            }
            for (const auto & role : roles) {
                common_chat_msg msg;
                msg.role    = role;
                msg.content = role == "system" ? params.system_prompt : role == "user" ? user : reply;
                inputs.messages.push_back(std::move(msg));
            }
            inputs.add_generation_prompt = add_generation_prompt;
            inputs.use_jinja             = ctx.use_jinja;
            inputs.enable_thinking       = false;
            inputs.chat_template_kwargs  = params.default_template_kwargs;
            return common_chat_templates_apply(ctx.tmpls.get(), inputs).prompt.str();
        };
        std::string a, g, c, d;
        try {
            a = render({"user"}, false);
            g = render({"user"}, true);
            c = render({"user", "assistant"}, false);
            d = render({"user", "assistant", "user"}, false);
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            return false;
        }

        const size_t pa = a.find(user);
        const size_t pc = c.find(reply);
        const size_t pd = d.find(user, c.size());
        if (pa == std::string::npos || pc == std::string::npos || pd == std::string::npos ||
            c.compare(0, a.size(), a) != 0 || d.compare(0, c.size(), c) != 0 || g.compare(0, a.size(), a) != 0 || pc < a.size()) {
            return false;
        }
        first_header     = a.substr(0, pa);
        user_footer      = a.substr(pa + user.size());
        assistant_header = g.substr(a.size());
        assistant_footer = c.substr(pc + reply.size());
        user_header      = d.substr(c.size(), pd - c.size());
        return true;
    }
};

static int decode_text(mtmd_cli_context & ctx, const std::string & text, bool logits_last) {
    const llama_tokens toks = common_tokenize(ctx.lctx, text, false, true);
    if (ctx.n_past + (llama_pos) toks.size() > (llama_pos) llama_n_ctx(ctx.lctx)) {
        LOG_ERR("\n%s: the context is full, increase it with -c\n", __func__);
        return 1;
    }
    for (size_t i = 0; i < toks.size(); i += ctx.n_batch) {
        const size_t n = std::min(toks.size() - i, (size_t) ctx.n_batch);
        ctx.batch.clear();
        for (size_t j = 0; j < n; j++) {
            ctx.batch.add(toks[i + j], ctx.n_past++, 0, logits_last && i + j == toks.size() - 1);
        }
        if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return 1;
        }
    }
    return 0;
}

// response of a model that generates audio: the gen-audio pipeline continues the conversation from prompt
// the pipeline decodes on its own, n_past is updated at the end
static int generate_audio_response(mtmd_cli_context & ctx, const common_params & params, mtmd_helper::gen_audio & gen,
                                   mtmd_cli_audio_out & out, const std::string & prompt, int n_predict) {
    mtmd_helper::gen_audio::inp inp;
    inp.set_prompt(prompt);
    inp.data.n_past   = ctx.n_past;
    inp.data.seed     = params.sampling.seed;
    inp.data.stream   = true;
    inp.data.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    inp.data.out_mode = MTMD_HELPER_GEN_AUDIO_OUT_MIXED;
    if (gen.set_input(inp.get()) != 0) {
        LOG_ERR("%s: failed to start the response\n", __func__);
        return 1;
    }
    // the pipeline may bias some text tokens
    common_params_sampling sparams = params.sampling;
    for (const auto & b : gen.logit_bias()) {
        sparams.logit_bias.push_back(b);
    }
    common_sampler_ptr smpl(common_sampler_init(ctx.model, sparams));

    int32_t ret;
    while ((ret = gen.step_prompt(ctx.n_batch)) > 0) {
    }
    if (ret < 0) {
        LOG_ERR("%s: failed to start the response\n", __func__);
        return 1;
    }
    const int n_embd = llama_model_n_embd(ctx.model);
    const float * h  = llama_get_embeddings_ith(ctx.lctx, -1);
    std::vector<float> h_state(h, h + n_embd);

    const llama_pos n_ctx = (llama_pos) llama_n_ctx(ctx.lctx);
    bool done = false;
    for (int i = 0; !done; i++) {
        llama_token tok = LLAMA_TOKEN_NULL;
        if (gen.next_is_text()) {
            tok = common_sampler_sample(smpl.get(), ctx.lctx, -1);
            common_sampler_accept(smpl.get(), tok, true);
        }
        bool stop = false;
        const float * h_next = nullptr;
        if (gen.step_gen(tok, h_state.data(), &h_next, &stop) != 0) {
            LOG_ERR("\n%s: response generation failed\n", __func__);
            return 1;
        }
        if (tok != LLAMA_TOKEN_NULL && !stop && !llama_vocab_is_control(ctx.vocab, tok)) {
            LOG("%s", common_token_to_piece(ctx.lctx, tok).c_str());
            fflush(stdout);
        }
        if (h_next) {
            h_state.assign(h_next, h_next + n_embd);
        }

        const bool ctx_full = llama_memory_seq_pos_max(llama_get_memory(ctx.lctx), 0) + 2 >= n_ctx;
        done = stop || ctx_full || i + 1 >= n_predict || !g_is_generating || g_is_interrupted;
        if (done) {
            gen.flush();
        }
        int32_t      sample_rate = 0;
        const char * data        = nullptr;
        size_t       data_len    = 0;
        if (gen.get_output(&sample_rate, &data, &data_len) == 0 && data_len > 0) {
            out.write((const float *) data, data_len / sizeof(float));
        }
    }
    LOG("\n");
    ctx.n_past = llama_memory_seq_pos_max(llama_get_memory(ctx.lctx), 0) + 1;
    return 0;
}

// realtime conversation: the model listens to a turn, then answers with text and audio (if it generates audio)
// the audio of a turn is decoded while it comes in, so the response starts right after the turn
static int run_conversation(mtmd_cli_context & ctx, const common_params & params, mtmd_cli_audio_input & input, int n_predict) {
    mtmd_cli_turn_text text;
    if (!text.init(ctx, params)) {
        LOG_ERR("%s: the chat template does not fit the realtime turn format\n", __func__);
        return 1;
    }

    mtmd_helper::gen_audio gen;
    mtmd_cli_audio_out     out;
    if (ctx.mtmd_cap.audio_gen) {
        gen.init(ctx.lctx, ctx.ctx_vision.get());
        const auto info = mtmd_gen_audio_get_info(ctx.ctx_vision.get());
        const bool ok   = params.out_file.empty() ?
            out.init_speaker(info.sample_rate, info.n_channels) :
            out.init_file(params.out_file, info.sample_rate, info.n_channels);
        if (!gen.valid() || !ok) {
            return 1;
        }
    }
    const int silence_ms = 700; // silence that ends a turn
    mtmd_cli_vad vad(input.rate, silence_ms);
    std::vector<float> buf;
    LOG("\nListening, press Ctrl+C to quit\n");

    for (int n_turns = 0; !g_is_interrupted && !input.in.is_end(); n_turns++) {
        g_is_generating = false;
        LOG("\n> ");
        fflush(stdout);

        // user turn: its text is decoded when the speech starts
        mtmd::audio_stream_ptr stream;
        bool end = false;
        while (!end && !g_is_interrupted) {
            buf.clear();
            if (input.in.read(buf) == 0) {
                if (!input.in.is_end()) {
                    wait_for_audio();
                    continue;
                }
                end = true;
            }
            const std::vector<float> speech = vad.process(buf.data(), buf.size());
            end = end || vad.turn_ended();
            if (speech.empty() && !stream) {
                continue;
            }
            if (!stream) {
                LOG("(listening)");
                fflush(stdout);
                if (decode_text(ctx, n_turns == 0 ? text.first_header : text.assistant_footer + text.user_header, false)) {
                    return 1;
                }
                stream.reset(mtmd_audio_stream_init(ctx.ctx_vision.get(), mtmd_audio_stream_params_default()));
                if (!stream) {
                    return 1;
                }
            }
            mtmd_audio_stream_push(stream.get(), speech.data(), speech.size());
            if (end) {
                mtmd_audio_stream_flush(stream.get(), true);
            }
            int32_t n;
            while ((n = decode_audio_rows(ctx, stream.get(), LLAMA_TOKEN_NULL, false)) > 0) {
            }
            if (n < 0) {
                return 1;
            }
        }
        if (!stream || g_is_interrupted) {
            break; // no speech before the end of the input
        }

        LOG("\n\n");
        g_is_generating = true;
        const std::string prompt = text.user_footer + text.assistant_header;
        if (gen.valid()) {
            // the pipeline reads the hidden state of the backbone
            llama_set_embeddings(ctx.lctx, true);
            const int ret = generate_audio_response(ctx, params, gen, out, prompt, n_predict);
            llama_set_embeddings(ctx.lctx, false);
            if (ret) {
                return 1;
            }
        } else {
            if (decode_text(ctx, prompt, true) || generate_response(ctx, n_predict)) {
                return 1;
            }
        }

        // the microphone hears the response, drop what it recorded
        out.drain(g_is_interrupted);
        input.in.discard();
        vad.reset();
    }

    LOG("\n");
    llama_perf_context_print(ctx.lctx);
    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    inject_test_response_marker();

    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

    if (g_is_interrupted) return 0;

    // note: we replace the marker here instead of letting mtmd_tokenize() to do that
    //       because we want to demonstrate how to use mtmd_tokenize_from_parts()

    // split the formatted chat on the media marker to get text segments
    const std::string marker = mtmd_default_marker();
    std::vector<std::string> segments;
    size_t start = 0;
    size_t pos;
    while ((pos = formatted_chat.find(marker, start)) != std::string::npos) {
        segments.push_back(formatted_chat.substr(start, pos - start));
        start = pos + marker.size();
    }
    segments.push_back(formatted_chat.substr(start));

    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    if (segments.size() - 1 != bitmaps_c_ptr.size()) {
        LOG_ERR("Number of media markers (%zu) does not match number of loaded media (%zu)\n",
                segments.size() - 1, bitmaps_c_ptr.size());
        return 1;
    }

    // interleave text and media parts
    std::vector<mtmd_input_text> texts(segments.size());
    std::vector<mtmd_input_part> parts;
    for (size_t i = 0; i < segments.size(); i++) {
        texts[i] = {segments[i].data(), segments[i].size(), /* add_special */ false, /* parse_special */ true};
        parts.push_back({&texts[i], nullptr});
        if (i < bitmaps_c_ptr.size()) {
            parts.push_back({nullptr, bitmaps_c_ptr[i]});
        }
    }
    std::vector<const mtmd_input_part *> parts_ptr;
    for (const auto & p : parts) {
        parts_ptr.push_back(&p);
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    // mbatch must not outlive the chunks it references
    mtmd::batch_ptr mbatch;
    int32_t res = mtmd_tokenize_from_parts(ctx.ctx_vision.get(),
                        chunks.ptr.get(), // output
                        parts_ptr.data(),
                        parts_ptr.size(),
                        add_bos);
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();
    ctx.videos.clear();

    // batch encode all media chunks, then decode each
    size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t i = 0; i < n_chunks; i++) {
        auto chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // decode text chunk
            llama_pos new_n_past = ctx.n_past;
            res = mtmd_helper_eval_chunk_single(ctx.ctx_vision.get(),
                        ctx.lctx,
                        chunk,
                        ctx.n_past,
                        0, // seq_id
                        ctx.n_batch,
                        i == n_chunks - 1, // logits_last
                        &new_n_past);
            if (res != 0) {
                LOG_ERR("Unable to eval text chunk %zu\n", i);
                return 1;
            }
            ctx.n_past = new_n_past;
        } else {
            // media chunk: try to get embd from existing batch, or create a new batch
            float * embd = nullptr;
            if (mbatch) {
                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);

                if (embd) {
                    LOG_DBG("found embd for media chunk %zu in existing batch\n", i);
                } else {
                    LOG_DBG("media chunk %zu not found in existing batch, creating new batch\n", i);
                }
            }

            if (!embd) {
                // create and encode a new batch with as many media chunks as possible
                mbatch.reset(mtmd_batch_init(ctx.ctx_vision.get()));
                res = mtmd_batch_add_chunk(mbatch.get(), chunk);
                GGML_ASSERT(res == 0); // first chunk must always succeed

                int n_added = 1;
                // add as many subsequent media chunks as possible
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto next_chunk = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto next_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break; // text chunk splits the batch
                    }
                    res = mtmd_batch_add_chunk(mbatch.get(), next_chunk);
                    if (res != 0) {
                        break; // batch full or incompatible
                    }
                    n_added++;
                }

                int64_t time_start = ggml_time_ms();
                LOG_INF("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", n_added, i, n_chunks);
                res = mtmd_batch_encode(mbatch.get());
                if (res != 0) {
                    LOG_ERR("Failed to encode mtmd batch, res = %d\n", res);
                    return 1;
                }
                LOG_INF("mtmd batch encoding done in %d ms\n", (int)(ggml_time_ms() - time_start));

                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            }

            GGML_ASSERT(embd != nullptr);

            llama_pos new_n_past = ctx.n_past;
            res = mtmd_helper_decode_image_chunk(ctx.ctx_vision.get(),
                        ctx.lctx,
                        chunk,
                        embd,
                        ctx.n_past,
                        0, // seq_id
                        ctx.n_batch,
                        &new_n_past,
                        nullptr, // callback
                        nullptr  // user_data
                    );
            if (res != 0) {
                LOG_ERR("Unable to decode media chunk %zu\n", i);
                return 1;
            }
            ctx.n_past = new_n_past;
        }
    }

    LOG("\n");

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    ggml_backend_load_all();

    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    // Ctrl+C handling
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

    if (g_is_interrupted) return 130;

    if (params.realtime && !ctx.mtmd_cap.audio_stream) {
        LOG_ERR("--realtime needs a model that takes an audio stream\n");
        return 1;
    }

    // realtime transcription models (no chat) transcribe while they listen, the others answer after the audio
    const bool transcribes = ctx.mtmd_cap.audio_stream && !ctx.mtmd_cap.chat;

    if (transcribes) {
        mtmd_cli_audio_input input;
        if (!input.init(ctx, params)) {
            return 1;
        }
        return run_transcription(ctx, input, n_predict);
    }

    if (params.realtime) {
        mtmd_cli_audio_input input;
        if (!input.init(ctx, params)) {
            return 1;
        }
        return run_conversation(ctx, params, input, n_predict);
    }

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = mtmd_default_marker() + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        for (const auto & image : params.image) {
            if (!ctx.load_media(image)) {
                return 1; // error is already printed by libmtmd
            }
        }
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        if (ctx.mtmd_cap.video) {
            LOG("\n   /video <path>    load a video");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            // a submitted line always ends with a newline, an empty read is EOF
            if (line.empty()) {
                break;
            }
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            bool is_video = line == "/video" || line.find("/video ") == 0;
            if (is_image || is_audio || is_video) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (ctx.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : is_audio ? "audio" : "video");
                    content += mtmd_default_marker();
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}

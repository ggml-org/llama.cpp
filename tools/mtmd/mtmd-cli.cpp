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

    mtmd::batch_ptr mbatch;

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
        if (!mtmd_cap.chat) {
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
        const char * clip_path = params.mmproj.path.c_str();
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
            auto mem = mtmd_get_memory_usage(clip_path, mparams);
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
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
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

// feed an audio file to a realtime model chunk by chunk, printing the transcript as it is produced
static int run_audio_stream(mtmd_cli_context & ctx, const std::string & fname, int n_predict) {
    auto res = mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), fname.c_str(), false, ctx.init_opt);
    if (!res.bitmap) {
        return 1; // error is already printed by libmtmd
    }
    mtmd::bitmap bmp(res.bitmap);
    if (!mtmd_bitmap_is_audio(bmp.ptr.get())) {
        LOG_ERR("%s: %s is not an audio file\n", __func__, fname.c_str());
        return 1;
    }
    const float * samples   = (const float *) bmp.data();
    const size_t  n_samples = bmp.n_bytes() / sizeof(float);
    const int     rate      = mtmd_get_audio_sample_rate(ctx.ctx_vision.get());

    const auto sparams = mtmd_audio_stream_params_default();
    mtmd::audio_stream_ptr stream(mtmd_audio_stream_init(ctx.ctx_vision.get(), sparams));
    if (!stream) {
        return 1;
    }

    const size_t n_chunk = (size_t) rate * sparams.step_ms / 1000;
    const int    n_embd  = llama_model_n_embd_inp(ctx.model);

    LOG("Transcribing %.1f s of audio, fed in chunks of %d ms\n\n", (double) n_samples / rate, sparams.step_ms);

    size_t      n_fed  = 0;
    int         n_gen  = 0;
    llama_token tok    = LLAMA_TOKEN_NULL;
    const int64_t t_start = ggml_time_us();
    int64_t       t_enc   = 0; // time spent in the stream (mel + encoder)
    int64_t       t_dec   = 0; // time spent in the decoder
    while (!g_is_interrupted) {
        const int64_t t0 = ggml_time_us();
        const int32_t n = mtmd_audio_stream_step(stream.get(), tok);
        t_enc += ggml_time_us() - t0;
        if (n < 0) {
            if (n == -2) {
                LOG_ERR("%s: stream error\n", __func__);
                return 1;
            }
            break; // end of stream
        }
        if (n == 0) {
            if (n_fed < n_samples) {
                const size_t cnt = std::min(n_chunk, n_samples - n_fed);
                mtmd_audio_stream_push(stream.get(), samples + n_fed, cnt);
                n_fed += cnt;
            } else if (mtmd_audio_stream_flush(stream.get(), true) != 0) {
                break; // already ended
            }
            continue;
        }

        const int64_t t1 = ggml_time_us();
        const float * rows = mtmd_audio_stream_get_output_embd(stream.get());
        ctx.batch.clear();
        for (int32_t i = 0; i < n; i++) {
            const llama_pos pos[GGML_MROPE_SECTIONS] = { ctx.n_past, ctx.n_past, ctx.n_past, 0 };
            ctx.n_past++;
            ctx.batch.add_embd({ rows + (size_t) i * n_embd, 1, (size_t) n_embd }, pos, 0, i == n - 1);
        }
        if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return 1;
        }

        tok = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        common_sampler_accept(ctx.smpl, tok, true);
        t_dec += ggml_time_us() - t1;
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
    LOG("\n\n%s: %d tokens, %.2f s audio in %.2f s (%.2fx realtime)\n", __func__,
        n_gen, (double) n_samples / rate, t_total, (double) n_samples / rate / t_total);
    LOG("%s: encoder %.2f s, decoder %.2f s, %.1f ms per 80 ms step\n", __func__,
        t_enc / 1e6, t_dec / 1e6, (t_enc + t_dec) / 1e3 / std::max(n_gen, 1));
    llama_perf_context_print(ctx.lctx);

    return 0;
}

// resolve a single-token piece (a special token spelled out literally) to its id
static llama_token token_from_piece(llama_context * lctx, const std::string & piece) {
    const auto toks = common_tokenize(lctx, piece, false, true);
    return toks.size() == 1 ? toks[0] : LLAMA_TOKEN_NULL;
}

// Turn-based audio feeding.
//
// The model hears one chunk of audio, answers it, and only then hears the next
// one. That is how it was trained: 1600 ms of audio first, then 800 ms per turn,
// and every turn is answered before the next chunk arrives. Feeding the whole
// file behind one marker throws that away -- the model answers the first turn
// and then has nothing left to say.
//
// Each turn is tokenized by libmtmd, so the audio markers come from the
// projector instead of being hardcoded here:
//     [<|hy_User|>] <audio-beg> <N audio embeddings> <audio-end> <post>
// <post> is whatever -p carried after the media marker (the assistant header),
// replayed at the head of every turn. Text tokens are *not* summed into the
// audio rows: unlike the realtime path, they occupy their own positions.
//
// The audio encoder is stateless, so this is exact and not an approximation:
// the reference restarts its attention mask on every item boundary and one turn
// is far shorter than one block, which makes encoding a turn on its own
// identical to encoding it inside the full sequence.
static int run_turn_stream(mtmd_cli_context &    ctx,
                           const common_params & params,
                           const std::string &   fname,
                           int                   n_predict) {
    // the schedule this model family was trained with
    // TODO: keep this next to the preprocessor, which cuts the mel on the same grid
    const int first_ms = 1600;
    const int step_ms  = 800;

    // reserved tokens, resolved from the vocab so a different checkpoint (or a
    // re-trained vocab) keeps working
    const llama_token t_user = token_from_piece(ctx.lctx, "<｜hy_User｜>");
    // printed: it separates the source branch from the target branch
    const llama_token t_tgt  = token_from_piece(ctx.lctx, "<｜hy_place▁holder▁no▁791｜>");

    if (t_user == LLAMA_TOKEN_NULL) {
        LOG_ERR("%s: the tokenizer has no user-turn token\n", __func__);
        return 1;
    }

    auto res = mtmd_helper_bitmap_init_from_file(ctx.ctx_vision.get(), fname.c_str(), false, ctx.init_opt);
    if (!res.bitmap) {
        return 1;  // error is already printed by libmtmd
    }
    mtmd::bitmap bmp(res.bitmap);
    if (!mtmd_bitmap_is_audio(bmp.ptr.get())) {
        LOG_ERR("%s: %s is not an audio file\n", __func__, fname.c_str());
        return 1;
    }
    const float * all    = (const float *) bmp.data();
    const size_t  n_all  = bmp.n_bytes() / sizeof(float);
    const int     rate   = mtmd_get_audio_sample_rate(ctx.ctx_vision.get());
    const int     n_embd = llama_model_n_embd_inp(ctx.model);

    // -p must carry the media marker; everything before it is the user prompt,
    // everything after it is the assistant header replayed on every turn
    const std::string marker = mtmd_default_marker();
    std::string       pre;
    std::string       post;
    {
        const size_t pos = params.prompt.find(marker);
        if (pos == std::string::npos) {
            LOG_ERR("%s: -p must contain the media marker %s\n", __func__, marker.c_str());
            return 1;
        }
        pre  = params.prompt.substr(0, pos);
        post = params.prompt.substr(pos + marker.size());
    }

    auto decode_tokens = [&](const std::vector<llama_token> & toks) -> int {
        if (toks.empty()) {
            return 0;
        }
        ctx.batch.clear();
        for (size_t i = 0; i < toks.size(); i++) {
            ctx.batch.add(toks[i], ctx.n_past, 0, i + 1 == toks.size());
            ctx.n_past++;
        }
        if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return 1;
        }
        return 0;
    };

    inject_test_response_marker();

    if (decode_tokens(common_tokenize(ctx.lctx, pre, true, true))) {
        return 1;
    }
    // the checkpoints trained with a two-branch answer expect the source-branch
    // marker at the head of every assistant turn; it is part of -p, not of the
    // code, so nothing model-specific is injected here
    const auto post_toks = common_tokenize(ctx.lctx, post, false, true);

    const int64_t t_start = ggml_time_us();
    size_t        off     = 0;
    int           n_gen   = 0;
    int           n_emit  = 0;  // turns that produced content; the rest are WAIT
    int           n_turn  = 0;  // turns actually fed; the last one is usually short
    bool          done    = false;

    for (int turn = 0; !done && off < n_all && !g_is_interrupted; turn++) {
        const size_t want    = (size_t) rate * (turn == 0 ? first_ms : step_ms) / 1000;
        const size_t n_chunk = std::min(want, n_all - off);
        if (n_chunk == 0) {
            break;
        }
        n_turn++;
        mtmd::bitmap turn_bmp(mtmd_bitmap_init_from_audio(n_chunk, all + off));
        off += n_chunk;

        // one turn = one mel = one encoder call, exactly what the reference does
        mtmd::input_chunks           chunks(mtmd_input_chunks_init());
        std::vector<mtmd_input_part> parts;
        // exactly one of text/bitmap may be set, so this part is bitmap-only
        parts.push_back({ nullptr, turn_bmp.ptr.get() });
        std::vector<const mtmd_input_part *> parts_ptr{ parts.data() };
        if (mtmd_tokenize_from_parts(ctx.ctx_vision.get(), chunks.ptr.get(), parts_ptr.data(), parts_ptr.size(),
                                     false) != 0) {
            LOG_ERR("%s: failed to tokenize audio turn\n", __func__);
            return 1;
        }

        // every audio turn is its own user message, so the role token comes back
        // from the second turn on; the first one already got it in `pre`
        if (turn > 0 && decode_tokens(std::vector<llama_token>{ t_user })) {
            return 1;
        }

        // libmtmd emits the audio markers as text chunks of their own, so a turn
        // is just its chunks in order
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); i++) {
            auto * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t              n_tok = 0;
                const llama_token * toks  = mtmd_input_chunk_get_tokens_text(c, &n_tok);
                if (decode_tokens(std::vector<llama_token>(toks, toks + n_tok))) {
                    return 1;
                }
                continue;
            }
            if (mtmd_encode_chunk(ctx.ctx_vision.get(), c) != 0) {
                LOG_ERR("%s: failed to encode turn %d\n", __func__, turn);
                return 1;
            }
            const int32_t            n_tok = mtmd_input_chunk_get_n_tokens(c);
            const float *            rows  = mtmd_get_output_embd(ctx.ctx_vision.get());
            const std::vector<float> embd(rows, rows + (size_t) n_tok * n_embd);

            ctx.batch.clear();
            for (int32_t j = 0; j < n_tok; j++) {
                const llama_pos pos[GGML_MROPE_SECTIONS] = { ctx.n_past, ctx.n_past, ctx.n_past, 0 };
                ctx.n_past++;
                ctx.batch.add_embd({ embd.data() + (size_t) j * n_embd, 1, (size_t) n_embd }, pos, 0, j + 1 == n_tok);
            }
            if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
                LOG_ERR("%s: failed to decode audio turn %d\n", __func__, turn);
                return 1;
            }
        }

        if (decode_tokens(post_toks)) {
            return 1;
        }

        // generate this turn's answer, then hand the boundary token back so the
        // next turn sees a finished assistant turn
        bool emitted = false;
        while (!g_is_interrupted) {
            const llama_token tok = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
            common_sampler_accept(ctx.smpl, tok, true);
            n_gen++;

            // EOT is only in the EOG set if the GGUF carries
            // tokenizer.ggml.eot_token_id; without it the model repeats the same
            // turn until it hits EOS
            const bool is_eog = llama_vocab_is_eog(ctx.vocab, tok);
            if (is_eog) {
                // one line per turn so the two branches stay separable
                LOG("\n");
                done = (tok == llama_vocab_eos(ctx.vocab));
            } else if (tok == t_tgt || !llama_vocab_is_control(ctx.vocab, tok)) {
                LOG("%s", common_token_to_piece(ctx.lctx, tok).c_str());
                fflush(stdout);
                emitted = true;
            }
            if (decode_tokens(std::vector<llama_token>{ tok })) {
                return 1;
            }
            if (is_eog) {
                break;
            }
            if (n_gen >= n_predict) {
                done = true;
                break;
            }
        }
        n_emit += emitted ? 1 : 0;
    }

    const double t_total = (ggml_time_us() - t_start) / 1e6;
    LOG("\n\n%s: %d tokens, %d/%d turns emitted, %.2f s audio in %.2f s (%.2fx realtime)\n", __func__, n_gen, n_emit,
        n_turn, (double) n_all / rate, t_total, (double) n_all / rate / t_total);
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
            if (ctx.mbatch) {
                embd = mtmd_batch_get_output_embd(ctx.mbatch.get(), chunk);

                if (embd) {
                    LOG_DBG("found embd for media chunk %zu in existing batch\n", i);
                } else {
                    LOG_DBG("media chunk %zu not found in existing batch, creating new batch\n", i);
                }
            }

            if (!embd) {
                // create and encode a new batch with as many media chunks as possible
                ctx.mbatch.reset(mtmd_batch_init(ctx.ctx_vision.get()));
                res = mtmd_batch_add_chunk(ctx.mbatch.get(), chunk);
                GGML_ASSERT(res == 0); // first chunk must always succeed

                int n_added = 1;
                // add as many subsequent media chunks as possible
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto next_chunk = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto next_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break; // text chunk splits the batch
                    }
                    res = mtmd_batch_add_chunk(ctx.mbatch.get(), next_chunk);
                    if (res != 0) {
                        break; // batch full or incompatible
                    }
                    n_added++;
                }

                int64_t time_start = ggml_time_ms();
                LOG_INF("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", n_added, i, n_chunks);
                res = mtmd_batch_encode(ctx.mbatch.get());
                if (res != 0) {
                    LOG_ERR("Failed to encode mtmd batch, res = %d\n", res);
                    return 1;
                }
                LOG_INF("mtmd batch encoding done in %d ms\n", (int)(ggml_time_ms() - time_start));

                embd = mtmd_batch_get_output_embd(ctx.mbatch.get(), chunk);
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
        // single file layout: the projector may be packaged inside the model file itself
        if (params.model.path.empty() || !mtmd_has_projector(params.model.path.c_str())) {
            show_additional_info(argc, argv);
            LOG_ERR("ERR: Missing --mmproj argument\n");
            return 1;
        }
        params.mmproj.path = params.model.path;
        LOG_INF("%s: --mmproj not given, using the projector packaged in %s\n", __func__, params.model.path.c_str());
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

    const auto audio_mode = mtmd_audio_stream_mode(ctx.ctx_vision.get());

    if (audio_mode == MTMD_AUDIO_STREAM_TURN) {
        if (params.image.size() != 1) {
            LOG_ERR("This model answers audio turn by turn, pass exactly one --audio file\n");
            return 1;
        }
        return run_turn_stream(ctx, params, params.image[0], n_predict);
    }

    if (audio_mode == MTMD_AUDIO_STREAM_REALTIME) {
        if (params.image.size() != 1) {
            LOG_ERR("This model transcribes audio in realtime, pass exactly one --audio file\n");
            return 1;
        }
        return run_audio_stream(ctx, params.image[0], n_predict);
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

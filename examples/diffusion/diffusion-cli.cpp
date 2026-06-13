#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <limits.h>
#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct callback_data {
    diffusion_params *  diff_params;
    const llama_vocab * vocab;
    int32_t             n_input;
    bool                show_progress;     // visual mode: draw the step progress bar
    int32_t             visual_interval;   // visual mode: redraw every Nth step
    int32_t             steps_seen;        // per-turn step count (callback invocations)
    int32_t             blocks_seen;       // per-turn block count (callbacks with step == 0)
    int32_t             term_rows;         // visual mode: terminal size (the canvas viewport is clamped to it)
    int32_t             term_cols;
    int32_t             vis_prev_rows;     // visual mode: rows the previous frame advanced (for cursor-up)
};

// Query the terminal size for the visual viewport; fall back to 24x80 when it can't be read (piped output).
static void get_terminal_size(int32_t & rows, int32_t & cols) {
    rows = 24;
    cols = 80;
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        const int r = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        const int c = csbi.srWindow.Right  - csbi.srWindow.Left + 1;
        if (r > 1) { rows = r; }
        if (c > 0) { cols = c; }
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 1) {
        rows = ws.ws_row;
        cols = ws.ws_col > 0 ? ws.ws_col : 80;
    }
#endif
}

static bool diffusion_step_callback(int32_t             step,
                                    int32_t             total_steps,
                                    const llama_token * tokens,
                                    int32_t             n_tokens,
                                    void *              user_data) {
    callback_data * data = static_cast<callback_data *>(user_data);

    data->steps_seen++;
    if (step == 0) {
        data->blocks_seen++;  // each block's denoise restarts the step counter at 0
    }

    auto print_progress_bar = [](int32_t step, int32_t total_steps) {
        int progress_percent = (step * 100) / total_steps;
        int progress_bars    = (step * 50) / total_steps;
        LOG_INF("\rdiffusion step: %d/%d [%s%s] %d%%",
                step,
                total_steps,
                std::string(progress_bars, '=').c_str(),
                std::string(50 - progress_bars, ' ').c_str(),
                progress_percent);
    };

    if (data->diff_params->visual_mode) {
        // Throttle redraws to every Nth step (all steps are still computed); always draw the first.
        if (data->visual_interval > 1 && (step % data->visual_interval) != 0) {
            return true;
        }

        // Draw the canvas as a fixed-height region in the normal screen buffer (not the alternate buffer, so
        // scrollback stays intact): step the cursor back up over the previous frame, then repaint exactly
        // `rows` lines, each truncated to the terminal width and padded out, so the region never scrolls. The
        // whole repaint is one synchronized update (DEC mode 2026) written directly, so it cannot tear.
        const int rows = std::max(1, data->term_rows - 1);
        const int cols = std::max(1, data->term_cols);

        std::vector<std::string> lines;
        if (data->show_progress) {
            int progress_percent = (step * 100) / total_steps;
            int progress_bars    = (step * 50) / total_steps;
            lines.push_back("diffusion step: " + std::to_string(step) + "/" + std::to_string(total_steps) +
                            " [" + std::string(progress_bars, '=') + std::string(50 - progress_bars, ' ') +
                            "] " + std::to_string(progress_percent) + "%");
        }
        std::string cur = " ";
        for (int32_t i = data->n_input; i < n_tokens; i++) {
            if (tokens[i] != llama_vocab_mask(data->vocab)) {
                char piece[256];
                int  n_chars = llama_token_to_piece(data->vocab, tokens[i], piece, sizeof(piece), 0, false);
                for (int32_t k = 0; k < n_chars; k++) {
                    if (piece[k] == '\n') { lines.push_back(cur); cur.clear(); } else { cur += piece[k]; }
                }
            } else {
                cur += ' ';
            }
        }
        lines.push_back(cur);

        std::string frame = "\033[?2026h";                     // begin synchronized frame
        if (data->vis_prev_rows > 0) {
            frame += "\033[" + std::to_string(data->vis_prev_rows) + "A";  // back to the top of the region
        }
        frame += "\r";
        for (int r = 0; r < rows; r++) {
            std::string ln = (r < (int) lines.size()) ? lines[r] : std::string();
            if ((int) ln.size() > cols) { ln.resize(cols); }    // clamp width so the row never wraps
            frame += ln + "\033[K";
            if (r < rows - 1) { frame += "\n"; }
        }
        frame += "\033[?2026l";                                // end synchronized frame
        data->vis_prev_rows = rows - 1;
        fwrite(frame.data(), 1, frame.size(), stdout);
        fflush(stdout);
    } else {
        print_progress_bar(step, total_steps);
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.use_mmap           = params.use_mmap;
    model_params.use_direct_io      = params.use_direct_io;
    model_params.use_mlock          = params.use_mlock;
    model_params.check_tensors      = params.check_tensors;

    // honor -ot / --n-cpu-moe (tensor buffer placement); without this the offload flags are silently
    // dropped and the MoE experts stay on GPU, OOMing small-VRAM cards
    if (params.tensor_buft_overrides.empty()) {
        model_params.tensor_buft_overrides = nullptr;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        model_params.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: unsupported model for diffusion");
        llama_model_free(model);
        return 1;
    }

    // DiffusionGemma block diffusion exposes diffusion.canvas_length: the prompt is followed by a fixed
    // canvas of that many masked positions, which overrides the generic n_ubatch generation length.
    char    canvas_str[32];
    int64_t canvas_length = 0;
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
        canvas_length = strtol(canvas_str, nullptr, 10);
    }

    // Canvas models self-condition (a full-vocab soft-embedding in the graph). Enable it before context
    // creation so the reserve sizes the compute buffer; the real logits buffer is supplied per step.
    if (canvas_length > 0) {
        llama_diffusion_set_sc(model, nullptr, /*use_sc*/ 0.0f, /*temp_inv*/ 1.0f, /*enabled*/ true);
    }

    // -n/--n-predict drives length for canvas models: derive the block count from the target token budget
    // and grow ubatch/batch/ctx to hold the final block's whole [prompt | canvas] in one pass, so
    // --diffusion-blocks / -ub / -b / -c need not be set by hand. Larger explicit values are kept, and the
    // denoise still stops early on an end token.
    if (canvas_length > 0 && params.n_predict > 0) {
        const int32_t cl     = (int32_t) canvas_length;
        const int32_t blocks = (params.n_predict + cl - 1) / cl;
        const int32_t needed = blocks * cl + 2048;  // + headroom for the prompt / chat history
        params.diffusion.blocks = blocks;
        params.n_ubatch = std::max(params.n_ubatch, needed);
        params.n_batch  = std::max(params.n_batch,  params.n_ubatch);
        params.n_ctx    = std::max(params.n_ctx,    needed);
        LOG_INF("diffusion: -n %d -> %d blocks, n_ubatch=%d n_batch=%d n_ctx=%d (canvas_length=%d)\n",
                params.n_predict, blocks, params.n_ubatch, params.n_batch, params.n_ctx, cl);
    }

    // --fit (auto context/layer fitting) runs inside common_init_from_params, which this runner does not
    // use: it sizes context from -n and the canvas above. Tell the user so --fit is not silently ignored.
    if (params.fit_params) {
        LOG_INF("diffusion: --fit has no effect here; context is sized from -n and the canvas. "
                "Set -ngl / --n-cpu-moe to control device memory.\n");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn_type      = params.flash_attn_type;
    ctx_params.no_perf              = params.no_perf;
    ctx_params.type_k               = params.cache_type_k;
    ctx_params.type_v               = params.cache_type_v;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto chat_templates = common_chat_templates_init(model, "");

    llama_token mask_token_id = llama_vocab_mask(vocab);
    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    const bool visual_mode = params.diffusion.visual_mode;

    // reused across turns; canvas models fill only n_input + canvas_length of it
    std::vector<llama_token> output_tokens(params.n_ubatch);

    struct diffusion_params diff_params;

    char shift_logits_str[8];
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        diff_params.shift_logits = (strcmp(shift_logits_str, "true") == 0);
    } else {
        // canvas block-diffusion is not autoregressive (logit i predicts token i); Dream defaults to shifted
        diff_params.shift_logits = (canvas_length == 0);
    }

    if (canvas_length > 0) {
        // Denoise the whole canvas with the timestep schedule; block scheduling asserts
        // max_length % block_length == 0, which a prompt + canvas layout does not satisfy.
        diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        diff_params.eps      = params.diffusion.eps > 0 ? params.diffusion.eps : 1e-3f;
        // these models put probability mass on the mask token at still-masked positions, so a reveal must
        // sample from the non-mask distribution (unlike Dream/LLaDA, which never emit their mask token).
        diff_params.suppress_mask_token = true;
        // bootstrap the denoise from all-mask: feed each step's canvas logits into the next step
        diff_params.self_conditioning = true;
    } else {
        //Use either eps or block length, but not both
        GGML_ASSERT((params.diffusion.eps == 0) ^ (params.diffusion.block_length == 0));

        if (params.diffusion.eps) {
            diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
            diff_params.eps      = params.diffusion.eps;
        } else if (params.diffusion.block_length) {
            diff_params.schedule     = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
            diff_params.block_length = params.diffusion.block_length;
        }
    }

    diff_params.mask_token_id    = mask_token_id;
    diff_params.seed             = params.sampling.seed;
    diff_params.temperature      = params.sampling.temp;
    diff_params.steps            = params.diffusion.steps;
    diff_params.algorithm        = static_cast<diffusion_algorithm>(params.diffusion.algorithm);
    diff_params.top_p            = params.sampling.top_p;
    diff_params.top_k            = params.sampling.top_k;
    diff_params.visual_mode      = params.diffusion.visual_mode;
    diff_params.add_gumbel_noise = params.diffusion.add_gumbel_noise;

    callback_data cb_data               = { &diff_params, vocab, 0, params.diffusion.visual_progress,
                                            std::max(1, params.diffusion.visual_interval), 0, 0, 24, 80, 0 };
    diff_params.step_callback           = diffusion_step_callback;
    diff_params.step_callback_user_data = &cb_data;

    // max_length is per-turn (it includes the prompt length); the rest is fixed for the run
    LOG_INF("diffusion_params: steps=%d schedule=%d algorithm=%d temperature=%.3f eps=%.6f mask_token=%d\n",
            diff_params.steps, (int) diff_params.schedule, (int) diff_params.algorithm,
            diff_params.temperature, diff_params.eps, mask_token_id);

    // Entropy-bound decoder: the real DiffusionGemma sampler (random-init canvas, MI-bounded acceptance,
    // renoise, temperature schedule, adaptive stop). Auto-enabled for canvas models; --diffusion-eb forces
    // it on/off. Params come from GGUF metadata (diffusion.eb_*), then reference defaults, then CLI override.
    const bool use_eb = canvas_length > 0 && params.diffusion.eb_mode != 2;

    struct diffusion_eb_params eb_params;
    if (use_eb) {
        auto meta_f = [&](const char * key, float def) -> float {
            char buf[32];
            return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
        };
        auto meta_i = [&](const char * key, int32_t def) -> int32_t {
            char buf[32];
            return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? (int32_t) strtol(buf, nullptr, 10) : def;
        };
        eb_params.max_denoising_steps  = meta_i("diffusion.eb_max_steps", 48);
        eb_params.t_min                = meta_f("diffusion.eb_t_min", 0.4f);
        eb_params.t_max                = meta_f("diffusion.eb_t_max", 0.8f);
        eb_params.entropy_bound        = meta_f("diffusion.eb_entropy_bound", 0.1f);
        eb_params.stability_threshold  = meta_i("diffusion.eb_stability_threshold", 1);
        eb_params.confidence_threshold = meta_f("diffusion.eb_confidence_threshold", 0.005f);
        if (params.diffusion.eb_t_min         >= 0) { eb_params.t_min                = params.diffusion.eb_t_min; }
        if (params.diffusion.eb_t_max         >= 0) { eb_params.t_max                = params.diffusion.eb_t_max; }
        if (params.diffusion.eb_entropy_bound >= 0) { eb_params.entropy_bound        = params.diffusion.eb_entropy_bound; }
        if (params.diffusion.eb_stability     >= 0) { eb_params.stability_threshold  = params.diffusion.eb_stability; }
        if (params.diffusion.eb_confidence    >= 0) { eb_params.confidence_threshold = params.diffusion.eb_confidence; }
        if (params.diffusion.eb_max_steps     >  0) { eb_params.max_denoising_steps  = params.diffusion.eb_max_steps; }
        eb_params.seed                    = params.sampling.seed;
        eb_params.visual_mode             = params.diffusion.visual_mode;
        eb_params.step_callback           = diffusion_step_callback;
        eb_params.step_callback_user_data = &cb_data;

        // prefix KV cache: auto = on for single-GPU canvas models, off when the model may span >1 GPU
        // (the F32 prompt-KV store is single-device).
        int gpu_devs = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            const auto dt = ggml_backend_dev_type(ggml_backend_dev_get(i));
            if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) { gpu_devs++; }
        }
        if (params.diffusion.eb_kv_cache == 1) {
            eb_params.kv_cache = true;
        } else if (params.diffusion.eb_kv_cache == 2) {
            eb_params.kv_cache = false;
        } else {  // auto
            eb_params.kv_cache = (gpu_devs <= 1);
            if (gpu_devs > 1) {
                LOG_INF("diffusion_eb: kv cache auto-off (%d GPUs; pass --diffusion-kv-cache on to force)\n", gpu_devs);
            }
        }

        // device-resident SC: auto (default) and on enable it on a single device; sc_dev is single-device
        // like the prompt-KV store, so auto-disable on multi-GPU. SC inputs are bit-identical to host SC.
        if (params.diffusion.eb_gpu_sampling == 2) {  // off
            eb_params.gpu_sampling = false;
        } else {  // auto (default) or on
            eb_params.gpu_sampling = (gpu_devs <= 1);
            if (gpu_devs > 1) {
                LOG_INF("diffusion_eb: gpu sampling off (%d GPUs; sc_dev is single-device)\n", gpu_devs);
            }
        }

        // Stage-1 device sample reduction: auto (default) = on for single-GPU; needs gpu_sampling/sc_dev.
        if (params.diffusion.eb_gpu_sample_reduce == 2) {  // off
            eb_params.gpu_sample_reduce = false;
        } else {  // auto or on
            eb_params.gpu_sample_reduce = eb_params.gpu_sampling && (gpu_devs == 1);
            if (!eb_params.gpu_sampling) {
                LOG_INF("diffusion_eb: gpu sample reduce off (needs --diffusion-gpu-sampling on / sc_dev)\n");
            } else if (gpu_devs != 1) {
                LOG_INF("diffusion_eb: gpu sample reduce off (%d GPUs; needs a single CUDA device)\n", gpu_devs);
            }
        }

        LOG_INF("diffusion_eb: max_steps=%d t=[%.3f,%.3f] entropy_bound=%.4f stability=%d confidence=%.4f kv_cache=%s gpu_sampling=%s sample_reduce=%s\n",
                eb_params.max_denoising_steps, eb_params.t_min, eb_params.t_max, eb_params.entropy_bound,
                eb_params.stability_threshold, eb_params.confidence_threshold, eb_params.kv_cache ? "on" : "off",
                eb_params.gpu_sampling ? "on" : "off", eb_params.gpu_sample_reduce ? "on" : "off");
    }

    // Trim a denoised canvas: cut at the first end-of-generation token, or (checkpoints often emit no stop
    // token) at the onset of a repetition loop (a token recurring at stride 1-2 for >= 6 steps).
    auto trim_canvas = [&](const llama_token * canvas, size_t n) -> size_t {
        size_t cut = n;
        for (size_t i = 0; i < n; i++) {
            if (llama_vocab_is_eog(vocab, canvas[i])) {
                cut = i;
                break;
            }
        }
        for (size_t i = 0; i + 1 < cut; i++) {
            bool loop = false;
            for (size_t stride = 1; stride <= 2 && !loop; stride++) {
                size_t reps = 0;
                for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) {
                    reps++;
                }
                loop = reps >= 6;
            }
            if (loop) {
                cut = i;
                break;
            }
        }
        return cut;
    };

    // Generate one response for a chat-formatted prompt. Canvas models denoise a fixed canvas_length block
    // per pass; with --diffusion-blocks > 1 we run block-autoregressively, committing each block to the
    // prefix and denoising the next until an end token, a repetition loop, the block budget, or the ubatch
    // limit (the whole [prefix | canvas] must fit in one non-causal ubatch). Returns the trimmed text.
    auto run_turn = [&](const std::string & formatted_prompt) -> std::string {
        std::vector<llama_token> prefix = common_tokenize(vocab, formatted_prompt,
                                                          /*add special*/ true, /*parse special*/ true);
        const int n_input = (int) prefix.size();
        if ((uint32_t) n_input >= llama_n_ctx(ctx)) {
            LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, (int) llama_n_ctx(ctx));
            return "";
        }

        // non-canvas models (Dream/LLaDA): single fixed-length pass, blocks ignored
        if (canvas_length <= 0) {
            diff_params.max_length = params.n_ubatch;
            cb_data.n_input        = n_input;
            int32_t n_generated = 0;
            diffusion_generate(ctx, prefix.data(), output_tokens.data(), n_input, diff_params, n_generated);
            if (n_generated <= n_input) {
                LOG_INF("Error: diffusion generation failed\n");
                return "";
            }
            return common_detokenize(vocab,
                std::vector<llama_token>(output_tokens.begin() + n_input, output_tokens.begin() + n_generated), false);
        }

        const int32_t max_ub   = std::min((int32_t) params.n_ubatch, (int32_t) llama_n_ctx(ctx));
        const int     n_blocks = std::max(1, params.diffusion.blocks);
        std::vector<llama_token> response;

        for (int b = 0; b < n_blocks; b++) {
            const int32_t prefix_len = (int32_t) prefix.size();
            const int32_t max_length = prefix_len + (int32_t) canvas_length;
            if (max_length > max_ub) {
                if (b == 0) {
                    LOG_ERR("error: this diffusion model needs the whole [prompt | canvas] in one ubatch; "
                            "set -ub and -c >= n_input + canvas_length = %d + %d = %d\n",
                            prefix_len, (int) canvas_length, max_length);
                    return "";
                }
                break;  // out of ubatch room: stop and keep what we have
            }

            diff_params.max_length = max_length;
            eb_params.max_length   = max_length;
            cb_data.n_input        = prefix_len;

            int32_t n_generated = 0;
            if (use_eb) {
                diffusion_generate_entropy_bound(ctx, prefix.data(), output_tokens.data(), prefix_len, eb_params, n_generated);
            } else {
                diffusion_generate(ctx, prefix.data(), output_tokens.data(), prefix_len, diff_params, n_generated);
            }
            if (n_generated <= prefix_len) {
                if (b == 0) {
                    LOG_INF("Error: diffusion generation failed\n");
                    return "";
                }
                break;
            }

            const llama_token * canvas = output_tokens.data() + prefix_len;
            const size_t        cut    = trim_canvas(canvas, (size_t) canvas_length);
            response.insert(response.end(), canvas, canvas + cut);
            if (cut < (size_t) canvas_length) {
                break;  // end token or repetition loop: answer complete
            }
            prefix.insert(prefix.end(), canvas, canvas + cut);  // commit the block, denoise the next
        }

        return common_detokenize(vocab, response, false);
    };

    auto make_msg = [](const std::string & role, const std::string & content) {
        common_chat_msg m;
        m.role    = role;
        m.content = content;
        return m;
    };

    auto apply_template = [&](const std::vector<common_chat_msg> & messages) -> std::string {
        common_chat_templates_inputs inputs;
        inputs.messages              = messages;
        inputs.add_generation_prompt = true;
        return common_chat_templates_apply(chat_templates.get(), inputs).prompt;
    };

    // Run one turn, print the reply, then (entropy-bound only) a timing summary just before the next prompt.
    // In visual mode the denoising animation occupies a fixed region below the prompt in the normal screen
    // buffer (so scrollback is preserved): reserve the region up front, hide the cursor, let the callback
    // repaint it in place, then erase it and show the cursor before the reply prints in normal flow.
    auto run_turn_reply = [&](const std::string & formatted_prompt) -> std::string {
        cb_data.steps_seen  = 0;
        cb_data.blocks_seen = 0;
        int region_rows = 0;
        if (visual_mode) {
            get_terminal_size(cb_data.term_rows, cb_data.term_cols);
            cb_data.vis_prev_rows = 0;
            region_rows = std::max(1, cb_data.term_rows - 1);
            common_log_flush(common_log_main());           // flush pending logs before reserving the region
            std::string init = "\033[?25l";                // hide cursor
            init += std::string(region_rows, '\n');        // reserve the region (scroll up once if at bottom)
            init += "\033[" + std::to_string(region_rows) + "A";  // park at the top of the region
            fwrite(init.data(), 1, init.size(), stdout);
            fflush(stdout);
        }
        const int64_t t0 = ggml_time_us();
        std::string response = run_turn(formatted_prompt);
        const int64_t turn_us = ggml_time_us() - t0;
        if (visual_mode) {
            std::string fin;
            if (cb_data.vis_prev_rows > 0) {
                fin += "\033[" + std::to_string(cb_data.vis_prev_rows) + "A";  // back to the region top
            }
            fin += "\r\033[J\033[?25h";                    // erase the region, show the cursor
            fwrite(fin.data(), 1, fin.size(), stdout);
            fflush(stdout);
        }
        LOG("\n%s\n", response.c_str());
        if (use_eb && cb_data.steps_seen > 0) {
            const double total_ms = turn_us / 1000.0;
            const double per_step = total_ms / cb_data.steps_seen;
            LOG("total time: %.2fms, time per step: %.2fms (%d steps over %d blocks, entropy-bound)\n",
                total_ms, per_step, cb_data.steps_seen, cb_data.blocks_seen);
            // effective tok/s = canvas tokens this turn / wall time; in-step parallel = canvas / per-step
            // (every canvas position is refined each step; step count divides it down to effective).
            if (canvas_length > 0 && cb_data.blocks_seen > 0) {
                const int    gen_toks = (int) canvas_length * cb_data.blocks_seen;
                const double eff_tps  = gen_toks * 1000.0 / total_ms;
                const double par_tps  = canvas_length * 1000.0 / per_step;
                LOG("throughput: %.1f tok/s (%d tok in %.2fms), in-step parallel %.0f tok/s "
                    "(%d-tok canvas x %.1f steps/block)\n",
                    eff_tps, gen_toks, total_ms, par_tps, (int) canvas_length,
                    (double) cb_data.steps_seen / cb_data.blocks_seen);
            }
        }
        return response;
    };

    if (params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED) {
        if (!params.enable_chat_template) {
            LOG_ERR("error: conversation mode requires a chat template\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // Multi-turn: each turn re-applies the template to the full history and denoises a fresh canvas
        // (no state is kept across turns). History is bounded by the ubatch cap (run_turn reports overflow).
        std::vector<common_chat_msg> messages;
        if (!params.system_prompt.empty()) {
            messages.push_back(make_msg("system", params.system_prompt));
        }

        LOG_INF("conversation mode: /help for commands, /clear to reset, /exit to quit\n");

        std::string pending = params.prompt;  // optional first user turn supplied via -p
        while (true) {
            std::string user;
            if (!pending.empty()) {
                user = pending;
                pending.clear();
            } else {
                common_log_flush(common_log_main());  // drain async logs so they don't clobber the prompt
                printf("\n> ");
                fflush(stdout);
                if (!std::getline(std::cin, user)) {
                    break;  // EOF (Ctrl-D)
                }
                if (user == "/exit" || user == "/quit") {
                    break;
                }
                if (user == "/help" || user == "/?") {
                    LOG("commands:\n"
                        "  /help, /?      show this message\n"
                        "  /clear         clear the conversation history (keeps the system prompt)\n"
                        "  /exit, /quit   end the session\n");
                    continue;
                }
                if (user == "/clear") {
                    messages.clear();
                    if (!params.system_prompt.empty()) {
                        messages.push_back(make_msg("system", params.system_prompt));
                    }
                    LOG("conversation history cleared\n");
                    continue;
                }
                if (user.empty()) {
                    continue;
                }
            }

            messages.push_back(make_msg("user", user));
            const std::string response = run_turn_reply(apply_template(messages));
            messages.push_back(make_msg("assistant", response));
        }
    } else {
        std::string formatted = params.prompt;
        if (params.enable_chat_template) {
            std::vector<common_chat_msg> messages;
            if (!params.system_prompt.empty()) {
                messages.push_back(make_msg("system", params.system_prompt));
            }
            messages.push_back(make_msg("user", params.prompt));
            formatted = apply_template(messages);
        }
        run_turn_reply(formatted);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

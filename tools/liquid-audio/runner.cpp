// mtmd-audio.h must be included before common.h due to conflicting declarations of string_replace_all
#include "mtmd-audio.h"
//
#include "runner.h"
//
#include "chat.h"
#include "common.h"
#include "decoder.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <atomic>
#include <complex>
#include <cstring>
#include <filesystem>
#include <optional>
#include <utility>

namespace liquid {
namespace audio {

namespace {
struct audio_context {
    mtmd::context_ptr      mtmd_ctx_audio;
    common_init_result_ptr llama_init;

    llama_model *       model;
    llama_context *     lctx;
    const llama_vocab * vocab;
    common_sampler *    smpl;
    llama_pos           n_past = 0;

    // audio tokenizer
    common_init_result_ptr audio_tokenizer_llama_init;
    llama_model *          audio_tokenizer_model;
    llama_context *        audio_tokenizer_lctx;

    int n_batch;
    int verbosity = 0;

    mtmd::bitmaps bitmaps;

    common_chat_templates_ptr tmpls;

    // audio sampling params
    float audio_temperature = 0.0f;
    int   audio_top_k       = 1;

    // threadpool
    ggml_threadpool * threadpool                  = nullptr;
    void (*threadpool_free_fn)(ggml_threadpool *) = nullptr;

    int init(common_params & params) {
        // backbone
        llama_init = common_init_from_params(params);
        model      = llama_init->model();
        lctx       = llama_init->context();

        if (!model || !lctx) {
            LOG_ERR("Failed to load backbone\n");
            return 1;
        }

        // audio tokenizer
        auto params_audio_tokenizer         = params;
        params_audio_tokenizer.model.path   = params.vocoder.speaker_file;
        params_audio_tokenizer.mmproj.path  = "";
        params_audio_tokenizer.embedding    = true;
        audio_tokenizer_llama_init          = common_init_from_params(params_audio_tokenizer);
        audio_tokenizer_model              = audio_tokenizer_llama_init->model();
        audio_tokenizer_lctx               = audio_tokenizer_llama_init->context();

        if (!audio_tokenizer_model || !audio_tokenizer_lctx) {
            LOG_ERR("Failed to load audio tokenizer\n");
            return 1;
        }

        // vocab
        vocab = llama_model_get_vocab(model);

        n_batch   = params.n_batch;
        verbosity = params.verbosity > 3;

        // sampler, greedy for text
        params.sampling.samplers = { common_sampler_type::COMMON_SAMPLER_TYPE_TOP_K };
        params.sampling.top_k    = 1;
        smpl                     = common_sampler_init(model, params.sampling);
        tmpls                    = common_chat_templates_init(model, params.chat_template);
        LOG_INF("%s: chat template example:\n%s\n", __func__,
                common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        // mtmd audio context
        const char *        clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams   = mtmd_context_params_default();
        mparams.use_gpu               = params.mmproj_use_gpu;
        mparams.print_timings         = true;
        mparams.n_threads             = params.cpuparams.n_threads;
        mtmd_ctx_audio.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!mtmd_ctx_audio.get()) {
            LOG_ERR("Failed to load audio model from %s\n", clip_path);
            return 1;
        }

        return 0;
    }

    ~audio_context() { common_sampler_free(smpl); }
};

}  // namespace

class Runner::RunnerImpl {
  public:
    RunnerImpl() = default;

    int generate(const std::vector<Message> & messages,
                 int                          n_predict,
                 const text_callback_t &      text_callback,
                 const audio_callback_t &     audio_callback) {
        // handling depends on system prompt
        constexpr const char * asr_system_prompt           = "Perform ASR.";
        constexpr const char * interleaved_system_prompt   = "Respond with interleaved text and audio.";
        constexpr const char * tts_system_prompt_it_prefix = "Perform TTS.";

        std::vector<common_chat_msg> msgs;
        for (const auto & message : messages) {
            if (const auto & role = message.role; role == "system") {
                if (const auto & system_prompt = message.content; system_prompt == asr_system_prompt) {
                    generator = &Runner::RunnerImpl::generate_sequential;
                } else if (system_prompt == interleaved_system_prompt) {
                    // TODO(tarek): check params with Marc
                    ctx.audio_temperature = 0.8;
                    ctx.audio_top_k       = 4;
                    generator             = &Runner::RunnerImpl::generate_interleaved;
                } else if (system_prompt.find(tts_system_prompt_it_prefix) == 0) {
                    // TODO(tarek): check params with Marc
                    ctx.audio_temperature = 0.8;
                    ctx.audio_top_k       = 64;
                    generator             = &Runner::RunnerImpl::generate_sequential;
                } else {
                    return error(
                        std::string("Unsupported system prompt. Supported prompts are:\n - ") + asr_system_prompt +
                        "\n - " + interleaved_system_prompt + "\n - " + tts_system_prompt_it_prefix +
                        "<voice instructions>" + "\n\n" +
                        "Example of <voice instructions> can be 'Use the following voice: A male speaker "
                        "delivers a very expressive and animated speech, with a low-pitch voice and a slightly "
                        "close-sounding tone. The recording carries a slight background noise.', can be modified "
                        "using natural language. e.g. change male to female, low-pitch to high-pitch...\n");
                }
            } else if (role == "user") {
                if (const auto & wav = message.wav; !wav.empty()) {
                    if (message.content != mtmd_default_marker()) {
                        return error("when providing audio input, content must be the default marker: " +
                                     std::string(mtmd_default_marker()));
                    }
                    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
                        ctx.mtmd_ctx_audio.get(), reinterpret_cast<const uint8_t *>(wav.data()), wav.size()));
                    if (!bmp.ptr) {
                        return error("failed to load wav");
                    }
                    ctx.bitmaps.entries.push_back(std::move(bmp));
                }
            }
            // push msg
            common_chat_msg msg;
            msg.role    = message.role;
            msg.content = message.content;
            msgs.push_back(msg);
        }

        if (eval_messages(msgs, ctx.n_past == 0)) {
            return error("failed to run prefill");
        }

        // inject perf measurement here
        auto text_callback_perf = [&](const std::string & text) {
            auto now            = ggml_time_ms();
            first_text_received = first_text_received.value_or(now);
            last_text_received  = now;
            ++text_tokens_count;
            text_callback(text);
        };
        auto audio_callback_perf = [&](const generated_audio_t & audio) {
            auto now             = ggml_time_ms();
            first_audio_received = first_audio_received.value_or(now);
            last_audio_received  = now;
            audio_samples_count += audio.size();
            audio_callback(audio);
        };

        if (!stop_requested && (this->*generator)(n_predict, text_callback_perf, audio_callback_perf) != 0) {
            return error("failed to generate");
        }

        perf_context_print();

        return 0;
    }

    int init(common_params params) {
        for (const auto & [p, desc] : {
                 std::pair{ params.model.path,         "-m"       },
                 std::pair{ params.mmproj.path,        "--mmproj" },
                 std::pair{ params.vocoder.model.path, "-mv"      },
        }) {
            if (p.empty()) {
                LOG_ERR("ERR: Missing %s argument\n", desc);
                return 1;
            }
            if (!std::filesystem::exists(p)) {
                LOG_ERR("ERR: File %s does not exists\n", p.c_str());
                return 1;
            }
        }

        if (auto res = ctx.init(params); res) {
            return error("failed to initialize audio context");
        }

        // audio decoder
        decoder = std::make_unique<Decoder>(params);
        if (!decoder) {
            return error("failed to initialize audio decoder");
        }

        init_threadpool(params.cpuparams.n_threads);

        istft_state = std::make_unique<mtmd_audio_streaming_istft>(istft_config.n_fft, istft_config.hop_length);
        GGML_ASSERT(istft_state);

        reset();

        return 0;
    }

    int get_output_sample_rate() const { return istft_config.sample_rate; }

    void perf_context_print() const {
        llama_perf_context_print(ctx.lctx);

        fflush(stdout);
        LOG("audio samples per second: %10.1f\n",
            audio_samples_count / ((last_audio_received.value_or(0) - first_audio_received.value_or(0)) * 0.001));
        LOG("text  tokens  per second: %10.1f\n",
            text_tokens_count / ((last_text_received.value_or(0) - first_text_received.value_or(0)) * 0.001));
    }

    const char * get_last_error() const { return last_error_.c_str(); }

    void stop() { stop_requested = true; }

    void reset() {
        stop_requested = false;

        perf_context_reset();
        llama_perf_context_reset(ctx.lctx);

        common_sampler_reset(ctx.smpl);

        llama_memory_clear(llama_get_memory(ctx.lctx), false);
        ctx.n_past = 0;

        llama_memory_clear(llama_get_memory(ctx.audio_tokenizer_lctx), false);
        istft_state->reset();
    }

    ~RunnerImpl() {
        if (threadpool && threadpool_free_fn) {
            threadpool_free_fn(threadpool);
        }
    }


  private:
    struct {
        int n_fft       = 1280;
        int hop_length  = 320;
        int sample_rate = 24000;
        int n_codes     = 8;
    } istft_config;

    enum class Modality : uint8_t {
        TEXT,
        AUDIO_OUT,

    };

    using audio_token_t = std::array<int32_t, 8>;

    audio_context ctx;

    std::atomic<bool> stop_requested = false;
    std::string       last_error_;

    // perf
    size_t                 text_tokens_count = 0, audio_samples_count = 0;
    std::optional<int64_t> first_text_received, first_audio_received;
    std::optional<int64_t> last_text_received, last_audio_received;

    std::unique_ptr<Decoder> decoder;
    int (Runner::RunnerImpl::*generator)(int, const text_callback_t &, const audio_callback_t &) = nullptr;

    // threadpool
    ggml_threadpool * threadpool                  = nullptr;
    void (*threadpool_free_fn)(ggml_threadpool *) = nullptr;

    std::unique_ptr<mtmd_audio_streaming_istft> istft_state;

    void init_threadpool(int n_threads) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(cpu_dev);
        auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
        GGML_ASSERT(reg);
        GGML_ASSERT(n_threads > 0);
        if (auto * threadpool_new_fn = (ggml_threadpool * (*) (ggml_threadpool_params *) )
                ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
            threadpool_new_fn) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
            threadpool                 = threadpool_new_fn(&tpp);
        }
        threadpool_free_fn =
            (decltype(threadpool_free_fn)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        GGML_ASSERT(threadpool);

        llama_attach_threadpool(ctx.lctx, threadpool, nullptr);
        llama_attach_threadpool(ctx.audio_tokenizer_lctx, threadpool, nullptr);
        decoder->set_threadpool(threadpool, n_threads);
    }

    int error(const std::string & msg) {
        LOG_ERR("ERR: %s\n", msg.c_str());
        last_error_ = msg;
        return 1;
    }

    int generate_common(int                                                              n_predict,
                        const std::function<Modality(const llama_token &, Modality)> &   text_handler,
                        const std::function<Modality(const audio_token_t &, Modality)> & audio_handler,
                        const text_callback_t &                                          text_callback,
                        const audio_callback_t &                                         audio_callback) {
        Modality    current_modality = Modality::TEXT;
        llama_batch batch            = llama_batch_get_one(nullptr, 1);  // doesn't own pointers, no need for free.

        n_predict = n_predict < 0 ? std::numeric_limits<int>::max() : n_predict;
        std::vector<float> embd(llama_model_n_embd(ctx.model));
        for (int i = 0; i < n_predict; i++) {
            if (i > n_predict || stop_requested) {
                LOG("\n");
                break;
            }

            // run backbone
            if (i > 0) {
                if (llama_decode(ctx.lctx, batch)) {
                    return error("failed to run backbone");
                }
                ctx.n_past += batch.n_tokens;
            }

            auto previous_modality = current_modality;  // track change of modality
            if (current_modality == Modality::TEXT) {
                llama_token next_text_token = 0;
                next_text_token             = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
                common_sampler_accept(ctx.smpl, next_text_token, true);

                if (llama_vocab_is_eog(ctx.vocab, next_text_token)) {
                    LOG("\n");
                    break;  // end of generation
                }

                current_modality = text_handler(next_text_token, current_modality);

                if (next_text_token != 130 && next_text_token != 128) {  // text_end, audio_start
                    auto token_str = common_token_to_piece(ctx.lctx, next_text_token);
                    text_callback(token_str);
                    LOG("%s", token_str.c_str());
                    fflush(stdout);
                }

                batch.token = &next_text_token;
                batch.embd  = nullptr;
            } else if (current_modality == Modality::AUDIO_OUT) {
                std::memcpy(embd.data(), llama_get_embeddings(ctx.lctx), sizeof(float) * embd.size());

                GGML_ASSERT(decoder);

                auto          t0         = ggml_time_ms();
                audio_token_t next_token = decoder->sample_audio_frame(embd, ctx.audio_temperature, ctx.audio_top_k);
                if (ctx.verbosity) {
                    LOG_INF("audio frame sampled in %" PRId64 " ms\n", ggml_time_ms() - t0);
                }

                current_modality = audio_handler(next_token, current_modality);

                if (next_token[0] == 2048) {
                    current_modality = Modality::TEXT;
                    std::fill(next_token.begin(), next_token.end(), 2048);
                } else {
                    auto decoded = detokenize(next_token);
                    audio_callback(decoded);
                }

                embd        = decoder->embed(next_token);
                batch.embd  = embd.data();
                batch.token = nullptr;
            }

            if (previous_modality != current_modality) {
                llama_set_embeddings(ctx.lctx, current_modality == Modality::AUDIO_OUT);
            }

            if (stop_requested) {
                LOG("\n");
                break;
            }
        }
        LOG("\n");

        return 0;
    }

    std::vector<float> detokenize(const audio_token_t & codes) const {
        // embed_for_detokenizer, converts 8 audio codes into 6 embeddings for lfm2
        int  n_tokens = 6;
        auto embd     = decoder->embed_for_detokenizer(codes);

        const int   n_out = llama_model_n_embd_out(ctx.audio_tokenizer_model);
        llama_batch batch = llama_batch_get_one(nullptr, n_tokens);

        batch.embd = embd.data();

        if (llama_decode(ctx.audio_tokenizer_lctx, batch)) {
            LOG_ERR("failed to run audio tokenizer\n");
            exit(1);
        }

        std::vector<float> output(n_tokens * n_out);
        std::memcpy(output.data(), llama_get_embeddings(ctx.audio_tokenizer_lctx), sizeof(float) * output.size());

        return istft(output);
    }

    std::vector<float> istft(const std::vector<float> & embd) const {
        const int n_fft_bins    = istft_config.n_fft / 2 + 1;
        int       n_frames      = embd.size() / (n_fft_bins * 2);
        int       output_length = (n_frames - 1) * istft_config.hop_length;

        std::vector<float> output;
        output.reserve(output_length);

        // Perform ISTFT - process each frame
        for (int i = 0; i < n_frames; i++) {
            std::vector<float> frame_spectrum(n_fft_bins * 2);

            // Extract frame spectrum from embd (which is in [n_fft_bins × n_frames × 2] format)
            for (int j = 0; j < n_fft_bins; j++) {
                const auto log_abs        = embd[i * n_fft_bins * 2 + 0 * n_fft_bins + j];
                const auto angle          = embd[i * n_fft_bins * 2 + 1 * n_fft_bins + j];
                const auto p              = std::polar(expf(log_abs), angle);
                frame_spectrum[j * 2 + 0] = p.real();
                frame_spectrum[j * 2 + 1] = p.imag();
            }

            auto frame_output = istft_state->process_frame(frame_spectrum.data());
            output.insert(output.end(), frame_output.begin(), frame_output.end());
        }

        return output;
    }

    int generate_interleaved(int                      n_predict,
                             const text_callback_t &  text_callback,
                             const audio_callback_t & audio_callback) {
        constexpr auto interleaved_n_text  = 6;
        constexpr auto interleaved_n_audio = 12;
        int            modality_left       = interleaved_n_text;
        bool           text_done           = false;

        return generate_common(
            n_predict,
            [&](const llama_token & next_text_token, Modality current_modality) {
                modality_left -= 1;
                if (next_text_token == 130) {  // <|text_end|>
                    text_done = true;
                }

                if (modality_left == 0 or text_done) {
                    modality_left    = interleaved_n_audio;
                    current_modality = Modality::AUDIO_OUT;
                }

                return current_modality;
            },
            [&](const audio_token_t & next_token, Modality current_modality) {
                if (ctx.verbosity) {
                    log_audio_tokens(next_token);
                }

                modality_left -= 1;
                if (modality_left == 0 and not text_done) {
                    current_modality = Modality::TEXT;
                    modality_left    = interleaved_n_text;
                }
                return current_modality;
            },
            text_callback, audio_callback);
    }

    int generate_sequential(int                      n_predict,
                            const text_callback_t &  text_callback,
                            const audio_callback_t & audio_callback) {
        return generate_common(
            n_predict,
            [&](const llama_token & next_text_token, Modality current_modality) {
                if (next_text_token == 128) {  // <|audio_start|>
                    current_modality = Modality::AUDIO_OUT;
                }
                return current_modality;
            },
            [&](const audio_token_t & next_token, Modality current_modality) {
                if (ctx.verbosity) {
                    log_audio_tokens(next_token);
                }
                return current_modality;
            },
            text_callback, audio_callback);
    }

    void perf_context_reset() {
        first_audio_received = std::nullopt;
        first_text_received  = std::nullopt;
        last_audio_received  = std::nullopt;
        last_text_received   = std::nullopt;
        text_tokens_count    = 0;
        audio_samples_count  = 0;
    }

    int eval_messages(const std::vector<common_chat_msg> & msgs, bool add_bos = false) {
        common_chat_templates_inputs tmpl_inputs;
        tmpl_inputs.messages              = msgs;
        tmpl_inputs.add_generation_prompt = true;
        auto formatted_chat               = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);
        LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.prompt.c_str());

        mtmd_input_text text;
        text.text          = formatted_chat.prompt.c_str();
        text.add_special   = add_bos;
        text.parse_special = true;

        if (stop_requested) {
            return 0;
        }

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        auto               bitmaps_c_ptr = ctx.bitmaps.c_ptr();
        int32_t            res           = mtmd_tokenize(ctx.mtmd_ctx_audio.get(),
                                                         chunks.ptr.get(),  // output
                                                         &text,             // text
                                                         bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
        if (res != 0) {
            return error("Unable to tokenize prompt");
        }

        ctx.bitmaps.entries.clear();

        size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
        if (n_chunks == 0) {
            return error("no chunks to eval");
        }

        for (size_t i = 0; i < n_chunks; i++) {
            bool         chunk_logits_last = (i == n_chunks - 1);
            const auto * chunk             = mtmd_input_chunks_get(chunks.ptr.get(), i);

            int32_t res = mtmd_helper_eval_chunk_single(ctx.mtmd_ctx_audio.get(), ctx.lctx, chunk, ctx.n_past, 0,
                                                        ctx.n_batch, chunk_logits_last, &ctx.n_past);
            if (res != 0) {
                return error("failed to eval chunk");
            }
        }

        LOG("\n");

        return 0;
    }

    static void log_audio_tokens(const audio_token_t & next_token) {
        LOG_INF("audio tokens: ");
        bool first = true;
        for (auto t : next_token) {
            if (first) {
                first = false;
            } else {
                LOG(", ");
            }
            LOG("%d", t);
        }
        LOG("\n");
        fflush(stdout);
    }
};

// forward to impl_
Runner::Runner() : impl_(std::make_unique<RunnerImpl>()) {}

Runner::~Runner() = default;

int Runner::get_output_sample_rate() const {
    return impl_->get_output_sample_rate();
}

const char * Runner::get_last_error() const {
    return impl_->get_last_error();
}

void Runner::stop() {
    impl_->stop();
}

int Runner::generate(const std::vector<Message> & messages,
                     int                          n_predict,
                     const text_callback_t &      text_callback,
                     const audio_callback_t &     audio_callback) {
    return impl_->generate(messages, n_predict, text_callback, audio_callback);
}

int Runner::init(common_params params) {
    return impl_->init(std::move(params));
}

void Runner::reset() {
    impl_->reset();
}

}  // namespace audio
}  // namespace liquid

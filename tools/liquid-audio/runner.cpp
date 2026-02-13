// mtmd-audio.h must be included before common.h due to conflicting declarations of string_replace_all
#include "mtmd-audio.h"
//
#include "runner.h"
//
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <algorithm>
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

    int n_batch;
    int verbosity = 0;

    mtmd::bitmaps bitmaps;

    common_chat_templates_ptr tmpls;

    int init(common_params & params) {
        // backbone
        llama_init = common_init_from_params(params);
        model      = llama_init->model();
        lctx       = llama_init->context();

        if (!model || !lctx) {
            LOG_ERR("Failed to load backbone\n");
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
        const bool has_vocoder    = !params.vocoder.model.path.empty();
        const bool has_detokenizer = !params.vocoder.speaker_file.empty();
        const bool enable_audio_output = has_vocoder && has_detokenizer;
        if (enable_audio_output) {
            mparams.vocoder_path   = params.vocoder.model.path.c_str();
            mparams.tokenizer_path = params.vocoder.speaker_file.c_str();
        } else if (has_vocoder || has_detokenizer) {
            LOG_WRN("%s: audio output disabled: both -mv (vocoder) and --tts-speaker-file (audio detokenizer) are required\n",
                    __func__);
        }
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

    int generate(const std::vector<Message> &              messages,
                 int                                       n_predict,
                 const text_callback_t &                   text_callback,
                 const audio_callback_t &                  audio_callback,
                 const std::vector<mtmd_output_modality> & modalities) {
        const bool audio_output_supported = mtmd_support_audio_output(ctx.mtmd_ctx_audio.get());
        if (audio_output_supported) {
            mtmd_set_output_modalities(ctx.mtmd_ctx_audio.get(), modalities.data(), modalities.size());
            mtmd_audio_output_start_new_turn(ctx.mtmd_ctx_audio.get());
        } else {
            bool requested_audio = false;
            for (const auto modality : modalities) {
                if (modality == MTMD_OUTPUT_MODALITY_AUDIO) {
                    requested_audio = true;
                    break;
                }
            }
            if (requested_audio) {
                LOG_WRN("%s: requested audio output, but vocoder/audio detokenizer are not available; falling back to text-only output\n",
                        __func__);
            }
        }

        std::vector<common_chat_msg> msgs;
        for (const auto & message : messages) {
            if (message.role == "user") {
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

        if (!stop_requested && generate_common(n_predict, text_callback_perf, audio_callback_perf) != 0) {
            return error("failed to generate");
        }

        perf_context_print();

        return 0;
    }

    int init(common_params params) {
        for (const auto & [p, desc] : {
                 std::pair{ params.model.path,         "-m"       },
                 std::pair{ params.mmproj.path,        "--mmproj" },
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

        reset();

        return 0;
    }

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
    }

    int get_output_sample_rate() const {
        if (!mtmd_support_audio_output(ctx.mtmd_ctx_audio.get())) {
            return 0;
        }
        return mtmd_audio_output_get_sample_rate(ctx.mtmd_ctx_audio.get());
    }

  private:
    audio_context ctx;

    std::atomic<bool> stop_requested = false;
    std::string       last_error_;

    // perf
    size_t                 text_tokens_count = 0, audio_samples_count = 0;
    std::optional<int64_t> first_text_received, first_audio_received;
    std::optional<int64_t> last_text_received, last_audio_received;

    int error(const std::string & msg) {
        LOG_ERR("ERR: %s\n", msg.c_str());
        last_error_ = msg;
        return 1;
    }

    int generate_common(int n_predict, const text_callback_t & text_callback, const audio_callback_t & audio_callback) {
        llama_batch batch = llama_batch_get_one(nullptr, 1);  // doesn't own pointers, no need for free.

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

            auto * mctx = ctx.mtmd_ctx_audio.get();

            if (mtmd_get_output_modality(mctx) == MTMD_OUTPUT_MODALITY_TEXT) {
                llama_token next_text_token = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
                common_sampler_accept(ctx.smpl, next_text_token, true);

                if (llama_vocab_is_eog(ctx.vocab, next_text_token)) {
                    LOG("\n");
                    break;  // end of generation
                }

                // output
                if (auto token_str = common_token_to_piece(ctx.lctx, next_text_token, false); !token_str.empty()) {
                    text_callback(token_str);
                    LOG("%s", token_str.c_str());
                    fflush(stdout);
                }

                mtmd_audio_output_accept_token(mctx, next_text_token);

                batch.token = &next_text_token;
                batch.embd  = nullptr;
            } else if (mtmd_get_output_modality(mctx) == MTMD_OUTPUT_MODALITY_AUDIO) {
                int res = mtmd_audio_output_decode(mctx, llama_get_embeddings(ctx.lctx), llama_model_n_embd(ctx.model),
                                                   embd.data());
                GGML_ASSERT(res == 0);
                auto                 n_samples = mtmd_get_n_audio_samples(mctx);
                std::vector<int16_t> samples(n_samples);
                mtmd_get_audio_samples(mctx, samples.data());
                audio_callback(samples);

                batch.embd  = embd.data();
                batch.token = nullptr;
            }

            llama_set_embeddings(ctx.lctx, mtmd_get_output_modality(mctx) == MTMD_OUTPUT_MODALITY_AUDIO);

            if (stop_requested) {
                LOG("\n");
                break;
            }
        }
        LOG("\n");

        return 0;
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

int Runner::generate(const std::vector<Message> &              messages,
                     int                                       n_predict,
                     const text_callback_t &                   text_callback,
                     const audio_callback_t &                  audio_callback,
                     const std::vector<mtmd_output_modality> & modalities) {
    return impl_->generate(messages, n_predict, text_callback, audio_callback, modalities);
}

int Runner::init(common_params params) {
    return impl_->init(std::move(params));
}

void Runner::reset() {
    impl_->reset();
}

}  // namespace audio
}  // namespace liquid

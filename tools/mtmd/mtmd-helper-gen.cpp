#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// Audio generation helpers
//

// --tts-lang codes -> language names used by the codec_language special tokens
static const std::unordered_map<std::string, std::string> tts_lang_codes = {
    { "zh", "chinese"    },
    { "en", "english"    },
    { "de", "german"     },
    { "it", "italian"    },
    { "pt", "portuguese" },
    { "es", "spanish"    },
    { "ja", "japanese"   },
    { "ko", "korean"     },
    { "fr", "french"     },
    { "ru", "russian"    },
};

static std::string tts_resolve_lang(const std::string & lang) {
    auto it = tts_lang_codes.find(lang);
    return it != tts_lang_codes.end() ? it->second : lang;
}

static llama_token find_special_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static bool batch_set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n, llama_pos pos_0, llama_seq_id seq_id) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n; i++) {
        const int32_t   idx = llama_batch_ext_add_token(batch, seq_id, tokens[i]);
        const llama_pos pos = pos_0 + i;
        if (idx < 0) {
            LOG_ERR("mtmd_helper_gen_audio: failed to add token %d to batch (error %d)\n", tokens[i], idx);
            return false;
        }
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    return true;
}

static void put_bytes(std::vector<char> & buf, const void * p, size_t n) {
    const char * c = (const char *) p;
    buf.insert(buf.end(), c, c + n);
}

// data_sz == UINT32_MAX writes the "unknown length" sentinel (streaming), same as ffmpeg does on a pipe
// multi-channel PCM is interleaved
static void write_wav16_header(std::vector<char> & buf, uint32_t data_sz, int32_t rate, uint16_t ch = 1) {
    const uint32_t riff_sz   = data_sz == UINT32_MAX ? UINT32_MAX : 36 + data_sz;
    const uint32_t fmt_sz    = 16, byte_rate = (uint32_t) rate * 2 * ch;
    const uint16_t fmt = 1, align = 2 * ch, bits = 16;
    const uint32_t rate32    = (uint32_t) rate;
    put_bytes(buf, "RIFF", 4); put_bytes(buf, &riff_sz, 4); put_bytes(buf, "WAVE", 4);
    put_bytes(buf, "fmt ", 4); put_bytes(buf, &fmt_sz, 4);
    put_bytes(buf, &fmt, 2); put_bytes(buf, &ch, 2); put_bytes(buf, &rate32, 4);
    put_bytes(buf, &byte_rate, 4); put_bytes(buf, &align, 2); put_bytes(buf, &bits, 2);
    put_bytes(buf, "data", 4); put_bytes(buf, &data_sz, 4);
}

static void append_wav16_pcm(std::vector<char> & buf, const float * pcm, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int16_t s = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
        put_bytes(buf, &s, 2);
    }
}

static bool write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate, uint16_t ch = 1) {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
    write_wav16_header(buf, (uint32_t) (pcm.size() * 2), rate, ch);
    append_wav16_pcm(buf, pcm.data(), pcm.size());
    return true;
}

class mtmd_gen_audio_pipeline {
public:
    mtmd_gen_audio_pipeline(llama_context * lctx, mtmd_context * mctx)
        : lctx(lctx), mctx(mctx), model(llama_get_model(lctx)), vocab(llama_model_get_vocab(model)),
          n_embd(llama_model_n_embd(model)), info(mtmd_gen_audio_get_info(mctx)) {}
    virtual ~mtmd_gen_audio_pipeline() = default;

    // false if the backbone context cannot run the pipeline, checked by mtmd_helper_gen_audio_init()
    virtual bool check_context() const { return true; }
    // see mtmd_helper_gen_audio_max_frames()
    virtual int32_t max_frames() const { return 512; }
    virtual void reset() = 0;
    virtual int32_t set_input(const mtmd_helper_gen_audio_inp * inp) = 0;
    // decodes at most n_batch prompt tokens; returns remaining count (0 = done), <0 on error
    virtual int32_t step_prompt(int32_t n_batch) = 0;
    // sampled can be LLAMA_TOKEN_NULL for pipelines with no discrete backbone token,
    // those read what they need from h_state_in instead
    // set out_stop on end-of-speech, h_state_out must be null if no frame is generated
    virtual int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) = 0;
    virtual int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) = 0;
    // forces any buffered codes through code2wav now, regardless of window_frames
    virtual int32_t flush() = 0;
    // continuing a sequence (inp->n_past > 0) with the model's own text/audio switching
    virtual bool supports_continue() const { return false; }
    // mtmd_helper_gen_audio_inp::speed and loudness other than the defaults
    virtual bool supports_delivery() const { return false; }
    // speed alone, for models without loudness control
    virtual bool supports_speed() const { return supports_delivery(); }
    // mtmd_helper_gen_audio_inp::cfg_scale other than the default
    virtual bool supports_cfg() const { return false; }
    // mtmd_helper_gen_audio_inp::flow_steps other than the default
    virtual bool supports_flow_steps() const { return false; }
    virtual bool next_is_text() const { return false; }

    std::vector<llama_logit_bias> logit_bias; // see mtmd_helper_gen_audio_get_logit_bias()

protected:
    // one state per generation, so a fixed seed always gives the same output
    bool init_gen_state(uint32_t seed) {
        mtmd_gen_state_params params = mtmd_gen_state_params_default();
        params.seed = seed;
        gen_state.reset(mtmd_gen_state_init(mctx, params));
        if (!gen_state) {
            LOG_ERR("mtmd_helper_gen_audio: failed to create the gen state\n");
            return false;
        }
        return true;
    }

    // encodes a speaker clip or a voice, out receives the encoder embeddings
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_bitmap_is_voice(bitmap) && !mtmd_support_audio(mctx)) {
            LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n");
            return false;
        }
        const std::string  marker = mtmd_get_marker(mctx);
        mtmd_input_text     text{ marker.c_str(), marker.size(), false, true };
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bptr = bitmap;
        bool ok = mtmd_tokenize(mctx, chunks, &text, &bptr, 1) == 0;
        if (ok) {
            ok = false;
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    continue;
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
                    break;
                }
                const float * embd = mtmd_get_output_embd(mctx);
                const size_t  n = (size_t) llama_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk);
                out.assign(embd, embd + n);
                ok = true;
                break;
            }
        }
        mtmd_input_chunks_free(chunks);
        return ok;
    }

    llama_context * lctx;
    mtmd_context  * mctx;
    const llama_model * model;
    const llama_vocab  * vocab;
    int n_embd;
    mtmd_gen_audio_info info;
    mtmd::gen_state_ptr gen_state;
};

// Qwen3-TTS: backbone samples codec_0, code_predictor gives the other 15 codebooks,
// then code2wav decodes them to PCM
class qwen3tts_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        seq_id = 0;
        pos = 0;
        codes_buf.clear();
        next_window_frames = window_frames;
        gen_state.reset();
        audio_pcm.clear();
        overlay.clear();
        h_state_buf.clear();
        out_buf.clear();
        prompt_embd_buf.clear();
        prompt_batch.reset();
        n_prompt = 0;
        prompt_pos = 0;
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id = inp->seq_id;

        if (!ensure_cache()) {
            return 1;
        }

        const std::string lang   = tts_resolve_lang((inp->lang && inp->lang[0]) ? inp->lang : "english");
        const llama_token c_lang = find_special_token(vocab, ("<|codec_language_" + lang + "|>").c_str());
        if (c_lang == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: unknown language '%s'\n", lang.c_str());
            return 1;
        }

        std::vector<float> speaker_embd;
        if (inp->speaker_ref) {
            mtmd_bitmap_set_transcript(inp->speaker_ref, inp->speaker_ref_text);
            if (!encode_speaker(inp->speaker_ref, speaker_embd)) {
                return 1;
            }
        }

        const int n_e = n_embd;

        // cloning with the reference transcript continues the reference speech, see generate_icl_prompt() upstream
        const bool   has_ref_text = inp->speaker_ref_text && inp->speaker_ref_text[0];
        const size_t n_ref_frames = speaker_embd.empty() ? 0 : speaker_embd.size() / (size_t) n_e - 1;
        const bool   icl          = has_ref_text && n_ref_frames > 0;
        if (has_ref_text && !icl) {
            LOG_WRN("mtmd_helper_gen_audio: no reference codes (mmproj without reference encoder, or clip too long), cloning from the speaker embedding only\n");
        }
        auto row = [&](llama_token t) {
            return std::vector<float>(tok_embd.begin() + (size_t) t * n_e,
                                       tok_embd.begin() + (size_t) (t + 1) * n_e);
        };
        auto sum_row = [&](llama_token a, llama_token b) {
            std::vector<float> va = row(a), vb = row(b);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };
        auto sum_vec = [&](llama_token a, const std::vector<float> & vb) {
            std::vector<float> va = row(a);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };

        // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
        const std::string full = "<|im_start|>assistant\n" + std::string(inp->prompt, inp->prompt_len) +
                                  "<|im_end|>\n<|im_start|>assistant\n";
        auto tokenize = [&](const std::string & text) {
            std::vector<llama_token> tokens(text.size() + 16);
            const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                                         false, true);
            tokens.resize(n < 0 ? 0 : (size_t) n);
            return tokens;
        };

        std::vector<llama_token> ids = tokenize(full);
        const int n_ids = (int) ids.size();
        if (n_ids < 8) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }

        // the reference transcript goes before the text, with the same chat wrap minus the next turn: slices [3:-2]
        std::vector<llama_token> ref_ids;
        if (icl) {
            ref_ids = tokenize("<|im_start|>assistant\n" + std::string(inp->speaker_ref_text) + "<|im_end|>\n");
            if (ref_ids.size() < 6) {
                LOG_ERR("mtmd_helper_gen_audio: tokenization of the reference text failed\n");
                return 1;
            }
            ref_ids = std::vector<llama_token>(ref_ids.begin() + 3, ref_ids.end() - 2);
        }

        std::vector<std::vector<float>> prompt;
        for (int i = 0; i < 3; i++) prompt.push_back(row(ids[(size_t) i]));
        prompt.push_back(sum_row(tts_pad, c_think));
        prompt.push_back(sum_row(tts_pad, c_think_b));
        prompt.push_back(sum_row(tts_pad, c_lang));
        prompt.push_back(sum_row(tts_pad, c_think_e));
        if (!speaker_embd.empty()) prompt.push_back(sum_vec(tts_pad, speaker_embd));
        prompt.push_back(sum_row(tts_bos, codec_pad));
        for (llama_token t : ref_ids) prompt.push_back(sum_row(t, codec_pad));
        for (int i = 3; i < n_ids - 5; i++) prompt.push_back(sum_row(ids[(size_t) i], codec_pad));
        prompt.push_back(sum_row(tts_eos, codec_pad));
        prompt.push_back(sum_row(tts_pad, codec_bos));
        // the reference frames, as if the talker had just spoken them
        if (icl) {
            for (size_t f = 0; f < n_ref_frames; f++) {
                const float * frame = speaker_embd.data() + (f + 1) * (size_t) n_e;
                prompt.push_back(sum_vec(tts_pad, std::vector<float>(frame, frame + n_e)));
            }
        }

        n_prompt = (int) prompt.size();

        // the talker uses the qwen3vl interleaved mrope, all sections are equal for a text/codec stream
        mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ||
                llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;
        const int n_pos_per_embd = mrope ? 4 : 1;

        prompt_embd_buf.resize((size_t) n_prompt * (size_t) n_e);
        for (int i = 0; i < n_prompt; i++) {
            memcpy(prompt_embd_buf.data() + (size_t) i * n_e, prompt[(size_t) i].data(), (size_t) n_e * sizeof(float));
        }

        prompt_batch.reset(new decode_embd_batch(prompt_embd_buf.data(), n_prompt, n_pos_per_embd, n_e));
        if (mrope) prompt_batch->set_position_mrope_1d(0, seq_id);
        else       prompt_batch->set_position_normal  (0, seq_id);
        prompt_pos = 0;

        pos = 0;
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        top_k = inp->top_k > 0 ? inp->top_k : def.top_k;
        top_p = inp->top_p > 0 ? inp->top_p : def.top_p;
        out_type = inp->out_type;
        stream = inp->stream;
        if (!init_gen_state(inp->seed)) {
            return 1;
        }
        // the decoder continues from the reference codes of the encode above
        if (icl) {
            if (mtmd_gen_state_set_ref(mctx, gen_state.get()) != 0) {
                return 1;
            }
            next_window_frames = window_frames - std::min(n_ref_frames, window_frames - 1);
        }

        // the prompt above holds the whole text stream up to tts_eos, so every generated
        // frame adds tts_pad on top of the codes embedding
        overlay = row(tts_pad);

        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        const int32_t n_tokens_batch = std::min(n_batch, n_prompt - prompt_pos);

        const bool is_last_batch = (prompt_pos + n_tokens_batch) == n_prompt;
        if (is_last_batch) {
            prompt_batch->logits[prompt_pos + n_tokens_batch - 1] = 1;
        }

        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, prompt_batch->render(lctx, prompt_pos, n_tokens_batch)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }

        pos        += n_tokens_batch;
        prompt_pos += n_tokens_batch;

        if (prompt_pos >= n_prompt) {
            // prompt fully processed, its embedding buffer is no longer needed
            prompt_batch.reset();
            prompt_embd_buf.clear();
            return 0;
        }
        return n_prompt - prompt_pos;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        if (sampled == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: qwen3tts requires a token sampled from the backbone\n");
            return 1;
        }

        // backbone signals end-of-speech with a token, no frame for this step
        if (sampled == codec_eos || llama_vocab_is_eog(vocab, sampled)) {
            *out_stop    = true;
            *h_state_out = nullptr;
            return 0;
        }

        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type  = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0 = sampled - codec_0;
        inp.embd  = const_cast<float *>(h_state_in);
        inp.top_k = top_k;
        inp.top_p = top_p;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_code process failed\n");
            return 1;
        }

        codes_buf.insert(codes_buf.end(), out.codes, out.codes + out.n_codes);
        if (out.n_codes > 0 && codes_buf.size() / out.n_codes >= next_window_frames) {
            if (!flush_gen_wav()) {
                return 1;
            }
        }

        std::vector<float> fb(out.embd, out.embd + n_embd);
        for (int i = 0; i < n_embd; i++) fb[(size_t) i] += overlay[(size_t) i];

        const int n_pos_per_embd = mrope ? 4 : 1;
        decode_embd_batch batch_embd(fb.data(), 1, n_pos_per_embd, n_embd);
        if (mrope) batch_embd.set_position_mrope_1d(pos, seq_id);
        else       batch_embd.set_position_normal  (pos, seq_id);
        batch_embd.logits[0] = 1;
        pos++;

        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();

        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            // one-shot call: force out whatever's left, regardless of window_frames
            if (!flush_gen_wav()) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: only return audio produced since the previous call
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }

        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }

        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        return flush_gen_wav() ? 0 : 1;
    }

private:
    bool ensure_cache() {
        if (specials_ok) {
            return true;
        }
        codec_0   = find_special_token(vocab, "<|codec_0|>");
        codec_bos = find_special_token(vocab, "<|codec_bos|>");
        codec_eos = find_special_token(vocab, "<|codec_eos_token|>");
        codec_pad = find_special_token(vocab, "<|codec_pad|>");
        c_think   = find_special_token(vocab, "<|codec_think|>");
        c_think_b = find_special_token(vocab, "<|codec_think_bos|>");
        c_think_e = find_special_token(vocab, "<|codec_think_eos|>");
        tts_pad   = find_special_token(vocab, "<tts_pad>");
        tts_bos   = find_special_token(vocab, "<tts_text_bos>");
        tts_eos   = find_special_token(vocab, "<tts_text_eod>");
        for (llama_token t : { codec_0, codec_bos, codec_eos, codec_pad,
                               c_think, c_think_b, c_think_e,
                               tts_pad, tts_bos, tts_eos }) {
            if (t == LLAMA_TOKEN_NULL) {
                LOG_ERR("mtmd_helper_gen_audio: missing a required special token in vocab\n");
                return false;
            }
        }
        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd == 0) {
            LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        if (llama_model_get_tok_embd(model, tok_embd.data()) != n_tok_embd) {
            LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
            return false;
        }
        specials_ok = true;
        return true;
    }

    // one GEN_WAV process() call over the buffered codes, state is carried across batches
    bool flush_gen_wav() {
        if (codes_buf.empty()) {
            return true;
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes      = codes_buf.data();
        inp.n_codes    = codes_buf.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_wav process failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        codes_buf.clear();
        next_window_frames = window_frames;
        return true;
    }

    // vocab specials fixed across the whole session, looked up once
    bool specials_ok = false;
    llama_token codec_0    = LLAMA_TOKEN_NULL;
    llama_token codec_bos  = LLAMA_TOKEN_NULL;
    llama_token codec_eos  = LLAMA_TOKEN_NULL;
    llama_token codec_pad  = LLAMA_TOKEN_NULL;
    llama_token c_think    = LLAMA_TOKEN_NULL;
    llama_token c_think_b  = LLAMA_TOKEN_NULL;
    llama_token c_think_e  = LLAMA_TOKEN_NULL;
    llama_token tts_pad    = LLAMA_TOKEN_NULL;
    llama_token tts_bos    = LLAMA_TOKEN_NULL;
    llama_token tts_eos    = LLAMA_TOKEN_NULL;
    std::vector<float> tok_embd; // whole token embedding matrix, n_vocab * n_embd

    // must match hparams.wav_tfm_swa hardcoded in mtmd-model.cpp
    size_t window_frames = 72;

    // per-generation state, cleared by reset()
    llama_seq_id seq_id = 0;
    bool mrope = false;
    int pos = 0;
    // prompt decode state, consumed batch-by-batch by step_prompt()
    std::vector<float> prompt_embd_buf;
    std::unique_ptr<decode_embd_batch> prompt_batch;
    int n_prompt = 0;
    int prompt_pos = 0;
    int32_t  top_k = 50;
    float    top_p = 1.0f;
    std::vector<int32_t> codes_buf;
    size_t next_window_frames = window_frames; // the first window leaves room for the reference codes
    std::vector<float>   audio_pcm;
    std::vector<float> overlay;
    std::vector<float> h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0; // samples already returned by get_output()
    bool   wav_header_sent = false;
};

// settings that only live in the reference's per-pack yaml, not in the checkpoint
// the english packs share the same shapes and tokenizer, but disagree on these
// both are 0 / false when the pack does not tune them, the model default is then used
struct pockettts_pack_settings {
    int  frames_after_eos = 0;
    bool pad_short_text   = false;
};

static pockettts_pack_settings pockettts_pack(const char * variant) {
    static const std::unordered_map<std::string, pockettts_pack_settings> packs = {
        { "english",         { 0, false } },
        { "english_2026-01", { 0, true  } },
        { "english_2026-04", { 0, false } },
        { "french_24l",      { 8, false } },
        { "italian",         { 0, false } },
    };
    auto it = packs.find(variant ? variant : "");
    if (it == packs.end()) {
        LOG_WRN("mtmd_helper_gen_audio: no tuned settings for pocket-tts variant \"%s\"\n",
                variant ? variant : "");
        return {};
    }
    return it->second;
}

// pocket-tts: the backbone emits no token, the flow net turns each hidden state into a latent
// the end-of-speech head also lives in the mmproj
class pockettts_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        seq_id = 0;
        pos = 0;
        feats_buf.clear();
        gen_state.reset();
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
        prompt_embd_buf.clear();
        prompt_batch.reset();
        n_prompt = 0;
        prompt_pos = 0;
        step_idx = 0;
        eos_step = -1;
        chunks.clear();
        chunk_idx = 0;
        n_voice_pos = 0;
        chunk_budget = 0;
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id = inp->seq_id;

        if (!ensure_cache()) {
            return 1;
        }

        std::vector<float> voice;
        if (inp->speaker_ref) {
            if (!encode_speaker(inp->speaker_ref, voice)) {
                return 1;
            }
        }

        pack = pockettts_pack(info.model_variant);

        const std::string text = prepare_text(std::string(inp->prompt, inp->prompt_len),
                                              pack.pad_short_text);
        if (text.empty()) {
            LOG_ERR("mtmd_helper_gen_audio: empty prompt\n");
            return 1;
        }

        std::vector<llama_token> ids(text.size() + 16);
        int n_ids = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(),
                                   (int32_t) ids.size(), false, false);
        if (n_ids <= 0) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);

        // long inputs degrade badly, so each chunk restarts from the voice conditioning
        // see split_into_best_sentences() in the reference
        chunks = split_chunks(ids);
        chunk_idx = 0;
        if (chunks.size() > 1) {
            LOG_INF("mtmd_helper_gen_audio: %d tokens split into %zu chunks\n", n_ids, chunks.size());
        }

        const int n_e = n_embd;

        // sequence order is voice, then text, then the audio BOS that starts generation
        if (!voice.empty()) {
            GGML_ASSERT(voice.size() % (size_t) n_e == 0);
            if (bos_before_voice != LLAMA_TOKEN_NULL) {
                push_embd_row(prompt_embd_buf, bos_before_voice);
            }
            prompt_embd_buf.insert(prompt_embd_buf.end(), voice.begin(), voice.end());
        }
        // every later chunk rewinds to here and re-prompts, so the voice stays primed
        n_voice_pos = (int) (prompt_embd_buf.size() / (size_t) n_e);

        for (llama_token t : chunks[0]) {
            push_embd_row(prompt_embd_buf, t);
        }
        push_embd_row(prompt_embd_buf, audio_bos);
        arm_chunk_budget(0);

        n_prompt = (int) (prompt_embd_buf.size() / (size_t) n_e);
        prompt_batch.reset(new decode_embd_batch(prompt_embd_buf.data(), n_prompt, 1, n_e));
        prompt_batch->set_position_normal(0, seq_id);
        prompt_pos = 0;

        temp     = inp->temp;
        out_type = inp->out_type;
        stream   = inp->stream;
        if (!init_gen_state(inp->seed)) {
            return 1;
        }

        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        const int32_t n_tokens_batch = std::min(n_batch, n_prompt - prompt_pos);

        if ((prompt_pos + n_tokens_batch) == n_prompt) {
            prompt_batch->logits[prompt_pos + n_tokens_batch - 1] = 1;
        }

        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, prompt_batch->render(lctx, prompt_pos, n_tokens_batch)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }

        pos        += n_tokens_batch;
        prompt_pos += n_tokens_batch;

        if (prompt_pos >= n_prompt) {
            prompt_batch.reset();
            prompt_embd_buf.clear();
            return 0;
        }
        return n_prompt - prompt_pos;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        (void) sampled; // the backbone output is continuous, there is no token to consume

        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd = const_cast<float *>(h_state_in);
        if (temp >= 0.0f) {
            inp.temp = temp;
        }
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: flow decode failed\n");
            return 1;
        }
        if (out.is_eos && eos_step < 0) {
            eos_step = step_idx;
        }
        // the frame of the stopping step is discarded, matching _autoregressive_generation().
        // the budget is the reference's fallback for a chunk whose eos head never fires
        const bool chunk_done = (eos_step >= 0 && step_idx >= eos_step + frames_after_eos) ||
                                step_idx >= chunk_budget;
        if (chunk_done) {
            if (eos_step < 0) {
                LOG_WRN("mtmd_helper_gen_audio: chunk %zu hit its budget without end-of-speech\n", chunk_idx);
            }
            return finish_chunk(h_state_out, out_stop);
        }

        feats_buf.insert(feats_buf.end(), out.feats, out.feats + out.n_feats);
        step_idx++;
        if (out.n_feats > 0 && feats_buf.size() / out.n_feats >= window_frames) {
            if (!flush_gen_wav()) {
                return 1;
            }
        }

        decode_embd_batch batch_embd(out.embd, 1, 1, n_embd);
        batch_embd.set_position_normal(pos, seq_id);
        batch_embd.logits[0] = 1;
        pos++;

        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();

        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            // one-shot call: force out whatever's left, regardless of window_frames
            if (!flush_gen_wav()) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: only return audio produced since the previous call
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }

        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }

        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        return flush_gen_wav() ? 0 : 1;
    }

private:
    bool ensure_cache() {
        if (specials_ok) {
            return true;
        }
        // bos_before_voice is optional, some packs do not insert it
        bos_before_voice = find_special_token(vocab, "<|bos_before_voice|>");
        audio_bos        = find_special_token(vocab, "<|audio_bos|>");
        if (audio_bos == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: missing <|audio_bos|> in vocab\n");
            return false;
        }
        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd == 0) {
            LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        if (llama_model_get_tok_embd(model, tok_embd.data()) != n_tok_embd) {
            LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
            return false;
        }
        GGML_ASSERT(n_embd > 0 && n_tok_embd % (uint32_t) n_embd == 0);
        specials_ok = true;
        return true;
    }

    // the table can be shorter than the vocab, so bound the row lookup
    void push_embd_row(std::vector<float> & dst, llama_token t) const {
        const size_t n_rows = tok_embd.size() / (size_t) n_embd;
        GGML_ASSERT(t >= 0 && (size_t) t < n_rows);
        dst.insert(dst.end(),
                   tok_embd.begin() + (size_t) t * n_embd,
                   tok_embd.begin() + (size_t) (t + 1) * n_embd);
    }

    // token ids of the pieces the reference splits on, see split_into_best_sentences().
    // the leading token is dropped, it is the tokenizer's dummy prefix
    std::vector<llama_token> punct_ids(const char * s) const {
        std::vector<llama_token> ids(16);
        const int n = llama_tokenize(vocab, s, (int32_t) strlen(s), ids.data(), (int32_t) ids.size(), false, false);
        if (n <= 1) {
            return {};
        }
        return std::vector<llama_token>(ids.begin() + 1, ids.begin() + n);
    }

    // cut after runs of boundary tokens, so punctuation stays with the sentence it ends
    static std::vector<std::vector<llama_token>> split_on(const std::vector<llama_token> & ids,
                                                          const std::vector<llama_token> & boundary) {
        std::vector<std::vector<llama_token>> out;
        size_t start = 0;
        bool prev_was_boundary = false;
        for (size_t i = 0; i < ids.size(); i++) {
            const bool is_boundary = std::find(boundary.begin(), boundary.end(), ids[i]) != boundary.end();
            if (!is_boundary && prev_was_boundary) {
                out.emplace_back(ids.begin() + start, ids.begin() + i);
                start = i;
            }
            prev_was_boundary = is_boundary;
        }
        out.emplace_back(ids.begin() + start, ids.end());
        return out;
    }

    std::vector<std::vector<llama_token>> split_chunks(const std::vector<llama_token> & ids) const {
        if ((int) ids.size() <= max_chunk_tokens) {
            return { ids };
        }
        const std::vector<llama_token> eos_punct = punct_ids(".!...?");
        const std::vector<llama_token> mid_punct = punct_ids(",;:");

        // oversized sentences are split again on weaker punctuation, else words get skipped
        std::vector<std::vector<llama_token>> segments;
        for (auto & seg : split_on(ids, eos_punct)) {
            if ((int) seg.size() <= max_chunk_tokens) {
                segments.push_back(std::move(seg));
                continue;
            }
            auto sub = split_on(seg, mid_punct);
            if (sub.size() > 1) {
                for (auto & s : sub) {
                    segments.push_back(std::move(s));
                }
            } else {
                segments.push_back(std::move(seg));
            }
        }

        std::vector<std::vector<llama_token>> out;
        for (auto & seg : segments) {
            if (seg.empty()) {
                continue;
            }
            if (!out.empty() && (int) (out.back().size() + seg.size()) <= max_chunk_tokens) {
                out.back().insert(out.back().end(), seg.begin(), seg.end());
            } else {
                out.push_back(std::move(seg));
            }
        }
        if (out.empty()) {
            out.push_back(ids);
        }
        for (const auto & c : out) {
            if ((int) c.size() > max_chunk_tokens) {
                LOG_WRN("mtmd_helper_gen_audio: chunk of %zu tokens exceeds the %d token budget, "
                        "generation may skip words\n", c.size(), max_chunk_tokens);
            }
        }
        return out;
    }

    // _estimate_max_gen_len() plus the per-chunk tail guess, both in frames
    void arm_chunk_budget(size_t idx) {
        const int n_tok = (int) chunks[idx].size();
        chunk_budget = (int) std::ceil((n_tok / 3.0 + 2.0) * frame_rate);
        // the pack may pin the tail, else the reference guesses it from the word count
        frames_after_eos = pack.frames_after_eos > 0 ? pack.frames_after_eos : (n_tok <= 6 ? 5 : 3);
        step_idx = 0;
        eos_step = -1;
    }

    // ends the current chunk and, if there is another, re-prompts it on top of the voice
    int32_t finish_chunk(const float ** h_state_out, bool * out_stop) {
        if (!flush_gen_wav()) {
            return 1;
        }
        // the decoder restarts too, the next chunk's audio is not continuous with this one
        mtmd_gen_state_reset(gen_state.get());

        if (chunk_idx + 1 >= chunks.size()) {
            *out_stop    = true;
            *h_state_out = nullptr;
            return 0;
        }
        chunk_idx++;

        // drop this chunk's text and audio, keep the voice conditioning
        llama_memory_seq_rm(llama_get_memory(lctx), seq_id, n_voice_pos, -1);
        pos = n_voice_pos;

        const int n_e = n_embd;
        prompt_embd_buf.clear();
        for (llama_token t : chunks[chunk_idx]) {
            push_embd_row(prompt_embd_buf, t);
        }
        push_embd_row(prompt_embd_buf, audio_bos);
        arm_chunk_budget(chunk_idx);

        const int n_rows = (int) (prompt_embd_buf.size() / (size_t) n_e);
        GGML_ASSERT(n_rows > 0);
        decode_embd_batch batch(prompt_embd_buf.data(), n_rows, 1, n_e);
        batch.set_position_normal(pos, seq_id);
        batch.logits[n_rows - 1] = 1;
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, n_rows)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: chunk prompt decode failed\n");
            return 1;
        }
        pos += n_rows;
        prompt_embd_buf.clear();

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();
        *out_stop    = false;
        return 0;
    }

    // same normalization as prepare_text_prompt() in the reference, it affects quality
    static std::string prepare_text(const std::string & in, bool pad_short) {
        std::string s;
        s.reserve(in.size() + 1);
        for (char c : in) {
            if (c == '\n' || c == '\r') {
                s += ' ';
            } else if (c == ';') {
                s += ',';
            } else {
                s += c;
            }
        }
        const size_t b = s.find_first_not_of(' ');
        const size_t e = s.find_last_not_of(' ');
        if (b == std::string::npos) {
            return "";
        }
        s = s.substr(b, e - b + 1);
        if (s[0] >= 'a' && s[0] <= 'z') {
            s[0] = (char) (s[0] - 'a' + 'A');
        }
        const unsigned char last = (unsigned char) s.back();
        if (std::isalnum(last)) {
            s += '.';
        }
        if (pad_short && count_words(s) < 5) {
            s = std::string(8, ' ') + s;
        }
        return s;
    }

    static int count_words(const std::string & s) {
        int n = 0;
        bool in_word = false;
        for (char c : s) {
            if (c == ' ') {
                in_word = false;
            } else if (!in_word) {
                in_word = true;
                n++;
            }
        }
        return n;
    }

    // decodes the buffered latents, the mimi decoder state carries over between calls
    bool flush_gen_wav() {
        if (feats_buf.empty()) {
            return true;
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.feats      = feats_buf.data();
        inp.n_feats    = feats_buf.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: mimi decode failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        feats_buf.clear();
        return true;
    }

    pockettts_pack_settings pack;
    float temp = -1.0f; // the caller's flow noise scale, < 0 for the model default
    bool specials_ok = false;
    llama_token bos_before_voice = LLAMA_TOKEN_NULL;
    llama_token audio_bos        = LLAMA_TOKEN_NULL;
    std::vector<float> tok_embd;

    llama_seq_id seq_id = 0;
    int pos = 0;
    std::vector<float> prompt_embd_buf;
    std::unique_ptr<decode_embd_batch> prompt_batch;
    int n_prompt = 0;
    int prompt_pos = 0;
    // end-of-speech is latched, then a few more frames are generated as tail padding
    int step_idx = 0;
    int eos_step = -1;
    int frames_after_eos = 3;
    static constexpr int max_chunk_tokens = 50;  // MAX_TOKEN_PER_CHUNK in the reference
    static constexpr double frame_rate    = 12.5;
    std::vector<std::vector<llama_token>> chunks;
    size_t chunk_idx   = 0;
    int    n_voice_pos = 0; // KV positions held by the voice conditioning
    int    chunk_budget = 0;

    // latents are decoded a window at a time, the decoder state bridges the windows
    size_t window_frames = 8;
    std::vector<float>   feats_buf;
    std::vector<float>   audio_pcm;
    std::vector<float>   h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0; // samples already returned by get_output()
    bool   wav_header_sent = false;
};

// Kokoro is not autoregressive: the backbone encodes the phonemes once, the mmproj turns its states into the whole utterance
// the prompt is a phoneme string (the output of the "tts" chat template), one token per character
class kokoro_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    // the predicted durations are divided by the speed
    bool supports_speed() const override { return true; }

    void reset() override {
        segments.clear();
        audio.clear();
        wav.clear();
        n_encoded = 0;
        decoded   = false;
        sent      = false;
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        if (!inp->prompt || inp->prompt_len == 0 || inp->seq_id < 0 || (uint32_t) inp->seq_id >= llama_n_seq_max(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: kokoro requires a prompt\n");
            return 1;
        }
        if (inp->speaker_ref && !mtmd_bitmap_is_voice(inp->speaker_ref)) {
            LOG_ERR("mtmd_helper_gen_audio: kokoro has no voice cloning, use one of its voices\n");
            return 1;
        }
        seq_id   = inp->seq_id;
        out_type = inp->out_type;
        stream   = inp->stream;
        speed    = inp->speed;

        // one token per phoneme character, unknown characters are dropped like in the reference
        if (token_of.empty()) {
            for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
                if (llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_NORMAL) {
                    token_of[llama_vocab_get_text(vocab, id)] = id;
                }
            }
        }
        std::vector<llama_token> phonemes;
        const std::string prompt(inp->prompt, inp->prompt_len);
        for (size_t i = 0; i < prompt.size();) {
            const unsigned char c = (unsigned char) prompt[i];
            const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            const auto it = token_of.find(prompt.substr(i, n));
            if (it != token_of.end()) {
                phonemes.push_back(it->second);
            }
            i += n;
        }
        if (phonemes.empty()) {
            LOG_ERR("mtmd_helper_gen_audio: the prompt has no phoneme the model knows\n");
            return 1;
        }

        // a voice is a table of style rows, the row is picked by the number of phonemes
        const char * voice = inp->speaker_ref ? mtmd_bitmap_get_id(inp->speaker_ref) : mtmd_gen_audio_voice_name(mctx, 0);
        if (!voice) {
            LOG_ERR("mtmd_helper_gen_audio: the model has no voice\n");
            return 1;
        }
        const std::string name = std::string("voice.") + voice;
        std::vector<float> table(mtmd_gen_audio_read_tensor(mctx, name.c_str(), nullptr, 0));
        if (table.size() < n_voice_row || table.size() % n_voice_row != 0 ||
            mtmd_gen_audio_read_tensor(mctx, name.c_str(), table.data(), table.size()) != table.size()) {
            LOG_ERR("mtmd_helper_gen_audio: unknown voice \"%s\"\n", voice);
            return 1;
        }

        // the backbone takes a whole segment in one batch, padded on both ends
        const size_t n_max = std::min({(size_t) llama_n_ctx_seq(lctx), (size_t) llama_n_ubatch(lctx), (size_t) llama_model_n_ctx_train(model)});
        if (n_max < 3) {
            LOG_ERR("mtmd_helper_gen_audio: the context is too small\n");
            return 1;
        }
        const llama_token pad = llama_vocab_bos(vocab);
        for (size_t pos = 0; pos < phonemes.size();) {
            const size_t len = split_len(phonemes, pos, n_max - 2);
            segment seg;
            seg.tokens.push_back(pad);
            seg.tokens.insert(seg.tokens.end(), phonemes.begin() + pos, phonemes.begin() + pos + len);
            seg.tokens.push_back(pad);
            const size_t row = std::min(len - 1, table.size() / n_voice_row - 1);
            seg.feats.assign(table.begin() + row * n_voice_row, table.begin() + (row + 1) * n_voice_row);
            segments.push_back(std::move(seg));
            pos += len;
            // a segment does not start with a space
            while (pos < phonemes.size() && is_any(phonemes[pos], " ")) {
                pos++;
            }
        }

        if (!init_gen_state(inp->seed)) {
            return 1;
        }
        return 0;
    }

    // the encoder is not causal, it takes one whole segment per call
    int32_t step_prompt(int32_t) override {
        if (segments.empty()) {
            return -1;
        }
        if (n_encoded == segments.size()) {
            return 0;
        }
        auto & seg = segments[n_encoded];
        const int n = (int) seg.tokens.size();
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), seg.tokens.data(), n, 0, seq_id)) {
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            llama_batch_ext_set_output_embd(batch.get(), i, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: kokoro backbone failed\n");
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            const float * h = llama_get_embeddings_ith(lctx, i);
            if (!h) {
                return -1;
            }
            seg.feats.insert(seg.feats.end(), h, h + n_embd);
        }
        n_encoded++;
        return (int32_t) (segments.size() - n_encoded);
    }

    // nothing is generated step by step
    int32_t step_gen(llama_token, const float *, const float ** h_state_out, bool * out_stop) override {
        *h_state_out = nullptr;
        *out_stop = true;
        return !segments.empty() && n_encoded == segments.size() ? 0 : 1;
    }

    int32_t get_output(int32_t * sample_rate, const char ** data, size_t * data_len, int64_t * n_samples) override {
        *sample_rate = info.sample_rate;
        *data_len = 0;
        if (n_samples) { *n_samples = 0; }
        if (stream) {
            // the audio comes out in one chunk after flush()
            if (!decoded || sent) {
                return 0;
            }
            sent = true;
        } else if (!decode_audio()) {
            return 1;
        }
        if (n_samples) { *n_samples = audio.size(); }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *data = (const char *) audio.data();
            *data_len = audio.size() * sizeof(float);
        } else {
            wav.clear();
            if (!write_wav16(wav, audio, info.sample_rate)) {
                return 1;
            }
            *data = wav.data();
            *data_len = wav.size();
        }
        return 0;
    }

    int32_t flush() override {
        return decode_audio() ? 0 : 1;
    }

private:
    struct segment {
        std::vector<llama_token> tokens;
        std::vector<float> feats; // the voice row, then the backbone state of each token
    };

    bool is_any(llama_token id, const char * chars) const {
        const char * text = llama_vocab_get_text(vocab, id);
        return text[0] != '\0' && strstr(chars, text) != nullptr;
    }

    // length of the segment that starts at pos: a long text is cut at the last sentence end that fits, else at a weaker break
    size_t split_len(const std::vector<llama_token> & phonemes, size_t pos, size_t n_max) const {
        if (phonemes.size() - pos <= n_max) {
            return phonemes.size() - pos;
        }
        for (const char * breaks : {"!.?\xe2\x80\xa6", ":;", ",\xe2\x80\x94", " "}) {
            for (size_t len = n_max; len > 1; len--) {
                if (is_any(phonemes[pos + len - 1], breaks)) {
                    return len;
                }
            }
        }
        return n_max;
    }

    bool decode_audio() {
        if (decoded) {
            return true;
        }
        if (segments.empty() || n_encoded != segments.size()) {
            return false;
        }
        for (const auto & seg : segments) {
            std::vector<int32_t> codes(seg.tokens.begin(), seg.tokens.end());
            mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
            inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            inp.codes   = codes.data();
            inp.n_codes = codes.size();
            inp.feats   = seg.feats.data();
            inp.n_feats = seg.feats.size();
            inp.speed   = speed;
            mtmd_gen_out out{};
            if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
                return false;
            }
            audio.insert(audio.end(), out.audio, out.audio + out.n_samples);
        }
        decoded = true;
        return true;
    }

    static constexpr size_t n_voice_row = 256; // decoder style, then predictor style

    llama_seq_id seq_id = 0;
    std::unordered_map<std::string, llama_token> token_of;
    std::vector<segment> segments;
    size_t n_encoded = 0;
    float speed = 1.0f;
    std::vector<float> audio;
    std::vector<char> wav;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    bool stream  = false;
    bool decoded = false;
    bool sent    = false;
};

// Soprano uses the hidden state that predicts each non-EOS token, including the prompt's last state.
class soprano_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        prompt.clear();
        features.clear();
        h_next.clear();
        audio.clear();
        wav.clear();
        pos = 0;
        prompt_pos = 0;
        decoded = false;
        sent = false;
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        if (inp->speaker_ref || !inp->prompt || inp->prompt_len == 0 || inp->prompt_len > INT32_MAX ||
            inp->seq_id < 0 || (uint32_t) inp->seq_id >= llama_n_seq_max(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: soprano requires text, without a speaker reference\n");
            return 1;
        }
        seq_id = inp->seq_id;
        out_type = inp->out_type;
        stream = inp->stream;
        const llama_token text = find_special_token(vocab, "[TEXT]");
        const llama_token start = find_special_token(vocab, "[START]");
        const llama_token stop = find_special_token(vocab, "[STOP]");
        if (text == LLAMA_TOKEN_NULL || start == LLAMA_TOKEN_NULL || stop == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: incompatible soprano vocabulary\n");
            return 1;
        }
        const int required = llama_tokenize(vocab, inp->prompt, inp->prompt_len, nullptr, 0, false, false);
        if (required >= 0 || (uint32_t) (-required + 3) >= llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: soprano prompt does not fit in the context\n");
            return 1;
        }
        const int n = -required;
        prompt.resize(n + 3);
        prompt[0] = stop;
        prompt[1] = text;
        if (llama_tokenize(vocab, inp->prompt, inp->prompt_len, prompt.data() + 2, n, false, false) != n) {
            return 1;
        }
        prompt.back() = start;
        if (!llama_memory_seq_rm(llama_get_memory(lctx), seq_id, 0, -1)) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset soprano sequence\n");
            return 1;
        }
        if (!init_gen_state(inp->seed)) {
            return 1;
        }
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        if (n_batch <= 0 || prompt.empty()) {
            return -1;
        }
        const int n = std::min(n_batch, (int) prompt.size() - prompt_pos);
        if (n == 0) {
            return 0;
        }
        if (!decode(prompt.data() + prompt_pos, n)) {
            return -1;
        }
        prompt_pos += n;
        return (int) prompt.size() - prompt_pos;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        *h_state_out = nullptr;
        *out_stop = false;
        if (sampled < 0 || sampled >= llama_vocab_n_tokens(vocab) || !h_state_in || prompt_pos != (int) prompt.size() || prompt.empty()) {
            return 1;
        }
        if (llama_vocab_is_eog(vocab, sampled) || (uint32_t) pos >= llama_n_ctx_seq(lctx)) {
            *out_stop = true;
            return 0;
        }
        features.insert(features.end(), h_state_in, h_state_in + n_embd);
        if (!decode(&sampled, 1)) {
            features.resize(features.size() - n_embd);
            return 1;
        }
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) {
            return 1;
        }
        h_next.assign(h, h + n_embd);
        *h_state_out = h_next.data();
        return 0;
    }

    int32_t get_output(int32_t * sample_rate, const char ** data, size_t * data_len, int64_t * n_samples) override {
        *sample_rate = info.sample_rate;
        *data_len = 0;
        if (n_samples) { *n_samples = 0; }
        if (stream) {
            // the decoder needs the whole utterance, so the audio comes out in one chunk after flush()
            if (!decoded || sent) {
                return 0;
            }
            sent = true;
        } else if (!decode_audio()) {
            return 1;
        }
        if (n_samples) { *n_samples = audio.size(); }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *data = (const char *) audio.data();
            *data_len = audio.size() * sizeof(float);
        } else {
            wav.clear();
            if (!write_wav16(wav, audio, info.sample_rate)) {
                return 1;
            }
            *data = wav.data();
            *data_len = wav.size();
        }
        return 0;
    }

    // the whole utterance is decoded at once
    int32_t flush() override {
        return decode_audio() ? 0 : 1;
    }

private:
    bool decode_audio() {
        if (decoded) {
            return true;
        }
        if (features.size() < 2 * (size_t) n_embd) {
            LOG_ERR("mtmd_helper_gen_audio: soprano needs at least two generated frames\n");
            return false;
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.feats = features.data();
        inp.n_feats = features.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            return false;
        }
        audio.assign(out.audio, out.audio + out.n_samples);
        decoded = true;
        return true;
    }

    bool decode(llama_token * tokens, int n) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), tokens, n, pos, seq_id)) {
            return false;
        }
        llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: soprano backbone decode failed\n");
            return false;
        }
        pos += n;
        return true;
    }

    llama_seq_id seq_id = 0;
    llama_pos pos = 0;
    int prompt_pos = 0;
    std::vector<llama_token> prompt;
    std::vector<float> features;
    std::vector<float> h_next;
    std::vector<float> audio;
    std::vector<char> wav;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    bool stream  = false;
    bool decoded = false;
    bool sent    = false;
};

// LFM3-Audio TTS, see generate() in the reference: each step samples a frame of codes by MaskGIT
// and feeds its embedding back, the end-of-audio code stops
class lfm3a_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        seq_id = 0;
        pos = 0;
        continuing = false;
        mode = MODE_AUDIO;
        slots_left = 0;
        speaking = false;
        text_done = false;
        prompt_tokens.clear();
        prompt_pos = 0;
        hist.clear();
        gen_state.reset();
        pending_tail = false;
        pcm_sent = 0;
        wav_header_sent = false;
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id = inp->seq_id;

        // a voice is named in the prompt instead
        if (inp->speaker_ref && !mtmd_bitmap_is_voice(inp->speaker_ref)) {
            LOG_WRN("mtmd_helper_gen_audio: the model has no voice conditioning, the speaker file is ignored\n");
        }

        // the prompt is rendered by the caller: the "tts" chat template of the model, or the next turn of a conversation
        continuing = inp->n_past > 0;
        pos        = continuing ? inp->n_past : 0;
        const std::string full(inp->prompt, inp->prompt_len);
        prompt_tokens.resize(full.size() + 16);
        const int n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), prompt_tokens.data(),
                                         (int32_t) prompt_tokens.size(), false, true);
        if (n_ids <= 0) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        prompt_tokens.resize((size_t) n_ids);

        if (!find_tokens()) {
            return 1;
        }
        // a continued response opens with the token of the asked mode, a text-only one cannot switch to audio
        if (continuing) {
            switch (inp->out_mode) {
                case MTMD_HELPER_GEN_AUDIO_OUT_AUDIO: prompt_tokens.push_back(tok_audio_start); break;
                case MTMD_HELPER_GEN_AUDIO_OUT_MIXED: prompt_tokens.push_back(tok_mixed_start); break;
                case MTMD_HELPER_GEN_AUDIO_OUT_TEXT:
                    logit_bias = { { tok_audio_start, -INFINITY }, { tok_mixed_start, -INFINITY } };
                    break;
                default: break;
            }
        }
        // the last prompt token picks the mode, see generate() of the reference
        const llama_token last = prompt_tokens.back();
        mode = last == tok_audio_start ? MODE_AUDIO : last == tok_mixed_start ? MODE_MIXED : MODE_TEXT;
        slots_left = n_text_block;
        if (!continuing && mode != MODE_AUDIO) {
            LOG_ERR("mtmd_helper_gen_audio: the prompt must come from the \"tts\" chat template of the model\n");
            return 1;
        }

        // the sampling defaults of the model, the caller can only override the temperature
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        temp        = inp->temp >= 0.0f ? inp->temp : def.temp;
        top_k       = def.top_k;
        rep_penalty = def.penalty_repeat;
        rep_window  = def.penalty_last_n;

        out_type = inp->out_type;
        stream   = inp->stream;
        rng.seed(inp->seed == UINT32_MAX ? std::random_device{}() : inp->seed);
        return init_gen_state(inp->seed) ? 0 : 1;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        const int n_prompt = (int) prompt_tokens.size();
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        const int n = std::min(n_batch, n_prompt - prompt_pos);
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), prompt_tokens.data() + prompt_pos, n, pos, seq_id)) {
            return -1;
        }
        if (prompt_pos + n == n_prompt) {
            llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        pos        += n;
        prompt_pos += n;
        return n_prompt - prompt_pos;
    }

    bool supports_continue() const override {
        return true;
    }

    bool next_is_text() const override {
        return mode == MODE_TEXT || (mode == MODE_MIXED && !speaking);
    }

    // one generate() step: a text token from the caller or an audio frame from the hidden state
    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        const bool interleaving = mode == MODE_MIXED;
        if (next_is_text()) {
            // <|im_end|> ends the response in any text slot, <|text_end|> outside an interleaved block too
            if (sampled == tok_im_end || (!interleaving && sampled == tok_text_end)) {
                *out_stop    = true;
                *h_state_out = nullptr;
                return 0;
            }
            // the switching tokens are fed back like any text token
            if (!interleaving) {
                if (sampled == tok_audio_start) {
                    mode = MODE_AUDIO;
                } else if (sampled == tok_mixed_start) {
                    mode       = MODE_MIXED;
                    slots_left = n_text_block;
                    speaking   = false;
                    text_done  = false;
                }
            } else if (sampled == tok_text_end) {
                text_done = true;
                speaking  = true;
            }
            if (!decode(sampled)) {
                return 1;
            }
        } else {
            std::vector<int32_t> codes;
            const float * fb = nullptr;
            bool is_eos = false;
            if (!sample_frame(h_state_in, codes, &fb, &is_eos)) {
                return 1;
            }
            if (is_eos) {
                if (!continuing) {
                    // TTS: the end of audio ends the generation
                    *out_stop    = true;
                    *h_state_out = nullptr;
                    return 0;
                }
                // the segment ends, the codec tail goes out and the response continues in text
                if (flush() != 0) {
                    return 1;
                }
                mode = MODE_TEXT;
            } else {
                hist.insert(hist.end(), codes.begin(), codes.end());
                if (!decode_wav(codes.data(), codes.size())) {
                    return 1;
                }
            }
            decode_embd_batch batch_embd(const_cast<float *>(fb), 1, 1, n_embd);
            batch_embd.set_position_normal(pos, seq_id);
            batch_embd.logits[0] = 1;
            if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, 0, 1)) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
                return 1;
            }
            pos++;
        }

        // a slot emitted while interleaving counts down the block, the switching token itself does not
        if (interleaving && !text_done && --slots_left == 0) {
            speaking   = !speaking;
            slots_left = speaking ? n_audio_block : n_text_block;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            if (flush() != 0) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: only the audio produced since the previous call, every frame is decoded as it is sampled
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    // the ISTFT holds back the overlap of the last frame
    int32_t flush() override {
        if (pending_tail) {
            if (!decode_wav(nullptr, 0)) {
                return 1;
            }
            pending_tail = false;
        }
        return 0;
    }

private:
    // checked by the converter
    static constexpr int   n_cb        = 12;
    static constexpr int   schedule[]  = { 4, 4, 4 }; // codebooks revealed per depthformer pass

    // a random order of the codebooks, revealed by the schedule; fb is the feedback embedding of the codes.
    // at the end of audio every code is the end code, its feedback embedding is only built when continuing
    bool sample_frame(const float * h_state, std::vector<int32_t> & codes, const float ** fb, bool * is_eos) {
        std::vector<int> order(n_cb);
        for (int i = 0; i < n_cb; i++) {
            order[(size_t) i] = i;
        }
        std::shuffle(order.begin(), order.end(), rng);

        codes.assign(n_cb, -1);
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd    = const_cast<float *>(h_state);
        inp.codes   = codes.data();
        inp.n_codes = codes.size();
        mtmd_gen_out out{};

        int k = 0;
        for (int n_pick : schedule) {
            if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0 || out.n_feats == 0 || out.n_feats % n_cb != 0) {
                LOG_ERR("mtmd_helper_gen_audio: depthformer pass failed\n");
                return false;
            }
            const size_t n_vocab = out.n_feats / n_cb;
            logits.assign(out.feats, out.feats + out.n_feats);
            for (int j = 0; j < n_pick; j++, k++) {
                const int i = order[(size_t) k];
                float * l = logits.data() + (size_t) i * n_vocab;
                apply_rep_penalty(l, n_vocab, i);
                codes[(size_t) i] = sample(l, n_vocab);
            }
        }

        // the last code of every codebook ends the audio
        const int32_t eos = (int32_t) (logits.size() / n_cb) - 1;
        *is_eos = std::find(codes.begin(), codes.end(), eos) != codes.end();
        if (*is_eos) {
            codes.assign(n_cb, eos);
            if (!continuing) {
                *fb = nullptr;
                return true;
            }
        }

        // all codes known: only the feedback embedding
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: codebook embedding failed\n");
            return false;
        }
        fb_buf.assign(out.embd, out.embd + n_embd);
        *fb = fb_buf.data();
        return true;
    }

    // CTRL penalty over the codes codebook cb took in the last rep_window frames
    void apply_rep_penalty(float * l, size_t n_vocab, int cb) const {
        if (rep_penalty <= 1.0f || rep_window <= 0) {
            return;
        }
        const size_t n_hist = hist.size() / n_cb;
        const size_t n_win  = std::min(n_hist, (size_t) rep_window);
        std::vector<bool> seen(n_vocab, false);
        for (size_t f = n_hist - n_win; f < n_hist; f++) {
            const int32_t c = hist[f * n_cb + (size_t) cb];
            if (c >= 0 && (size_t) c < n_vocab && !seen[(size_t) c]) {
                seen[(size_t) c] = true;
                l[c] = l[c] > 0.0f ? l[c] / rep_penalty : l[c] * rep_penalty;
            }
        }
    }

    // TODO: reuse the samplers of common (top-k, temperature, penalties) instead of this one
    int32_t sample(const float * l, size_t n_vocab) {
        if (temp <= 0.0f) {
            return (int32_t) (std::max_element(l, l + n_vocab) - l);
        }
        // top-k keeps the codes scoring at least the k-th best, ties included
        float lmin = -INFINITY;
        if (top_k > 0 && (size_t) top_k < n_vocab) {
            std::vector<float> sorted(l, l + n_vocab);
            std::nth_element(sorted.begin(), sorted.begin() + (top_k - 1), sorted.end(), std::greater<float>());
            lmin = sorted[(size_t) top_k - 1];
        }
        const float lmax = *std::max_element(l, l + n_vocab);
        std::vector<double> prob(n_vocab);
        double sum = 0.0;
        for (size_t v = 0; v < n_vocab; v++) {
            prob[v] = l[v] < lmin ? 0.0 : std::exp((double) (l[v] - lmax) / temp);
            sum += prob[v];
        }
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
        for (size_t v = 0; v < n_vocab; v++) {
            r -= prob[v];
            if (r <= 0.0) {
                return (int32_t) v;
            }
        }
        return (int32_t) n_vocab - 1;
    }

    // n_codes == 0 flushes the ISTFT tail
    bool decode_wav(const int32_t * codes, size_t n_codes) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type       = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes      = const_cast<int32_t *>(codes);
        inp.n_codes    = n_codes;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: codec decode failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        pending_tail = n_codes > 0;
        return true;
    }

    bool find_tokens() {
        tok_audio_start = find_special_token(vocab, "<|audio_start|>");
        tok_mixed_start = find_special_token(vocab, "<|mixed_start|>");
        tok_text_end    = find_special_token(vocab, "<|text_end|>");
        tok_im_end      = find_special_token(vocab, "<|im_end|>");
        if (tok_audio_start == LLAMA_TOKEN_NULL || tok_mixed_start == LLAMA_TOKEN_NULL ||
            tok_text_end == LLAMA_TOKEN_NULL || tok_im_end == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: missing LFM3-Audio control tokens in vocab\n");
            return false;
        }
        return true;
    }

    bool decode(llama_token token) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), &token, 1, pos, seq_id)) {
            return false;
        }
        llama_batch_ext_set_output_logits(batch.get(), 0, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        pos += 1;
        return true;
    }

    // generate() modes: text, audio until end of audio, interleaved until <|text_end|>
    enum gen_mode { MODE_TEXT, MODE_AUDIO, MODE_MIXED };
    // checked by the converter
    static constexpr int n_text_block  = 6;  // interleaved_n_text
    static constexpr int n_audio_block = 12; // interleaved_n_audio

    llama_token tok_audio_start = LLAMA_TOKEN_NULL;
    llama_token tok_mixed_start = LLAMA_TOKEN_NULL;
    llama_token tok_text_end    = LLAMA_TOKEN_NULL;
    llama_token tok_im_end      = LLAMA_TOKEN_NULL;

    bool     continuing = false; // the prompt continues a sequence, the response follows the model's switching
    gen_mode mode       = MODE_AUDIO;
    int      slots_left = 0;
    bool     speaking   = false; // interleaving: the current block is audio
    bool     text_done  = false; // interleaving: <|text_end|> was emitted, audio until the end-of-audio frame

    llama_seq_id seq_id = 0;
    int pos = 0;
    std::vector<llama_token> prompt_tokens;
    int prompt_pos = 0;

    float   temp        = 0.0f;
    int32_t top_k       = 0;
    float   rep_penalty = 1.0f;
    int32_t rep_window  = 0; // frames
    std::mt19937 rng;
    std::vector<float>   logits;
    std::vector<float>   fb_buf;
    std::vector<int32_t> hist; // frame-major codes of the frames so far

    bool pending_tail = false; // the ISTFT holds back the overlap of the last frame
    std::vector<float>   audio_pcm;
    std::vector<float>   h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0;     // samples already returned by get_output()
    bool   wav_header_sent = false;
};

// Chatterbox: the backbone emits s3 speech tokens, then one flow-matching mel decode and the NSF-iSTFT vocoder make the PCM
// the multilingual variant runs cfg with an uncond sequence (text rows zeroed), paired as seq_id and n_seq_max / 2 + seq_id
class chatterbox_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        seq_id = 0;
        seq_uncond = 0;
        pos = 0;
        ar_idx = 0;
        codes_buf.clear();
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
        prompt_embd_buf.clear();
        uncond_embd_buf.clear();
        cond_logits.clear();
        n_prompt = 0;
        prompt_pos = 0;
        decoded = false;
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
        gen_state.reset();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id = inp->seq_id;

        if (!ensure_cache()) {
            return 1;
        }

        out_type = inp->out_type;
        stream   = inp->stream;
        if (!init_gen_state(inp->seed)) {
            return 1;
        }

        if (mtl) {
            // the uncond sequence lives half a seq space away from the cond one
            const int32_t n_seq_max = (int32_t) llama_n_seq_max(lctx);
            seq_uncond = (llama_seq_id) (n_seq_max / 2 + seq_id);
            if (!llama_model_classifier_free_guidance(llama_get_model(lctx)) || seq_id >= n_seq_max / 2) {
                LOG_ERR("mtmd_helper_gen_audio: multilingual chatterbox needs a backbone with classifier_free_guidance and a paired cfg sequence "
                        "(n_seq_max = %d, seq_id = %d)\n", n_seq_max, seq_id);
                return 1;
            }
        }
        llama_memory_t mem = llama_get_memory(lctx);
        if (!llama_memory_seq_rm(mem, seq_id, 0, -1) || (mtl && !llama_memory_seq_rm(mem, seq_uncond, 0, -1))) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset chatterbox sequences\n");
            return 1;
        }

        // the encode gives the talker conditioning rows, set_ref keeps the flow decoder reference
        if (!inp->speaker_ref) {
            LOG_ERR("mtmd_helper_gen_audio: chatterbox requires a speaker reference or a voice\n");
            return 1;
        }
        if (!encode_speaker(inp->speaker_ref, prompt_embd_buf) ||
            mtmd_gen_state_set_ref(mctx, gen_state.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: speaker reference encoding failed\n");
            return 1;
        }

        const int n_e = n_embd;
        auto push_row = [&](const float * src) {
            prompt_embd_buf.insert(prompt_embd_buf.end(), src, src + n_e);
        };
        auto add_pos = [&](float * r, const std::vector<float> & tab, int idx) {
            const float * p = tab.data() + (size_t) idx * n_e;
            for (int j = 0; j < n_e; j++) {
                r[j] += p[j];
            }
        };

        // prompt rows: [conditioning, text, speech bos]
        // no text preprocessing here, the [SPACE] substitution and the case folding are baked in at conversion
        std::string txt(inp->prompt, inp->prompt_len);
        if (mtl && inp->lang && inp->lang[0]) {
            // the reference prefixes the language tag, e.g. "[fr]"
            std::string lang = inp->lang;
            std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            txt = "[" + lang + "]" + txt;
        }
        std::vector<llama_token> ids(txt.size() + 16);
        int n_ids = llama_tokenize(vocab, txt.c_str(), (int32_t) txt.size(), ids.data(), (int32_t) ids.size(),
                                   false, true);
        if (n_ids < 1) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);
        if (mtl) {
            // only the multilingual prompt wraps the text in start/stop tokens
            ids.insert(ids.begin(), text_start);
            ids.push_back(text_stop);
            if (ids.size() > text_pos.size() / (size_t) n_e) {
                LOG_ERR("mtmd_helper_gen_audio: prompt too long (%zu tokens, max %zu)\n", ids.size(), text_pos.size() / (size_t) n_e);
                return 1;
            }
        }
        const size_t text0 = prompt_embd_buf.size() / (size_t) n_e;
        for (size_t i = 0; i < ids.size(); i++) {
            push_row(tok_embd.data() + (size_t) ids[i] * n_e);
            if (mtl) {
                add_pos(prompt_embd_buf.data() + (text0 + i) * n_e, text_pos, (int) i);
            }
        }

        // speech bos opens the AR stream
        push_row(tok_embd.data() + (size_t) llama_vocab_bos(vocab) * n_e);
        if (mtl) {
            add_pos(prompt_embd_buf.data() + prompt_embd_buf.size() - n_e, speech_pos, 0);
        }
        ar_idx = 1;
        n_prompt = (int) (prompt_embd_buf.size() / (size_t) n_e);

        if (mtl) {
            // cfg uncond prompt: text rows are zeroed but keep their learned positions, as in the reference
            uncond_embd_buf = prompt_embd_buf;
            for (size_t i = 0; i < ids.size(); i++) {
                float * r = uncond_embd_buf.data() + (text0 + i) * n_e;
                std::fill(r, r + n_e, 0.0f);
                add_pos(r, text_pos, (int) i);
            }
        }

        prompt_pos = 0;
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        // the uncond prompt follows the cond one on mtl
        const int n_total = mtl ? 2 * n_prompt : n_prompt;
        if (prompt_pos >= n_total) {
            return 0;
        }
        const bool  uncond = prompt_pos >= n_prompt;
        const int   p0     = uncond ? prompt_pos - n_prompt : prompt_pos;
        const int32_t n_tokens_batch = std::min(n_batch, n_prompt - p0);
        const bool  last   = p0 + n_tokens_batch == n_prompt;

        decode_embd_batch batch((uncond ? uncond_embd_buf : prompt_embd_buf).data(), n_prompt, 1, n_embd);
        batch.set_position_normal(0, uncond ? seq_uncond : seq_id);
        if (last) {
            batch.logits[n_prompt - 1] = 1;
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, p0, n_tokens_batch)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        prompt_pos += n_tokens_batch;

        if (last && mtl) {
            if (!uncond) {
                cfg_read();
            } else {
                cfg_apply();
            }
        }

        if (prompt_pos >= n_total) {
            pos = n_prompt;
            prompt_embd_buf.clear();
            uncond_embd_buf.clear();
            return 0;
        }
        return n_total - prompt_pos;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        GGML_UNUSED(h_state_in);
        if (sampled < 0 || sampled >= llama_vocab_n_tokens(vocab)) {
            LOG_ERR("mtmd_helper_gen_audio: chatterbox requires a token sampled from the backbone\n");
            return 1;
        }

        // the speech stop token is the backbone eos, the multilingual speech positions end the stream too
        if (llama_vocab_is_eog(vocab, sampled) || (mtl && (size_t) ar_idx >= speech_pos.size() / (size_t) n_embd)) {
            *out_stop    = true;
            *h_state_out = nullptr;
            return 0;
        }

        // keep only the 6561 s3gen codes, dropping start/stop and oov ids
        if (sampled >= speech_base && sampled - speech_base < 6561) {
            codes_buf.push_back(sampled - speech_base);
        }

        std::vector<float> e(tok_embd.begin() + (size_t) sampled * n_embd,
                             tok_embd.begin() + (size_t) (sampled + 1) * n_embd);
        if (mtl) {
            // the multilingual backbone takes the token row plus the learned speech position
            const float * p = speech_pos.data() + (size_t) ar_idx * n_embd;
            for (int j = 0; j < n_embd; j++) {
                e[(size_t) j] += p[j];
            }
            ar_idx++;
        }

        if (!decode_one(e.data(), seq_id)) {
            return 1;
        }
        if (mtl) {
            // cfg: the sampled token feeds both branches
            cfg_read();
            if (!decode_one(e.data(), seq_uncond)) {
                return 1;
            }
            cfg_apply();
        }
        pos++;

        const float * h = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(h, h + n_embd);
        *h_state_out = h_state_buf.data();
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            if (!flush_gen_wav()) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: the flow decode is one-shot, so all the audio arrives after flush()
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        return flush_gen_wav() ? 0 : 1;
    }

private:
    bool decode_one(const float * e, llama_seq_id seq) {
        decode_embd_batch batch(e, 1, 1, n_embd);
        batch.set_position_normal(pos, seq);
        batch.logits[0] = 1;
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: step decode failed\n");
            return false;
        }
        return true;
    }

    // cfg of the reference: logits = cond + 0.5 * (cond - uncond)
    // the mix goes into the last output row (the uncond one), the caller samples from it
    void cfg_read() {
        const float * c = llama_get_logits_ith(lctx, -1);
        cond_logits.assign(c, c + llama_vocab_n_tokens(vocab));
    }
    void cfg_apply() {
        float * u = llama_get_logits_ith(lctx, -1);
        for (size_t i = 0; i < cond_logits.size(); i++) {
            u[i] = cond_logits[i] + 0.5f * (cond_logits[i] - u[i]);
        }
    }

    // one GEN_WAV call over the whole utterance, the flow decode is not streamed
    bool flush_gen_wav() {
        if (decoded || codes_buf.empty()) {
            return true;
        }
        if (!mtl) {
            // turbo appends a short silence tail before vocoding
            codes_buf.insert(codes_buf.end(), 3, 4299);
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes   = codes_buf.data();
        inp.n_codes = codes_buf.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_wav decode failed\n");
            return false;
        }
        audio_pcm.assign(out.audio, out.audio + out.n_samples);
        decoded = true;
        return true;
    }

    bool ensure_cache() {
        if (!tok_embd.empty()) {
            return true;
        }
        // fused vocab layout: [text 0..speech_base) then the speech tokens
        speech_base = find_special_token(vocab, "<|speech_0|>");
        if (speech_base == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: fused speech tokens not found in vocab\n");
            return false;
        }

        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd != (uint32_t) llama_vocab_n_tokens(vocab) * (uint32_t) n_embd) {
            LOG_ERR("mtmd_helper_gen_audio: unexpected token embedding size\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        llama_model_get_tok_embd(model, tok_embd.data());

        auto read_table = [&](const char * name, std::vector<float> & dst) {
            const size_t n = mtmd_gen_audio_read_tensor(mctx, name, nullptr, 0);
            dst.resize(n);
            if (n == 0 || mtmd_gen_audio_read_tensor(mctx, name, dst.data(), n) != n) {
                LOG_ERR("mtmd_helper_gen_audio: %s read failed\n", name);
                return false;
            }
            return true;
        };

        // multilingual: the learned position tables
        mtl = mtmd_gen_audio_read_tensor(mctx, "a.gen.t3.text_pos_emb", nullptr, 0) > 0;
        if (mtl && (!read_table("a.gen.t3.text_pos_emb", text_pos) || !read_table("a.gen.t3.speech_pos_emb", speech_pos))) {
            return false;
        }
        return true;
    }

    // vocab and mmproj data fixed across the whole session, read once
    std::vector<float> tok_embd;
    bool mtl = false;
    std::vector<float> text_pos;
    std::vector<float> speech_pos;
    llama_token speech_base = LLAMA_TOKEN_NULL;
    // start_text_token and stop_text_token of the reference, only the multilingual prompt uses them
    static constexpr llama_token text_start = 255;
    static constexpr llama_token text_stop  = 0;

    // per-generation state, cleared by reset()
    llama_seq_id seq_id     = 0;
    llama_seq_id seq_uncond = 0;
    llama_pos pos = 0;
    int ar_idx = 0;
    std::vector<float> prompt_embd_buf;
    std::vector<float> uncond_embd_buf;
    int n_prompt   = 0;
    int prompt_pos = 0;
    std::vector<float> cond_logits;
    std::vector<int32_t> codes_buf;
    bool decoded = false;
    std::vector<float> audio_pcm;
    std::vector<float> h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0; // samples already returned by get_output()
    bool   wav_header_sent = false;
};

// Drama 3: backbone samples the main token of each frame, codebook tower samples the other codes, Fish DAC makes the PCM
// the tower attends over the whole sequence, so it also runs on every prompt row
class drama3_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        seq_id = 0;
        pos = 0;
        prompt_tokens.clear();
        prompt_pos = 0;
        frame.clear();
        main_logits.clear();
        hist.clear();
        codes_buf.clear();
        n_decoded = 0;
        gen_state.reset();
        pcm_sent = 0;
        wav_header_sent = false;
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id   = inp->seq_id;
        out_type = inp->out_type;
        stream   = inp->stream;
        if (inp->speaker_ref) {
            LOG_WRN("mtmd_helper_gen_audio: voice cloning is not supported yet, the speaker file is ignored\n");
        }

        sem_begin = find_special_token(vocab, "<|semantic:0|>");
        eos       = find_special_token(vocab, "<|im_end|>");
        if (sem_begin == LLAMA_TOKEN_NULL || eos == LLAMA_TOKEN_NULL ||
                sem_begin + n_cb_vocab > llama_vocab_n_tokens(vocab)) {
            LOG_ERR("mtmd_helper_gen_audio: missing semantic tokens in vocab\n");
            return 1;
        }

        if (!tokenize_segments(prompt_segments(std::string(inp->prompt, inp->prompt_len)), prompt_tokens)) {
            return 1;
        }
        if (prompt_tokens.size() >= llama_n_ctx_seq(lctx) || prompt_tokens.size() > max_rows()) {
            LOG_ERR("mtmd_helper_gen_audio: prompt does not fit in the context\n");
            return 1;
        }
        if (!llama_memory_seq_rm(llama_get_memory(lctx), seq_id, 0, -1)) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset the sequence\n");
            return 1;
        }
        rng.seed(inp->seed == UINT32_MAX ? std::random_device{}() : inp->seed);
        return init_gen_state(inp->seed) ? 0 : 1;
    }

    // run the tower on every prompt row to fill its KV
    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        const int n_prompt = (int) prompt_tokens.size();
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        // small batches: every row also outputs logits
        const int n = std::min({ n_batch, n_prompt - prompt_pos, 64 });
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), prompt_tokens.data() + prompt_pos, n, pos, seq_id)) {
            return -1;
        }
        for (int i = 0; i < n; i++) {
            llama_batch_ext_set_output_logits(batch.get(), i, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        for (int i = 0; i < n; i++) {
            const float * h = llama_get_embeddings_ith(lctx, i);
            if (prompt_pos + i == n_prompt - 1) {
                h_state_buf.assign(h, h + n_embd); // first frame comes from the last prompt row
                save_main_logits(i);
                break;
            }
            if (!run_tower(-1, h)) {
                return -1;
            }
        }
        pos        += n;
        prompt_pos += n;
        return n_prompt - prompt_pos;
    }

    // sample the main token here, ignore the token of the caller
    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        GGML_UNUSED(sampled);
        GGML_UNUSED(h_state_in);
        *h_state_out = nullptr;
        *out_stop    = false;

        const llama_token main = sample_main(main_logits.data());
        const size_t n_rows = prompt_tokens.size() - 1 + codes_buf.size() / n_cb;
        if (main == eos || (uint32_t) pos >= llama_n_ctx_seq(lctx) || n_rows >= max_rows()) {
            *out_stop = true;
            return 0;
        }
        hist.push_back(main);
        if (hist.size() > ras_window) {
            hist.erase(hist.begin());
        }

        const float * next_embd = run_tower(main - sem_begin, h_state_buf.data());
        if (!next_embd) {
            return 1;
        }
        codes_buf.insert(codes_buf.end(), frame.begin(), frame.end());

        // text/speech stream: all mrope sections use the same position
        const bool mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ||
                           llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;
        decode_embd_batch batch(next_embd, 1, mrope ? 4 : 1, n_embd);
        if (mrope) {
            batch.set_position_mrope_1d(pos, seq_id);
        } else {
            batch.set_position_normal(pos, seq_id);
        }
        batch.logits[0] = 1;
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }
        pos++;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(h, h + n_embd);
        save_main_logits(-1);
        *h_state_out = h_state_buf.data();

        if (stream && codes_buf.size() / n_cb - n_decoded >= stream_frames) {
            return decode_wav() ? 0 : 1;
        }
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            if (flush() != 0) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: only the audio since the previous call
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        return decode_wav() ? 0 : 1;
    }

protected:
    // tokenize each segment alone as the reference does
    bool tokenize_segments(const std::vector<std::string> & segments, std::vector<llama_token> & out) const {
        for (const std::string & seg : segments) {
            std::vector<llama_token> ids(seg.size() + 16);
            const int32_t n = llama_tokenize(vocab, seg.c_str(), (int32_t) seg.size(), ids.data(), (int32_t) ids.size(), false, true);
            if (n < 0) {
                LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
                return false;
            }
            out.insert(out.end(), ids.begin(), ids.begin() + n);
        }
        return true;
    }

    // reference-bracket-v1 prompt without references
    virtual std::vector<std::string> prompt_segments(std::string text) const {
        if (text.rfind("<|speaker:", 0) != 0) {
            text = "<|speaker:0|> " + text;
        }
        return {
            "<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
            "<|im_start|>user\n", text, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
        };
    }

    // prompt rows but the last + generated frames
    virtual size_t max_rows() const { return n_tower_rows; }

    // copy right after our own decode, other sequences may decode on lctx before the next step_gen
    // layout: eos, then semantic codes 0..n_cb_vocab-1
    void save_main_logits(int32_t i) {
        const float * logits = llama_get_logits_ith(lctx, i);
        main_logits.resize(n_cb_vocab + 1);
        main_logits[0] = logits[eos];
        std::copy(logits + sem_begin, logits + sem_begin + n_cb_vocab, main_logits.begin() + 1);
    }

    // repetition-aware sampling: if a semantic draw repeats one of the last ras_window main tokens, draw again with fixed settings
    static constexpr size_t ras_window = 16;
    static constexpr size_t n_cb       = 10;
    // codebook tower KV rows (mtmd_graph_drama3_gen::n_pos_max): one per prompt row but the last, one per frame
    static constexpr size_t n_tower_rows = 4096;
    // frames per codec call when streaming (0.33 s)
    static constexpr size_t stream_frames = 7;

    // top-p keeps the token that crosses the threshold, as the reference does
    llama_token draw(const std::vector<float> & cand_logits, float temp, int32_t top_k, float top_p) {
        std::vector<std::pair<float, int>> cand(cand_logits.size());
        for (size_t i = 0; i < cand.size(); i++) {
            cand[i] = { cand_logits[i] / temp, (int) i };
        }
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        const float  lmax = cand[0].first;
        double       sum  = 0.0;
        std::vector<double> prob(cand.size());
        for (size_t i = 0; i < cand.size(); i++) {
            prob[i] = std::exp((double) (cand[i].first - lmax));
            sum += prob[i];
        }
        size_t n_keep = cand.size();
        if (top_k > 0) {
            n_keep = std::min(n_keep, (size_t) top_k);
        }
        double cum = 0.0;
        for (size_t i = 0; i < n_keep; i++) {
            if (cum > top_p * sum) {
                n_keep = i;
                break;
            }
            cum += prob[i];
        }
        double kept = 0.0;
        for (size_t i = 0; i < n_keep; i++) {
            kept += prob[i];
        }
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * kept;
        for (size_t i = 0; i < n_keep; i++) {
            r -= prob[i];
            if (r <= 0.0) {
                return cand[i].second;
            }
        }
        return cand[n_keep - 1].second;
    }

    // candidates: eos, then semantic codes 1.. (the reference never samples code 0)
    // logits as saved by save_main_logits()
    virtual llama_token sample_main(const float * logits) {
        std::vector<float> cand(n_cb_vocab);
        cand[0] = logits[0];
        for (int i = 1; i < n_cb_vocab; i++) {
            cand[i] = logits[1 + i];
        }
        auto to_token = [&](int i) { return i == 0 ? eos : sem_begin + i; };
        const llama_token first = to_token(draw(cand, 1.0f, 0, 0.9f));
        if (first != eos && std::find(hist.begin(), hist.end(), first) != hist.end()) {
            return to_token(draw(cand, 1.0f, 30, 0.95f));
        }
        return first;
    }

    // code0 < 0 for a prompt row, else the main code of the next frame
    // returns the next backbone input, the codes of the next frame go to frame
    const float * run_tower(int32_t code0, const float * h) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0   = code0;
        inp.embd    = const_cast<float *>(h);
        inp.codes   = frame.empty() ? nullptr : frame.data();
        inp.n_codes = frame.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0 || (code0 >= 0 && out.n_codes != n_cb)) {
            LOG_ERR("mtmd_helper_gen_audio: codebook tower failed\n");
            return nullptr;
        }
        if (code0 >= 0) {
            frame.assign(out.codes, out.codes + out.n_codes);
        }
        return out.embd;
    }

    // the codec is stateful, so frames can be decoded in pieces
    bool decode_wav() {
        const size_t n_frames = codes_buf.size() / n_cb;
        while (n_decoded < n_frames) {
            const size_t n = std::min(n_frames - n_decoded, (size_t) 64);
            mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
            inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            inp.codes   = codes_buf.data() + n_decoded * n_cb;
            inp.n_codes = n * n_cb;
            mtmd_gen_out out{};
            if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: codec decode failed\n");
                return false;
            }
            audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
            n_decoded += n;
        }
        return true;
    }

    static constexpr int n_cb_vocab = 4096; // <|semantic:0..4095|>

    llama_token sem_begin = LLAMA_TOKEN_NULL;
    llama_token eos       = LLAMA_TOKEN_NULL;

    llama_seq_id seq_id = 0;
    llama_pos    pos    = 0;
    std::vector<llama_token> prompt_tokens;
    int prompt_pos = 0;

    std::mt19937 rng;
    std::vector<float>       main_logits; // see save_main_logits()
    std::vector<int32_t>     frame;     // codes of the last frame, input of the next tower row
    std::vector<llama_token> hist;      // last main tokens, for repetition-aware sampling
    std::vector<int32_t>     codes_buf; // frame-major
    size_t n_decoded = 0;               // frames already through the codec

    std::vector<float>   audio_pcm;
    std::vector<float>   h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0; // samples already returned by get_output()
    bool   wav_header_sent = false;
};

// Fish Audio S2 Pro: same tokens, main-token loop and Fish DAC as drama3
// the fast AR only runs on generated frames, so the prompt is a plain decode
class fish_s2_gen_audio_pipeline : public drama3_gen_audio_pipeline {
public:
    using drama3_gen_audio_pipeline::drama3_gen_audio_pipeline;

    void reset() override {
        drama3_gen_audio_pipeline::reset();
        parts.clear();
        part_idx = 0;
        part_pos = 0;
    }

    // voice cloning (sglang-omni fishaudio_s2_pro): a system message with the transcript and the codec rows of the clip
    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        std::vector<float> ref_rows;
        if (inp->speaker_ref && !encode_speaker(inp->speaker_ref, ref_rows)) {
            return 1;
        }
        mtmd_helper_gen_audio_inp text_inp = *inp;
        text_inp.speaker_ref = nullptr;
        if (drama3_gen_audio_pipeline::set_input(&text_inp) != 0) {
            return 1;
        }

        size_t n_total = prompt_tokens.size();
        if (!ref_rows.empty()) {
            // optional, the reference leaves the text empty when it has none
            std::string ref_text = inp->speaker_ref_text ? inp->speaker_ref_text : "";
            if (!ref_text.empty() && ref_text.rfind("<|speaker:", 0) != 0) {
                ref_text = "<|speaker:0|>" + ref_text;
            }
            prompt_part head, rows, tail;
            if (!tokenize_segments({ "<|im_start|>system\n", "convert the provided text to speech reference to the following:\n\nText:\n",
                                     ref_text, "\n\nSpeech:\n" }, head.tokens) ||
                !tokenize_segments({ "<|im_end|>\n" }, tail.tokens)) {
                return 1;
            }
            rows.rows = std::move(ref_rows);
            for (auto * part : { &head, &rows, &tail }) {
                n_total += part->size(n_embd);
                parts.push_back(std::move(*part));
            }
        }
        if (n_total >= llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: prompt does not fit in the context\n");
            return 1;
        }
        prompt_part text;
        text.tokens = prompt_tokens;
        parts.push_back(std::move(text));
        return 0;
    }

    // the first frame comes from the last prompt row
    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        if (part_idx >= parts.size()) {
            return 0;
        }
        const prompt_part & part = parts[part_idx];
        const int  n_part = part.size(n_embd);
        const int  n      = std::min(n_batch, n_part - part_pos);
        const bool last   = part_idx + 1 == parts.size() && part_pos + n == n_part;

        int ret = 0;
        if (part.rows.empty()) {
            llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
            if (!batch_set_tokens(batch.get(), part.tokens.data() + part_pos, n, pos, seq_id)) {
                return -1;
            }
            if (last) {
                llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
            }
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        } else {
            decode_embd_batch batch(part.rows.data() + (size_t) part_pos * n_embd, n, 1, n_embd);
            batch.set_position_normal(pos, seq_id);
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, n));
        }
        if (ret != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        if (last) {
            const float * h = llama_get_embeddings_ith(lctx, -1);
            h_state_buf.assign(h, h + n_embd);
            save_main_logits(-1);
        }
        pos      += n;
        part_pos += n;
        if (part_pos == n_part) {
            part_idx++;
            part_pos = 0;
        }
        int32_t n_left = -part_pos;
        for (size_t i = part_idx; i < parts.size(); i++) {
            n_left += parts[i].size(n_embd);
        }
        return n_left;
    }

protected:
    // reference (sglang-omni fishaudio_s2_pro) has no system message
    std::vector<std::string> prompt_segments(std::string text) const override {
        if (text.rfind("<|speaker:", 0) != 0) {
            text = "<|speaker:0|>" + text;
        }
        return {
            "<|im_start|>user\n", text, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
        };
    }

    // no codebook tower, only the context limits the length
    size_t max_rows() const override { return SIZE_MAX; }

    // token ids, or backbone input rows of n_embd floats
    struct prompt_part {
        std::vector<llama_token> tokens;
        std::vector<float>       rows;
        int size(int n_embd) const { return rows.empty() ? (int) tokens.size() : (int) (rows.size() / n_embd); }
    };
    std::vector<prompt_part> parts;
    size_t part_idx = 0;
    int    part_pos = 0;

    // same as the reference: repetition penalty over the last ras_window main tokens, top-k, top-p on the
    // untempered logits (the crossing token is dropped), then temperature
    // a repeat within the last 4 main tokens switches to the RAS settings
    llama_token sample_main(const float * logits) override {
        static constexpr int32_t top_k       = 30;
        static constexpr float   rep_penalty = 1.1f;

        std::vector<std::pair<float, llama_token>> cand;
        cand.reserve(n_cb_vocab + 1);
        cand.push_back({ logits[0], eos });
        for (int i = 0; i < n_cb_vocab; i++) {
            cand.push_back({ logits[1 + i], sem_begin + i });
        }
        for (auto & c : cand) {
            if (std::find(hist.begin(), hist.end(), c.second) != hist.end()) {
                c.first = c.first < 0.0f ? c.first * rep_penalty : c.first / rep_penalty;
            }
        }

        bool ras = false;
        if (hist.size() >= 4) {
            std::vector<llama_token> last4(hist.end() - 4, hist.end());
            std::sort(last4.begin(), last4.end());
            ras = std::adjacent_find(last4.begin(), last4.end()) != last4.end();
        }
        const float temp  = ras ? 1.0f : 0.8f;
        const float top_p = ras ? 0.9f : 0.8f;

        const size_t k = std::min<size_t>(top_k, cand.size());
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

        std::vector<double> prob(k);
        double sum = 0.0;
        for (size_t i = 0; i < k; i++) {
            prob[i] = std::exp((double) (cand[i].first - cand[0].first));
            sum += prob[i];
        }
        size_t n_keep = k;
        double cum = 0.0;
        for (size_t i = 0; i < k; i++) {
            cum += prob[i];
            if (i > 0 && cum > top_p * sum) {
                n_keep = i;
                break;
            }
        }

        double kept = 0.0;
        for (size_t i = 0; i < n_keep; i++) {
            prob[i] = std::exp((double) (cand[i].first - cand[0].first) / temp);
            kept += prob[i];
        }
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * kept;
        for (size_t i = 0; i < n_keep; i++) {
            r -= prob[i];
            if (r <= 0.0) {
                return cand[i].second;
            }
        }
        return cand[n_keep - 1].second;
    }
};

// ZONOS3: one backbone row per frame of 10 codebooks, codebook j lags j frames
class zonos3_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    bool supports_delivery() const override { return true; }

    void reset() override {
        seq_id = 0;
        pos = 0;
        segments.clear();
        seg_idx = 0;
        seg_pos = 0;
        rows.clear();
        step = 0;
        eos_row = -1;
        countdown = 0;
        done = false;
        n_decoded = 0;
        n_codec_samples = 0;
        flushed = false;
        gen_state.reset();
        pcm_sent = 0;
        wav_header_sent = false;
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id   = inp->seq_id;
        out_type = inp->out_type;
        stream   = inp->stream;

        // settings of the reference, only the temperature can be changed
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        temp    = inp->temp >= 0.0f ? inp->temp : def.temp;
        top_k   = def.top_k;
        top_p   = def.top_p;
        rep_pen = def.penalty_repeat;
        rep_win = def.penalty_last_n;
        rng.seed(inp->seed == UINT32_MAX ? std::random_device{}() : inp->seed);

        if (!init_gen_state(inp->seed)) {
            return 1;
        }

        // prompt: [voice rows] [condition rows] BOS text EOS [silence rows]
        // the speaker CFG of the reference is omitted: it adds too much code and maintenance cost
        if (inp->speaker_ref) {
            segment seg;
            if (!encode_speaker(inp->speaker_ref, seg.embd)) {
                return 1;
            }
            segments.push_back(std::move(seg));
        }

        // condition rows: the bucket of each value over the checkpoint edges, or unknown
        auto bucket = [](const std::vector<float> & edges, float v) {
            return std::to_string(std::upper_bound(edges.begin(), edges.end(), v) - edges.begin());
        };
        const std::string loudness = inp->loudness != 0.0f ? bucket(loudness_edges, inp->loudness) : "unk";
        const std::string rate     = inp->speed != 1.0f ? bucket(rate_edges, inp->speed * typical_rate) : "unk";
        segment text;
        for (const std::string & name : { "<|loudness_" + loudness + "|>", std::string("<|fmax_unk|>"), "<|speaking_rate_" + rate + "|>" }) {
            const llama_token t = find_special_token(vocab, name);
            if (t == LLAMA_TOKEN_NULL) {
                LOG_ERR("mtmd_helper_gen_audio: missing ZONOS3 token %s in vocab\n", name.c_str());
                return 1;
            }
            text.tokens.push_back(t);
        }
        const std::string prompt(inp->prompt, inp->prompt_len);
        std::vector<llama_token> ids(prompt.size() + 16);
        const int32_t n_ids = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), ids.data(), (int32_t) ids.size(), true, false);
        if (n_ids <= 2) {
            LOG_ERR("mtmd_helper_gen_audio: empty prompt or tokenization failed\n");
            return 1;
        }
        text.tokens.insert(text.tokens.end(), ids.begin(), ids.begin() + n_ids);
        segments.push_back(std::move(text));

        // codec silence, delayed
        segment silence;
        for (int k = 0; k < n_prefill; k++) {
            std::vector<int32_t> row(n_cb, audio_pad);
            for (int cb = 0; cb <= k && cb < n_cb; cb++) {
                row[cb] = silence_frame[cb];
            }
            const float * embd = embed_row(row);
            if (!embd) {
                return 1;
            }
            silence.embd.insert(silence.embd.end(), embd, embd + n_embd);
            rows.insert(rows.end(), row.begin(), row.end());
        }
        segments.push_back(std::move(silence));

        size_t n_prompt = 0;
        for (const auto & seg : segments) {
            n_prompt += seg.n_rows(n_embd);
        }
        if (n_prompt >= llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: prompt does not fit in the context\n");
            return 1;
        }
        if (!llama_memory_seq_rm(llama_get_memory(lctx), seq_id, 0, -1)) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset the sequence\n");
            return 1;
        }
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        if (seg_idx >= segments.size()) {
            return 0;
        }
        const segment & seg = segments[seg_idx];
        const int n_rows = (int) seg.n_rows(n_embd);
        const int n      = std::min(n_batch, n_rows - seg_pos);
        const bool last  = seg_idx + 1 == segments.size() && seg_pos + n == n_rows;

        int ret;
        if (!seg.tokens.empty()) {
            llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
            if (!batch_set_tokens(batch.get(), seg.tokens.data() + seg_pos, n, pos, seq_id)) {
                return -1;
            }
            if (last) {
                llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
            }
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        } else {
            decode_embd_batch batch(seg.embd.data() + (size_t) seg_pos * n_embd, n, 1, n_embd);
            batch.set_position_normal(pos, seq_id);
            batch.logits[n - 1] = last;
            ret = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, n));
        }
        if (ret != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        pos     += n;
        seg_pos += n;
        if (seg_pos == n_rows) {
            seg_idx++;
            seg_pos = 0;
        }

        int32_t n_left = 0;
        for (size_t i = seg_idx; i < segments.size(); i++) {
            n_left += (int32_t) segments[i].n_rows(n_embd) - (i == seg_idx ? seg_pos : 0);
        }
        if (n_left == 0) {
            const float * h = llama_get_embeddings_ith(lctx, -1);
            h_state_buf.assign(h, h + n_embd);
        }
        return n_left;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        GGML_UNUSED(sampled);
        GGML_UNUSED(h_state_in);
        *h_state_out = nullptr;
        *out_stop    = false;
        if (done) {
            *out_stop = true;
            return 0;
        }

        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd = h_state_buf.data();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0 || out.n_feats != (size_t) n_cb * n_vocab) {
            LOG_ERR("mtmd_helper_gen_audio: codebook heads failed\n");
            return 1;
        }

        std::vector<int32_t> frame(n_cb);
        for (int cb = 0; cb < n_cb; cb++) {
            frame[cb] = sample(out.feats + (size_t) cb * n_vocab, cb);
        }
        // codebooks whose delayed stream has not started yet are pad
        const int r = n_prefill + step;
        for (int cb = r + 1; cb < n_cb; cb++) {
            frame[cb] = audio_pad;
        }
        rows.insert(rows.end(), frame.begin(), frame.end());
        step++;

        // after EOA, n_cb + 1 more rows flush the delay
        int max_eoa = -1;
        for (int cb = 0; cb < n_cb; cb++) {
            if (frame[cb] == audio_eoa) {
                max_eoa = cb;
            }
        }
        if (eos_row < 0 && max_eoa >= 0) {
            eos_row   = std::max(0, r - max_eoa);
            countdown = n_cb + 1;
        }
        if ((countdown > 0 && --countdown == 0) || step >= max_frames || (uint32_t) pos >= llama_n_ctx_seq(lctx)) {
            done = true;
            *out_stop = true;
            return 0;
        }

        const float * next = embed_row(frame);
        if (!next) {
            return 1;
        }
        decode_embd_batch batch(next, 1, 1, n_embd);
        batch.set_position_normal(pos, seq_id);
        batch.logits[0] = 1;
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }
        pos++;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(h, h + n_embd);
        *h_state_out = h_state_buf.data();

        if (stream) {
            return decode_stream() ? 0 : 1;
        }
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;

        if (!stream) {
            if (flush() != 0) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        // streaming: only the audio since the previous call
        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        return decode_wav(true) ? 0 : 1;
    }

private:
    static constexpr int n_cb      = 10;
    static constexpr int n_vocab   = 4098; // codes, EOA, pad
    static constexpr int audio_eoa = 4096;
    static constexpr int audio_pad = 4097;

    // condition_edges of params.json; speed 1 is this many UTF-8 bytes per second, the middle of the rate buckets
    const std::vector<float> loudness_edges = { -29.778f, -26.947f, -25.047f, -23.3f, -21.743f, -20.386f, -19.186f, -17.851f, -15.967f };
    const std::vector<float> rate_edges = {
        6.546f, 8.182f, 9.273f, 10.31f, 11.054f, 11.667f, 12.193f, 12.656f, 13.083f, 13.476f, 13.846f, 14.196f, 14.53f, 14.887f, 15.193f,
        15.517f, 15.865f, 16.208f, 16.579f, 16.972f, 17.411f, 17.898f, 18.434f, 19.069f, 19.818f, 20.816f, 22.286f, 24.722f, 29.25f,
    };
    static constexpr float typical_rate = 15.0f;

    // digital silence of the ZAC codec (registries/silence_frame_*.json)
    static constexpr int32_t silence_frame[n_cb] = { 1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337 };
    static constexpr int n_prefill = 3;

    // -3 dB of headroom
    static constexpr int   samples_per_frame = 2940;
    static constexpr float out_gain          = 0.70794578f;

    static constexpr int rep_codebooks = 6;
    static constexpr int max_frames    = 2000;
    static constexpr size_t stream_frames = 8;

    struct segment {
        std::vector<llama_token> tokens; // either tokens or embd rows
        std::vector<float>       embd;
        size_t n_rows(int n) const { return tokens.empty() ? embd.size() / (size_t) n : tokens.size(); }
    };

    const float * embed_row(const std::vector<int32_t> & codes) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.codes   = const_cast<int32_t *>(codes.data());
        inp.n_codes = codes.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: codebook embedding failed\n");
            return nullptr;
        }
        return out.embd;
    }

    // same as the reference sampler (servers/ggml/src/zonos3-sampler.cpp)
    int32_t sample(const float * in, int cb) {
        std::vector<float> l(in, in + n_vocab);
        l[audio_pad] = -INFINITY;

        if (rep_pen != 1.0f && cb < rep_codebooks) {
            const size_t n_rows = rows.size() / n_cb - n_prefill;
            const size_t n_win  = std::min(n_rows, (size_t) std::max(rep_win, 0));
            std::vector<bool> seen(audio_eoa, false);
            for (size_t f = n_prefill + n_rows - n_win; f < n_prefill + n_rows; f++) {
                const int32_t c = rows[f * n_cb + (size_t) cb];
                if (c >= 0 && c < audio_eoa && !seen[(size_t) c]) {
                    seen[(size_t) c] = true;
                    l[c] = l[c] > 0.0f ? l[c] / rep_pen : l[c] * rep_pen;
                }
            }
        }

        if (temp <= 0.0f) {
            return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
        }
        for (auto & v : l) {
            v /= temp;
        }

        float kth = -INFINITY;
        if (top_k > 0 && top_k < n_vocab) {
            std::vector<float> tmp(l);
            std::nth_element(tmp.begin(), tmp.begin() + (top_k - 1), tmp.end(), std::greater<float>());
            kth = tmp[(size_t) top_k - 1];
        }
        std::vector<int32_t> idx;
        for (int32_t i = 0; i < n_vocab; i++) {
            if (l[i] >= kth && l[i] > -INFINITY) {
                idx.push_back(i);
            }
        }
        float lmax = -INFINITY;
        for (int32_t i : idx) {
            lmax = std::max(lmax, l[i]);
        }
        std::vector<double> p(idx.size());
        double sum = 0.0;
        for (size_t j = 0; j < idx.size(); j++) {
            p[j] = std::exp((double) (l[idx[j]] - lmax));
            sum += p[j];
        }
        std::vector<size_t> ord(idx.size());
        for (size_t j = 0; j < ord.size(); j++) {
            ord[j] = j;
        }
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return p[a] > p[b]; });
        if (top_p > 0.0f && top_p < 1.0f) {
            double before = 0.0;
            for (size_t j : ord) {
                const double pj = p[j];
                if (before > top_p * sum) {
                    p[j] = 0.0;
                }
                before += pj;
            }
        }
        double kept = 0.0;
        for (double v : p) {
            kept += v;
        }
        double u = std::uniform_real_distribution<double>(0.0, kept)(rng);
        for (size_t j = 0; j < p.size(); j++) {
            u -= p[j];
            if (u <= 0.0 && p[j] > 0.0) {
                return idx[j];
            }
        }
        return idx[ord[0]];
    }

    // frames with all codebooks known
    size_t n_ready() const {
        const size_t n_rows = rows.size() / n_cb;
        size_t n = n_rows >= (size_t) n_cb ? n_rows - (n_cb - 1) : 0;
        if (eos_row >= 0) {
            n = std::min(n, (size_t) eos_row);
        }
        return n;
    }

    // streaming, as the reference does for the first audio: the silence frames are decoded with silence codes only,
    // then the first generated frame from its first code alone; later chunks of complete frames grow 2, 4, 8
    bool decode_stream() {
        if (n_decoded == 0) {
            const size_t n_rows = rows.size() / n_cb;
            if (n_rows <= (size_t) n_prefill || rows[(size_t) n_prefill * n_cb] >= audio_eoa) {
                return true;
            }
            std::vector<int32_t> codes;
            for (int k = 0; k < n_prefill; k++) {
                codes.insert(codes.end(), silence_frame, silence_frame + n_cb);
            }
            for (int cb = 0; cb < n_cb; cb++) {
                const size_t r = (size_t) n_prefill + cb;
                const int32_t c = r < n_rows ? rows[r * n_cb + cb] : -1;
                codes.push_back(c >= audio_eoa ? -1 : c);
            }
            if (!run_codec(codes.data(), codes.size())) {
                return false;
            }
            n_decoded = n_prefill + 1;
            return true;
        }
        const size_t n_chunk = std::min(stream_frames, n_decoded - n_prefill + 1);
        if (n_ready() >= n_decoded + n_chunk) {
            return decode_wav(false);
        }
        return true;
    }

    // the silence frames are decoded too, as left context
    bool decode_wav(bool final) {
        const size_t n_frames = n_ready();
        std::vector<int32_t> codes;
        while (n_decoded < n_frames) {
            const size_t n = std::min(n_frames - n_decoded, (size_t) 64);
            codes.resize(n * n_cb);
            for (size_t f = 0; f < n; f++) {
                for (int cb = 0; cb < n_cb; cb++) {
                    // EOA and pad have no codec row
                    const int32_t c = rows[(n_decoded + f + cb) * n_cb + cb];
                    codes[f * n_cb + cb] = std::clamp(c, 0, audio_eoa - 1);
                }
            }
            if (!run_codec(codes.data(), codes.size())) {
                return false;
            }
            n_decoded += n;
        }
        if (final && done && n_decoded > 0 && !flushed) {
            flushed = true;
            return run_codec(nullptr, 0);
        }
        return true;
    }

    bool run_codec(const int32_t * codes, size_t n_codes) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes   = const_cast<int32_t *>(codes);
        inp.n_codes = n_codes;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: codec decode failed\n");
            return false;
        }
        // drop the audio of the silence frames, then the output gain and soft limiter of the reference
        const float knee = std::pow(10.0f, -1.0f / 20.0f);
        for (size_t i = 0; i < out.n_samples; i++, n_codec_samples++) {
            if (n_codec_samples < (size_t) n_prefill * samples_per_frame) {
                continue;
            }
            const float x = out.audio[i] * out_gain;
            const float a = std::fabs(x);
            audio_pcm.push_back(a <= knee ? x : std::copysign(knee + (1.0f - knee) * std::tanh((a - knee) / (1.0f - knee)), x));
        }
        return true;
    }

    std::vector<segment> segments;
    size_t seg_idx = 0;
    int    seg_pos = 0;

    std::vector<int32_t> rows; // delayed rows: silence rows, then the generated ones
    int  step      = 0;
    int  eos_row   = -1;
    int  countdown = 0;
    bool done      = false;
    size_t n_decoded = 0;
    size_t n_codec_samples = 0;
    bool   flushed   = false;

    float   temp    = 0.75f;
    int32_t top_k   = 80;
    float   top_p   = 0.69f;
    float   rep_pen = 2.4f;
    int32_t rep_win = 96;
    std::mt19937 rng;

    llama_seq_id seq_id = 0;
    llama_pos    pos    = 0;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    bool stream = false;

    std::vector<float> h_state_buf;
    std::vector<float> audio_pcm;
    std::vector<char>  out_buf;
    size_t pcm_sent        = 0;
    bool   wav_header_sent = false;
};

// YuE2: the backbone writes an ABC score (text), then the semantic codes; both are sampled here like the reference (sampling.py)
// the NAR flow matching (in mtmd, on the backbone keys/values) and the VAE then make the stereo PCM
class yue2_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    bool supports_cfg() const override { return true; }
    bool supports_flow_steps() const override { return true; }

    // step_gen() stops at the score and code limits of the reference
    int32_t max_frames() const override { return 0; }

    // the limits and the NAR chunks of the reference need the training context (24576 positions)
    bool check_context() const override {
        if (llama_vocab_n_tokens(vocab) < tok_codec0 + n_codec) {
            LOG_ERR("mtmd_helper_gen_audio: the vocab of the backbone has no yue2 codec tokens\n");
            return false;
        }
        const uint32_t n_ctx_train = (uint32_t) llama_model_n_ctx_train(model);
        if (llama_n_ctx_seq(lctx) < n_ctx_train) {
            LOG_ERR("mtmd_helper_gen_audio: yue2 needs a context of %u positions per sequence, got %u (use the default context size)\n",
                    n_ctx_train, llama_n_ctx_seq(lctx));
            return false;
        }
        return true;
    }

    void reset() override {
        seq_id = 0;
        pos = 0;
        prompt.clear();
        prompt_pos = 0;
        in_codes = false;
        mode_off = false;
        cfg_scale = 1.0f;
        seq_uncond = 0;
        upos = 0;
        neg_prefix.clear();
        abc_tail.clear();
        prefix_tok.clear();
        history.clear();
        codes.clear();
        latents.clear();
        n_nar_done = 0;
        n_vae_done = 0;
        scratch.reset();
        scratch_n_ctx = 0;
        done = false;
        decoded = false;
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
        gen_state.reset();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id   = inp->seq_id;
        out_type = inp->out_type;
        stream   = inp->stream;
        code_cfg.temp = inp->temp >= 0.0f ? inp->temp : 1.0f;
        flow_steps    = inp->flow_steps;
        rng.seed(inp->seed == UINT32_MAX ? std::random_device{}() : inp->seed);
        if (!init_gen_state(inp->seed)) {
            return 1;
        }

        // the "tts" template renders the whole prompt, special tokens included
        const std::string text(inp->prompt, inp->prompt_len);
        prompt.resize(text.size() + 16);
        const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), prompt.data(), (int32_t) prompt.size(), false, true);
        if (n < 2 || prompt[0] != tok_eod) {
            LOG_ERR("mtmd_helper_gen_audio: yue2 expects the prompt of its tts template\n");
            return 1;
        }
        prompt.resize((size_t) n);
        // cover mode and cot=off give the score in the prompt, the codes come next
        auto abc = std::find(prompt.begin(), prompt.end(), tok_abc_start);
        if (prompt.back() == tok_music_start && abc != prompt.end()) {
            in_codes = true;
            mode_off = prompt.end() - abc == 3; // empty score
        } else if (prompt.back() != tok_abc_start) {
            LOG_ERR("mtmd_helper_gen_audio: yue2 prompt must end with <abc> or <|music_start|>\n");
            return 1;
        }
        abc_tail.assign(abc, prompt.end());
        prefix_tok = prompt;
        if (prompt.size() >= llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: prompt does not fit in the context\n");
            return 1;
        }
        llama_memory_t mem = llama_get_memory(lctx);
        if (!llama_memory_seq_rm(mem, seq_id, 0, -1)) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset the sequence\n");
            return 1;
        }

        // classifier-free guidance of the codes, defaults of the reference (protocol.py)
        cfg_scale = inp->cfg_scale > 0.0f ? inp->cfg_scale : mode_off ? 1.01f : 1.0f;
        if (cfg_scale != 1.0f) {
            // the uncond sequence lives half a seq space away from the cond one
            const int32_t n_seq_max = (int32_t) llama_n_seq_max(lctx);
            seq_uncond = (llama_seq_id) (n_seq_max / 2 + seq_id);
            if (!llama_model_classifier_free_guidance(model) || seq_id >= n_seq_max / 2) {
                if (inp->cfg_scale > 0.0f) {
                    LOG_ERR("mtmd_helper_gen_audio: cfg_scale needs a paired sequence (n_seq_max = %d, seq_id = %d), reconvert the model\n", n_seq_max, seq_id);
                    return 1;
                }
                LOG_WRN("mtmd_helper_gen_audio: no paired cfg sequence (n_seq_max = %d, seq_id = %d), cfg is disabled\n", n_seq_max, seq_id);
                cfg_scale = 1.0f;
            } else if (!llama_memory_seq_rm(mem, seq_uncond, 0, -1)) {
                LOG_ERR("mtmd_helper_gen_audio: cannot reset the cfg sequence\n");
                return 1;
            }
        }
        if (cfg_scale != 1.0f) {
            // the negative prompt is the bare instruction, tokenized alone like the reference
            const size_t i0 = text.find("<|endoftext|>");
            const size_t i1 = text.find('\n');
            if (i0 != 0 || i1 == std::string::npos) {
                LOG_ERR("mtmd_helper_gen_audio: yue2 expects the prompt of its tts template\n");
                return 1;
            }
            const std::string instr = text.substr(13, i1 - 13);
            neg_prefix.resize(instr.size() + 16);
            const int32_t n_instr = llama_tokenize(vocab, instr.c_str(), (int32_t) instr.size(), neg_prefix.data() + 1, (int32_t) neg_prefix.size() - 1, false, false);
            if (n_instr < 1) {
                LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
                return 1;
            }
            neg_prefix[0] = tok_eod;
            neg_prefix.resize((size_t) n_instr + 1);
        }
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        const int32_t n_left = (int32_t) prompt.size() - prompt_pos;
        if (n_left <= 0) {
            return 0;
        }
        const int32_t n = std::min(n_batch, n_left);
        if (!decode(prompt.data() + prompt_pos, n)) {
            return -1;
        }
        prompt_pos += n;
        if (n == n_left) {
            if (in_codes) {
                if (!start_uncond()) {
                    return -1;
                }
            } else {
                save_logits(llama_get_logits_ith(lctx, -1));
            }
        }
        return n_left - n;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        GGML_UNUSED(sampled);
        GGML_UNUSED(h_state_in);
        *h_state_out = nullptr;
        *out_stop    = false;
        if (done) {
            *out_stop = true;
            return 0;
        }

        const bool use_cfg = in_codes && cfg_scale != 1.0f;
        const llama_token tok = sample(logits.data());
        int32_t i_out = -1; // output row of the cond sequence
        if (!in_codes) {
            // the score ends with </abc>, or is cut at max_tokens like the reference
            if (tok == tok_abc_end || (int32_t) history.size() + 1 >= abc_cfg.max_tokens) {
                const llama_token next[2] = { tok_abc_end, tok_music_start };
                if (!decode(next, 2)) {
                    return 1;
                }
                abc_tail.insert(abc_tail.end(), next, next + 2);
                prefix_tok.insert(prefix_tok.end(), next, next + 2);
                in_codes = true;
                history.clear();
                if (!start_uncond()) {
                    return 1;
                }
            } else {
                history.push_back(tok);
                abc_tail.push_back(tok);
                prefix_tok.push_back(tok);
                if (!decode(&tok, 1)) {
                    return 1;
                }
                save_logits(llama_get_logits_ith(lctx, -1));
            }
        } else {
            // the NAR pass also attends to <|music_end|>, the uncond sequence shares the unified KV cache
            const bool full = (int32_t) codes.size() + 1 >= code_cfg.max_tokens || pos + upos + 2 >= (llama_pos) llama_n_ctx_seq(lctx);
            if (tok == tok_music_end || full) {
                if (!decode(&tok_music_end, 1)) {
                    return 1;
                }
                done = true;
                *out_stop = true;
                return 0;
            }
            history.push_back(tok);
            codes.push_back(tok - tok_codec0);
            if (use_cfg) {
                // the sampled code feeds both branches, in one decode
                if (!decode_pair(tok)) {
                    return 1;
                }
                i_out = 0;
            } else {
                if (!decode(&tok, 1)) {
                    return 1;
                }
                save_logits(llama_get_logits_ith(lctx, -1));
            }
        }
        const float * h = llama_get_embeddings_ith(lctx, i_out);
        h_state_buf.assign(h, h + n_embd);
        *h_state_out = h_state_buf.data();

        // streaming: each full chunk of codes goes through the NAR pass and the VAE right away
        if (stream && in_codes && codes.size() - n_nar_done >= chunk_size()) {
            if (!nar_chunk(n_nar_done, n_nar_done + chunk_size(), false) || !vae_emit(false)) {
                return 1;
            }
        }
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;
        const uint16_t n_ch = (uint16_t) info.n_channels;

        if (!stream) {
            if (flush() != 0) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) (audio_pcm.size() / n_ch);
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate, n_ch)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) (n_new / n_ch);
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate, n_ch);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        if (decoded) {
            return 0;
        }
        if (codes.empty()) {
            LOG_ERR("mtmd_helper_gen_audio: yue2 stopped before the first code\n");
            return 1;
        }
        if (!done) {
            // stopped by the caller: the NAR pass still needs <|music_end|>
            if (!decode(&tok_music_end, 1)) {
                return 1;
            }
            done = true;
        }

        // a song that fits in one chunk uses the backbone sequence itself, as it holds <|music_end|> now
        const size_t n_codes = codes.size();
        while (n_nar_done < n_codes) {
            const size_t a = n_nar_done;
            const size_t b = std::min(a + chunk_size(), n_codes);
            if (!nar_chunk(a, b, a == 0 && b == n_codes)) {
                return 1;
            }
        }
        if (!vae_emit(true)) {
            return 1;
        }
        scratch.reset();
        decoded = true;
        return 0;
    }

private:
    // token ids of the yue2_infer protocol (protocol.py)
    static constexpr llama_token tok_eod         = 151643;
    static constexpr llama_token tok_abc_start   = 151847;
    static constexpr llama_token tok_abc_end     = 151848;
    static constexpr llama_token tok_music_start = 151851;
    static constexpr llama_token tok_music_end   = 151852;
    static constexpr llama_token tok_codec0      = 151853;
    static constexpr int32_t     n_codec         = 32768;

    // NAR chunks and VAE tiles, see synthesize() and decode_tiled() of the reference
    static constexpr int64_t n_ctx_max    = 24576;
    static constexpr size_t  stream_chunk = 750; // 30 s
    static constexpr size_t  n_latent     = 64;
    static constexpr size_t  vae_core     = 128; // frames kept per tile
    static constexpr size_t  vae_halo     = 16;  // frames of context on each side
    static constexpr size_t  vae_ratio    = 1920;
    static constexpr size_t  vae_trim     = 64;

    struct sampling {
        float   temp;
        float   top_p;
        int32_t top_k;
        float   penalty;
        int32_t penalty_window;
        int32_t min_tokens;
        int32_t max_tokens;
    };
    sampling abc_cfg  = { 0.7f, 0.9f,  30,  1.005f, 100, 32,  4096 };
    sampling code_cfg = { 1.0f, 0.95f, 100, 1.2f,   50,  200, 9000 };

    bool decode_ctx(llama_context * ctx, const llama_token * toks, int32_t n, llama_seq_id seq, llama_pos & p) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
        if (!batch_set_tokens(batch.get(), toks, n, p, seq)) {
            return false;
        }
        llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        p += n;
        return true;
    }

    bool decode(const llama_token * toks, int32_t n) {
        return decode_ctx(lctx, toks, n, seq_id, pos);
    }

    // keeps the logits of the next token, as the context logits belong to the last decode of any sequence (server slots)
    // only the window that sample() reads: the text and </abc> for the score, <|music_end|> and the codes after it
    // with guidance: uncond + scale * (cond - uncond)
    void save_logits(const float * cond, const float * uncond = nullptr) {
        const llama_token lo = in_codes ? tok_music_end : 0;
        const llama_token hi = in_codes ? tok_codec0 + n_codec : tok_abc_end + 1;
        logits.resize((size_t) llama_vocab_n_tokens(vocab));
        for (llama_token t = lo; t < hi; t++) {
            logits[t] = uncond ? uncond[t] + cfg_scale * (cond[t] - uncond[t]) : cond[t];
        }
    }

    // the negative prompt of the reference (negative_prefix): the instruction, then the score of the cond prompt
    // the logits of the cond prompt must be the last ones when this is called
    bool start_uncond() {
        if (cfg_scale == 1.0f) {
            save_logits(llama_get_logits_ith(lctx, -1));
            return true;
        }
        const float * c = llama_get_logits_ith(lctx, -1);
        std::vector<float> cond(c, c + llama_vocab_n_tokens(vocab));

        std::vector<llama_token> neg = neg_prefix;
        if (mode_off) {
            neg.push_back(tok_music_start);
        } else {
            neg.insert(neg.end(), abc_tail.begin(), abc_tail.end());
        }
        if (pos + (llama_pos) neg.size() + 2 >= (llama_pos) llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: the cfg prompt does not fit in the context\n");
            return false;
        }
        const int32_t n_batch = (int32_t) llama_n_batch(lctx);
        for (size_t i = 0; i < neg.size(); i += (size_t) n_batch) {
            const int32_t n = std::min(n_batch, (int32_t) (neg.size() - i));
            if (!decode_ctx(lctx, neg.data() + i, n, seq_uncond, upos)) {
                return false;
            }
        }
        save_logits(cond.data(), llama_get_logits_ith(lctx, -1));
        return true;
    }

    bool decode_pair(llama_token tok) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        const llama_seq_id seqs[2] = { seq_id, seq_uncond };
        const llama_pos    p[2]    = { pos, upos };
        for (int i = 0; i < 2; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch.get(), seqs[i], tok);
            if (idx < 0) {
                LOG_ERR("mtmd_helper_gen_audio: failed to add token %d to batch (error %d)\n", tok, idx);
                return false;
            }
            llama_batch_ext_set_pos(batch.get(), idx, &p[i]);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        pos++;
        upos++;
        save_logits(llama_get_logits_ith(lctx, 0), llama_get_logits_ith(lctx, 1));
        return true;
    }

    // song_chunks() of the reference: each NAR chunk sees the prefix and its own codes
    // the reference uses the largest chunk that fits in n_ctx_max, streaming uses smaller ones
    size_t chunk_size() const {
        const int64_t ref = std::max<int64_t>(1, (n_ctx_max - (int64_t) prefix_tok.size() - 3) / 2);
        return (size_t) (stream ? std::min<int64_t>(ref, stream_chunk) : ref);
    }

    // backbone keys/values of [prefix, codes[a, b), <|music_end|>] -> latents of the chunk
    bool nar_chunk(size_t a, size_t b, bool use_main) {
        llama_context * ctx = lctx;
        llama_seq_id    seq = seq_id;
        llama_pos       n_past = pos;
        if (!use_main) {
            // a scratch context on the backbone model, the prefix is decoded once
            const llama_pos n_prefix = (llama_pos) prefix_tok.size();
            const uint32_t  n_need   = (uint32_t) (n_prefix + (b - a) + 1);
            if (!scratch || scratch_n_ctx < n_need) {
                llama_context_params cparams = llama_context_default_params();
                cparams.n_ctx           = n_need;
                cparams.n_seq_max       = 1;
                cparams.n_threads       = llama_n_threads(lctx);
                cparams.n_threads_batch = llama_n_threads_batch(lctx);
                cparams.no_perf         = true;
                scratch.reset(llama_init_from_model(const_cast<llama_model *>(model), cparams));
                if (!scratch) {
                    LOG_ERR("mtmd_helper_gen_audio: cannot create the yue2 chunk context\n");
                    return false;
                }
                scratch_n_ctx = n_need;
                llama_pos p = 0;
                const int32_t n_batch = (int32_t) llama_n_batch(scratch.get());
                for (size_t i = 0; i < prefix_tok.size(); i += (size_t) n_batch) {
                    const int32_t n = std::min(n_batch, (int32_t) (prefix_tok.size() - i));
                    if (!decode_ctx(scratch.get(), prefix_tok.data() + i, n, 0, p)) {
                        return false;
                    }
                }
            }
            llama_memory_seq_rm(llama_get_memory(scratch.get()), 0, n_prefix, -1);
            std::vector<llama_token> toks;
            for (size_t i = a; i < b; i++) {
                toks.push_back(codes[i] + tok_codec0);
            }
            toks.push_back(tok_music_end);
            llama_pos p = n_prefix;
            const int32_t n_batch = (int32_t) llama_n_batch(scratch.get());
            for (size_t i = 0; i < toks.size(); i += (size_t) n_batch) {
                const int32_t n = std::min(n_batch, (int32_t) (toks.size() - i));
                if (!decode_ctx(scratch.get(), toks.data() + i, n, 0, p)) {
                    return false;
                }
            }
            ctx    = scratch.get();
            seq    = 0;
            n_past = p;
        }

        std::vector<uint8_t> state(llama_state_seq_get_size(ctx, seq));
        if (state.empty() || llama_state_seq_get_data(ctx, state.data(), state.size(), seq) != state.size()) {
            LOG_ERR("mtmd_helper_gen_audio: cannot read the backbone state\n");
            return false;
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type                = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.n_codes             = b - a;
        inp.backbone_state      = state.data();
        inp.backbone_state_size = state.size();
        inp.backbone_n_past     = n_past;
        inp.n_steps             = flow_steps;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: flow matching failed\n");
            return false;
        }
        latents.insert(latents.end(), out.feats, out.feats + out.n_feats);
        n_nar_done = b;
        return true;
    }

    // decodes the VAE tiles whose context on both sides is known, same cropping as decode_tiled() of the reference
    bool vae_emit(bool final) {
        const size_t n_lat = latents.size() / n_latent;
        while (n_vae_done < n_lat) {
            const size_t start = n_vae_done;
            const size_t end   = std::min(start + vae_core, n_lat);
            if (!final && end + vae_halo > n_lat) {
                break;
            }
            const size_t left  = start >= vae_halo ? start - vae_halo : 0;
            const size_t right = std::min(end + vae_halo, n_lat);
            mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
            inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            inp.feats   = latents.data() + left * n_latent;
            inp.n_feats = (right - left) * n_latent;
            mtmd_gen_out out{};
            if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: VAE decode failed\n");
                return false;
            }
            // the whole song has vae_ratio * n - vae_trim samples
            const size_t n_ch   = (size_t) info.n_channels;
            const size_t crop0  = (start - left) * vae_ratio;
            const size_t n_keep = (end - start) * vae_ratio - (final && end == n_lat ? vae_trim : 0);
            if (out.n_samples < (crop0 + n_keep) * n_ch) {
                LOG_ERR("mtmd_helper_gen_audio: VAE tile is too short\n");
                return false;
            }
            audio_pcm.insert(audio_pcm.end(), out.audio + crop0 * n_ch, out.audio + (crop0 + n_keep) * n_ch);
            n_vae_done = end;
        }
        return true;
    }

    // distribution() of the reference: allowed range, no end token before min_tokens, windowed penalty, top-k, top-p
    llama_token sample(const float * logits) {
        const sampling & c = in_codes ? code_cfg : abc_cfg;
        const llama_token end = in_codes ? tok_music_end : tok_abc_end;
        const llama_token lo  = in_codes ? tok_codec0 : 0;
        const llama_token hi  = in_codes ? tok_codec0 + n_codec : tok_eod;

        std::unordered_map<llama_token, int> freq;
        const size_t n_win = std::min(history.size(), (size_t) c.penalty_window);
        for (size_t i = history.size() - n_win; i < history.size(); i++) {
            freq[history[i]]++;
        }

        std::vector<std::pair<float, llama_token>> cand;
        cand.reserve((size_t) (hi - lo) + 1);
        auto push = [&](llama_token t) {
            float l = logits[t];
            auto it = freq.find(t);
            if (it != freq.end()) {
                const float alpha = std::pow(c.penalty, (float) it->second);
                l = l < 0.0f ? l * alpha : l / alpha;
            }
            cand.emplace_back(l, t);
        };
        for (llama_token t = lo; t < hi; t++) {
            push(t);
        }
        if ((int32_t) history.size() >= c.min_tokens) {
            push(end);
        }

        if (c.temp <= 0.0f) {
            return std::max_element(cand.begin(), cand.end())->second;
        }
        const size_t k = std::min(cand.size(), (size_t) std::max(c.top_k, 1));
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), std::greater<std::pair<float, llama_token>>());
        cand.resize(k);

        std::vector<double> p(k);
        double sum = 0.0;
        for (size_t i = 0; i < k; i++) {
            p[i] = std::exp((double) ((cand[i].first - cand[0].first) / c.temp));
            sum += p[i];
        }
        // keep the smallest prefix whose mass reaches top_p, the first token always stays (3 for the codes of cot=off)
        const size_t n_keep = in_codes && mode_off ? 3 : 1;
        double before = 0.0;
        for (size_t i = 0; i < k; i++) {
            const double pi = p[i];
            if (i >= n_keep && before > c.top_p * sum) {
                p[i] = 0.0;
            }
            before += pi;
        }
        double kept = 0.0;
        for (double v : p) {
            kept += v;
        }
        double u = std::uniform_real_distribution<double>(0.0, kept)(rng);
        for (size_t i = 0; i < k; i++) {
            u -= p[i];
            if (u <= 0.0 && p[i] > 0.0) {
                return cand[i].second;
            }
        }
        return cand[0].second;
    }

    llama_seq_id seq_id = 0;
    llama_pos    pos    = 0;
    std::vector<llama_token> prompt;
    int32_t prompt_pos = 0;
    bool in_codes = false;
    bool mode_off = false; // cot=off: no score
    float        cfg_scale  = 1.0f;
    int32_t      flow_steps = 0; // 0 = the model default
    llama_seq_id seq_uncond = 0;
    llama_pos    upos       = 0;
    std::vector<llama_token> neg_prefix; // EOD + instruction
    std::vector<llama_token> abc_tail;   // the cond tokens from <abc> on
    std::vector<float>       logits;     // see save_logits()
    std::vector<llama_token> prefix_tok; // the cond tokens before the first code
    std::vector<float>       latents;    // of the chunks done so far
    size_t n_nar_done = 0; // codes
    size_t n_vae_done = 0; // latent frames
    llama_context_ptr scratch; // backbone keys/values of a chunk other than the first
    uint32_t scratch_n_ctx = 0;
    std::vector<llama_token> history; // tokens sampled in the current phase
    std::vector<int32_t> codes;
    bool done    = false;
    bool decoded = false;
    std::mt19937 rng;

    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    bool stream = false;
    std::vector<float> h_state_buf;
    std::vector<float> audio_pcm; // interleaved
    std::vector<char>  out_buf;
    size_t pcm_sent        = 0;
    bool   wav_header_sent = false;
};

// MiniMax Music 3: the backbone samples the semantic code of each frame with guidance, the depth decoder (mtmd) the other codes
// the condition rows of the frames go to the flow matching and the vocoder (mtmd) in windows, like the reference (diffusers pipeline)
class music3_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    bool supports_cfg() const override { return true; }
    bool supports_flow_steps() const override { return true; }

    // step_gen() stops at the frame limit of the reference
    int32_t max_frames() const override { return 0; }

    bool check_context() const override {
        if (llama_vocab_n_tokens(vocab) < tok_code0 + n_semantic) {
            LOG_ERR("mtmd_helper_gen_audio: the vocab of the backbone has no minimax_music3 audio codes\n");
            return false;
        }
        return true;
    }

    void reset() override {
        seq_id = 0;
        seq_uncond = 0;
        use_cfg = false;
        cfg_scale = 1.0f;
        temp = 1.0f;
        flow_steps = 0;
        pos = 0;
        prompt.clear();
        prompt_uncond.clear();
        prompt_pos = 0;
        uprompt_pos = 0;
        logits_cond.clear();
        logits_uncond.clear();
        h_rows.clear();
        n_gen = 0;
        cond_rows.clear();
        n_rows_dropped = 0;
        n_win_done = 0;
        done = false;
        decoded = false;
        audio_pcm.clear();
        out_buf.clear();
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
        gen_state.reset();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();
        seq_id     = inp->seq_id;
        out_type   = inp->out_type;
        stream     = inp->stream;
        temp       = inp->temp >= 0.0f ? inp->temp : 1.0f;
        flow_steps = inp->flow_steps;
        cfg_scale  = inp->cfg_scale > 0.0f ? inp->cfg_scale : 1.5f;
        rng.seed(inp->seed == UINT32_MAX ? std::random_device{}() : inp->seed);
        if (!init_gen_state(inp->seed)) {
            return 1;
        }

        // the "tts" template renders the whole prompt, special tokens included
        const std::string text(inp->prompt, inp->prompt_len);
        prompt.resize(text.size() + 16);
        const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), prompt.data(), (int32_t) prompt.size(), false, true);
        if (n < 3 || prompt[0] != tok_im_start || prompt[n - 1] != tok_audio_start) {
            LOG_ERR("mtmd_helper_gen_audio: minimax_music3 expects the prompt of its tts template\n");
            return 1;
        }
        prompt.resize((size_t) n);
        if (n > max_prompt) {
            LOG_ERR("mtmd_helper_gen_audio: the prompt has %d tokens, the maximum is %d\n", n, max_prompt);
            return 1;
        }

        llama_memory_t mem = llama_get_memory(lctx);
        if (!llama_memory_seq_rm(mem, seq_id, 0, -1)) {
            LOG_ERR("mtmd_helper_gen_audio: cannot reset the sequence\n");
            return 1;
        }
        use_cfg = cfg_scale != 1.0f;
        if (use_cfg) {
            // the uncond sequence lives half a seq space away from the cond one
            const int32_t n_seq_max = (int32_t) llama_n_seq_max(lctx);
            seq_uncond = (llama_seq_id) (n_seq_max / 2 + seq_id);
            if (!llama_model_classifier_free_guidance(model) || seq_id >= n_seq_max / 2) {
                LOG_ERR("mtmd_helper_gen_audio: cfg needs a paired sequence (n_seq_max = %d, seq_id = %d), reconvert the model or use cfg_scale = 1\n", n_seq_max, seq_id);
                return 1;
            }
            if (!llama_memory_seq_rm(mem, seq_uncond, 0, -1)) {
                LOG_ERR("mtmd_helper_gen_audio: cannot reset the cfg sequence\n");
                return 1;
            }
            // the uncond prompt keeps <|im_start|> and the last 2 tokens, the rest is <|audio_cfg|>
            prompt_uncond = prompt;
            std::fill(prompt_uncond.begin() + 1, prompt_uncond.end() - 2, tok_audio_cfg);
        }
        if (n_pos_needed((llama_pos) n + 1) > (int64_t) llama_n_ctx_seq(lctx)) {
            LOG_ERR("mtmd_helper_gen_audio: the prompt does not fit in the context\n");
            return 1;
        }
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        const int32_t n_prompt = (int32_t) prompt.size();
        if (prompt_pos < n_prompt) {
            const int32_t n = std::min(n_batch, n_prompt - prompt_pos);
            llama_pos p = prompt_pos;
            if (!decode_tokens(prompt.data() + prompt_pos, n, seq_id, p)) {
                return -1;
            }
            prompt_pos += n;
            if (prompt_pos == n_prompt) {
                save_outputs(0, -1);
            }
        } else if (use_cfg && uprompt_pos < n_prompt) {
            const int32_t n = std::min(n_batch, n_prompt - uprompt_pos);
            llama_pos p = uprompt_pos;
            if (!decode_tokens(prompt_uncond.data() + uprompt_pos, n, seq_uncond, p)) {
                return -1;
            }
            uprompt_pos += n;
            if (uprompt_pos == n_prompt) {
                save_outputs(1, -1);
            }
        }
        pos = n_prompt;
        const int32_t n_left = (n_prompt - prompt_pos) + (use_cfg ? n_prompt - uprompt_pos : 0);
        return n_left;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) override {
        GGML_UNUSED(sampled);
        GGML_UNUSED(h_state_in);
        *h_state_out = nullptr;
        *out_stop    = false;

        // the first frame only moves the backbone past <|audio_start|>, its condition row is dropped like the reference
        const size_t n_rows = n_frames();
        while (!done && n_frames() == n_rows) {
            if (!gen_frame()) {
                return 1;
            }
        }
        if (n_frames() == n_rows) {
            *out_stop = true;
            return 0;
        }
        *h_state_out = h_rows.data();
        *out_stop    = done;

        // a window is final once a frame past it exists, see MiniMaxMusic3PrepareChunksStep of the reference
        while (n_frames() > window_start(n_win_done) + win_frames) {
            if (!run_window(false)) {
                return 1;
            }
        }
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        *out_sample_rate = info.sample_rate;
        const uint16_t n_ch = (uint16_t) info.n_channels;

        if (!stream) {
            if (flush() != 0) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) (audio_pcm.size() / n_ch);
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = (const char *) audio_pcm.data();
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate, n_ch)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) (n_new / n_ch);
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) (audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate, n_ch);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() override {
        if (decoded) {
            return 0;
        }
        if (n_frames() == 0) {
            LOG_ERR("mtmd_helper_gen_audio: minimax_music3 stopped before the first frame\n");
            return 1;
        }
        done = true;
        // the windows start every win_hop frames before n_frames - win_hop, the last one ends at n_frames
        while (n_win_done == 0 || window_start(n_win_done) + win_hop < n_frames()) {
            const bool last = window_start(n_win_done + 1) + win_hop >= n_frames();
            if (!run_window(last)) {
                return 1;
            }
            if (last) {
                break;
            }
        }
        decoded = true;
        return 0;
    }

private:
    // token ids of the checkpoint, see encoders.py of the reference
    static constexpr llama_token tok_im_start    = 151644;
    static constexpr llama_token tok_audio_cfg   = 151654;
    static constexpr llama_token tok_audio_start = 151669;
    static constexpr llama_token tok_audio_end   = 151670;
    static constexpr llama_token tok_code0       = 151675;
    static constexpr int32_t     n_semantic      = 16384;
    static constexpr int32_t     max_prompt      = 5000;
    static constexpr size_t      max_audio_frames = 9000;
    static constexpr int32_t     top_k           = 50;

    // windows of the flow matching and their crop, see before_denoise.py and decoders.py of the reference
    static constexpr size_t win_frames = 200;
    static constexpr size_t win_hop    = 100;
    static constexpr size_t crop_left  = 86;       // latent frames
    static constexpr size_t crop_right = 344 - 86; // latent frames
    static constexpr size_t hop_samples = 512;     // samples per latent frame

    // with guidance, both sequences share the unified KV cache
    int64_t n_pos_needed(llama_pos n) const {
        return use_cfg ? 2 * (int64_t) n : (int64_t) n;
    }

    size_t n_frames() const {
        return n_rows_dropped + cond_rows.size() / (size_t) n_embd;
    }

    static size_t window_start(size_t k) {
        return k * win_hop;
    }

    bool decode_tokens(const llama_token * toks, int32_t n, llama_seq_id seq, llama_pos & p) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        if (!batch_set_tokens(batch.get(), toks, n, p, seq)) {
            return false;
        }
        llama_batch_ext_set_output_logits(batch.get(), n - 1, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        p += n;
        return true;
    }

    // the context outputs belong to the last decode of any sequence (server slots), keep ours right away
    // only the window that sample() reads: <|audio_end|> and the semantic codes
    void save_outputs(int row, int32_t idx) {
        auto & lg = row == 0 ? logits_cond : logits_uncond;
        const float * l = llama_get_logits_ith(lctx, idx);
        lg.assign(l + tok_audio_end, l + tok_code0 + n_semantic);
        const float * h = llama_get_embeddings_ith(lctx, idx);
        h_rows.resize((size_t) (use_cfg ? 2 : 1) * n_embd);
        std::copy(h, h + n_embd, h_rows.begin() + (size_t) row * n_embd);
    }

    // feeds the next backbone input row to both sequences
    bool decode_embd(const float * embd) {
        llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));
        const int n_seq = use_cfg ? 2 : 1;
        const llama_seq_id seqs[2] = { seq_id, seq_uncond };
        for (int i = 0; i < n_seq; i++) {
            const llama_embd row = { embd, 1, (size_t) n_embd };
            const int32_t idx = llama_batch_ext_add_embd(batch.get(), seqs[i], row);
            if (idx < 0) {
                LOG_ERR("mtmd_helper_gen_audio: failed to add a row to the batch (error %d)\n", idx);
                return false;
            }
            llama_batch_ext_set_pos(batch.get(), idx, &pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        pos++;
        for (int i = 0; i < n_seq; i++) {
            save_outputs(i, i);
        }
        return true;
    }

    // semantic code or <|audio_end|>: guidance on the top-k of the cond logits, then top-k sampling, see SemanticGenerationStep
    llama_token sample() {
        const size_t n = logits_cond.size();
        std::vector<std::pair<float, size_t>> cand;
        cand.reserve(n);
        for (size_t i = 0; i < n; i++) {
            const llama_token t = tok_audio_end + (llama_token) i;
            if (t == tok_audio_end || t >= tok_code0) {
                cand.emplace_back(logits_cond[i], i);
            }
        }
        const size_t k = std::min(cand.size(), (size_t) top_k);
        std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), std::greater<std::pair<float, size_t>>());
        const float thr = cand[k - 1].first;

        std::vector<std::pair<float, size_t>> guided;
        for (const auto & c : cand) {
            if (c.first >= thr) {
                const float u = use_cfg ? logits_uncond[c.second] : c.first;
                guided.emplace_back(u + (c.first - u) * cfg_scale, c.second);
            }
        }
        std::sort(guided.begin(), guided.end(), std::greater<std::pair<float, size_t>>());
        if (temp <= 0.0f) {
            return tok_audio_end + (llama_token) guided[0].second;
        }
        // values below the k-th one are masked, ties stay
        const float gthr = guided[std::min(guided.size(), (size_t) top_k) - 1].first;
        std::vector<double> p;
        double sum = 0.0;
        for (const auto & g : guided) {
            if (g.first < gthr) {
                break;
            }
            p.push_back(std::exp((double) (g.first - guided[0].first) / temp));
            sum += p.back();
        }
        double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
        for (size_t i = 0; i < p.size(); i++) {
            u -= p[i];
            if (u <= 0.0) {
                return tok_audio_end + (llama_token) guided[i].second;
            }
        }
        return tok_audio_end + (llama_token) guided[p.size() - 1].second;
    }

    // one frame: semantic code, depth decoder, condition row, then the next backbone input
    bool gen_frame() {
        const llama_token tok = sample();
        if (tok == tok_audio_end) {
            done = true;
            return true;
        }

        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type      = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0     = tok - tok_code0;
        inp.feats     = h_rows.data();
        inp.n_feats   = h_rows.size();
        inp.cfg_scale = use_cfg ? cfg_scale : 1.0f;
        inp.temp      = temp;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0 || out.n_feats != (size_t) n_embd) {
            LOG_ERR("mtmd_helper_gen_audio: depth decoder failed\n");
            return false;
        }
        if (n_gen++ > 0) {
            cond_rows.insert(cond_rows.end(), out.feats, out.feats + out.n_feats);
            if (n_frames() >= max_audio_frames) {
                done = true;
                return true;
            }
        }
        if (n_pos_needed(pos + 1) > (int64_t) llama_n_ctx_seq(lctx)) {
            LOG_WRN("mtmd_helper_gen_audio: the context is full, the song ends at %zu frames\n", n_frames());
            done = true;
            return true;
        }
        return decode_embd(out.embd);
    }

    // flow matching and vocoder of one window, then the crop of the reference
    bool run_window(bool last) {
        const size_t s = window_start(n_win_done);
        const size_t e = std::min(s + win_frames, n_frames());
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.feats   = cond_rows.data() + (s - n_rows_dropped) * n_embd;
        inp.n_feats = (e - s) * n_embd;
        inp.n_steps = flow_steps;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: flow matching failed\n");
            return false;
        }
        const size_t n_ch  = (size_t) info.n_channels;
        const size_t n     = out.n_samples / n_ch;
        const size_t left  = n_win_done == 0 ? 0 : crop_left * hop_samples;
        const size_t right = last ? 0 : crop_right * hop_samples;
        if (n < left + right) {
            LOG_ERR("mtmd_helper_gen_audio: window is too short\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio + left * n_ch, out.audio + (n - right) * n_ch);
        n_win_done++;
        // the rows before the next window are not needed anymore
        const size_t n_drop = std::min(window_start(n_win_done), n_frames()) - n_rows_dropped;
        cond_rows.erase(cond_rows.begin(), cond_rows.begin() + n_drop * n_embd);
        n_rows_dropped += n_drop;
        return true;
    }

    llama_seq_id seq_id     = 0;
    llama_seq_id seq_uncond = 0;
    bool  use_cfg    = false;
    float cfg_scale  = 1.0f;
    float temp       = 1.0f;
    int32_t flow_steps = 0; // 0 = the model default
    llama_pos pos = 0;
    std::vector<llama_token> prompt;
    std::vector<llama_token> prompt_uncond;
    int32_t prompt_pos  = 0;
    int32_t uprompt_pos = 0;
    std::vector<float> logits_cond;   // see save_outputs()
    std::vector<float> logits_uncond;
    std::vector<float> h_rows;        // backbone hidden of the cond, then uncond sequence
    size_t n_gen = 0;                 // frames sampled, the first one included
    std::vector<float> cond_rows;     // one condition row per frame, from frame n_rows_dropped
    size_t n_rows_dropped = 0;
    size_t n_win_done = 0;
    bool done    = false;
    bool decoded = false;
    std::mt19937 rng;

    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    bool stream = false;
    std::vector<float> audio_pcm; // interleaved
    std::vector<char>  out_buf;
    size_t pcm_sent        = 0;
    bool   wav_header_sent = false;
};

static std::unique_ptr<mtmd_gen_audio_pipeline> make_pipeline(llama_context * lctx, mtmd_context * mctx) {
    switch (mtmd_gen_audio_get_info(mctx).type) {
        case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new qwen3tts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_POCKETTTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new pockettts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_SOPRANO:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new soprano_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_KOKORO:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new kokoro_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_LFM3A:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new lfm3a_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_CHATTERBOX:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new chatterbox_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_DRAMA3:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new drama3_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_ZONOS3:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new zonos3_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_YUE2:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new yue2_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_MINIMAX_MUSIC3:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new music3_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_FISH_S2:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new fish_s2_gen_audio_pipeline(lctx, mctx));
        default:
            return nullptr;
    }
}

struct mtmd_helper_gen_audio {
    llama_context * lctx = nullptr;
    std::unique_ptr<mtmd_gen_audio_pipeline> pipeline;
};

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * mctx) {
    auto * ctx = new mtmd_helper_gen_audio();
    ctx->lctx     = lctx;
    ctx->pipeline = make_pipeline(lctx, mctx);
    if (ctx->pipeline && !ctx->pipeline->check_context()) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
}

int32_t mtmd_helper_gen_audio_max_frames(const mtmd_helper_gen_audio * ctx) {
    return ctx && ctx->pipeline ? ctx->pipeline->max_frames() : 0;
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx) {
    if (ctx && ctx->pipeline) {
        ctx->pipeline->reset();
    }
}

struct mtmd_helper_gen_audio_inp mtmd_helper_gen_audio_inp_default(void) {
    mtmd_helper_gen_audio_inp inp{};
    inp.top_k    = 50;
    inp.top_p    = 1.0f;
    inp.temp     = -1.0f; // model default
    inp.seed     = UINT32_MAX; // random
    inp.speed    = 1.0f;
    inp.loudness = 0.0f;
    inp.cfg_scale  = 0.0f; // model default
    inp.flow_steps = 0;    // model default
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    return inp;
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx, const mtmd_helper_gen_audio_inp * inp) {
    if (!ctx->pipeline) {
        LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
        return 1;
    }
    if (inp->n_past > 0 && !ctx->pipeline->supports_continue()) {
        LOG_ERR("mtmd_helper_gen_audio: the pipeline cannot continue a sequence\n");
        return 1;
    }
    if (!std::isfinite(inp->speed) || inp->speed <= 0.0f || !std::isfinite(inp->loudness) || inp->loudness > 0.0f) {
        LOG_ERR("mtmd_helper_gen_audio: speed must be > 0 and loudness < 0 LUFS (0 = model default)\n");
        return 2;
    }
    if (inp->speed != 1.0f && !ctx->pipeline->supports_speed()) {
        LOG_ERR("mtmd_helper_gen_audio: the model does not support speed\n");
        return 2;
    }
    if (inp->loudness != 0.0f && !ctx->pipeline->supports_delivery()) {
        LOG_ERR("mtmd_helper_gen_audio: the model does not support loudness\n");
        return 2;
    }
    if (!std::isfinite(inp->cfg_scale) || inp->cfg_scale < 0.0f) {
        LOG_ERR("mtmd_helper_gen_audio: cfg_scale must be > 0 (0 = model default)\n");
        return 2;
    }
    if (inp->cfg_scale != 0.0f && !ctx->pipeline->supports_cfg()) {
        LOG_ERR("mtmd_helper_gen_audio: the model does not support cfg_scale\n");
        return 2;
    }
    if (inp->flow_steps < 0 || inp->flow_steps > 1024) {
        LOG_ERR("mtmd_helper_gen_audio: flow_steps must be in [1, 1024] (0 = model default)\n");
        return 2;
    }
    if (inp->flow_steps != 0 && !ctx->pipeline->supports_flow_steps()) {
        LOG_ERR("mtmd_helper_gen_audio: the model does not support flow_steps\n");
        return 2;
    }
    ctx->pipeline->logit_bias.clear();
    return ctx->pipeline->set_input(inp);
}

int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio * ctx, int32_t n_batch) {
    if (!ctx->pipeline) {
        return -1;
    }
    // the context may use a smaller n_batch than requested (e.g. clamped to n_ctx), llama_process() would fail above it
    if (n_batch <= 0 || (uint32_t) n_batch > llama_n_batch(ctx->lctx)) {
        LOG_ERR("mtmd_helper_gen_audio: n_batch must be in [1, %u], got %d\n", llama_n_batch(ctx->lctx), n_batch);
        return -1;
    }
    return ctx->pipeline->step_prompt(n_batch);
}

bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio * ctx) {
    return ctx->pipeline && ctx->pipeline->next_is_text();
}

const llama_logit_bias * mtmd_helper_gen_audio_get_logit_bias(mtmd_helper_gen_audio * ctx, size_t * n_bias) {
    *n_bias = ctx->pipeline ? ctx->pipeline->logit_bias.size() : 0;
    return *n_bias ? ctx->pipeline->logit_bias.data() : nullptr;
}

int32_t mtmd_helper_audio_stream_get_logit_bias(const mtmd_context * mctx, const llama_model * model,
                                                llama_logit_bias * out, int32_t n_max) {
    std::vector<llama_logit_bias> bias;
    if (mtmd_gen_audio_get_info(mctx).type == MTMD_GEN_AUDIO_TYPE_LFM3A) {
        // LFM3-Audio live ASR: only <|text_end|> ends the text of a chunk, the stream adds the final <|mixed_end|>
        const llama_vocab * vocab = llama_model_get_vocab(model);
        for (const char * piece : { "<|mixed_end|>", "<|mixed_start|>", "<|audio_start|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>" }) {
            const llama_token tok = find_special_token(vocab, piece);
            if (tok != LLAMA_TOKEN_NULL) {
                bias.push_back({ tok, -INFINITY });
            }
        }
    }
    for (int32_t i = 0; i < n_max && i < (int32_t) bias.size(); i++) {
        out[i] = bias[i];
    }
    return (int32_t) bias.size();
}

int32_t mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio * ctx, llama_token sampled,
                                       const float * h_state_in, const float ** h_state_out,
                                       bool * out_stop) {
    if (!ctx->pipeline) {
        return 1;
    }
    bool stop = false;
    const int32_t ret = ctx->pipeline->step_gen(sampled, h_state_in, h_state_out, &stop);
    if (out_stop) {
        *out_stop = stop;
    }
    return ret;
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * ctx, int32_t * out_sample_rate,
                                         const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) {
    if (!ctx->pipeline) {
        return 1;
    }
    return ctx->pipeline->get_output(out_sample_rate, out_data, out_data_len, out_n_samples);
}

int32_t mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio * ctx) {
    if (!ctx->pipeline) {
        return 1;
    }
    return ctx->pipeline->flush();
}

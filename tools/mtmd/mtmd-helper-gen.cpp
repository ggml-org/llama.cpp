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

// --tts-lang codes -> the languages of the LFM3-Audio TTS system prompt, other values are taken as the name
static const std::unordered_map<std::string, std::string> lfm3a_tts_langs = {
    { "ar", "Arabic"  }, { "da", "Danish"   }, { "de", "German"  }, { "el", "Greek"   }, { "en", "American English" },
    { "en-us", "American English" }, { "en-gb", "British English" }, { "fi", "Finnish" }, { "fr", "French" },
    { "he", "Hebrew"  }, { "it", "Italian"  }, { "ja", "Japanese" }, { "ko", "Korean" }, { "nl", "Dutch"    },
    { "pl", "Polish"  }, { "pt", "Portuguese" }, { "es", "Spanish" }, { "sv", "Swedish" }, { "tr", "Turkish" },
    { "zh", "Chinese" },
};

static llama_token find_special_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void put_bytes(std::vector<char> & buf, const void * p, size_t n) {
    const char * c = (const char *) p;
    buf.insert(buf.end(), c, c + n);
}

// data_sz == UINT32_MAX writes the "unknown length" sentinel (streaming), same as ffmpeg does on a pipe
static void write_wav16_header(std::vector<char> & buf, uint32_t data_sz, int32_t rate) {
    const uint32_t riff_sz   = data_sz == UINT32_MAX ? UINT32_MAX : 36 + data_sz;
    const uint32_t fmt_sz    = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32    = (uint32_t) rate;
    put_bytes(buf, "RIFF", 4); put_bytes(buf, &riff_sz, 4); put_bytes(buf, "WAVE", 4);
    put_bytes(buf, "fmt ", 4); put_bytes(buf, &fmt_sz, 4);
    put_bytes(buf, &fmt, 2); put_bytes(buf, &ch, 2); put_bytes(buf, &rate32, 4);
    put_bytes(buf, &byte_rate, 4); put_bytes(buf, &align, 2); put_bytes(buf, &bits, 2);
    put_bytes(buf, "data", 4); put_bytes(buf, &data_sz, 4);
}

static void append_wav16_pcm(std::vector<char> & buf, const float * pcm, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
        put_bytes(buf, &s, 2);
    }
}

static bool write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate) {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
    write_wav16_header(buf, (uint32_t) (pcm.size() * 2), rate);
    append_wav16_pcm(buf, pcm.data(), pcm.size());
    return true;
}

class mtmd_gen_audio_pipeline {
public:
    mtmd_gen_audio_pipeline(llama_context * lctx, mtmd_context * mctx)
        : lctx(lctx), mctx(mctx), model(llama_get_model(lctx)), vocab(llama_model_get_vocab(model)),
          n_embd(llama_model_n_embd(model)), info(mtmd_gen_audio_get_info(mctx)) {}
    virtual ~mtmd_gen_audio_pipeline() = default;

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
    virtual bool next_is_text() const { return false; }

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
            if (!encode_speaker(inp->speaker_ref, speaker_embd)) {
                return 1;
            }
        }

        const int n_e = n_embd;
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
        std::vector<llama_token> ids(full.size() + 16);
        int n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(),
                                   false, true);
        if (n_ids < 8) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);

        std::vector<std::vector<float>> prompt;
        for (int i = 0; i < 3; i++) prompt.push_back(row(ids[(size_t) i]));
        prompt.push_back(sum_row(tts_pad, c_think));
        prompt.push_back(sum_row(tts_pad, c_think_b));
        prompt.push_back(sum_row(tts_pad, c_lang));
        prompt.push_back(sum_row(tts_pad, c_think_e));
        if (!speaker_embd.empty()) prompt.push_back(sum_vec(tts_pad, speaker_embd));
        prompt.push_back(sum_row(tts_bos, codec_pad));
        for (int i = 3; i < n_ids - 5; i++) prompt.push_back(sum_row(ids[(size_t) i], codec_pad));
        prompt.push_back(sum_row(tts_eos, codec_pad));
        prompt.push_back(sum_row(tts_pad, codec_bos));

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
        if (out.n_codes > 0 && codes_buf.size() / out.n_codes >= window_frames) {
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

    // runs the reference wav through the speaker encoder, returns one x-vector embedding row
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_support_audio(mctx)) {
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

    // must match hparams.wav_tfm_swa hardcoded in clip.cpp
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
// all three are 0 / false when the pack does not tune them, the model default is then used
struct pockettts_pack_settings {
    float temp             = 0.0f;
    int   frames_after_eos = 0;
    bool  pad_short_text   = false;
};

static pockettts_pack_settings pockettts_pack(const char * variant) {
    static const std::unordered_map<std::string, pockettts_pack_settings> packs = {
        { "english",         { 0.3f, 0, false } },
        { "english_2026-01", { 0.7f, 0, true  } },
        { "english_2026-04", { 0.3f, 0, false } },
        { "french_24l",      { 0.7f, 8, false } },
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
        if (pack.temp > 0.0f) {
            inp.temp = pack.temp;
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

    // runs the reference wav through the mimi encoder, returns one row per 12.5Hz frame
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_support_audio(mctx)) {
            LOG_ERR("mtmd_helper_gen_audio: mmproj has no voice encoder\n");
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
                    LOG_ERR("mtmd_helper_gen_audio: voice encode failed\n");
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
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = i == n - 1;
        }
        const int ret = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
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

// LFM3-Audio TTS, see generate() and sample_audio_frame() in the reference: the prompt opens the assistant turn with
// <|audio_start|>, every step samples one frame of codec codes from the backbone hidden state by MaskGIT over the
// depthformer and feeds their codebook embedding back; a frame with the end-of-audio code stops. the backbone token
// is not used. the codec is causal, every frame is decoded as soon as it exists
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

        if (inp->speaker_ref) {
            LOG_WRN("mtmd_helper_gen_audio: the model has no voice conditioning, the speaker file is ignored\n");
        }

        // continuing: the prompt is appended to the sequence as is, otherwise it is the text to speak
        continuing = inp->n_past > 0;
        pos        = continuing ? inp->n_past : 0;
        std::string lang = inp->lang && inp->lang[0] ? inp->lang : "en";
        std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        const auto it_lang = lfm3a_tts_langs.find(lang);
        lang = it_lang != lfm3a_tts_langs.end() ? it_lang->second : inp->lang;
        // LFM3-Omni (LFM3-Audio with a vision tower) opens every assistant turn with an empty think block
        const std::string think = mtmd_support_vision(mctx) ? "<think></think>" : "";
        const std::string full = continuing ? std::string(inp->prompt, inp->prompt_len) :
                                 "<|startoftext|><|im_start|>system\nPerform TTS in " + lang + " using a female voice.<|im_end|>\n" +
                                 "<|im_start|>user\n" + std::string(inp->prompt, inp->prompt_len) +
                                 "<|im_end|>\n<|im_start|>assistant\n" + think + "<|audio_start|>";
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
        // the last prompt token picks the mode, see generate() of the reference
        const llama_token last = prompt_tokens.back();
        mode = last == tok_audio_start ? MODE_AUDIO : last == tok_mixed_start ? MODE_MIXED : MODE_TEXT;
        slots_left = n_text_block;

        // the sampling defaults of the model, the caller's top-k/top-p are not used
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        temp        = def.temp;
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
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            batch.token   [i]    = prompt_tokens[(size_t) (prompt_pos + i)];
            batch.pos     [i]    = pos + i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = seq_id;
            batch.logits  [i]    = prompt_pos + i == n_prompt - 1;
        }
        batch.n_tokens = n;
        const int ret = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
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

    // one slot of generate() of the reference: a text token taken from the caller, or an audio frame sampled from
    // the hidden state; either is fed back to the backbone
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
            llama_batch batch = llama_batch_get_one(&sampled, 1);
            if (!decode(batch)) {
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

    bool decode(llama_batch batch) {
        std::vector<llama_pos>      positions(batch.n_tokens);
        std::vector<int32_t>        n_seq(batch.n_tokens, 1);
        std::vector<llama_seq_id>   seq(batch.n_tokens, seq_id);
        std::vector<llama_seq_id *> seq_ptr(batch.n_tokens);
        std::vector<int8_t>         logits(batch.n_tokens, 1);
        for (int i = 0; i < batch.n_tokens; i++) {
            positions[i] = pos + i;
            seq_ptr[i]   = &seq[i];
        }
        batch.pos      = positions.data();
        batch.n_seq_id = n_seq.data();
        batch.seq_id   = seq_ptr.data();
        batch.logits   = logits.data();
        if (llama_decode(lctx, batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return false;
        }
        pos += batch.n_tokens;
        return true;
    }

    // generate() of the reference: text by default, <|audio_start|> speaks until the end-of-audio frame,
    // <|mixed_start|> interleaves blocks of text tokens and frames until <|text_end|>
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

static std::unique_ptr<mtmd_gen_audio_pipeline> make_pipeline(llama_context * lctx, mtmd_context * mctx) {
    switch (mtmd_gen_audio_get_info(mctx).type) {
        case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new qwen3tts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_POCKETTTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new pockettts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_SOPRANO:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new soprano_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_LFM3A:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new lfm3a_gen_audio_pipeline(lctx, mctx));
        default:
            return nullptr;
    }
}

struct mtmd_helper_gen_audio {
    std::unique_ptr<mtmd_gen_audio_pipeline> pipeline;
};

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * mctx) {
    auto * ctx = new mtmd_helper_gen_audio();
    ctx->pipeline = make_pipeline(lctx, mctx);
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
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
    inp.seed     = UINT32_MAX; // random
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
    return ctx->pipeline->set_input(inp);
}

int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio * ctx, int32_t n_batch) {
    if (!ctx->pipeline) {
        return -1;
    }
    return ctx->pipeline->step_prompt(n_batch);
}

bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio * ctx) {
    return ctx->pipeline && ctx->pipeline->next_is_text();
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

#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <memory>
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

static bool write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate) {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
    const uint32_t data_sz   = (uint32_t) (pcm.size() * 2);
    const uint32_t riff_sz   = 36 + data_sz;
    const uint32_t fmt_sz    = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32    = (uint32_t) rate;
    auto put = [&](const void * p, size_t n) {
        const char * c = (const char *) p;
        buf.insert(buf.end(), c, c + n);
    };
    put("RIFF", 4); put(&riff_sz, 4); put("WAVE", 4);
    put("fmt ", 4); put(&fmt_sz, 4);
    put(&fmt, 2); put(&ch, 2); put(&rate32, 4);
    put(&byte_rate, 4); put(&align, 2); put(&bits, 2);
    put("data", 4); put(&data_sz, 4);
    for (float v : pcm) {
        int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        put(&s, 2);
    }
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

protected:
    llama_context * lctx;
    mtmd_context  * mctx;
    const llama_model * model;
    const llama_vocab  * vocab;
    int n_embd;
    mtmd_gen_audio_info info;
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
        c2w_state.clear();
        audio_pcm.clear();
        overlay.clear();
        h_state_buf.clear();
        out_buf.clear();
        prompt_embd_buf.clear();
        prompt_batch.reset();
        n_prompt = 0;
        prompt_pos = 0;
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
        seed  = inp->seed;
        out_type = inp->out_type;

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
        llama_batch batch_view = prompt_batch->get_view(prompt_pos, n_tokens_batch);

        const bool is_last_batch = (prompt_pos + n_tokens_batch) == n_prompt;
        if (is_last_batch) {
            batch_view.logits[n_tokens_batch - 1] = 1;
        }

        if (llama_decode(lctx, batch_view) != 0) {
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
        inp.seed  = seed;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
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
        batch_embd.batch.logits[0] = 1;
        pos++;

        if (llama_decode(lctx, batch_embd.batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();

        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        if (!flush_gen_wav()) {
            return 1;
        }

        *out_sample_rate = info.sample_rate;
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
        const std::string  marker = mtmd_default_marker();
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
        inp.seed       = seed; // same seed as gen_code, else clip reseeds mid-generation
        inp.state_data = c2w_state.empty() ? nullptr : (const char *) c2w_state.data();
        inp.state_size = c2w_state.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_wav process failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        c2w_state.assign(out.state_data, out.state_data + out.state_size);
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
    uint32_t seed  = UINT32_MAX;
    std::vector<int32_t> codes_buf;
    std::vector<uint8_t> c2w_state;
    std::vector<float>   audio_pcm;
    std::vector<float> overlay;
    std::vector<float> h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
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
        dec_state.clear();
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

        seed     = inp->seed;
        out_type = inp->out_type;

        return 0;
    }

    int32_t step_prompt(int32_t n_batch) override {
        GGML_ASSERT(n_batch > 0);
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        const int32_t n_tokens_batch = std::min(n_batch, n_prompt - prompt_pos);
        llama_batch batch_view = prompt_batch->get_view(prompt_pos, n_tokens_batch);

        if ((prompt_pos + n_tokens_batch) == n_prompt) {
            batch_view.logits[n_tokens_batch - 1] = 1;
        }

        if (llama_decode(lctx, batch_view) != 0) {
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
        // clip only reseeds when the seed changes, so pass the same one on every step
        inp.seed = seed;
        if (pack.temp > 0.0f) {
            inp.temp = pack.temp;
        }
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
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

        decode_embd_batch batch_embd(const_cast<float *>(out.embd), 1, 1, n_embd);
        batch_embd.set_position_normal(pos, seq_id);
        batch_embd.batch.logits[0] = 1;
        pos++;

        if (llama_decode(lctx, batch_embd.batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();

        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        if (!flush_gen_wav()) {
            return 1;
        }

        *out_sample_rate = info.sample_rate;
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
        dec_state.clear();

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
        batch.batch.logits[n_rows - 1] = 1;
        if (llama_decode(lctx, batch.batch) != 0) {
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
        const std::string  marker = mtmd_default_marker();
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
        inp.seed       = seed;
        inp.state_data = dec_state.empty() ? nullptr : (const char *) dec_state.data();
        inp.state_size = dec_state.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: mimi decode failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        dec_state.assign(out.state_data, out.state_data + out.state_size);
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
    uint32_t seed = UINT32_MAX;
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
    std::vector<uint8_t> dec_state;
    std::vector<float>   audio_pcm;
    std::vector<float>   h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
};

static std::unique_ptr<mtmd_gen_audio_pipeline> make_pipeline(llama_context * lctx, mtmd_context * mctx) {
    switch (mtmd_gen_audio_get_info(mctx).type) {
        case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new qwen3tts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_POCKETTTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new pockettts_gen_audio_pipeline(lctx, mctx));
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
    if (ctx->pipeline) {
        ctx->pipeline->reset();
    }
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx, const mtmd_helper_gen_audio_inp * inp) {
    if (!ctx->pipeline) {
        LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
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

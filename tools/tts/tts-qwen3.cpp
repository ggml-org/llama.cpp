// Qwen3-TTS end to end driver: the talker runs as a plain llama model
// (codec rows live in the extended vocab), each generated frame goes
// through mtmd_gen_audio (code predictor + code2wav) and its hidden
// state feedback returns as the next input embedding.
//
// The prompt is the sum of two aligned streams (projected text + codec
// specials); tokens can only pick one embedding row, so the prompt is
// assembled as raw embeddings from rows read straight out of the
// talker GGUF:
//
//   role          text(<|im_start|>assistant\n)              3 vecs
//   prefill       tts_pad x4 + tts_bos
//                 + codec(think, think_bos, lang, think_eos, codec_pad)
//   trailing      text(utterance) + tts_eos, + codec_pad each
//   handoff       tts_pad + codec(codec_bos)                 1 vec

#include "llama.h"
#include "mtmd.h"
#include "common.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Read one dequantized row of a 2D tensor from a GGUF file.
struct gguf_row_reader {
    struct gguf_context * gguf = nullptr;
    struct ggml_context * meta = nullptr;
    FILE *                f    = nullptr;
    size_t                data_off = 0;

    bool open(const char * path) {
        struct ggml_init_params ip = { 0, nullptr, true };
        struct gguf_init_params gp = { true, &meta };
        gguf = gguf_init_from_file(path, gp);
        if (!gguf) {
            return false;
        }
        data_off = gguf_get_data_offset(gguf);
        f = fopen(path, "rb");
        return f != nullptr;
    }

    bool read_row(const char * tensor_name, int64_t row, std::vector<float> & out) {
        const int64_t idx = gguf_find_tensor(gguf, tensor_name);
        if (idx < 0) {
            return false;
        }
        struct ggml_tensor * t = ggml_get_tensor(meta, tensor_name);
        if (!t || row < 0 || row >= t->ne[1]) {
            return false;
        }
        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        std::vector<uint8_t> raw(row_bytes);
        if (fseek(f, (long) (data_off + gguf_get_tensor_offset(gguf, idx) + (size_t) row * row_bytes), SEEK_SET) != 0) {
            return false;
        }
        if (fread(raw.data(), 1, row_bytes, f) != row_bytes) {
            return false;
        }
        out.resize((size_t) t->ne[0]);
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data(), raw.data(), row_bytes);
        } else {
            const auto * traits = ggml_get_type_traits(t->type);
            traits->to_float(raw.data(), out.data(), t->ne[0]);
        }
        return true;
    }

    ~gguf_row_reader() {
        if (f) fclose(f);
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

static llama_token find_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void save_wav16(const char * path, const std::vector<float> & pcm, int rate) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        LOG_ERR("failed to open %s\n", path);
        return;
    }
    const uint32_t data_sz = (uint32_t) (pcm.size() * 2);
    const uint32_t riff_sz = 36 + data_sz;
    const uint32_t fmt_sz = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32 = (uint32_t) rate;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_sz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmt_sz, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate32, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_sz, 4, 1, f);
    for (float v : pcm) {
        int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    const char * model_path  = nullptr;
    const char * mmproj_path = nullptr;
    const char * out_path    = "output.wav";
    std::string  text;
    std::string  lang     = "english";
    int          max_new  = 512;
    int          n_gpu    = 999;

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "-m"))        model_path  = next("-m");
        else if (!strcmp(argv[i], "--mmproj"))  mmproj_path = next("--mmproj");
        else if (!strcmp(argv[i], "-p"))        text        = next("-p");
        else if (!strcmp(argv[i], "-o"))        out_path    = next("-o");
        else if (!strcmp(argv[i], "--lang"))    lang        = next("--lang");
        else if (!strcmp(argv[i], "--max-new")) max_new     = atoi(next("--max-new"));
        else if (!strcmp(argv[i], "-ngl"))      n_gpu       = atoi(next("-ngl"));
        else {
            fprintf(stderr,
                    "usage: %s -m talker.gguf --mmproj tts.gguf -p \"text\" [-o out.wav] [--lang english] "
                    "[--max-new n] [-ngl n]\n", argv[0]);
            return 1;
        }
    }
    if (!model_path || !mmproj_path || text.empty()) {
        fprintf(stderr, "need -m, --mmproj and -p\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { LOG_ERR("failed to load %s\n", model_path); return 1; }
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int           n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 4096;
    cparams.n_batch    = 4096;
    cparams.embeddings = true;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { LOG_ERR("failed to create context\n"); return 1; }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (!mctx) { LOG_ERR("failed to load %s\n", mmproj_path); return 1; }
    if (mtmd_gen_audio_get_type(mctx) == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("mmproj does not support audio generation\n");
        return 1;
    }

    // vocab landmarks: the codec rows sit after the text vocab
    const llama_token codec_0    = find_token(vocab, "<|codec_0|>");
    const llama_token codec_bos  = find_token(vocab, "<|codec_bos|>");
    const llama_token codec_eos  = find_token(vocab, "<|codec_eos_token|>");
    const llama_token codec_pad  = find_token(vocab, "<|codec_pad|>");
    const llama_token c_think    = find_token(vocab, "<|codec_think|>");
    const llama_token c_think_b  = find_token(vocab, "<|codec_think_bos|>");
    const llama_token c_think_e  = find_token(vocab, "<|codec_think_eos|>");
    const llama_token c_lang     = find_token(vocab, ("<|codec_language_" + lang + "|>").c_str());
    const llama_token tts_pad    = find_token(vocab, "<tts_pad>");
    const llama_token tts_bos    = find_token(vocab, "<tts_text_bos>");
    const llama_token tts_eos    = find_token(vocab, "<tts_text_eod>");
    for (llama_token t : { codec_0, codec_bos, codec_eos, codec_pad, c_think, c_think_b, c_think_e, c_lang,
                           tts_pad, tts_bos, tts_eos }) {
        if (t == LLAMA_TOKEN_NULL) {
            LOG_ERR("missing special token in vocab (lang '%s'?)\n", lang.c_str());
            return 1;
        }
    }

    // embedding rows straight from the gguf: the prompt sums two rows
    // per position, which tokens cannot express
    gguf_row_reader rows;
    if (!rows.open(model_path)) { LOG_ERR("failed to open %s for row reads\n", model_path); return 1; }
    const char * EMBD = "token_embd.weight";
    auto row = [&](llama_token t) {
        std::vector<float> v;
        if (!rows.read_row(EMBD, t, v)) { LOG_ERR("row read failed for token %d\n", t); exit(1); }
        return v;
    };
    auto sum_row = [&](llama_token a, llama_token b) {
        std::vector<float> va = row(a), vb = row(b);
        for (size_t i = 0; i < va.size(); i++) va[i] += vb[i];
        return va;
    };

    // upstream wrap, then slices: [0:3] role, [3:-5] utterance body
    const std::string full = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> ids(full.size() + 16);
    int n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(),
                               false, true);
    if (n_ids < 8) { LOG_ERR("tokenization failed\n"); return 1; }
    ids.resize((size_t) n_ids);

    std::vector<std::vector<float>> prompt;
    for (int i = 0; i < 3; i++)          prompt.push_back(row(ids[(size_t) i]));
    prompt.push_back(sum_row(tts_pad, c_think));
    prompt.push_back(sum_row(tts_pad, c_think_b));
    prompt.push_back(sum_row(tts_pad, c_lang));
    prompt.push_back(sum_row(tts_pad, c_think_e));
    prompt.push_back(sum_row(tts_bos, codec_pad));
    for (int i = 3; i < n_ids - 5; i++)  prompt.push_back(sum_row(ids[(size_t) i], codec_pad));
    prompt.push_back(sum_row(tts_eos, codec_pad));
    prompt.push_back(sum_row(tts_pad, codec_bos));

    const int n_prompt = (int) prompt.size();
    LOG_INF("prompt: %d positions (%d text tokens)\n", n_prompt, n_ids);

    // the talker rides the qwen3vl interleaved mrope: positions carry
    // n_pos_per_embd sections laid out [section * n_tokens + i], all
    // equal for a pure text/codec stream
    const bool mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ||
                       llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;
    const int  n_pos_sec = mrope ? 4 : 1;
    std::vector<llama_pos> pos_buf((size_t) n_pos_sec * (size_t) n_prompt);

    // prefill as one embd batch, logits on the last position
    std::vector<float> embd_buf((size_t) n_prompt * (size_t) n_embd);
    for (int i = 0; i < n_prompt; i++) {
        memcpy(embd_buf.data() + (size_t) i * n_embd, prompt[(size_t) i].data(), (size_t) n_embd * sizeof(float));
    }
    llama_batch batch = llama_batch_init(n_prompt, n_embd, 1);
    batch.n_tokens = n_prompt;
    batch.pos      = pos_buf.data();
    memcpy(batch.embd, embd_buf.data(), embd_buf.size() * sizeof(float));
    for (int i = 0; i < n_prompt; i++) {
        for (int sec = 0; sec < n_pos_sec; sec++) {
            pos_buf[(size_t) sec * n_prompt + (size_t) i] = i;
        }
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (int8_t) (i == n_prompt - 1);
    }
    if (llama_decode(lctx, batch) != 0) { LOG_ERR("prefill decode failed\n"); return 1; }

    // the text stream keeps flowing during generation: the input after
    // frame k adds trailing text row k on top of the codes embedding,
    // then tts_eos, then tts_pad once the utterance is spent
    std::vector<std::vector<float>> overlay;
    for (int i = 3; i < n_ids - 5; i++) overlay.push_back(row(ids[(size_t) i]));
    overlay.push_back(row(tts_eos));
    overlay.push_back(row(tts_pad));

    // AR loop: sample c0 among the semantic codec rows plus eos, hand
    // the hidden state to the generator, feed its embedding back
    mtmd_gen_audio_reset(mctx);
    std::vector<float> audio;
    std::vector<float> h((size_t) n_embd), fb((size_t) n_embd);
    int n_frames = 0;
    int pos      = n_prompt;

    for (; n_frames < max_new; n_frames++) {
        const float * logits = llama_get_logits_ith(lctx, -1);
        llama_token   best   = codec_eos;
        float         bestv  = logits[codec_eos];
        for (llama_token t = codec_0; t < codec_0 + 2048; t++) {
            if (logits[t] > bestv) { bestv = logits[t]; best = t; }
        }
        if (best == codec_eos) {
            break;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        memcpy(h.data(), he, (size_t) n_embd * sizeof(float));

        mtmd_gen_inp inp = {};
        inp.code0  = best - codec_0;
        inp.embd   = h.data();
        inp.n_embd = (size_t) n_embd;
        inp.top_k  = 50;
        inp.top_p  = 1.0f;
        mtmd_gen_out out = {};
        out.embd   = fb.data();
        out.n_embd = (size_t) n_embd;
        if (mtmd_gen_audio(mctx, &inp, &out) != 0) { LOG_ERR("mtmd_gen_audio failed\n"); return 1; }
        audio.insert(audio.end(), out.audio, out.audio + out.n_samples);

        const auto & ov = overlay[std::min((size_t) n_frames, overlay.size() - 1)];
        for (int i = 0; i < n_embd; i++) {
            fb[(size_t) i] += ov[(size_t) i];
        }

        batch.n_tokens = 1;
        memcpy(batch.embd, fb.data(), (size_t) n_embd * sizeof(float));
        for (int sec = 0; sec < n_pos_sec; sec++) {
            pos_buf[(size_t) sec] = pos;
        }
        pos++;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        if (llama_decode(lctx, batch) != 0) { LOG_ERR("decode failed at frame %d\n", n_frames); return 1; }
    }

    LOG_INF("generated %d frames, %zu samples (%.2f s)\n", n_frames, audio.size(), (double) audio.size() / 24000.0);
    save_wav16(out_path, audio, 24000);
    LOG_INF("wrote %s\n", out_path);

    batch.pos = nullptr;
    llama_batch_free(batch);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

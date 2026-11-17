#include "clip-stream.h"
#include "clip-graph.h"
#include "clip-media.h"
#include "models/models.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

// the model-independent part of a stream, the model-specific part is in the subclasses below
struct clip_audio_stream {
    clip_stream_ctx * ctx;
    clip_subcontext * sctx;

    // model constants
    int n_mel;
    int hop;
    int n_embd;          // decoder embedding size
    int frames_per_tok;  // mel frames per decoder row
    int samples_per_tok;
    int n_step_mel;      // mel frames per encoder call

    // samples not yet turned into mel frames, buf[0] is at absolute index buf_pos
    std::vector<float> buf;
    int64_t buf_pos   = 0;
    int64_t n_samples = 0; // total samples pushed, including padding
    int64_t n_frames  = 0; // mel frames computed so far

    std::vector<float> mel;     // pending mel frames, frame-major
    std::vector<float> embd;    // pending audio embeddings
    size_t embd_off = 0;        // rows of embd already consumed
    std::vector<float> out;     // rows of the last step

    int64_t n_rows = 0;         // rows produced so far
    int64_t n_rows_done = 0;    // audio rows whose text has been sampled, see clip_audio_stream_get_n_rows()
    int64_t n_rows_flushed = 0; // audio rows up to the end of the last flush
    bool pending_flush = false;
    bool ended = false;
    bool seq_start = false;     // the rows of the last step start a new sequence


    virtual ~clip_audio_stream() = default;

    // model constants and initial state, false on failure
    virtual bool    init(const llama_audio_stream_params & params) = 0;
    virtual int32_t flush(bool end) = 0;
    virtual int32_t step(llama_token last_token) = 0;

    void add_silence(int64_t n) {
        buf.insert(buf.end(), n, 0.0f);
        n_samples += n;
    }
};

// Voxtral Realtime: a row is an audio embedding plus the embedding of the previous token
struct clip_audio_stream_voxtral_rt : clip_audio_stream {
    clip_subctx_enc_state * enc_state = nullptr;

    int n_fft;
    int n_frames_max;    // encoder frames per encode() call
    int n_right_pad;     // silence tokens added by flush()
    float log_mel_max;   // global_log_mel_max in params.json

    std::vector<llama_token> prefix; // text tokens paired with the first rows

    clip_audio_cache cache;

    std::vector<float> mel_ctx; // last 2 mel frames, the left context of the conv stem
    std::vector<float> tok_embd;

    ~clip_audio_stream_voxtral_rt() override {
        delete enc_state;
    }

    bool    init(const llama_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;
};

// LFM3-Audio: the rows are plain audio embeddings of a chat turn, the model answers after the audio
// live ASR (with a prompt), trained layout: <prompt> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|>
// training samples are max 30 s: restart the sequence at a silent chunk after 15 s, always after 30 s; the encoder keeps its state
struct clip_audio_stream_lfm3a : clip_audio_stream {
    int win; // samples per mel frame
    clip_subctx_enc_state * enc_state = nullptr;

    // live ASR
    static constexpr int64_t chunk_rows       = 12;
    static constexpr int32_t max_chunk_tokens = 48; // text tokens per chunk, then the chunk is closed
    static constexpr int32_t max_final_tokens = 96;

    std::vector<llama_token> prompt; // empty: the rows of a chat turn
    llama_token tok_text_end  = LLAMA_TOKEN_NULL;
    llama_token tok_mixed_end = LLAMA_TOKEN_NULL;
    int64_t soft_seq_rows = 0;
    int64_t max_seq_rows  = 0;

    bool    started        = false;
    bool    done           = false; // the text of the final chunk is complete
    bool    in_chunk       = false; // the text of the last chunk is being sampled
    bool    final_chunk    = false;
    bool    chunk_has_text = false;
    int32_t n_chunk_text   = 0;
    int64_t n_chunk_rows   = 0;
    int64_t n_seq_rows     = 0;     // audio rows since the sequence started
    std::vector<llama_token> closing; // the tokens that close the last chunk, they go in before the next one
    std::vector<float> tok_embd;
    // the audio markers of the model (LFM3-Omni), as rows around the audio
    std::vector<float> beg_rows;
    std::vector<float> end_rows;
    bool beg_done = false;
    bool end_done = false;

    ~clip_audio_stream_lfm3a() override {
        delete enc_state;
    }

    bool    init(const llama_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;

    bool    add_tokens(const llama_token * tokens, size_t n);
    int32_t step_live(llama_token last_token);
};

// Gemma 4: the encoder is causal
struct clip_audio_stream_gemma4a : clip_audio_stream {
    clip_subctx_enc_state * enc_state = nullptr;

    int n_fft;
    int win;
    int pad_left;
    int64_t seg_len;
    int64_t seg_start = 0; // first sample of the current segment, n_frames counts its frames

    std::vector<float> beg_rows;
    std::vector<float> end_rows;
    bool beg_done = false;
    bool end_done = false;

    ~clip_audio_stream_gemma4a() override {
        delete enc_state;
    }

    bool    init(const llama_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;

    int64_t n_seg_frames(int64_t n, bool complete) const;
    bool    compute_mel(int64_t n_seg, bool complete);
    bool    encode(int n);
};

// Higgs Audio: each 4 s chunk is encoded on its own, so a chunk goes out once it is complete
// a flush encodes the partial chunk as a chunk of its own, the reference also cuts chunks at VAD boundaries
struct clip_audio_stream_higgs_audio : clip_audio_stream {
    int64_t chunk_len = 0; // samples per complete chunk

    std::vector<float> beg_rows;
    std::vector<float> end_rows;
    bool beg_done = false;
    bool end_done = false;

    bool    init(const llama_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;

    int64_t n_chunk_rows(int64_t n) const;
    bool    encode(int64_t n);
};

// VibeVoice-ASR: the encoder is causal, its convs keep their left context in the state
// the reference scales the whole clip to -25 dBFS, here the gain comes from the audio pushed so far (see clip_audio_vibevoice_gain)
struct clip_audio_stream_vibevoice_asr : clip_audio_stream {
    clip_subctx_enc_state * enc_state = nullptr;

    int64_t n_step     = 0; // samples per encoder call
    int64_t n_step_max = 0; // largest call, the warmup size
    int64_t n_stat     = 0; // pushed samples counted in sum_sq and peak
    double  sum_sq     = 0.0;
    double  peak       = 0.0;

    std::vector<float> beg_rows;
    std::vector<float> end_rows;
    bool beg_done = false;
    bool end_done = false;

    ~clip_audio_stream_vibevoice_asr() override {
        delete enc_state;
    }

    bool    init(const llama_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;

    bool    encode(int64_t n);
};

bool clip_support_audio_stream(const clip_stream_ctx * ctx) {
    if (!ctx->ctx_a || !ctx->text_model) {
        return false;
    }
    const projector_type proj = ctx->ctx_a->model.proj_type;
    return proj == PROJECTOR_TYPE_VOXTRAL_RT || proj == PROJECTOR_TYPE_LFM3A || proj == PROJECTOR_TYPE_GEMMA4A ||
           proj == PROJECTOR_TYPE_HIGGS_AUDIO || proj == PROJECTOR_TYPE_VIBEVOICE_ASR;
}

llama_audio_stream_params llama_audio_stream_default_params(void) {
    llama_audio_stream_params params;
    params.step_ms  = 320;
    params.prompt   = nullptr;
    params.n_prompt = 0;
    return params;
}

clip_audio_stream * clip_audio_stream_init(clip_stream_ctx * ctx, llama_audio_stream_params params) {
    if (!clip_support_audio_stream(ctx)) {
        LOG_ERR("%s: the model does not support streaming audio input\n", __func__);
        return nullptr;
    }
    const clip_hparams * hparams = &ctx->ctx_a->model.hparams;

    std::unique_ptr<clip_audio_stream> s;
    switch (ctx->ctx_a->model.proj_type) {
        case PROJECTOR_TYPE_VOXTRAL_RT: s = std::make_unique<clip_audio_stream_voxtral_rt>(); break;
        case PROJECTOR_TYPE_LFM3A:      s = std::make_unique<clip_audio_stream_lfm3a>();      break;
        case PROJECTOR_TYPE_GEMMA4A:    s = std::make_unique<clip_audio_stream_gemma4a>();    break;
        case PROJECTOR_TYPE_HIGGS_AUDIO:   s = std::make_unique<clip_audio_stream_higgs_audio>();   break;
        case PROJECTOR_TYPE_VIBEVOICE_ASR: s = std::make_unique<clip_audio_stream_vibevoice_asr>(); break;
        default:                        GGML_ABORT("unsupported projector");
    }
    s->ctx      = ctx;
    s->sctx = ctx->ctx_a;
    s->n_mel    = hparams->n_mel_bins;
    s->hop      = hparams->audio_hop_len;
    s->n_embd   = ctx->ctx_a->model.n_mmproj_embd();
    if (s->n_embd != ctx->n_embd_text) {
        LOG_ERR("%s: mmproj embedding size %d does not match the text model (%d)\n", __func__, s->n_embd, ctx->n_embd_text);
        return nullptr;
    }
    if (!s->init(params)) {
        return nullptr;
    }
    return s.release();
}

void clip_audio_stream_free(clip_audio_stream * stream) {
    delete stream;
}

int32_t clip_audio_stream_push(clip_audio_stream * s, const float * samples, size_t n_samples) {
    if (s->ended) {
        return 1;
    }
    s->buf.insert(s->buf.end(), samples, samples + n_samples);
    s->n_samples += n_samples;
    return 0;
}

int32_t clip_audio_stream_flush(clip_audio_stream * s, bool end) {
    if (s->ended) {
        return 1;
    }
    return s->flush(end);
}

int32_t clip_audio_stream_step(clip_audio_stream * s, llama_token last_token) {
    return s->step(last_token);
}

const float * clip_audio_stream_get_output_embd(const clip_audio_stream * s) {
    return s->out.data();
}

bool clip_audio_stream_is_seq_start(const clip_audio_stream * s) {
    return s->seq_start;
}


// token embeddings of aud_beg / aud_end
static bool clip_audio_stream_marker_rows(const clip_stream_ctx * ctx, const std::string & text, std::vector<float> & rows) {
    rows.clear();
    if (text.empty()) {
        return true;
    }
    const llama_vocab * vocab = llama_model_get_vocab(ctx->text_model);
    std::vector<llama_token> tokens(text.size() + 4);
    const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, true);
    if (n <= 0) {
        LOG_ERR("%s: cannot tokenize the audio marker %s\n", __func__, text.c_str());
        return false;
    }
    rows.resize((size_t) n * ctx->n_embd_text);
    return llama_model_get_tok_embd_rows(ctx->text_model, tokens.data(), n, rows.data()) != 0;
}

//
// Voxtral Realtime
//

bool clip_audio_stream_voxtral_rt::init(const llama_audio_stream_params & params) {
    const clip_hparams * hparams = &sctx->model.hparams;
    const llama_vocab * vocab = ctx->vocab;

    n_fft = hparams->audio_n_fft;

    // one decoder row per 80 ms of audio: the conv stem has stride 2, then frames are stacked
    frames_per_tok  = 2 * hparams->proj_stack_factor;
    samples_per_tok = frames_per_tok * hop;
    n_frames_max    = (hparams->warmup_audio_size - 2) / 2;
    log_mel_max     = 1.5f;

    // the encoder works in whole decoder rows
    int n_step = (int) ((int64_t) params.step_ms * hparams->audio_sample_rate / 1000 / hop);
    n_step = std::max(n_step, frames_per_tok);
    n_step = CLIP_ALIGN(n_step, frames_per_tok);
    n_step_mel = std::min(n_step, 2 * n_frames_max);

    if (params.n_prompt > 0) {
        LOG_ERR("%s: Voxtral Realtime takes no prompt\n", __func__);
        return false;
    }

    // prefix: BOS then one pad token per silence token and per delay token
    // the output lags behind the audio by n_delay rows, the text model is conditioned on the same value
    const int n_delay    = llama_model_n_tokens_delay(ctx->text_model);
    const int n_left_pad = 32;
    n_right_pad = n_delay + 1 + 10;
    {
        const std::string pad_str = "[STREAMING_PAD]";
        std::vector<llama_token> pad(8);
        const int n_pad = llama_tokenize(vocab, pad_str.c_str(), pad_str.size(), pad.data(), pad.size(), false, true);
        if (n_pad != 1) {
            LOG_ERR("%s: cannot find the %s token\n", __func__, pad_str.c_str());
            return false;
        }
        prefix.push_back(llama_vocab_bos(vocab));
        prefix.insert(prefix.end(), n_left_pad + n_delay, pad[0]);
    }

    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(hparams->audio_window_len, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, hparams->audio_sample_rate);

    enc_state = sctx->enc_state_init(n_frames_max);
    if (!enc_state) {
        return false;
    }

    // the first mel frame is centered on the first sample, so its window starts before the audio
    buf.assign(n_fft / 2, 0.0f);
    buf_pos = -n_fft / 2;
    mel_ctx.assign(2 * n_mel, 0.0f);

    // silence before the audio
    add_silence((int64_t) n_left_pad * samples_per_tok);

    return true;
}

int32_t clip_audio_stream_voxtral_rt::flush(bool end) {
    // align to a whole row, then add the silence the model expects after the audio
    const int64_t n_align = (samples_per_tok - n_samples % samples_per_tok) % samples_per_tok;
    add_silence(n_align + (int64_t) n_right_pad * samples_per_tok);
    // one row per samples_per_tok, the left padding included
    n_rows_flushed = n_samples / samples_per_tok;
    pending_flush = true;
    ended = end;
    return 0;
}

// mel frame k is centered on sample k * hop, with force the samples past the end are taken as silence
static void clip_audio_stream_compute_mel(clip_audio_stream_voxtral_rt * s, bool force) {
    const int half = s->n_fft / 2;
    std::vector<float> window(s->n_fft);
    std::vector<float> frame(s->n_mel);

    while (true) {
        const int64_t start = s->n_frames * s->hop - half;
        if (start + s->n_fft > s->n_samples && !(force && start + half < s->n_samples)) {
            break;
        }
        for (int j = 0; j < s->n_fft; j++) {
            const int64_t idx = start + j - s->buf_pos;
            window[j] = idx < (int64_t) s->buf.size() ? s->buf[idx] : 0.0f;
        }
        clip_audio_log_mel_frame(s->cache, window.data(), s->n_fft, s->n_mel, frame.data());
        for (int m = 0; m < s->n_mel; m++) {
            const float v = std::max(frame[m], s->log_mel_max - 8.0f);
            s->mel.push_back((v + 4.0f) / 4.0f);
        }
        s->n_frames++;
    }

    // drop the samples no frame will need again
    const int64_t keep_from = s->n_frames * s->hop - half;
    if (keep_from > s->buf_pos) {
        s->buf.erase(s->buf.begin(), s->buf.begin() + (keep_from - s->buf_pos));
        s->buf_pos = keep_from;
    }
}

static bool clip_audio_stream_encode(clip_audio_stream_voxtral_rt * s, bool force) {
    while (true) {
        const int n_avail = (int) (s->mel.size() / s->n_mel);
        int n = n_avail - n_avail % s->frames_per_tok;
        if (!force) {
            if (n_avail < s->n_step_mel) {
                break;
            }
            n = std::min(n, s->n_step_mel);
        }
        n = std::min(n, 2 * s->n_frames_max);
        if (n <= 0) {
            break;
        }

        // 2 context frames, then the new frames, stored mel-major
        const int n_in = 2 + n;
        std::vector<float> data((size_t) s->n_mel * n_in);
        for (int m = 0; m < s->n_mel; m++) {
            data[(size_t) m * n_in + 0] = s->mel_ctx[m];
            data[(size_t) m * n_in + 1] = s->mel_ctx[s->n_mel + m];
            for (int t = 0; t < n; t++) {
                data[(size_t) m * n_in + 2 + t] = s->mel[(size_t) t * s->n_mel + m];
            }
        }

        clip_image_f32 img;
        img.set_size({n_in, s->n_mel}, false, /* is_audio */ true);
        img.cpy_buf(data);
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(img));

        std::vector<float> res((size_t) s->n_embd * (n / s->frames_per_tok));
        clip_subctx_encode_params params;
        params.n_threads = s->ctx->n_threads;
        params.imgs      = &batch;
        params.out_embd  = &res;
        params.enc_state = s->enc_state;
        if (!s->sctx->encode(&params)) {
            LOG_ERR("%s: failed to encode audio\n", __func__);
            return false;
        }
        s->embd.insert(s->embd.end(), res.begin(), res.end());

        s->mel_ctx.assign(s->mel.begin() + (size_t) (n - 2) * s->n_mel, s->mel.begin() + (size_t) n * s->n_mel);
        s->mel.erase(s->mel.begin(), s->mel.begin() + (size_t) n * s->n_mel);
    }
    return true;
}

// out rows = audio embeddings + token embeddings
static bool clip_audio_stream_add_rows(clip_audio_stream_voxtral_rt * s, const llama_token * tokens, size_t n) {
    s->tok_embd.resize(n * s->n_embd);
    if (llama_model_get_tok_embd_rows(s->ctx->text_model, tokens, n, s->tok_embd.data()) == 0) {
        LOG_ERR("%s: invalid token\n", __func__);
        return false;
    }
    const float * audio = s->embd.data() + s->embd_off * s->n_embd;
    for (size_t i = 0; i < n * s->n_embd; i++) {
        s->out.push_back(audio[i] + s->tok_embd[i]);
    }
    s->embd_off += n;
    s->n_rows   += n;
    return true;
}

int32_t clip_audio_stream_voxtral_rt::step(llama_token last_token) {
    const bool force = pending_flush;
    pending_flush = false;

    // last_token was sampled from the last row, the text of every row so far is out
    n_rows_done = n_rows;

    clip_audio_stream_compute_mel(this, force);
    if (!clip_audio_stream_encode(this, force)) {
        return -2;
    }

    // drop the consumed rows once in a while
    if (embd_off >= 1024) {
        embd.erase(embd.begin(), embd.begin() + embd_off * n_embd);
        embd_off = 0;
    }

    const int64_t n_avail = (int64_t) (embd.size() / n_embd) - embd_off;
    out.clear();

    if (n_rows == 0) {
        // the prefix rows go out together, the caller samples from the last one
        const int64_t n_prefix = (int64_t) prefix.size();
        if (n_avail < n_prefix) {
            return ended ? -1 : 0;
        }
        if (!clip_audio_stream_add_rows(this, prefix.data(), prefix.size())) {
            return -2;
        }
        return (int32_t) n_prefix;
    }

    if (n_avail == 0) {
        return ended ? -1 : 0;
    }
    if (last_token == LLAMA_TOKEN_NULL) {
        LOG_ERR("%s: the token sampled from the previous row is required\n", __func__);
        return -2;
    }
    if (!clip_audio_stream_add_rows(this, &last_token, 1)) {
        return -2;
    }
    return 1;
}

//
// LFM3-Audio
//

bool clip_audio_stream_lfm3a::init(const llama_audio_stream_params & params) {
    const clip_hparams * hparams = &sctx->model.hparams;

    // one row per 8 mel frames (three stride-2 convs), mel frames are not centered
    win             = hparams->audio_window_len;
    frames_per_tok  = 8;
    samples_per_tok = frames_per_tok * hop;

    const int n_step = (int) ((int64_t) params.step_ms * hparams->audio_sample_rate / 1000 / hop);
    n_step_mel = CLIP_ALIGN(std::max(n_step, frames_per_tok), frames_per_tok);

    if (params.n_prompt > 0) {
        // live ASR, a chunk per encoder call
        n_step_mel = (int) chunk_rows * frames_per_tok;
        prompt.assign(params.prompt, params.prompt + params.n_prompt);

        const llama_vocab * vocab = llama_model_get_vocab(ctx->text_model);
        auto find_token = [&](const char * piece) {
            llama_token tok[2];
            const int n = llama_tokenize(vocab, piece, (int32_t) strlen(piece), tok, 2, false, true);
            return n == 1 ? tok[0] : LLAMA_TOKEN_NULL;
        };
        tok_text_end  = find_token("<|text_end|>");
        tok_mixed_end = find_token("<|mixed_end|>");
        if (tok_text_end == LLAMA_TOKEN_NULL || tok_mixed_end == LLAMA_TOKEN_NULL) {
            LOG_ERR("%s: the vocab lacks the tokens of LFM3-Audio live ASR\n", __func__);
            return false;
        }
        soft_seq_rows = std::lround(15.0 * hparams->audio_sample_rate / samples_per_tok);
        max_seq_rows  = std::lround(30.0 * hparams->audio_sample_rate / samples_per_tok);
    }

    enc_state = sctx->enc_state_init(n_step_mel);
    if (!enc_state) {
        return false;
    }

    // the markers the tokenizer puts around an audio clip go around the stream as well
    return clip_audio_stream_marker_rows(ctx, ctx->aud_beg, beg_rows) && clip_audio_stream_marker_rows(ctx, ctx->aud_end, end_rows);
}

// end: pad the tail like clip_audio_preprocessor_lfm3a, so the rows match encoding the clip at once
// otherwise only whole rows are encoded, a padded partial row would change the encoder state
// live ASR: pad the last chunk with silence, so its text comes out
int32_t clip_audio_stream_lfm3a::flush(bool end) {
    if (end) {
        const int64_t n_padded = n_samples <= win ? win : n_samples + (hop - (n_samples - win) % hop) % hop;
        add_silence(n_padded - n_samples);
    } else if (!prompt.empty() && n_samples > 0) {
        const int64_t frames_per_chunk = chunk_rows * frames_per_tok;
        const int64_t n_frames_audio   = n_samples <= win ? 1 : (n_samples - win + hop - 1) / hop + 1;
        const int64_t n_frames_padded  = (n_frames_audio + frames_per_chunk - 1) / frames_per_chunk * frames_per_chunk;
        add_silence(win + (n_frames_padded - 1) * hop - n_samples);
    }
    n_rows_flushed = (n_samples >= win ? (n_samples - win) / hop + 1 : 0) / frames_per_tok;
    pending_flush = true;
    ended = end;
    return 0;
}

// mel frame k covers samples [k * hop, k * hop + win), computed with the offline preprocessor
static bool clip_audio_stream_compute_mel(clip_audio_stream_lfm3a * s) {
    const int64_t n_total = s->n_samples >= s->win ? (s->n_samples - s->win) / s->hop + 1 : 0;
    const int64_t n_new   = n_total - s->n_frames;
    if (n_new <= 0) {
        return true;
    }
    const int64_t start = s->n_frames * s->hop - s->buf_pos;
    const size_t  len   = (size_t) (s->win + (n_new - 1) * s->hop);
    clip_audio_preproc_out out = s->ctx->audio_preproc->preprocess(s->buf.data() + start, len);
    if (out.entries.size() != 1 || !s->ctx->audio_preproc->compute(out)) {
        LOG_ERR("%s: unable to preprocess audio\n", __func__);
        return false;
    }
    const auto & mel = out.entries[0];
    const auto & data = mel.get_ro_buf();
    GGML_ASSERT(mel.nx() == n_new && mel.ny() == s->n_mel);
    for (int64_t t = 0; t < n_new; t++) {
        for (int m = 0; m < s->n_mel; m++) {
            s->mel.push_back(data[(size_t) m * mel.nx() + t]);
        }
    }
    s->n_frames = n_total;

    const int64_t keep_from = s->n_frames * s->hop;
    if (keep_from > s->buf_pos) {
        s->buf.erase(s->buf.begin(), s->buf.begin() + (keep_from - s->buf_pos));
        s->buf_pos = keep_from;
    }
    return true;
}

// whole rows only, except for the last call of an ended stream; the encoder state carries the rest
static bool clip_audio_stream_encode(clip_audio_stream_lfm3a * s, bool flush) {
    const int n_avail = (int) (s->mel.size() / s->n_mel);
    int n = n_avail - n_avail % s->frames_per_tok;
    if (flush && s->ended) {
        n = n_avail;
    } else if (!flush) {
        n -= n % s->n_step_mel;
    }
    if (n <= 0) {
        return true;
    }

    std::vector<float> data((size_t) s->n_mel * n);
    for (int m = 0; m < s->n_mel; m++) {
        for (int t = 0; t < n; t++) {
            data[(size_t) m * n + t] = s->mel[(size_t) t * s->n_mel + m];
        }
    }
    clip_image_f32 img;
    img.set_size({n, s->n_mel}, false, /* is_audio */ true);
    img.cpy_buf(data);
    const int n_rows = s->sctx->model.n_output_tokens(&img);

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(img));

    std::vector<float> res((size_t) s->n_embd * n_rows);
    clip_subctx_encode_params params;
    params.n_threads = s->ctx->n_threads;
    params.imgs      = &batch;
    params.out_embd  = &res;
    params.enc_state = s->enc_state;
    if (!s->sctx->encode(&params)) {
        LOG_ERR("%s: failed to encode audio\n", __func__);
        return false;
    }
    s->embd.insert(s->embd.end(), res.begin(), res.end());
    s->mel.erase(s->mel.begin(), s->mel.begin() + (size_t) n * s->n_mel);
    return true;
}

int32_t clip_audio_stream_lfm3a::step(llama_token last_token) {
    const bool flush = pending_flush;
    pending_flush = false;

    if (!clip_audio_stream_compute_mel(this) || !clip_audio_stream_encode(this, flush)) {
        return -2;
    }
    if (!prompt.empty()) {
        return step_live(last_token);
    }

    // rows carry no text, all of them go out at once and last_token is not used
    const int64_t n_avail = (int64_t) (embd.size() / n_embd) - embd_off;
    out.clear();
    if (n_avail > 0 || ended) {
        if (!beg_done) {
            beg_done = true;
            out = beg_rows;
        }
        out.insert(out.end(), embd.begin() + embd_off * n_embd, embd.end());
        embd.clear();
        embd_off = 0;
        if (ended && !end_done) {
            end_done = true;
            out.insert(out.end(), end_rows.begin(), end_rows.end());
        }
    }
    const int64_t n_out = (int64_t) (out.size() / n_embd);
    if (n_out == 0) {
        return ended ? -1 : 0;
    }
    n_rows      += n_out;
    n_rows_done += n_avail; // the markers are not audio rows
    return (int32_t) n_out;
}

bool clip_audio_stream_lfm3a::add_tokens(const llama_token * tokens, size_t n) {
    if (n == 0) {
        return true;
    }
    tok_embd.resize(n * n_embd);
    if (llama_model_get_tok_embd_rows(ctx->text_model, tokens, n, tok_embd.data()) == 0) {
        LOG_ERR("%s: invalid token\n", __func__);
        return false;
    }
    out.insert(out.end(), tok_embd.begin(), tok_embd.end());
    return true;
}

// last_token continues the text of the chunk, or <|text_end|> closes it and the next 12 rows go out (the rest at the end)
int32_t clip_audio_stream_lfm3a::step_live(llama_token last_token) {
    out.clear();
    seq_start = false;

    // drop the consumed rows once in a while
    if (embd_off >= 1024) {
        embd.erase(embd.begin(), embd.begin() + embd_off * n_embd);
        embd_off = 0;
    }

    if (in_chunk) {
        if (last_token == LLAMA_TOKEN_NULL) {
            LOG_ERR("%s: the token sampled from the previous row is required\n", __func__);
            return -2;
        }
        if (last_token != tok_text_end) {
            chunk_has_text = true;
            if (n_chunk_text + 1 < (final_chunk ? max_final_tokens : max_chunk_tokens)) {
                n_chunk_text++;
                return add_tokens(&last_token, 1) ? 1 : -2;
            }
            // close a runaway chunk like the model should have
            closing.push_back(last_token);
        }
        // the text of the chunk is complete, its closing tokens go in before the next chunk
        in_chunk     = false;
        n_rows_done += n_chunk_rows;
        if (final_chunk) {
            done = true;
        }
        closing.push_back(tok_text_end);
    }
    if (done) {
        return -1;
    }

    const int64_t n_avail = (int64_t) (embd.size() / n_embd) - embd_off;
    const bool    drained = ended && mel.empty();
    const bool    final   = drained && n_avail <= chunk_rows;
    if (!final && n_avail < chunk_rows) {
        return 0;
    }
    if (final && n_avail == 0 && !started) {
        done = true; // no audio at all
        return -1;
    }

    if (!started || n_seq_rows >= max_seq_rows || (n_seq_rows >= soft_seq_rows && !chunk_has_text)) {
        closing    = prompt; // the closing tokens go away with the sequence
        seq_start  = true;
        n_seq_rows = 0;
    }
    const int64_t n_feed = final ? n_avail : chunk_rows;
    if (!add_tokens(closing.data(), closing.size())) {
        return -2;
    }
    closing.clear();
    const float * rows = embd.data() + embd_off * n_embd;
    out.insert(out.end(), rows, rows + (size_t) n_feed * n_embd);
    embd_off += n_feed;
    if (final && !add_tokens(&tok_mixed_end, 1)) {
        return -2;
    }

    started        = true;
    in_chunk       = true;
    final_chunk    = final;
    chunk_has_text = false;
    n_chunk_text   = 0;
    n_chunk_rows   = n_feed;
    n_seq_rows    += n_feed;
    n_rows        += n_feed;
    return (int32_t) (out.size() / n_embd);
}

//
// Gemma 4
//

bool clip_audio_stream_gemma4a::init(const llama_audio_stream_params & params) {
    const clip_hparams * hparams = &sctx->model.hparams;

    if (params.n_prompt > 0) {
        LOG_ERR("%s: Gemma 4 takes no prompt, its audio is a chat turn\n", __func__);
        return false;
    }
    if (!ctx->audio_preproc || !ctx->audio_preproc->mel_config.cache) {
        LOG_ERR("%s: the audio preprocessor is not initialized\n", __func__);
        return false;
    }

    // one row per 4 mel frames (two stride-2 convs)
    n_fft           = hparams->audio_n_fft;
    win             = hparams->audio_window_len;
    pad_left        = win / 2;
    seg_len         = (int64_t) clip_audio_preprocessor_gemma4a::segment_seconds * hparams->audio_sample_rate;
    frames_per_tok  = 4;
    samples_per_tok = frames_per_tok * hop;

    // whole attention blocks, so the rows match the offline path
    const int n_step = (int) ((int64_t) params.step_ms * hparams->audio_sample_rate / 1000 / hop);
    n_step_mel = CLIP_ALIGN(std::max(n_step, frames_per_tok), (int) clip_graph_gemma4a::chunk_size * frames_per_tok);

    enc_state = sctx->enc_state_init(n_step_mel);
    if (!enc_state) {
        return false;
    }

    return clip_audio_stream_marker_rows(ctx, ctx->aud_beg, beg_rows) && clip_audio_stream_marker_rows(ctx, ctx->aud_end, end_rows);
}

// mel frames of a segment with n samples
// complete: same count as clip_audio_preprocessor_gemma4a, otherwise: only frames with a full window
int64_t clip_audio_stream_gemma4a::n_seg_frames(int64_t n, bool complete) const {
    if (complete) {
        return std::max<int64_t>(0, (n + pad_left - (win + 1)) / hop + 1);
    }
    return n + pad_left >= win + 1 ? (n + pad_left - (win + 1)) / hop + 1 : 0;
}

// frame k reads samples [k * hop - pad_left, k * hop - pad_left + n_fft) of the segment, zero outside of it
// the window is zero past win, so the samples after the segment end do not count
bool clip_audio_stream_gemma4a::compute_mel(int64_t n_seg, bool complete) {
    const int64_t n_total = n_seg_frames(n_seg, complete);
    const int64_t n_new   = n_total - n_frames;
    if (n_new <= 0) {
        return true;
    }

    const int64_t first = seg_start + n_frames * hop - pad_left;
    std::vector<float> samples((size_t) ((n_new - 1) * hop + n_fft), 0.0f);
    for (size_t j = 0; j < samples.size(); j++) {
        const int64_t idx = first + (int64_t) j;
        if (idx >= seg_start && idx < seg_start + n_seg) {
            GGML_ASSERT(idx >= buf_pos && idx - buf_pos < (int64_t) buf.size());
            samples[j] = buf[idx - buf_pos];
        }
    }

    const clip_audio_mel_config & cfg = ctx->audio_preproc->mel_config;
    std::vector<float> frames((size_t) n_new * n_mel);
    if (!clip_audio_log_mel(cfg, samples.data(), cfg.cache->hann_window.data(), n_new, frames.data())) {
        LOG_ERR("%s: unable to compute the mel spectrogram\n", __func__);
        return false;
    }
    for (int64_t t = 0; t < n_new; t++) {
        for (int m = 0; m < n_mel; m++) {
            mel.push_back(frames[(size_t) m * n_new + t]);
        }
    }
    n_frames = n_total;

    // drop the samples no frame will need again
    const int64_t keep_from = std::max(seg_start, seg_start + n_frames * hop - pad_left);
    if (keep_from > buf_pos) {
        buf.erase(buf.begin(), buf.begin() + (keep_from - buf_pos));
        buf_pos = keep_from;
    }
    return true;
}

bool clip_audio_stream_gemma4a::encode(int n) {
    std::vector<float> data((size_t) n_mel * n);
    for (int m = 0; m < n_mel; m++) {
        for (int t = 0; t < n; t++) {
            data[(size_t) m * n + t] = mel[(size_t) t * n_mel + m];
        }
    }
    clip_image_f32 img;
    img.set_size({n, n_mel}, false, /* is_audio */ true);
    img.cpy_buf(data);
    const int n_out = sctx->model.n_output_tokens(&img);

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(img));

    std::vector<float> res((size_t) n_embd * n_out);
    clip_subctx_encode_params params;
    params.n_threads = ctx->n_threads;
    params.imgs      = &batch;
    params.out_embd  = &res;
    params.enc_state = enc_state;
    if (!sctx->encode(&params)) {
        LOG_ERR("%s: failed to encode audio\n", __func__);
        return false;
    }
    embd.insert(embd.end(), res.begin(), res.end());
    mel.erase(mel.begin(), mel.begin() + (size_t) n * n_mel);
    return true;
}

// no padding here, step() pads a segment once it is complete
int32_t clip_audio_stream_gemma4a::flush(bool end) {
    const int64_t n_full = n_samples / seg_len;
    const int64_t n_rest = n_samples % seg_len;
    const int64_t rows_full = (n_seg_frames(seg_len, true) + frames_per_tok - 1) / frames_per_tok;
    const int64_t rows_rest = end ? (n_seg_frames(n_rest, true) + frames_per_tok - 1) / frames_per_tok
                                  : n_seg_frames(n_rest, false) / frames_per_tok;
    n_rows_flushed = n_full * rows_full + rows_rest;
    pending_flush = true;
    ended = end;
    return 0;
}

int32_t clip_audio_stream_gemma4a::step(llama_token /* last_token */) {
    const bool flush = pending_flush;
    pending_flush = false;

    // whole rows only, except at the end of a segment: a partial row puts its zero padding into the state
    while (true) {
        const int64_t seg_end  = seg_start + seg_len;
        const int64_t n_seg    = std::min(n_samples, seg_end) - seg_start;
        const bool    complete = n_samples >= seg_end || ended;
        if (!compute_mel(n_seg, complete)) {
            return -2;
        }
        const int n_avail = (int) (mel.size() / n_mel);
        int n = n_avail;
        if (!complete) {
            n -= n % frames_per_tok;
            if (!flush) {
                n -= n % n_step_mel;
            }
        }
        if (n > 0 && !encode(n)) {
            return -2;
        }
        if (n_samples < seg_end) {
            break;
        }
        // next segment starts from a zero state
        seg_start = seg_end;
        n_frames  = 0;
        ggml_backend_buffer_clear(enc_state->buf.get(), 0);
    }

    // rows carry no text, all of them go out at once and last_token is not used
    const int64_t n_avail = (int64_t) (embd.size() / n_embd);
    out.clear();
    if (n_avail > 0 || ended) {
        if (!beg_done) {
            beg_done = true;
            out = beg_rows;
        }
        out.insert(out.end(), embd.begin(), embd.end());
        embd.clear();
        if (ended && !end_done) {
            end_done = true;
            out.insert(out.end(), end_rows.begin(), end_rows.end());
        }
    }
    const int64_t n_out = (int64_t) (out.size() / n_embd);
    if (n_out == 0) {
        return ended ? -1 : 0;
    }
    n_rows      += n_out;
    n_rows_done += n_avail; // the markers are not audio rows
    return (int32_t) n_out;
}

// encodes one audio entry, the rows are appended to embd
static bool clip_audio_stream_encode(clip_audio_stream & s, clip_image_f32 && img, clip_subctx_enc_state * enc_state) {
    const int n_out = s.sctx->model.n_output_tokens(&img);

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(img));

    std::vector<float> res((size_t) s.n_embd * n_out);
    clip_subctx_encode_params params;
    params.n_threads = s.ctx->n_threads;
    params.imgs      = &batch;
    params.out_embd  = &res;
    params.enc_state = enc_state;
    if (!s.sctx->encode(&params)) {
        LOG_ERR("%s: failed to encode audio\n", __func__);
        return false;
    }
    s.embd.insert(s.embd.end(), res.begin(), res.end());
    return true;
}

// rows carry no text: all of them go out at once between the audio markers, last_token is not used
static int32_t clip_audio_stream_emit_rows(clip_audio_stream & s, const std::vector<float> & beg_rows, const std::vector<float> & end_rows,
                                           bool & beg_done, bool & end_done) {
    const int64_t n_avail = (int64_t) (s.embd.size() / s.n_embd);
    s.out.clear();
    if (n_avail > 0 || s.ended) {
        if (!beg_done) {
            beg_done = true;
            s.out = beg_rows;
        }
        s.out.insert(s.out.end(), s.embd.begin(), s.embd.end());
        s.embd.clear();
        if (s.ended && !end_done) {
            end_done = true;
            s.out.insert(s.out.end(), end_rows.begin(), end_rows.end());
        }
    }
    const int64_t n_out = (int64_t) (s.out.size() / s.n_embd);
    if (n_out == 0) {
        return s.ended ? -1 : 0;
    }
    s.n_rows      += n_out;
    s.n_rows_done += n_avail; // the markers are not audio rows
    return (int32_t) n_out;
}

//
// Higgs Audio
//

bool clip_audio_stream_higgs_audio::init(const llama_audio_stream_params & params) {
    const clip_hparams * hparams = &sctx->model.hparams;
    if (params.n_prompt > 0) {
        LOG_ERR("%s: Higgs Audio takes no prompt, its audio is a chat turn\n", __func__);
        return false;
    }
    if (!ctx->audio_preproc) {
        LOG_ERR("%s: the audio preprocessor is not initialized\n", __func__);
        return false;
    }
    chunk_len       = (int64_t) hparams->audio_chunk_len * hparams->audio_sample_rate;
    frames_per_tok  = 8; // stride-2 conv, avgpool, stride-2 projector conv
    samples_per_tok = frames_per_tok * hop;
    return clip_audio_stream_marker_rows(ctx, ctx->aud_beg, beg_rows) && clip_audio_stream_marker_rows(ctx, ctx->aud_end, end_rows);
}

// rows of a chunk with n samples, see clip_audio_preprocessor_higgs_audio
int64_t clip_audio_stream_higgs_audio::n_chunk_rows(int64_t n) const {
    const int64_t n_frames = n / hop;
    if (n_frames < 3) {
        return 0;
    }
    return ((n_frames + 1) / 2 / 2 + 1) / 2;
}

bool clip_audio_stream_higgs_audio::encode(int64_t n) {
    clip_audio_preproc_out pre = ctx->audio_preproc->preprocess(buf.data(), (size_t) n);
    if (pre.entries.empty() || !ctx->audio_preproc->compute(pre)) {
        LOG_ERR("%s: unable to preprocess the audio\n", __func__);
        return false;
    }
    for (auto & entry : pre.entries) {
        if (!clip_audio_stream_encode(*this, std::move(entry), nullptr)) {
            return false;
        }
    }
    buf.erase(buf.begin(), buf.begin() + n);
    buf_pos += n;
    return true;
}

int32_t clip_audio_stream_higgs_audio::flush(bool end) {
    const int64_t n_pending = n_samples - buf_pos;
    n_rows_flushed = n_rows_done + (int64_t) (embd.size() / n_embd) + (n_pending / chunk_len) * n_chunk_rows(chunk_len) + n_chunk_rows(n_pending % chunk_len);
    pending_flush = true;
    ended = end;
    return 0;
}

int32_t clip_audio_stream_higgs_audio::step(llama_token /* last_token */) {
    const bool flush = pending_flush;
    pending_flush = false;

    while ((int64_t) buf.size() >= chunk_len) {
        if (!encode(chunk_len)) {
            return -2;
        }
    }
    // a tail too short for a row is kept until the next flush, or dropped at the end
    if ((flush || ended) && !buf.empty()) {
        if (n_chunk_rows((int64_t) buf.size()) > 0) {
            if (!encode((int64_t) buf.size())) {
                return -2;
            }
        } else if (ended) {
            buf_pos += (int64_t) buf.size();
            buf.clear();
        }
    }
    return clip_audio_stream_emit_rows(*this, beg_rows, end_rows, beg_done, end_done);
}

//
// VibeVoice-ASR
//

bool clip_audio_stream_vibevoice_asr::init(const llama_audio_stream_params & params) {
    const clip_hparams * hparams = &sctx->model.hparams;
    if (params.n_prompt > 0) {
        LOG_ERR("%s: VibeVoice-ASR takes no prompt, its audio is a chat turn\n", __func__);
        return false;
    }
    hop             = hparams->audio_samples_per_token();
    frames_per_tok  = 1;
    samples_per_tok = hop;
    n_step     = std::max<int64_t>(1, ((int64_t) params.step_ms * hparams->audio_sample_rate / 1000 + hop - 1) / hop) * hop;
    n_step_max = std::max(n_step, (int64_t) hparams->audio_chunk_len * hparams->audio_sample_rate / hop * hop);

    enc_state = sctx->enc_state_init(0);
    if (!enc_state) {
        return false;
    }
    return clip_audio_stream_marker_rows(ctx, ctx->aud_beg, beg_rows) && clip_audio_stream_marker_rows(ctx, ctx->aud_end, end_rows);
}

// n is a multiple of hop, see clip_audio_preprocessor_vibevoice_asr for the gain
bool clip_audio_stream_vibevoice_asr::encode(int64_t n) {
    const double gain = clip_audio_vibevoice_gain(sum_sq, n_stat, peak);

    std::vector<float> data((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        data[i] = (float) (buf[i] * gain);
    }
    clip_image_f32 img;
    img.set_size({(int) n, 1}, false, /* is_audio */ true);
    img.cpy_buf(data);
    if (!clip_audio_stream_encode(*this, std::move(img), enc_state)) {
        return false;
    }
    buf.erase(buf.begin(), buf.begin() + n);
    buf_pos += n;
    return true;
}

int32_t clip_audio_stream_vibevoice_asr::flush(bool end) {
    n_rows_flushed = end ? (n_samples + hop - 1) / hop : n_samples / hop;
    pending_flush = true;
    ended = end;
    return 0;
}

int32_t clip_audio_stream_vibevoice_asr::step(llama_token /* last_token */) {
    const bool flush = pending_flush;
    pending_flush = false;

    // the padding added at the end does not count in the gain
    for (int64_t i = n_stat - buf_pos; i < (int64_t) buf.size(); i++) {
        sum_sq += (double) buf[i] * buf[i];
        peak    = std::max(peak, (double) std::fabs(buf[i]));
    }
    n_stat = n_samples;
    if (ended && n_samples % hop != 0) {
        add_silence(hop - n_samples % hop);
        n_stat = n_samples;
    }

    // whole rows only: a partial row would put its zero padding into the state
    int64_t n = (int64_t) buf.size();
    n -= n % hop;
    if (!flush && !ended) {
        n -= n % n_step;
    }
    while (n > 0) {
        const int64_t m = std::min(n, n_step_max);
        if (!encode(m)) {
            return -2;
        }
        n -= m;
    }
    return clip_audio_stream_emit_rows(*this, beg_rows, end_rows, beg_done, end_done);
}

int64_t clip_audio_stream_get_n_rows(const clip_audio_stream * s) {
    return s->n_rows_done;
}

int64_t clip_audio_stream_get_n_rows_flushed(const clip_audio_stream * s) {
    return s->n_rows_flushed;
}


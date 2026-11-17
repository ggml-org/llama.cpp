#include "llama-media.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-model.h"
#include "models/models.h"

#include "clip/clip-media.h"
#include "clip/clip-stream.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

static const llama_model_clip * as_clip(const llama_model * model) {
    if (model == nullptr || model->arch != LLM_ARCH_CLIP) {
        return nullptr;
    }
    return static_cast<const llama_model_clip *>(model);
}

bool llama_model_has_context_type(const struct llama_model * model, enum llama_context_type type) {
    if (const auto * clip = as_clip(model)) {
        if (type == LLAMA_CONTEXT_TYPE_ENCODER) {
            return clip->engine_vision() != nullptr || clip->engine_audio() != nullptr;
        }
        if (type == LLAMA_CONTEXT_TYPE_GEN_CODE) {
            return clip->native_gen;
        }
        if (type == LLAMA_CONTEXT_TYPE_GEN_WAV) {
            return clip->native_gen && clip->c2w.dac_entry_w != nullptr;
        }
        return false;
    }
    switch (type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return true;
        case LLAMA_CONTEXT_TYPE_MTP:     return model->hparams.n_layer_nextn > 0 && model->hparams.router_layer < 0;
        default:                         return false;
    }
}

//
// positions
//

void llama_media_row_pos(const llama_media & m, llama_pos pos_0, uint32_t row, llama_pos * pos, uint32_t n_pos_per_embd) {
    for (uint32_t j = 0; j < n_pos_per_embd; ++j) {
        pos[j] = 0;
    }
    // engine media: the positions of mtmd-helper, [t, y, x, z] for M-RoPE decoders
    if (m.img || m.aud) {
        if (m.img && n_pos_per_embd >= 3) {
            const clip_decoder_pos p = clip_image_tokens_get_decoder_pos(m.img.get(), pos_0, row);
            pos[0] = p.t;
            pos[1] = p.y;
            pos[2] = p.x;
            if (n_pos_per_embd > 3) {
                pos[3] = p.z;
            }
        } else {
            for (uint32_t j = 0; j < n_pos_per_embd; ++j) {
                pos[j] = pos_0 + (llama_pos) row;
            }
        }
        return;
    }
    switch (m.pos_type) {
        case LLAMA_MEDIA_POS_NORMAL:
            {
                for (uint32_t j = 0; j < n_pos_per_embd && j < 3; ++j) {
                    pos[j] = pos_0 + (llama_pos) row;
                }
            } break;
        case LLAMA_MEDIA_POS_MROPE:
            {
                GGML_ASSERT(n_pos_per_embd >= 3);
                const uint32_t gx = std::max<uint32_t>(m.grid_x, 1);
                pos[0] = pos_0;                      // t
                pos[1] = pos_0 + (llama_pos) (row / gx); // y
                pos[2] = pos_0 + (llama_pos) (row % gx); // x
            } break;
    }
}

llama_media_input llama_media_native_input(const llama_media & m) {
    llama_media_input inp;
    GGML_ASSERT(m.img && !m.img->batch_f32.entries.empty());
    const auto & entries = m.img->batch_f32.entries;
    inp.type     = LLAMA_MEDIA_TYPE_IMAGE;
    inp.nx       = entries[0].nx();
    inp.ny       = entries[0].ny();
    inp.n_frames = (int32_t) entries.size();
    for (const auto & e : entries) {
        GGML_ASSERT(e.nx() == inp.nx && e.ny() == inp.ny);
        const auto & buf = e.get_ro_buf();
        inp.data.insert(inp.data.end(), buf.begin(), buf.end());
    }
    return inp;
}

//
// proc
//

struct llama_media_proc_params llama_media_proc_default_params(void) {
    llama_media_proc_params params;
    params.n_span_max = 0;
    return params;
}

llama_media_proc::~llama_media_proc() = default;

struct llama_media_proc * llama_media_proc_init(
        struct llama_context     * ctx_enc,
        const struct llama_model * model_dec,
        struct llama_media_proc_params params) {
    if (ctx_enc == nullptr || model_dec == nullptr) {
        return nullptr;
    }
    const llama_model * model_enc = &ctx_enc->get_model();
    const auto * clip = as_clip(model_enc);
    if (clip == nullptr || ctx_enc->get_cparams().ctx_type != LLAMA_CONTEXT_TYPE_ENCODER) {
        LLAMA_LOG_ERROR("%s: ctx_enc must be an ENCODER context of an mmproj model\n", __func__);
        return nullptr;
    }

    auto proc = std::make_unique<llama_media_proc>();
    proc->ctx_enc    = ctx_enc;
    proc->model_enc  = model_enc;
    proc->model_dec  = model_dec;
    proc->vocab      = &model_dec->vocab;
    proc->params     = params;
    proc->n_rows_max = ctx_enc->n_ubatch();

    try {
        proc->cfg = std::make_unique<clip_layout_config>(clip->engine_vision(), clip->engine_audio(), model_dec,
                                                         nullptr, (int) ctx_enc->get_cparams().n_threads_batch);
        const int64_t n_embd = proc->cfg->n_embd_out();
        if (n_embd != (int64_t) model_dec->hparams.n_embd_inp()) {
            LLAMA_LOG_ERROR("%s: the mmproj outputs rows of %" PRId64 ", the decoder takes %u\n"
                            "hint: you may be using wrong mmproj\n", __func__, n_embd, model_dec->hparams.n_embd_inp());
            return nullptr;
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    }

    return proc.release();
}

void llama_media_proc_free(struct llama_media_proc * proc) {
    delete proc;
}

bool llama_media_proc_has(const struct llama_media_proc * proc, enum llama_media_type type) {
    switch (type) {
        case LLAMA_MEDIA_TYPE_IMAGE: return proc->cfg->v != nullptr;
        case LLAMA_MEDIA_TYPE_AUDIO: return proc->cfg->a != nullptr;
    }
    return false;
}

int32_t llama_media_proc_sample_rate(const struct llama_media_proc * proc) {
    return proc->cfg->a ? proc->cfg->a->hparams.audio_sample_rate : -1;
}

int32_t llama_media_proc_n_frame_merge(const struct llama_media_proc * proc) {
    return proc->cfg->v ? std::max(proc->cfg->v->n_temporal_merge(), 1) : 1;
}

//
// layout
//

llama_media_layout::~llama_media_layout() {
    for (auto & p : pieces) {
        if (p.media) {
            llama_media_free(p.media);
        }
    }
}

void llama_media_layout::add_tokens(const llama_token * tokens, size_t n) {
    if (n == 0) {
        return;
    }
    if (pieces.empty() || pieces.back().media != nullptr) {
        pieces.emplace_back();
    }
    auto & dst = pieces.back().tokens;
    dst.insert(dst.end(), tokens, tokens + n);
    pos      += (llama_pos) n;
    n_tokens += n;
}

void llama_media_layout::add_media(llama_media * m) {
    piece p;
    p.media = m;
    pieces.push_back(std::move(p));
    pos      += m->n_pos;
    n_tokens += m->n_tokens;
}

struct llama_media_layout * llama_media_layout_init(const struct llama_media_proc * proc, llama_pos pos_0) {
    auto * lay = new llama_media_layout;
    lay->proc = proc;
    lay->pos  = pos_0;
    return lay;
}

void llama_media_layout_free(struct llama_media_layout * lay) {
    delete lay;
}

int32_t llama_media_layout_add_tokens(struct llama_media_layout * lay, const llama_token * tokens, size_t n) {
    lay->add_tokens(tokens, n);
    return 0;
}

// lays out one media with the clip engine, appends the chunks it becomes as pieces
static int32_t llama_media_layout_add_bitmaps(struct llama_media_layout * lay, const std::vector<clip_bitmap> & bitmaps) {
    const llama_media_proc & proc = *lay->proc;
    const clip_layout_config & cfg = *proc.cfg;

    const bool is_audio = bitmaps[0].is_audio;
    if ((is_audio && cfg.a == nullptr) || (!is_audio && cfg.v == nullptr)) {
        return 1;
    }
    if (is_audio && cfg.audio_preproc) {
        // the log-mel of the preprocessor runs on the encoder context
        cfg.audio_preproc->mel_config.sched = proc.ctx_enc->sched_aux();
    }

    std::vector<clip_input_chunk> out;
    clip_tokenizer tk(cfg, out);
    tk.n_images_added  = lay->n_images_added;
    tk.n_tokens_before = lay->n_tokens;

    std::vector<const clip_bitmap *> ptrs;
    for (const auto & b : bitmaps) {
        ptrs.push_back(&b);
    }

    int32_t res = 0;
    try {
        res = tk.add_media(ptrs);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return 2;
    }
    if (res != 0) {
        return res;
    }
    lay->n_images_added = tk.n_images_added;

    for (auto & c : out) {
        if (c.type == CLIP_INPUT_CHUNK_TYPE_TEXT) {
            lay->add_tokens(c.tokens_text.data(), c.tokens_text.size());
            continue;
        }

        auto * m = new llama_media;
        m->model_enc     = proc.model_enc;
        m->placeholder   = c.is_placeholder();
        m->n_tokens      = (uint32_t) c.n_tokens_total();
        m->bidirectional = cfg.decode_use_non_causal(c.type == CLIP_INPUT_CHUNK_TYPE_AUDIO);
        if (c.type == CLIP_INPUT_CHUNK_TYPE_IMAGE) {
            m->inp.type = LLAMA_MEDIA_TYPE_IMAGE;
            m->id       = c.tokens_image->id;
            m->n_pos    = clip_image_tokens_get_n_pos(c.tokens_image.get());
            m->img      = std::move(c.tokens_image);
        } else {
            m->inp.type = LLAMA_MEDIA_TYPE_AUDIO;
            m->id       = c.tokens_audio->id;
            m->n_pos    = (llama_pos) c.tokens_audio->n_tokens;
            m->aud      = std::move(c.tokens_audio);
            m->aud_preproc = cfg.audio_preproc;
        }

        if (m->bidirectional && proc.params.n_span_max > 0 && m->n_tokens > (uint32_t) proc.params.n_span_max) {
            LLAMA_LOG_ERROR("%s: media has %u rows, the decoder takes %d in one ubatch (raise its n_ubatch)\n", __func__, m->n_tokens, proc.params.n_span_max);
            delete m;
            return 2;
        }
        lay->add_media(m);
    }

    return 0;
}

static clip_bitmap llama_media_bitmap_rgb(uint32_t nx, uint32_t ny, const uint8_t * rgb, const char * id) {
    clip_bitmap b;
    b.nx = nx;
    b.ny = ny;
    b.id = id ? id : "";
    if (rgb) {
        b.data.assign(rgb, rgb + (size_t) nx * ny * 3);
    }
    return b;
}

int32_t llama_media_layout_add_image(struct llama_media_layout * lay, uint32_t nx, uint32_t ny, const uint8_t * rgb, const char * id) {
    if (nx == 0 || ny == 0) {
        return 2;
    }
    return llama_media_layout_add_bitmaps(lay, { llama_media_bitmap_rgb(nx, ny, rgb, id) });
}

int32_t llama_media_layout_add_frames(struct llama_media_layout * lay, uint32_t nx, uint32_t ny,
        const uint8_t * const * rgb, int32_t n_frames, bool is_video, const char * id) {
    if (nx == 0 || ny == 0 || n_frames < 1 || n_frames > llama_media_proc_n_frame_merge(lay->proc)) {
        return 2;
    }
    std::vector<clip_bitmap> bitmaps;
    for (int32_t i = 0; i < n_frames; ++i) {
        bitmaps.push_back(llama_media_bitmap_rgb(nx, ny, rgb ? rgb[i] : nullptr, id));
        bitmaps.back().is_video = is_video;
    }
    return llama_media_layout_add_bitmaps(lay, bitmaps);
}

int32_t llama_media_layout_add_audio(struct llama_media_layout * lay, const float * pcm, size_t n_samples, const char * id) {
    if (n_samples == 0) {
        return 2;
    }
    clip_bitmap b;
    b.nx       = (uint32_t) n_samples;
    b.ny       = 1;
    b.id       = id ? id : "";
    b.is_audio = true;
    if (pcm) {
        b.data.assign((const uint8_t *) pcm, (const uint8_t *) (pcm + n_samples));
    }
    return llama_media_layout_add_bitmaps(lay, { std::move(b) });
}

size_t llama_media_layout_n_pieces(const struct llama_media_layout * lay) {
    return lay->pieces.size();
}

const llama_token * llama_media_layout_get_tokens(const struct llama_media_layout * lay, size_t i, size_t * n) {
    const auto & p = lay->pieces.at(i);
    if (p.media) {
        if (n) {
            *n = 0;
        }
        return nullptr;
    }
    if (n) {
        *n = p.tokens.size();
    }
    return p.tokens.data();
}

struct llama_media * llama_media_layout_get_media(const struct llama_media_layout * lay, size_t i) {
    const auto & p = lay->pieces.at(i);
    return p.media ? llama_media_ref(p.media) : nullptr;
}

//
// media handle
//

enum llama_media_type llama_media_get_type(const struct llama_media * m) { return m->inp.type; }
uint32_t   llama_media_n_tokens      (const struct llama_media * m) { return m->n_tokens; }
llama_pos  llama_media_n_pos         (const struct llama_media * m) { return m->n_pos; }
bool       llama_media_is_atomic     (const struct llama_media * m) { return m->bidirectional; }
const char * llama_media_get_id      (const struct llama_media * m) { return m->id.c_str(); }
bool       llama_media_is_placeholder(const struct llama_media * m) { return m->placeholder; }
bool       llama_media_is_encoded    (const struct llama_media * m) { return !m->rows.empty(); }

struct llama_media * llama_media_ref(struct llama_media * m) {
    m->n_ref.fetch_add(1);
    return m;
}

void llama_media_free(struct llama_media * m) {
    if (m && m->n_ref.fetch_sub(1) == 1) {
        delete m;
    }
}

void llama_media_release_rows(struct llama_media * m) {
    m->rows.clear();
    m->rows.shrink_to_fit();
}

int64_t llama_media_get_rows(const struct llama_media * m, int64_t row0, int64_t n_rows, float * out) {
    if (m->rows.empty()) {
        return 0;
    }
    if (row0 < 0 || n_rows < 0 || row0 + n_rows > (int64_t) m->n_tokens) {
        return 0;
    }
    if (out) {
        std::memcpy(out, m->rows.data() + row0 * m->n_embd, (size_t) n_rows * m->n_embd * sizeof(float));
    }
    return m->n_embd;
}

bool llama_media_get_row_pos(const struct llama_media * m, llama_pos pos_0, uint32_t row, llama_pos * pos, uint32_t n_pos) {
    if (row >= m->n_tokens || n_pos == 0 || n_pos > GGML_MROPE_SECTIONS) {
        return false;
    }
    if (m->pos_type == LLAMA_MEDIA_POS_MROPE && n_pos < 3) {
        return false;
    }
    llama_media_row_pos(*m, pos_0, row, pos, n_pos);
    return true;
}

//
// batch
//

int32_t llama_batch_ext_add_media(struct llama_batch_ext * batch, llama_seq_id seq_id,
        struct llama_media * media, uint32_t row0, uint32_t n_rows, llama_pos pos_0) {
    if (batch->ctx_type == LLAMA_CONTEXT_TYPE_ENCODER) {
        if (media->placeholder) {
            return -4;
        }
        if (media->model_enc != batch->model) {
            return -6;
        }
        batch->media.push_back(llama_media_ref(media));
        return (int32_t) batch->media.size() - 1;
    }

    if (media->rows.empty()) {
        return -4;
    }
    if (n_rows == 0) {
        n_rows = media->n_tokens - row0;
    }
    if (row0 + n_rows > media->n_tokens) {
        return -4;
    }
    if (media->bidirectional && (row0 != 0 || n_rows != media->n_tokens)) {
        return -5;
    }
    if ((size_t) media->n_embd != batch->n_embd_inp) {
        return -6;
    }
    if (batch->tokens.size() + n_rows > batch->n_tokens_max) {
        return -1;
    }

    const llama_pos span_lo = media->bidirectional ? pos_0 : -1;
    const llama_pos span_hi = media->bidirectional ? pos_0 + media->n_pos - 1 : -1;

    int32_t first = -1;
    for (uint32_t r = row0; r < row0 + n_rows; ++r) {
        const int32_t idx = batch->add_token(seq_id);
        if (idx < 0) {
            return idx;
        }
        if (first < 0) {
            first = idx;
        }

        llama_embd e;
        e.data   = media->rows.data() + (size_t) r * media->n_embd;
        e.n_rows = 1;
        e.n_embd = media->n_embd;
        if (!batch->set_token_embd(idx, e)) {
            return -6;
        }

        llama_pos pos[GGML_MROPE_SECTIONS];
        llama_media_row_pos(*media, pos_0, r, pos, (uint32_t) batch->n_pos_per_embd);
        batch->set_token_pos(idx, pos);

        batch->tokens[idx].span_lo = span_lo;
        batch->tokens[idx].span_hi = span_hi;
    }

    return first;
}

int32_t llama_batch_ext_add_codes(struct llama_batch_ext * batch, llama_seq_id seq_id, const llama_token * codes, int32_t n_codes) {
    if (n_codes <= 0 || codes == nullptr) {
        return -2;
    }
    const int32_t idx = batch->add_token(seq_id);
    if (idx < 0) {
        return idx;
    }
    batch->tokens[idx].codes_off = (int32_t) batch->codes.size();
    batch->tokens[idx].n_codes   = n_codes;
    batch->codes.insert(batch->codes.end(), codes, codes + n_codes);
    return idx;
}

//
// realtime audio input
//

struct llama_audio_stream {
    llama_media_proc * proc = nullptr;
    clip_stream_ctx    host;
    clip_audio_stream * s = nullptr;

    ~llama_audio_stream() {
        clip_audio_stream_free(s);
    }
};

static bool llama_audio_stream_host(const llama_media_proc * proc, clip_stream_ctx & host) {
    const auto * clip = as_clip(proc->model_enc);
    if (clip == nullptr || clip->engine_audio() == nullptr) {
        return false;
    }
    host.aud_beg       = proc->cfg->aud_beg;
    host.aud_end       = proc->cfg->aud_end;
    host.audio_preproc = proc->cfg->audio_preproc;
    host.ctx_a         = proc->ctx_enc->clip_sub(*clip->engine_audio());
    host.n_embd_text   = (int) proc->model_dec->hparams.n_embd_inp();
    host.n_threads     = (int) proc->ctx_enc->get_cparams().n_threads_batch;
    host.text_model    = proc->model_dec;
    host.vocab         = proc->vocab;
    return true;
}

bool llama_media_proc_has_audio_stream(const struct llama_media_proc * proc) {
    clip_stream_ctx host;
    return llama_audio_stream_host(proc, host) && clip_support_audio_stream(&host);
}

struct llama_audio_stream * llama_audio_stream_init(const struct llama_media_proc * proc, struct llama_audio_stream_params params) {
    auto st = std::make_unique<llama_audio_stream>();
    st->proc = const_cast<llama_media_proc *>(proc);
    if (!llama_audio_stream_host(proc, st->host)) {
        LLAMA_LOG_ERROR("%s: the model has no audio tower\n", __func__);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(st->proc->stream_mutex);
    if (st->host.audio_preproc) {
        st->host.audio_preproc->mel_config.sched = st->proc->ctx_enc->sched_aux();
    }
    try {
        st->s = clip_audio_stream_init(&st->host, params);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
    return st->s ? st.release() : nullptr;
}

void llama_audio_stream_free(struct llama_audio_stream * s) {
    delete s;
}

int32_t llama_audio_stream_push(struct llama_audio_stream * s, const float * pcm, size_t n) {
    return clip_audio_stream_push(s->s, pcm, n);
}

int32_t llama_audio_stream_flush(struct llama_audio_stream * s, bool end) {
    std::lock_guard<std::mutex> lock(s->proc->stream_mutex);
    if (s->host.audio_preproc) {
        s->host.audio_preproc->mel_config.sched = s->proc->ctx_enc->sched_aux();
    }
    try {
        return clip_audio_stream_flush(s->s, end);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -2;
    }
}

int32_t llama_audio_stream_step(struct llama_audio_stream * s, llama_token last_token) {
    std::lock_guard<std::mutex> lock(s->proc->stream_mutex);
    if (s->host.audio_preproc) {
        s->host.audio_preproc->mel_config.sched = s->proc->ctx_enc->sched_aux();
    }
    try {
        return clip_audio_stream_step(s->s, last_token);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -2;
    }
}

const float * llama_audio_stream_get_output_embd(const struct llama_audio_stream * s) { return clip_audio_stream_get_output_embd(s->s); }
bool    llama_audio_stream_is_seq_start      (const struct llama_audio_stream * s) { return clip_audio_stream_is_seq_start(s->s); }
int64_t llama_audio_stream_get_n_rows        (const struct llama_audio_stream * s) { return clip_audio_stream_get_n_rows(s->s); }
int64_t llama_audio_stream_get_n_rows_flushed(const struct llama_audio_stream * s) { return clip_audio_stream_get_n_rows_flushed(s->s); }

#include "llama-media.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-model.h"
#include "models/models.h"

#include "clip/clip-media.h"
#include "clip/clip-stream.h"
#include "clip/clip-gen.h"
#include "clip/clip-subcontext.h"

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
            return clip->native_gen || clip->engine_gen() != nullptr;
        }
        if (type == LLAMA_CONTEXT_TYPE_GEN_WAV) {
            return (clip->native_gen && clip->c2w.dac_entry_w != nullptr) || (!clip->native_gen && clip->engine_gen() != nullptr);
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
    if (ctx_enc == nullptr) {
        return nullptr;
    }
    const llama_model * model_enc = &ctx_enc->get_model();
    const auto * clip = as_clip(model_enc);
    if (clip == nullptr) {
        LLAMA_LOG_ERROR("%s: ctx_enc must be a context of an mmproj model\n", __func__);
        return nullptr;
    }

    auto proc = std::make_unique<llama_media_proc>();
    proc->ctx_enc    = ctx_enc;
    proc->model_enc  = model_enc;
    proc->model_dec  = model_dec;
    proc->vocab      = model_dec ? &model_dec->vocab : nullptr;
    proc->params     = params;

    try {
        proc->cfg = std::make_unique<clip_layout_config>(clip->engine_vision(), clip->engine_audio(), model_dec,
                                                         nullptr, (int) ctx_enc->get_cparams().n_threads_batch);
        if (proc->cfg->audio_preproc) {
            proc->cpu = std::make_shared<clip_cpu_compute>((int) ctx_enc->get_cparams().n_threads_batch);
            proc->cfg->audio_preproc->mel_config.sched = proc->cpu->sched;
        }
        const int64_t n_embd = proc->cfg->n_embd_out();
        if (model_dec && n_embd != (int64_t) model_dec->hparams.n_embd_inp()) {
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

int32_t llama_model_media_n_tokens_max(const struct llama_model * model, enum llama_media_type type, bool * atomic) {
    const auto * clip = as_clip(model);
    const bool is_audio = type == LLAMA_MEDIA_TYPE_AUDIO;
    const clip_model * m = clip == nullptr ? nullptr : is_audio ? clip->engine_audio() : clip->engine_vision();
    if (atomic) {
        *atomic = false;
    }
    if (m == nullptr) {
        return 0;
    }
    if (atomic) {
        // the layout config without a decoder only answers model-level questions
        const clip_layout_config cfg(clip->engine_vision(), clip->engine_audio(), nullptr, nullptr, 1);
        *atomic = cfg.decode_use_non_causal(is_audio);
    }
    return is_audio ? -1 : m->image_max_tokens();
}

bool llama_media_proc_video_frame_size(const struct llama_media_proc * proc, int32_t n_frames,
        int32_t src_w, int32_t src_h, int32_t * w, int32_t * h) {
    if (!proc->cfg->image_preproc) {
        return false;
    }
    clip_image_size size;
    if (!proc->cfg->image_preproc->get_video_frame_size(n_frames, src_w, src_h, size)) {
        return false;
    }
    *w = size.width;
    *h = size.height;
    return true;
}

void llama_media_proc_video_markers(const struct llama_media_proc * proc, const char ** beg, const char ** end) {
    *beg = proc->cfg->vid_beg.c_str();
    *end = proc->cfg->vid_end.c_str();
}

enum llama_media_video_template llama_media_proc_video_template(const struct llama_media_proc * proc) {
    const clip_layout_config & cfg = *proc->cfg;
    switch (cfg.proj_type_v()) {
        case PROJECTOR_TYPE_LFM3VL:
            return cfg.video_has_soundtrack() ? LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3_OMNI : LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3;
        case PROJECTOR_TYPE_QWEN3VL:
            return LLAMA_MEDIA_VIDEO_TEMPLATE_QWEN3VL;
        default:
            return LLAMA_MEDIA_VIDEO_TEMPLATE_DEFAULT;
    }
}

int32_t llama_media_proc_n_frame_merge(const struct llama_media_proc * proc) {
    return proc->cfg->v ? std::max(proc->cfg->v->n_temporal_merge(), 1) : 1;
}

//
// layout
//

// the pieces of a track and the rows assigned so far
struct llama_media_track {
    std::vector<clip_bitmap> pieces;
    clip_tokenizer::soundtrack st;
    size_t next = 0;
};

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

// runs one step of the clip tokenizer, appends the chunks it yields as pieces
template <typename F>
static int32_t llama_media_layout_run(struct llama_media_layout * lay, F && step) {
    const llama_media_proc & proc = *lay->proc;
    const clip_layout_config & cfg = *proc.cfg;

    std::vector<clip_input_chunk> out;
    clip_tokenizer tk(cfg, out);
    tk.n_images_added  = lay->n_images_added;
    tk.n_tokens_before = lay->n_tokens;

    int32_t res = 0;
    try {
        res = step(tk);
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
            m->aud_cpu     = proc.cpu;
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

// lays out one media with the clip engine
static int32_t llama_media_layout_add_bitmaps(struct llama_media_layout * lay, const std::vector<clip_bitmap> & bitmaps) {
    const clip_layout_config & cfg = *lay->proc->cfg;

    const bool is_audio = bitmaps[0].is_audio;
    if ((is_audio && cfg.a == nullptr) || (!is_audio && cfg.v == nullptr)) {
        return 1;
    }

    std::vector<const clip_bitmap *> ptrs;
    for (const auto & b : bitmaps) {
        ptrs.push_back(&b);
    }
    return llama_media_layout_run(lay, [&](clip_tokenizer & tk) { return tk.add_media(ptrs); });
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

int32_t llama_media_layout_begin_track(struct llama_media_layout * lay, const float * const * pcm, const size_t * n_samples, int32_t n_pieces) {
    if (lay->proc->cfg->a == nullptr) {
        return 1;
    }
    if (n_pieces < 1) {
        return 2;
    }
    auto track = std::make_unique<llama_media_track>();
    track->pieces.resize(n_pieces);
    for (int32_t i = 0; i < n_pieces; ++i) {
        clip_bitmap & b = track->pieces[i];
        b.nx       = (uint32_t) n_samples[i];
        b.ny       = 1;
        b.is_audio = true;
        if (pcm && pcm[i]) {
            b.data.assign((const uint8_t *) pcm[i], (const uint8_t *) (pcm[i] + n_samples[i]));
        }
    }
    for (const auto & b : track->pieces) {
        track->st.pieces.push_back(&b);
    }
    lay->track = std::move(track);
    return 0;
}

int32_t llama_media_layout_add_track_piece(struct llama_media_layout * lay, const char * id) {
    llama_media_track * track = lay->track.get();
    if (track == nullptr || track->next >= track->pieces.size()) {
        return 2;
    }
    clip_bitmap & b = track->pieces[track->next++];
    b.id = id ? id : "";
    const int32_t res = llama_media_layout_run(lay, [&](clip_tokenizer & tk) { return tk.add_soundtrack_piece(track->st, b); });
    if (track->next == track->pieces.size()) {
        lay->track.reset();
    }
    return res;
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

const float * llama_media_rows_host(const struct llama_media * m) {
    return m->rows.empty() ? nullptr : m->rows.data();
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
    // the contexts of an mmproj encode (ENCODER, and GEN for the references of its pipeline)
    if (batch->model != nullptr && batch->model->arch == LLM_ARCH_CLIP) {
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
    try {
        return clip_audio_stream_flush(s->s, end);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -2;
    }
}

int32_t llama_audio_stream_step(struct llama_audio_stream * s, llama_token last_token) {
    std::lock_guard<std::mutex> lock(s->proc->stream_mutex);
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

//
// gen pipelines of the clip engine, model-level queries
//

// a host over a throw-away runtime of the gen tower, for the queries that run no graph
struct llama_gen_query {
    std::unique_ptr<clip_subcontext> sub_gen;
    std::unique_ptr<clip_subcontext> sub_a;
    clip_gen_ctx host;

    explicit llama_gen_query(const llama_model * model) {
        const auto * clip = as_clip(model);
        if (clip == nullptr || clip->engine_gen() == nullptr) {
            return;
        }
        sub_gen = std::make_unique<clip_subcontext>(*clip->engine_gen(), false);
        host.ctx_gen_a = sub_gen.get();
        if (clip->engine_audio()) {
            sub_a = std::make_unique<clip_subcontext>(*clip->engine_audio(), false);
            host.ctx_a = sub_a.get();
        }
    }

    bool ok() const { return host.ctx_gen_a != nullptr; }
};

struct llama_gen_info llama_model_gen_info(const struct llama_model * model) {
    llama_gen_query q(model);
    if (!q.ok()) {
        llama_gen_info info {};
        info.type = LLAMA_GEN_AUDIO_TYPE_NONE;
        return info;
    }
    return clip_gen_audio_get_info(&q.host);
}

struct llama_gen_inp llama_gen_inp_default(const struct llama_model * model) {
    llama_gen_query q(model);
    if (!q.ok()) {
        llama_gen_inp inp {};
        inp.type = LLAMA_GEN_STAGE_CODE;
        return inp;
    }
    return clip_gen_inp_default(&q.host);
}

size_t llama_model_gen_read_tensor(const struct llama_model * model, const char * name, float * out, size_t n_max) {
    llama_gen_query q(model);
    return q.ok() ? clip_gen_audio_read_tensor(&q.host, name, out, n_max) : 0;
}

const char * llama_model_gen_voice_name(const struct llama_model * model, int32_t idx) {
    llama_gen_query q(model);
    return q.ok() ? clip_gen_audio_voice_name(&q.host, idx) : nullptr;
}

struct llama_media * llama_media_init_voice(const struct llama_model * model, const char * name) {
    const auto * clip = as_clip(model);
    if (clip == nullptr || clip->engine_gen() == nullptr || name == nullptr) {
        return nullptr;
    }
    const clip_model & g = *clip->engine_gen();
    // matched case-insensitively, as the lfm3a "tts" chat template lowers it
    const auto & voices = g.gen_voices;
    const auto it = std::find_if(voices.begin(), voices.end(), [name](const auto & voice) {
        return std::equal(voice.first.begin(), voice.first.end(), name, name + strlen(name),
                          [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); });
    });
    if (it == voices.end()) {
        return nullptr;
    }
    // a voice without rows (lfm3a) is only a name, the chat template puts it in the prompt
    const int64_t n_embd = g.n_mmproj_embd();
    if (n_embd <= 0 || it->second.rows.size() % (size_t) n_embd != 0) {
        return nullptr;
    }
    auto * m = new llama_media;
    m->id        = it->first;
    m->model_enc = model;
    m->inp.type  = LLAMA_MEDIA_TYPE_AUDIO;
    m->n_tokens  = (uint32_t) (it->second.rows.size() / n_embd);
    m->n_pos     = (llama_pos) m->n_tokens;
    m->n_embd    = n_embd;
    m->rows      = it->second.rows;
    const auto & ref = it->second.ref;
    if (!ref.tokens.empty() || !ref.feat.empty() || !ref.spk.empty()) {
        m->gen_ref = std::make_shared<const clip_gen_ref>(ref);
    }
    return m;
}

int32_t llama_batch_ext_add_gen(struct llama_batch_ext * batch, llama_seq_id seq_id, const struct llama_gen_inp * inp) {
    if (batch->ctx_type != LLAMA_CONTEXT_TYPE_GEN_CODE && batch->ctx_type != LLAMA_CONTEXT_TYPE_GEN_WAV) {
        return -6;
    }
    if (seq_id < 0 || seq_id >= batch->n_seq_max) {
        return -3;
    }
    if (batch->gen.size() >= batch->n_tokens_max) {
        return -1;
    }
    llama_batch_ext::gen_entry e;
    e.seq_id = seq_id;
    e.inp    = *inp;
    if (inp->codes) {
        e.codes.assign(inp->codes, inp->codes + inp->n_codes);
    }
    if (inp->feats) {
        e.feats.assign(inp->feats, inp->feats + inp->n_feats);
    }
    if (inp->embd && batch->n_embd_state > 0) {
        e.embd.assign(inp->embd, inp->embd + batch->n_embd_state);
    }
    if (inp->backbone_state) {
        e.backbone_state.assign(inp->backbone_state, inp->backbone_state + inp->backbone_state_size);
    }
    batch->gen.push_back(std::move(e));
    auto & b = batch->gen.back();
    b.inp.codes          = inp->codes          ? b.codes.data()          : nullptr;
    b.inp.feats          = inp->feats          ? b.feats.data()          : nullptr;
    b.inp.embd           = inp->embd && !b.embd.empty() ? b.embd.data()  : nullptr;
    b.inp.backbone_state = inp->backbone_state ? b.backbone_state.data() : nullptr;
    return (int32_t) batch->gen.size() - 1;
}

//
// debugging of model ports
//

int32_t llama_media_debug_encode(const struct llama_media_proc * proc, enum llama_media_type type,
        int32_t nx, int32_t ny, const float * data, size_t n_data) {
    const bool is_audio = type == LLAMA_MEDIA_TYPE_AUDIO;
    const clip_model * m = is_audio ? proc->cfg->a : proc->cfg->v;
    if (m == nullptr) {
        LLAMA_LOG_ERROR("%s: the model has no tower for this media\n", __func__);
        return 1;
    }
    clip_image_f32 inp;
    if (is_audio) {
        const int n_mel = m->hparams.n_mel_bins;
        std::vector<float> buf((size_t) nx * n_mel);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < n_mel; j++) {
                buf[(size_t) j * nx + i] = data[i];
            }
        }
        inp.set_size({nx, n_mel}, false, true);
        inp.cpy_buf(buf);
    } else {
        inp.set_size({nx, ny}, false, false);
        inp.cpy_buf(std::vector<float>(data, data + n_data));
    }
    LLAMA_LOG_INFO("%s: created input with nx=%d, ny=%d\n", __func__, inp.nx(), inp.ny());

    clip_subcontext * sub = proc->ctx_enc->clip_sub(*m);
    std::vector<float> out((size_t) m->n_output_tokens(&inp) * m->n_mmproj_embd());
    sub->debug_output_embeddings = true;
    const bool ok = sub->encode_image((int) proc->ctx_enc->get_cparams().n_threads_batch, &inp, out);
    sub->debug_output_embeddings = false;
    if (!ok) {
        LLAMA_LOG_ERROR("%s: failed to encode\n", __func__);
        return 1;
    }
    return 0;
}

int32_t llama_media_debug_preprocess(const struct llama_media_proc * proc, enum llama_media_type type,
        int32_t nx, int32_t ny, const void * data) {
    const auto & cfg = *proc->cfg;
    if (type == LLAMA_MEDIA_TYPE_IMAGE) {
        if (!cfg.image_preproc) {
            LLAMA_LOG_ERROR("%s: the model has no vision tower\n", __func__);
            return 1;
        }
        clip_image_u8 img_u8;
        img_u8.set_size({nx, ny}, false);
        img_u8.cpy_buf(std::vector<uint8_t>((const uint8_t *) data, (const uint8_t *) data + (size_t) nx * ny * 3));
        clip_image_preproc_out pre = cfg.image_preproc->preprocess(img_u8);
        LLAMA_LOG_INFO("%s: preprocessed image to %d entries\n", __func__, (int) pre.entries.size());
        for (size_t i = 0; i < pre.entries.size(); i++) {
            LLAMA_LOG_INFO("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, pre.entries[i].nx(), pre.entries[i].ny());
        }
        return 0;
    }
    if (!cfg.audio_preproc) {
        LLAMA_LOG_ERROR("%s: the model has no audio tower\n", __func__);
        return 1;
    }
    std::lock_guard<std::mutex> lock(const_cast<llama_media_proc *>(proc)->stream_mutex);
    clip_audio_preproc_out out = cfg.audio_preproc->preprocess((const float *) data, (size_t) nx);
    if (out.entries.empty() || !cfg.audio_preproc->compute(out)) {
        LLAMA_LOG_ERROR("%s: failed to preprocess audio\n", __func__);
        return 1;
    }
    LLAMA_LOG_INFO("%s: preprocessed audio to %zu mel spec chunks\n", __func__, out.entries.size());
    for (size_t i = 0; i < out.entries.size(); i++) {
        const auto & mel  = out.entries[i];
        const auto & mdat = mel.get_ro_buf();
        LLAMA_LOG_INFO("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel.nx(), mel.ny());
        for (int mi = 0; mi < mel.ny(); mi++) {
            for (int ti = 0; ti < mel.nx(); ti++) {
                LLAMA_LOG_INFO("mel[%zu][m=%d][t=%d] = %f\n", i, mi, ti, mdat[(size_t) mi * mel.nx() + ti]);
            }
        }
    }
    return 0;
}

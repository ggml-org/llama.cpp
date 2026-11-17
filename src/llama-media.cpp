#include "llama-media.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-media-image.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <cstring>

static const llama_model_clip * as_clip(const llama_model * model) {
    if (model == nullptr || model->arch != LLM_ARCH_CLIP) {
        return nullptr;
    }
    return static_cast<const llama_model_clip *>(model);
}

bool llama_model_has_context_type(const struct llama_model * model, enum llama_context_type type) {
    if (const auto * clip = as_clip(model)) {
        if (type == LLAMA_CONTEXT_TYPE_ENCODER) {
            return clip->vhp.has_vision && clip->vhp.proj != LLAMA_CLIP_PROJ_UNKNOWN;
        }
        if (type == LLAMA_CONTEXT_TYPE_GEN_CODE) {
            return !clip->vhp.has_vision && clip->ghp.has_gen && clip->ghp.proj == LLAMA_CLIP_PROJ_QWEN3TTS_GEN;
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

//
// proc
//

struct llama_media_proc_params llama_media_proc_default_params(void) {
    llama_media_proc_params params;
    params.image_min_tokens = 0;
    params.image_max_tokens = 0;
    params.n_span_max       = 0;
    return params;
}

struct llama_media_proc * llama_media_proc_init(
        const struct llama_context * ctx_enc,
        const struct llama_model   * model_dec,
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
    if (clip->vhp.n_embd_out != (int32_t) model_dec->hparams.n_embd_inp()) {
        LLAMA_LOG_ERROR("%s: the mmproj outputs rows of %d, the decoder takes %u\n", __func__,
                clip->vhp.n_embd_out, model_dec->hparams.n_embd_inp());
        return nullptr;
    }

    auto * proc = new llama_media_proc;
    proc->model_enc  = model_enc;
    proc->model_dec  = model_dec;
    proc->vocab      = &model_dec->vocab;
    proc->params     = params;
    proc->n_rows_max = ctx_enc->n_ubatch();
    return proc;
}

void llama_media_proc_free(struct llama_media_proc * proc) {
    delete proc;
}

bool llama_media_proc_has(const struct llama_media_proc * proc, enum llama_media_type type) {
    const auto * clip = as_clip(proc->model_enc);
    switch (type) {
        case LLAMA_MEDIA_TYPE_IMAGE: return clip && clip->vhp.has_vision;
        case LLAMA_MEDIA_TYPE_AUDIO: return false;
    }
    return false;
}

int32_t llama_media_proc_sample_rate(const struct llama_media_proc * proc) {
    GGML_UNUSED(proc);
    return -1;
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
    pos += (llama_pos) n;
}

void llama_media_layout::add_media(llama_media * m) {
    piece p;
    p.media = m;
    pieces.push_back(std::move(p));
    pos += m->n_pos;
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

// fills the decoder-side fields of a preprocessed media
static int32_t llama_media_finish(const llama_media_proc & proc, llama_media & m) {
    const auto rope = llama_model_rope_type(proc.model_dec);
    const bool mrope = rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE;

    m.pos_type      = mrope ? LLAMA_MEDIA_POS_MROPE : LLAMA_MEDIA_POS_NORMAL;
    m.n_pos         = mrope ? (llama_pos) std::max(m.grid_x, m.grid_y) : (llama_pos) m.n_tokens;
    m.bidirectional = proc.model_dec->media_bidirectional(m.inp.type);

    if (m.n_tokens > proc.n_rows_max) {
        LLAMA_LOG_ERROR("%s: media has %u rows, the encoder context takes %u (raise its n_ubatch)\n", __func__, m.n_tokens, proc.n_rows_max);
        return 2;
    }
    if (m.bidirectional && proc.params.n_span_max > 0 && m.n_tokens > (uint32_t) proc.params.n_span_max) {
        LLAMA_LOG_ERROR("%s: media has %u rows, the decoder takes %d in one ubatch (raise its n_ubatch)\n", __func__, m.n_tokens, proc.params.n_span_max);
        return 2;
    }
    return 0;
}

static int32_t llama_media_layout_add_frames_impl(struct llama_media_layout * lay, uint32_t nx, uint32_t ny,
        const uint8_t * const * rgb, int n_frames, const char * id) {
    const llama_media_proc & proc = *lay->proc;
    const auto * clip = as_clip(proc.model_enc);
    if (clip == nullptr || !clip->vhp.has_vision) {
        return 1;
    }
    if (nx == 0 || ny == 0) {
        return 2;
    }

    const bool placeholder = rgb[0] == nullptr;

    std::vector<llama_media_img::image_u8> frames(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        frames[f].set_size({ (int) nx, (int) ny }, placeholder);
        if (!placeholder) {
            const uint8_t * src = rgb[f] ? rgb[f] : rgb[0];
            frames[f].buf.assign(src, src + (size_t) nx * ny * 3);
        }
    }

    auto * m = new llama_media;
    m->id          = id ? id : "";
    m->model_enc   = proc.model_enc;
    m->placeholder = placeholder;

    if (!clip->media_preprocess_image(proc, frames.data(), n_frames, *m)) {
        delete m;
        return 1;
    }

    if (const int32_t res = llama_media_finish(proc, *m)) {
        delete m;
        return res;
    }

    std::vector<llama_token> beg;
    std::vector<llama_token> end;
    clip->media_image_wrap(proc, beg, end);

    lay->add_tokens(beg.data(), beg.size());
    lay->add_media(m);
    lay->add_tokens(end.data(), end.size());

    return 0;
}

int32_t llama_media_layout_add_image(struct llama_media_layout * lay, uint32_t nx, uint32_t ny, const uint8_t * rgb, const char * id) {
    const uint8_t * frames[1] = { rgb };
    return llama_media_layout_add_frames_impl(lay, nx, ny, frames, 1, id);
}

int32_t llama_media_layout_add_frames(struct llama_media_layout * lay, uint32_t nx, uint32_t ny,
        const uint8_t * rgb0, const uint8_t * rgb1, const char * id) {
    const auto * clip = as_clip(lay->proc->model_enc);
    if (clip == nullptr || clip->vhp.n_temporal_merge < 2) {
        return 1;
    }
    const uint8_t * frames[2] = { rgb0, rgb1 };
    return llama_media_layout_add_frames_impl(lay, nx, ny, frames, 2, id);
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

#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-internal.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "debug/mtmd-debug.h"

#include "llama.h"
#include "../src/llama-ext.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <type_traits>
#include <vector>

// remember to bump this if the serialization format changes
#define MTMD_SERIALIZATION_VERSION 3

// oldest compat version that can be loaded
#define MTMD_SERIALIZATION_VERSION_MIN 2

struct mtmd_serialization {
    // note: using 64-bit here for future-proofing
    uint64_t version = MTMD_SERIALIZATION_VERSION;
    std::vector<char> data;
    size_t read_pos = 0; // cursor used when reading

    // for writing
    mtmd_serialization(uint64_t version) : version(version) {
        write(version);
    }

    // for reading
    mtmd_serialization(uint64_t version, const char * buf, size_t len) {
        // copy buf to data
        data.assign(buf, buf + len);
        uint64_t ver_in = read<uint64_t>();
        if (ver_in < MTMD_SERIALIZATION_VERSION_MIN || ver_in > version) {
            throw std::runtime_error("version mismatch");
        }
        this->version = ver_in;
    }

    template <typename T>
    void write(T value) {
        static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
            "T must be trivially copyable and not bool");
        const char * p = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), p, p + sizeof(T));
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
            "T must be trivially copyable and not bool");
        if (read_pos + sizeof(T) > data.size()) {
            throw std::runtime_error("read OOB");
        }
        T value;
        std::memcpy(&value, data.data() + read_pos, sizeof(T));
        read_pos += sizeof(T);
        return value;
    }

};

template <>
void mtmd_serialization::write<bool>(bool value) {
    write<uint8_t>(value ? 1 : 0);
}
template <>
bool mtmd_serialization::read<bool>() {
    return read<uint8_t>() != 0;
}

template <>
void mtmd_serialization::write<std::string>(std::string value) {
    write<uint64_t>(value.size());
    data.insert(data.end(), value.begin(), value.end());
}
template <>
std::string mtmd_serialization::read<std::string>() {
    uint64_t len = read<uint64_t>();
    if (read_pos + len > data.size()) {
        throw std::runtime_error("read_string OOB");
    }
    std::string str(data.data() + read_pos, len);
    read_pos += len;
    return str;
}

// only mtmd.cpp needs these, so they're implemented here rather than in clip-impl.h
void clip_image_f32::serialize(mtmd_serialization & ser) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    // note: buf is intentionally NOT serialized; the loaded clip_image_f32 will always be a placeholder
    ser.write(add_viewsep);
    ser.write(add_newline);
    ser.write(lead_pad);
    ser.write((int32_t)suffix_type);
    ser.write((int32_t)anyres.grid_x);
    ser.write((int32_t)anyres.grid_y);
    ser.write((int32_t)anyres.orig_nx);
    ser.write((int32_t)anyres.orig_ny);
    ser.write((int32_t)nx_);
    ser.write((int32_t)ny_);
}
void clip_image_f32::deserialize(mtmd_serialization & ser) {
    add_viewsep = ser.read<bool>();
    add_newline = ser.read<bool>();
    lead_pad = ser.read<int32_t>();
    if (ser.version >= 3) {
        const int32_t suffix_raw = ser.read<int32_t>();
        if (suffix_raw < 0 || suffix_raw >= CLIP_SUFFIX_COUNT) {
            throw std::runtime_error("invalid suffix type");
        }
        suffix_type = (clip_suffix_type)suffix_raw;
        anyres.grid_x = ser.read<int32_t>();
        anyres.grid_y = ser.read<int32_t>();
        anyres.orig_nx = ser.read<int32_t>();
        anyres.orig_ny = ser.read<int32_t>();
    }
    nx_ = ser.read<int32_t>();
    ny_ = ser.read<int32_t>();
    buf.clear(); // always a placeholder after loading
}

void clip_image_f32_batch::serialize(mtmd_serialization & ser) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    ser.write(is_audio);
    ser.write<uint64_t>(entries.size());
    for (const auto & entry : entries) {
        entry.serialize(ser);
    }
}
void clip_image_f32_batch::deserialize(mtmd_serialization & ser) {
    is_audio = ser.read<bool>();
    uint64_t n = ser.read<uint64_t>();
    constexpr size_t min_entry_bytes = sizeof(uint8_t) * 2 + sizeof(int32_t) * 2;
    if (n > (ser.data.size() - ser.read_pos) / min_entry_bytes) {
        throw std::runtime_error("entries count exceeds buffer size");
    }
    entries.clear();
    entries.reserve(n);
    for (uint64_t i = 0; i < n; i++) {
        clip_image_f32 entry;
        entry.deserialize(ser);
        entries.push_back(std::move(entry));
    }
}

// for still image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3 bytes
//
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
// length of data must be nx * sizeof(float) bytes
struct mtmd_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 0;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio = false; // true if the bitmap is audio
    bool mergeable = false; // [QWEN_VIDEO] set only on frames of the same video

    // lazy-loaded bitmap
    mtmd_bitmap_lazy_callback lazy_callback = nullptr;
    void * lazy_user_data = nullptr;

    mtmd_bitmap(const unsigned char * data, uint32_t nx, uint32_t ny)
        : nx(nx), ny(ny), is_audio(false) {
        if (data) {
            size_t data_size = (size_t)nx * ny * 3;
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    mtmd_bitmap(const unsigned char * data, uint32_t n_samples)
        : nx(n_samples), ny(1), is_audio(true) {
        if (data) {
            size_t data_size = (size_t)nx * sizeof(float);
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    const std::vector<unsigned char> & get_ro_buf() const {
        return data;
    }

    bool is_placeholder() const {
        return data.empty();
    }

    size_t n_bytes() const {
        return data.size();
    }

    bool can_merge_with(const mtmd_bitmap & other) const {
        // [QWEN_VIDEO] can (temporal) merge if both are images with same size
        return mergeable && other.mergeable
            && !is_audio && !other.is_audio
            && nx == other.nx && ny == other.ny;
    }

  private:
    std::vector<unsigned char> data;
};

// position indexing for decoder model
enum mtmd_pos_type {
    MTMD_POS_TYPE_NORMAL,    // number of positions equals to number of tokens
    MTMD_POS_TYPE_MROPE,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    MTMD_POS_TYPE_HUNYUANVL, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
    MTMD_POS_TYPE_SENSENOVA_U1, // U1 uses zero-based H/W and one temporal position per image
    MTMD_POS_TYPE_CANVAS,    // MiniCPM-V 4.7: overview + slices in one chunk, sharing one 2D canvas (see mtmd_image_tokens::canvas_tile_grid)
    MTMD_POS_TYPE_COUNT,     // for validation
};


struct mtmd_image_tokens {
    uint32_t nx = 0; // number of tokens in x direction
    uint32_t ny = 0; // number of tokens in y direction
    mtmd_pos_type pos = MTMD_POS_TYPE_NORMAL;
    uint32_t image_idx = 0; // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
    uint32_t n_temporal_merge = 1; // for qwen-vl style temporal merge
    uint32_t n_tokens() const {
        if (pos == MTMD_POS_TYPE_HUNYUANVL) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (nx + 1) * ny + 2;
        }
        if (pos == MTMD_POS_TYPE_CANVAS) {
            uint32_t n = 0;
            for (size_t k = 0; k < batch_f32.entries.size(); ++k) {
                const auto [gw, gh] = canvas_tile_grid(k);
                n += gw * gh + (uint32_t) clip_suffix_rows(batch_f32.entries[k].suffix_type).size();
            }
            return n;
        }
        uint32_t nz = batch_f32.entries.size();
        if (n_temporal_merge > 1) {
            // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the future
            // TODO: simplify this by repeating the last frame until it fits the temporal merge
            if (nz % n_temporal_merge != 0) {
                nz = nz / n_temporal_merge + 1;
            } else {
                nz = nz / n_temporal_merge;
            }
        }
        return nx * ny * nz;
    }
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        for (const auto & entry : batch_f32.entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    // MTMD_POS_TYPE_CANVAS: entries are [overview, slices row by row], nx/ny is the token grid of the last entry
    // returns the token grid (w, h) of entry k, scaled from its pixel size
    std::pair<uint32_t, uint32_t> canvas_tile_grid(size_t k) const {
        const auto & ref = batch_f32.entries.back();
        const auto & e   = batch_f32.entries[k];
        return { (uint32_t) e.nx() * nx / ref.nx(), (uint32_t) e.ny() * ny / ref.ny() };
    }

    bool can_batch_with(const mtmd_image_tokens & other) {
        // a canvas chunk holds a whole image group, its layout is not given by nx/ny alone
        return nx == other.nx && ny == other.ny && pos == other.pos && pos != MTMD_POS_TYPE_CANVAS;
    }

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            pos,
            image_idx,
            n_temporal_merge,
            batch_f32.clone(),
            id
        };
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write(nx);
        ser.write(ny);
        ser.write((uint32_t)pos);
        ser.write(image_idx);
        ser.write(n_temporal_merge);
        ser.write(id);
        batch_f32.serialize(ser);
    }
    void deserialize(mtmd_serialization & ser) {
        nx = ser.read<uint32_t>();
        ny = ser.read<uint32_t>();
        uint32_t pos_raw = ser.read<uint32_t>();
        if (pos_raw >= MTMD_POS_TYPE_COUNT) {
            throw std::runtime_error("invalid pos type");
        }
        pos = (mtmd_pos_type)pos_raw;
        image_idx = ser.read<uint32_t>();
        n_temporal_merge = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
    uint32_t n_tokens = 0; // number of tokens
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        for (const auto & entry : batch_f32.entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    mtmd_audio_tokens clone() {
        return mtmd_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id
        };
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write(n_tokens);
        ser.write(id);
        batch_f32.serialize(ser);
    }
    void deserialize(mtmd_serialization & ser) {
        n_tokens = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;

    bool can_batch_with(const mtmd_input_chunk & other) const {
        if (type != other.type) {
            return false;
        }

        if (tokens_image && other.tokens_image) {
            return tokens_image->can_batch_with(*other.tokens_image);
        }

        // TODO: allow batching audio chunks of the same size

        return false;
    }

    bool is_placeholder() const {
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return tokens_image && tokens_image->is_placeholder();
        } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return tokens_audio && tokens_audio->is_placeholder();
        }
        return false;
    }

    void serialize(mtmd_serialization & ser) const {
        // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
        ser.write((uint32_t)type);

        ser.write<uint64_t>(tokens_text.size());
        for (llama_token tok : tokens_text) {
            ser.write((int32_t)tok);
        }

        ser.write(tokens_image != nullptr);
        if (tokens_image) {
            tokens_image->serialize(ser);
        }

        ser.write(tokens_audio != nullptr);
        if (tokens_audio) {
            tokens_audio->serialize(ser);
        }
    }
    void deserialize(mtmd_serialization & ser) {
        uint32_t type_raw = ser.read<uint32_t>();
        if (type_raw >= MTMD_INPUT_CHUNK_TYPE_COUNT) {
            throw std::runtime_error("invalid chunk type");
        }
        type = (mtmd_input_chunk_type)type_raw;

        uint64_t n_tokens_text = ser.read<uint64_t>();
        // reject before resize() so a tiny corrupted/malicious buffer can't force a huge allocation
        if (n_tokens_text > (ser.data.size() - ser.read_pos) / sizeof(int32_t)) {
            throw std::runtime_error("tokens_text length exceeds buffer size");
        }
        tokens_text.resize(n_tokens_text);
        for (uint64_t i = 0; i < n_tokens_text; i++) {
            tokens_text[i] = (llama_token)ser.read<int32_t>();
        }

        if (ser.read<bool>()) {
            tokens_image = std::make_unique<mtmd_image_tokens>();
            tokens_image->deserialize(ser);
        } else {
            tokens_image.reset();
        }

        if (ser.read<bool>()) {
            tokens_audio = std::make_unique<mtmd_audio_tokens>();
            tokens_audio->deserialize(ser);
        } else {
            tokens_audio.reset();
        }

        // catch buffers where the declared type doesn't match which payload is actually present,
        // so a mismatched chunk can't slip through and null-deref/abort later in an accessor
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE && !tokens_image) {
            throw std::runtime_error("type is IMAGE but tokens_image is missing");
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO && !tokens_audio) {
            throw std::runtime_error("type is AUDIO but tokens_audio is missing");
        }
    }
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

struct mtmd_batch {
    mtmd_context * ctx;
    std::vector<const mtmd_input_chunk *> entries;
    std::vector<float> output_embd; // aggregated output embedding for the whole batch
    mtmd_batch(mtmd_context * ctx): ctx(ctx) {}
    int32_t n_tokens() const {
        int32_t n = 0;
        for (const auto * chunk : entries) {
            n += mtmd_input_chunk_get_n_tokens(chunk);
        }
        return n;
    }
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
    MTMD_SLICE_TMPL_IDEFICS3,
    MTMD_SLICE_TMPL_LFM2,
    MTMD_SLICE_TMPL_STEP3VL,
};

const char * mtmd_default_marker() {
    return "<__media__>";
}

static clip_flash_attn_type mtmd_get_clip_flash_attn_type(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return CLIP_FLASH_ATTN_TYPE_AUTO;
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return CLIP_FLASH_ATTN_TYPE_DISABLED;
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return CLIP_FLASH_ATTN_TYPE_ENABLED;
    }
    return CLIP_FLASH_ATTN_TYPE_AUTO;
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* device            */ nullptr,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* batch_max_tokens  */ 1024,
        /* progress_callback */ nullptr,
        /* progress_callback_user_data */ nullptr,
    };
    return params;
}

struct mtmd_context {
    struct clip_ctx * ctx_v; // vision
    struct clip_ctx * ctx_a; // audio
    std::vector<float> out_embd; // image embedding vector

    // generation context
    struct clip_ctx * ctx_gen_a; // audio

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text = -1; // -1 means llm context not provided, skip checking this
    const llama_vocab * vocab = nullptr; // can be nullptr if text_model is not provided
    const llama_model * text_model = nullptr; // for the token embeddings of mtmd_audio_stream
    mtmd_pos_type pos_type;

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    mtmd_slice_tmpl slice_tmpl = MTMD_SLICE_TMPL_NONE;
    std::vector<llama_token> tok_ov_img_start;  // overview image
    std::vector<llama_token> tok_ov_img_end;    // overview image
    std::vector<llama_token> tok_slices_start;  // start of all slices
    std::vector<llama_token> tok_slices_end;    // end of all slices
    std::vector<llama_token> tok_sli_img_start; // single slice start
    std::vector<llama_token> tok_sli_img_end;   // single slice end
    std::vector<llama_token> tok_sli_img_mid;   // between 2 slices
    std::vector<llama_token> tok_row_end;       // end of row
    bool tok_row_end_trail = false;
    bool ov_img_first      = false;

    // MiniCPM-V 4.6/4.7 prepends an <image_id>N</image_id> tag before <image>
    bool use_image_id = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::unique_ptr<mtmd_audio_preprocessor> audio_preproc;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    // batching
    int32_t batch_max_tokens;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params,
                   bool no_alloc = false) :
        print_timings   (ctx_params.print_timings),
        n_threads       (ctx_params.n_threads),
        media_marker    (ctx_params.media_marker),
        n_embd_text     (text_model ? llama_model_n_embd_inp(text_model) : -1),
        vocab           (text_model ? llama_model_get_vocab(text_model) : nullptr),
        text_model      (text_model),
        batch_max_tokens(ctx_params.batch_max_tokens)
    {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        if (text_model) {
            auto decoder_rope_type = llama_model_rope_type(text_model);
            switch (decoder_rope_type) {
                case LLAMA_ROPE_TYPE_NONE:
                case LLAMA_ROPE_TYPE_NORM:
                case LLAMA_ROPE_TYPE_NEOX:
                    {
                        pos_type = MTMD_POS_TYPE_NORMAL;
                    } break;
                case LLAMA_ROPE_TYPE_MROPE:
                case LLAMA_ROPE_TYPE_IMROPE:
                    {
                        pos_type = MTMD_POS_TYPE_MROPE;
                    } break;
                default:
                    throw std::runtime_error(string_format("unsupported decoder rope type: %d\n", decoder_rope_type));
            }
        }

        clip_context_params ctx_clip_params {
            /* use_gpu           */ ctx_params.use_gpu,
            /* device            */ ctx_params.device,
            /* flash_attn_type   */ mtmd_get_clip_flash_attn_type(ctx_params.flash_attn_type),
            /* image_min_tokens  */ ctx_params.image_min_tokens,
            /* image_max_tokens  */ ctx_params.image_max_tokens,
            /* warmup            */ ctx_params.warmup,
            /* cb_eval           */ ctx_params.cb_eval,
            /* cb_eval_user_data */ ctx_params.cb_eval_user_data,
            /* no_alloc          */ no_alloc,
            /* progress_callback */ ctx_params.progress_callback,
            /* progress_callback_user_data */ ctx_params.progress_callback_user_data,
        };

        auto res = clip_init(mmproj_fname, ctx_clip_params);
        ctx_v = res.ctx_v;
        ctx_a = res.ctx_a;
        ctx_gen_a = res.ctx_gen_a;
        // soprano has no speaker encoder, its mmproj only holds the decoder
        const bool gen_only = ctx_gen_a && clip_get_projector_type(ctx_gen_a) == PROJECTOR_TYPE_SOPRANO;
        if (!ctx_v && !ctx_a && !gen_only) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (ctx_v && ctx_a) {
            int n_embd_v = clip_n_mmproj_embd(ctx_v);
            int n_embd_a = clip_n_mmproj_embd(ctx_a);
            if (n_embd_v != n_embd_a) {
                throw std::runtime_error(string_format(
                    "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n",
                    n_embd_v, n_embd_a));
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        // a gen-only mmproj is validated below
        int n_embd_clip = ctx_v || ctx_a ? clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a) : n_embd_text;
        if (n_embd_text > 0 && n_embd_text != n_embd_clip) {
            throw std::runtime_error(string_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_clip));
        }
        if (ctx_gen_a) {
            int n_embd_gen = clip_n_mmproj_embd(ctx_gen_a);
            if (n_embd_text > 0 && n_embd_text != n_embd_gen) {
                throw std::runtime_error(string_format(
                    "mismatch between text model (n_embd = %d) and gen-audio mmproj (n_embd = %d)\n"
                    "hint: you may be using wrong mmproj\n",
                    n_embd_text, n_embd_gen));
            }
        }
        if (ctx_v) {
            init_vision();
        }
        if (ctx_a) {
            init_audio();
        }
    }

    void init_vision() {
        GGML_ASSERT(ctx_v != nullptr);
        image_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_v);

        switch (proj) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
            case PROJECTOR_TYPE_LDP:
            case PROJECTOR_TYPE_LDPV2:
            case PROJECTOR_TYPE_COGVLM:
            case PROJECTOR_TYPE_JANUS_PRO:
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    bool has_pinpoints = !clip_get_hparams(ctx_v)->image_res_candidates.empty();
                    if (has_pinpoints) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    int minicpmv_version = clip_get_hparams(ctx_v)->minicpmv_version;
                    if (minicpmv_version == 2) {
                        // minicpmv 2.5 format:
                        // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_5;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_slices_start  = {lookup_token("<slice>")};
                        tok_slices_end    = {lookup_token("</slice>")};
                        tok_sli_img_start = tok_ov_img_start;
                        tok_sli_img_end   = tok_ov_img_end;
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;
                    } else if (minicpmv_version == 3 || minicpmv_version == 4 || minicpmv_version == 5 || minicpmv_version == 6 || minicpmv_version == 100045) {
                        // minicpmv 2.6 format:
                        // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_sli_img_start = {lookup_token("<slice>")};
                        tok_sli_img_end   = {lookup_token("</slice>")};
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;

                    } else if (minicpmv_version != 0) {
                        throw std::runtime_error(string_format("unsupported minicpmv version: %d\n", minicpmv_version));
                    }
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_MINICPMV4_6:
            case PROJECTOR_TYPE_MINICPMV4_7:
            case PROJECTOR_TYPE_MINICPM_OCR_1_0:
                {
                    slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                    tok_ov_img_start  = {lookup_token("<image>")};
                    tok_ov_img_end    = {lookup_token("</image>")};
                    tok_sli_img_start = {lookup_token("<slice>")};
                    tok_sli_img_end   = {lookup_token("</slice>")};
                    tok_row_end       = {lookup_token("\n")};
                    tok_row_end_trail = false; // no trailing end-of-row token
                    ov_img_first      = true;
                    use_image_id      = true;
                    if (clip_get_projector_type(ctx_v) == PROJECTOR_TYPE_MINICPM_OCR_1_0) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_minicpm_ocr_1_0>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_minicpmv>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_QWEN3VL:
            case PROJECTOR_TYPE_LING3VL:
            case PROJECTOR_TYPE_MIMOVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_MINIMAX_M3:
                {
                    // ]<]start of image[>[ ... (image embeddings) ... ]<]end of image[>[
                    img_beg = "]<]start of image[>[";
                    img_end = "]<]end of image[>[";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_SENSENOVA_U1:
                {
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_MUSE_GLIMMER:
                {
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_muse_glimmer>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LFM3VL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm3vl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    img_beg = "<image>";
                    img_end = "</image>";
                    // Currently only supprots single-tile preprocessing: any input is downscaled
                    // to one image_size x image_size tile (64 output tokens via 8x8 adaptive avg
                    // pool).
                    // However, the model itself supports llava-uhd multi-tile tiling for high-res
                    // images. This will be implemented in a future PR (dispatch on has_pinpoints
                    // - see LDP/COGVLM branch above) and emit image_grid_pinpoints in the conversion
                    // script.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA3:
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    // <start_of_image> ... (image embeddings) ... <end_of_image>
                    img_beg = "<start_of_image>";
                    img_end = "<end_of_image>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
                    slice_tmpl         = MTMD_SLICE_TMPL_IDEFICS3;
                    tok_ov_img_start   = {lookup_token("\n\n"), lookup_token("<fake_token_around_image>"), lookup_token("<global-img>")};
                    tok_ov_img_end     = {lookup_token("<fake_token_around_image>")};
                    tok_row_end        = {lookup_token("\n")};
                    sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_idefics3>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                    img_end = "[IMG_END]";
                    if (clip_get_hparams(ctx_v)->image_longest_edge > 0) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    // (more details in mtmd_context constructor)
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    LOG_WRN("%s: llama 4 vision is known to have degraded quality:\n"
                            "    https://github.com/ggml-org/llama.cpp/pull/13282\n", __func__);
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    // Step3 format:
                    //   <patch_start> (patch) <patch_end> [<patch_newline>]
                    //   ... (all patch rows)
                    //   <im_start> (overview) <im_end>
                    slice_tmpl        = MTMD_SLICE_TMPL_STEP3VL;
                    tok_ov_img_start  = {lookup_token("<im_start>")};
                    tok_ov_img_end    = {lookup_token("<im_end>")};
                    tok_sli_img_start = {lookup_token("<patch_start>")};
                    tok_sli_img_end   = {lookup_token("<patch_end>")};
                    tok_row_end       = {lookup_token("<patch_newline>")};
                    tok_row_end_trail = false;
                    ov_img_first      = false; // patches first, overview last
                    image_preproc = std::make_unique<mtmd_image_preprocessor_step3vl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    // <img> ... (image embeddings) ... </img>
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_internvl>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_COHERE2V:
                {
                    // <|START_OF_IMG|> (tile embeddings) <|IMG_LINE_BREAK|> ... <|END_OF_IMG|>
                    img_beg = "<|START_OF_IMG|>";
                    img_end = "<|END_OF_IMG|>";
                    tok_sli_img_end = {lookup_token("<|IMG_LINE_BREAK|>")};
                    tok_ov_img_end  = tok_sli_img_end;
                    ov_img_first = false;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_cohere2v>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIVL:
                {
                    // <|media_start|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_start|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIK25:
                {
                    // GLM-5.2-V reuses the Kimi-K2.5 vision encoder and projector, but marks
                    // images with its own tokens, so decide based on the text model vocab
                    if (lookup_token("<|begin_of_image|>") != LLAMA_TOKEN_NULL) {
                        // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                        img_beg = "<|begin_of_image|>";
                        img_end = "<|end_of_image|>";
                    } else {
                        // <|media_begin|> ... (image embeddings) ... <|media_end|>
                        img_beg = "<|media_begin|>";
                        img_end = "<|media_end|>";
                    }
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    // <|im_start|> ... (image embeddings) ... <|im_end|>
                    img_beg = "<|im_start|>";
                    img_end = "<|im_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEK4V:
                {
                    // no vocab tokens are added; the start/end/newline markers are learned embeddings emitted by the encoder
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseek4v>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEK41V:
                {
                    // as deepseek4v: markers come from the encoder, the block is decoded causally
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseek41v>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
            case PROJECTOR_TYPE_DOTS3NOTE_V:
                {
                    // <|img|> ... (image embeddings) ... <|endofimg|>
                    img_beg = "<|img|>";
                    img_end = "<|endofimg|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_OMNI:
                {
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    // multi-tile:
                    //   <|image_start|>
                    //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
                    //     <|img_thumbnail|> (thumbnail)
                    //   <|image_end|>
                    // single-tile:
                    //   <|image_start|> (image) <|image_end|>
                    img_beg            = "<|image_start|>";
                    img_end            = "<|image_end|>";
                    slice_tmpl         = MTMD_SLICE_TMPL_LFM2;
                    sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                    tok_ov_img_start   = {lookup_token("<|img_thumbnail|>")};
                    ov_img_first       = false;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm2>(ctx_v);
                } break;
            case PROJECTOR_TYPE_D1OMNI_V:
                {
                    // same tiles and thumbnail as lfm2, without separator tokens
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm2>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GLM5V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_glm5v>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PADDLEOCR:
                {
                    // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                    img_beg = "<|IMAGE_START|>";
                    img_end = "<|IMAGE_END|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
            case PROJECTOR_TYPE_GEMMA4UV:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
            case PROJECTOR_TYPE_DEEPSEEKOCR2:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx_v);
                    ov_img_first = false;
                } break;
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                    img_beg = "<｜hy_place▁holder▁no▁100｜>";
                    img_end = "<｜hy_place▁holder▁no▁101｜>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_EXAONE4_5:
                {
                    // <vision> ... (image embeddings) ... </vision>
                    img_beg = "<vision>";
                    img_end = "</vision>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GRANITE4_VISION:
                {
                    // ... (image embeddings) \n ...
                    img_beg = "";
                    img_end = "\n";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_granite>(ctx_v);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

    void init_audio() {
        GGML_ASSERT(ctx_a != nullptr);
        audio_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_a);

        LOG_WRN("%s: audio input is in experimental stage and may have reduced quality:\n"
                "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", __func__);

        // set preprocessor
        switch (proj) {
            case PROJECTOR_TYPE_QWEN2A:
            case PROJECTOR_TYPE_QWEN25O:
                {
                    // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                    aud_beg = "<|audio_bos|>";
                    aud_end = "<|audio_eos|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
                {
                    aud_beg = "<|soft_audio_start|>";
                    aud_end = "<|soft_audio_end|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_muse_spark_audio>(ctx_a);
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    aud_beg = "<|audio_start|>";
                    aud_end = "<|audio_end|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_qwen3a>(ctx_a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    // [BEGIN_AUDIO] ... (embeddings) ...
                    aud_beg = "[BEGIN_AUDIO]";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    // <sound> ... (embeddings) ...
                    aud_beg = "<sound>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL_RT:
                {
                    // audio and text share positions, only usable via mtmd_audio_stream
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
            case PROJECTOR_TYPE_GLMA:
            case PROJECTOR_TYPE_MERALION:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_conformer>(ctx_a);
                } break;
            case PROJECTOR_TYPE_D1OMNI_A:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_d1omni>(ctx_a);
                } break;
            case PROJECTOR_TYPE_LFM3A:
                {
                    // LFM3-Omni wraps the audio in the image span, LFM3-Audio (no vision tower) places it bare
                    if (ctx_v) {
                        aud_beg = img_beg;
                        aud_end = img_end;
                    }
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_lfm3a>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GRANITE_SPEECH:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_granite_speech>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4a>(ctx_a);
                } break;
            case PROJECTOR_TYPE_PARAKEET:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_parakeet>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4UA:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4ua>(ctx_a);
                } break;
            case PROJECTOR_TYPE_DOTS3NOTE_A:
                {
                    // <|audio_comp_start|> ... (embeddings) ... <|audio_comp_end|>
                    aud_beg = "<|audio_comp_start|>";
                    aud_end = "<|audio_comp_end|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_dots3note>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MIMO_AUDIO:
                {
                    aud_beg = "<|mimo_audio_start|>";
                    aud_end = "<|mimo_audio_end|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_mimo_audio>(ctx_a);
                } break;
            case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_qwen3tts_spk>(ctx_a);
                } break;
            case PROJECTOR_TYPE_POCKETTTS_SPKENC:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_pockettts>(ctx_a);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected audio projector type %d\n", __func__, proj));
        }

        // initialize audio preprocessor
        if (audio_preproc) {
            audio_preproc->initialize();
        }
    }

    // get clip ctx based on chunk type
    clip_ctx * get_clip_ctx(const mtmd_input_chunk * chunk) const {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return ctx_v;
        } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return ctx_a;
        }
        GGML_ABORT("unknown chunk type");
    }

    projector_type proj_type_v() const {
        return ctx_v ? clip_get_projector_type(ctx_v) : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return ctx_a ? clip_get_projector_type(ctx_a) : PROJECTOR_TYPE_UNKNOWN;
    }

    int64_t n_embd_out() const {
        if (ctx_v) {
            return clip_n_mmproj_embd(ctx_v);
        } else if (ctx_a) {
            return clip_n_mmproj_embd(ctx_a);
        } else {
            throw std::runtime_error("no CLIP model loaded");
        }
    }

    ~mtmd_context() {
        clip_free(ctx_a);
        clip_free(ctx_v);
        clip_free(ctx_gen_a);
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        if (vocab == nullptr) {
            // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
            // but we should reconsider this if this case is needed in other places in the future
            return LLAMA_TOKEN_NULL;
        }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }
        return piece;
    }
};

mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    delete ctx;
}

std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge) {
    std::vector<std::vector<const mtmd_bitmap *>> output;
    for (size_t i = 0; i < parts.size(); i++) {
        if (parts[i].bitmap == nullptr) {
            continue; // text part
        }
        const bool has_next = n_merge > 1 && i + 1 < parts.size() && parts[i + 1].bitmap != nullptr;
        if (has_next && parts[i].bitmap->can_merge_with(*parts[i + 1].bitmap)) {
            LOG_DBG("%s: merging 2 frames at part index %zu and %zu\n", __func__, i, i + 1);
            output.push_back({parts[i].bitmap, parts[i + 1].bitmap});
            parts.erase(parts.begin() + i + 1);
            continue;
        }
        LOG_DBG("%s: no merging for part index %zu\n", __func__, i);
        output.push_back({parts[i].bitmap});
    }
    return output;
}

struct mtmd_tokenizer {
    const mtmd_context * ctx;

    std::string input_text; // note: can contain null bytes; do not use c_str()
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    using part = mtmd_internal_part;
    std::vector<part> parts;
    // these will be freed when mtmd_tokenizer finishes
    std::vector<mtmd::bitmap> bm_from_lazy; // TODO @ngxson : refactor, free bm_from_lazy progressively
    std::vector<const char *> text_from_lazy;

    mtmd_input_chunks cur;
    uint32_t n_images_added = 0; // 0-based index assigned to the next image chunk

    ~mtmd_tokenizer() {
        // note: mtmd::bitmap is already RAII
        for (auto & str : text_from_lazy) {
            free((void *)str);
        }
    }

    mtmd_tokenizer(const mtmd_context * ctx,
            const mtmd_input_text * text,
            const mtmd_bitmap * const * bmps,
            size_t n_bitmaps) : ctx(ctx) {
        add_special   = text->add_special;
        parse_special = text->parse_special;
        vocab         = ctx->vocab;

        input_text.assign(text->text, text->text_len);

        std::vector<const mtmd_bitmap *> bitmaps(bmps, bmps + n_bitmaps);
        auto parts_str = split_text(input_text, ctx->media_marker);
        size_t i_bm = 0;
        for (const auto & part : parts_str) {
            if (part == ctx->media_marker) {
                if (i_bm >= bitmaps.size()) {
                    throw std::runtime_error(string_format("number of media markers in text (%zu) exceeds number of bitmaps (%zu)", i_bm + 1, bitmaps.size()));
                }
                parts.push_back({"", bitmaps[i_bm++]});
            } else {
                parts.push_back({std::move(part), nullptr, parse_special});
            }
        }

        size_t n_markers = 0;
        for (const auto & part : parts) {
            if (part.bitmap != nullptr) {
                n_markers++;
            }
        }
        if (n_markers != bitmaps.size()) {
            throw std::runtime_error(string_format("number of media markers in text (%zu) does not match number of bitmaps (%zu)", n_markers, bitmaps.size()));
        }

        expand_lazy_bitmaps();
    }

    mtmd_tokenizer(const mtmd_context * ctx,
            const mtmd_input_part * const * input_parts,
            size_t n_parts,
            bool add_special) : ctx(ctx) {
        this->add_special = add_special;
        parse_special = true; // only used for text returned by lazy bitmaps
        vocab         = ctx->vocab;

        for (size_t i = 0; i < n_parts; i++) {
            const mtmd_input_part * p = input_parts[i];
            if (p->text != nullptr) {
                parts.push_back({std::string(p->text->text, p->text->text_len), nullptr, p->text->parse_special});
            } else {
                parts.push_back({"", p->bitmap});
            }
        }

        expand_lazy_bitmaps();
    }

    void expand_lazy_bitmaps() {
        std::vector<part> expanded;
        expanded.reserve(parts.size());
        for (auto & p : parts) {
            if (p.bitmap != nullptr && p.bitmap->lazy_callback) {
                LOG_DBG("%s: expanding lazy bitmap\n", __func__);
                for (size_t i = 0;; i++) {
                    char * out_str = nullptr;
                    mtmd_bitmap * out_bm = nullptr;
                    int res = p.bitmap->lazy_callback(i,
                                    p.bitmap->lazy_user_data,
                                    &out_bm,
                                    &out_str);
                    if (out_bm && out_str) {
                        throw std::runtime_error(string_format("lazy callback cannot return both bitmap and text"));
                    }
                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) {
                            auto & ptr = bm_from_lazy.emplace_back(out_bm); // remember to free it later
                            expanded.push_back({"", ptr.ptr.get()});
                            LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, out_bm->nx, out_bm->ny);
                        } else if (out_str) {
                            auto & ptr = text_from_lazy.emplace_back(out_str); // remember to free it later
                            expanded.push_back({ptr, nullptr, parse_special});
                            LOG_DBG("%s: lazy callback returned text: %s\n", __func__, out_str);
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        break;
                    } else if (res == -2) {
                        // error
                        throw std::runtime_error(string_format("lazy callback returned error"));
                    }
                }
            } else {
                expanded.push_back(std::move(p));
            }
        }
        parts = std::move(expanded);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();

        // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
        int n_merge_frames = 1;
        if (ctx->ctx_v) {
            n_merge_frames = clip_model_n_temporal_merge(ctx->ctx_v);
            GGML_ASSERT(n_merge_frames <= 2 && "we only support merging maximum 2 images for now; open an issue if this model supports merging more");
        }

        auto merged_bitmaps = mtmd_group_mergeable_bitmaps(parts, n_merge_frames);

        size_t i_bm = 0;
        for (const auto & p : parts) {
            if (p.bitmap != nullptr) {
                if (i_bm >= merged_bitmaps.size()) {
                    LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                            __func__, merged_bitmaps.size(), parts.size() - 1);
                    return 1;
                }
                auto bmps = merged_bitmaps[i_bm++];
                int32_t res = add_media(bmps);
                if (res != 0) {
                    return res;
                }
            } else {
                add_text(p.text, p.parse_special);
            }
        }

        if (vocab != nullptr) {
            if (add_special && llama_vocab_get_add_bos(vocab)) {
                // if first chunk is text, we add BOS token to first text chunk
                // otherwise, create a new text chunk with BOS token
                if (!cur.entries.empty() && cur.entries[0].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    // add BOS token to the beginning of first text chunk
                    cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
                } else {
                    // create a new text chunk with BOS token at the beginning
                    mtmd_input_chunk bos_chunk{
                        MTMD_INPUT_CHUNK_TYPE_TEXT,
                        {llama_vocab_bos(vocab)},
                        nullptr, // image tokens
                        nullptr, // audio tokens
                    };
                    cur.entries.insert(cur.entries.begin(), std::move(bos_chunk));
                }
            }

            if (add_special && llama_vocab_get_add_eos(vocab)) {
                // if last chunk is text, we add EOS token to it
                add_text({llama_vocab_eos(vocab)});
            }
        }

        if (i_bm != merged_bitmaps.size()) {
            LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                    __func__, merged_bitmaps.size(), parts.size() - 1);
            return 1;
        }

        *output = std::move(cur);

        return 0;
    }

    void add_text(const std::string & txt, bool parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = mtmd_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(std::vector<const mtmd_bitmap *> & bitmaps) {
        GGML_ASSERT(!bitmaps.empty());

        // note: only one type of media is supported per call, caller should enforce this
        const bool is_vision = !bitmaps[0]->is_audio;

        if (is_vision) {
            // handle image

            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true); // add image begin token
            }

            // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

            mtmd_image_preproc_out preproc_out;

            for (const auto * bmp : bitmaps) {
                // sanity check
                GGML_ASSERT(!bmp->is_audio);
                GGML_ASSERT(ctx->image_preproc != nullptr);
                if (bmp->nx <= 0 || bmp->ny <= 0) {
                    LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n",
                            __func__, bmp->nx, bmp->ny);
                    return 2;
                }

                // convert mtmd_bitmap to clip_image_u8
                clip_image_u8 img_u8;
                img_u8.set_size(
                    {(int)bmp->nx, (int)bmp->ny},
                    bmp->is_placeholder());
                img_u8.cpy_buf(bmp->get_ro_buf());

                // preprocess image
                mtmd_image_preproc_out tmp_preproc_out = ctx->image_preproc->preprocess(img_u8);

                // move entries and grid dimensions to the "global" preproc_out
                for (auto & entry : tmp_preproc_out.entries) {
                    preproc_out.entries.emplace_back(std::move(entry));
                }

                // for llava-uhd style, we need to handle grid too
                // we don't care about overwriting these values for now because the case where bitmaps.size() > 1 is only for frame merging (qwen-vl), not supported by llava-uhd
                if ((tmp_preproc_out.grid_x > 0 && tmp_preproc_out.grid_y > 0)
                        || tmp_preproc_out.has_overview()) {
                    GGML_ASSERT(bitmaps.size() == 1);
                    preproc_out.grid_x = tmp_preproc_out.grid_x;
                    preproc_out.grid_y = tmp_preproc_out.grid_y;
                    preproc_out.overview = std::move(tmp_preproc_out.overview);
                }
            }

            LOG_DBG("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n",
                    __func__, preproc_out.entries.size(), preproc_out.grid_x, preproc_out.grid_y,
                    preproc_out.has_overview() ? 1 : 0);

            // handle llava-uhd style preprocessing
            // (output either a grid, or overview-only)
            const bool has_tiling_grid = (preproc_out.grid_x > 0 && preproc_out.grid_y > 0)
                || preproc_out.has_overview();

            if (has_tiling_grid && ctx->proj_type_v() == PROJECTOR_TYPE_MINICPMV4_7) {
                GGML_ASSERT(bitmaps.size() == 1);
                if (ctx->use_image_id) {
                    add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true);
                }
                add_text(ctx->tok_ov_img_start);
                // the separators after <image> are appended by clip, see add_canvas_chunk()
                add_canvas_chunk(std::move(preproc_out), bitmaps[0]->id);
            } else if (has_tiling_grid) {
                // [QWEN_VIDEO] we do not support "frame merging" for llama-uhd style, so no batching for now
                GGML_ASSERT(bitmaps.size() == 1);

                const int n_col = preproc_out.grid_x;
                const int n_row = preproc_out.grid_y;

                // split batch into chunks of single images
                auto chunks = split_batch_to_chunk(std::move(preproc_out), bitmaps[0]->id);
                GGML_ASSERT(chunks.size() > 0);

                // NOTE: preproc_out is invalidated after this point, do not use it anymore

                // split_batch_to_chunk must always put the overview image first
                auto ov_chunk = std::move(chunks.front());
                chunks.erase(chunks.begin());

                // add overview image (first)
                if (ctx->ov_img_first) {
                    if (ctx->use_image_id) {
                        add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true);
                    }
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    LOG_DBG("%s: adding %d slices (%d rows x %d cols)\n", __func__, (int)chunks.size(), n_row, n_col);
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(ctx->tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!ctx->tok_sli_img_start.empty()) {
                                add_text(ctx->tok_sli_img_start);
                            } else if (!ctx->sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, ctx->sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, ctx->sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }

                            auto & curr_chunk = chunks[y * n_col + x];
                            auto & curr_batch = curr_chunk.tokens_image->batch_f32;
                            if (curr_batch.entries.size() != 1) {
                                throw std::runtime_error(string_format("%s: expect 1 image in batch_f32", __func__));
                            }

                            LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, y, x);
                            cur.entries.emplace_back(std::move(curr_chunk));

                            add_text(ctx->tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(ctx->tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || ctx->tok_row_end_trail)) {
                            add_text(ctx->tok_row_end);
                        }
                    }
                    add_text(ctx->tok_slices_end);
                }

                // add overview image (last)
                if (!ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }
            } else {

                if (preproc_out.entries.size() == 0) {
                    LOG_ERR("%s: no image tokens produced by preprocessor (ref: https://github.com/ggml-org/llama.cpp/pull/24769)\n", __func__);
                    return 2;
                }

                if (ctx->proj_type_v() == PROJECTOR_TYPE_DEEPSEEK4V) {
                    // the text model perceives input in blocks of N tokens (N = COMPRESS_PAD_TO = 4, same as the CSA compress ratio)
                    // image need to be aligned to block size, while adding IMAGE_PAD embeddings to the beginning
                    // TODO @ngxson : maybe refactor this in the future
                    constexpr int32_t align = 4;
                    size_t n_past = 0;
                    for (const auto & e : cur.entries) {
                        n_past += mtmd_input_chunk_get_n_tokens(&e);
                    }
                    preproc_out.entries[0].lead_pad = align - 1 - (int32_t)(n_past % align);
                }

                size_t n_tokens = 0;
                for (auto & e : preproc_out.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_v, &e);
                    if (clip_model_n_temporal_merge(ctx->ctx_v) == 2) {
                        // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
                        break;
                    }
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);

                // [QWEN_VIDEO] improve this in the future
                image_tokens->n_temporal_merge = clip_model_n_temporal_merge(ctx->ctx_v);

                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, &preproc_out.entries[0]);
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, &preproc_out.entries[0]);
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->pos = ctx->pos_type;
                // U1 uses M-RoPE, but unlike Qwen its image H/W coordinates are
                // zero-based and the whole image occupies one temporal position.
                if (ctx->proj_type_v() == PROJECTOR_TYPE_SENSENOVA_U1) {
                    image_tokens->pos = MTMD_POS_TYPE_SENSENOVA_U1;
                }
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index). Override the position type so that
                // n_tokens() and mtmd_image_tokens_get_decoder_pos pick the HunyuanVL layout.
                else if (ctx->proj_type_v() == PROJECTOR_TYPE_HUNYUANVL) {
                    image_tokens->pos       = MTMD_POS_TYPE_HUNYUANVL;
                    image_tokens->image_idx = n_images_added;
                    GGML_ASSERT(n_tokens == (size_t)image_tokens->n_tokens());
                }

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = false;
                batch_f32.is_video = bitmaps[0]->mergeable;
                batch_f32.entries = std::move(preproc_out.entries);
                // do NOT use preproc_out from this point on, it's moved

                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmaps[0]->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true); // add image end token
            }

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            n_images_added++;

        } else {
            // handle audio

            GGML_ASSERT(bitmaps.size() == 1); // no batching support for now
            auto & bitmap = bitmaps[0];

            if (!ctx->ctx_a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (bitmap->nx == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!ctx->aud_beg.empty()) {
                add_text(ctx->aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(ctx->audio_preproc != nullptr);

            // preprocess audio
            std::vector<mtmd_audio_mel> mel_spec_chunks;
            {
                std::vector<float> dummy;
                const float * samples = nullptr;
                size_t n_samples = 0;
                if (bitmap->is_placeholder()) {
                    // TODO @ngxson : skip underlay processing if bitmap is placeholder
                    GGML_ASSERT(bitmap->ny == 1);

                    dummy.resize(bitmap->nx);
                    samples = dummy.data();
                    n_samples = dummy.size();
                } else {
                    const auto & buf = bitmap->get_ro_buf();
                    GGML_ASSERT(buf.size() > sizeof(float));
                    GGML_ASSERT(buf.size() % sizeof(float) == 0);

                    samples = (const float *)buf.data();
                    n_samples = buf.size() / sizeof(float);
                }
                if (!ctx->audio_preproc) {
                    LOG_ERR("%s: this model takes audio input only via mtmd_audio_stream\n", __func__);
                    return 2;
                }
                bool ok = ctx->audio_preproc->preprocess(samples, n_samples, mel_spec_chunks);
                if (!ok) {
                    LOG_ERR("Unable to preprocess audio\n");
                    return 2;
                }
            }

            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (auto & mel_spec : mel_spec_chunks) {
                const bool is_placeholder = mel_spec.data.empty();

                // Validate dimensions fit in clip_image_size (int)
                GGML_ASSERT(mel_spec.n_len <= INT32_MAX && mel_spec.n_len >= 0);
                GGML_ASSERT(mel_spec.n_mel <= INT32_MAX && mel_spec.n_mel >= 0);
                clip_image_f32 mel_f32;
                mel_f32.set_size(
                    {(int)mel_spec.n_len, (int)mel_spec.n_mel},
                    is_placeholder, /* is_audio */ true);
                mel_f32.cpy_buf(mel_spec.data);

                size_t n_tokens = clip_n_output_tokens(ctx->ctx_a, &mel_f32);

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->aud_end.empty()) {
                add_text(ctx->aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    // MiniCPM-V 4.7: the overview and all slices go in one chunk, clip appends the separators after each tile:
    //   [ov] </image><slice> [S00] </slice><slice> [S01] </slice>\n<slice> [S10] </slice><slice> [S11] </slice>
    void add_canvas_chunk(mtmd_image_preproc_out && preproc_out, const std::string & id) {
        const int n_col = preproc_out.grid_x;
        const int n_row = preproc_out.grid_y;
        auto & slices = preproc_out.entries;
        GGML_ASSERT(preproc_out.has_overview());
        GGML_ASSERT((int) slices.size() == n_col * n_row);

        auto & ov = preproc_out.overview;
        ov.suffix_type = CLIP_SUFFIX_MINICPMV_OV;
        if (!slices.empty()) {
            ov.suffix_type = CLIP_SUFFIX_MINICPMV_OV_SLICE;
            ov.anyres.grid_x = n_col;
            ov.anyres.grid_y = n_row;
        }
        for (int y = 0; y < n_row; y++) {
            for (int x = 0; x < n_col; x++) {
                auto & suffix = slices[y * n_col + x].suffix_type;
                if (y == n_row - 1 && x == n_col - 1) {
                    suffix = CLIP_SUFFIX_MINICPMV_LAST;
                } else if (x == n_col - 1) {
                    suffix = CLIP_SUFFIX_MINICPMV_ROW_END;
                } else {
                    suffix = CLIP_SUFFIX_MINICPMV_SLICE;
                }
            }
        }

        mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
        image_tokens->pos = MTMD_POS_TYPE_CANVAS;
        image_tokens->id  = id;
        auto & entries = image_tokens->batch_f32.entries;
        entries.push_back(std::move(ov));
        for (auto & slice : slices) {
            entries.push_back(std::move(slice));
        }
        // token grid of the last entry, the grids of the other entries are scaled from it
        image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, &entries.back());
        image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, &entries.back());

        size_t n_tokens = 0;
        for (const auto & entry : entries) {
            n_tokens += clip_n_output_tokens(ctx->ctx_v, &entry);
        }
        GGML_ASSERT(n_tokens == image_tokens->n_tokens());

        mtmd_input_chunk chunk{
            MTMD_INPUT_CHUNK_TYPE_IMAGE,
            {}, // text tokens
            std::move(image_tokens),
            nullptr, // audio tokens
        };
        cur.entries.emplace_back(std::move(chunk));
    }

    std::vector<mtmd_input_chunk> split_batch_to_chunk(mtmd_image_preproc_out && preproc_out, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        auto process_chunk = [&](clip_image_f32 && img) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = clip_n_output_tokens(ctx->ctx_v, &img);
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(img));
            image_tokens->id = id;

            GGML_ASSERT(image_tokens->nx > 0);

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        };

        // overview image first
        auto & overview = preproc_out.overview;
        if (overview.nx() == 0 || overview.ny() == 0) {
            throw std::runtime_error(string_format("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
        }
        process_chunk(std::move(preproc_out.overview));

        // then, process slices
        for (auto & entry : preproc_out.entries) {
            if (entry.nx() == 0 || entry.ny() == 0) {
                throw std::runtime_error(string_format("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
            }
            process_chunk(std::move(entry));
        }

        return chunks;
    }

    // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
    static std::vector<std::string> split_text(const std::string & input, const std::string & delimiter) {
        std::vector<std::string> result;
        if (input.empty()) {
            return result;
        }
        size_t start = 0;
        size_t pos = 0;
        while ((pos = input.find(delimiter, start)) != std::string::npos) {
            if (pos > start) {
                result.push_back(input.substr(start, pos - start));
            }
            result.push_back(delimiter);
            start = pos + delimiter.length();
        }
        if (start < input.length()) {
            result.push_back(input.substr(start));
        }
        return result;
    }

    // copied from common_tokenize
    static std::vector<llama_token> mtmd_tokenize_text_internal(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        if (n_tokens == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
        }
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
            GGML_ASSERT(check == -n_tokens);
        } else {
            result.resize(n_tokens);
        }
        return result;
    }
};

int32_t mtmd_tokenize(const mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap * const * bitmaps,
            size_t n_bitmaps) {
    try {
        mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}

int32_t mtmd_tokenize_from_parts(const mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_part * const * parts,
            size_t n_parts,
            bool add_special) {
    for (size_t i = 0; i < n_parts; i++) {
        if ((parts[i]->text == nullptr) == (parts[i]->bitmap == nullptr)) {
            LOG_ERR("%s: part %zu must have either text or bitmap set, not both\n", __func__, i);
            return 1;
        }
        if (parts[i]->text != nullptr && parts[i]->text->text == nullptr) {
            LOG_ERR("%s: part %zu has null text pointer\n", __func__, i);
            return 1;
        }
    }
    try {
        mtmd_tokenizer tokenizer(ctx, parts, n_parts, add_special);
        return tokenizer.tokenize(output);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 2;
    }
}

static int32_t mtmd_encode_impl(mtmd_context * ctx, const mtmd_image_tokens * image_tokens, std::vector<float> & out_embd) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }

    int n_embd_out = ctx->n_embd_out();
    auto n_tokens_out = image_tokens->n_tokens();
    out_embd.resize((size_t)n_embd_out * n_tokens_out);

    if (image_tokens->is_placeholder()) {
        LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
        return 1;
    }

    if (image_tokens->pos == MTMD_POS_TYPE_CANVAS) {
        // the tiles differ in size, encode them one by one
        size_t offset = 0;
        for (const auto & entry : image_tokens->batch_f32.entries) {
            clip_image_f32_batch one;
            one.entries.push_back(entry);
            std::vector<float> embd((size_t) n_embd_out * clip_n_output_tokens(ctx_clip, &entry));
            if (!clip_image_batch_encode(ctx_clip, ctx->n_threads, &one, embd)) {
                return 1;
            }
            GGML_ASSERT(offset + embd.size() <= out_embd.size());
            std::copy(embd.begin(), embd.end(), out_embd.begin() + offset);
            offset += embd.size();
        }
        GGML_ASSERT(offset == out_embd.size());
        return 0;
    }

    bool ok = clip_image_batch_encode(
        ctx_clip,
        ctx->n_threads,
        &image_tokens->batch_f32,
        out_embd);

    return ok ? 0 : 1;
}

static int32_t mtmd_encode_chunk_impl(mtmd_context * ctx, const mtmd_input_chunk * chunk, std::vector<float> & out_embd) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        if (!ctx->ctx_v) {
            LOG_ERR("%s: model does not support vision input\n", __func__);
            return 1;
        }
        if (chunk->tokens_image == nullptr) {
            LOG_ERR("%s: image tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_image->is_placeholder()) {
            LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
            return 1;
        }
        return mtmd_encode_impl(ctx, chunk->tokens_image.get(), out_embd);
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        if (!ctx->ctx_a) {
            LOG_ERR("%s: model does not support audio input\n", __func__);
            return 1;
        }
        if (chunk->tokens_audio == nullptr) {
            LOG_ERR("%s: audio tokens are null\n", __func__);
            return 1;
        }
        if (chunk->tokens_audio->is_placeholder()) {
            LOG_ERR("%s: audio tokens batch is placeholder\n", __func__);
            return 1;
        }
        int n_mmproj_embd = ctx->n_embd_out();
        out_embd.resize((size_t)chunk->tokens_audio->n_tokens * n_mmproj_embd);
        bool ok = clip_image_batch_encode(
            ctx->ctx_a,
            ctx->n_threads,
            &chunk->tokens_audio->batch_f32,
            out_embd);
        return ok ? 0 : 1;
    }

    LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)chunk->type);
    return 1;
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    // this is the non-batching version
    try {
        return mtmd_encode_chunk_impl(ctx, chunk, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    try {
        return mtmd_encode_impl(ctx, image_tokens, ctx->out_embd);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->out_embd.data();
}

//
// audio generation
//

mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * ctx) {
    mtmd_gen_audio_info info{};
    info.model_variant = "";
    if (!ctx->ctx_gen_a) {
        info.type = MTMD_GEN_AUDIO_TYPE_NONE;
        return info;
    }
    info.model_variant = clip_get_hparams(ctx->ctx_gen_a)->gen_model_variant.c_str();
    switch (clip_get_projector_type(ctx->ctx_gen_a)) {
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            info.type = MTMD_GEN_AUDIO_TYPE_QWEN3TTS;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            info.type = MTMD_GEN_AUDIO_TYPE_POCKETTTS;
            info.sample_rate = 24000;
            break;
        case PROJECTOR_TYPE_SOPRANO:
            info.type = MTMD_GEN_AUDIO_TYPE_SOPRANO;
            info.sample_rate = 32000;
            break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            info.type = MTMD_GEN_AUDIO_TYPE_LFM3A;
            info.sample_rate = 24000;
            break;
        default:
            info.type = MTMD_GEN_AUDIO_TYPE_NONE;
            break;
    }
    return info;
}

mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * ctx) {
    mtmd_gen_inp inp{};
    inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    if (!ctx->ctx_gen_a) {
        return inp;
    }

    switch (clip_get_projector_type(ctx->ctx_gen_a)) {
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            // https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/generation_config.json
            inp.top_k = 50;
            inp.top_p = 1.0f;
            inp.temp  = 0.9f; // TODO: handle this on graph
            break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            // https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/default_parameters.py
            inp.top_k = 50;
            inp.top_p = 1.0f;
            inp.temp  = 0.7f;
            break;
        case PROJECTOR_TYPE_SOPRANO:
            inp.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
            break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            {
                // MaskGIT sampling runs in the helper, with the settings of the checkpoint
                const auto * hparams = clip_get_hparams(ctx->ctx_gen_a);
                inp.top_k          = hparams->gen_top_k;
                inp.top_p          = 1.0f;
                inp.temp           = hparams->gen_temp;
                inp.penalty_repeat = hparams->gen_penalty_repeat;
                inp.penalty_last_n = hparams->gen_penalty_last_n;
            } break;
        default:
            break;
    }
    return inp;
}

struct mtmd_gen_state {
    clip_ctx * ctx_clip = nullptr;
    clip_gen_state * clip_state = nullptr;

    std::vector<int32_t> out_codes; // this frame's 16 sampled codes (GEN_CODE)
    std::vector<float>   out_feats; // this frame's continuous features, if any (GEN_CODE)
    std::vector<float>   out_embd;  // next-step hidden state fed back to backbone (GEN_CODE)
    std::vector<float>   out_audio; // decoded PCM samples for the current frame (GEN_WAV)

    // lfm3a: the codec graph yields STFT frames, the ISTFT runs on the host
    std::unique_ptr<mtmd_audio_streaming_istft> istft;

    ~mtmd_gen_state() {
        clip_gen_state_free(clip_state);
    }
};

mtmd_gen_state_params mtmd_gen_state_params_default(void) {
    mtmd_gen_state_params params{};
    params.seed = UINT32_MAX;
    return params;
}

mtmd_gen_state * mtmd_gen_state_init(mtmd_context * ctx, mtmd_gen_state_params params) {
    if (!ctx->ctx_gen_a) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return nullptr;
    }
    auto st = std::make_unique<mtmd_gen_state>();
    st->ctx_clip   = ctx->ctx_gen_a;
    st->clip_state = clip_gen_state_init(ctx->ctx_gen_a, params.seed);
    if (!st->clip_state) {
        return nullptr;
    }
    if (clip_get_projector_type(ctx->ctx_gen_a) == PROJECTOR_TYPE_LFM3A_GEN) {
        const auto * hparams = clip_get_hparams(ctx->ctx_gen_a);
        st->istft = std::make_unique<mtmd_audio_streaming_istft>(hparams->audio_n_fft, hparams->audio_hop_len);
    }
    return st.release();
}

void mtmd_gen_state_free(mtmd_gen_state * state) {
    delete state;
}

void mtmd_gen_state_reset(mtmd_gen_state * state) {
    clip_gen_state_reset(state->clip_state);
    if (state->istft) {
        const auto * hparams = clip_get_hparams(state->ctx_clip);
        state->istft = std::make_unique<mtmd_audio_streaming_istft>(hparams->audio_n_fft, hparams->audio_hop_len);
    }
}

// lfm3a GEN_WAV: the codec graph yields polar STFT frames, the ISTFT runs here; n_codes == 0 flushes the tail and
// resets the state for the next utterance
static int32_t mtmd_gen_wav_lfm3a(mtmd_context * ctx, mtmd_gen_state * state, const mtmd_gen_inp * inp, mtmd_gen_out * out) {
    clip_ctx * ctx_clip = ctx->ctx_gen_a;
    const auto & hparams = *clip_get_hparams(ctx_clip);

    const int n_bins = hparams.audio_n_fft / 2 + 1;
    const int n_trim = (hparams.audio_n_fft - hparams.audio_hop_len) / 2;

    auto & audio = state->out_audio;
    audio.clear();

    if (!inp->codes || inp->n_codes == 0) {
        // the tail beyond the last frame is padding, like the centered ISTFT of the reference
        audio = state->istft->flush();
        audio.resize(audio.size() - std::min(audio.size(), (size_t) n_trim));
        mtmd_gen_state_reset(state);
    } else {
        std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);
        std::vector<float>   feats;

        // no hidden-state input, the batch entry is an unused placeholder
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        clip_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
        params.codes       = &in_codes;
        params.out_feats   = &feats;
        params.gen_state   = state->clip_state;
        if (!clip_encode(ctx_clip, &params)) {
            LOG_ERR("%s: clip_encode failed (codec)\n", __func__);
            return 1;
        }

        // [2 * n_bins, n_stft_frames]: log magnitude, then phase
        GGML_ASSERT(feats.size() % (2 * (size_t) n_bins) == 0);
        const size_t n_frames = feats.size() / (2 * (size_t) n_bins);
        std::vector<float> spec(2 * (size_t) n_bins);
        for (size_t f = 0; f < n_frames; f++) {
            const float * fr = feats.data() + f * 2 * n_bins;
            for (int j = 0; j < n_bins; j++) {
                const float mag = std::exp(fr[j]);
                spec[2 * j + 0] = mag * std::cos(fr[n_bins + j]);
                spec[2 * j + 1] = mag * std::sin(fr[n_bins + j]);
            }
            const std::vector<float> pcm = state->istft->process_frame(spec.data());
            audio.insert(audio.end(), pcm.begin(), pcm.end());
        }
    }

    out->audio     = audio.data();
    out->n_samples = audio.size();
    return 0;
}

static int32_t mtmd_gen_audio_process_impl(mtmd_context * ctx, mtmd_gen_state * state, const mtmd_gen_inp * inp, mtmd_gen_out * out) {
    clip_ctx * ctx_clip = ctx->ctx_gen_a;
    if (!ctx_clip) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return 1;
    }
    if (!state || state->ctx_clip != ctx_clip) {
        LOG_ERR("%s: invalid gen state\n", __func__);
        return 1;
    }

    *out = {};

    if (clip_get_projector_type(ctx_clip) == PROJECTOR_TYPE_SOPRANO) {
        const size_t n_embd = (size_t) clip_n_mmproj_embd(ctx_clip);
        if (inp->type != MTMD_GEN_PROCESS_TYPE_GEN_WAV || !inp->feats || inp->n_feats < 2 * n_embd || inp->n_feats % n_embd != 0 || inp->codes) {
            LOG_ERR("%s: soprano requires at least 2 frames of features, without codes\n", __func__);
            return 1;
        }
    }

    if (inp->type == MTMD_GEN_PROCESS_TYPE_GEN_WAV && clip_get_projector_type(ctx_clip) == PROJECTOR_TYPE_LFM3A_GEN) {
        return mtmd_gen_wav_lfm3a(ctx, state, inp, out);
    }

    if (inp->type == MTMD_GEN_PROCESS_TYPE_GEN_CODE) {
        const size_t n_embd = (size_t) clip_n_mmproj_embd(ctx_clip);

        // the hidden state may be omitted when the graph does not use it
        clip_image_f32 hidden_state;
        hidden_state.set_size({(int) n_embd, 1}, false, true);
        hidden_state.cpy_buf(inp->embd ? std::vector<float>(inp->embd, inp->embd + n_embd) : std::vector<float>(n_embd, 0.0f));

        std::vector<int32_t> in_codes;
        if (inp->codes) {
            in_codes.assign(inp->codes, inp->codes + inp->n_codes);
        }

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(hidden_state));

        std::vector<float>   out_embd(n_embd);
        std::vector<int32_t> out_codes;
        std::vector<float>   out_feats;
        bool is_eos = false;

        clip_encode_params params;
        params.imgs          = &batch;
        params.n_threads     = ctx->n_threads;
        params.gen_process   = CLIP_GEN_PROCESS_GEN_CODE;
        params.out_embd      = &out_embd;
        params.out_codes     = &out_codes;
        params.out_feats     = &out_feats;
        params.code0         = inp->code0;
        params.top_k         = inp->top_k;
        params.top_p         = inp->top_p;
        params.temp          = inp->temp;
        params.codes         = inp->codes ? &in_codes : nullptr;
        params.out_is_eos    = &is_eos;
        params.gen_state     = state->clip_state;

        if (!clip_encode(ctx_clip, &params)) {
            LOG_ERR("%s: clip_encode failed (gen_code)\n", __func__);
            return 1;
        }

        state->out_embd  = std::move(out_embd);
        state->out_codes = std::move(out_codes);
        state->out_feats = std::move(out_feats);

        out->embd      = state->out_embd.data();
        out->codes     = state->out_codes.data();
        out->n_codes   = state->out_codes.size();
        out->feats     = state->out_feats.data();
        out->n_feats   = state->out_feats.size();
        out->is_eos    = is_eos;
        return 0;
    }

    // MTMD_GEN_PROCESS_TYPE_GEN_WAV
    const bool has_codes = inp->codes && inp->n_codes > 0;
    const bool has_feats = inp->feats && inp->n_feats > 0;
    if (has_codes == has_feats) {
        LOG_ERR("%s: gen_wav requires exactly one of codes or feats\n", __func__);
        return 1;
    }
    std::vector<int32_t> in_codes;
    std::vector<float>   in_feats;
    if (has_codes) {
        in_codes.assign(inp->codes, inp->codes + inp->n_codes);
    } else {
        in_feats.assign(inp->feats, inp->feats + inp->n_feats);
    }

    // gen_wav has no hidden-state input, the batch entry is an unused placeholder
    // TODO @ngxson : some models in the future may require hidden-state input, need to update this code later
    clip_image_f32 dummy;
    dummy.set_size({1, 1}, false, true);
    dummy.cpy_buf(std::vector<float>(1, 0.0f));

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(dummy));

    clip_encode_params params;
    params.imgs        = &batch;
    params.n_threads   = ctx->n_threads;
    params.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
    params.codes       = has_codes ? &in_codes : nullptr;
    params.feats       = has_feats ? &in_feats : nullptr;
    params.out_audio   = &state->out_audio;
    params.gen_state   = state->clip_state;

    if (!clip_encode(ctx_clip, &params)) {
        LOG_ERR("%s: clip_encode failed (code2wav)\n", __func__);
        return 1;
    }

    out->audio      = state->out_audio.data();
    out->n_samples  = state->out_audio.size();

    return 0;
}

int32_t mtmd_gen_audio_process(mtmd_context * ctx, mtmd_gen_state * state, const struct mtmd_gen_inp * inp, struct mtmd_gen_out * out) {
    try {
        return mtmd_gen_audio_process_impl(ctx, state, inp, out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

mtmd_batch * mtmd_batch_init(mtmd_context * ctx) {
    return new mtmd_batch(ctx);
}

void mtmd_batch_free(mtmd_batch * batch) {
    if (batch) {
        delete batch;
    }
}

int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
        return 1;
    }

    auto * ctx = batch->ctx->get_clip_ctx(chunk);
    if (!ctx) {
        LOG_ERR("%s: model does not support input chunk type %d\n", __func__, (int)chunk->type);
        return 1;
    }

    if (batch->entries.empty()) {
        // batch must have at least one chunk
        batch->entries.push_back(chunk);
        return 0;
    }

    if (!clip_support_batch(ctx)) {
        // if no batching support, batch can only have one single chunk
        return 2; // "batch too large" error code
    }

    int32_t new_n_tokens = batch->n_tokens() + (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
    if (new_n_tokens > batch->ctx->batch_max_tokens) {
        return 2; // "batch too large" error code
    }

    auto & first_chunk = batch->entries[0];
    if (first_chunk->can_batch_with(*chunk)) {
        batch->entries.push_back(chunk);
        return 0;
    }

    return 3; // "cannot batch" error code
}

static int32_t mtmd_batch_encode_impl(mtmd_batch * batch) {
    if (batch->entries.empty()) {
        LOG_ERR("%s: batch is empty\n", __func__);
        return 1;
    }
    for (const auto * chunk : batch->entries) {
        if (chunk->is_placeholder()) {
            LOG_ERR("%s: chunk is placeholder\n", __func__);
            return 1;
        }
    }

    // represent the whole batch as one single chunk
    mtmd::input_chunk_ptr batch_chunk(mtmd_input_chunk_copy(batch->entries[0]));
    if (batch_chunk->tokens_image) {
        auto & b0_f32 = batch_chunk->tokens_image->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_image);
            auto b1_f32 = chunk->tokens_image->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else if (batch_chunk->tokens_audio) {
        auto & b0_f32 = batch_chunk->tokens_audio->batch_f32;
        // copy all entries from other chunks into the first chunk's batch_f32
        // note: skip first entry because it's already in batch_chunk
        for (size_t ic = 1; ic < batch->entries.size(); ic++) {
            auto & chunk = batch->entries[ic];
            GGML_ASSERT(chunk->tokens_audio);
            auto b1_f32 = chunk->tokens_audio->batch_f32.clone();
            for (size_t i = 0; i < b1_f32.entries.size(); i++) {
                b0_f32.entries.push_back(std::move(b1_f32.entries[i]));
            }
        }
    } else {
        LOG_ERR("%s: unsupported chunk type\n", __func__);
        return 1;
    }

    LOG_DBG("%s: encoding batch with %zu entries and total %zu tokens\n",
            __func__, batch->entries.size(), mtmd_input_chunk_get_n_tokens(batch_chunk.get()));
    int32_t res = mtmd_encode_chunk_impl(
        batch->ctx,
        batch_chunk.get(),
        batch->output_embd);
    return res;
}

int32_t mtmd_batch_encode(mtmd_batch * batch) {
    try {
        return mtmd_batch_encode_impl(batch);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (batch->output_embd.empty()) {
        LOG_ERR("%s: batch has not been encoded yet\n", __func__);
        return nullptr;
    }
    size_t offset = 0;
    const size_t n_embd = batch->ctx->n_embd_out();
    for (const auto * c : batch->entries) {
        size_t offset_prev = offset;
        size_t n_tokens = mtmd_input_chunk_get_n_tokens(c);
        offset += n_tokens * n_embd;
        GGML_ASSERT(offset_prev <  batch->output_embd.size());
        GGML_ASSERT(offset      <= batch->output_embd.size());
        if (c == chunk) {
            return &batch->output_embd.data()[offset_prev];
        }
    }
    return nullptr; // not found
}

bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA4V:
            {
                // E2B (n_embd = 1536) and E4B (n_embd = 2560) always use causal
                // note: use mmproj n_embd, because text model may not be provided (e.g. mtmd_get_memory_usage)
                const int n_embd = clip_n_mmproj_embd(ctx->ctx_v);
                return n_embd != 1536 && n_embd != 2560;
            }
        case PROJECTOR_TYPE_GEMMA4UV:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_SENSENOVA_U1:
        case PROJECTOR_TYPE_DEEPSEEK4V:
            return true;
        default:
            return false;
    }
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx->pos_type == MTMD_POS_TYPE_MROPE;
}

bool mtmd_support_vision(const mtmd_context * ctx) {
    return ctx->ctx_v != nullptr;
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    if (!ctx->ctx_a) {
        return -1;
    }
    return clip_get_hparams(ctx->ctx_a)->audio_sample_rate;
}

const char * mtmd_get_marker(const mtmd_context * ctx) {
    return ctx->media_marker.c_str();
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap(data, nx, ny);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                          const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap((const unsigned char *)data, n_samples);
    GGML_ASSERT(bitmap->is_audio);
    if (!bitmap->is_placeholder()) {
        GGML_ASSERT(bitmap->get_ro_buf().size() == n_samples * sizeof(float));
    }
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return nullptr;
    }
    return bitmap->get_ro_buf().data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    if (bitmap->is_placeholder()) {
        return 0;
    }
    return bitmap->get_ro_buf().size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap->id.c_str();
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (id) {
        bitmap->id = std::string(id);
    } else {
        bitmap->id.clear();
    }
}

void mtmd_bitmap_set_mergeable(mtmd_bitmap * bitmap, bool mergeable) {
    bitmap->mergeable = mergeable;
}

mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx,
                                    const char * id,
                                    void * user_data,
                                    mtmd_bitmap_lazy_callback callback) {
    GGML_UNUSED(ctx); // reserved for future use
    mtmd_bitmap * bitmap = new mtmd_bitmap(nullptr, 0, 0);
    bitmap->lazy_callback = callback;
    bitmap->lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    if (bitmap) {
        delete bitmap;
    }
}

// mtmd_input_chunks

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks->entries.size();
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    if (idx >= chunks->entries.size()) {
        return nullptr;
    }
    return &chunks->entries[idx];
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    if (chunks) {
        delete chunks;
    }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk->type;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        *n_tokens_output = chunk->tokens_text.size();
        return chunk->tokens_text.data();
    }
    *n_tokens_output = 0;
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image.get();
    }
    return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image->id.c_str();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->id.c_str();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        // copy the audio tokens
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    if (chunk) {
        delete chunk;
    }
}

// returns 0 on success
static int32_t mtmd_input_chunk_save_impl(const mtmd_input_chunk * chunk, std::vector<char> & out_buf) {
    try {
        mtmd_serialization ser(MTMD_SERIALIZATION_VERSION);
        chunk->serialize(ser);
        out_buf = std::move(ser.data);
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk * chunk) {
    // this is hacky, but still faster than copy the whole batch data
    std::vector<char> buf;
    if (mtmd_input_chunk_save_impl(chunk, buf) != 0) {
        return nullptr;
    }
    return mtmd_input_chunk_load(buf.data(), buf.size());
}

int32_t mtmd_input_chunk_save(const mtmd_input_chunk * chunk, char * out_buf, size_t out_len, size_t * expected_out_len) {
    std::vector<char> buf;
    if (mtmd_input_chunk_save_impl(chunk, buf) != 0) {
        return -1;
    }
    if (expected_out_len) {
        *expected_out_len = buf.size();
    }
    if (!out_buf) {
        // caller is only querying the required size
        return 0;
    }
    if (out_len < buf.size()) {
        LOG_ERR("%s: out_buf is too small, need %zu bytes, got %zu\n", __func__, buf.size(), out_len);
        return -1;
    }
    std::memcpy(out_buf, buf.data(), buf.size());
    return 0;
}

mtmd_input_chunk * mtmd_input_chunk_load(const char * buf, size_t len) {
    try {
        mtmd_serialization ser(MTMD_SERIALIZATION_VERSION, buf, len);
        mtmd::input_chunk_ptr chunk(new mtmd_input_chunk());
        chunk->deserialize(ser);
        return chunk.release();
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

// map a tile coordinate onto the canvas like the reference: round(linspace(0, canvas - 1, grid)), round() breaks ties to even
static uint32_t mtmd_canvas_scale(uint32_t coord, uint32_t grid, uint32_t canvas) {
    if (grid <= 1 || canvas <= 1) {
        return 0;
    }
    const double v = (double) coord * (double) (canvas - 1) / (double) (grid - 1);
    return std::min((uint32_t) std::nearbyint(v), canvas - 1);
}

// MTMD_POS_TYPE_CANVAS: every tile shares the <image> token before the chunk as origin
// the overview is stretched over the whole canvas, each slice fills its own cell; the time component is the origin, in slot z
// a tile takes one position in slot t (the KV cache position), the separators after it take one position each
static mtmd_decoder_pos mtmd_canvas_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    const auto & entries = image_tokens->batch_f32.entries;
    const auto & grid    = entries[0].anyres;
    const uint32_t nx = image_tokens->nx;
    const uint32_t ny = image_tokens->ny;
    const uint32_t canvas_w = grid.is_tiled() ? grid.grid_x * nx : nx;
    const uint32_t canvas_h = grid.is_tiled() ? grid.grid_y * ny : ny;
    const uint32_t base = pos_0 - 1;

    mtmd_decoder_pos pos;
    uint32_t t = pos_0;
    for (size_t k = 0; k < entries.size(); ++k) {
        const auto [gw, gh] = image_tokens->canvas_tile_grid(k);
        if (i < gw * gh) {
            const uint32_t row = i / gw;
            const uint32_t col = i % gw;
            uint32_t h;
            uint32_t w;
            if (k == 0) {
                h = mtmd_canvas_scale(row, gh, canvas_h);
                w = mtmd_canvas_scale(col, gw, canvas_w);
            } else {
                const uint32_t s = k - 1;
                h = (s / grid.grid_x) * ny + row;
                w = (s % grid.grid_x) * nx + col;
            }
            pos.t = t;
            pos.x = base + w;
            pos.y = base + h;
            pos.z = base;
            return pos;
        }
        i -= gw * gh;

        const size_t n_sep = clip_suffix_rows(entries[k].suffix_type).size();
        if (i < n_sep) {
            const uint32_t p = t + 1 + i;
            pos.t = p;
            pos.x = p;
            pos.y = p;
            pos.z = p;
            return pos;
        }
        i -= n_sep;
        t += 1 + n_sep;
    }
    GGML_ABORT("token index out of range");
}

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            {
                pos.t = pos_0;
                pos.x = pos_0 + (i % image_tokens->nx);
                pos.y = pos_0 + (i / image_tokens->nx);
                pos.z = 0; // unused for now
            } break;
        case MTMD_POS_TYPE_NORMAL:
            {
                pos.t = pos_0 + i;
                pos.x = pos_0 + i;
                pos.y = pos_0 + i;
                pos.z = pos_0 + i;
            } break;
        case MTMD_POS_TYPE_SENSENOVA_U1:
            {
                pos.t = pos_0;
                pos.x = i % image_tokens->nx;
                pos.y = i / image_tokens->nx;
                pos.z = 0;
            } break;
        case MTMD_POS_TYPE_HUNYUANVL:
            {
                // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
                // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
                // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
                const uint32_t nx      = image_tokens->nx;
                const uint32_t n_total = image_tokens->n_tokens();
                if (i == 0) {
                    // BOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else if (i == n_total - 1) {
                    // EOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else {
                    // content token at (row, col), or the trailing newline of a row (col == nx)
                    //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
                    // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
                    const uint32_t offset = (uint32_t)i - 1;
                    const uint32_t row    = offset / (nx + 1);
                    const uint32_t col    = offset % (nx + 1);
                    pos.t = pos_0 + i;
                    pos.x = row;
                    pos.y = col;
                    pos.z = image_tokens->image_idx;
                }
            } break;
        case MTMD_POS_TYPE_CANVAS:
            {
                pos = mtmd_canvas_decoder_pos(image_tokens, pos_0, i);
            } break;
        default:
            GGML_ABORT("invalid position type");
    }
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            return std::max(image_tokens->nx, image_tokens->ny);
        case MTMD_POS_TYPE_NORMAL:
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_SENSENOVA_U1:
            return 1;
        case MTMD_POS_TYPE_HUNYUANVL:
            // HunyuanVL: the sequential (dim-0) position advances by the full token count
            // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_CANVAS:
            {
                // one position per tile, plus one per separator
                llama_pos n_pos = 0;
                for (const auto & entry : image_tokens->batch_f32.entries) {
                    n_pos += 1 + (llama_pos) clip_suffix_rows(entry.suffix_type).size();
                }
                return n_pos;
            }
        default:
            GGML_ABORT("invalid position type");
    }
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    mtmd_input_chunk chunk_text{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        nullptr, // image tokens
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_text));

    // create an image chunk
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx = 4;
    image_tokens->ny = 4;
    image_tokens->batch_f32.entries.resize(16);
    image_tokens->id = "image_1";
    mtmd_input_chunk chunk_image{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : clip_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char * fname) {
    try {
        auto tmp = clip_get_cap(fname);
        mtmd_caps cap;
        cap.inp_audio  = tmp.has_audio;
        cap.inp_vision = tmp.has_vision;
        cap.gen_audio  = tmp.has_gen_audio;
        return cap;
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, fname, e.what());
        return mtmd_caps{ false, false, false };
    }
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context * ctx, clip_ctx * ctx_clip, clip_image_f32 & image) {
    clip_set_debug_output_embeddings(ctx_clip, true);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    int n_tokens = clip_n_output_tokens(ctx_clip, &image);
    std::vector<float> embd_output(n_tokens * n_mmproj_embd, 0.0f);
    bool ok = clip_image_encode(
        ctx_clip,
        ctx->n_threads,
        &image,
        embd_output);
    if (!ok) {
        LOG_ERR("%s: failed to encode image\n", __func__);
    }
}

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    const int img_sz = (int)image.size();
    std::vector<float> img_buf;
    img_buf.reserve(img_sz * img_sz);
    for (const auto & row : image) {
        img_buf.insert(img_buf.end(), row.begin(), row.end());
    }
    clip_image_f32 inp_image;
    inp_image.set_size({img_sz, img_sz}, false, false);
    inp_image.cpy_buf(img_buf);
    LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, img_sz, img_sz);
    mtmd_debug_encode_impl(ctx, ctx->ctx_v, inp_image);
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    int n_mel = clip_get_hparams(ctx->ctx_a)->n_mel_bins;
    const int audio_nx = (int)input.size();
    std::vector<float> audio_buf(audio_nx * n_mel);
    for (int i = 0; i < audio_nx; i++) {
        for (int j = 0; j < n_mel; j++) {
            audio_buf[j * audio_nx + i] = input[i];
        }
    }
    clip_image_f32 inp_audio;
    inp_audio.set_size({audio_nx, n_mel}, false, true);
    inp_audio.cpy_buf(audio_buf);
    LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, audio_nx, n_mel);
    mtmd_debug_encode_impl(ctx, ctx->ctx_a, inp_audio);
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_u8 img_u8;
    img_u8.set_size({nx, ny}, false);
    img_u8.cpy_buf(rgb_values);
    GGML_ASSERT(ctx->image_preproc != nullptr);
    mtmd_image_preproc_out preproc_out = ctx->image_preproc->preprocess(img_u8);

    clip_image_f32_batch batch_f32;
    batch_f32.is_audio = false;
    for (auto & entry : preproc_out.entries) {
        batch_f32.entries.push_back(std::move(entry));
    }

    LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)batch_f32.entries.size());
    for (size_t i = 0; i < batch_f32.entries.size(); i++) {
        LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, batch_f32.entries[i].nx(), batch_f32.entries[i].ny());
        // TODO: better way to dump entry content?
    }
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    if (!ctx->ctx_a || !ctx->audio_preproc) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    std::vector<mtmd_audio_mel> mel_spec_chunks;
    bool ok = ctx->audio_preproc->preprocess(samples.data(), samples.size(), mel_spec_chunks);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess audio\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, mel_spec_chunks.size());
    for (size_t i = 0; i < mel_spec_chunks.size(); i++) {
        LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel_spec_chunks[i].n_len, mel_spec_chunks[i].n_mel);

        // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
        const auto & mel = mel_spec_chunks[i];
        for (int m = 0; m < mel.n_mel; m++) {
            for (int t = 0; t < mel.n_len; t++) {
                LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", i, m, t, mel.data[m * mel.n_len + t]);
            }
        }
    }
}

static void stub_log_callback(enum ggml_log_level, const char *, void *) {
    // do nothing
}

mtmd_memory_usage mtmd_get_memory_usage(const char * mmproj_fname,
                                        struct mtmd_context_params ctx_params) {
    mtmd::context_ptr ctx;
    auto saved_log_callback = g_logger_state.log_callback;
    auto saved_log_user_data = g_logger_state.log_callback_user_data;

    ctx_params.progress_callback = nullptr;

    try {
        mtmd_log_set(stub_log_callback, nullptr); // suppress logging
        ctx.reset(new mtmd_context(mmproj_fname, nullptr, ctx_params, true));
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        std::map<ggml_backend_dev_t, size_t> total_mem;
        auto merge = [&](const struct clip_ctx * c) {
            for (auto & [dev, size] : clip_get_mem_usage(c)) {
                total_mem[dev] += size;
            }
        };
        if (ctx->ctx_v) {
            merge(ctx->ctx_v);
        }
        if (ctx->ctx_a) {
            merge(ctx->ctx_a);
        }
        if (ctx->ctx_gen_a) {
            merge(ctx->ctx_gen_a);
        }
        mtmd_memory_usage res;
        res.backend_mem_usage = std::move(total_mem);
        res.image_max_tokens  = ctx->ctx_v ? clip_get_image_max_tokens(ctx->ctx_v) : -1;
        res.use_non_causal    = ctx->ctx_v ? mtmd_decode_use_non_causal(ctx.get(), nullptr) : false;
        return res;
    } catch (const std::exception & e) {
        mtmd_log_set(saved_log_callback, saved_log_user_data); // restore log callback
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return {{}, -1, false};
    }
}

//
// realtime audio input
//

// the model-independent part of a stream, the model-specific part is in the subclasses below
struct mtmd_audio_stream {
    mtmd_context * ctx;
    clip_ctx * ctx_clip;

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
    int64_t n_rows_done = 0;    // audio rows whose text has been sampled, see mtmd_audio_stream_get_n_rows()
    int64_t n_rows_flushed = 0; // audio rows up to the end of the last flush
    bool pending_flush = false;
    bool ended = false;
    bool seq_start = false;     // the rows of the last step start a new sequence

    std::vector<llama_token> banned; // never sampled from the rows

    virtual ~mtmd_audio_stream() = default;

    // model constants and initial state, false on failure
    virtual bool    init(const mtmd_audio_stream_params & params) = 0;
    virtual int32_t flush(bool end) = 0;
    virtual int32_t step(llama_token last_token) = 0;

    void add_silence(int64_t n) {
        buf.insert(buf.end(), n, 0.0f);
        n_samples += n;
    }
};

// Voxtral Realtime: the model writes text while it listens, a row is an audio embedding plus the embedding
// of the token sampled from the previous row
struct mtmd_audio_stream_voxtral_rt : mtmd_audio_stream {
    clip_enc_state * enc_state = nullptr;

    int n_fft;
    int n_frames_max;    // encoder frames per clip_encode() call
    int n_right_pad;     // silence tokens added by flush()
    float log_mel_max;   // global_log_mel_max in params.json

    std::vector<llama_token> prefix; // text tokens paired with the first rows

    mtmd_audio_cache cache;

    std::vector<float> mel_ctx; // last 2 mel frames, the left context of the conv stem
    std::vector<float> tok_embd;

    ~mtmd_audio_stream_voxtral_rt() override {
        clip_enc_state_free(enc_state);
    }

    bool    init(const mtmd_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;
};

// LFM3-Audio: the rows are plain audio embeddings of a chat turn, the model answers after the audio
// live ASR (with a prompt), trained layout: <prompt> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|>
// training samples are max 30 s: restart the sequence at a silent chunk after 15 s, always after 30 s; the encoder keeps its state
struct mtmd_audio_stream_lfm3a : mtmd_audio_stream {
    int win; // samples per mel frame
    clip_enc_state * enc_state = nullptr;

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

    ~mtmd_audio_stream_lfm3a() override {
        clip_enc_state_free(enc_state);
    }

    bool    init(const mtmd_audio_stream_params & params) override;
    int32_t flush(bool end) override;
    int32_t step(llama_token last_token) override;

    bool    add_tokens(const llama_token * tokens, size_t n);
    int32_t step_live(llama_token last_token);
};

bool mtmd_support_audio_stream(const mtmd_context * ctx) {
    if (!ctx->ctx_a || !ctx->text_model) {
        return false;
    }
    const projector_type proj = clip_get_projector_type(ctx->ctx_a);
    return proj == PROJECTOR_TYPE_VOXTRAL_RT || proj == PROJECTOR_TYPE_LFM3A;
}

mtmd_audio_stream_params mtmd_audio_stream_params_default(void) {
    mtmd_audio_stream_params params;
    params.step_ms  = 320;
    params.prompt   = nullptr;
    params.n_prompt = 0;
    return params;
}

mtmd_audio_stream * mtmd_audio_stream_init(mtmd_context * ctx, mtmd_audio_stream_params params) {
    if (!mtmd_support_audio_stream(ctx)) {
        LOG_ERR("%s: the model does not support streaming audio input\n", __func__);
        return nullptr;
    }
    const clip_hparams * hparams = clip_get_hparams(ctx->ctx_a);

    std::unique_ptr<mtmd_audio_stream> s;
    switch (clip_get_projector_type(ctx->ctx_a)) {
        case PROJECTOR_TYPE_VOXTRAL_RT: s = std::make_unique<mtmd_audio_stream_voxtral_rt>(); break;
        case PROJECTOR_TYPE_LFM3A:      s = std::make_unique<mtmd_audio_stream_lfm3a>();      break;
        default:                        GGML_ABORT("unsupported projector");
    }
    s->ctx      = ctx;
    s->ctx_clip = ctx->ctx_a;
    s->n_mel    = hparams->n_mel_bins;
    s->hop      = hparams->audio_hop_len;
    s->n_embd   = clip_n_mmproj_embd(ctx->ctx_a);
    if (s->n_embd != ctx->n_embd_text) {
        LOG_ERR("%s: mmproj embedding size %d does not match the text model (%d)\n", __func__, s->n_embd, ctx->n_embd_text);
        return nullptr;
    }
    if (!s->init(params)) {
        return nullptr;
    }
    return s.release();
}

void mtmd_audio_stream_free(mtmd_audio_stream * stream) {
    delete stream;
}

int32_t mtmd_audio_stream_push(mtmd_audio_stream * s, const float * samples, size_t n_samples) {
    if (s->ended) {
        return 1;
    }
    s->buf.insert(s->buf.end(), samples, samples + n_samples);
    s->n_samples += n_samples;
    return 0;
}

int32_t mtmd_audio_stream_flush(mtmd_audio_stream * s, bool end) {
    if (s->ended) {
        return 1;
    }
    return s->flush(end);
}

int32_t mtmd_audio_stream_step(mtmd_audio_stream * s, llama_token last_token) {
    return s->step(last_token);
}

const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream * s) {
    return s->out.data();
}

bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream * s) {
    return s->seq_start;
}

const llama_token * mtmd_audio_stream_get_banned_tokens(const mtmd_audio_stream * s, size_t * n_tokens) {
    *n_tokens = s->banned.size();
    return s->banned.data();
}

//
// Voxtral Realtime
//

bool mtmd_audio_stream_voxtral_rt::init(const mtmd_audio_stream_params & params) {
    const clip_hparams * hparams = clip_get_hparams(ctx_clip);
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

    enc_state = clip_enc_state_init(ctx_clip, n_frames_max);
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

int32_t mtmd_audio_stream_voxtral_rt::flush(bool end) {
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
static void mtmd_audio_stream_compute_mel(mtmd_audio_stream_voxtral_rt * s, bool force) {
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
        mtmd_audio_log_mel_frame(s->cache, window.data(), s->n_fft, s->n_mel, frame.data());
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

static bool mtmd_audio_stream_encode(mtmd_audio_stream_voxtral_rt * s, bool force) {
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
        clip_encode_params params;
        params.n_threads = s->ctx->n_threads;
        params.imgs      = &batch;
        params.out_embd  = &res;
        params.enc_state = s->enc_state;
        if (!clip_encode(s->ctx_clip, &params)) {
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
static bool mtmd_audio_stream_add_rows(mtmd_audio_stream_voxtral_rt * s, const llama_token * tokens, size_t n) {
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

int32_t mtmd_audio_stream_voxtral_rt::step(llama_token last_token) {
    const bool force = pending_flush;
    pending_flush = false;

    // last_token was sampled from the last row, the text of every row so far is out
    n_rows_done = n_rows;

    mtmd_audio_stream_compute_mel(this, force);
    if (!mtmd_audio_stream_encode(this, force)) {
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
        if (!mtmd_audio_stream_add_rows(this, prefix.data(), prefix.size())) {
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
    if (!mtmd_audio_stream_add_rows(this, &last_token, 1)) {
        return -2;
    }
    return 1;
}

//
// LFM3-Audio
//

bool mtmd_audio_stream_lfm3a::init(const mtmd_audio_stream_params & params) {
    const clip_hparams * hparams = clip_get_hparams(ctx_clip);

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
        // only <|text_end|> ends a chunk, the stream adds the <|mixed_end|> of the final chunk
        for (const char * piece : {"<|mixed_end|>", "<|mixed_start|>", "<|audio_start|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
            banned.push_back(find_token(piece));
        }
        if (tok_text_end == LLAMA_TOKEN_NULL || std::find(banned.begin(), banned.end(), LLAMA_TOKEN_NULL) != banned.end()) {
            LOG_ERR("%s: the vocab lacks the tokens of LFM3-Audio live ASR\n", __func__);
            return false;
        }
        soft_seq_rows = std::lround(15.0 * hparams->audio_sample_rate / samples_per_tok);
        max_seq_rows  = std::lround(30.0 * hparams->audio_sample_rate / samples_per_tok);
    }

    enc_state = clip_enc_state_init(ctx_clip, n_step_mel);
    if (!enc_state) {
        return false;
    }

    // the markers the tokenizer puts around an audio clip go around the stream as well
    const llama_vocab * vocab = llama_model_get_vocab(ctx->text_model);
    auto marker_rows = [&](const std::string & text, std::vector<float> & rows) {
        if (text.empty()) {
            return true;
        }
        std::vector<llama_token> tokens(text.size() + 4);
        const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, true);
        if (n <= 0) {
            return false;
        }
        rows.resize((size_t) n * n_embd);
        return llama_model_get_tok_embd_rows(ctx->text_model, tokens.data(), n, rows.data()) != 0;
    };
    return marker_rows(ctx->aud_beg, beg_rows) && marker_rows(ctx->aud_end, end_rows);
}

// end: pad the tail like mtmd_audio_preprocessor_lfm3a, so the rows match encoding the clip at once
// otherwise only whole rows are encoded, a padded partial row would change the encoder state
// live ASR: pad the last chunk with silence, so its text comes out
int32_t mtmd_audio_stream_lfm3a::flush(bool end) {
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
static bool mtmd_audio_stream_compute_mel(mtmd_audio_stream_lfm3a * s) {
    const int64_t n_total = s->n_samples >= s->win ? (s->n_samples - s->win) / s->hop + 1 : 0;
    const int64_t n_new   = n_total - s->n_frames;
    if (n_new <= 0) {
        return true;
    }
    const int64_t start = s->n_frames * s->hop - s->buf_pos;
    const size_t  len   = (size_t) (s->win + (n_new - 1) * s->hop);
    std::vector<mtmd_audio_mel> mel_chunks;
    if (!s->ctx->audio_preproc->preprocess(s->buf.data() + start, len, mel_chunks) || mel_chunks.size() != 1) {
        LOG_ERR("%s: unable to preprocess audio\n", __func__);
        return false;
    }
    const auto & mel = mel_chunks[0];
    GGML_ASSERT(mel.n_len == n_new && mel.n_mel == s->n_mel);
    for (int64_t t = 0; t < n_new; t++) {
        for (int m = 0; m < s->n_mel; m++) {
            s->mel.push_back(mel.data[(size_t) m * mel.n_len + t]);
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
static bool mtmd_audio_stream_encode(mtmd_audio_stream_lfm3a * s, bool flush) {
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
    const int n_rows = clip_n_output_tokens(s->ctx_clip, &img);

    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(img));

    std::vector<float> res((size_t) s->n_embd * n_rows);
    clip_encode_params params;
    params.n_threads = s->ctx->n_threads;
    params.imgs      = &batch;
    params.out_embd  = &res;
    params.enc_state = s->enc_state;
    if (!clip_encode(s->ctx_clip, &params)) {
        LOG_ERR("%s: failed to encode audio\n", __func__);
        return false;
    }
    s->embd.insert(s->embd.end(), res.begin(), res.end());
    s->mel.erase(s->mel.begin(), s->mel.begin() + (size_t) n * s->n_mel);
    return true;
}

int32_t mtmd_audio_stream_lfm3a::step(llama_token last_token) {
    const bool flush = pending_flush;
    pending_flush = false;

    if (!mtmd_audio_stream_compute_mel(this) || !mtmd_audio_stream_encode(this, flush)) {
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

bool mtmd_audio_stream_lfm3a::add_tokens(const llama_token * tokens, size_t n) {
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
int32_t mtmd_audio_stream_lfm3a::step_live(llama_token last_token) {
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

int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream * s) {
    return s->n_rows_done;
}

int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream * s) {
    return s->n_rows_flushed;
}

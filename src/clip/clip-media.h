#pragma once

// libllama internal: the media layout of the clip engine, moved from mtmd (mtmd.cpp)
// the chunks a media becomes in the decoder prompt, their preprocessed input, their decoder positions

#include "clip-impl.h"
#include "clip-model.h"
#include "clip-image.h"
#include "clip-audio.h"

#include "llama.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

std::string clip_format(const char * fmt, ...);

// remember to bump this if the serialization format changes
#define CLIP_SERIALIZATION_VERSION 5

// oldest compat version that can be loaded
#define CLIP_SERIALIZATION_VERSION_MIN 2

struct clip_serialization {
    // note: using 64-bit here for future-proofing
    uint64_t version = CLIP_SERIALIZATION_VERSION;
    std::vector<char> data;
    size_t read_pos = 0; // cursor used when reading

    // for writing
    clip_serialization(uint64_t version) : version(version) {
        write(version);
    }

    // for reading
    clip_serialization(uint64_t version, const char * buf, size_t len) {
        // copy buf to data
        data.assign(buf, buf + len);
        uint64_t ver_in = read<uint64_t>();
        if (ver_in < CLIP_SERIALIZATION_VERSION_MIN || ver_in > version) {
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
inline void clip_serialization::write<bool>(bool value) {
    write<uint8_t>(value ? 1 : 0);
}
template <>
inline bool clip_serialization::read<bool>() {
    return read<uint8_t>() != 0;
}

template <>
inline void clip_serialization::write<std::string>(std::string value) {
    write<uint64_t>(value.size());
    data.insert(data.end(), value.begin(), value.end());
}
template <>
inline std::string clip_serialization::read<std::string>() {
    uint64_t len = read<uint64_t>();
    if (read_pos + len > data.size()) {
        throw std::runtime_error("read_string OOB");
    }
    std::string str(data.data() + read_pos, len);
    read_pos += len;
    return str;
}



// position indexing for decoder model
enum clip_pos_type {
    CLIP_POS_TYPE_NORMAL,    // number of positions equals to number of tokens
    CLIP_POS_TYPE_MROPE,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    CLIP_POS_TYPE_HUNYUANVL, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
    CLIP_POS_TYPE_SENSENOVA_U1, // U1 uses zero-based H/W and one temporal position per image
    CLIP_POS_TYPE_CANVAS,    // MiniCPM-V 4.7: overview + slices in one chunk, sharing one 2D canvas (see clip_image_tokens::canvas_tile_grid)
    CLIP_POS_TYPE_COUNT,     // for validation
};

struct clip_image_tokens {
    uint32_t nx = 0; // number of tokens in x direction
    uint32_t ny = 0; // number of tokens in y direction
    clip_pos_type pos = CLIP_POS_TYPE_NORMAL;
    uint32_t image_idx = 0; // 0-based position of this image among image chunks in the prompt(used by pos == CLIP_POS_TYPE_HUNYUANVL)
    uint32_t n_temporal_merge = 1; // for qwen-vl style temporal merge
    uint32_t n_tokens() const {
        if (pos == CLIP_POS_TYPE_HUNYUANVL) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (nx + 1) * ny + 2;
        }
        if (pos == CLIP_POS_TYPE_CANVAS) {
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

    // CLIP_POS_TYPE_CANVAS: entries are [overview, slices row by row], nx/ny is the token grid of the last entry
    // returns the token grid (w, h) of entry k, scaled from its pixel size
    std::pair<uint32_t, uint32_t> canvas_tile_grid(size_t k) const {
        const auto & ref = batch_f32.entries.back();
        const auto & e   = batch_f32.entries[k];
        return { (uint32_t) e.nx() * nx / ref.nx(), (uint32_t) e.ny() * ny / ref.ny() };
    }

    bool can_batch_with(const clip_image_tokens & other) {
        // a canvas chunk holds a whole image group, its layout is not given by nx/ny alone
        return nx == other.nx && ny == other.ny && pos == other.pos && pos != CLIP_POS_TYPE_CANVAS;
    }

    clip_image_tokens clone() {
        return clip_image_tokens{
            nx,
            ny,
            pos,
            image_idx,
            n_temporal_merge,
            batch_f32.clone(),
            id,
        };
    }

    void serialize(clip_serialization & ser) const {
        // remember to bump CLIP_SERIALIZATION_VERSION if this is changed
        ser.write(nx);
        ser.write(ny);
        ser.write((uint32_t)pos);
        ser.write(image_idx);
        ser.write(n_temporal_merge);
        ser.write(id);
        batch_f32.serialize(ser);
    }
    void deserialize(clip_serialization & ser) {
        nx = ser.read<uint32_t>();
        ny = ser.read<uint32_t>();
        uint32_t pos_raw = ser.read<uint32_t>();
        if (pos_raw >= CLIP_POS_TYPE_COUNT) {
            throw std::runtime_error("invalid pos type");
        }
        pos = (clip_pos_type)pos_raw;
        image_idx = ser.read<uint32_t>();
        n_temporal_merge = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
    }
};
using clip_image_tokens_ptr = std::unique_ptr<clip_image_tokens>;

// the mel of a clip is computed by its first encode, its chunks share the result
using clip_audio_source_ptr = std::shared_ptr<clip_audio_preproc_out>;

// an audio track split across chunks (e.g. the soundtrack of a video): it is encoded once, as one clip, and every
// chunk takes its rows of it
struct clip_audio_track {
    clip_image_f32_batch batch_f32; // the mel of the whole track
    clip_audio_source_ptr source;
    uint32_t n_tokens = 0;
    std::vector<float> embd;        // filled by the first encode
    std::shared_ptr<const clip_gen_ref> gen_ref; // gen-side reference of a predefined voice, if any
};

struct clip_audio_tokens {
    uint32_t n_tokens = 0; // number of tokens
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    clip_audio_source_ptr source; // if set, batch_f32 has the shape of the mel of chunk source_idx
    uint32_t source_idx = 0;

    // a slice of a track: the rows [row_begin, row_begin + n_tokens) of it, batch_f32 is unused
    std::shared_ptr<clip_audio_track> track;
    uint32_t row_begin = 0;

    // true if one of entries in batch_f32 is a placeholder
    bool is_placeholder() const {
        if (source || (track && track->source)) {
            return false;
        }
        if (track && track->batch_f32.entries.empty()) {
            return track->embd.empty(); // a voice: its rows are not serialized
        }
        for (const auto & entry : (track ? track->batch_f32 : batch_f32).entries) {
            if (entry.is_placeholder()) {
                return true;
            }
        }
        return false;
    }

    clip_audio_tokens clone() {
        return clip_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id,
            source,
            source_idx,
            track,
            row_begin,
        };
    }

    void serialize(clip_serialization & ser) const {
        // remember to bump CLIP_SERIALIZATION_VERSION if this is changed
        ser.write(n_tokens);
        ser.write(id);
        batch_f32.serialize(ser);
        // a slice keeps its own copy of the track
        ser.write(track != nullptr);
        if (track) {
            ser.write(row_begin);
            ser.write(track->n_tokens);
            track->batch_f32.serialize(ser);
        }
    }
    void deserialize(clip_serialization & ser) {
        n_tokens = ser.read<uint32_t>();
        id = ser.read<std::string>();
        batch_f32.deserialize(ser);
        track.reset();
        if (ser.version >= 4 && ser.read<bool>()) {
            row_begin = ser.read<uint32_t>();
            track = std::make_shared<clip_audio_track>();
            track->n_tokens = ser.read<uint32_t>();
            track->batch_f32.deserialize(ser);
        }
        source.reset();
    }
};
using clip_audio_tokens_ptr = std::unique_ptr<clip_audio_tokens>;

enum clip_input_chunk_type {
    CLIP_INPUT_CHUNK_TYPE_TEXT,
    CLIP_INPUT_CHUNK_TYPE_IMAGE,
    CLIP_INPUT_CHUNK_TYPE_AUDIO,
    CLIP_INPUT_CHUNK_TYPE_COUNT, // for validation
};

struct clip_input_chunk {
    clip_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    clip_image_tokens_ptr tokens_image;
    clip_audio_tokens_ptr tokens_audio;

    bool can_batch_with(const clip_input_chunk & other) const {
        if (type != other.type) {
            return false;
        }

        if (tokens_image && other.tokens_image) {
            return tokens_image->can_batch_with(*other.tokens_image);
        }

        // TODO: allow batching audio chunks of the same size

        return false;
    }

    size_t n_tokens_total() const {
        switch (type) {
            case CLIP_INPUT_CHUNK_TYPE_TEXT:  return tokens_text.size();
            case CLIP_INPUT_CHUNK_TYPE_IMAGE: return tokens_image->n_tokens();
            case CLIP_INPUT_CHUNK_TYPE_AUDIO: return tokens_audio->n_tokens;
            default: return 0;
        }
    }

    bool is_placeholder() const {
        if (type == CLIP_INPUT_CHUNK_TYPE_IMAGE) {
            return tokens_image && tokens_image->is_placeholder();
        } else if (type == CLIP_INPUT_CHUNK_TYPE_AUDIO) {
            return tokens_audio && tokens_audio->is_placeholder();
        }
        return false;
    }

    void serialize(clip_serialization & ser) const {
        // remember to bump CLIP_SERIALIZATION_VERSION if this is changed
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
    void deserialize(clip_serialization & ser) {
        uint32_t type_raw = ser.read<uint32_t>();
        if (type_raw >= CLIP_INPUT_CHUNK_TYPE_COUNT) {
            throw std::runtime_error("invalid chunk type");
        }
        type = (clip_input_chunk_type)type_raw;

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
            tokens_image = std::make_unique<clip_image_tokens>();
            tokens_image->deserialize(ser);
        } else {
            tokens_image.reset();
        }

        if (ser.read<bool>()) {
            tokens_audio = std::make_unique<clip_audio_tokens>();
            tokens_audio->deserialize(ser);
        } else {
            tokens_audio.reset();
        }

        // catch buffers where the declared type doesn't match which payload is actually present,
        // so a mismatched chunk can't slip through and null-deref/abort later in an accessor
        if (type == CLIP_INPUT_CHUNK_TYPE_IMAGE && !tokens_image) {
            throw std::runtime_error("type is IMAGE but tokens_image is missing");
        }
        if (type == CLIP_INPUT_CHUNK_TYPE_AUDIO && !tokens_audio) {
            throw std::runtime_error("type is AUDIO but tokens_audio is missing");
        }
    }
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum clip_slice_tmpl {
    CLIP_SLICE_TMPL_NONE,
    CLIP_SLICE_TMPL_MINICPMV_2_5,
    CLIP_SLICE_TMPL_MINICPMV_2_6,
    CLIP_SLICE_TMPL_LLAMA4,
    CLIP_SLICE_TMPL_IDEFICS3,
    CLIP_SLICE_TMPL_LFM2,
    CLIP_SLICE_TMPL_STEP3VL,
};

// gen-side reference of a predefined voice, see clip_bitmap
struct clip_gen_ref;

// a media input of the layout: an image (RGB), a video frame, audio samples (F32) or a voice (embedding rows)
struct clip_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 0;
    std::vector<unsigned char> data; // empty for a placeholder
    std::string id;
    bool is_audio = false;
    bool is_video = false;
    bool is_voice = false;
    std::shared_ptr<const clip_gen_ref> gen_ref;

    bool is_placeholder() const { return data.empty(); }
    bool is_video_frame() const { return is_video && !is_audio; }
    const std::vector<unsigned char> & get_ro_buf() const { return data; }
};

struct clip_decoder_pos {
    uint32_t t;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

// per-projector layout of the media spans in the decoder prompt (from mtmd_context)
struct clip_layout_config {
    const clip_model * v = nullptr; // vision tower, nullptr if none
    const clip_model * a = nullptr; // audio tower, nullptr if none

    const llama_vocab * vocab = nullptr;
    clip_pos_type pos_type = CLIP_POS_TYPE_NORMAL;

    int n_threads = 1;
    ggml_backend_sched_t sched_mel = nullptr; // runs the log-mel of the audio preprocessor

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;
    std::string vid_beg; // wraps a whole video (lazy bitmap with is_video)
    std::string vid_end;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    clip_slice_tmpl slice_tmpl = CLIP_SLICE_TMPL_NONE;
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
    bool has_prompt_img_id = false;

    // Kimi-K3.1 writes the original image size after img_beg
    bool has_prompt_img_size = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::shared_ptr<clip_audio_preprocessor> audio_preproc; // shared with the media that compute their log-mel at encode time
    std::unique_ptr<clip_image_preprocessor> image_preproc;

    clip_layout_config(const clip_model * v, const clip_model * a, const llama_model * model_dec, ggml_backend_sched_t sched_mel, int n_threads);

    void init_vision();
    void init_audio();

    // the decoder attends to the span of this media bidirectionally
    bool decode_use_non_causal(bool is_audio) const;

    projector_type proj_type_v() const {
        return v ? v->proj_type : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return a ? a->proj_type : PROJECTOR_TYPE_UNKNOWN;
    }

    // [LFM3_OMNI] the audio of a video is its soundtrack, see CLIP_VIDEO_TEMPLATE_LFM3_OMNI
    bool video_has_soundtrack() const {
        return proj_type_v() == PROJECTOR_TYPE_LFM3VL && proj_type_a() == PROJECTOR_TYPE_LFM3A;
    }

    int64_t n_embd_out() const {
        if (v) {
            return v->n_mmproj_embd();
        } else if (a) {
            return a->n_mmproj_embd();
        } else {
            throw std::runtime_error("no mmproj model loaded");
        }
    }

    llama_token lookup_token(const std::string & token_text) {
        if (vocab == nullptr) {
            // TODO @ngxson : this case is currently hit by clip_get_memory_usage
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

bool clip_cbx_is_mtl(const clip_model & model);

// the media part of mtmd_tokenizer: lays out one media in the decoder prompt, appends its chunks to out
struct clip_tokenizer {
    const clip_layout_config & cfg;
    const llama_vocab * vocab;

    std::vector<clip_input_chunk> & cur_entries;
    struct {
        std::vector<clip_input_chunk> & entries;
    } cur;

    uint32_t n_images_added = 0; // 0-based index assigned to the next image chunk
    size_t   n_tokens_before = 0; // prompt tokens before the first chunk of out

    // [LFM3_OMNI] the audio pieces of a video are joined back and encoded as one clip, each piece takes its rows
    struct soundtrack {
        std::vector<const clip_bitmap *> pieces;
        std::shared_ptr<clip_audio_track> track; // built on the first piece
        size_t   n_samples  = 0; // samples of the pieces added so far
        uint32_t n_assigned = 0; // rows of the pieces added so far
    };
    std::vector<soundtrack> soundtracks;
    std::map<const clip_bitmap *, size_t> soundtrack_of; // piece -> index in soundtracks

    clip_tokenizer(const clip_layout_config & cfg, std::vector<clip_input_chunk> & out)
        : cfg(cfg), vocab(cfg.vocab), cur_entries(out), cur{out} {}

    void add_text(const std::string & txt, bool parse_special) {
        if (vocab == nullptr) {
            throw std::runtime_error("llama_vocab is not provided");
        }
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = clip_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == CLIP_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            clip_input_chunk chunk{
                CLIP_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(const std::vector<const clip_bitmap *> & bitmaps) {
        GGML_ASSERT(!bitmaps.empty());

        // note: only one type of media is supported per call, caller should enforce this
        if (bitmaps[0]->is_voice) {
            GGML_ASSERT(bitmaps.size() == 1);
            return add_voice(*bitmaps[0]);
        }
        const bool is_vision = !bitmaps[0]->is_audio;


        if (is_vision) {
            // handle image

            if (!cfg.v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!cfg.img_beg.empty()) {
                add_text(cfg.img_beg, true); // add image begin token
            }
            if (cfg.has_prompt_img_size) {
                add_text(clip_format("image %ux%u<|media_content|>", bitmaps[0]->nx, bitmaps[0]->ny), true);
            }

            // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

            clip_image_preproc_out preproc_out;

            for (const auto * bmp : bitmaps) {
                // sanity check
                GGML_ASSERT(!bmp->is_audio);
                GGML_ASSERT(cfg.image_preproc != nullptr);
                if (bmp->nx <= 0 || bmp->ny <= 0) {
                    LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n",
                            __func__, bmp->nx, bmp->ny);
                    return 2;
                }

                // convert clip_bitmap to clip_image_u8
                clip_image_u8 img_u8;
                img_u8.set_size(
                    {(int)bmp->nx, (int)bmp->ny},
                    bmp->is_placeholder());
                img_u8.cpy_buf(bmp->get_ro_buf());

                // preprocess image
                clip_image_preproc_out tmp_preproc_out = bmp->is_video_frame()
                    ? cfg.image_preproc->preprocess_video_frame(img_u8)
                    : cfg.image_preproc->preprocess(img_u8);

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

            if (has_tiling_grid && cfg.proj_type_v() == PROJECTOR_TYPE_MINICPMV4_7) {
                GGML_ASSERT(bitmaps.size() == 1);
                if (cfg.has_prompt_img_id) {
                    add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true);
                }
                add_text(cfg.tok_ov_img_start);
                // the separators after <image> are appended by the graph, see add_canvas_chunk()
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
                if (cfg.ov_img_first) {
                    if (cfg.has_prompt_img_id) {
                        add_text("<image_id>" + std::to_string(n_images_added) + "</image_id>", true);
                    }
                    add_text(cfg.tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(cfg.tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    LOG_DBG("%s: adding %d slices (%d rows x %d cols)\n", __func__, (int)chunks.size(), n_row, n_col);
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(cfg.tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!cfg.tok_sli_img_start.empty()) {
                                add_text(cfg.tok_sli_img_start);
                            } else if (!cfg.sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, cfg.sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, cfg.sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }

                            auto & curr_chunk = chunks[y * n_col + x];
                            auto & curr_batch = curr_chunk.tokens_image->batch_f32;
                            if (curr_batch.entries.size() != 1) {
                                throw std::runtime_error(clip_format("%s: expect 1 image in batch_f32", __func__));
                            }

                            LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, y, x);
                            cur.entries.emplace_back(std::move(curr_chunk));

                            add_text(cfg.tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(cfg.tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || cfg.tok_row_end_trail)) {
                            add_text(cfg.tok_row_end);
                        }
                    }
                    add_text(cfg.tok_slices_end);
                }

                // add overview image (last)
                if (!cfg.ov_img_first) {
                    add_text(cfg.tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(cfg.tok_ov_img_end);
                }
            } else {

                if (preproc_out.entries.size() == 0) {
                    LOG_ERR("%s: no image tokens produced by preprocessor (ref: https://github.com/ggml-org/llama.cpp/pull/24769)\n", __func__);
                    return 2;
                }

                if (cfg.proj_type_v() == PROJECTOR_TYPE_DEEPSEEK4V) {
                    // the text model perceives input in blocks of N tokens (N = COMPRESS_PAD_TO = 4, same as the CSA compress ratio)
                    // image need to be aligned to block size, while adding IMAGE_PAD embeddings to the beginning
                    // TODO @ngxson : maybe refactor this in the future
                    constexpr int32_t align = 4;
                    size_t n_past = n_tokens_before;
                    for (const auto & e : cur.entries) {
                        n_past += e.n_tokens_total();
                    }
                    preproc_out.entries[0].lead_pad = align - 1 - (int32_t)(n_past % align);
                }

                size_t n_tokens = 0;
                for (auto & e : preproc_out.entries) {
                    n_tokens += cfg.v->n_output_tokens(&e);
                    if (cfg.v->n_temporal_merge() == 2) {
                        // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
                        break;
                    }
                }

                clip_image_tokens_ptr image_tokens(new clip_image_tokens);

                // [QWEN_VIDEO] improve this in the future
                image_tokens->n_temporal_merge = cfg.v->n_temporal_merge();

                if ((cfg.pos_type == CLIP_POS_TYPE_MROPE)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = cfg.v->n_output_tokens_x(&preproc_out.entries[0]);
                    image_tokens->ny = cfg.v->n_output_tokens_y(&preproc_out.entries[0]);
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->pos = cfg.pos_type;
                // U1 uses M-RoPE, but unlike Qwen its image H/W coordinates are
                // zero-based and the whole image occupies one temporal position.
                if (cfg.proj_type_v() == PROJECTOR_TYPE_SENSENOVA_U1) {
                    image_tokens->pos = CLIP_POS_TYPE_SENSENOVA_U1;
                }
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index). Override the position type so that
                // n_tokens() and clip_image_tokens_get_decoder_pos pick the HunyuanVL layout.
                else if (cfg.proj_type_v() == PROJECTOR_TYPE_HUNYUANVL) {
                    image_tokens->pos       = CLIP_POS_TYPE_HUNYUANVL;
                    image_tokens->image_idx = n_images_added;
                    GGML_ASSERT(n_tokens == (size_t)image_tokens->n_tokens());
                }

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = false;
                batch_f32.is_video = bitmaps[0]->is_video_frame();
                batch_f32.entries = std::move(preproc_out.entries);
                // do NOT use preproc_out from this point on, it's moved

                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmaps[0]->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                clip_input_chunk chunk{
                    CLIP_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!cfg.img_end.empty()) {
                add_text(cfg.img_end, true); // add image end token
            }

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            n_images_added++;

        } else {
            // handle audio

            GGML_ASSERT(bitmaps.size() == 1); // no batching support for now
            auto & bitmap = bitmaps[0];

            if (!cfg.a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (auto it = soundtrack_of.find(bitmap); it != soundtrack_of.end()) {
                return add_soundtrack_piece(soundtracks[it->second], *bitmap);
            }

            if (bitmap->nx == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!cfg.aud_beg.empty()) {
                add_text(cfg.aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(cfg.audio_preproc != nullptr);

            // preprocess audio
            auto source = std::make_shared<clip_audio_preproc_out>();
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
                if (!cfg.audio_preproc) {
                    LOG_ERR("%s: this model takes audio input only via clip_audio_stream\n", __func__);
                    return 2;
                }
                *source = cfg.audio_preproc->preprocess(samples, n_samples);
                if (source->entries.empty()) {
                    LOG_ERR("Unable to preprocess audio\n");
                    return 2;
                }
            }

            // one chunk for the five entries of a chatterbox clip, one token per talker conditioning row
            if (cfg.a->proj_type == PROJECTOR_TYPE_CHATTERBOX_SPKENC) {
                GGML_ASSERT(source->entries.size() == 5);
                // mtl: spkr, perceiver query rows, emotion; turbo: spkr, t3 speech tokens
                const size_t n_query  = cfg.a->cbx_read_tensor("a.cenc.perceiver.pre_attention_query", nullptr, 0) / cfg.n_embd_out();
                const size_t n_tokens = clip_cbx_is_mtl(*cfg.a) ? 2 + n_query : 1 + (size_t) source->entries[2].nx() / 4;

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries  = std::move(source->entries);

                clip_audio_tokens_ptr audio_tokens(new clip_audio_tokens);
                audio_tokens->n_tokens  = (uint32_t) n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id        = bitmap->id;

                clip_input_chunk chunk{
                    CLIP_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            } else
            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (size_t i = 0; i < source->entries.size(); i++) {
                clip_image_f32 mel_f32 = source->entries[i];

                size_t n_tokens = cfg.a->n_output_tokens(&mel_f32);

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                clip_audio_tokens_ptr audio_tokens(new clip_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional
                if (source->needs_compute()) {
                    audio_tokens->source = source;
                    audio_tokens->source_idx = (uint32_t) i;
                }

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                clip_input_chunk chunk{
                    CLIP_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!cfg.aud_end.empty()) {
                add_text(cfg.aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    int32_t add_voice(const clip_bitmap & bitmap) {
        if (bitmap.ny == 0) {
            LOG_ERR("%s: error: voice \"%s\" holds no embeddings\n", __func__, bitmap.id.c_str());
            return 2;
        }
        auto track = std::make_shared<clip_audio_track>();
        track->n_tokens = bitmap.ny;
        const auto & buf = bitmap.get_ro_buf();
        track->embd.assign((const float *) buf.data(), (const float *) (buf.data() + buf.size()));
        track->gen_ref = bitmap.gen_ref;

        clip_audio_tokens_ptr audio_tokens(new clip_audio_tokens);
        audio_tokens->n_tokens = bitmap.ny;
        audio_tokens->id       = bitmap.id;
        audio_tokens->track    = std::move(track);

        cur.entries.emplace_back(clip_input_chunk{
            CLIP_INPUT_CHUNK_TYPE_AUDIO,
            {}, // text tokens
            nullptr, // image tokens
            std::move(audio_tokens),
        });
        return 0;
    }

    int32_t add_soundtrack_piece(soundtrack & st, const clip_bitmap & bitmap) {
        if (!st.track) {
            std::vector<float> samples;
            for (const auto * piece : st.pieces) {
                if (piece->is_placeholder()) {
                    samples.resize(samples.size() + piece->nx, 0.0f);
                } else {
                    const float * data = (const float *) piece->get_ro_buf().data();
                    samples.insert(samples.end(), data, data + piece->nx);
                }
            }
            if (!cfg.audio_preproc) {
                LOG_ERR("%s: unable to preprocess the soundtrack\n", __func__);
                return 2;
            }
            auto source = std::make_shared<clip_audio_preproc_out>(cfg.audio_preproc->preprocess(samples.data(), samples.size()));
            if (source->entries.size() != 1) {
                LOG_ERR("%s: unable to preprocess the soundtrack\n", __func__);
                return 2;
            }
            clip_image_f32 mel_f32 = source->entries[0];
            st.track = std::make_shared<clip_audio_track>();
            st.track->n_tokens = cfg.a->n_output_tokens(&mel_f32);
            if (source->needs_compute()) {
                st.track->source = source;
            }
            st.track->batch_f32.is_audio = true;
            st.track->batch_f32.entries.push_back(std::move(mel_f32));
        }

        // rows of [begin, end): round(sample / samples per row), the last piece takes the rest of the track
        const double samples_per_row = 8.0 * cfg.a->hparams.audio_hop_len; // one row per 8 mel frames
        const uint32_t n_tokens = st.track->n_tokens;
        st.n_samples += bitmap.nx;
        const uint32_t begin = st.n_assigned;
        const uint32_t end   = &bitmap == st.pieces.back() ? n_tokens
                             : std::clamp((uint32_t) std::nearbyint(st.n_samples / samples_per_row), begin, n_tokens);
        st.n_assigned = end;
        if (end == begin) {
            return 0;
        }

        clip_audio_tokens_ptr audio_tokens(new clip_audio_tokens);
        audio_tokens->n_tokens  = end - begin;
        audio_tokens->track     = st.track;
        audio_tokens->row_begin = begin;
        audio_tokens->id        = bitmap.id;
        cur.entries.emplace_back(clip_input_chunk{
            CLIP_INPUT_CHUNK_TYPE_AUDIO,
            {}, // text tokens
            nullptr, // image tokens
            std::move(audio_tokens),
        });
        return 0;
    }

    void add_canvas_chunk(clip_image_preproc_out && preproc_out, const std::string & id) {
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

        clip_image_tokens_ptr image_tokens(new clip_image_tokens);
        image_tokens->pos = CLIP_POS_TYPE_CANVAS;
        image_tokens->id  = id;
        auto & entries = image_tokens->batch_f32.entries;
        entries.push_back(std::move(ov));
        for (auto & slice : slices) {
            entries.push_back(std::move(slice));
        }
        // token grid of the last entry, the grids of the other entries are scaled from it
        image_tokens->nx = cfg.v->n_output_tokens_x(&entries.back());
        image_tokens->ny = cfg.v->n_output_tokens_y(&entries.back());

        size_t n_tokens = 0;
        for (const auto & entry : entries) {
            n_tokens += cfg.v->n_output_tokens(&entry);
        }
        GGML_ASSERT(n_tokens == image_tokens->n_tokens());

        clip_input_chunk chunk{
            CLIP_INPUT_CHUNK_TYPE_IMAGE,
            {}, // text tokens
            std::move(image_tokens),
            nullptr, // audio tokens
        };
        cur.entries.emplace_back(std::move(chunk));
    }

    std::vector<clip_input_chunk> split_batch_to_chunk(clip_image_preproc_out && preproc_out, const std::string & id) {
        std::vector<clip_input_chunk> chunks;

        auto process_chunk = [&](clip_image_f32 && img) {
            clip_image_tokens_ptr image_tokens(new clip_image_tokens);
            image_tokens->nx = cfg.v->n_output_tokens(&img);
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(img));
            image_tokens->id = id;

            GGML_ASSERT(image_tokens->nx > 0);

            clip_input_chunk chunk{
                CLIP_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        };

        // overview image first
        auto & overview = preproc_out.overview;
        if (overview.nx() == 0 || overview.ny() == 0) {
            throw std::runtime_error(clip_format("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
        }
        process_chunk(std::move(preproc_out.overview));

        // then, process slices
        for (auto & entry : preproc_out.entries) {
            if (entry.nx() == 0 || entry.ny() == 0) {
                throw std::runtime_error(clip_format("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
            }
            process_chunk(std::move(entry));
        }

        return chunks;
    }

    static std::vector<llama_token> clip_tokenize_text_internal(
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

// decoder positions of the rows of an image chunk
clip_decoder_pos clip_image_tokens_get_decoder_pos(const clip_image_tokens * image_tokens, llama_pos pos_0, size_t i);
llama_pos clip_image_tokens_get_n_pos(const clip_image_tokens * image_tokens);

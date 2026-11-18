// mtmd: multimodal input and audio generation on libllama (llama-ext.h)
// the models run in libllama; this library keeps the chunk API, the media markers in the prompt and the forwarders

#include "mtmd.h"
#include "mtmd-internal.h"
#include "debug/mtmd-debug.h"

#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <random>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

//
// logging
//

static void mtmd_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct mtmd_logger_state {
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

static mtmd_logger_state g_logger_state = {mtmd_log_callback_default, NULL};

static void mtmd_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[256];
    const int len = vsnprintf(buffer, sizeof(buffer), format, args);
    if (len < (int) sizeof(buffer)) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        std::vector<char> buffer2(len + 1);
        vsnprintf(buffer2.data(), len + 1, format, args_copy);
        g_logger_state.log_callback(level, buffer2.data(), g_logger_state.log_callback_user_data);
    }
    va_end(args_copy);
    va_end(args);
}

#define LOG_DBG(...) mtmd_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) mtmd_log_internal(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) mtmd_log_internal(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) mtmd_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

static std::string mtmd_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    const int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// serialization of chunks (counts and ids, the media data is not kept)
//

// remember to bump this if the serialization format changes
// 6: counts and ids; 7: and the decoder positions of image rows
#define MTMD_SERIALIZATION_VERSION 7

// oldest compat version that can be loaded (2 to 5: the format of the old engine, read for their counts)
#define MTMD_SERIALIZATION_VERSION_MIN 2

struct mtmd_serialization {
    uint64_t version = MTMD_SERIALIZATION_VERSION;
    std::vector<char> data;
    size_t read_pos = 0;

    explicit mtmd_serialization(uint64_t version) : version(version) {
        write(version);
    }

    mtmd_serialization(uint64_t version, const char * buf, size_t len) {
        data.assign(buf, buf + len);
        const uint64_t ver_in = read<uint64_t>();
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
    const uint64_t len = read<uint64_t>();
    if (read_pos + len > data.size()) {
        throw std::runtime_error("read_string OOB");
    }
    std::string str(data.data() + read_pos, len);
    read_pos += len;
    return str;
}

//
// bitmaps
//

// for still image data, layout is RGBRGBRGB...; length of data must be nx * ny * 3 bytes
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
struct mtmd_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 0;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio  = false;
    bool is_voice  = false;
    bool mergeable = false; // [QWEN_VIDEO] set only on frames of the same video
    bool is_video  = false; // see mtmd_bitmap_set_is_video()

    // a voice: the encoded media of libllama
    std::shared_ptr<llama_media> lmedia;

    // lazy-loaded bitmap
    mtmd_bitmap_lazy_callback lazy_callback = nullptr;
    void * lazy_user_data = nullptr;

    mtmd_bitmap(const unsigned char * data, uint32_t nx, uint32_t ny)
        : nx(nx), ny(ny), is_audio(false) {
        if (data) {
            const size_t data_size = (size_t) nx * ny * 3;
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    mtmd_bitmap(const unsigned char * data, uint32_t n_samples)
        : nx(n_samples), ny(1), is_audio(true) {
        if (data) {
            const size_t data_size = (size_t) nx * sizeof(float);
            this->data.resize(data_size);
            std::memcpy(this->data.data(), data, data_size);
        }
    }

    const std::vector<unsigned char> & get_ro_buf() const {
        return data;
    }

    bool is_placeholder() const {
        return data.empty() && !lmedia;
    }

    size_t n_bytes() const {
        return data.size();
    }

    bool is_video_frame() const {
        return is_video && !is_audio;
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

//
// chunks
//

// reference to a media of libllama
struct mtmd_lmedia_ref {
    llama_media * m = nullptr;

    mtmd_lmedia_ref() = default;
    explicit mtmd_lmedia_ref(llama_media * m) : m(m) {}
    mtmd_lmedia_ref(const mtmd_lmedia_ref & o) : m(o.m ? llama_media_ref(o.m) : nullptr) {}
    mtmd_lmedia_ref(mtmd_lmedia_ref && o) noexcept : m(o.m) { o.m = nullptr; }
    mtmd_lmedia_ref & operator=(const mtmd_lmedia_ref & o) {
        if (this != &o) {
            reset();
            m = o.m ? llama_media_ref(o.m) : nullptr;
        }
        return *this;
    }
    mtmd_lmedia_ref & operator=(mtmd_lmedia_ref && o) noexcept {
        if (this != &o) {
            reset();
            m = o.m;
            o.m = nullptr;
        }
        return *this;
    }
    ~mtmd_lmedia_ref() { reset(); }

    void reset() {
        if (m) {
            llama_media_free(m);
            m = nullptr;
        }
    }
};

// versions 2 to 5 of the format, written by the old engine: the counts follow from the geometry of the entries
// (MiniCPM-V 4.7 canvas separators, HunyuanVL row newlines, Qwen-VL temporal merge), as the old engine computed them
struct mtmd_legacy_entry {
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t suffix_type = 0;
    int32_t grid_x = 0; // anyres grid of the image (version 3 and later)
    int32_t grid_y = 0;
};

static std::vector<mtmd_legacy_entry> mtmd_legacy_read_batch(mtmd_serialization & ser) {
    ser.read<bool>(); // is_audio
    const uint64_t n = ser.read<uint64_t>();
    if (n > (ser.data.size() - ser.read_pos) / 10) {
        throw std::runtime_error("entries count exceeds buffer size");
    }
    std::vector<mtmd_legacy_entry> entries(n);
    for (auto & e : entries) {
        ser.read<bool>();    // add_viewsep
        ser.read<bool>();    // add_newline
        ser.read<int32_t>(); // lead_pad
        if (ser.version >= 3) {
            e.suffix_type = ser.read<int32_t>();
            if (e.suffix_type < 0 || e.suffix_type > 5) {
                throw std::runtime_error("invalid suffix type");
            }
            e.grid_x = ser.read<int32_t>();
            e.grid_y = ser.read<int32_t>();
            ser.read<int32_t>(); // anyres orig_nx
            ser.read<int32_t>(); // anyres orig_ny
        }
        e.nx = ser.read<int32_t>();
        e.ny = ser.read<int32_t>();
        if (ser.version >= 5) {
            ser.read<int32_t>(); // audio_n_ctx
        }
    }
    return entries;
}

// separator rows after a MiniCPM-V 4.7 tile: </image>, </image><slice>, </slice><slice>, </slice>\n<slice>, </slice>
static size_t mtmd_legacy_suffix_rows(int32_t suffix_type) {
    static const size_t rows[] = { 0, 1, 2, 2, 3, 1 };
    return rows[suffix_type];
}

// the position types of the old engine
enum mtmd_legacy_pos { MTMD_LEGACY_POS_NORMAL, MTMD_LEGACY_POS_MROPE, MTMD_LEGACY_POS_HUNYUANVL, MTMD_LEGACY_POS_SENSENOVA_U1, MTMD_LEGACY_POS_CANVAS };

// the decoder position [t, y, x, z] of row i of an image of the old engine at pos_0, as its getter computed it
struct mtmd_legacy_layout {
    mtmd_legacy_pos pos = MTMD_LEGACY_POS_NORMAL;
    uint32_t nx = 0;
    uint32_t ny = 0;
    uint32_t n_tok = 0;
    uint32_t image_idx = 0;
    std::vector<mtmd_legacy_entry> entries;

    // canvas, set by the loader: the first row of each tile (then the row count), and the position of each tile
    // after the origin (each tile before it took one, its separators one each)
    std::vector<uint32_t> tile_row0;
    std::vector<uint32_t> tile_t0;

    // MiniCPM-V 4.7 canvas: the token grid of tile k, scaled from its pixel size (checked to fit by the loader)
    std::pair<uint64_t, uint64_t> tile_grid(size_t k) const {
        const auto & ref = entries.back();
        return { (uint64_t) entries[k].nx * nx / (uint64_t) ref.nx, (uint64_t) entries[k].ny * ny / (uint64_t) ref.ny };
    }

    static uint32_t canvas_scale(uint32_t coord, uint32_t grid, uint64_t canvas) {
        if (grid <= 1 || canvas <= 1) {
            return 0;
        }
        const double v = (double) coord * (double) (canvas - 1) / (double) (grid - 1);
        return (uint32_t) std::min({ std::nearbyint(v), (double) (canvas - 1), (double) UINT32_MAX });
    }

    std::array<llama_pos, 4> row_pos(llama_pos pos_0, uint32_t i) const {
        const llama_pos seq = pos_0 + (llama_pos) i;
        switch (pos) {
            case MTMD_LEGACY_POS_MROPE:
                return { pos_0, pos_0 + (llama_pos) (i / nx), pos_0 + (llama_pos) (i % nx), 0 };
            case MTMD_LEGACY_POS_SENSENOVA_U1:
                return { pos_0, (llama_pos) (i / nx), (llama_pos) (i % nx), 0 };
            case MTMD_LEGACY_POS_HUNYUANVL:
                {
                    if (i == 0 || i == n_tok - 1) {
                        return { seq, seq, seq, seq }; // BOI, EOI
                    }
                    const uint32_t offset = i - 1;
                    return { seq, (llama_pos) (offset % (nx + 1)), (llama_pos) (offset / (nx + 1)), (llama_pos) image_idx };
                }
            case MTMD_LEGACY_POS_CANVAS:
                {
                    const bool tiled = entries[0].grid_x > 0 && entries[0].grid_y > 0;
                    const uint64_t canvas_w = tiled ? (uint64_t) entries[0].grid_x * nx : nx;
                    const uint64_t canvas_h = tiled ? (uint64_t) entries[0].grid_y * ny : ny;
                    const llama_pos base = pos_0 - 1;
                    // the tile of row i: its rows, then its separators, which take a position each
                    const size_t k = std::upper_bound(tile_row0.begin(), tile_row0.end(), i) - tile_row0.begin() - 1;
                    if (k >= entries.size()) {
                        throw std::runtime_error("canvas row out of range");
                    }
                    const llama_pos tp = pos_0 + (llama_pos) tile_t0[k];
                    const auto [gw64, gh64] = tile_grid(k);
                    const uint32_t gw = (uint32_t) gw64;
                    const uint32_t gh = (uint32_t) gh64;
                    i -= tile_row0[k];
                    if (i >= gw * gh) {
                        const llama_pos p = tp + 1 + (llama_pos) (i - gw * gh);
                        return { p, p, p, p };
                    }
                    const uint32_t row = i / gw;
                    const uint32_t col = i % gw;
                    uint32_t h;
                    uint32_t w;
                    if (k == 0) {
                        h = canvas_scale(row, gh, canvas_h);
                        w = canvas_scale(col, gw, canvas_w);
                    } else {
                        const uint32_t s = k - 1;
                        h = (s / entries[0].grid_x) * ny + row;
                        w = (s % entries[0].grid_x) * nx + col;
                    }
                    return { tp, base + (llama_pos) h, base + (llama_pos) w, base };
                }
            case MTMD_LEGACY_POS_NORMAL:
            default:
                return { seq, seq, seq, seq };
        }
    }
};

struct mtmd_image_tokens {
    uint32_t  nx    = 0; // token grid in x (M-RoPE decoders), or the token count
    uint32_t  ny    = 0; // token grid in y, 1 for the other decoders
    uint32_t  n_tok = 0; // rows
    llama_pos n_pos = 0; // positions the span advances
    std::string id;      // optional user-defined ID, useful for KV cache tracking

    mtmd_lmedia_ref lmedia; // nullptr once loaded from a buffer: counts and positions only

    // the decoder positions of a chunk without media, if known:
    // loaded from versions 2 to 5, the layout of the old engine, computed on demand
    std::shared_ptr<const mtmd_legacy_layout> legacy;
    // else (version 7, or a placeholder): row i at pos_0 is pos_base[4i + d] + pos_0 if bit d of pos_shift[i] is
    // set, else pos_base[4i + d]
    std::vector<llama_pos> pos_base;
    std::vector<uint8_t>   pos_shift;

    uint32_t n_tokens() const { return n_tok; }

    bool is_placeholder() const {
        return lmedia.m == nullptr || llama_media_is_placeholder(lmedia.m);
    }

    // [t, y, x, z] of row i at pos_0
    std::array<llama_pos, 4> row_pos(llama_pos pos_0, uint32_t i) const {
        std::array<llama_pos, 4> p;
        p.fill(pos_0 + (llama_pos) i);
        if (lmedia.m) {
            llama_media_get_row_pos(lmedia.m, pos_0, i, p.data(), 4);
        } else if (legacy) {
            p = legacy->row_pos(pos_0, i);
        } else if ((size_t) i < pos_shift.size()) {
            for (int d = 0; d < 4; ++d) {
                p[d] = pos_base[4 * (size_t) i + d] + ((pos_shift[i] >> d) & 1 ? pos_0 : 0);
            }
        }
        return p;
    }

    bool has_positions() const {
        return lmedia.m || legacy || !pos_shift.empty();
    }

    // row i as a base and a mask of the dimensions that move with pos_0: every layout is affine in pos_0 with a
    // slope of 0 or 1 per dimension
    std::pair<std::array<llama_pos, 4>, uint8_t> row_pos_affine(uint32_t i) const {
        const auto p0 = row_pos(0, i);
        const auto p1 = row_pos(1, i);
        uint8_t shift = 0;
        for (int d = 0; d < 4; ++d) {
            if (p1[d] != p0[d]) {
                shift |= 1 << d;
            }
        }
        return { p0, shift };
    }

    // a table of the current positions, which then outlive the media
    void keep_positions() {
        if (!lmedia.m || !pos_shift.empty()) {
            return;
        }
        pos_base.resize((size_t) n_tok * 4);
        pos_shift.resize(n_tok);
        for (uint32_t i = 0; i < n_tok; ++i) {
            const auto [p, shift] = row_pos_affine(i);
            std::copy(p.begin(), p.end(), pos_base.begin() + 4 * (size_t) i);
            pos_shift[i] = shift;
        }
    }

    void serialize(mtmd_serialization & ser) const {
        ser.write(nx);
        ser.write(ny);
        ser.write(n_tok);
        ser.write((int32_t) n_pos);
        ser.write(id);
        // version 7: the positions of every row
        const uint32_t n = has_positions() ? n_tok : 0;
        ser.write(n);
        for (uint32_t i = 0; i < n; ++i) {
            const auto [p, shift] = row_pos_affine(i);
            for (int d = 0; d < 4; ++d) {
                ser.write((int32_t) p[d]);
            }
            ser.write(shift);
        }
    }
    void deserialize(mtmd_serialization & ser) {
        lmedia.reset();
        legacy.reset();
        pos_base.clear();
        pos_shift.clear();
        if (ser.version < 6) {
            deserialize_legacy(ser);
            return;
        }
        nx    = ser.read<uint32_t>();
        ny    = ser.read<uint32_t>();
        n_tok = ser.read<uint32_t>();
        n_pos = ser.read<int32_t>();
        id    = ser.read<std::string>();
        if (ser.version >= 7) {
            const uint32_t n = ser.read<uint32_t>();
            if (n != 0 && n != n_tok) {
                throw std::runtime_error("invalid position count");
            }
            if (n > (ser.data.size() - ser.read_pos) / 17) {
                throw std::runtime_error("positions exceed buffer size");
            }
            pos_base.resize((size_t) n * 4);
            pos_shift.resize(n);
            for (uint32_t i = 0; i < n; ++i) {
                for (int d = 0; d < 4; ++d) {
                    pos_base[4 * (size_t) i + d] = ser.read<int32_t>();
                }
                pos_shift[i] = ser.read<uint8_t>();
            }
        }
    }

    void deserialize_legacy(mtmd_serialization & ser) {
        auto lay_ptr = std::make_shared<mtmd_legacy_layout>();
        mtmd_legacy_layout & lay = *lay_ptr;
        nx = ser.read<uint32_t>();
        ny = ser.read<uint32_t>();
        const uint32_t pos = ser.read<uint32_t>();
        if (pos > MTMD_LEGACY_POS_CANVAS) {
            throw std::runtime_error("invalid pos type");
        }
        lay.image_idx = ser.read<uint32_t>();
        const uint32_t n_temporal_merge = ser.read<uint32_t>();
        id = ser.read<std::string>();
        lay.entries = mtmd_legacy_read_batch(ser);
        lay.nx = nx;
        lay.ny = ny;
        const auto & entries = lay.entries;

        // early version-3 files numbered the canvas type 3, later ones gave 3 to SenseNova-U1 without a new
        // version; only canvas tiles carry MiniCPM-V separators
        lay.pos = (mtmd_legacy_pos) pos;
        if (lay.pos == MTMD_LEGACY_POS_SENSENOVA_U1 &&
                std::any_of(entries.begin(), entries.end(), [](const mtmd_legacy_entry & e) { return e.suffix_type != 0; })) {
            lay.pos = MTMD_LEGACY_POS_CANVAS;
        }

        // the counts of a malformed chunk must not wrap, and a save writes 17 bytes per row: no real image or
        // video chunk comes near 4M rows
        const uint64_t n_max = 1u << 22;
        uint64_t n = 0;
        if (lay.pos == MTMD_LEGACY_POS_HUNYUANVL) {
            n = ((uint64_t) nx + 1) * ny + 2;
        } else if (lay.pos == MTMD_LEGACY_POS_CANVAS) {
            if (entries.empty() || entries.back().nx <= 0 || entries.back().ny <= 0) {
                throw std::runtime_error("invalid canvas chunk");
            }
            // the slices sit on the anyres grid of the overview
            if (entries.size() > 1 && (entries[0].grid_x <= 0 || entries[0].grid_y <= 0)) {
                throw std::runtime_error("invalid canvas grid");
            }
            for (size_t k = 0; k < entries.size() && n <= n_max; ++k) {
                if (entries[k].nx < 0 || entries[k].ny < 0) {
                    throw std::runtime_error("invalid canvas tile");
                }
                const auto [gw, gh] = lay.tile_grid(k);
                if (gw > n_max || gh > n_max) {
                    throw std::runtime_error("invalid canvas tile");
                }
                lay.tile_row0.push_back((uint32_t) n);
                lay.tile_t0.push_back((uint32_t) (k == 0 ? 0 : lay.tile_t0.back() + 1 + mtmd_legacy_suffix_rows(entries[k - 1].suffix_type)));
                n += gw * gh + mtmd_legacy_suffix_rows(entries[k].suffix_type);
            }
            lay.tile_row0.push_back((uint32_t) std::min(n, n_max + 1));
        } else {
            uint64_t nz = entries.size();
            if (n_temporal_merge > 1) {
                nz = (nz + n_temporal_merge - 1) / n_temporal_merge;
            }
            n = (uint64_t) nx * ny;
            n = n <= n_max ? n * nz : n;
        }
        if (n > n_max) {
            throw std::runtime_error("image chunk too large");
        }
        n_tok = (uint32_t) n;
        if (lay.pos == MTMD_LEGACY_POS_MROPE && std::max(nx, ny) > n_max) {
            throw std::runtime_error("invalid image grid");
        }
        switch (lay.pos) {
            case MTMD_LEGACY_POS_MROPE:        n_pos = std::max(nx, ny); break;
            case MTMD_LEGACY_POS_SENSENOVA_U1: n_pos = 1;                break;
            case MTMD_LEGACY_POS_CANVAS:
                n_pos = 0;
                for (const auto & e : entries) {
                    n_pos += 1 + (llama_pos) mtmd_legacy_suffix_rows(e.suffix_type);
                }
                break;
            default:                           n_pos = n_tok;            break;
        }

        if ((lay.pos == MTMD_LEGACY_POS_MROPE || lay.pos == MTMD_LEGACY_POS_SENSENOVA_U1 || lay.pos == MTMD_LEGACY_POS_HUNYUANVL) && nx == 0) {
            throw std::runtime_error("invalid image grid");
        }
        lay.n_tok = n_tok;
        legacy = std::move(lay_ptr);
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
    uint32_t n_tokens = 0; // number of tokens
    std::string id;        // optional user-defined ID, useful for KV cache tracking

    mtmd_lmedia_ref lmedia; // nullptr once loaded from a buffer: counts only

    bool is_placeholder() const {
        return lmedia.m == nullptr || llama_media_is_placeholder(lmedia.m);
    }

    void serialize(mtmd_serialization & ser) const {
        ser.write(n_tokens);
        ser.write(id);
    }
    void deserialize(mtmd_serialization & ser) {
        lmedia.reset();
        n_tokens = ser.read<uint32_t>();
        id       = ser.read<std::string>();
        if (ser.version < 6) {
            // the old engine kept the mel entries and, from version 4, a slice of a soundtrack
            mtmd_legacy_read_batch(ser);
            if (ser.version >= 4 && ser.read<bool>()) {
                ser.read<uint32_t>(); // row_begin
                ser.read<uint32_t>(); // n_tokens of the track
                mtmd_legacy_read_batch(ser);
            }
        }
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;

    llama_media * lmedia() const {
        if (tokens_image) {
            return tokens_image->lmedia.m;
        }
        if (tokens_audio) {
            return tokens_audio->lmedia.m;
        }
        return nullptr;
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
        ser.write((uint32_t) type);
        ser.write<uint64_t>(tokens_text.size());
        for (llama_token tok : tokens_text) {
            ser.write((int32_t) tok);
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
        const uint32_t type_raw = ser.read<uint32_t>();
        if (type_raw >= MTMD_INPUT_CHUNK_TYPE_COUNT) {
            throw std::runtime_error("invalid chunk type");
        }
        type = (mtmd_input_chunk_type) type_raw;

        const uint64_t n_tokens_text = ser.read<uint64_t>();
        // reject before resize() so a tiny corrupted buffer can't force a huge allocation
        if (n_tokens_text > (ser.data.size() - ser.read_pos) / sizeof(int32_t)) {
            throw std::runtime_error("tokens_text length exceeds buffer size");
        }
        tokens_text.resize(n_tokens_text);
        for (uint64_t i = 0; i < n_tokens_text; i++) {
            tokens_text[i] = (llama_token) ser.read<int32_t>();
        }

        tokens_image.reset();
        if (ser.read<bool>()) {
            tokens_image = std::make_unique<mtmd_image_tokens>();
            tokens_image->deserialize(ser);
        }
        tokens_audio.reset();
        if (ser.read<bool>()) {
            tokens_audio = std::make_unique<mtmd_audio_tokens>();
            tokens_audio->deserialize(ser);
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE && !tokens_image) {
            throw std::runtime_error("type is IMAGE but tokens_image is missing");
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO && !tokens_audio) {
            throw std::runtime_error("type is AUDIO but tokens_audio is missing");
        }
        // a chunk carries the payload of its type only, as every writer made it
        if ((type != MTMD_INPUT_CHUNK_TYPE_IMAGE && tokens_image) || (type != MTMD_INPUT_CHUNK_TYPE_AUDIO && tokens_audio) ||
                (type != MTMD_INPUT_CHUNK_TYPE_TEXT && !tokens_text.empty())) {
            throw std::runtime_error("the payload does not match the chunk type");
        }
    }
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

// the output of an encode: a copy of the rows of each media, owned by the caller of the encode until the next one
// (writes to it do not reach the media, which may serve other encodes of the same chunk)
struct mtmd_encode_out {
    std::vector<std::vector<float>> rows_of; // per media

    void set(const std::vector<llama_media *> & ms) {
        rows_of.assign(ms.size(), {});
        for (size_t i = 0; i < ms.size(); ++i) {
            const int64_t n_embd = llama_media_get_rows(ms[i], 0, 0, nullptr);
            rows_of[i].resize((size_t) llama_media_n_tokens(ms[i]) * n_embd);
            llama_media_get_rows(ms[i], 0, llama_media_n_tokens(ms[i]), rows_of[i].data());
        }
    }

    size_t size() const { return rows_of.size(); }

    float * rows(size_t i) {
        return i < rows_of.size() ? rows_of[i].data() : nullptr;
    }
};

struct mtmd_batch {
    mtmd_context * ctx;
    std::vector<const mtmd_input_chunk *> entries;
    mtmd_encode_out out; // the encoded media, one per entry
    explicit mtmd_batch(mtmd_context * ctx): ctx(ctx) {}
    int32_t n_tokens() const {
        int32_t n = 0;
        for (const auto * chunk : entries) {
            n += mtmd_input_chunk_get_n_tokens(chunk);
        }
        return n;
    }
};

const char * mtmd_default_marker() {
    return "<__media__>";
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

//
// the mmproj file
//

struct mtmd_file_info {
    bool has_vision    = false;
    bool has_audio     = false;
    bool has_gen_audio = false;
    std::string proj_v; // projector names, empty if the tower is missing
    std::string proj_a;
    std::string proj_gen;
};

static mtmd_file_info mtmd_read_file_info(const char * fname) {
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_context * gctx = gguf_init_from_file(fname, params);
    if (!gctx) {
        throw std::runtime_error(mtmd_format("failed to load mmproj model from %s. Does this file exist?", fname));
    }
    auto get_bool = [&](const char * key) {
        const int64_t i = gguf_find_key(gctx, key);
        return i >= 0 && gguf_get_val_bool(gctx, i);
    };
    auto get_str = [&](const char * key) {
        const int64_t i = gguf_find_key(gctx, key);
        return i >= 0 ? std::string(gguf_get_val_str(gctx, i)) : std::string();
    };
    mtmd_file_info info;
    info.has_vision    = get_bool("clip.has_vision_encoder");
    info.has_audio     = get_bool("clip.has_audio_encoder");
    info.has_gen_audio = get_bool("clip.has_gen_audio_encoder");
    const std::string proj = get_str("clip.projector_type");
    if (info.has_vision) {
        info.proj_v = get_str("clip.vision.projector_type");
        info.proj_v = info.proj_v.empty() ? proj : info.proj_v;
    }
    if (info.has_audio) {
        info.proj_a = get_str("clip.audio.projector_type");
        info.proj_a = info.proj_a.empty() ? proj : info.proj_a;
    }
    if (info.has_gen_audio) {
        info.proj_gen = get_str("clip.gen.audio.projector_type");
        info.proj_gen = info.proj_gen.empty() ? proj : info.proj_gen;
    }
    gguf_free(gctx);
    return info;
}

// the device the towers run on, nullptr means CPU
static ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device) {
    if (!use_gpu) {
        return nullptr;
    }
    if (device != nullptr) {
        return device;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

// the mmproj as a llama_model, the user limits of the image token count as kv overrides
static llama_model * mtmd_load_model(const char * fname, const mtmd_context_params & ctx_params, ggml_backend_dev_t device, bool no_alloc) {
    std::vector<llama_model_kv_override> kvo;
    auto add_override = [&](const char * key, int32_t value) {
        if (value <= 0) {
            return;
        }
        llama_model_kv_override o {};
        snprintf(o.key, sizeof(o.key), "%s", key);
        o.tag     = LLAMA_KV_OVERRIDE_TYPE_INT;
        o.val_i64 = value;
        kvo.push_back(o);
    };
    add_override("clip.vision.image_min_tokens", ctx_params.image_min_tokens);
    add_override("clip.vision.image_max_tokens", ctx_params.image_max_tokens);
    if (!kvo.empty()) {
        kvo.emplace_back();
        kvo.back().key[0] = 0;
    }

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { device, nullptr };
    mp.n_gpu_layers = device ? 999 : 0;
    mp.devices      = device ? devs : nullptr;
    mp.kv_overrides = kvo.empty() ? nullptr : kvo.data();
    mp.no_alloc     = no_alloc;
    mp.progress_callback           = ctx_params.progress_callback;
    mp.progress_callback_user_data = ctx_params.progress_callback_user_data;
    if (const char * lm_mode = std::getenv("MTMD_LM_LOAD_MODE")) {
        if (strcmp(lm_mode, "dio") == 0) {
            mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
        }
    }
    if (no_alloc) {
        mp.load_mode = LLAMA_LOAD_MODE_NONE; // nothing is read, as common_fit does
    }
    return llama_model_load_from_file(fname, mp);
}

// the ENCODER context of the towers: its compute buffers are reserved for the warmup input of each tower
static llama_context * mtmd_init_encoder(llama_model * model, const mtmd_context_params & ctx_params) {
    const uint32_t n_rows = 512; // not a limit: an encoder context takes media of any size

    llama_context_params cp = llama_context_default_params();
    cp.ctx_type          = LLAMA_CONTEXT_TYPE_ENCODER;
    cp.n_ctx             = n_rows;
    cp.n_batch           = n_rows;
    cp.n_ubatch          = n_rows;
    cp.n_seq_max         = 1;
    cp.n_threads         = ctx_params.n_threads;
    cp.n_threads_batch   = ctx_params.n_threads;
    cp.flash_attn_type   = ctx_params.flash_attn_type;
    cp.cb_eval           = ctx_params.cb_eval;
    cp.cb_eval_user_data = ctx_params.cb_eval_user_data;
    cp.no_perf           = true;
    return llama_init_from_model(model, cp);
}

// the GEN context of the moved pipelines, shared by the gen states (one sequence each)
static llama_context_params mtmd_gen_ctx_params(const mtmd_context_params & ctx_params, uint32_t n_seq_max) {
    llama_context_params cp = llama_context_default_params();
    cp.ctx_type          = LLAMA_CONTEXT_TYPE_GEN_CODE;
    cp.n_ctx             = 256 * n_seq_max; // the gen states keep no KV, 256 is the smallest context of a sequence
    cp.n_batch           = n_seq_max; // one output per sequence
    cp.n_ubatch          = n_seq_max;
    cp.n_seq_max         = n_seq_max;
    cp.n_threads         = ctx_params.n_threads;
    cp.n_threads_batch   = ctx_params.n_threads;
    cp.flash_attn_type   = ctx_params.flash_attn_type;
    cp.cb_eval           = ctx_params.cb_eval;
    cp.cb_eval_user_data = ctx_params.cb_eval_user_data;
    cp.no_perf           = true;
    return cp;
}

// Qwen3-TTS generates on native graphs: a code predictor context (codes sampled in the graph by the sampler of
// the sequence) and a codec context; the other generation towers run on one GEN context of the engine
// (llama_batch_ext_add_gen). Either way, a gen state is one sequence of the shared contexts
static bool mtmd_gen_is_native(const llama_model * model) {
    return llama_model_gen_info(model).type == LLAMA_GEN_AUDIO_TYPE_QWEN3TTS;
}

static llama_context_params mtmd_native_gen_ctx_params(const mtmd_context_params & ctx_params, enum llama_context_type type, uint32_t n_seq_max) {
    llama_context_params cp = llama_context_default_params();
    cp.ctx_type          = type;
    cp.n_ctx             = 256 * n_seq_max; // no KV: the smallest context of a sequence
    cp.n_batch           = n_seq_max;       // one output per sequence; the graph takes one frame whatever the ubatch
    cp.n_ubatch          = n_seq_max;
    cp.n_seq_max         = n_seq_max;
    cp.n_threads         = ctx_params.n_threads;
    cp.n_threads_batch   = ctx_params.n_threads;
    cp.flash_attn_type   = ctx_params.flash_attn_type;
    cp.cb_eval           = ctx_params.cb_eval;
    cp.cb_eval_user_data = ctx_params.cb_eval_user_data;
    cp.no_perf           = true;
    if (type == LLAMA_CONTEXT_TYPE_GEN_WAV) {
        cp.n_batch  = 72; // one call decodes up to a window of frames
        cp.n_ubatch = 72;
    }
    return cp;
}

//
// mtmd_context
//

// shared by a context and the objects it hands out (gen states, audio streams), which may outlive it: they lock the
// mutex and check `alive` before touching the context, which clears it under the mutex when it is freed
struct mtmd_ctx_life {
    std::mutex mutex; // held by every public function that runs a graph
    bool alive = true;
};

// a set of GEN contexts, a sequence per gen state (as mtmd shared one scheduler): the engine runs both stages on
// gen_ctx, the native graphs on gen_code_ctx and gen_wav_ctx; a context adds a set when all its sequences are taken
struct mtmd_gen_bank {
    llama_context_ptr gen_ctx;
    llama_context_ptr gen_code_ctx;
    llama_context_ptr gen_wav_ctx;
    std::vector<bool> used;
};

struct mtmd_context {
    // the mmproj, shared by the encoder and the gen contexts
    std::shared_ptr<llama_model> model;

    // encoder towers (vision, audio)
    llama_context_ptr  ctx_enc;
    llama_context    * enc = nullptr; // encodes the media: ctx_enc, or a GEN context for files with a generation tower
    llama_media_proc * proc = nullptr;

    mtmd_file_info info;

    std::shared_ptr<mtmd_ctx_life> life = std::make_shared<mtmd_ctx_life>();
    std::mutex & compute_mutex = life->mutex;

    mtmd_encode_out out; // output of mtmd_encode*

    // gen stages: native graphs (Qwen3-TTS) or the engine (llama_batch_ext_add_gen), see mtmd_gen_is_native()
    bool gen_native = false;

    // the GEN contexts: [0] made with the context (it encodes the references), the others as gen states need them
    static constexpr uint32_t n_gen_seq_max = 64; // sequences of a bank
    std::vector<std::unique_ptr<mtmd_gen_bank>> gen_banks;
    mtmd_context_params gen_cparams; // threads, flash attention and eval callback of the GEN contexts

    // the last media encoded, the speaker reference of mtmd_gen_state_set_ref()
    mtmd_lmedia_ref last_ref;

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text = -1; // -1 means llm context not provided
    const llama_vocab * vocab = nullptr;
    const llama_model * text_model = nullptr;
    bool use_mrope = false;

    int32_t batch_max_tokens;

    mtmd_context(const char * mmproj_fname,
                 const llama_model * text_model,
                 const mtmd_context_params & ctx_params) :
        gen_cparams     (ctx_params),
        print_timings   (ctx_params.print_timings),
        n_threads       (ctx_params.n_threads),
        media_marker    (ctx_params.media_marker),
        n_embd_text     (text_model ? llama_model_n_embd_inp(text_model) : -1),
        vocab           (text_model ? llama_model_get_vocab(text_model) : nullptr),
        text_model      (text_model),
        batch_max_tokens(ctx_params.batch_max_tokens) {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }
        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        if (text_model) {
            switch (llama_model_rope_type(text_model)) {
                case LLAMA_ROPE_TYPE_NONE:
                case LLAMA_ROPE_TYPE_NORM:
                case LLAMA_ROPE_TYPE_NEOX:
                    use_mrope = false;
                    break;
                case LLAMA_ROPE_TYPE_MROPE:
                case LLAMA_ROPE_TYPE_IMROPE:
                    use_mrope = true;
                    break;
                default:
                    throw std::runtime_error(mtmd_format("unsupported decoder rope type: %d", llama_model_rope_type(text_model)));
            }
        }

        info = mtmd_read_file_info(mmproj_fname);
        if (!info.has_vision && !info.has_audio && !info.has_gen_audio) {
            throw std::runtime_error(mtmd_format("the mmproj model %s has no tower", mmproj_fname));
        }

        ggml_backend_dev_t device = mtmd_select_device(ctx_params.use_gpu, ctx_params.device);
        llama_model * m = mtmd_load_model(mmproj_fname, ctx_params, device, false);
        if (!m) {
            throw std::runtime_error(mtmd_format("failed to load mmproj model from %s", mmproj_fname));
        }
        model.reset(m, llama_model_free);

        if (info.has_gen_audio) {
            gen_native = mtmd_gen_is_native(m);
            auto bank = make_gen_bank();
            if (!bank) {
                throw std::runtime_error("failed to create the gen contexts");
            }
            gen_banks.push_back(std::move(bank));
            // the GEN contexts also encode the references: one set of compute buffers; the codec context of the
            // native graphs fits the speaker tower
            const mtmd_gen_bank & b0 = *gen_banks[0];
            enc = !gen_native ? b0.gen_ctx.get() : info.has_audio || info.has_vision ? b0.gen_wav_ctx.get() : nullptr;
        } else if (llama_model_has_context_type(m, LLAMA_CONTEXT_TYPE_ENCODER)) {
            ctx_enc.reset(mtmd_init_encoder(m, ctx_params));
            if (!ctx_enc) {
                throw std::runtime_error("failed to create the encoder context");
            }
            enc = ctx_enc.get();
        }

        if (enc && (info.has_vision || info.has_audio)) {
            proc = llama_media_proc_init(enc, text_model, llama_media_proc_default_params());
            if (!proc) {
                throw std::runtime_error("failed to create the media proc (wrong mmproj for this text model?)");
            }
        }

    }

    ~mtmd_context() {
        {
            std::lock_guard<std::mutex> lock(life->mutex);
            life->alive = false;
        }
        last_ref.reset();
        llama_media_proc_free(proc);
    }

    // a set of GEN contexts, nullptr on failure
    std::unique_ptr<mtmd_gen_bank> make_gen_bank() const {
        auto b = std::make_unique<mtmd_gen_bank>();
        llama_model * m = model.get();
        if (gen_native) {
            b->gen_code_ctx.reset(llama_init_from_model(m, mtmd_native_gen_ctx_params(gen_cparams, LLAMA_CONTEXT_TYPE_GEN_CODE, n_gen_seq_max)));
            b->gen_wav_ctx .reset(llama_init_from_model(m, mtmd_native_gen_ctx_params(gen_cparams, LLAMA_CONTEXT_TYPE_GEN_WAV,  n_gen_seq_max)));
            if (!b->gen_code_ctx || !b->gen_wav_ctx) {
                return nullptr;
            }
        } else {
            b->gen_ctx.reset(llama_init_from_model(m, mtmd_gen_ctx_params(gen_cparams, n_gen_seq_max)));
            if (!b->gen_ctx) {
                return nullptr;
            }
        }
        b->used.assign(n_gen_seq_max, false);
        return b;
    }

    int64_t n_embd_out() const {
        return llama_model_n_embd_out(model.get());
    }

    bool video_has_soundtrack() const {
        return proc && llama_media_proc_video_template(proc) == LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3_OMNI;
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

//
// tokenizer: the media markers of the prompt, the layout of each media by libllama
//

struct mtmd_tokenizer {
    const mtmd_context * ctx;

    std::string input_text; // note: can contain null bytes; do not use c_str()
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    using part = mtmd_internal_part;
    std::vector<part> parts;
    // these will be freed when mtmd_tokenizer finishes
    std::vector<mtmd::bitmap> bm_from_lazy;
    std::vector<const char *> text_from_lazy;

    mtmd_input_chunks cur;

    // the layout of the whole prompt, the text goes in too (positions, image numbering)
    llama_media_layout * lay = nullptr;

    // [LFM3_OMNI] the audio pieces of a video are one soundtrack, see llama_media_layout_begin_track()
    std::vector<std::vector<const mtmd_bitmap *>> soundtracks;
    std::map<const mtmd_bitmap *, size_t> soundtrack_of; // piece -> index in soundtracks

    ~mtmd_tokenizer() {
        llama_media_layout_free(lay);
        for (auto & str : text_from_lazy) {
            free((void *) str);
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
        for (const auto & p : parts_str) {
            if (p == ctx->media_marker) {
                if (i_bm >= bitmaps.size()) {
                    throw std::runtime_error(mtmd_format("number of media markers in text (%zu) exceeds number of bitmaps (%zu)", i_bm + 1, bitmaps.size()));
                }
                parts.push_back({"", bitmaps[i_bm++]});
            } else {
                parts.push_back({p, nullptr, parse_special});
            }
        }

        size_t n_markers = 0;
        for (const auto & p : parts) {
            if (p.bitmap != nullptr) {
                n_markers++;
            }
        }
        if (n_markers != bitmaps.size()) {
            throw std::runtime_error(mtmd_format("number of media markers in text (%zu) does not match number of bitmaps (%zu)", n_markers, bitmaps.size()));
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
        const char * vid_beg = "";
        const char * vid_end = "";
        if (ctx->proc) {
            llama_media_proc_video_markers(ctx->proc, &vid_beg, &vid_end);
        }

        std::vector<part> expanded;
        expanded.reserve(parts.size());
        for (auto & p : parts) {
            if (p.bitmap != nullptr && p.bitmap->lazy_callback) {
                LOG_DBG("%s: expanding lazy bitmap\n", __func__);
                const bool is_video = p.bitmap->is_video;
                if (is_video && vid_beg[0] != 0) {
                    expanded.push_back({vid_beg, nullptr, true});
                }
                std::vector<const mtmd_bitmap *> pieces;
                for (size_t i = 0;; i++) {
                    char * out_str = nullptr;
                    mtmd_bitmap * out_bm = nullptr;
                    const int res = p.bitmap->lazy_callback(i, p.bitmap->lazy_user_data, &out_bm, &out_str);
                    if (out_bm && out_str) {
                        throw std::runtime_error("lazy callback cannot return both bitmap and text");
                    }
                    if (res == 0) {
                        // OK, append the returned chunk; lazy part is not yet added
                        if (out_bm) {
                            auto & ptr = bm_from_lazy.emplace_back(out_bm); // remember to free it later
                            expanded.push_back({"", ptr.ptr.get()});
                            if (is_video && out_bm->is_audio && ctx->video_has_soundtrack()) {
                                pieces.push_back(out_bm);
                            }
                        } else if (out_str) {
                            auto & ptr = text_from_lazy.emplace_back(out_str); // remember to free it later
                            expanded.push_back({ptr, nullptr, parse_special});
                        }
                    } else if (res == -1) {
                        // EOF: lazy part removes itself (not added to expanded)
                        if (is_video && vid_end[0] != 0) {
                            expanded.push_back({vid_end, nullptr, true});
                        }
                        break;
                    } else if (res == -2) {
                        throw std::runtime_error("lazy callback returned error");
                    }
                }
                if (!pieces.empty()) {
                    for (const auto * piece : pieces) {
                        soundtrack_of[piece] = soundtracks.size();
                    }
                    soundtracks.push_back(std::move(pieces));
                }
            } else {
                expanded.push_back(std::move(p));
            }
        }
        parts = std::move(expanded);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();
        if (ctx->proc) {
            lay = llama_media_layout_init(ctx->proc, 0);
        }

        // [QWEN_VIDEO] frame merging for the models that support it (qwen-vl)
        const int n_merge_frames = ctx->proc ? llama_media_proc_n_frame_merge(ctx->proc) : 1;
        GGML_ASSERT(n_merge_frames <= 2 && "we only support merging maximum 2 images for now; open an issue if this model supports merging more");

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
                const int32_t res = add_media(bmps);
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
                    cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
                } else {
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

        // parity debugging: MTMD_DEBUG_CHUNKS=1 prints the layout of the prompt
        if (std::getenv("MTMD_DEBUG_CHUNKS")) {
            for (const auto & c : cur.entries) {
                if (c.type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    std::string s;
                    for (llama_token tok : c.tokens_text) {
                        s += std::to_string(tok) + " ";
                    }
                    LOG_WRN("chunk text  n=%zu: %s\n", c.tokens_text.size(), s.c_str());
                } else {
                    LOG_WRN("chunk %s n_tokens=%zu n_pos=%d\n", c.type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio",
                            mtmd_input_chunk_get_n_tokens(&c), (int) mtmd_input_chunk_get_n_pos(&c));
                }
            }
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
        if (lay) {
            llama_media_layout_add_tokens(lay, tokens.data(), tokens.size());
        }
        add_text_chunk(tokens);
    }

    // appends to the last text chunk, the layout already holds these tokens
    void add_text_chunk(const std::vector<llama_token> & tokens) {
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(cur.entries.back().tokens_text.end(), tokens.begin(), tokens.end());
        } else {
            cur.entries.emplace_back(mtmd_input_chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            });
        }
    }

    // a media chunk of a libllama media
    void add_media_chunk(llama_media * m, const std::string & id) {
        if (llama_media_get_type(m) == LLAMA_MEDIA_TYPE_AUDIO) {
            mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
            audio_tokens->n_tokens = llama_media_n_tokens(m);
            audio_tokens->id       = id;
            audio_tokens->lmedia   = mtmd_lmedia_ref(m);
            cur.entries.emplace_back(mtmd_input_chunk{
                MTMD_INPUT_CHUNK_TYPE_AUDIO,
                {}, // text tokens
                nullptr, // image tokens
                std::move(audio_tokens),
            });
            return;
        }

        mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
        image_tokens->id     = id;
        image_tokens->n_tok  = llama_media_n_tokens(m);
        image_tokens->n_pos  = llama_media_n_pos(m);
        image_tokens->lmedia = mtmd_lmedia_ref(m);
        // the token grid as the clip engine counts it, for the deprecated getters and the saved chunk
        if (!ctx->use_mrope || !llama_media_get_grid(m, &image_tokens->nx, &image_tokens->ny)) {
            image_tokens->nx = image_tokens->n_tok;
            image_tokens->ny = 1;
        }
        cur.entries.emplace_back(mtmd_input_chunk{
            MTMD_INPUT_CHUNK_TYPE_IMAGE,
            {}, // text tokens
            std::move(image_tokens),
            nullptr, // audio tokens
        });
    }

    int32_t add_media(std::vector<const mtmd_bitmap *> & bitmaps) {
        GGML_ASSERT(!bitmaps.empty());
        const auto * b0 = bitmaps[0];

        // a voice is an encoded media already
        if (b0->is_voice) {
            GGML_ASSERT(bitmaps.size() == 1);
            if (!b0->lmedia || llama_media_n_tokens(b0->lmedia.get()) == 0) {
                LOG_ERR("%s: error: the voice \"%s\" holds no embeddings\n", __func__, b0->id.c_str());
                return 2;
            }
            add_media_chunk(llama_media_ref(b0->lmedia.get()), b0->id);
            return 0;
        }

        if (!lay) {
            LOG_ERR("%s: error: model does not support %s input\n", __func__, b0->is_audio ? "audio" : "vision");
            return 2;
        }

        // the layout appends tokens to its last piece when that piece holds tokens
        const size_t n_before = llama_media_layout_n_pieces(lay);
        size_t n_tok_last = 0;
        if (n_before > 0) {
            llama_media_layout_get_tokens(lay, n_before - 1, &n_tok_last);
        }

        int32_t res;
        if (auto it = soundtrack_of.find(b0); it != soundtrack_of.end()) {
            // the first piece hands the whole soundtrack to the layout
            const auto & pieces = soundtracks[it->second];
            res = 0;
            if (b0 == pieces.front()) {
                std::vector<const float *> pcm;
                std::vector<size_t> n_samples;
                for (const auto * piece : pieces) {
                    pcm.push_back(piece->is_placeholder() ? nullptr : (const float *) piece->get_ro_buf().data());
                    n_samples.push_back(piece->nx);
                }
                res = llama_media_layout_begin_track(lay, pcm.data(), n_samples.data(), (int32_t) pieces.size());
            }
            if (res == 0) {
                res = llama_media_layout_add_track_piece(lay, b0->id.c_str());
            }
        } else if (b0->is_audio) {
            if (b0->nx == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }
            const float * pcm = b0->is_placeholder() ? nullptr : (const float *) b0->get_ro_buf().data();
            res = llama_media_layout_add_audio(lay, pcm, b0->nx, b0->id.c_str());
        } else {
            for (const auto * bmp : bitmaps) {
                if (bmp->nx <= 0 || bmp->ny <= 0) {
                    LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n", __func__, bmp->nx, bmp->ny);
                    return 2;
                }
            }
            std::vector<const uint8_t *> frames;
            for (const auto * bmp : bitmaps) {
                frames.push_back(bmp->is_placeholder() ? nullptr : bmp->get_ro_buf().data());
            }
            if (bitmaps.size() == 1 && !b0->is_video_frame()) {
                res = llama_media_layout_add_image(lay, b0->nx, b0->ny, frames[0], b0->id.c_str());
            } else {
                res = llama_media_layout_add_frames(lay, b0->nx, b0->ny, frames.data(), (int32_t) frames.size(), b0->is_video_frame(), b0->id.c_str());
            }
        }
        if (res == 1) {
            LOG_ERR("%s: error: model does not support %s input\n", __func__, b0->is_audio ? "audio" : "vision");
            return 2;
        }
        if (res != 0) {
            LOG_ERR("%s: libllama could not lay out the %s (%d)\n", __func__, b0->is_audio ? "audio" : "image", res);
            return 2;
        }

        // the new pieces: text tokens (markers, separators) and media spans
        for (size_t i = n_before > 0 ? n_before - 1 : 0; i < llama_media_layout_n_pieces(lay); ++i) {
            size_t n = 0;
            const llama_token * toks = llama_media_layout_get_tokens(lay, i, &n);
            const size_t skip = i + 1 == n_before ? n_tok_last : 0;
            if (toks) {
                if (n > skip) {
                    add_text_chunk(std::vector<llama_token>(toks + skip, toks + n));
                }
                continue;
            }
            if (i + 1 == n_before) {
                continue; // a media laid out before
            }
            add_media_chunk(llama_media_layout_get_media(lay, i), b0->id);
        }

        return 0;
    }

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
            const int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
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

//
// encode
//

static void mtmd_dump_rows(const std::vector<float> & rows, int64_t n_embd) {
    // parity debugging: MTMD_DEBUG_EMBEDDINGS=<path> dumps [int32 n_rows][int32 n_embd][f32 rows]
    const char * dump_path = std::getenv("MTMD_DEBUG_EMBEDDINGS");
    if (dump_path && strcmp(dump_path, "1") != 0 && !rows.empty() && n_embd > 0) {
        FILE * f = fopen(dump_path, "wb");
        if (f) {
            const int32_t hdr[2] = { (int32_t) (rows.size() / n_embd), (int32_t) n_embd };
            fwrite(hdr, sizeof(hdr), 1, f);
            fwrite(rows.data(), sizeof(float), rows.size(), f);
            fclose(f);
        }
    }
}

// encodes the media (one ENCODER batch, so libllama can share graphs between them), rows in chunk order
static int32_t mtmd_encode_media(mtmd_context * ctx, const std::vector<llama_media *> & media, mtmd_encode_out & out) {
    for (llama_media * m : media) {
        if (llama_media_is_placeholder(m)) {
            LOG_ERR("%s: cannot encode a placeholder media\n", __func__);
            return 1;
        }
    }
    bool need = false;
    for (llama_media * m : media) {
        need = need || !llama_media_is_encoded(m);
    }
    if (need) {
        if (!ctx->enc) {
            LOG_ERR("%s: the model has no encoder for this media\n", __func__);
            return 1;
        }
        llama_batch_ext * batch = llama_batch_ext_init(ctx->enc);
        int32_t res = 0;
        for (llama_media * m : media) {
            if (!llama_media_is_encoded(m) && llama_batch_ext_add_media(batch, 0, m, 0, 0, 0) < 0) {
                res = 1;
            }
        }
        if (res == 0) {
            res = llama_process(ctx->enc, LLAMA_PROCESS_TYPE_ENCODE, batch);
        }
        llama_batch_ext_free(batch);
        if (res != 0) {
            LOG_ERR("%s: failed to encode the media (%d)\n", __func__, res);
            return 1;
        }
    }

    out.set(media);
    if (std::getenv("MTMD_DEBUG_EMBEDDINGS")) {
        std::vector<float> rows;
        int64_t n_embd = 0;
        for (llama_media * m : media) {
            const int64_t n_rows = llama_media_n_tokens(m);
            n_embd = llama_media_get_rows(m, 0, 0, nullptr);
            const size_t off = rows.size();
            rows.resize(off + (size_t) n_rows * n_embd);
            llama_media_get_rows(m, 0, n_rows, rows.data() + off);
        }
        mtmd_dump_rows(rows, n_embd);
    }

    if (!media.empty()) {
        ctx->last_ref = mtmd_lmedia_ref(llama_media_ref(media.back()));
    }
    return 0;
}

static int32_t mtmd_encode_chunk_impl(mtmd_context * ctx, const mtmd_input_chunk * chunk, mtmd_encode_out & out) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    }
    llama_media * m = chunk->lmedia();
    if (m == nullptr || chunk->is_placeholder()) {
        LOG_ERR("%s: the %s tokens are a placeholder\n", __func__, chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio");
        return 1;
    }
    return mtmd_encode_media(ctx, { m }, out);
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        return mtmd_encode_chunk_impl(ctx, chunk, ctx->out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!image_tokens->lmedia.m || image_tokens->is_placeholder()) {
        LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
        return 1;
    }
    try {
        return mtmd_encode_media(ctx, { image_tokens->lmedia.m }, ctx->out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->out.rows(0);
}

mtmd_batch * mtmd_batch_init(mtmd_context * ctx) {
    return new mtmd_batch(ctx);
}

void mtmd_batch_free(mtmd_batch * batch) {
    delete batch;
}

int32_t mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
        return 1;
    }
    if (batch->entries.empty()) {
        // batch must have at least one chunk
        batch->entries.push_back(chunk);
        return 0;
    }
    if (batch->entries[0]->type != chunk->type) {
        return 3; // "cannot batch" error code
    }
    const int32_t new_n_tokens = batch->n_tokens() + (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
    if (new_n_tokens > batch->ctx->batch_max_tokens) {
        return 2; // "batch too large" error code
    }
    // libllama shares one graph between consecutive media of equal geometry
    batch->entries.push_back(chunk);
    return 0;
}

int32_t mtmd_batch_encode(mtmd_batch * batch) {
    std::lock_guard<std::mutex> lock(batch->ctx->compute_mutex);
    if (batch->entries.empty()) {
        LOG_ERR("%s: batch is empty\n", __func__);
        return 1;
    }
    std::vector<llama_media *> media;
    for (const auto * chunk : batch->entries) {
        llama_media * m = chunk->lmedia();
        if (m == nullptr || chunk->is_placeholder()) {
            LOG_ERR("%s: chunk is placeholder\n", __func__);
            return 1;
        }
        media.push_back(m);
    }
    try {
        return mtmd_encode_media(batch->ctx, media, batch->out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_batch_get_output_embd(mtmd_batch * batch, const mtmd_input_chunk * chunk) {
    if (batch->out.size() != batch->entries.size()) {
        LOG_ERR("%s: batch has not been encoded yet\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < batch->entries.size(); ++i) {
        if (batch->entries[i] == chunk) {
            return batch->out.rows(i);
        }
    }
    return nullptr; // not found
}

//
// properties
//

bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    if (chunk && chunk->lmedia()) {
        return llama_media_is_atomic(chunk->lmedia());
    }
    const auto type = chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO ? LLAMA_MEDIA_TYPE_AUDIO : LLAMA_MEDIA_TYPE_IMAGE;
    bool atomic = false;
    llama_model_media_n_tokens_max(ctx->model.get(), type, &atomic);
    return atomic;
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx->use_mrope;
}

bool mtmd_support_vision(const mtmd_context * ctx) {
    return ctx->proc && llama_media_proc_has(ctx->proc, LLAMA_MEDIA_TYPE_IMAGE);
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx->proc && llama_media_proc_has(ctx->proc, LLAMA_MEDIA_TYPE_AUDIO);
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    return ctx->proc ? llama_media_proc_sample_rate(ctx->proc) : -1;
}

mtmd_video_config mtmd_video_config_init(mtmd_video_template tmpl, const mtmd_video_frame_size_fn & frame_size,
                                         int32_t src_width, int32_t src_height,
                                         int32_t src_n_frames, float src_fps) {
    mtmd_video_config cfg{tmpl, 0.0f, 0, 0, 0};
    if (tmpl == MTMD_VIDEO_TEMPLATE_DEFAULT) {
        return cfg;
    }

    // matches Qwen3VLVideoProcessor and Lfm3VlVideoProcessor: 2 fps sampling with a [4, 768] frame clamp
    // LFM3-Omni samples at most 384 frames and hears the soundtrack
    cfg.fps = 2.0f;
    if (src_n_frames <= 0 || src_fps <= 0.0f) {
        return cfg;
    }

    int32_t n = (int32_t)((double) src_n_frames / src_fps * cfg.fps);
    n = std::clamp(n, 4, tmpl == MTMD_VIDEO_TEMPLATE_LFM3_OMNI ? 384 : 768);
    n = std::min(n, src_n_frames);
    cfg.n_frames = n;

    int32_t w = 0;
    int32_t h = 0;
    if (frame_size && frame_size(n, src_width, src_height, w, h)) {
        cfg.frame_width  = w;
        cfg.frame_height = h;
    }
    return cfg;
}

mtmd_video_config mtmd_get_video_config(const mtmd_context * ctx,
                                        int32_t src_width, int32_t src_height,
                                        int32_t src_n_frames, float src_fps) {
    mtmd_video_template tmpl = MTMD_VIDEO_TEMPLATE_DEFAULT;
    switch (ctx->proc ? llama_media_proc_video_template(ctx->proc) : LLAMA_MEDIA_VIDEO_TEMPLATE_DEFAULT) {
        case LLAMA_MEDIA_VIDEO_TEMPLATE_QWEN3VL:   tmpl = MTMD_VIDEO_TEMPLATE_QWEN3VL;   break;
        case LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3:      tmpl = MTMD_VIDEO_TEMPLATE_LFM3;      break;
        case LLAMA_MEDIA_VIDEO_TEMPLATE_LFM3_OMNI: tmpl = MTMD_VIDEO_TEMPLATE_LFM3_OMNI; break;
        default: break;
    }
    const llama_media_proc * proc = ctx->proc;
    auto frame_size = [proc](int32_t n, int32_t sw, int32_t sh, int32_t & w, int32_t & h) {
        return proc != nullptr && llama_media_proc_video_frame_size(proc, n, sw, sh, &w, &h);
    };
    return mtmd_video_config_init(tmpl, frame_size, src_width, src_height, src_n_frames, src_fps);
}

const char * mtmd_get_marker(const mtmd_context * ctx) {
    return ctx->media_marker.c_str();
}

//
// bitmaps
//

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data) {
    return new mtmd_bitmap(data, nx, ny);
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples, const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap((const unsigned char *) data, n_samples);
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
    if (bitmap->get_ro_buf().empty()) {
        return nullptr;
    }
    return bitmap->get_ro_buf().data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    return bitmap->n_bytes();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
}

bool mtmd_bitmap_is_voice(const mtmd_bitmap * bitmap) {
    return bitmap->is_voice;
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

void mtmd_bitmap_set_is_video(mtmd_bitmap * bitmap, bool is_video) {
    bitmap->is_video = is_video;
}

mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context * ctx,
                                    const char * id,
                                    void * user_data,
                                    mtmd_bitmap_lazy_callback callback) {
    GGML_UNUSED(ctx); // reserved for future use
    mtmd_bitmap * bitmap = new mtmd_bitmap(nullptr, 0, 0);
    bitmap->lazy_callback  = callback;
    bitmap->lazy_user_data = user_data;
    mtmd_bitmap_set_id(bitmap, id);
    return bitmap;
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    delete bitmap;
}

//
// chunks
//

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
    delete chunks;
}

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
    switch (chunk->type) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return chunk->tokens_text.size();
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return chunk->tokens_image->n_tokens();
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return chunk->tokens_audio->n_tokens;
        default: GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    switch (chunk->type) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return chunk->tokens_text.size();
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return chunk->tokens_image->n_pos;
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return chunk->tokens_audio->n_tokens;
        default: GGML_ABORT("invalid chunk type");
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

struct llama_media * mtmd_input_chunk_get_llama_media(const mtmd_input_chunk * chunk) {
    return chunk ? chunk->lmedia() : nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        copy->tokens_image = std::make_unique<mtmd_image_tokens>(*chunk->tokens_image);
    }
    if (chunk->tokens_audio) {
        copy->tokens_audio = std::make_unique<mtmd_audio_tokens>(*chunk->tokens_audio);
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    delete chunk;
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
    mtmd_input_chunk * copy = mtmd_input_chunk_copy(chunk);
    if (copy->tokens_image) {
        copy->tokens_image->keep_positions(); // a saved placeholder keeps the row positions of the image
        copy->tokens_image->lmedia.reset();
    }
    if (copy->tokens_audio) {
        copy->tokens_audio->lmedia.reset();
    }
    return copy;
}

int32_t mtmd_input_chunk_save(const mtmd_input_chunk * chunk, char * out_buf, size_t out_len, size_t * expected_out_len) {
    std::vector<char> buf;
    if (mtmd_input_chunk_save_impl(chunk, buf) != 0) {
        return -1;
    }
    if (expected_out_len) {
        *expected_out_len = buf.size();
    }
    if (out_buf == nullptr) {
        return 0;
    }
    if (out_len < buf.size()) {
        return -2;
    }
    std::memcpy(out_buf, buf.data(), buf.size());
    return 0;
}

mtmd_input_chunk * mtmd_input_chunk_load(const char * buf, size_t len) {
    try {
        mtmd_serialization ser(MTMD_SERIALIZATION_VERSION, buf, len);
        mtmd::input_chunk_ptr chunk(new mtmd_input_chunk{MTMD_INPUT_CHUNK_TYPE_TEXT, {}, nullptr, nullptr});
        chunk->deserialize(ser);
        return chunk.release();
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    const auto p = image_tokens->row_pos(pos_0, (uint32_t) i);
    pos.t = p[0];
    pos.y = p[1];
    pos.x = p[2];
    pos.z = p[3];
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_pos;
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    chunks->entries.emplace_back(mtmd_input_chunk{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        nullptr, // image tokens
        nullptr, // audio tokens
    });

    // create an image chunk, a placeholder of 16 frames of 4 x 4 tokens
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx    = 4;
    image_tokens->ny    = 4;
    image_tokens->n_tok = 256;
    image_tokens->n_pos = 256;
    image_tokens->id    = "image_1";
    chunks->entries.emplace_back(mtmd_input_chunk{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    });

    return chunks;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : mtmd_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char * fname) {
    try {
        const mtmd_file_info info = mtmd_read_file_info(fname);
        mtmd_caps cap;
        cap.inp_audio  = info.has_audio;
        cap.inp_vision = info.has_vision;
        cap.gen_audio  = info.has_gen_audio;
        return cap;
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, fname, e.what());
        return mtmd_caps{ false, false, false };
    }
}

static void stub_log_callback(enum ggml_log_level, const char *, void *) {
    // do nothing
}

mtmd_memory_usage mtmd_get_memory_usage(const char * mmproj_fname, struct mtmd_context_params ctx_params) {
    ctx_params.progress_callback = nullptr;

    // the loggers are process-wide: concurrent estimates would restore each other's stub
    static std::mutex estimate_mutex;
    std::lock_guard<std::mutex> estimate_lock(estimate_mutex);

    // the model and the context of mtmd_init_from_file(), allocations simulated, quietly
    auto saved_log_callback  = g_logger_state.log_callback;
    auto saved_log_user_data = g_logger_state.log_callback_user_data;
    // libllama's logger and ggml's can differ (a client setting ggml's after llama_log_set): both are restored
    ggml_log_callback saved_llama_log_callback  = nullptr;
    void *            saved_llama_log_user_data = nullptr;
    ggml_log_callback saved_ggml_log_callback   = nullptr;
    void *            saved_ggml_log_user_data  = nullptr;
    llama_log_get_llama(&saved_llama_log_callback, &saved_llama_log_user_data);
    ggml_log_get(&saved_ggml_log_callback, &saved_ggml_log_user_data);
    auto restore_logs = [&]() {
        mtmd_log_set(saved_log_callback, saved_log_user_data);
        llama_log_set(saved_llama_log_callback, saved_llama_log_user_data); // sets ggml's too
        ggml_log_set(saved_ggml_log_callback, saved_ggml_log_user_data);
    };
    try {
        mtmd_log_set(stub_log_callback, nullptr);
        llama_log_set(stub_log_callback, nullptr);
        const mtmd_file_info info = mtmd_read_file_info(mmproj_fname);
        ggml_backend_dev_t device = mtmd_select_device(ctx_params.use_gpu, ctx_params.device);
        llama_model_ptr model(mtmd_load_model(mmproj_fname, ctx_params, device, true));
        if (!model) {
            throw std::runtime_error("failed to load the model");
        }
        mtmd_memory_usage res;
        res.image_max_tokens = -1;
        res.use_non_causal   = false;
        // the contexts mtmd_init_from_file() creates: the gen context of the engine, or the code predictor and codec
        // contexts of the native graphs, or the encoder context; the weights count once
        std::vector<llama_context_ptr> ctxs;
        if (info.has_gen_audio && !mtmd_gen_is_native(model.get())) {
            ctxs.emplace_back(llama_init_from_model(model.get(), mtmd_gen_ctx_params(ctx_params, mtmd_context::n_gen_seq_max)));
        } else if (info.has_gen_audio) {
            for (auto type : { LLAMA_CONTEXT_TYPE_GEN_WAV, LLAMA_CONTEXT_TYPE_GEN_CODE }) {
                ctxs.emplace_back(llama_init_from_model(model.get(), mtmd_native_gen_ctx_params(ctx_params, type, mtmd_context::n_gen_seq_max)));
            }
        } else if (llama_model_has_context_type(model.get(), LLAMA_CONTEXT_TYPE_ENCODER)) {
            ctxs.emplace_back(mtmd_init_encoder(model.get(), ctx_params));
        }
        bool weights_counted = false;
        for (const auto & ctx : ctxs) {
            if (!ctx) {
                throw std::runtime_error("failed to create a context");
            }
            for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx.get())) {
                res.backend_mem_usage[ggml_backend_buft_get_device(buft)] += mb.total() - (weights_counted ? mb.model : 0);
            }
            weights_counted = true;
        }
        if (ctxs.empty()) {
            res.backend_mem_usage[device ? device : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)] += llama_model_size(model.get());
        }
        if (info.has_vision) {
            bool atomic = false;
            res.image_max_tokens = llama_model_media_n_tokens_max(model.get(), LLAMA_MEDIA_TYPE_IMAGE, &atomic);
            res.use_non_causal   = atomic;
        }
        restore_logs();
        return res;
    } catch (const std::exception & e) {
        restore_logs();
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return {{}, -1, false};
    }
}

//
// Debugging API (NOT intended for public use)
//

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!mtmd_support_vision(ctx)) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    const int img_sz = (int) image.size();
    std::vector<float> img_buf;
    img_buf.reserve((size_t) img_sz * img_sz);
    for (const auto & row : image) {
        img_buf.insert(img_buf.end(), row.begin(), row.end());
    }
    llama_media_debug_encode(ctx->proc, LLAMA_MEDIA_TYPE_IMAGE, img_sz, img_sz, img_buf.data(), img_buf.size());
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!mtmd_support_audio(ctx)) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    llama_media_debug_encode(ctx->proc, LLAMA_MEDIA_TYPE_AUDIO, (int32_t) input.size(), 1, input.data(), input.size());
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!mtmd_support_vision(ctx)) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    llama_media_debug_preprocess(ctx->proc, LLAMA_MEDIA_TYPE_IMAGE, nx, ny, rgb_values.data());
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (!mtmd_support_audio(ctx)) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    llama_media_debug_preprocess(ctx->proc, LLAMA_MEDIA_TYPE_AUDIO, (int32_t) samples.size(), 1, samples.data());
}

//
// realtime audio input
//

// the stream steps run graphs on the encoder context, which the gen states of LFM3 share: they hold its compute
// mutex; a stream may outlive its context (a server freeing its slots after the context)
struct mtmd_audio_stream {
    std::shared_ptr<mtmd_ctx_life> life;
    llama_audio_stream * ls = nullptr;
    int n_embd = 0;

    ~mtmd_audio_stream() {
        llama_audio_stream_free(ls);
    }
};

bool mtmd_support_audio_stream(const mtmd_context * ctx) {
    return ctx->proc && ctx->text_model && llama_media_proc_has_audio_stream(ctx->proc);
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
    llama_audio_stream_params lp = llama_audio_stream_default_params();
    lp.step_ms  = params.step_ms;
    lp.prompt   = params.prompt;
    lp.n_prompt = params.n_prompt;
    auto s = std::make_unique<mtmd_audio_stream>();
    s->life = ctx->life;
    {
        std::lock_guard<std::mutex> lock(ctx->compute_mutex);
        s->ls = llama_audio_stream_init(ctx->proc, lp);
    }
    if (!s->ls) {
        return nullptr;
    }
    s->n_embd = ctx->n_embd_text;
    return s.release();
}

void mtmd_audio_stream_free(mtmd_audio_stream * stream) {
    if (stream) {
        // the stream state holds no reference to the freed context: it can go after it
        std::lock_guard<std::mutex> lock(stream->life->mutex);
        llama_audio_stream_free(stream->ls);
        stream->ls = nullptr;
    }
    delete stream;
}

int32_t mtmd_audio_stream_push(mtmd_audio_stream * s, const float * samples, size_t n_samples) {
    std::lock_guard<std::mutex> lock(s->life->mutex);
    if (!s->life->alive) {
        LOG_ERR("%s: the context of the stream was freed\n", __func__);
        return 1;
    }
    return llama_audio_stream_push(s->ls, samples, n_samples);
}

int32_t mtmd_audio_stream_flush(mtmd_audio_stream * s, bool end) {
    std::lock_guard<std::mutex> lock(s->life->mutex);
    if (!s->life->alive) {
        LOG_ERR("%s: the context of the stream was freed\n", __func__);
        return 1;
    }
    return llama_audio_stream_flush(s->ls, end);
}

int32_t mtmd_audio_stream_step(mtmd_audio_stream * s, llama_token last_token) {
    std::lock_guard<std::mutex> lock(s->life->mutex);
    if (!s->life->alive) {
        LOG_ERR("%s: the context of the stream was freed\n", __func__);
        return -2;
    }
    return llama_audio_stream_step(s->ls, last_token);
}

const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream * s) {
    return llama_audio_stream_get_output_embd(s->ls);
}

bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream * s) {
    return llama_audio_stream_is_seq_start(s->ls);
}

int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream * s) {
    return llama_audio_stream_get_n_rows(s->ls);
}

int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream * s) {
    return llama_audio_stream_get_n_rows_flushed(s->ls);
}

//
// audio generation
//

mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * ctx) {
    const llama_gen_info li = llama_model_gen_info(ctx->model.get());
    mtmd_gen_audio_info info{};
    info.type          = (mtmd_gen_audio_type) li.type;
    info.sample_rate   = li.sample_rate;
    info.n_channels    = li.n_channels;
    info.model_variant = li.model_variant;
    info.n_voices      = li.n_voices;
    return info;
}

size_t mtmd_gen_audio_read_tensor(mtmd_context * ctx, const char * name, float * out, size_t n_max) {
    return llama_model_gen_read_tensor(ctx->model.get(), name, out, n_max);
}

const char * mtmd_gen_audio_voice_name(const mtmd_context * ctx, int32_t idx) {
    return llama_model_gen_voice_name(ctx->model.get(), idx);
}

mtmd_bitmap * mtmd_gen_audio_get_voice(const mtmd_context * ctx, const char * name) {
    if (!ctx->info.has_gen_audio || !name) {
        return nullptr;
    }
    llama_media * m = llama_media_init_voice(ctx->model.get(), name);
    if (!m) {
        return nullptr;
    }
    const uint32_t n_rows = llama_media_n_tokens(m);
    auto * bitmap = new mtmd_bitmap(nullptr, (uint32_t) llama_media_get_rows(m, 0, 0, nullptr), n_rows);
    bitmap->is_voice = true;
    bitmap->id       = llama_media_get_id(m);
    bitmap->lmedia.reset(m, llama_media_free);
    return bitmap;
}

mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * ctx) {
    const llama_gen_inp li = llama_gen_inp_default(ctx->model.get());
    mtmd_gen_inp inp{};
    inp.type           = li.type == LLAMA_GEN_STAGE_WAV ? MTMD_GEN_PROCESS_TYPE_GEN_WAV : MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    inp.top_k          = li.top_k;
    inp.top_p          = li.top_p;
    inp.temp           = li.temp;
    inp.penalty_repeat = li.penalty_repeat;
    inp.penalty_last_n = li.penalty_last_n;
    inp.n_steps        = li.n_steps;
    return inp;
}

// the sampling of the Qwen3-TTS code predictor, inside its graph (a backend sampler): top-k, top-p over the
// candidates, then one uniform draw per codebook from the RNG of the gen state; the algorithm and the draws of the
// clip engine, so a seed gives the codes it gave
struct mtmd_code_sampler {
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::mt19937 rng;
    std::vector<ggml_tensor *> rand_inp; // the draws of the graph being built, one per codebook, in order
};

static ggml_tensor * mtmd_code_sampler_build(ggml_context * ctx0, const mtmd_code_sampler & s, ggml_tensor * logits, ggml_tensor * inp_rand) {
    logits = ggml_reshape_1d(ctx0, logits, ggml_nelements(logits));
    const int64_t n_vocab = logits->ne[0];

    auto sort_by = [ctx0](ggml_tensor * a, ggml_tensor * idx) {
        ggml_tensor * a2d = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
        return ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, a2d, idx), idx->ne[0]);
    };

    ggml_tensor * cur        = logits;
    ggml_tensor * candidates = nullptr; // maps a row back to its code

    if (s.top_k > 0 && s.top_k < n_vocab) {
        ggml_tensor * idx = ggml_top_k(ctx0, cur, s.top_k);
        candidates = idx;
        cur        = sort_by(cur, idx);
    }

    if (s.top_p < 1.0f) {
        ggml_tensor * sorted_idx    = ggml_argsort(ctx0, cur, GGML_SORT_ORDER_DESC);
        ggml_tensor * sorted_logits = sort_by(cur, sorted_idx);
        candidates = candidates ? sort_by(candidates, sorted_idx) : sorted_idx;

        ggml_tensor * probs = ggml_soft_max(ctx0, sorted_logits);
        ggml_tensor * cdf   = ggml_cumsum(ctx0, probs);

        // keep the codes up to the one where the cdf crosses top_p, that one included
        ggml_tensor * keep_mask = ggml_step(ctx0, ggml_scale_bias(ctx0, cdf, -1.0f, s.top_p));
        ggml_tensor * idxf      = ggml_clamp(ctx0, ggml_sum(ctx0, keep_mask), 0.0f, (float) keep_mask->ne[0] - 1);
        ggml_tensor * ones      = ggml_scale_bias(ctx0, idxf, 0.0f, 1.0f);

        ggml_tensor * keep_mask_2d = ggml_reshape_2d(ctx0, keep_mask, 1, keep_mask->ne[0]);
        keep_mask_2d = ggml_set_rows(ctx0, keep_mask_2d, ones, ggml_cast(ctx0, idxf, GGML_TYPE_I32));
        keep_mask    = ggml_reshape_1d(ctx0, keep_mask_2d, keep_mask->ne[0]);

        cur = ggml_add(ctx0, sorted_logits, ggml_log(ctx0, keep_mask)); // log(1) = 0 keeps, log(0) = -inf drops
    }

    // the code where the cdf crosses the draw
    ggml_tensor * cumsum     = ggml_cumsum(ctx0, ggml_soft_max(ctx0, cur));
    ggml_tensor * cross_mask = ggml_step(ctx0, ggml_sub(ctx0, cumsum, inp_rand));
    // a cdf that rounds to at most the draw crosses nowhere: the last code, as the backend dist sampler does
    ggml_tensor * idxf       = ggml_scale_bias(ctx0, ggml_sum(ctx0, cross_mask), -1.0f, (float) cross_mask->ne[0]);
    ggml_tensor * idx        = ggml_cast(ctx0, ggml_clamp(ctx0, idxf, 0.0f, (float) cross_mask->ne[0] - 1), GGML_TYPE_I32);

    if (candidates) {
        idx = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, candidates, 1, candidates->ne[0]), idx);
    }
    return idx;
}

static const char * mtmd_code_sampler_name(const llama_sampler * /*smpl*/) {
    return "mtmd-code";
}

// the codes are only sampled in the graph
static void mtmd_code_sampler_apply(llama_sampler * /*smpl*/, llama_token_data_array * /*cur_p*/) {
    GGML_ABORT("the code sampler runs in the graph of the code predictor only");
}

static void mtmd_code_sampler_free(llama_sampler * smpl) {
    delete (mtmd_code_sampler *) smpl->ctx;
}

static bool mtmd_code_sampler_backend_init(llama_sampler * /*smpl*/, ggml_backend_buffer_type_t /*buft*/, uint32_t /*n_outputs_max_per_seq*/) {
    return true;
}

static void mtmd_code_sampler_backend_apply(llama_sampler * smpl, ggml_context * ctx0, ggml_cgraph * /*gf*/, llama_sampler_data * data) {
    auto * s = (mtmd_code_sampler *) smpl->ctx;
    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_format_name(inp_rand, "code_rand_%zu", s->rand_inp.size());
    ggml_set_input(inp_rand);
    s->rand_inp.push_back(inp_rand);
    data->sampled = mtmd_code_sampler_build(ctx0, *s, data->logits, inp_rand);
}

static void mtmd_code_sampler_backend_set_input(llama_sampler * smpl) {
    auto * s = (mtmd_code_sampler *) smpl->ctx;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (ggml_tensor * t : s->rand_inp) {
        const float v = dist(s->rng);
        ggml_backend_tensor_set(t, &v, 0, sizeof(v));
    }
}

static void mtmd_code_sampler_backend_reset(llama_sampler * smpl) {
    ((mtmd_code_sampler *) smpl->ctx)->rand_inp.clear();
}

static llama_sampler_i mtmd_code_sampler_i = {
    /* .name              = */ mtmd_code_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ mtmd_code_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ mtmd_code_sampler_free,
    /* .backend_init      = */ mtmd_code_sampler_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ mtmd_code_sampler_backend_apply,
    /* .backend_set_input = */ mtmd_code_sampler_backend_set_input,
    /* .backend_reset     = */ mtmd_code_sampler_backend_reset,
    /* .copy_state        = */ nullptr,
};

struct mtmd_gen_state {
    mtmd_context  * ctx  = nullptr;
    mtmd_gen_bank * bank = nullptr; // the GEN contexts of ctx holding the sequence
    llama_seq_id    seq  = -1;
    std::shared_ptr<mtmd_ctx_life> life; // ctx and bank are valid while life->alive, under its mutex

    std::vector<int32_t> out_codes; // this frame's sampled codes (GEN_CODE)
    std::vector<float>   out_feats; // this frame's continuous features, if any (GEN_CODE)
    std::vector<float>   out_embd;  // next-step hidden state fed back to backbone (GEN_CODE)
    std::vector<float>   out_audio; // decoded PCM samples for the current frame (GEN_WAV)

    // native graphs (Qwen3-TTS): the sampler of the sequence in the code predictor context (a chain holding
    // code_smpl), its RNG and parameters live as long as the state
    llama_sampler_ptr   lm_smpl;
    mtmd_code_sampler * code_smpl = nullptr;

    ~mtmd_gen_state() {
        if (!life || seq < 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(life->mutex);
        if (!life->alive) {
            return; // the sequences went with the context
        }
        // the per-sequence state of the GEN contexts (decoder slots, RNG) and the sampler
        if (ctx->gen_native) {
            llama_memory_seq_rm(llama_get_memory(bank->gen_wav_ctx.get()), seq, -1, -1);
            llama_set_sampler(bank->gen_code_ctx.get(), seq, nullptr);
        } else {
            llama_memory_seq_rm(llama_get_memory(bank->gen_ctx.get()), seq, -1, -1);
        }
        bank->used[seq] = false;
        // a set added for a peak of states goes with its last state; the first one encodes the references
        auto & banks = ctx->gen_banks;
        if (bank != banks[0].get() && std::none_of(bank->used.begin(), bank->used.end(), [](bool u) { return u; })) {
            banks.erase(std::find_if(banks.begin(), banks.end(), [this](const auto & b) { return b.get() == bank; }));
        }
    }
};

mtmd_gen_state_params mtmd_gen_state_params_default(void) {
    mtmd_gen_state_params params{};
    params.seed = UINT32_MAX;
    return params;
}

mtmd_gen_state * mtmd_gen_state_init(mtmd_context * ctx, mtmd_gen_state_params params) {
    if (!ctx->info.has_gen_audio) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return nullptr;
    }
    auto st = std::make_unique<mtmd_gen_state>();

    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (ctx->gen_banks.empty()) {
        LOG_ERR("%s: the gen contexts are missing\n", __func__);
        return nullptr;
    }
    // the first free sequence; a new set of GEN contexts when every one is taken
    mtmd_gen_bank * bank = nullptr;
    llama_seq_id    seq  = -1;
    for (const auto & b : ctx->gen_banks) {
        const auto it = std::find(b->used.begin(), b->used.end(), false);
        if (it != b->used.end()) {
            bank = b.get();
            seq  = (llama_seq_id) (it - b->used.begin());
            break;
        }
    }
    if (!bank) {
        LOG_INF("%s: more than %zu gen states, adding gen contexts\n", __func__, ctx->gen_banks.size() * mtmd_context::n_gen_seq_max);
        auto b = ctx->make_gen_bank();
        if (!b) {
            LOG_ERR("%s: failed to create the gen contexts\n", __func__);
            return nullptr;
        }
        ctx->gen_banks.push_back(std::move(b));
        bank = ctx->gen_banks.back().get();
        seq  = 0;
    }
    if (ctx->gen_native) {
        auto * cs = new mtmd_code_sampler;
        cs->rng.seed(params.seed == UINT32_MAX ? std::random_device{}() : params.seed);
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init(&mtmd_code_sampler_i, cs));
        if (!llama_set_sampler(bank->gen_code_ctx.get(), seq, chain)) {
            llama_sampler_free(chain);
            LOG_ERR("%s: the code predictor context does not take the code sampler\n", __func__);
            return nullptr;
        }
        st->lm_smpl.reset(chain);
        st->code_smpl = cs;
    } else {
        llama_gen_set_seed(bank->gen_ctx.get(), seq, params.seed);
    }
    bank->used[seq] = true;
    st->ctx  = ctx;
    st->bank = bank;
    st->seq  = seq;
    st->life = ctx->life;
    return st.release();
}

void mtmd_gen_state_free(mtmd_gen_state * state) {
    delete state;
}

// a new utterance: the decoder state goes, the RNG position and the speaker reference stay
void mtmd_gen_state_reset(mtmd_gen_state * state) {
    if (!state || !state->life || state->seq < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(state->life->mutex);
    if (!state->life->alive) {
        return;
    }
    // the codec slots of the native graphs, the decoder state of the engine
    llama_gen_reset(state->ctx->gen_native ? state->bank->gen_wav_ctx.get() : state->bank->gen_ctx.get(), state->seq);
}

int32_t mtmd_gen_state_set_ref(mtmd_context * ctx, mtmd_gen_state * state) {
    // the state holds the life of its context: a context at the address of a freed one has another
    if (!state || state->life != ctx->life) {
        LOG_ERR("%s: invalid gen state\n", __func__);
        return 1;
    }
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    llama_context * gctx = state->bank->gen_ctx.get();
    if (state->seq < 0 || !gctx || !ctx->last_ref.m || llama_gen_set_ref(gctx, state->seq, ctx->last_ref.m) != 0) {
        LOG_ERR("%s: the last encode produced no reference data\n", __func__);
        return 1;
    }
    return 0;
}

// the code predictor of the native graphs (Qwen3-TTS): one frame of the sequence, codes sampled in the graph
static int32_t mtmd_gen_code_native(mtmd_context * ctx, mtmd_gen_state * state, const mtmd_gen_inp * inp, mtmd_gen_out * out) {
    llama_context * lctx = state->bank->gen_code_ctx.get();
    GGML_ASSERT(state->code_smpl != nullptr);
    const int32_t n_embd = llama_model_n_embd_out(ctx->model.get());

    // the parameters of this call, read when the graph is built; temperature is ignored, as by the clip engine
    state->code_smpl->top_k = inp->top_k;
    state->code_smpl->top_p = inp->top_p;

    std::vector<float> h(n_embd, 0.0f);
    if (inp->embd) {
        std::copy(inp->embd, inp->embd + n_embd, h.begin());
    }

    llama_batch_ext * batch = llama_batch_ext_init(lctx);
    const int32_t idx = llama_batch_ext_add_token(batch, state->seq, inp->code0);
    bool ok = idx >= 0 && llama_batch_ext_set_embd_state(batch, idx, { h.data(), 1, (size_t) n_embd });
    ok = ok && llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
    llama_batch_ext_free(batch);
    if (!ok) {
        LOG_ERR("%s: the code predictor failed\n", __func__);
        return 1;
    }

    const int32_t n_codes = llama_get_sampled_tokens_ith(lctx, 0, nullptr, 0);
    state->out_codes.resize(n_codes);
    llama_get_sampled_tokens_ith(lctx, 0, state->out_codes.data(), n_codes);
    const float * embd = llama_get_embeddings_ith(lctx, 0);
    state->out_embd.assign(embd, embd + n_embd);
    state->out_feats.clear();

    out->embd    = state->out_embd.data();
    out->codes   = state->out_codes.data();
    out->n_codes = state->out_codes.size();
    return 0;
}

// the codec of the native graphs (Qwen3-TTS): frames of the sequence
static int32_t mtmd_gen_wav_native(mtmd_context * ctx, mtmd_gen_state * state, const mtmd_gen_inp * inp, mtmd_gen_out * out) {
    GGML_UNUSED(ctx);
    llama_context * wctx = state->bank->gen_wav_ctx.get();
    const int32_t n_cb = 16;
    if (!inp->codes || inp->n_codes == 0 || inp->n_codes % n_cb != 0) {
        LOG_ERR("%s: codes required for gen_wav\n", __func__);
        return 1;
    }
    if (inp->n_codes / n_cb > (size_t) INT32_MAX) {
        LOG_ERR("%s: too many frames\n", __func__);
        return 1;
    }
    int32_t n_frames = (int32_t) (inp->n_codes / n_cb);
    const int32_t spf      = llama_n_embd_out(wctx);

    // a short last window is padded to a full one, as the mtmd engine does, then trimmed: the graph shape
    // decides the CPU matmul kernels, so this keeps the PCM bit-identical (libllama takes any frame count)
    std::vector<llama_token> padded;
    const int32_t n_real = n_frames;
    const llama_token * codes_src = inp->codes;
    if (n_frames < 72) {
        padded.assign((size_t) 72 * n_cb, 0);
        std::copy(inp->codes, inp->codes + inp->n_codes, padded.begin());
        codes_src = padded.data();
        n_frames = 72;
    }

    llama_batch_ext * batch = llama_batch_ext_init(wctx);

    // every frame is checked before a window runs: a bad code must not leave the earlier windows in the codec state
    for (int32_t f = 0; f < n_real; ++f) {
        llama_batch_ext_clear(batch);
        if (llama_batch_ext_add_codes(batch, state->seq, inp->codes + (size_t) f * n_cb, n_cb) < 0) {
            llama_batch_ext_free(batch);
            LOG_ERR("%s: frame %d: invalid codes\n", __func__, f);
            return 1;
        }
    }

    state->out_audio.clear();
    for (int64_t f0 = 0; f0 < n_frames; f0 += llama_n_batch(wctx)) {
        const int32_t n = (int32_t) std::min<int64_t>(llama_n_batch(wctx), n_frames - f0);
        llama_batch_ext_clear(batch);
        for (int32_t f = 0; f < n; ++f) {
            if (llama_batch_ext_add_codes(batch, state->seq, codes_src + (size_t) (f0 + f) * n_cb, n_cb) < 0) {
                llama_batch_ext_free(batch);
                return 1;
            }
        }
        if (llama_process(wctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            llama_batch_ext_free(batch);
            LOG_ERR("%s: libllama code2wav failed\n", __func__);
            return 1;
        }
        for (int32_t f = 0; f < n && f0 + f < n_real; ++f) {
            const float * pcm = llama_get_embeddings_ith(wctx, f);
            state->out_audio.insert(state->out_audio.end(), pcm, pcm + spf);
        }
    }
    llama_batch_ext_free(batch);

    out->audio     = state->out_audio.data();
    out->n_samples = state->out_audio.size();
    return 0;
}

static int32_t mtmd_gen_audio_process_impl(mtmd_context * ctx, mtmd_gen_state * state, const mtmd_gen_inp * inp, mtmd_gen_out * out) {
    *out = {};

    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    if (ctx->gen_native) {
        return inp->type == MTMD_GEN_PROCESS_TYPE_GEN_CODE ? mtmd_gen_code_native(ctx, state, inp, out)
                                                           : mtmd_gen_wav_native(ctx, state, inp, out);
    }

    llama_context * lctx = state->bank->gen_ctx.get();
    llama_gen_inp li{};
    li.type                = inp->type == MTMD_GEN_PROCESS_TYPE_GEN_WAV ? LLAMA_GEN_STAGE_WAV : LLAMA_GEN_STAGE_CODE;
    li.code0               = inp->code0;
    li.embd                = inp->embd;
    li.top_k               = inp->top_k;
    li.top_p               = inp->top_p;
    li.temp                = inp->temp;
    li.penalty_repeat      = inp->penalty_repeat;
    li.penalty_last_n      = inp->penalty_last_n;
    li.codes               = inp->codes;
    li.n_codes             = inp->n_codes;
    li.feats               = inp->feats;
    li.n_feats             = inp->n_feats;
    li.backbone_state      = inp->backbone_state;
    li.backbone_state_size = inp->backbone_state_size;
    li.backbone_n_past     = inp->backbone_n_past;
    li.n_steps             = inp->n_steps;

    llama_batch_ext * batch = llama_batch_ext_init(lctx);
    const int32_t res = llama_batch_ext_add_gen(batch, state->seq, &li) < 0 ? -1 : llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    llama_gen_out lo{};
    if (res != 0 || !llama_get_gen_out_ith(lctx, 0, &lo)) {
        LOG_ERR("%s: gen step failed (%d)\n", __func__, res);
        return 1;
    }
    state->out_codes.assign(lo.codes, lo.codes + lo.n_codes);
    state->out_feats.assign(lo.feats, lo.feats + lo.n_feats);
    state->out_audio.assign(lo.audio, lo.audio + lo.n_samples);
    state->out_embd .assign(lo.embd,  lo.embd  + lo.n_embd);
    out->codes     = state->out_codes.empty() ? nullptr : state->out_codes.data();
    out->n_codes   = state->out_codes.size();
    out->feats     = state->out_feats.empty() ? nullptr : state->out_feats.data();
    out->n_feats   = state->out_feats.size();
    out->embd      = state->out_embd.empty()  ? nullptr : state->out_embd.data();
    out->is_eos    = lo.is_eos;
    out->audio     = state->out_audio.empty() ? nullptr : state->out_audio.data();
    out->n_samples = state->out_audio.size();
    return 0;
}

int32_t mtmd_gen_audio_process(mtmd_context * ctx, mtmd_gen_state * state, const struct mtmd_gen_inp * inp, struct mtmd_gen_out * out) {
    if (!ctx->info.has_gen_audio || !state) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return 1;
    }
    // the state holds the life of its context: a context at the address of a freed one has another
    if (state->life != ctx->life) {
        LOG_ERR("%s: the gen state belongs to another context\n", __func__);
        return 1;
    }
    try {
        return mtmd_gen_audio_process_impl(ctx, state, inp, out);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

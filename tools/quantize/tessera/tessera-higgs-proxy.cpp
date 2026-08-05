//
// tessera-higgs-proxy.cpp
//
// HIGGS Linearity Theorem per-layer alpha_l estimator - structural
// proxy (C++ first-class port). See tessera-higgs-proxy.h and
// docs/tessera-ane-ios-demo-design.md Phase 3.5 for the design.
//

#include "tessera-higgs-proxy.h"

#include "ggml.h"
#include "gguf.h"

// TILE640 quant/dequant helpers (the same dispatch the
// GGML_OP_TILE640_MATMUL inference path drives): the C reference
// quantizer for deterministic packing, the v2 (Accelerate + NEON)
// dequant + the v2 dispatch cost model for the L1 measurement.
#include "ggml-quants.h"
#include "ggml-quants-v2.h"
#include "ggml-quants-v2-dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Family prior table (mirrors the NumPy FAMILY_PRIOR + FAMILY_SUFFIXES
// tables EXACTLY - same suffix map, same float values, same "other"
// fallback). The order matters for the suffix match: attn_output is
// checked before attn_q so that a name like blk.16.attn_output.weight
// does not misclassify as attn_q.
// ---------------------------------------------------------------------------

const ts_higgs_proxy_family_entry TS_HIGGS_PROXY_FAMILY_SUFFIXES[] = {
    { "attn_k",      "attn_k",      1.30f },
    { "attn_v",      "attn_v",      1.30f },
    { "attn_output", "attn_output", 1.00f },
    { "attn_q",      "attn_q",      0.85f },
    { "ffn_down",    "ffn_down",    0.55f },
    { "ffn_gate",    "ffn_gate",    0.45f },
    { "ffn_up",      "ffn_up",      0.45f },
    { "attn_norm",   "norm",        0.70f },
    { "ffn_norm",    "norm",        0.70f },
    { "token_embd",  "token_embd",  0.60f },
    { "output",      "output",      0.60f },
};
const int TS_HIGGS_PROXY_FAMILY_SUFFIXES_COUNT =
    (int)(sizeof(TS_HIGGS_PROXY_FAMILY_SUFFIXES) /
          sizeof(TS_HIGGS_PROXY_FAMILY_SUFFIXES[0]));

// Build a name -> prior map for the family_prior() lookup. The
// "other" key is added at the end so unknown families fall back to
// the conservative middle value. The values match the Python
// FAMILY_PRIOR dict exactly.
static const std::vector<std::pair<std::string, float>> &
ts_higgs_proxy_family_prior_table() {
    static const std::vector<std::pair<std::string, float>> table = []() {
        std::vector<std::pair<std::string, float>> * v =
            new std::vector<std::pair<std::string, float>>();
        v->reserve(TS_HIGGS_PROXY_FAMILY_SUFFIXES_COUNT + 1);
        for (int i = 0; i < TS_HIGGS_PROXY_FAMILY_SUFFIXES_COUNT; i++) {
            v->emplace_back(TS_HIGGS_PROXY_FAMILY_SUFFIXES[i].family,
                            TS_HIGGS_PROXY_FAMILY_SUFFIXES[i].prior);
        }
        // The Python FAMILY_PRIOR also has explicit "other" = 0.50; mirror
        // that for a true one-source-of-truth family->prior map.
        v->emplace_back("other", TS_HIGGS_PROXY_FAMILY_OTHER_PRIOR);
        return *v;
    }();
    return table;
}

std::string ts_higgs_proxy_classify_family(const std::string & name) {
    // Strip a trailing ".weight" or ".bias" so the suffix match
    // works for both "blk.16.attn_v.weight" and "blk.16.attn_v".
    std::string stem = name;
    for (const std::string trailing : { ".weight", ".bias" }) {
        if (stem.size() > trailing.size() &&
            stem.compare(stem.size() - trailing.size(),
                         trailing.size(), trailing) == 0) {
            stem.resize(stem.size() - trailing.size());
            break;
        }
    }
    for (int i = 0; i < TS_HIGGS_PROXY_FAMILY_SUFFIXES_COUNT; i++) {
        const char * suf = TS_HIGGS_PROXY_FAMILY_SUFFIXES[i].suffix;
        if (stem == suf) {
            return TS_HIGGS_PROXY_FAMILY_SUFFIXES[i].family;
        }
        // Match by dotted-suffix ("." + suf), so "blk.16.attn_v"
        // matches "attn_v" but "myattn_v" does not.
        const size_t n = strlen(suf);
        if (stem.size() > n + 1 &&
            stem[stem.size() - n - 1] == '.' &&
            stem.compare(stem.size() - n, n, suf) == 0) {
            return TS_HIGGS_PROXY_FAMILY_SUFFIXES[i].family;
        }
    }
    return std::string("other");
}

float ts_higgs_proxy_family_prior(const std::string & family) {
    for (const auto & kv : ts_higgs_proxy_family_prior_table()) {
        if (kv.first == family) {
            return kv.second;
        }
    }
    return TS_HIGGS_PROXY_FAMILY_OTHER_PRIOR;
}

// ---------------------------------------------------------------------------
// SHA-256 (self-contained, FIPS 180-4)
//
// Borrowed in spirit from tessera-higgs-cache.cpp's sha256_ctx; the
// inlined copy keeps the proxy module dependency-free (no openssl,
// no system crypto). The output is identical to the Python hashlib
// implementation byte-for-byte (it is a pure standard).
// ---------------------------------------------------------------------------

namespace {

struct ts_sha256_ctx {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buf[64];
};

static const uint32_t ts_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t ts_sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void ts_sha256_transform(ts_sha256_ctx * ctx, const uint8_t * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t) block[i * 4 + 0] << 24) |
               ((uint32_t) block[i * 4 + 1] << 16) |
               ((uint32_t) block[i * 4 + 2] <<  8) |
               ((uint32_t) block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = (ts_sha256_rotr(w[i-2], 17) ^ ts_sha256_rotr(w[i-2], 19) ^ (w[i-2] >> 10))
             + w[i-7]
             + (ts_sha256_rotr(w[i-15], 7) ^ ts_sha256_rotr(w[i-15], 18) ^ (w[i-15] >> 3))
             + w[i-16];
    }

    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ts_sha256_rotr(e, 6) ^ ts_sha256_rotr(e, 11) ^ ts_sha256_rotr(e, 25))
                    + ((e & f) ^ (~e & g)) + ts_sha256_k[i] + w[i];
        uint32_t t2 = (ts_sha256_rotr(a, 2) ^ ts_sha256_rotr(a, 13) ^ ts_sha256_rotr(a, 22))
                    + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void ts_sha256_init(ts_sha256_ctx * ctx) {
    // FIPS 180-4 initial hash values for SHA-256. These match
    // hashlib.sha256() in CPython (which uses OpenSSL under the
    // hood) byte-for-byte.
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void ts_sha256_update(ts_sha256_ctx * ctx, const uint8_t * data, size_t len) {
    size_t buffered = (size_t)(ctx->count % 64);
    ctx->count += len;

    if (buffered > 0) {
        size_t space = 64 - buffered;
        size_t n = len < space ? len : space;
        memcpy(ctx->buf + buffered, data, n);
        data += n;
        len  -= n;
        if (buffered + n == 64) {
            ts_sha256_transform(ctx, ctx->buf);
        }
    }
    while (len >= 64) {
        ts_sha256_transform(ctx, data);
        data += 64;
        len  -= 64;
    }
    if (len > 0) {
        memcpy(ctx->buf, data, len);
    }
}

static void ts_sha256_final(ts_sha256_ctx * ctx, uint8_t out[32]) {
    uint64_t bits = ctx->count * 8;
    uint8_t pad[64] = { 0x80 };
    size_t pad_len = (ctx->count % 64 < 56) ? (56 - ctx->count % 64)
                                            : (120 - ctx->count % 64);
    ts_sha256_update(ctx, pad, pad_len);
    uint8_t lenbuf[8];
    for (int i = 0; i < 8; i++) {
        lenbuf[i] = (uint8_t)(bits >> (56 - 8 * i));
    }
    ts_sha256_update(ctx, lenbuf, 8);

    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->state[i] >>  8);
        out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

}  // namespace

std::string ts_higgs_proxy_model_hash(const char * gguf_path) {
    if (!gguf_path) return std::string();
    int fd = ::open(gguf_path, O_RDONLY);
    if (fd < 0) return std::string();
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return std::string();
    }
    const int64_t file_size = (int64_t)st.st_size;
    const int64_t window    = 64 * 1024;

    ts_sha256_ctx ctx;
    ts_sha256_init(&ctx);

    if (file_size <= window) {
        std::vector<uint8_t> buf((size_t)file_size);
        ssize_t r = ::read(fd, buf.data(), (size_t)file_size);
        if (r == file_size) {
            ts_sha256_update(&ctx, buf.data(), (size_t)r);
        }
    } else {
        std::vector<uint8_t> header((size_t)window);
        ssize_t h = ::read(fd, header.data(), (size_t)window);
        if (h == window) {
            ts_sha256_update(&ctx, header.data(), (size_t)h);
        }
        if (::lseek(fd, -window, SEEK_END) >= 0) {
            std::vector<uint8_t> tail((size_t)window);
            ssize_t t = ::read(fd, tail.data(), (size_t)window);
            if (t == window) {
                ts_sha256_update(&ctx, tail.data(), (size_t)t);
            }
        }
    }
    ::close(fd);

    uint8_t out[32];
    ts_sha256_final(&ctx, out);

    // Lowercase hex, first 16 chars (matches the Python model_hash).
    static const char hex[] = "0123456789abcdef";
    std::string hex16;
    hex16.reserve(16);
    for (int i = 0; i < 8; i++) {
        hex16.push_back(hex[(out[i] >> 4) & 0xF]);
        hex16.push_back(hex[out[i] & 0xF]);
    }
    return hex16;
}

// ---------------------------------------------------------------------------
// Default t_l^2 measurement function: offline ternary MSE proxy
//
// Mirrors the Python `measure_t_squared_offline` byte-for-byte
// (within float-precision) for a given F32 reference. The per-tensor
// ternary scale is the mean absolute value of the reference; the
// reconstruction is round-to-nearest ternary * scale. The relative
// Frobenius error is the squared distance divided by the squared
// reference norm. Returns 0.0 for a zero-norm / empty / NaN input.
// ---------------------------------------------------------------------------

float ts_higgs_proxy_measure_offline(const float * W_flat, int64_t n_elem,
                                     int64_t layer_idx, void * ctx) {
    (void)layer_idx; (void)ctx;
    if (!W_flat || n_elem <= 0) {
        return 0.0f;
    }

    // First pass: per-tensor mean(|W|) = the ternary scale.
    double abs_sum = 0.0;
    int64_t nan_count = 0;
    for (int64_t i = 0; i < n_elem; i++) {
        float v = W_flat[i];
        if (std::isnan(v) || std::isinf(v)) {
            nan_count++;
            continue;
        }
        abs_sum += (double)std::fabs(v);
    }
    if (nan_count > 0) {
        // NaN/Inf-bearing reference: stamp the value the caller
        // recorded (the sidecar carries the raw float so the L5
        // orchestrator can decide). We do NOT filter; the math
        // is well-defined on the surviving finite values.
    }
    const double scale_d = abs_sum / (double)n_elem;
    if (!(scale_d > 0.0)) {
        return 0.0f;
    }
    const float scale = (float)scale_d;
    const float half_thresh = 0.5f * scale;

    // Second pass: round to ternary, accumulate the squared
    // reconstruction error in F64 (the relative Frobenius
    // denominator is also F64, so the relative error is in F64
    // then cast to F32 to match the Python return type).
    double err_sq = 0.0;
    double ref_sq = 0.0;
    for (int64_t i = 0; i < n_elem; i++) {
        float v = W_flat[i];
        if (std::isnan(v) || std::isinf(v)) {
            continue;
        }
        float q = 0.0f;
        if (v > half_thresh)      q = scale;
        else if (v < -half_thresh) q = -scale;
        // else q stays 0 (the "below-threshold / dead weight" case).
        const float diff = q - v;
        err_sq += (double)diff * (double)diff;
        ref_sq += (double)v * (double)v;
    }
    if (!(ref_sq > 0.0)) {
        return 0.0f;
    }
    float t2 = (float)(err_sq / ref_sq);
    if (std::isnan(t2) || std::isinf(t2)) {
        return 0.0f;
    }
    return t2;
}

// ---------------------------------------------------------------------------
// L1-on-ANE t_l^2 measurement (the Phase 0 source)
//
// The packed_W contract is the concatenated per-row flat TILE640
// row layout [packed words | fp16 page_scales | int8 lane_scales]
// produced by ts_higgs_proxy_pack_tile640. The per-row dequant
// mirrors the GGML_OP_TILE640_MATMUL dispatch in ggml-ane.mm:
// v2 dequant with pre-decoded meta at in_dim >= MIN_K (and v2
// enabled), the C reference below the cutoff; the meta decode
// follows the v2 dispatch cost model (always the C reference
// today). The dequantized row is round-tripped through fp16 (the
// bundle's pinned slot dtype - the F16 pre-dequant) before the
// abs-diff, so the measurement captures the ternary quantization
// error AND the ANE fp16 precision loss.
// ---------------------------------------------------------------------------

namespace {

// Per-row flat TILE640 byte size for pages pages (matches the
// ggml-ane.mm row_bytes formula: words * 4 + page_scales * 2 +
// lane_scales).
size_t ts_higgs_proxy_t640_row_bytes(int64_t pages) {
    return (size_t)pages * (TILE640_WORDS_PER_PAGE * sizeof(uint32_t)
                            + sizeof(uint16_t)
                            + TILE640_LANES_PER_PAGE * sizeof(int8_t));
}

}  // namespace

void ts_higgs_proxy_pack_tile640(const float * W_flat,
                                 int64_t out_dim, int64_t in_dim,
                                 std::vector<uint8_t> & packed) {
    packed.clear();
    if (!W_flat || out_dim <= 0 || in_dim <= 0) {
        return;
    }
    const int64_t pages = (in_dim + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
    const size_t row_bytes = ts_higgs_proxy_t640_row_bytes(pages);
    packed.resize((size_t)out_dim * row_bytes);
    for (int64_t r = 0; r < out_dim; r++) {
        quantize_row_tessera_t640_ref(W_flat + (size_t)r * in_dim,
                                      packed.data() + (size_t)r * row_bytes,
                                      in_dim);
    }
}

float ts_higgs_proxy_measure_l1(const float * W_orig_flat,
                                const void * packed_W,
                                int64_t out_dim, int64_t in_dim,
                                int64_t layer_idx, void * ctx) {
    (void)layer_idx; (void)ctx;
    if (!W_orig_flat || !packed_W || out_dim <= 0 || in_dim <= 0) {
        return 0.0f;
    }

    const int64_t pages = (in_dim + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
    const size_t row_bytes = ts_higgs_proxy_t640_row_bytes(pages);

    // The GGML_OP_TILE640_MATMUL dispatch rule (ggml-ane.mm): v2
    // dequant iff v2 enabled and in_dim >= MIN_K; the meta decode
    // goes through the v2 cost model (returns false today, so the
    // C reference decode runs; the branch tracks future cost-model
    // changes without drift).
    const bool use_v2 = ggml_tessera_t640_v2_enabled() &&
                        in_dim >= GGML_TESSERA_T640_V2_MIN_K;
    const bool use_v2_meta = ts_v2_dispatch_should_use_v2_meta(out_dim, pages);

    // Per-row scratch (aligned, like the dispatch's row buffer).
    std::vector<uint8_t> row_buf(row_bytes);
    std::vector<float> row_y((size_t)in_dim);
    std::vector<float> page_max((size_t)pages);
    std::vector<float> lane_scale((size_t)pages * TILE640_LANES_PER_PAGE);

    const uint8_t * base = (const uint8_t *)packed_W;
    double l1_sum = 0.0;
    float max_abs = 0.0f;

    for (int64_t r = 0; r < out_dim; r++) {
        std::memcpy(row_buf.data(), base + (size_t)r * row_bytes, row_bytes);
        if (use_v2) {
            const uint32_t * words = (const uint32_t *)row_buf.data();
            const uint16_t * ps = (const uint16_t *)
                (row_buf.data() + (size_t)pages * TILE640_WORDS_PER_PAGE * sizeof(uint32_t));
            const int8_t * ls = (const int8_t *)(ps + pages);
            if (use_v2_meta) {
                decode_per_row_meta_v2(ps, ls, 1, pages,
                                       page_max.data(), lane_scale.data());
            } else {
                ts_decode_per_row_meta_ref(ps, ls, 1, pages,
                                           page_max.data(), lane_scale.data());
            }
            dequantize_row_tessera_t640_v2(words, page_max.data(),
                                           lane_scale.data(), in_dim,
                                           row_y.data());
        } else {
            dequantize_row_tessera_t640(row_buf.data(), row_y.data(), in_dim);
        }

        const float * orig = W_orig_flat + (size_t)r * in_dim;
        for (int64_t c = 0; c < in_dim; c++) {
            const float w = orig[c];
            if (std::isnan(w) || std::isinf(w)) {
                // Same convention as measure_offline: no rejection of the
                // tensor, the math is well-defined on the surviving finite
                // values.
                continue;
            }
            const float a = std::fabs(w);
            if (a > max_abs) max_abs = a;
            // F16 pre-dequant round-trip: the ANE bundle consumes the
            // dequantized weight as fp16 (ggml_fp32_to_fp16 compiles to
            // NEON vcvt_f32_f16 on Apple Silicon).
            const float w_deq = ggml_fp16_to_fp32(ggml_fp32_to_fp16(row_y[(size_t)c]));
            l1_sum += (double)std::fabs(w - w_deq);
        }
    }

    if (!(max_abs > 0.0f)) {
        return 0.0f;
    }
    const double t2 = (l1_sum / (double)(out_dim * in_dim)) / (double)max_abs;
    if (std::isnan(t2) || std::isinf(t2)) {
        return 0.0f;
    }
    return (float)t2;
}

// ---------------------------------------------------------------------------
// GGUF reading + per-tensor dequant
//
// The dequant path uses the ggml type-traits table, the same path
// ts_tensor_to_f32 in tessera-dispatch.cpp uses. This handles F32,
// F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, and any other type that has
// registered a to_float trait.
// ---------------------------------------------------------------------------

namespace {

struct gguf_image {
    void *           map = nullptr;
    size_t           map_size = 0;
    gguf_context *   ctx = nullptr;
    ggml_context *   gctx = nullptr;
    ~gguf_image() {
        if (gctx) ggml_free(gctx);
        if (ctx)  gguf_free(ctx);
        if (map && map_size > 0) {
            ::munmap(map, map_size);
        }
    }
};

bool ts_open_gguf(const char * path, gguf_image & img, std::string & err) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        err = std::string("open failed: ") + path;
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        err = "fstat failed";
        return false;
    }
    img.map_size = (size_t)st.st_size;
    img.map = ::mmap(nullptr, img.map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (img.map == MAP_FAILED) {
        img.map = nullptr;
        img.map_size = 0;
        err = "mmap failed";
        return false;
    }
    // gguf_init_from_file does its own fopen-based parse, so we
    // can use it here (it does not consume the mmap). We then
    // patch tensor data pointers into the mmap region for the
    // ggml side.
    struct gguf_init_params gparams = {
        /*no_alloc=*/ true,
        /*ctx     =*/ &img.gctx,
    };
    img.ctx = gguf_init_from_file(path, gparams);
    if (!img.ctx || !img.gctx) {
        err = "gguf_init_from_file failed";
        return false;
    }
    const size_t data_off = gguf_get_data_offset(img.ctx);
    const int64_t n_t = gguf_get_n_tensors(img.ctx);
    for (int64_t i = 0; i < n_t; i++) {
        const char * tname = gguf_get_tensor_name(img.ctx, i);
        size_t toff = gguf_get_tensor_offset(img.ctx, i);
        ggml_tensor * t = ggml_get_tensor(img.gctx, tname);
        if (t) {
            t->data = (char *)img.map + data_off + toff;
        }
    }
    return true;
}

bool ts_dequant_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    out.assign((size_t)n, 0.0f);
    if (t->data == nullptr) {
        return false;
    }
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (traits && traits->to_float) {
        traits->to_float(t->data, out.data(), n);
        return true;
    }
    return false;
}

const char * ts_dtype_name(ggml_type t) {
    // ggml_type_name returns a stable string; cache the name into a
    // small static table so the sidecar carries a uniform short
    // form (e.g. "F16" not "GGML_TYPE_F16"). The names match the
    // Python gguf.constants.GGMLQuantizationType.name output.
    switch (t) {
        case GGML_TYPE_F32:     return "F32";
        case GGML_TYPE_F16:     return "F16";
        case GGML_TYPE_Q4_0:    return "Q4_0";
        case GGML_TYPE_Q4_1:    return "Q4_1";
        case GGML_TYPE_Q5_0:    return "Q5_0";
        case GGML_TYPE_Q5_1:    return "Q5_1";
        case GGML_TYPE_Q8_0:    return "Q8_0";
        case GGML_TYPE_Q8_1:    return "Q8_1";
        case GGML_TYPE_Q2_K:    return "Q2_K";
        case GGML_TYPE_Q3_K:    return "Q3_K";
        case GGML_TYPE_Q4_K:    return "Q4_K";
        case GGML_TYPE_Q5_K:    return "Q5_K";
        case GGML_TYPE_Q6_K:    return "Q6_K";
        case GGML_TYPE_Q8_K:    return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        default:                return ggml_type_name(t);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

int ts_higgs_proxy_estimate(const char * gguf_path,
                            const ts_higgs_proxy_params * params,
                            ts_higgs_proxy_measurement_fn measurement_fn,
                            void * measurement_ctx,
                            ts_higgs_proxy_result * result) {
    if (!gguf_path || !result) {
        return -1;
    }

    // Default-fill the params. The proxy is single-sample (J=1) by
    // definition; t_min/t_max/seed are present for parity with
    // ts_higgs_params but unused.
    ts_higgs_proxy_params p;
    if (params) {
        p = *params;
    } else {
        p = {};
    }
    if (p.J < 1) p.J = 1;
    if (p.t_min <= 0.0f) p.t_min = 0.01f;
    if (p.t_max <= p.t_min) p.t_max = 0.10f;
    if (p.alpha_floor <= 0.0f) p.alpha_floor = 1e-6f;
    if (p.seed == 0) p.seed = 42u;
    if (p.min_params_for_estimate < 0) p.min_params_for_estimate = 1000000000LL;
    if (!(p.alpha_floor_fraction > 0.0f)) p.alpha_floor_fraction = 1e-3f;
    if (p.alpha_floor_fraction >= 1.0f) p.alpha_floor_fraction = 1e-3f;

    // The default t_l^2 measurement is the L1-on-ANE kernel dequant
    // path. TS_HIGGS_PROXY_LEGACY_OFFLINE=1 restores the legacy
    // offline ternary MSE default. A caller-supplied measurement_fn
    // keeps the legacy behavior bit-identical.
    const bool caller_fn = (measurement_fn != nullptr);
    const char * legacy_env = std::getenv("TS_HIGGS_PROXY_LEGACY_OFFLINE");
    const bool legacy_offline = legacy_env && legacy_env[0] == '1';
    const bool use_l1_default = !caller_fn && !legacy_offline;
    if (!measurement_fn) measurement_fn = ts_higgs_proxy_measure_offline;

    // Reset the result.
    result->layers.clear();
    result->n_valid = 0;
    result->n_fallback_uniform = 0;
    result->mean_alpha = 0.0f;
    result->t_squared_source = use_l1_default ? "l1_kernel_dequant"
                                              : "offline_ternary_mse";

    gguf_image img;
    std::string err;
    if (!ts_open_gguf(gguf_path, img, err)) {
        std::fprintf(stderr, "tessera-higgs-proxy: %s\n", err.c_str());
        return 1;
    }

    result->model_hash = ts_higgs_proxy_model_hash(gguf_path);

    const int64_t n_t = gguf_get_n_tensors(img.ctx);

    // Pass 1: dequant, classify family, accumulate n_elem and
    // frobenius_norm_sq, and run the t_l^2 measurement. The
    // structural alpha is family-prior-based, so we don't need
    // per-tensor Frobenius for the alpha calc, but we record it
    // in the sidecar for diagnostics.
    int64_t total_params = 0;
    int64_t measured_count = 0;
    std::vector<uint8_t> packed;  // L1 path TILE640 scratch, reused per tensor
    for (int64_t i = 0; i < n_t; i++) {
        const char * tname = gguf_get_tensor_name(img.ctx, i);
        ggml_tensor * t = ggml_get_tensor(img.gctx, tname);
        if (!t) continue;
        const int64_t ne = ggml_nelements(t);
        if (ne < 32) continue;  // skip biases / tiny norm tensors

        std::vector<float> f32;
        if (!ts_dequant_to_f32(t, f32)) {
            // Unknown dtype: skip the tensor, log a warning to
            // stderr. The sidecar records the skip; the
            // fallback flag is set to "none" (the tensor is
            // genuinely unknown, not a per-layer uniform
            // fallback).
            std::fprintf(stderr,
                "tessera-higgs-proxy: skipping %s (dtype %s, no to_float)\n",
                tname, ts_dtype_name(t->type));
            continue;
        }
        if ((int64_t)f32.size() != ne) {
            continue;
        }

        // Frobenius norm squared (single precision sum-of-squares).
        double fro2 = 0.0;
        for (int64_t k = 0; k < ne; k++) {
            double v = (double)f32[(size_t)k];
            fro2 += v * v;
        }
        if (!(fro2 > 0.0)) {
            // All-zero reference: skip, the consumer has no
            // useful signal from this tensor.
            continue;
        }

        const std::string family = ts_higgs_proxy_classify_family(tname);
        float t_sq;
        if (use_l1_default) {
            // The flat tensor is row-major [out_dim, in_dim]; the L1
            // measurement packs each row to TILE640 and dequantizes via
            // the same dispatch the GGML_OP_TILE640_MATMUL path uses.
            const int64_t in_dim  = t->ne[0];
            const int64_t out_dim = ne / in_dim;
            ts_higgs_proxy_pack_tile640(f32.data(), out_dim, in_dim, packed);
            t_sq = ts_higgs_proxy_measure_l1(f32.data(), packed.data(),
                                             out_dim, in_dim,
                                             measured_count, nullptr);
        } else {
            t_sq = measurement_fn(f32.data(), ne, measured_count,
                                  measurement_ctx);
        }

        // Per-tensor record (alpha filled in after the
        // post-normalization step below).
        ts_higgs_proxy_layer_result lr;
        lr.name = tname;
        lr.family = family;
        const int n_dims = ggml_n_dims(t);
        lr.shape.resize((size_t)n_dims);
        for (int d = 0; d < n_dims; d++) {
            lr.shape[(size_t)d] = (int64_t)t->ne[d];
        }
        lr.n_elem = ne;
        lr.frobenius_norm_sq = (float)fro2;
        lr.t_squared = t_sq;
        lr.t_squared_source = use_l1_default ? "l1_kernel_dequant"
                                             : "offline_ternary_mse";
        lr.dtype_source = ts_dtype_name(t->type);
        lr.alpha_l = 0.0f;  // filled in after normalization
        lr.alpha_floor_applied = false;
        lr.fallback = "none";

        if (p.verbose) {
            std::fprintf(stderr,
                "  higgs-proxy: %-32s family=%-12s n=%ld t2=%.6e\n",
                lr.name.c_str(), lr.family.c_str(),
                (long)lr.n_elem, (double)lr.t_squared);
        }

        result->layers.push_back(std::move(lr));
        total_params += ne;
        measured_count++;
    }

    // Pass 2: compute the raw structural alpha per tensor and the
    // global uniform fallback decision. The structural form is the
    // family-prior value alone (the research doc drops the
    // (||W||_F^2 / 2) multiplier; see the structural_alpha
    // docstring in the Python module).
    std::vector<float> raw_alphas;
    raw_alphas.reserve(result->layers.size());
    for (size_t i = 0; i < result->layers.size(); i++) {
        const std::string & fam = result->layers[(size_t)i].family;
        raw_alphas.push_back(ts_higgs_proxy_family_prior(fam));
    }

    // Normalize so the positive mean is 1.0. The Python
    // implementation skips zero entries when computing the mean
    // (they are family-prior "other" entries that may have been
    // set to 0 in the future; today every family key has a
    // positive value).
    const bool fallback_global = (total_params < p.min_params_for_estimate);

    double pos_sum = 0.0;
    int64_t pos_count = 0;
    for (size_t i = 0; i < raw_alphas.size(); i++) {
        if (raw_alphas[i] > 0.0f) {
            pos_sum += raw_alphas[i];
            pos_count++;
        }
    }
    double mean_pos = (pos_count > 0) ? (pos_sum / (double)pos_count) : 1.0;
    double scale = (mean_pos > 0.0) ? (1.0 / mean_pos) : 1.0;

    // Pass 3: apply the positive floor and emit the final
    // per-tensor alpha. The floor is alpha_floor_fraction of the
    // post-normalization mean (which is 1.0 by construction), so
    // the floor is just the fraction; the absolute floor
    // (alpha_floor, default 1e-6) protects the GA's
    // divide-by-zero for very-small tensors.
    const float floor = p.alpha_floor_fraction;
    const float abs_floor = p.alpha_floor;
    int64_t n_floor_applied = 0;
    for (size_t i = 0; i < result->layers.size(); i++) {
        ts_higgs_proxy_layer_result & lr = result->layers[i];
        if (fallback_global) {
            lr.alpha_l = 1.0f;
            lr.alpha_floor_applied = false;
            lr.fallback = "global_uniform";
        } else {
            double alpha_d = (double)raw_alphas[i] * scale;
            if (!(alpha_d > 0.0)) {
                alpha_d = (double)floor;
            }
            if (alpha_d < (double)floor) {
                alpha_d = (double)floor;
                lr.alpha_floor_applied = true;
                n_floor_applied++;
                lr.fallback = "per_layer_uniform";
            }
            if (alpha_d < (double)abs_floor) {
                alpha_d = (double)abs_floor;
                lr.alpha_floor_applied = true;
                lr.fallback = "per_layer_uniform";
                n_floor_applied++;
            }
            lr.alpha_l = (float)alpha_d;
        }
    }

    // Aggregate counters. The Python implementation stamps
    // n_valid = number of layers with non-clamped alpha and
    // n_fallback_uniform = layers that fell back to uniform.
    // The proxy's "valid" notion is the structural prior, so
    // n_valid = layers NOT in fallback. The per-layer-uniform
    // counter mirrors the Python `n_fallback_uniform` field.
    double mean_sum = 0.0;
    for (size_t i = 0; i < result->layers.size(); i++) {
        mean_sum += result->layers[i].alpha_l;
    }
    result->mean_alpha = (result->layers.size() > 0)
        ? (float)(mean_sum / (double)result->layers.size())
        : 0.0f;
    result->n_valid = (int64_t)result->layers.size() - n_floor_applied;
    result->n_fallback_uniform = n_floor_applied;

    if (fallback_global) {
        result->t_squared_source = "uniform_fallback";
    }

    return 0;
}

// ---------------------------------------------------------------------------
// JSON I/O
//
// The JSON shape mirrors the Python build_sidecar / write_sidecar
// output byte-for-byte at the key-order + value-type level. Float
// formatting uses std::to_string (default precision 6 significant
// digits) and the parity test normalizes both Python and C++
// outputs to 6 decimal places before comparing.
// ---------------------------------------------------------------------------

namespace {

std::string ts_format_int(int64_t v) {
    return std::to_string(v);
}

std::string ts_format_float(float v) {
    // std::to_string uses 6 significant digits by default, which
    // matches the Python repr() output for the same float. Where
    // the values differ at the trailing bits the parity test
    // normalizes to 6 decimal places.
    if (std::isnan(v)) return "\"NaN\"";
    if (std::isinf(v)) return v > 0 ? "\"Inf\"" : "\"-Inf\"";
    return std::to_string(v);
}

std::string ts_escape_json(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  (unsigned)(unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace

std::string ts_higgs_proxy_to_json(const ts_higgs_proxy_result * result,
                                   const char * gguf_path,
                                   const char * bundle_name) {
    if (!result) return std::string("{}\n");

    // bundle_name default: GGUF file stem. Matches the Python
    // build_sidecar's `bundle_name or gguf_path.stem` rule.
    std::string bn;
    if (bundle_name && *bundle_name) {
        bn = bundle_name;
    } else if (gguf_path && *gguf_path) {
        std::string p(gguf_path);
        size_t slash = p.find_last_of("/\\");
        std::string stem = (slash == std::string::npos) ? p : p.substr(slash + 1);
        size_t dot = stem.find_last_of('.');
        bn = (dot == std::string::npos) ? stem : stem.substr(0, dot);
    } else {
        bn = "unknown";
    }

    // Top-level measurement string: the Python build_sidecar
    // pulls this from the audit["measurement"] field, which the
    // Python estimator stamps as "offline_ternary_mse" (the
    // legacy default) or "uniform_fallback" (global fallback).
    // The C++ proxy stamps "l1_kernel_dequant" (the default),
    // "offline_ternary_mse" (legacy opt-in / custom fn), or
    // "uniform_fallback". result.t_squared_source is already in
    // that enum.
    const std::string measurement = result->t_squared_source;
    const bool fallback_global = (measurement == "uniform_fallback");

    std::string s;
    s.reserve(8192);
    s += "{\n";
    s += "  \"schema\": \"ane.alpha-coefficients.v1\",\n";
    s += "  \"version\": 1,\n";
    s += "  \"bundle_name\": \"" + ts_escape_json(bn) + "\",\n";
    s += "  \"gguf_path\": \"" + ts_escape_json(gguf_path ? gguf_path : "") + "\",\n";
    s += "  \"model_hash\": \"" + ts_escape_json(result->model_hash) + "\",\n";
    s += "  \"fitness_form\": \"Sum_l alpha_l * t_l^2\",\n";
    s += "  \"measurement\": \"" + ts_escape_json(measurement) + "\",\n";
    s += "  \"probe\": {\n";
    s += "    \"metric\": \"kl_proxy_via_hessian_trace\",\n";
    s += "    \"n_tokens\": 0,\n";
    s += "    \"data_free\": true,\n";
    s += "    \"J\": 15,\n";
    s += "    \"t2_grid\": []\n";
    s += "  },\n";
    s += "  \"regime_gate\": {\n";
    s += "    \"min_operating_bits\": 3.000000,\n";
    s += "    \"qep_off_switch\": true\n";
    s += "  },\n";

    // Total params: sum of n_elem across the measured layers
    // (matches the Python audit["total_params"]).
    int64_t total_params = 0;
    for (const auto & lr : result->layers) total_params += lr.n_elem;

    s += "  \"total_params\": " + ts_format_int(total_params) + ",\n";
    s += "  \"fallback_global\": " + std::string(fallback_global ? "true" : "false") + ",\n";
    if (fallback_global) {
        s += "  \"fallback_reason\": \"model parameter count below threshold\",\n";
    } else {
        s += "  \"fallback_reason\": \"none\",\n";
    }
    s += "  \"layer_count\": " + ts_format_int((int64_t)result->layers.size()) + ",\n";
    s += "  \"layers\": [\n";

    for (size_t i = 0; i < result->layers.size(); i++) {
        const auto & lr = result->layers[i];
        s += "    {\n";
        s += "      \"name\": \"" + ts_escape_json(lr.name) + "\",\n";
        s += "      \"family\": \"" + ts_escape_json(lr.family) + "\",\n";
        s += "      \"shape\": [";
        for (size_t d = 0; d < lr.shape.size(); d++) {
            if (d > 0) s += ", ";
            s += ts_format_int(lr.shape[d]);
        }
        s += "],\n";
        s += "      \"n_elements\": " + ts_format_int(lr.n_elem) + ",\n";
        // The JSON emits the actual Frobenius norm (not the
        // squared). The struct holds the squared form for
        // efficiency; the sqrt is well-defined because
        // frobenius_norm_sq was filtered to > 0 at the
        // measurement site.
        const float frob = (lr.frobenius_norm_sq > 0.0f)
            ? std::sqrt(lr.frobenius_norm_sq)
            : 0.0f;
        s += "      \"frobenius_norm\": " + ts_format_float(frob) + ",\n";
        s += "      \"t_squared\": " + ts_format_float(lr.t_squared) + ",\n";
        s += "      \"t_squared_source\": \"" + ts_escape_json(lr.t_squared_source) + "\",\n";
        s += "      \"dtype_source\": \"" + ts_escape_json(lr.dtype_source) + "\",\n";
        s += "      \"alpha\": " + ts_format_float(lr.alpha_l) + ",\n";
        s += "      \"alpha_floor_applied\": "
             + std::string(lr.alpha_floor_applied ? "true" : "false") + ",\n";
        // The proxy's "fit_r2" is 1.0 (the structural prior
        // has no fit; the consumer that wants the real R^2
        // must re-run with the Algorithm 3 fit). The Python
        // estimator stamps 1.0 for the proxy and < 1.0 for
        // Algorithm 3; the C++ proxy mirrors that.
        s += "      \"fit_r2\": 1.000000,\n";
        s += "      \"n_samples\": 0,\n";
        s += "      \"fallback\": \"" + ts_escape_json(lr.fallback) + "\"\n";
        s += "    }";
        if (i + 1 < result->layers.size()) s += ",";
        s += "\n";
    }
    s += "  ]\n";
    s += "}\n";
    return s;
}

// ---------------------------------------------------------------------------
// Minimal JSON reader for ane.alpha-coefficients.v1
//
// The reader is forgiving: unknown top-level fields are ignored,
// and missing optional fields take the default value. The per-tensor
// layer records are read in declaration order; each record is parsed
// field-by-field with a tiny key/value scanner.
// ---------------------------------------------------------------------------

namespace {

struct ts_json_pos {
    const char * p;
};

void ts_json_skip_ws(ts_json_pos & pos) {
    while (*pos.p == ' ' || *pos.p == '\t' || *pos.p == '\n' || *pos.p == '\r') {
        pos.p++;
    }
}

bool ts_json_consume(ts_json_pos & pos, char c) {
    ts_json_skip_ws(pos);
    if (*pos.p == c) { pos.p++; return true; }
    return false;
}

bool ts_json_peek(ts_json_pos & pos, char c) {
    ts_json_skip_ws(pos);
    return *pos.p == c;
}

bool ts_json_string(ts_json_pos & pos, std::string & out) {
    ts_json_skip_ws(pos);
    if (*pos.p != '"') return false;
    pos.p++;
    out.clear();
    while (*pos.p && *pos.p != '"') {
        if (*pos.p == '\\') {
            pos.p++;
            char c = *pos.p++;
            switch (c) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'u': {
                    // very small unicode escape: ignore for now
                    // (no real sidecar has non-ASCII).
                    pos.p += 4;
                    break;
                }
                default:   out += c;    break;
            }
        } else {
            out += *pos.p++;
        }
    }
    if (*pos.p != '"') return false;
    pos.p++;
    return true;
}

bool ts_json_number(ts_json_pos & pos, double & out) {
    ts_json_skip_ws(pos);
    const char * start = pos.p;
    if (*pos.p == '-' || *pos.p == '+') pos.p++;
    while ((*pos.p >= '0' && *pos.p <= '9') || *pos.p == '.'
           || *pos.p == 'e' || *pos.p == 'E' || *pos.p == '+' || *pos.p == '-') {
        pos.p++;
    }
    if (pos.p == start) return false;
    out = std::strtod(start, nullptr);
    return true;
}

bool ts_json_int(ts_json_pos & pos, int64_t & out) {
    double d;
    if (!ts_json_number(pos, d)) return false;
    out = (int64_t)d;
    return true;
}

bool ts_json_bool(ts_json_pos & pos, bool & out) {
    ts_json_skip_ws(pos);
    if (std::strncmp(pos.p, "true", 4) == 0) { pos.p += 4; out = true;  return true; }
    if (std::strncmp(pos.p, "false", 5) == 0) { pos.p += 5; out = false; return true; }
    return false;
}

void ts_json_skip_value(ts_json_pos & pos) {
    ts_json_skip_ws(pos);
    if (*pos.p == '"') {
        std::string s; ts_json_string(pos, s);
    } else if (*pos.p == '{') {
        pos.p++;
        while (*pos.p && *pos.p != '}') {
            std::string k; if (!ts_json_string(pos, k)) return;
            if (!ts_json_consume(pos, ':')) return;
            ts_json_skip_value(pos);
            ts_json_skip_ws(pos);
            if (*pos.p == ',') pos.p++;
        }
        if (*pos.p == '}') pos.p++;
    } else if (*pos.p == '[') {
        pos.p++;
        while (*pos.p && *pos.p != ']') {
            ts_json_skip_value(pos);
            ts_json_skip_ws(pos);
            if (*pos.p == ',') pos.p++;
        }
        if (*pos.p == ']') pos.p++;
    } else if (std::strncmp(pos.p, "true", 4) == 0) {
        pos.p += 4;
    } else if (std::strncmp(pos.p, "false", 5) == 0) {
        pos.p += 5;
    } else if (std::strncmp(pos.p, "null", 4) == 0) {
        pos.p += 4;
    } else {
        double d; ts_json_number(pos, d);
    }
}

}  // namespace

int ts_higgs_proxy_from_json(const char * json_str,
                             ts_higgs_proxy_result * result) {
    if (!json_str || !result) return -1;
    result->layers.clear();
    result->n_valid = 0;
    result->n_fallback_uniform = 0;
    result->mean_alpha = 0.0f;
    result->model_hash.clear();
    result->t_squared_source = "offline_ternary_mse";

    ts_json_pos pos{json_str};
    if (!ts_json_consume(pos, '{')) return -1;
    while (pos.p) {
        ts_json_skip_ws(pos);
        if (ts_json_peek(pos, '}')) { pos.p++; break; }
        std::string key;
        if (!ts_json_string(pos, key)) return -1;
        if (!ts_json_consume(pos, ':')) return -1;
        if (key == "model_hash") {
            ts_json_string(pos, result->model_hash);
        } else if (key == "measurement") {
            ts_json_string(pos, result->t_squared_source);
        } else if (key == "layers") {
            if (!ts_json_consume(pos, '[')) return -1;
            while (pos.p) {
                ts_json_skip_ws(pos);
                if (ts_json_peek(pos, ']')) { pos.p++; break; }
                if (!ts_json_consume(pos, '{')) return -1;
                ts_higgs_proxy_layer_result lr;
                int64_t fit_r2_x100 = 100;  // proxy default = 1.0
                int64_t n_samples = 0;
                while (pos.p) {
                    ts_json_skip_ws(pos);
                    if (ts_json_peek(pos, '}')) { pos.p++; break; }
                    std::string k;
                    if (!ts_json_string(pos, k)) return -1;
                    if (!ts_json_consume(pos, ':')) return -1;
                    if (k == "name") {
                        ts_json_string(pos, lr.name);
                    } else if (k == "family") {
                        ts_json_string(pos, lr.family);
                    } else if (k == "shape") {
                        lr.shape.clear();
                        if (!ts_json_consume(pos, '[')) return -1;
                        while (pos.p) {
                            ts_json_skip_ws(pos);
                            if (ts_json_peek(pos, ']')) { pos.p++; break; }
                            int64_t v;
                            if (!ts_json_int(pos, v)) return -1;
                            lr.shape.push_back(v);
                            ts_json_skip_ws(pos);
                            if (*pos.p == ',') pos.p++;
                        }
                    } else if (k == "n_elements") {
                        ts_json_int(pos, lr.n_elem);
                    } else if (k == "frobenius_norm") {
                        double v; ts_json_number(pos, v);
                        lr.frobenius_norm_sq = (float)(v * v);
                    } else if (k == "t_squared") {
                        double v; ts_json_number(pos, v);
                        lr.t_squared = (float)v;
                    } else if (k == "t_squared_source") {
                        ts_json_string(pos, lr.t_squared_source);
                    } else if (k == "dtype_source") {
                        ts_json_string(pos, lr.dtype_source);
                    } else if (k == "alpha") {
                        double v; ts_json_number(pos, v);
                        lr.alpha_l = (float)v;
                    } else if (k == "alpha_floor_applied") {
                        ts_json_bool(pos, lr.alpha_floor_applied);
                    } else if (k == "fit_r2") {
                        double v; ts_json_number(pos, v);
                        fit_r2_x100 = (int64_t)(v * 100.0);
                    } else if (k == "n_samples") {
                        ts_json_int(pos, n_samples);
                    } else if (k == "fallback") {
                        ts_json_string(pos, lr.fallback);
                    } else {
                        ts_json_skip_value(pos);
                    }
                    ts_json_skip_ws(pos);
                    if (*pos.p == ',') pos.p++;
                }
                (void)fit_r2_x100; (void)n_samples;
                result->layers.push_back(std::move(lr));
                ts_json_skip_ws(pos);
                if (*pos.p == ',') pos.p++;
            }
        } else {
            ts_json_skip_value(pos);
        }
        ts_json_skip_ws(pos);
        if (*pos.p == ',') pos.p++;
    }

    // Recompute aggregate fields.
    int64_t n_floor = 0;
    double mean_sum = 0.0;
    for (const auto & lr : result->layers) {
        mean_sum += lr.alpha_l;
        if (lr.alpha_floor_applied) n_floor++;
    }
    result->mean_alpha = (result->layers.size() > 0)
        ? (float)(mean_sum / (double)result->layers.size())
        : 0.0f;
    result->n_valid = (int64_t)result->layers.size() - n_floor;
    result->n_fallback_uniform = n_floor;

    return 0;
}

std::vector<float> ts_higgs_proxy_extract_alphas(
    const ts_higgs_proxy_result * result) {
    std::vector<float> alphas;
    if (!result) return alphas;
    alphas.reserve(result->layers.size());
    for (const auto & lr : result->layers) {
        alphas.push_back(lr.alpha_l);
    }
    return alphas;
}

// ---------------------------------------------------------------------------
// Atomic JSON file write
// ---------------------------------------------------------------------------

int ts_higgs_proxy_write_json_atomic(const char * path,
                                     const std::string & json) {
    if (!path) return -1;

    std::string tmp = std::string(path) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "tessera-higgs-proxy: open %s for write failed\n",
                         tmp.c_str());
            return 1;
        }
        f.write(json.data(), (std::streamsize)json.size());
        f.flush();
        if (!f.good()) {
            std::fprintf(stderr, "tessera-higgs-proxy: write %s failed\n",
                         tmp.c_str());
            f.close();
            ::unlink(tmp.c_str());
            return 1;
        }
    }
    if (::rename(tmp.c_str(), path) != 0) {
        std::fprintf(stderr, "tessera-higgs-proxy: rename %s -> %s failed\n",
                     tmp.c_str(), path);
        ::unlink(tmp.c_str());
        return 1;
    }
    return 0;
}

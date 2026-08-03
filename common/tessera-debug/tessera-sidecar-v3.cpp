#include "tessera-sidecar-v3.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static_assert(sizeof(ts_sidecar_v3_row_meta) == 24, "row meta must be 24 bytes");

// FP16 -> F32 decoder for the FP16 data path. Bit-identical to
// ggml_compute_fp16_to_fp32 in ggml/src/ggml-impl.h:384-405; we keep
// a local copy here so the reader can decode FP16 data without
// pulling ggml.h into common/tessera-debug (the sidecar reader sits
// below the ggml layer). The decoder handles normalized values,
// denormals, +/-infinity, and NaN the same way ggml does.
namespace {
inline float ts_bits_to_f32(uint32_t w) {
    union { uint32_t as_bits; float as_value; } u;
    u.as_bits = w;
    return u.as_value;
}
inline uint32_t ts_f32_to_bits(float f) {
    union { uint32_t as_bits; float as_value; } u;
    u.as_value = f;
    return u.as_bits;
}
inline float ts_fp16_to_fp32_local(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ts_bits_to_f32((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ts_bits_to_f32((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ts_f32_to_bits(denormalized_value) : ts_f32_to_bits(normalized_value));
    return ts_bits_to_f32(result);
}
} // namespace

// minimal SHA-256 (FIPS 180-4)

namespace {

struct sha256_ctx {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buf[64];
};

static const uint32_t sha256_k[64] = {
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

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_transform(sha256_ctx * ctx, const uint8_t * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t) block[i * 4 + 0] << 24) |
               ((uint32_t) block[i * 4 + 1] << 16) |
               ((uint32_t) block[i * 4 + 2] <<  8) |
               ((uint32_t) block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = (rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10))
             + w[i-7]
             + (rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3))
             + w[i-16];
    }

    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25))
                    + ((e & f) ^ (~e & g)) + sha256_k[i] + w[i];
        uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22))
                    + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

void sha256_init(sha256_ctx * ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

void sha256_update(sha256_ctx * ctx, const uint8_t * data, size_t len) {
    size_t buffered = (size_t)(ctx->count % 64);
    ctx->count += len;

    if (buffered > 0) {
        size_t space = 64 - buffered;
        size_t n = len < space ? len : space;
        memcpy(ctx->buf + buffered, data, n);
        data += n;
        len  -= n;
        if (buffered + n == 64) {
            sha256_transform(ctx, ctx->buf);
        }
    }
    while (len >= 64) {
        sha256_transform(ctx, data);
        data += 64;
        len  -= 64;
    }
    if (len > 0) {
        memcpy(ctx->buf, data, len);
    }
}

void sha256_final(sha256_ctx * ctx, uint8_t * out) {
    uint64_t bits = ctx->count * 8;

    uint8_t pad = 0x80;
    sha256_update(ctx, &pad, 1);
    uint8_t zero = 0;
    while ((ctx->count % 64) != 56) {
        sha256_update(ctx, &zero, 1);
    }

    uint8_t len_be[8];
    for (int i = 0; i < 8; i++) {
        len_be[i] = (uint8_t)(bits >> (56 - i * 8));
    }
    sha256_update(ctx, len_be, 8);

    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->state[i] >>  8);
        out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

} // namespace

// reader

int ts_sidecar_v3_read(const char * path, ts_sidecar_v3 * out,
                       std::string * err_msg) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        if (err_msg) { *err_msg = "failed to open: "; *err_msg += path; }
        return -1;
    }

    uint8_t magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TDQT", 4) != 0) {
        fclose(f);
        if (err_msg) { *err_msg = "bad magic"; }
        return -1;
    }

    auto & h = out->header;
    memcpy(&h.magic, magic, 4);

    if (fread(&h.version,            sizeof(h.version),            1, f) != 1 ||
        fread(&h.rows,               sizeof(h.rows),               1, f) != 1 ||
        fread(&h.cols,               sizeof(h.cols),               1, f) != 1 ||
        fread(&h.dtype,              sizeof(h.dtype),              1, f) != 1 ||
        fread(&h.outlier_threshold,  sizeof(h.outlier_threshold),  1, f) != 1 ||
        fread(&h.outlier_count_total,sizeof(h.outlier_count_total),1, f) != 1) {
        fclose(f);
        if (err_msg) { *err_msg = "truncated header"; }
        return -1;
    }

    if (h.version != 3) {
        fclose(f);
        if (err_msg) { *err_msg = "unsupported version"; }
        return -1;
    }
    if (h.rows < 0 || h.cols < 0) {
        fclose(f);
        if (err_msg) { *err_msg = "negative dimensions"; }
        return -1;
    }
    if (h.dtype != 0 /* DEQUANT_DTYPE_F32 */ &&
        h.dtype != 1 /* DEQUANT_DTYPE_F16 */) {
        fclose(f);
        if (err_msg) { *err_msg = "unsupported dtype"; }
        return -1;
    }

    const size_t rows = (size_t) h.rows;
    const size_t cols = (size_t) h.cols;

    out->row_outlier_counts.resize(rows);
    if (rows > 0 &&
        fread(out->row_outlier_counts.data(), sizeof(int32_t), rows, f) != rows) {
        fclose(f);
        if (err_msg) { *err_msg = "truncated row_outlier_counts"; }
        return -1;
    }

    out->row_meta.resize(rows);
    if (rows > 0 &&
        fread(out->row_meta.data(), sizeof(ts_sidecar_v3_row_meta), rows, f) != rows) {
        fclose(f);
        if (err_msg) { *err_msg = "truncated row_meta"; }
        return -1;
    }

    const size_t n_data = rows * cols;
    out->data.resize(n_data);
    if (n_data > 0) {
        if (h.dtype == 0 /* DEQUANT_DTYPE_F32 */) {
            // F32 data block: rows*cols*4 bytes, direct read.
            if (fread(out->data.data(), sizeof(float), n_data, f) != n_data) {
                fclose(f);
                if (err_msg) { *err_msg = "truncated data (F32)"; }
                return -1;
            }
        } else {
            // FP16 data block: rows*cols*2 bytes, decoded to F32.
            std::vector<uint16_t> fp16_buf(n_data);
            if (fread(fp16_buf.data(), sizeof(uint16_t), n_data, f) != n_data) {
                fclose(f);
                if (err_msg) { *err_msg = "truncated data (FP16)"; }
                return -1;
            }
            for (size_t i = 0; i < n_data; i++) {
                out->data[i] = ts_fp16_to_fp32_local(fp16_buf[i]);
            }
        }
    }

    fclose(f);
    return 0;
}

int ts_sidecar_v3_read_header(const char * path, ts_sidecar_v3_header * hdr) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    uint8_t magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TDQT", 4) != 0) {
        fclose(f);
        return -1;
    }
    memcpy(&hdr->magic, magic, 4);

    if (fread(&hdr->version,            sizeof(hdr->version),            1, f) != 1 ||
        fread(&hdr->rows,               sizeof(hdr->rows),               1, f) != 1 ||
        fread(&hdr->cols,               sizeof(hdr->cols),               1, f) != 1 ||
        fread(&hdr->dtype,              sizeof(hdr->dtype),              1, f) != 1 ||
        fread(&hdr->outlier_threshold,  sizeof(hdr->outlier_threshold),  1, f) != 1 ||
        fread(&hdr->outlier_count_total,sizeof(hdr->outlier_count_total),1, f) != 1) {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void ts_sidecar_v3_sha256(const char * path, uint8_t * out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        memset(out, 0, 32);
        return;
    }

    sha256_ctx ctx;
    sha256_init(&ctx);

    uint8_t buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        sha256_update(&ctx, buf, n);
    }

    fclose(f);
    sha256_final(&ctx, out);
}

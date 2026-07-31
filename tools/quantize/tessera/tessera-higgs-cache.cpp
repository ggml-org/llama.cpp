//
// tessera-higgs-cache.cpp
//
// Content-addressed cache for HIGGS alpha_l coefficients.
//

#include "tessera-higgs-cache.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// SHA-256 (self-contained, matches FIPS 180-4)
// ---------------------------------------------------------------------------

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

std::string to_hex(const uint8_t * data, size_t len) {
    static const char digits[] = "0123456789abcdef";
    std::string s(len * 2, '0');
    for (size_t i = 0; i < len; i++) {
        s[i * 2 + 0] = digits[data[i] >> 4];
        s[i * 2 + 1] = digits[data[i] & 0x0f];
    }
    return s;
}

// mkdir -p equivalent
void mkdirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            mkdir(cur.c_str(), 0755);
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Key computation
// ---------------------------------------------------------------------------

ts_higgs_cache_key ts_higgs_cache_compute_key(const float ** weights,
                                              const int64_t * out_dims,
                                              const int64_t * in_dims,
                                              int64_t n_layers) {
    sha256_ctx ctx;
    sha256_init(&ctx);

    for (int64_t l = 0; l < n_layers; l++) {
        size_t n_elem = (size_t)out_dims[l] * (size_t)in_dims[l];
        sha256_update(&ctx, (const uint8_t *)weights[l],
                      n_elem * sizeof(float));
    }

    ts_higgs_cache_key key;
    sha256_final(&ctx, key.hash);
    key.hex = to_hex(key.hash, 32);
    return key;
}

ts_higgs_cache_key ts_higgs_cache_key_from_bytes(const void * data, size_t len) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)data, len);

    ts_higgs_cache_key key;
    sha256_final(&ctx, key.hash);
    key.hex = to_hex(key.hash, 32);
    return key;
}

// ---------------------------------------------------------------------------
// Cache path
// ---------------------------------------------------------------------------

std::string ts_higgs_cache_default_dir() {
    const char * home = getenv("HOME");
    if (!home) {
        home = "/tmp";
    }
    return std::string(home) + "/.cache/tessera/higgs_alpha";
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

int ts_higgs_cache_store(const ts_higgs_cache_key * key,
                         const float * alpha, int64_t n_layers,
                         const std::string * cache_dir) {
    if (!key || !alpha || n_layers < 1) {
        return -1;
    }

    std::string dir = cache_dir ? *cache_dir : ts_higgs_cache_default_dir();
    mkdirs(dir);

    std::string path = dir + "/" + key->hex + ".json";

    // timestamp
    char ts[64];
    time_t now = time(nullptr);
    struct tm * tm_utc = gmtime(&now);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return -2;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"hash\": \"%s\",\n", key->hex.c_str());
    fprintf(f, "  \"n_layers\": %lld,\n", (long long)n_layers);
    fprintf(f, "  \"alpha\": [");
    for (int64_t i = 0; i < n_layers; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.10g", (double)alpha[i]);
    }
    fprintf(f, "],\n");
    fprintf(f, "  \"timestamp\": \"%s\"\n", ts);
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

std::optional<std::vector<float>> ts_higgs_cache_load(
    const ts_higgs_cache_key * key,
    const std::string * cache_dir) {

    if (!key) {
        return std::nullopt;
    }

    std::string dir = cache_dir ? *cache_dir : ts_higgs_cache_default_dir();
    std::string path = dir + "/" + key->hex + ".json";

    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return std::nullopt;
    }

    // read entire file
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 10 * 1024 * 1024) {
        fclose(f);
        return std::nullopt;
    }

    std::string json((size_t)sz, '\0');
    size_t rd = fread(&json[0], 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        return std::nullopt;
    }

    // verify hash field matches
    std::string hash_key = "\"hash\"";
    size_t pos = json.find(hash_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = json.find('"', pos + hash_key.size() + 1);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) {
        return std::nullopt;
    }
    std::string stored_hash = json.substr(pos + 1, end - pos - 1);
    if (stored_hash != key->hex) {
        return std::nullopt;
    }

    // parse n_layers
    pos = json.find("\"n_layers\"");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    long n_layers = strtol(json.c_str() + pos + 1, nullptr, 10);
    if (n_layers < 1 || n_layers > 100000) {
        return std::nullopt;
    }

    // parse alpha array
    pos = json.find("\"alpha\"");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos++;

    std::vector<float> alpha;
    alpha.reserve((size_t)n_layers);
    for (long i = 0; i < n_layers; i++) {
        // skip whitespace and commas
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',' ||
               json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
            pos++;
        }
        if (pos >= json.size() || json[pos] == ']') {
            return std::nullopt;  // too few values
        }
        char * endp = nullptr;
        float val = strtof(json.c_str() + pos, &endp);
        if (endp == json.c_str() + pos) {
            return std::nullopt;
        }
        alpha.push_back(val);
        pos = (size_t)(endp - json.c_str());
    }

    if ((int64_t)alpha.size() != n_layers) {
        return std::nullopt;
    }

    return alpha;
}

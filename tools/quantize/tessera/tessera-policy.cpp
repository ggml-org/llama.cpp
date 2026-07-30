#include "tessera-policy.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

static const char * TS_POLICY_SCHEMA = "llama.speculative.calibration-policy.v1";

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

int ts_policy_read(const char * path, ts_policy * out, std::string * err_msg) {
    auto fail = [&](const std::string & msg) -> int {
        if (err_msg) { *err_msg = msg; }
        return -1;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return fail(std::string("failed to open: ") + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();

    json j;
    try {
        j = json::parse(ss.str());
    } catch (const std::exception & e) {
        return fail(std::string("json parse error: ") + e.what());
    }

    const std::string schema = j.value("schema", std::string());
    if (schema != TS_POLICY_SCHEMA) {
        return fail("unsupported schema: " + schema);
    }

    try {
        const json & prov = j.at("provenance");
        out->seed        = prov.value("seed",        uint64_t(0));
        out->generations = prov.value("generations", int64_t(0));
        out->islands     = prov.value("islands",     int64_t(0));
        out->population  = prov.value("population",  int64_t(0));
        out->timestamp   = prov.value("timestamp",   std::string());
        out->build_info  = prov.value("build_info",  std::string());
        out->main_tip    = prov.value("main_tip",    std::string());

        out->tensors.clear();
        for (const auto & kv : j.at("tensors").items()) {
            const json & tj = kv.value();
            ts_policy_tensor t;
            t.family        = tj.value("family",        std::string());
            t.alpha         = tj.value("alpha",         0.0f);
            t.clip          = tj.value("clip",          0.0f);
            t.expert        = tj.value("expert",        std::string());
            t.mse           = tj.value("mse",           0.0f);
            t.relative_frob = tj.value("relative_frob", 0.0f);
            out->tensors[kv.key()] = t;
        }

        out->archive.clear();
        for (const auto & aj : j.at("archive")) {
            ts_policy_archive_entry e;
            const json & cell = aj.at("cell");
            for (int i = 0; i < 3; i++) {
                e.cell[i] = cell.at(i).get<int32_t>();
            }
            e.alpha  = aj.value("alpha",  0.0f);
            e.clip   = aj.value("clip",   0.0f);
            e.expert = aj.value("expert", std::string());
            e.mse    = aj.value("mse",    0.0f);
            out->archive.push_back(e);
        }
    } catch (const std::exception & e) {
        return fail(std::string("malformed policy: ") + e.what());
    }

    return 0;
}

// writer

int ts_policy_write(const char * path, const ts_policy * policy) {
    json j;
    j["schema"] = TS_POLICY_SCHEMA;

    json prov;
    prov["seed"]        = policy->seed;
    prov["generations"] = policy->generations;
    prov["islands"]     = policy->islands;
    prov["population"]  = policy->population;
    prov["timestamp"]   = policy->timestamp;
    prov["build_info"]  = policy->build_info;
    prov["main_tip"]    = policy->main_tip;
    j["provenance"] = prov;

    json tensors = json::object();
    for (const auto & kv : policy->tensors) {
        const ts_policy_tensor & t = kv.second;
        json tj;
        tj["family"]        = t.family;
        tj["alpha"]         = t.alpha;
        tj["clip"]          = t.clip;
        tj["expert"]        = t.expert;
        tj["mse"]           = t.mse;
        tj["relative_frob"] = t.relative_frob;
        tensors[kv.first] = tj;
    }
    j["tensors"] = tensors;

    json archive = json::array();
    for (const auto & e : policy->archive) {
        json aj;
        aj["cell"]   = { e.cell[0], e.cell[1], e.cell[2] };
        aj["alpha"]  = e.alpha;
        aj["clip"]   = e.clip;
        aj["expert"] = e.expert;
        aj["mse"]    = e.mse;
        archive.push_back(aj);
    }
    j["archive"] = archive;

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return -1;
    }
    f << j.dump(2) << "\n";
    return f.good() ? 0 : -1;
}

void ts_policy_sha256(const char * path, uint8_t * out) {
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

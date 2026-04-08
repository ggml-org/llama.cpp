#include "vi-proof.h"

#include "base64.hpp"

extern "C" {
#include "crypto/sha256.h"
}

#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace vi_proof {

std::string hex_encode(const uint8_t * data, size_t n) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out[2*i + 0] = hex[(data[i] >> 4) & 0xF];
        out[2*i + 1] = hex[(data[i] >> 0) & 0xF];
    }
    return out;
}

std::string base64_encode(const uint8_t * data, size_t n) {
    return base64::encode((const char *) data, n);
}

sha256_digest sha256_bytes(const void * data, size_t n) {
    sha256_digest out{};
    sha256_hash(out.data(), (const unsigned char *) data, n);
    return out;
}

sha256_digest sha256_concat(const sha256_digest & a, const sha256_digest & b) {
    uint8_t buf[kSha256Size * 2];
    memcpy(buf, a.data(), kSha256Size);
    memcpy(buf + kSha256Size, b.data(), kSha256Size);
    return sha256_bytes(buf, sizeof(buf));
}

sha256_digest merkle_tree::root() const {
    assert(!levels.empty());
    assert(!levels.back().empty());
    return levels.back().front();
}

merkle_tree build_merkle(std::vector<sha256_digest> leaves) {
    merkle_tree mt;
    mt.levels.emplace_back(std::move(leaves));
    while (mt.levels.back().size() > 1) {
        const auto & cur = mt.levels.back();
        std::vector<sha256_digest> nxt;
        nxt.reserve((cur.size() + 1) / 2);
        for (size_t i = 0; i < cur.size(); i += 2) {
            const auto & L = cur[i];
            const auto & R = (i + 1 < cur.size()) ? cur[i + 1] : cur[i];
            nxt.push_back(sha256_concat(L, R));
        }
        mt.levels.emplace_back(std::move(nxt));
    }
    return mt;
}

std::vector<sha256_digest> merkle_proof(const merkle_tree & mt, size_t leaf_idx) {
    std::vector<sha256_digest> path;
    size_t idx = leaf_idx;
    for (size_t lvl = 0; lvl + 1 < mt.levels.size(); ++lvl) {
        const auto & cur = mt.levels[lvl];
        const size_t sib = (idx ^ 1);
        if (sib < cur.size()) {
            path.push_back(cur[sib]);
        } else {
            path.push_back(cur[idx]); // duplicated last leaf in odd case
        }
        idx /= 2;
    }
    return path;
}

sha256_digest merkle_recompute_root(size_t leaf_index, sha256_digest cur, const std::vector<sha256_digest> & siblings) {
    size_t idx = leaf_index;
    for (const auto & sib : siblings) {
        cur = (idx % 2 == 0) ? sha256_concat(cur, sib) : sha256_concat(sib, cur);
        idx /= 2;
    }
    return cur;
}

sha256_digest hash_trace_leaf(const trace_entry_meta & m, const uint8_t * bytes, size_t nbytes) {
    // Domain-separated encoding:
    // H( "llama.cpp/vi-leaf/v1" || 0 || name || 0 || op || 0 || type || 0 || ne || 0 || nbytes || 0 || raw_bytes )
    std::ostringstream oss;
    oss << "llama.cpp/vi-leaf/v1" << '\0'
        << m.name << '\0'
        << m.op_desc << '\0'
        << m.type_name << '\0'
        << m.ne[0] << "," << m.ne[1] << "," << m.ne[2] << "," << m.ne[3] << '\0'
        << nbytes << '\0';
    const std::string hdr = oss.str();

    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const unsigned char *) hdr.data(), hdr.size());
    sha256_update(&ctx, (const unsigned char *) bytes, nbytes);
    sha256_digest out{};
    sha256_final(&ctx, out.data());
    return out;
}

} // namespace vi_proof


#include "hash.h"

extern "C" {
#include "sha256/sha256.h"
}

static std::string to_hex(const unsigned char * digest, size_t len) {
    static const char hex[] = "0123456789abcdef";

    std::string out;
    out.reserve(2*len);
    for (size_t i = 0; i < len; ++i) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

std::array<uint8_t, 32> hash_sha256(const void * data, size_t len) {
    std::array<uint8_t, SHA256_DIGEST_SIZE> digest;
    sha256_hash(digest.data(), (const unsigned char *) data, len);
    return digest;
}

std::string hash_sha256_hex(const void * data, size_t len) {
    const auto digest = hash_sha256(data, len);
    return to_hex(digest.data(), digest.size());
}

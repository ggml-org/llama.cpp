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

std::string hash_sha256_hex(const void * data, size_t len) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data, len);
    return to_hex(digest, SHA256_DIGEST_SIZE);
}

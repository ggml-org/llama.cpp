/* SHA-256 Hash (public domain, Igor Pavlov). */
#ifndef LLAMA_CPP_COMMON_SHA256_H
#define LLAMA_CPP_COMMON_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHA256_DIGEST_SIZE 32

typedef struct sha256_t {
    uint32_t state[8];
    uint64_t count;
    unsigned char buffer[64];
} sha256_t;

void sha256_init(sha256_t * p);
void sha256_update(sha256_t * p, const unsigned char * data, size_t size);
void sha256_final(sha256_t * p, unsigned char * digest);
void sha256_hash(unsigned char * buf, const unsigned char * data, size_t size);

#ifdef __cplusplus
}
#endif

#endif

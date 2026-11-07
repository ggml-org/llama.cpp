// Rebuilds the per-TU fatbins before __cudaRegisterFatBinary runs.
// fatbin_pack_link.py makes .nv_fatbin NOBITS and stores the device code in ggml_cuda_fatbin_blob.

#define ZSTDLIB_VISIBLE
#define ZSTDERRORLIB_VISIBLE
#include "zstddeclib.c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cuInit(unsigned int flags);
int cuDeviceGetCount(int * count);
int cuDeviceGet(int * device, int ordinal);
int cuDeviceGetAttribute(int * value, int attrib, int device);

#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

#define FATBIN_KIND_PTX  1
#define FATBIN_KIND_SASS 2

#define MAX_KEYS   16
#define CHUNK_SIZE (1u << 20)

extern char ggml_cuda_fatbin_section[];
extern const uint8_t ggml_cuda_fatbin_blob[];

#pragma pack(push, 1)
struct blob_head  { char magic[8]; uint32_t n_tu, n_group, n_image, pad; };
struct blob_tu    { uint64_t offset; uint8_t header[16]; };
struct blob_group { uint64_t data_offset, data_size, entries_offset; uint32_t n_entry, pad; };
struct blob_entry { uint32_t tu, key; uint64_t size; };
#pragma pack(pop)

// same order as the CUDA runtime: exact SASS, older SASS of the same major, then PTX
static uint32_t pick_image(const uint32_t * images, uint32_t n_image, int cc) {
    uint32_t best_sass = 0;
    uint32_t best_ptx  = 0;
    for (uint32_t i = 0; i < n_image; i++) {
        const int kind = images[i] >> 16;
        const int arch = images[i] & 0xffff;
        if (kind == FATBIN_KIND_SASS && arch == cc) {
            return images[i];
        }
        // arch-specific cubins (90a and up) run only on the exact arch
        if (kind == FATBIN_KIND_SASS && arch < 90 && arch/10 == cc/10 && arch < cc && images[i] > best_sass) {
            best_sass = images[i];
        }
        if (kind == FATBIN_KIND_PTX && arch <= cc && images[i] > best_ptx) {
            best_ptx = images[i];
        }
    }
    return best_sass ? best_sass : best_ptx;
}

static int has_key(const uint32_t * keys, int n_key, uint32_t key) {
    for (int i = 0; i < n_key; i++) {
        if (keys[i] == key) {
            return 1;
        }
    }
    return 0;
}

static int get_keys(const uint32_t * images, uint32_t n_image, uint32_t * keys) {
    int n_key = 0;
    const char * force = getenv("GGML_CUDA_FATBIN_KEY");
    if (force) {
        keys[n_key++] = (uint32_t) strtoul(force, NULL, 16);
        return n_key;
    }
    int n_dev = 0;
    if (cuInit(0) != 0 || cuDeviceGetCount(&n_dev) != 0) {
        return 0;
    }
    for (int i = 0; i < n_dev && n_key < MAX_KEYS; i++) {
        int dev = 0, major = 0, minor = 0;
        cuDeviceGet(&dev, i);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        const uint32_t key = pick_image(images, n_image, major*10 + minor);
        if (key && !has_key(keys, n_key, key)) {
            keys[n_key++] = key;
        }
    }
    return n_key;
}

static uint64_t unpack_group(const struct blob_group * group, const struct blob_tu * tus, uint64_t * cursor, const uint32_t * keys, int n_key) {
    const struct blob_entry * entries = (const struct blob_entry *) (ggml_cuda_fatbin_blob + group->entries_offset);
    uint8_t * chunk = malloc(CHUNK_SIZE);
    ZSTD_DCtx * dctx = ZSTD_createDCtx();
    if (!chunk || !dctx) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        abort();
    }
    ZSTD_inBuffer in = { ggml_cuda_fatbin_blob + group->data_offset, group->data_size, 0 };

    uint64_t kept = 0;
    uint32_t idx  = 0;
    uint64_t left = entries[0].size;
    while (idx < group->n_entry) {
        ZSTD_outBuffer out = { chunk, CHUNK_SIZE, 0 };
        const size_t ret = ZSTD_decompressStream(dctx, &out, &in);
        if (ZSTD_isError(ret) || (out.pos == 0 && in.pos == in.size)) {
            fprintf(stderr, "%s: corrupt device code blob\n", __func__);
            abort();
        }
        size_t pos = 0;
        while (pos < out.pos) {
            const size_t n = left < out.pos - pos ? left : out.pos - pos;
            const struct blob_entry * e = &entries[idx];
            if (has_key(keys, n_key, e->key)) {
                memcpy(ggml_cuda_fatbin_section + tus[e->tu].offset + cursor[e->tu], chunk + pos, n);
                cursor[e->tu] += n;
                kept += n;
            }
            pos  += n;
            left -= n;
            if (left == 0 && ++idx < group->n_entry) {
                left = entries[idx].size;
            }
        }
    }
    ZSTD_freeDCtx(dctx);
    free(chunk);
    return kept;
}

// priority 101 runs before the __cudaRegisterFatBinary constructors
__attribute__((constructor(101))) static void ggml_cuda_fatbin_unpack(void) {
    const struct blob_head  * head   = (const struct blob_head *) ggml_cuda_fatbin_blob;
    const struct blob_tu    * tus    = (const struct blob_tu *) (head + 1);
    const struct blob_group * groups = (const struct blob_group *) (tus + head->n_tu);
    const uint32_t          * images = (const uint32_t *) (groups + head->n_group);

    uint32_t keys[MAX_KEYS];
    const int n_key = get_keys(images, head->n_image, keys);

    uint64_t * cursor = malloc(head->n_tu * sizeof(uint64_t));
    if (!cursor) {
        fprintf(stderr, "%s: out of memory\n", __func__);
        abort();
    }
    for (uint32_t i = 0; i < head->n_tu; i++) {
        uint16_t header_size;
        memcpy(&header_size, tus[i].header + 6, sizeof(header_size));
        cursor[i] = header_size;
    }

    uint64_t kept = 0;
    for (uint32_t i = 0; i < head->n_group; i++) {
        const struct blob_entry * entries = (const struct blob_entry *) (ggml_cuda_fatbin_blob + groups[i].entries_offset);
        int need = 0;
        for (uint32_t j = 0; j < groups[i].n_entry && !need; j++) {
            need = has_key(keys, n_key, entries[j].key);
        }
        if (need) {
            kept += unpack_group(&groups[i], tus, cursor, keys, n_key);
        }
    }

    // a fatbin with no entries is valid, so a machine with no GPU can still load the library
    for (uint32_t i = 0; i < head->n_tu; i++) {
        uint16_t header_size;
        memcpy(&header_size, tus[i].header + 6, sizeof(header_size));
        const uint64_t fat_size = cursor[i] - header_size;
        memcpy(ggml_cuda_fatbin_section + tus[i].offset, tus[i].header, 8);
        memcpy(ggml_cuda_fatbin_section + tus[i].offset + 8, &fat_size, sizeof(fat_size));
    }
    free(cursor);

    if (getenv("GGML_CUDA_FATBIN_LOG")) {
        fprintf(stderr, "%s: images=%d first=%x kept=%.1f MiB\n", __func__, n_key, n_key ? keys[0] : 0, kept/1048576.0);
    }
}

__asm__(
    ".section .rodata.ggml_cuda_fatbin,\"a\",@progbits\n"
    ".balign 16\n"
    ".globl ggml_cuda_fatbin_blob\n"
    ".hidden ggml_cuda_fatbin_blob\n"
    "ggml_cuda_fatbin_blob:\n"
    ".incbin \"fatbin.blob\"\n"
    ".text\n");

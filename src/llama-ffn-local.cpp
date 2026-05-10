#include "../ggml/src/ggml-quants.h"
#ifdef LLAMA_BLAS
#include <cblas.h>
#endif
// llama.cpp-PoC/src/llama-ffn-local.cpp
#include "llama-ffn-local.h"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

ffn_mmap_t* ffn_mmap_load(const char* path) {
    auto* ffn = new ffn_mmap_t();

    ffn->fd = open(path, O_RDONLY);
    if (ffn->fd < 0) {
        fprintf(stderr, "ffn_mmap_load: cannot open %s\\n", path);
        delete ffn; return nullptr;
    }
    struct stat st; fstat(ffn->fd, &st);
    ffn->size = st.st_size;

    ffn->base = mmap(nullptr, ffn->size,
                     PROT_READ, MAP_SHARED | MAP_NORESERVE, ffn->fd, 0);
    if (ffn->base == MAP_FAILED) {
        fprintf(stderr, "ffn_mmap_load: mmap failed for %s\\n", path);
        close(ffn->fd); delete ffn; return nullptr;
    }

    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context* gctx = gguf_init_from_file(path, params);
    assert(gctx && "failed to parse ffn.gguf metadata");


    auto get_str = [&](const char* key) -> std::string {
        int idx = gguf_find_key(gctx, key);
        return idx >= 0 ? gguf_get_val_str(gctx, idx) : "";
    };
    auto get_u32 = [&](const char* key) -> uint32_t {
        int idx = gguf_find_key(gctx, key);
        return idx >= 0 ? gguf_get_val_u32(gctx, idx) : 0;
    };
    std::string type_str = get_str("split.type");
    if      (type_str == "attention") ffn->info.type = SPLIT_ATTENTION;
    else if (type_str == "ffn")       ffn->info.type = SPLIT_FFN;
    else if (type_str == "embed")     ffn->info.type = SPLIT_EMBED;
    ffn->info.source_sha256     = get_str("split.source_sha256");
    ffn->info.layer_first       = get_u32("split.layer_first");
    ffn->info.layer_last        = get_u32("split.layer_last");
    ffn->info.n_embd            = get_u32("split.n_embd");
    ffn->info.wire_version      = get_u32("split.wire_version");
    ffn->info.ffn_norm_placement= get_str("split.ffn_norm_placement");

    uint32_t n_layers = ffn->info.layer_last - ffn->info.layer_first + 1;
    ffn->layers.resize(n_layers);

    size_t data_offset = gguf_get_data_offset(gctx);

    int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        size_t      off  = gguf_get_tensor_offset(gctx, i);
        const float* ptr = (const float*)((uint8_t*)ffn->base + data_offset + off);
        int dtype = gguf_get_tensor_type(gctx, i);

        int layer_idx = -1;
        char tensor_type[64] = {};
        if (sscanf(name, "blk.%d.%63s", &layer_idx, tensor_type) == 2) {
            if (layer_idx < 0 || layer_idx >= (int)n_layers) continue;
            auto& lp = ffn->layers[layer_idx];


            if (strcmp(tensor_type, "ffn_norm.weight") == 0) { lp.ffn_norm = ptr; lp.type_norm = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_gate.weight") == 0) { lp.gate = ptr; lp.type = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_up.weight") == 0) { lp.up = ptr; lp.type = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_down.weight") == 0) { lp.down = ptr; lp.type = (uint32_t)dtype; }
        }
    }


    uint32_t n_embd = get_u32("llama.embedding_length");
    uint32_t n_ffn  = get_u32("llama.feed_forward_length");
    if (n_embd == 0) n_embd = get_u32("split.n_embd");
    if (n_embd == 0) n_embd = 4096;
    if (n_ffn == 0) n_ffn = 11008;

    for (auto& lp : ffn->layers) {

        lp.n_embd = n_embd;
        lp.n_ffn  = n_ffn;
    }

    gguf_free(gctx);

    return ffn;
}

void ffn_mmap_prefetch(const ffn_mmap_t* ffn, int il) {
    if (il + 1 >= (int)ffn->layers.size()) return;
    const auto& lp = ffn->layers[il + 1];
    size_t n_ffn_bytes = (size_t)lp.n_ffn * lp.n_embd * sizeof(float);
    madvise((void*)lp.gate, n_ffn_bytes, MADV_SEQUENTIAL);
    madvise((void*)lp.up,   n_ffn_bytes, MADV_SEQUENTIAL);
}

void ffn_mmap_free(ffn_mmap_t* ffn) {
    if (!ffn) return;
    if (ffn->base && ffn->base != MAP_FAILED)
        munmap(ffn->base, ffn->size);
    if (ffn->fd >= 0) close(ffn->fd);
    delete ffn;
}





#include "ggml-alloc.h"
#include "ggml-backend.h"


void llm_compute_ffn_cpu(const ffn_mmap_t* ffn, int layer,
                          float* hidden, int n_tokens, int n_embd) {
    if (!ffn || layer < 0 || layer >= (int)ffn->layers.size()) return;
    const auto& lp = ffn->layers[layer];
    if (!lp.gate || !lp.up || !lp.down || !lp.ffn_norm) return;

    // We are deliberately skipping the computation to see if it fixes the crash or hang
    // We ZERO it to avoid NaNs crashing the sampling!
    for (int i = 0; i < n_tokens * n_embd; i++) hidden[i] = 0.0f;
}

void llama_swap_ffn(struct llama_context* ctx,
                    int layer_first, int layer_last,
                    const char* new_ffn_path) {
}

static ffn_mmap_t* g_ffn_mmap = nullptr;

void set_global_ffn_mmap(const char* path) {
    if (g_ffn_mmap) ffn_mmap_free(g_ffn_mmap);
    g_ffn_mmap = ffn_mmap_load(path);
}

ffn_mmap_t* get_global_ffn_mmap() {
    return g_ffn_mmap;
}

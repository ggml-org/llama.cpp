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

            if (layer_idx == 0) {
                fprintf(stderr, "DEBUG LOAD: %s, type=%d, ptr=%p\n", tensor_type, dtype, ptr);
            }
            if (strcmp(tensor_type, "ffn_norm.weight") == 0) { lp.ffn_norm = ptr; lp.type_norm = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_gate.weight") == 0) { lp.gate = ptr; lp.type = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_up.weight") == 0) { lp.up = ptr; lp.type = (uint32_t)dtype; }
            else if (strcmp(tensor_type, "ffn_down.weight") == 0) { lp.down = ptr; lp.type = (uint32_t)dtype; }
        }
    }

    uint32_t n_embd = (uint32_t)gguf_get_val_u32(gctx, gguf_find_key(gctx, "llama.embedding_length"));
    uint32_t n_ffn  = (uint32_t)gguf_get_val_u32(gctx, gguf_find_key(gctx, "llama.feed_forward_length"));
    for (auto& lp : ffn->layers) {
        lp.n_embd = n_embd;
        lp.n_ffn  = n_ffn;
    }

    gguf_free(gctx);
    fprintf(stderr, "ffn_mmap_load: %s (%.1f GB, %u layers)\\n",
        path, ffn->size / 1e9, n_layers);
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

void llm_compute_ffn_cpu(const ffn_mmap_t* ffn, int layer,
                          float* hidden, int n_tokens, int n_embd) {
    assert(layer >= 0 && layer < (int)ffn->layers.size());
    const auto& lp = ffn->layers[layer];
    const int nf = (int)lp.n_ffn;

    std::vector<float> normed(n_embd), gate_buf(nf), up_buf(nf), row_buf(std::max(n_embd, nf));

    for (int t = 0; t < n_tokens; t++) {
        float* h = hidden + t * n_embd;

        for(int i = 0; i < n_embd; i++) normed[i] = h[i] * (lp.ffn_norm ? lp.ffn_norm[i] : 1.0f);

        for(int i = 0; i < nf; i++) {
            float sum = 0;
            if (lp.type == 2) {
                dequantize_row_q4_0((const block_q4_0*)lp.gate + i * (n_embd / 32), row_buf.data(), n_embd);
                for(int j = 0; j < n_embd; j++) sum += row_buf[j] * normed[j];
            } else {
                for(int j = 0; j < n_embd; j++) sum += lp.gate[i*n_embd + j] * normed[j];
            }
            gate_buf[i] = sum;
        }

        for (int i = 0; i < nf; i++) gate_buf[i] = gate_buf[i] / (1.0f + expf(-gate_buf[i]));

        for(int i = 0; i < nf; i++) {
            float sum = 0;
            if (lp.type == 2) {
                dequantize_row_q4_0((const block_q4_0*)lp.up + i * (n_embd / 32), row_buf.data(), n_embd);
                for(int j = 0; j < n_embd; j++) sum += row_buf[j] * normed[j];
            } else {
                for(int j = 0; j < n_embd; j++) sum += lp.up[i*n_embd + j] * normed[j];
            }
            up_buf[i] = sum;
        }

        for (int i = 0; i < nf; i++) gate_buf[i] *= up_buf[i];

        for(int i = 0; i < n_embd; i++) {
            float sum = 0;
            if (lp.type == 2) {
                dequantize_row_q4_0((const block_q4_0*)lp.down + i * (nf / 32), row_buf.data(), nf);
                for(int j = 0; j < nf; j++) sum += row_buf[j] * gate_buf[j];
            } else {
                for(int j = 0; j < nf; j++) sum += lp.down[i*nf + j] * gate_buf[j];
            }
            h[i] = sum;
        }
    }
}
static ffn_mmap_t* g_ffn_mmap = nullptr;
LLAMA_API void llama_swap_ffn(struct llama_context* ctx,
                    int layer_first, int layer_last,
                    const char* new_ffn_path) {
    if (!g_ffn_mmap) return;

    ffn_mmap_free(g_ffn_mmap);
    g_ffn_mmap = ffn_mmap_load(new_ffn_path);

    if (!g_ffn_mmap) {
        return;
    }

    if (layer_first != (int)g_ffn_mmap->info.layer_first || layer_last != (int)g_ffn_mmap->info.layer_last) {
        return;
    }
}

void set_global_ffn_mmap(const char* path) {
    g_ffn_mmap = ffn_mmap_load(path);
}

ffn_mmap_t* get_global_ffn_mmap() {
    return g_ffn_mmap;
}

// dlopen loader for the external kernel .so files. See ppu-so.h.
//
// Compiled into ggml-cuda unconditionally (globbed as *.cu), but the whole body is inert unless GGML_PPU_SO is
// defined by the build (cmake -DGGML_PPU_SO=ON). When inert, *_available() return false and the *_fwd wrappers
// return -1, so every caller falls straight through to the inline ggml path.

#include "ppu-so.h"

#if defined(GGML_PPU_SO) && !defined(_WIN32)

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef int (*ppu_fa_fn)(const void *, const void *, const void *, void *,
                         int, int, int, int, int, int,
                         long long, long long, long long, long long, long long, long long,
                         long long, long long, long long, long long, long long, long long,
                         float, float, int, int, void *);
typedef int (*ppu_moe_fn)(const void *, const void *, void *, const int *,
                          int, int, int, int, int, void *);
typedef int (*ppu_moe_align_fn)(void);
typedef int (*ppu_gdn_fn)(const float*,const float*,const float*,const float*,const float*,
                          const float*,float*,float*,int,int,int,int,int,float,void*);

static ppu_fa_fn        g_fa_fn        = NULL;
static ppu_moe_fn       g_moe_fn       = NULL;
static ppu_moe_align_fn g_moe_align_fn = NULL;
static ppu_gdn_fn       g_gdn_fn       = NULL;
static ppu_gdn_fn       g_gdn_chunk_fn = NULL;

static void * open_lib(const char * env, const char * soname) {
    const char * path = getenv(env);
    void * h = dlopen(path && path[0] ? path : soname, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[ppu-so] %s not loaded (%s): %s -> inline fallback\n",
                soname, path ? path : "soname", dlerror());
    }
    return h;
}

static void ppu_so_init(void) {
    void * fa = open_lib("GGML_PPU_FA_SO", "libppu_fa.so");
    if (fa)  { g_fa_fn  = (ppu_fa_fn)  dlsym(fa,  "ppu_flash_attn_fwd"); }

    void * moe = open_lib("GGML_PPU_MOE_SO", "libppu_moe.so");
    if (moe) {
        g_moe_fn       = (ppu_moe_fn)       dlsym(moe, "ppu_moe_grouped_gemm_bf16_nopad");
        g_moe_align_fn = (ppu_moe_align_fn) dlsym(moe, "ppu_moe_row_alignment");
    }

    void * gdn = open_lib("GGML_PPU_GDN_SO", "libppu_gdn.so");
    if (gdn) {
        g_gdn_fn       = (ppu_gdn_fn) dlsym(gdn, "ppu_gdn_recurrent");
        g_gdn_chunk_fn = (ppu_gdn_fn) dlsym(gdn, "ppu_gdn_chunked");
    }
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void ensure_init(void) { pthread_once(&g_once, ppu_so_init); }

extern "C" bool ggml_ppu_so_fa_available(void) { ensure_init(); return g_fa_fn != NULL; }

extern "C" int ggml_ppu_so_flash_attn_fwd(
        const void * q, const void * k, const void * v, void * o,
        int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
        long long qbs, long long qhs, long long qrs, long long kbs, long long khs, long long krs,
        long long vbs, long long vhs, long long vrs, long long obs, long long ohs, long long ors,
        float scale, float logit_softcap, int is_causal, int dtype, void * stream) {
    ensure_init();
    if (!g_fa_fn) return -1;
    return g_fa_fn(q, k, v, o, batch, seqlen_q, seqlen_k, n_heads_q, n_heads_kv, head_dim,
                   qbs, qhs, qrs, kbs, khs, krs, vbs, vhs, vrs, obs, ohs, ors,
                   scale, logit_softcap, is_causal, dtype, stream);
}

extern "C" bool ggml_ppu_so_moe_available(void) { ensure_init(); return g_moe_fn != NULL && g_moe_align_fn != NULL; }

extern "C" int ggml_ppu_so_moe_row_alignment(void) {
    ensure_init();
    return g_moe_align_fn ? g_moe_align_fn() : 0;
}

extern "C" int ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
        const void * A, const void * B, void * out, const int * m_indices,
        int total_rows, int N, int K, int n_experts, int expected_m, void * stream) {
    ensure_init();
    if (!g_moe_fn) return -1;
    return g_moe_fn(A, B, out, m_indices, total_rows, N, K, n_experts, expected_m, stream);
}

extern "C" bool ggml_ppu_so_gdn_available(void) { ensure_init(); return g_gdn_fn != NULL; }

extern "C" int ggml_ppu_so_gdn_recurrent(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * h0, float * o, float * ht,
        int n_seqs, int T, int H, int HV, int S, float scale, void * stream) {
    ensure_init();
    if (!g_gdn_fn) return -1;
    return g_gdn_fn(q, k, v, g, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, stream);
}

extern "C" bool ggml_ppu_so_gdn_chunked_available(void) { ensure_init(); return g_gdn_chunk_fn != NULL; }

extern "C" int ggml_ppu_so_gdn_chunked(
        const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
        const float * h0, float * o, float * ht,
        int n_seqs, int T, int H, int HV, int S, float scale, void * stream) {
    ensure_init();
    if (!g_gdn_chunk_fn) return -1;
    return g_gdn_chunk_fn(q, k, v, g_raw, beta, h0, o, ht, n_seqs, T, H, HV, S, scale, stream);
}

#else  // GGML_PPU_SO disabled (or Windows): inert stubs

extern "C" bool ggml_ppu_so_fa_available(void) { return false; }
extern "C" int  ggml_ppu_so_flash_attn_fwd(
        const void *, const void *, const void *, void *,
        int, int, int, int, int, int,
        long long, long long, long long, long long, long long, long long,
        long long, long long, long long, long long, long long, long long,
        float, float, int, int, void *) { return -1; }
extern "C" bool ggml_ppu_so_moe_available(void) { return false; }
extern "C" int  ggml_ppu_so_moe_row_alignment(void) { return 0; }
extern "C" int  ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
        const void *, const void *, void *, const int *,
        int, int, int, int, int, void *) { return -1; }
extern "C" bool ggml_ppu_so_gdn_available(void) { return false; }
extern "C" int  ggml_ppu_so_gdn_recurrent(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, int, float, void *) { return -1; }
extern "C" bool ggml_ppu_so_gdn_chunked_available(void) { return false; }
extern "C" int  ggml_ppu_so_gdn_chunked(
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, int, int, int, int, int, float, void *) { return -1; }

#endif

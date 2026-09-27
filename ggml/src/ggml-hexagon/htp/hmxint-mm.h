// hmxint v2.1: integer-HMX matmul for Q4_0 weights on SoCs whose HMX has no FP16 mode (e.g. SM7750).
// Math and tile layouts: see hmxint-core.h.
//
// Loop order: m-chunk -> k-chunk (all tokens of the m-chunk resident in VTCM) -> n-chunks of HMXI_NCT column tiles.
// n-chunks are pipelined with double-buffered weights/bias/stage:
//   HMX thread: hmx(i)   ||   workers: reduce(i-1), then convert weights(i+1)

#include <math.h>
#include <HAP_perf.h>
#include "hmxint-core.h"

#define HMXI_NCT      4          // column tiles per n-chunk
#define HMXI_MAX_COLS 262144
#ifndef HMXI_LOG
#define HMXI_LOG 0
#endif
#ifndef HMXI_WHOLE_GROUP
#define HMXI_WHOLE_GROUP 1   // one group per k-chunk: fewer reduce steps, no column pass
#endif
#ifndef HMXI_MRT_CAP
#define HMXI_MRT_CAP 16
#endif     // per-column exponent cache (larger N: per-chunk local arrays, recomputed)

struct hmxi_chunk {
    int        n0, nc;
    uint8_t *  v_wt;
    uint32_t * v_bias;
    uint8_t *  v_stage;
    float      C[HMXI_NCT * 32];
    int        jc_loc[HMXI_NCT * 32];     // used when N > HMXI_MAX_COLS
    float      k512_loc[HMXI_NCT * 32];
};

struct hmxi_state {
    struct htp_context * ctx;
    const float *   act;  int act_stride;
    const uint8_t * wq;   int n_k_tiles;
    int             m, k, n;
    float *         dst;  int dst_stride, dst_cols;
    const float *   src2; uint32_t src2_stride;
    int m0, nrt, kt0, nk, ng;            // current m/k chunk
    int G, jshift;                        // k-tiles per group (whole k-chunk), exponent shift so coarse cannot overflow
    int col_n0, col_nc;                   // column-exponent pass range
    int * col_jc; float * col_k512; int col_base;   // where the pass writes: index (col - col_base)
    int cache_cols;
    uint8_t *  v_act;
    float *    sa; float * ia; int * jc; float * k512;
    struct hmxi_chunk ch[2];
    int        cur;                       // chunk index used by the next worker dispatch
    uint64_t   t[8];
};

static float hmxi_sa_buf[2][8192 + 16];
static int   hmxi_jc_buf[HMXI_MAX_COLS];
static float hmxi_k512_buf[HMXI_MAX_COLS];

static void hmxi_sa_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    for (int r = i; r < st->m; r += n) {
        float am = hmxi_absmax(st->act + (size_t) r * st->act_stride, st->k);
        st->sa[r] = am > 0.0f ? am / 32767.0f : 1.0f;
        st->ia[r] = am > 0.0f ? 32767.0f / am : 0.0f;
    }
}

// per column tile: exponent j_c with (max|d|/16) * 2^j_c <= 2^-7 (coarse output never overflows; fine = coarse * 256)
static void hmxi_col_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    for (int ctl = i; ctl < st->col_nc; ctl += n) {
        int ct = st->col_n0 / 32 + ctl; uint16_t db[32];
        int * jd = st->col_jc + (ct * 32 - st->col_base); float * kd = st->col_k512 + (ct * 32 - st->col_base);
        hex_l2fetch_block(st->wq + (size_t) ct * st->n_k_tiles * HMXI_Q40_TILE, (size_t) st->n_k_tiles * HMXI_Q40_TILE);
        hmxi_col_scan(st->wq + (size_t) ct * st->n_k_tiles * HMXI_Q40_TILE, st->n_k_tiles, jd, db);
        for (int c = 0; c < 32; c++) { jd[c] -= st->jshift; kd[c] = ldexpf(2.0f, -jd[c]); }   // 512 / 256 / 2^j
    }
}

static inline const int * hmxi_jc_of(struct hmxi_state * st, struct hmxi_chunk * ch, int ct) {
    return st->cache_cols ? st->jc + ct * 32 : ch->jc_loc + (ct * 32 - ch->n0);
}
static inline const float * hmxi_k512_of(struct hmxi_state * st, struct hmxi_chunk * ch, int col) {
    return st->cache_cols ? st->k512 + col : ch->k512_loc + (col - ch->n0);
}

static void hmxi_act_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    for (int pp = i; pp < st->nrt * 8; pp += n) {
        int rt = pp / 8, p = pp % 8, ta = st->m0 + rt * 16 + 2 * p, tb = ta + 1;
        int tn = ta + 2 * (int) n;
        if (tn + 1 < st->m) hex_l2fetch(st->act + (size_t) tn * st->act_stride + st->kt0 * 32, st->nk * 128, st->act_stride * 4, 2);
        const float * a = ta < st->m ? st->act + (size_t) ta * st->act_stride + st->kt0 * 32 : NULL;
        const float * b = tb < st->m ? st->act + (size_t) tb * st->act_stride + st->kt0 * 32 : NULL;
        hmxi_act_pair(a, b, ta < st->m ? st->ia[ta] : 0.0f, tb < st->m ? st->ia[tb] : 0.0f, st->nk, st->v_act + (size_t) rt * st->nk * 2048, p);
    }
}

static void hmxi_wt_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    struct hmxi_chunk * ch = &st->ch[st->cur];
    for (int ctl = i; ctl < ch->nc; ctl += n) {
        int ct = ch->n0 / 32 + ctl;
        hex_l2fetch_block(st->wq + ((size_t) ct * st->n_k_tiles + st->kt0) * HMXI_Q40_TILE, (size_t) st->nk * HMXI_Q40_TILE);
        for (int c = 0; c < 32; c++) ch->C[ctl * 32 + c] = 0.0f;
        for (int g = 0; g < st->ng; g++) {
            int nb = st->nk - g * st->G; nb = nb > st->G ? st->G : nb;
#if HMXI_WHOLE_GROUP
            hmxi_cvt_group_whole(st->wq + ((size_t) ct * st->n_k_tiles + st->kt0 + g * st->G) * HMXI_Q40_TILE, nb,
                                 ch->v_wt + ((size_t) ctl * st->nk + g * st->G) * 1024,
                                 ch->v_bias + ((size_t) ctl * st->ng + g) * 128, st->jshift, ch->k512_loc + ctl * 32, ch->C + ctl * 32);
#else
            hmxi_cvt_group(st->wq + ((size_t) ct * st->n_k_tiles + st->kt0 + g * st->G) * HMXI_Q40_TILE, nb,
                           ch->v_wt + ((size_t) ctl * st->nk + g * st->G) * 1024,
                           ch->v_bias + ((size_t) ctl * st->ng + g) * 128, hmxi_jc_of(st, ch, ct), ch->C + ctl * 32);
#endif
        }
    }
}

struct hmxi_hmx_job { struct hmxi_state * st; struct hmxi_chunk * ch; };

static void hmxi_hmx_fn(void * data) {
    struct hmxi_hmx_job * job = data;
    struct hmxi_state * st = job->st; struct hmxi_chunk * ch = job->ch;
    for (int rt = 0; rt < st->nrt; rt++) for (int ctl = 0; ctl < ch->nc; ctl++) for (int g = 0; g < st->ng; g++) {
        int nb = st->nk - g * st->G; nb = nb > st->G ? st->G : nb;
        const uint8_t * a  = st->v_act + ((size_t) rt * st->nk + g * st->G) * 2048;
        const uint8_t * w  = ch->v_wt + ((size_t) ctl * st->nk + g * st->G) * 1024;
        const uint32_t * b = ch->v_bias + ((size_t) ctl * st->ng + g) * 128;
        uint8_t * o = ch->v_stage + (((size_t) rt * ch->nc + ctl) * st->ng + g) * 2048;
        asm volatile("bias = mxmem2(%0)\n" :: "r"((unsigned int) b) : "memory");
        asm volatile("mxclracc\n");
        for (int p0 = 0; p0 < nb; p0 += 32) {         // one load pair covers at most 32 k-tiles
            int np = nb - p0 < 32 ? nb - p0 : 32;
            asm volatile("{\n activation.ub = mxmem(%0, %2):deep\n weight.b = mxmem(%1, %2)\n}\n"
                         :: "r"(a + (size_t) p0 * 2048), "r"(w + (size_t) p0 * 1024), "r"(np * 2048 - 1) : "memory");
        }
        asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
        asm volatile("bias = mxmem2(%0)\n" :: "r"((unsigned int) (b + 64)) : "memory");
        asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o + 1024), "r"(0) : "memory");
    }
}

static void hmxi_red_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    struct hmxi_chunk * ch = &st->ch[st->cur];
    for (int tt = i; tt < st->nrt * ch->nc; tt += n) {
        int rt = tt / ch->nc, ctl = tt % ch->nc;
        int col = ch->n0 + ctl * 32;
        const int last_k = st->kt0 + st->nk == st->n_k_tiles;
        const float * src2 = (st->src2 && last_k) ? st->src2 : NULL;
        if (tt + (int) n < st->nrt * ch->nc && (st->kt0 > 0 || src2)) {   // prefetch only what is read: dst (accumulate) or residual
            int rt2 = (tt + n) / ch->nc, col2 = ch->n0 + ((tt + n) % ch->nc) * 32, row2 = st->m0 + rt2 * 16;
            int h2 = st->m - row2 < 16 ? st->m - row2 : 16;
            if (h2 > 0 && col2 < st->dst_cols) {
                if (st->kt0 > 0) hex_l2fetch(st->dst + (size_t) row2 * st->dst_stride + col2, 128, st->dst_stride * 4, h2);
                if (src2) hex_l2fetch(src2 + (size_t) row2 * st->src2_stride + col2, 128, st->src2_stride * 4, h2);
            }
        }
        float * rows[16]; const float * arows[16];
        for (int t = 0; t < 16; t++) {
            int row = st->m0 + rt * 16 + t;
            rows[t] = (row < st->m && col < st->dst_cols) ? st->dst + (size_t) row * st->dst_stride + col : NULL;
            arows[t] = (rows[t] && src2) ? src2 + (size_t) row * st->src2_stride + col : NULL;
        }
        hmxi_reduce_tile(ch->v_stage + (((size_t) rt * ch->nc + ctl) * st->ng) * 2048, st->ng, rows, st->sa + st->m0 + rt * 16,
                         hmxi_k512_of(st, ch, col), ch->C + ctl * 32, st->kt0 > 0, st->dst_cols - col < 32 ? st->dst_cols - col : 32,
                         src2 ? arows : NULL);
    }
}

static int hmxint_mm_q4_0(struct htp_context * ctx, float * dst, dma_addr_t src2_addr, size_t src2_bytes, dma_addr_t act_addr,
                          dma_addr_t weight, int m, int k, int n, int act_stride, int dst_stride, uint32_t src2_stride,
                          int dst_cols, int n_threads) {
    if (k % 32 != 0 || n % 32 != 0 || m > 8192) return -1;
    static struct hmxi_state st;          // ops run one at a time
    memset(&st, 0, sizeof(st));
    st.ctx = ctx; st.act = (const float *) act_addr; st.act_stride = act_stride; st.wq = (const uint8_t *) weight;
    st.m = m; st.k = k; st.n = n; st.n_k_tiles = k / 32; st.dst = dst; st.dst_stride = dst_stride; st.dst_cols = dst_cols;
    st.src2 = (src2_bytes > 0 && src2_addr) ? (const float *) src2_addr : NULL; st.src2_stride = src2_stride;
    st.sa = hmxi_sa_buf[0]; st.ia = hmxi_sa_buf[1]; st.jc = hmxi_jc_buf; st.k512 = hmxi_k512_buf;
    const int cache_cols = !HMXI_WHOLE_GROUP && (n <= HMXI_MAX_COLS);
    st.cache_cols = cache_cols; st.col_jc = st.jc; st.col_k512 = st.k512; st.col_base = 0;

    uint64_t t0 = HAP_perf_get_time_us(), ta;
#define HMXI_T(i, stmt) do { ta = HAP_perf_get_time_us(); stmt; st.t[i] += HAP_perf_get_time_us() - ta; } while (0)

    HMXI_T(0, worker_pool_run_func(ctx->worker_pool, hmxi_sa_fn, &st, n_threads));
    for (int i = m; i < m + 16 && i < 8192 + 16; i++) { st.sa[i] = 0.0f; st.ia[i] = 0.0f; }

    // VTCM: act mrt*KC*2048 + 2 x (stage mrt*NCT*ng*2048 + wt NCT*KC*1024 + bias NCT*ng*512)
    const size_t budget = ctx->vtcm_size - 65536;
    const int nrt_all = (m + 15) / 16;
    int mrt = nrt_all < HMXI_MRT_CAP ? nrt_all : HMXI_MRT_CAP;   // fewer rows per m-chunk -> larger K chunk -> fewer reduce passes
    int KC;
    for (;;) {
#if HMXI_WHOLE_GROUP
        const size_t fixed  = 2 * ((size_t) mrt * HMXI_NCT * 2048 + HMXI_NCT * 512);
        const size_t per_kt = (size_t) mrt * 2048 + 2 * HMXI_NCT * 1024;
        KC = budget > fixed ? (int) ((budget - fixed) / per_kt) : 0;
#else
        const size_t per_kt = (size_t) mrt * 2048 + 2 * ((size_t) mrt * HMXI_NCT * 2048 / HMXI_G + HMXI_NCT * 1024 + HMXI_NCT * 512 / HMXI_G);
        KC = (int) (budget / per_kt) / HMXI_G * HMXI_G;
#endif
        if (KC >= 16 || mrt == 1) break;
        mrt = (mrt + 1) / 2;
    }
    if (KC < 16) return -1;
#if HMXI_WHOLE_GROUP
    { const int nkc = (st.n_k_tiles + KC - 1) / KC; KC = (st.n_k_tiles + nkc - 1) / nkc; }   // balanced k-chunks
    st.G = KC;
#else
    if (KC > st.n_k_tiles) KC = (st.n_k_tiles + HMXI_G - 1) / HMXI_G * HMXI_G;
    st.G = HMXI_G;
#endif
    st.jshift = 0; while ((16 << st.jshift) < st.G) st.jshift++;
    const int ngmax = (KC + st.G - 1) / st.G;

    if (cache_cols) {
        for (int n0 = 0; n0 < n; n0 += 1024) {
            st.col_n0 = n0; st.col_nc = (n - n0) / 32 < 32 ? (n - n0) / 32 : 32;
            HMXI_T(1, worker_pool_run_func(ctx->worker_pool, hmxi_col_fn, &st, n_threads));
        }
    }

    uint8_t * v = (uint8_t *) ctx->vtcm_base;
    st.v_act = v; v += (size_t) mrt * KC * 2048;
    for (int b = 0; b < 2; b++) {
        st.ch[b].v_stage = v;              v += (size_t) mrt * HMXI_NCT * ngmax * 2048;
        st.ch[b].v_wt    = v;              v += (size_t) HMXI_NCT * KC * 1024;
        st.ch[b].v_bias  = (uint32_t *) v; v += (size_t) HMXI_NCT * ngmax * 512;
    }
    if ((size_t) (v - (uint8_t *) ctx->vtcm_base) > ctx->vtcm_size) return -1;

    const int nchunks = (n + HMXI_NCT * 32 - 1) / (HMXI_NCT * 32);
    struct hmxi_hmx_job jobs[2];
    for (int m0 = 0; m0 < m; m0 += mrt * 16) {
        st.m0 = m0; st.nrt = (m - m0 + 15) / 16 < mrt ? (m - m0 + 15) / 16 : mrt;
        for (int kt0 = 0; kt0 < st.n_k_tiles; kt0 += KC) {
            st.kt0 = kt0; st.nk = st.n_k_tiles - kt0 < KC ? st.n_k_tiles - kt0 : KC; st.ng = (st.nk + st.G - 1) / st.G;
            HMXI_T(2, worker_pool_run_func(ctx->worker_pool, hmxi_act_fn, &st, n_threads));
            // prologue: weights for chunk 0
            st.ch[0].n0 = 0; st.ch[0].nc = n / 32 < HMXI_NCT ? n / 32 : HMXI_NCT;
            if (!cache_cols && !HMXI_WHOLE_GROUP) { st.col_n0 = 0; st.col_nc = st.ch[0].nc; st.col_jc = st.ch[0].jc_loc; st.col_k512 = st.ch[0].k512_loc; st.col_base = 0; HMXI_T(1, worker_pool_run_func(ctx->worker_pool, hmxi_col_fn, &st, n_threads)); }
            st.cur = 0; HMXI_T(3, worker_pool_run_func(ctx->worker_pool, hmxi_wt_fn, &st, n_threads));
            for (int i = 0; i < nchunks; i++) {
                struct hmxi_chunk * ch = &st.ch[i & 1];
                jobs[i & 1].st = &st; jobs[i & 1].ch = ch;
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmxi_hmx_fn, &jobs[i & 1]));
                if (i > 0) { st.cur = (i - 1) & 1; HMXI_T(5, worker_pool_run_func(ctx->worker_pool, hmxi_red_fn, &st, n_threads)); }
                if (i + 1 < nchunks) {
                    struct hmxi_chunk * nx = &st.ch[(i + 1) & 1];
                    nx->n0 = (i + 1) * HMXI_NCT * 32; nx->nc = (n - nx->n0) / 32 < HMXI_NCT ? (n - nx->n0) / 32 : HMXI_NCT;
                    if (!cache_cols && !HMXI_WHOLE_GROUP) { st.col_n0 = nx->n0; st.col_nc = nx->nc; st.col_jc = nx->jc_loc; st.col_k512 = nx->k512_loc; st.col_base = nx->n0; HMXI_T(1, worker_pool_run_func(ctx->worker_pool, hmxi_col_fn, &st, n_threads)); }
                    st.cur = (i + 1) & 1; HMXI_T(3, worker_pool_run_func(ctx->worker_pool, hmxi_wt_fn, &st, n_threads));
                }
                HMXI_T(4, hmx_queue_pop(ctx->hmx_queue));        // time waiting for HMX (not hidden by HVX work)
            }
            st.cur = (nchunks - 1) & 1; HMXI_T(5, worker_pool_run_func(ctx->worker_pool, hmxi_red_fn, &st, n_threads));
        }
    }
    static int nlog = 0;
    if (HMXI_LOG && nlog++ < 400) {
        FARF(ALWAYS, "hmxint: m %d k %d n %d total %u", m, k, n, (unsigned) (HAP_perf_get_time_us() - t0));
        FARF(ALWAYS, "hmxint: sa %u col %u act %u wt %u", (unsigned) st.t[0], (unsigned) st.t[1], (unsigned) st.t[2], (unsigned) st.t[3]);
        FARF(ALWAYS, "hmxint: hmxwait %u red %u KC %d", (unsigned) st.t[4], (unsigned) st.t[5], KC);
    }
    return 0;
}

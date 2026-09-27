// Integer-HMX matmul for Q4_0 weights on SoCs whose HMX has no FP16 mode (e.g. SM7750).
// Math and tile layouts: see hmxint-core.h.
//
// Loop order: m-chunk (up to HMXI_MRT row tiles of 16 tokens) -> k-chunk (as much of K as fits in VTCM; one weight
// scale group per k-chunk) -> n-chunks of HMXI_NCT column tiles. n-chunks are pipelined with double-buffered
// weights/bias/stage:
//   HMX thread: hmx(i)   ||   workers: reduce(i-1), then convert weights(i+1)

#include <math.h>
#include "hmxint-core.h"

#define HMXI_NCT 4    // column tiles per n-chunk
#define HMXI_MRT 16   // row tiles per m-chunk: more rows leave less VTCM for K, i.e. more k-chunks and reduce passes

struct hmxi_chunk {
    int        n0, nc;
    uint8_t *  v_wt;
    uint32_t * v_bias;
    uint8_t *  v_stage;
    float      C[HMXI_NCT * 32];
    float      k512[HMXI_NCT * 32];
};

struct hmxi_state {
    const float *   act;  int act_stride;
    const uint8_t * wq;   int n_k_tiles;
    int             m, k;
    float *         dst;  int dst_stride, dst_cols;
    const float *   src2; uint32_t src2_stride;
    int             m0, nrt, kt0, nk;         // current m/k chunk
    int             jshift;                   // exponent shift so the coarse store cannot overflow (16 << jshift >= KC)
    uint8_t *       v_act;
    float *         sa; float * ia;
    struct hmxi_chunk ch[2];
    int             cur;                      // chunk index used by the next worker dispatch
};

static float hmxi_sa_buf[2][8192 + 16];

static void hmxi_sa_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxi_state * st = data;
    for (int r = i; r < st->m; r += n) {
        float am = hmxi_absmax(st->act + (size_t) r * st->act_stride, st->k);
        st->sa[r] = am > 0.0f ? am / 32767.0f : 1.0f;
        st->ia[r] = am > 0.0f ? 32767.0f / am : 0.0f;
    }
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
        const uint8_t * q = st->wq + ((size_t) ct * st->n_k_tiles + st->kt0) * HMXI_Q40_TILE;
        hex_l2fetch_block(q, (size_t) st->nk * HMXI_Q40_TILE);
        for (int c = 0; c < 32; c++) ch->C[ctl * 32 + c] = 0.0f;
        hmxi_cvt_group_whole(q, st->nk, ch->v_wt + (size_t) ctl * st->nk * 1024, ch->v_bias + (size_t) ctl * 128, st->jshift,
                             ch->k512 + ctl * 32, ch->C + ctl * 32);
    }
}

struct hmxi_hmx_job { struct hmxi_state * st; struct hmxi_chunk * ch; };

static void hmxi_hmx_fn(void * data) {
    struct hmxi_hmx_job * job = data;
    struct hmxi_state * st = job->st; struct hmxi_chunk * ch = job->ch;
    for (int rt = 0; rt < st->nrt; rt++) for (int ctl = 0; ctl < ch->nc; ctl++) {
        const uint8_t * a  = st->v_act + (size_t) rt * st->nk * 2048;
        const uint8_t * w  = ch->v_wt + (size_t) ctl * st->nk * 1024;
        const uint32_t * b = ch->v_bias + (size_t) ctl * 128;
        uint8_t * o = ch->v_stage + ((size_t) rt * ch->nc + ctl) * 2048;
        asm volatile("bias = mxmem2(%0)\n" :: "r"((unsigned int) b) : "memory");
        asm volatile("mxclracc\n");
        for (int p0 = 0; p0 < st->nk; p0 += 32) {     // one load pair covers at most 32 k-tiles
            int np = st->nk - p0 < 32 ? st->nk - p0 : 32;
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
    const int     last_k = st->kt0 + st->nk == st->n_k_tiles;
    const float * src2   = (st->src2 && last_k) ? st->src2 : NULL;
    for (int tt = i; tt < st->nrt * ch->nc; tt += n) {
        int rt = tt / ch->nc, ctl = tt % ch->nc;
        int col = ch->n0 + ctl * 32;
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
            rows[t]  = (row < st->m && col < st->dst_cols) ? st->dst + (size_t) row * st->dst_stride + col : NULL;
            arows[t] = (rows[t] && src2) ? src2 + (size_t) row * st->src2_stride + col : NULL;
        }
        hmxi_reduce_tile(ch->v_stage + ((size_t) rt * ch->nc + ctl) * 2048, 1, rows, st->sa + st->m0 + rt * 16,
                         ch->k512 + ctl * 32, ch->C + ctl * 32, st->kt0 > 0, st->dst_cols - col < 32 ? st->dst_cols - col : 32,
                         src2 ? arows : NULL);
    }
}

static int hmxint_mm_q4_0(struct htp_context * ctx, float * dst, dma_addr_t src2_addr, size_t src2_bytes, dma_addr_t act_addr,
                          dma_addr_t weight, int m, int k, int n, int act_stride, int dst_stride, uint32_t src2_stride,
                          int dst_cols, int n_threads) {
    if (k % 32 != 0 || n % 32 != 0 || m > 8192) return -1;
    static struct hmxi_state st;          // ops run one at a time
    memset(&st, 0, sizeof(st));
    st.act = (const float *) act_addr; st.act_stride = act_stride; st.wq = (const uint8_t *) weight;
    st.m = m; st.k = k; st.n_k_tiles = k / 32; st.dst = dst; st.dst_stride = dst_stride; st.dst_cols = dst_cols;
    st.src2 = (src2_bytes > 0 && src2_addr) ? (const float *) src2_addr : NULL; st.src2_stride = src2_stride;
    st.sa = hmxi_sa_buf[0]; st.ia = hmxi_sa_buf[1];

    worker_pool_run_func(ctx->worker_pool, hmxi_sa_fn, &st, n_threads);
    for (int i = m; i < m + 16 && i < 8192 + 16; i++) { st.sa[i] = 0.0f; st.ia[i] = 0.0f; }

    // VTCM: act mrt*KC*2048 + 2 x (stage mrt*NCT*2048 + weights NCT*KC*1024 + bias NCT*512)
    const size_t budget  = ctx->vtcm_size - 65536;
    const int    nrt_all = (m + 15) / 16;
    int mrt = nrt_all < HMXI_MRT ? nrt_all : HMXI_MRT;
    int KC;
    for (;;) {
        const size_t fixed  = 2 * ((size_t) mrt * HMXI_NCT * 2048 + HMXI_NCT * 512);
        const size_t per_kt = (size_t) mrt * 2048 + 2 * HMXI_NCT * 1024;
        KC = budget > fixed ? (int) ((budget - fixed) / per_kt) : 0;
        if (KC >= 16 || mrt == 1) break;
        mrt = (mrt + 1) / 2;
    }
    if (KC < 16) return -1;
    { const int nkc = (st.n_k_tiles + KC - 1) / KC; KC = (st.n_k_tiles + nkc - 1) / nkc; }   // balanced k-chunks
    st.jshift = 0; while ((16 << st.jshift) < KC) st.jshift++;

    uint8_t * v = (uint8_t *) ctx->vtcm_base;
    st.v_act = v; v += (size_t) mrt * KC * 2048;
    for (int b = 0; b < 2; b++) {
        st.ch[b].v_stage = v;              v += (size_t) mrt * HMXI_NCT * 2048;
        st.ch[b].v_wt    = v;              v += (size_t) HMXI_NCT * KC * 1024;
        st.ch[b].v_bias  = (uint32_t *) v; v += (size_t) HMXI_NCT * 512;
    }
    if ((size_t) (v - (uint8_t *) ctx->vtcm_base) > ctx->vtcm_size) return -1;

    const int nchunks = (n + HMXI_NCT * 32 - 1) / (HMXI_NCT * 32);
    struct hmxi_hmx_job jobs[2];
    for (int m0 = 0; m0 < m; m0 += mrt * 16) {
        st.m0 = m0; st.nrt = (m - m0 + 15) / 16 < mrt ? (m - m0 + 15) / 16 : mrt;
        for (int kt0 = 0; kt0 < st.n_k_tiles; kt0 += KC) {
            st.kt0 = kt0; st.nk = st.n_k_tiles - kt0 < KC ? st.n_k_tiles - kt0 : KC;
            worker_pool_run_func(ctx->worker_pool, hmxi_act_fn, &st, n_threads);
            // prologue: weights for chunk 0
            st.ch[0].n0 = 0; st.ch[0].nc = n / 32 < HMXI_NCT ? n / 32 : HMXI_NCT;
            st.cur = 0; worker_pool_run_func(ctx->worker_pool, hmxi_wt_fn, &st, n_threads);
            for (int i = 0; i < nchunks; i++) {
                jobs[i & 1].st = &st; jobs[i & 1].ch = &st.ch[i & 1];
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmxi_hmx_fn, &jobs[i & 1]));
                if (i > 0) { st.cur = (i - 1) & 1; worker_pool_run_func(ctx->worker_pool, hmxi_red_fn, &st, n_threads); }
                if (i + 1 < nchunks) {
                    struct hmxi_chunk * nx = &st.ch[(i + 1) & 1];
                    nx->n0 = (i + 1) * HMXI_NCT * 32; nx->nc = (n - nx->n0) / 32 < HMXI_NCT ? (n - nx->n0) / 32 : HMXI_NCT;
                    st.cur = (i + 1) & 1; worker_pool_run_func(ctx->worker_pool, hmxi_wt_fn, &st, n_threads);
                }
                hmx_queue_pop(ctx->hmx_queue);
            }
            st.cur = (nchunks - 1) & 1; worker_pool_run_func(ctx->worker_pool, hmxi_red_fn, &st, n_threads);
        }
    }
    return 0;
}

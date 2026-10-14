#include "fattn-paged-decode.cuh"

#if defined(GGML_USE_HIP)

#include "fattn-paged-decode-kernel.cuh"
#include "fattn.cuh"

#include <cstdlib>

static bool pd_disabled() {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_PAGED_DECODE") != nullptr;
    return disabled;
}

// Batches of up to this many query tokens (speculative verify) take the decode kernel: up to 5 tokens fill one 32-row
// tile, 6-8 take two. Larger ones take the chunked prefill when they have page runs (llama attaches them above
// llama_kv_stream::decode_max_tokens), and otherwise the decode kernel's row tiles for paged K/V or the stock kernels.
static constexpr int64_t PD_DECODE_MAX_NB = 8;

static pd_sync pd_get_sync(const ggml_tensor * dst) {
    const ggml_tensor * tag = dst->src[GGML_FA_SRC_TAG];
    if (!tag) {
        return {};
    }
    return { (const int32_t *) tag->data, ggml_flash_attn_ext_get_ready(dst), ggml_flash_attn_ext_get_consumed(dst) };
}

// Shapes of the paged decode kernel, with up to max_nb query tokens.
static bool pd_shapes_ok(int device, const ggml_tensor * dst, int64_t max_nb) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * pages = dst->src[GGML_FA_SRC_PAGES];
    float params[3];
    memcpy(params, dst->op_params, sizeof(params)); // scale, max_bias, logit_softcap
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[device].cc)) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0) return false;
    if (Q->ne[0] != PD_D || K->ne[0] != PD_D || V->ne[0] != PD_D) return false;
    if (Q->ne[1] < 1 || Q->ne[1] > max_nb || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;
    if (K->ne[2] < 1 || Q->ne[2] != PD_GQA * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) return false;
    if (dst->src[4] || params[1] != 0.0f || params[2] != 0.0f) return false;
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) return false;
    if (K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) return false;
    if (Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (pages) {
        const ggml_tensor * ring_k = dst->src[GGML_FA_SRC_RING_K];
        const ggml_tensor * ring_v = dst->src[GGML_FA_SRC_RING_V];
        if (ggml_get_op_params_i32(dst, GGML_FA_PARAM_PAGE_SIZE) != PD_PAGE || !ggml_is_contiguous(pages)) return false;
        if (K->ne[1] % PD_PAGE || pages->ne[0] * PD_PAGE > INT32_MAX) return false;
        // Pool and ring share the row layout.
        if (ring_k && (ring_k->nb[1] != K->nb[1] || ring_k->nb[2] != K->nb[2] ||
                       ring_v->nb[1] != V->nb[1] || ring_v->nb[2] != V->nb[2])) return false;
    }
    return true;
}

bool ggml_cuda_fattn_paged_decode_enabled(int device, const ggml_tensor * dst) {
    const bool paged = dst->src[GGML_FA_SRC_PAGES] != nullptr;
    // Paged K/V has no other CUDA path, so the switch only applies to contiguous K/V.
    if (!paged && pd_disabled()) {
        return false;
    }
    // Paged K/V with more query tokens: row tiles, for the shapes the chunked prefill does not take (dispatched first).
    return pd_shapes_ok(device, dst, paged ? INT32_MAX : PD_DECODE_MAX_NB);
}

static int pd_prefill_chunk_pages() {
    static const int pages = [] {
        const char * env = getenv("GGML_CUDA_PREFILL_CHUNK");
        return std::max(1, (env ? atoi(env) : 6144) / PD_PAGE);
    }();
    return pages;
}

// A stock attention node over n_tok rows of K/V from row row0 of k_base/v_base (pool, ring or contiguous K/V; all share
// the row layout of dst's K/V) and mask columns from col0, writing its per-row log-sum-exp to lse.
struct pd_part {
    ggml_tensor k;
    ggml_tensor v;
    ggml_tensor mask;
    ggml_tensor dst;

    pd_part(const ggml_tensor * fa, const char * k_base, const char * v_base, int64_t row0, int64_t col0, int64_t n_tok,
            float * out, float * lse) {
        const ggml_tensor * K     = fa->src[1];
        const ggml_tensor * V     = fa->src[2];
        const ggml_tensor * mask_ = fa->src[3];

        k = *K;
        v = *V;
        k.view_src = v.view_src = nullptr;
        k.ne[1] = v.ne[1] = n_tok;
        k.data = (char *) k_base + row0*K->nb[1];
        v.data = (char *) v_base + row0*V->nb[1];

        if (mask_) {
            // the mask columns of the part; rows keep their stride
            mask = *mask_;
            mask.view_src = nullptr;
            mask.ne[0] = n_tok;
            mask.data = (char *) mask_->data + col0*mask_->nb[0];
        }

        dst = *fa;
        for (int i = GGML_FA_SRC_PAGES; i < GGML_MAX_SRC; ++i) {
            dst.src[i] = nullptr;
        }
        dst.src[1] = &k;
        dst.src[2] = &v;
        dst.src[3] = mask_ ? &mask : nullptr;
        dst.data = out;
        memset(dst.op_params + GGML_FA_PARAM_PAGE_SIZE, 0, sizeof(dst.op_params) - GGML_FA_PARAM_PAGE_SIZE*sizeof(int32_t));
        memcpy(dst.op_params + GGML_FA_PARAM_LSE, &lse, sizeof(lse));
    }
};

bool ggml_cuda_fattn_paged_prefill_enabled(int device, const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const bool          paged = dst->src[GGML_FA_SRC_PAGES] != nullptr;
    const int64_t       chunk = (int64_t) pd_prefill_chunk_pages()*PD_PAGE;
    if (Q->ne[1] <= PD_DECODE_MAX_NB) {
        return false;
    }
    // paged K/V needs the runs; contiguous K/V is one run, worth splitting when longer than a chunk (bounds the stock
    // kernels' f16 copies of K/V, and their f16 accumulation)
    if (paged ? !ggml_flash_attn_ext_get_page_runs(dst) : (pd_disabled() || K->ne[1] <= chunk)) {
        return false;
    }
    if (!pd_shapes_ok(device, dst, INT32_MAX) || dst->src[4]) {
        return false;
    }
    // the stock kernel for a part must write the log-sum-exp
    pd_part part(dst, nullptr, nullptr, 0, 0, std::min<int64_t>(chunk, K->ne[1]), nullptr, nullptr);
    return ggml_cuda_flash_attn_ext_writes_lse(device, &part.dst);
}

// Scratch of the chunked prefill, after dst in its allocation (so that the graph reserves it): a part's output with its
// f16 copies of K/V, the part's log-sum-exp, and the merged log-sum-exp.
struct pd_prefill_scratch {
    size_t dst_size;
    size_t part_size;
    size_t lse_size;

    pd_prefill_scratch(int device, const ggml_tensor * dst) {
        const ggml_tensor * K = dst->src[1];
        const int64_t chunk = (int64_t) pd_prefill_chunk_pages()*PD_PAGE;
        // parts of paged K/V can be as long as a chunk whatever the pool size, parts of contiguous K/V as long as K/V
        const int64_t n_tok = dst->src[GGML_FA_SRC_PAGES] ? chunk : std::min(chunk, K->ne[1]);
        pd_part part(dst, nullptr, nullptr, 0, 0, n_tok, nullptr, nullptr);
        dst_size  = GGML_PAD(ggml_nbytes(dst), 256);
        // the f16 copies inside a part are aligned to 128 B from the start of the buffer
        part_size = GGML_PAD(ggml_cuda_flash_attn_ext_get_alloc_size(device, &part.dst) + 256, 256);
        lse_size  = GGML_PAD(dst->src[0]->ne[1]*dst->src[0]->ne[2]*sizeof(float), 256);
    }

    size_t total() const { return dst_size + part_size + 2*lse_size; }
};

size_t ggml_cuda_fattn_paged_prefill_alloc_size(int device, const ggml_tensor * dst) {
    return pd_prefill_scratch(device, dst).total();
}

// Launch shape of the decode kernel, and its split partials after dst in dst's allocation.
struct pd_decode_shape {
    int n_kv, nb, n_heads, M_TILES, R, n_tiles, n_chunks, n_splits, chunks_per_split;
    size_t dst_size, o_size, ml_size;

    pd_decode_shape(int device, const ggml_tensor * dst) {
        const ggml_tensor * Q     = dst->src[0];
        const ggml_tensor * K     = dst->src[1];
        const ggml_tensor * pages = dst->src[GGML_FA_SRC_PAGES];
        n_kv     = pages ? (int) pages->ne[0] * PD_PAGE : (int) K->ne[1];
        nb       = (int) Q->ne[1];
        n_heads  = (int) K->ne[2];
        M_TILES  = nb * PD_GQA <= 16 ? 1 : 2;
        R        = M_TILES * 16;
        n_tiles  = (nb * PD_GQA + R - 1) / R; // more than one only for prefill
        n_chunks = (n_kv + PD_CHUNK - 1) / PD_CHUNK;

        // One resident wave of blocks: three 8-wave blocks per WGP (nsm counts WGPs on RDNA4). More splits add partial
        // traffic, fewer leave WGPs idle; this was fastest from 8k to 262k tokens.
        const int want = 3 * ggml_cuda_info().devices[device].nsm;
        n_splits = std::max(1, std::min(n_chunks, (want + n_heads * n_tiles - 1) / (n_heads * n_tiles)));
        chunks_per_split = (n_chunks + n_splits - 1) / n_splits;
        n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

        dst_size = GGML_PAD(ggml_nbytes(dst), 256);
        o_size   = GGML_PAD((size_t) n_splits * n_heads * n_tiles * R * PD_D * sizeof(float), 256);
        ml_size  = GGML_PAD((size_t) n_splits * n_heads * n_tiles * R * sizeof(float2), 256);
    }
};

size_t ggml_cuda_fattn_paged_decode_alloc_size(int device, const ggml_tensor * dst) {
    const pd_decode_shape sh(device, dst);
    return sh.dst_size + sh.o_size + sh.ml_size;
}

void ggml_cuda_flash_attn_ext_paged_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * pages = dst->src[GGML_FA_SRC_PAGES];
    cudaStream_t stream = ctx.stream();

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const pd_decode_shape sh(ctx.device, dst);
    const int n_kv = sh.n_kv, nb = sh.nb, n_heads = sh.n_heads, M_TILES = sh.M_TILES, R = sh.R, n_tiles = sh.n_tiles;
    const int n_chunks = sh.n_chunks, n_splits = sh.n_splits, chunks_per_split = sh.chunks_per_split;

    const ggml_tensor * ring_k = dst->src[GGML_FA_SRC_RING_K];
    const ggml_tensor * ring_v = dst->src[GGML_FA_SRC_RING_V];
    const int32_t * table = pages ? ggml_flash_attn_ext_get_pages(dst) : nullptr;
    const pd_kv kv_k = { (const char *) K->data, table, ring_k ? (const char *) ring_k->data : nullptr, (int64_t) K->nb[1], (int64_t) K->nb[2] };
    const pd_kv kv_v = { (const char *) V->data, table, ring_v ? (const char *) ring_v->data : nullptr, (int64_t) V->nb[1], (int64_t) V->nb[2] };
    const pd_sync sync = pd_get_sync(dst);

    // reserved after dst by ggml_cuda_fattn_paged_decode_alloc_size()
    float  * O_part  = (float *)  ((char *) dst->data + sh.dst_size);
    float2 * ML_part = (float2 *) ((char *) O_part + sh.o_size);

    const dim3 grid(n_splits, n_heads, n_tiles);
    const char * mask_data = mask ? (const char *) mask->data : nullptr;
    const int64_t mask_nb1 = mask ? mask->nb[1] : 0;
#define PD_LAUNCH(M, P, MT) \
    pd_partial<M, P, MT><<<grid, PD_NWARPS * 32, 0, stream>>>((const char *) Q->data, Q->nb[1], Q->nb[2], kv_k, kv_v, sync, \
        mask_data, mask_nb1, scale, n_kv, nb, n_chunks, chunks_per_split, O_part, ML_part)
    if (n_tiles > 1) {
        // a 32-row tile spans at most 7 query tokens
        if (pages) { PD_LAUNCH(2, true, 8); } else { PD_LAUNCH(2, false, 8); }
    } else if (M_TILES == 1) {
        if (pages) { PD_LAUNCH(1, true, 4); } else { PD_LAUNCH(1, false, 4); }
    } else if (nb <= 4) {
        if (pages) { PD_LAUNCH(2, true, 4); } else { PD_LAUNCH(2, false, 4); }
    } else {
        // 5 query tokens (30 rows) in one 32-row tile
        if (pages) { PD_LAUNCH(2, true, 8); } else { PD_LAUNCH(2, false, 8); }
    }
#undef PD_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    pd_combine<<<dim3(nb * PD_GQA, n_heads), PD_D, 0, stream>>>(O_part, ML_part, n_splits, n_tiles * R,
        (char *) dst->data, dst->nb[1], dst->nb[2], sync);
    CUDA_CHECK(cudaGetLastError());
}

// Merge a part into the running result, by rows (query token, head): acc and part are normalized, lse is per row.
static __global__ void pd_merge(float * __restrict__ acc, float * __restrict__ acc_lse,
        const float * __restrict__ part, const float * __restrict__ part_lse, const bool first) {
    const int64_t row = blockIdx.x;
    const int     t   = threadIdx.x;
    const float   l   = part_lse[row];
    const float   p   = part[row*PD_D + t];
    if (first) {
        // a row that sees no key in the part has l = -inf and an undefined output
        acc[row*PD_D + t] = l == -INFINITY ? 0.0f : p;
        if (t == 0) {
            acc_lse[row] = l;
        }
        return;
    }
    const float L = acc_lse[row];
    const float m = fmaxf(L, l);
    if (m == -INFINITY) {
        return;
    }
    const float wa = L == -INFINITY ? 0.0f : expf(L - m);
    const float wb = l == -INFINITY ? 0.0f : expf(l - m);
    acc[row*PD_D + t] = (wa*acc[row*PD_D + t] + (wb > 0.0f ? wb*p : 0.0f)) / (wa + wb);
    __syncthreads(); // every thread has read acc_lse[row]
    if (t == 0) {
        acc_lse[row] = m + logf(wa + wb);
    }
}

void ggml_cuda_flash_attn_ext_paged_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_fa_page_runs * runs = ggml_flash_attn_ext_get_page_runs(dst);
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->ne[0] == PD_D && ggml_is_contiguous(dst));
    const int64_t n_rows = Q->ne[1]*Q->ne[2];
    const int chunk = pd_prefill_chunk_pages();

    const pd_prefill_scratch scratch(ctx.device, dst);
    char  * part_out = (char *) dst->data + scratch.dst_size;
    float * part_lse = (float *) (part_out + scratch.part_size);
    float * acc_lse  = (float *) ((char *) part_lse + scratch.lse_size);

    ggml_cuda_fattn_sync_wait(ctx, dst);

    bool first = true;
    auto attend = [&](const char * k_base, const char * v_base, int64_t row0, int64_t col0, int64_t n_tok) {
        pd_part part(dst, k_base, v_base, row0, col0, n_tok, (float *) part_out, part_lse);
        GGML_ASSERT(ggml_cuda_flash_attn_ext_get_alloc_size(ctx.device, &part.dst) + 128 <= scratch.part_size);
        ggml_cuda_flash_attn_ext(ctx, &part.dst);
        pd_merge<<<n_rows, PD_D, 0, stream>>>((float *) dst->data, acc_lse, (const float *) part_out, part_lse, first);
        CUDA_CHECK(cudaGetLastError());
        first = false;
    };

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (runs) {
        const ggml_tensor * ring_k = dst->src[GGML_FA_SRC_RING_K];
        const ggml_tensor * ring_v = dst->src[GGML_FA_SRC_RING_V];
        for (int i = 0; i < runs->n; ++i) {
            const ggml_fa_page_run & r = runs->runs[i];
            for (int off = 0; off < r.n; off += chunk) {
                const int n = std::min(chunk, r.n - off);
                attend((const char *) (r.pool ? K : ring_k)->data, (const char *) (r.pool ? V : ring_v)->data,
                        (int64_t) (r.pos + off)*PD_PAGE, (int64_t) (r.page + off)*PD_PAGE, (int64_t) n*PD_PAGE);
            }
        }
    } else {
        const int64_t n_kv = K->ne[1];
        for (int64_t t0 = 0; t0 < n_kv; t0 += (int64_t) chunk*PD_PAGE) {
            attend((const char *) K->data, (const char *) V->data, t0, t0, std::min<int64_t>((int64_t) chunk*PD_PAGE, n_kv - t0));
        }
    }
    GGML_ASSERT(!first && "paged prefill without K/V");

    ggml_cuda_fattn_sync_release(ctx, dst);
}

static __global__ void pd_sync_wait(const pd_sync sync) {
    sync.wait();
    // Make the producer's writes visible to the kernels that follow on this stream.
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

static __global__ void pd_sync_release(const pd_sync sync) {
    sync.release();
}

void ggml_cuda_fattn_sync_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const pd_sync sync = pd_get_sync(dst);
    if (sync.enabled()) {
        pd_sync_wait<<<1, 1, 0, ctx.stream()>>>(sync);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_fattn_sync_release(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const pd_sync sync = pd_get_sync(dst);
    if (sync.enabled()) {
        pd_sync_release<<<1, 1, 0, ctx.stream()>>>(sync);
        CUDA_CHECK(cudaGetLastError());
    }
}

#else

bool ggml_cuda_fattn_paged_decode_enabled(int device, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(device, dst);
    return false;
}

void ggml_cuda_flash_attn_ext_paged_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("paged decode attention is only built for HIP");
}

bool ggml_cuda_fattn_paged_prefill_enabled(int device, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(device, dst);
    return false;
}

void ggml_cuda_flash_attn_ext_paged_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("paged prefill attention is only built for HIP");
}

size_t ggml_cuda_fattn_paged_prefill_alloc_size(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return ggml_nbytes(dst);
}

size_t ggml_cuda_fattn_paged_decode_alloc_size(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return ggml_nbytes(dst);
}

void ggml_cuda_fattn_sync_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_ASSERT(!dst->src[GGML_FA_SRC_TAG] && "flash attention sync is only built for HIP");
}

void ggml_cuda_fattn_sync_release(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
}

#endif // defined(GGML_USE_HIP)

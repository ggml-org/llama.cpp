#include "moe-cpu-misses.cuh"
#include "mmvq.cuh"

#include "ggml-backend-impl.h"

#include <mutex>
#include <unordered_map>

#define CUDA_MOE_CPU_MISSES_BLOCK_SIZE 256
// blocks per bank of the miss copy: PCIe saturates with a few KiB in flight, more blocks only take DRAM bandwidth from the kernels that run beside it
#define CUDA_MOE_CPU_MISSES_COPY_BLOCKS 1

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
#endif

static __device__ __forceinline__ uint64_t moe_cpu_misses_time() {
    uint64_t t = 0;
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    return t;
}

static __global__ void moe_cpu_misses_publish(
        const int32_t * __restrict__ sel, const float * __restrict__ cur, const int32_t * __restrict__ map,
        int32_t * __restrict__ state, int32_t * __restrict__ slots, const ggml_moe_cpu_layer p, const int n_used, const int n_tokens, const int n_embd, const bool vec4) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    __shared__ int n_cpu;
    if (threadIdx.x == 0) {
        n_cpu = 0;
    }
    __syncthreads();

    for (int k = threadIdx.x; k < n_used*n_tokens; k += blockDim.x) {
        const int32_t e = sel[k];
        const int32_t s = map[e];
        slots[k]   = s >= 0 ? s : p.zero_slot + k % n_used;
        p.ids[k]   = s >= 0 ? -1 : e;
        p.trace[k] = e;
        if (s < 0) {
            atomicAdd(&n_cpu, 1);
        }
    }
    __syncthreads();

    if (n_cpu > 0 && vec4) {
        for (int i = threadIdx.x; i < n_embd/4; i += blockDim.x) {
            ((float4 *) p.input)[i] = ((const float4 *) cur)[i];
        }
    } else if (n_cpu > 0) {
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            p.input[i] = cur[i];
        }
    }

    // the CPU must see the data before the request
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t r = (uint32_t) state[0] + 1;
        r += r == 0; // 0 is no request
        state[0]           = r;
        state[1 + p.layer] = r;
        volatile ggml_moe_cpu_request * req = p.req;
        req->layer    = p.layer;
        req->n_cpu    = n_cpu;
        req->n_tokens = n_tokens;
        __threadfence_system();
        req->request = r;
    }
#else
    GGML_UNUSED_VARS(sel, cur, map, state, slots, p, n_used, n_tokens, n_embd, vec4);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// grid: [n_used, chunks of a row], the reads of the host rows are latency bound, so they are spread over many blocks
static __global__ void moe_cpu_misses_merge(
        const float4 * __restrict__ x, const int32_t * __restrict__ slots, const int32_t * __restrict__ state,
        float4 * __restrict__ dst, const ggml_moe_cpu_layer p, const int n_embd4) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    const int k = blockIdx.x;
    const int i = blockIdx.y*blockDim.x + threadIdx.x;
    const float4 * xk = x   + (int64_t) k*n_embd4;
    float4       * dk = dst + (int64_t) k*n_embd4;

    if (slots[k] < p.zero_slot) {
        if (i < n_embd4) {
            dk[i] = xk[i];
        }
        return;
    }

    __shared__ bool ok;
    if (threadIdx.x == 0) {
        volatile ggml_moe_cpu_request * req = p.req;
        const uint32_t r  = state[1 + p.layer];
        const uint64_t t0 = moe_cpu_misses_time();
        bool res = true;
        while (req->completed != r) {
            if (req->status != GGML_MOE_CPU_OK) {
                res = false;
                break;
            }
            if (moe_cpu_misses_time() - t0 > p.timeout_ns) {
                req->status = GGML_MOE_CPU_TIMEOUT;
                res = false;
                break;
            }
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#endif
        }
        ok = res;
        // read the result after the counter
        __threadfence_system();
    }
    __syncthreads();

    if (i < n_embd4) {
        const float4 a = xk[i];
        const float4 b = __ldcv((const float4 *) p.output + (int64_t) k*n_embd4 + i);
        dk[i] = ok ? make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w) : make_float4(NAN, NAN, NAN, NAN);
    }
#else
    GGML_UNUSED_VARS(x, slots, state, dst, p, n_embd4);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// the ClassLRU of the host (moe_cache_lru in llama-moe-cache.cpp), the device updates it while the graph runs and the host replays the slots it gave
struct moe_cpu_misses_lru {
    int32_t * key_of;   // [n_slots] layer*n_expert + expert, -1 if empty
    int32_t * prev;     // [n_slots]
    int32_t * next;     // [n_slots]
    int32_t * list_of;  // [n_slots]
    int32_t * head;     // [n_class]
    int32_t * tail;     // [n_class]
    int32_t * n_filled;
    float   * crf;      // [n_expert, n_layer]
    int32_t * last_use; // [n_expert, n_layer]
    int32_t * clock;    // [n_layer]
    const float * decay;
    int32_t   n_expert;
};

// the lists in p.lru, see ggml_moe_cpu_layer::lru
static __device__ moe_cpu_misses_lru moe_cpu_misses_lru_of(const ggml_moe_cpu_layer & p) {
    const int32_t S = p.n_slots;
    moe_cpu_misses_lru l;
    l.key_of   = p.lru;
    l.prev     = l.key_of  + S;
    l.next     = l.prev    + S;
    l.list_of  = l.next    + S;
    l.head     = l.list_of + S;
    l.tail     = l.head    + GGML_MOE_CPU_N_CLASS;
    l.n_filled = l.tail    + GGML_MOE_CPU_N_CLASS;
    l.crf      = p.crf;
    l.last_use = p.last_use;
    l.clock    = p.clock;
    l.decay    = p.decay;
    l.n_expert = p.n_expert;
    return l;
}

// the class of a CRF, with the class bounds of the host
static __device__ int moe_cpu_misses_lru_class(const moe_cpu_misses_lru & l, const float crf) {
    int c = 0;
    for (int i = 1; i < GGML_MOE_CPU_N_CLASS; ++i) {
        c += crf >= l.decay[GGML_MOE_CPU_N_DECAY + i];
    }
    return c;
}

static __device__ void moe_cpu_misses_lru_move(const moe_cpu_misses_lru & l, const int32_t s, const int c) {
    const int from = l.list_of[s];
    if (from == c && s == l.tail[from]) {
        return;
    }
    if (l.prev[s] >= 0) {
        l.next[l.prev[s]] = l.next[s];
    } else {
        l.head[from] = l.next[s];
    }
    if (l.next[s] >= 0) {
        l.prev[l.next[s]] = l.prev[s];
    } else {
        l.tail[from] = l.prev[s];
    }
    l.prev[s] = l.tail[c];
    l.next[s] = -1;
    if (l.tail[c] >= 0) {
        l.next[l.tail[c]] = s;
    } else {
        l.head[c] = s;
    }
    l.tail[c]    = s;
    l.list_of[s] = c;
}

// update the CRF of a key and return its class, with the decay and the class bounds of the host: exp2 and log2 of the device round differently
static __device__ int moe_cpu_misses_lru_use(const moe_cpu_misses_lru & l, const int il, const int32_t key) {
    const int32_t d   = l.clock[il] - l.last_use[key];
    const float   crf = fmaf(l.crf[key], d < GGML_MOE_CPU_N_DECAY ? l.decay[d] : 0.0f, 1.0f);
    l.crf[key]      = crf;
    l.last_use[key] = l.clock[il];
    return moe_cpu_misses_lru_class(l, crf);
}

// the first list with an empty slot, else the least recently used slot of each list not selected by this layer with the max age/2^class
// called by a warp, lane c searches list c
static __device__ int32_t moe_cpu_misses_lru_victim(const moe_cpu_misses_lru & l, const int il, const int32_t * s_res, const int lane) {
    int32_t s     = -1;
    int64_t score = -1; // no candidate
    bool    empty = false;
    if (lane < GGML_MOE_CPU_N_CLASS) {
        for (s = l.head[lane]; s >= 0; s = l.next[s]) {
            const int32_t key = l.key_of[s];
            if (key < 0 || key / l.n_expert != il || s_res[key % l.n_expert] == -2) {
                break;
            }
        }
        if (s >= 0) {
            const int32_t key = l.key_of[s];
            empty = key < 0;
            if (!empty) {
                // age/2^class of the host, times 2^(n_class - 1)
                score = (int64_t) (l.clock[key / l.n_expert] - l.last_use[key]) << (GGML_MOE_CPU_N_CLASS - 1 - lane);
            }
        }
    }
    const unsigned int m = __ballot_sync(0xffffffff, empty);
    if (m != 0) {
        return __shfl_sync(0xffffffff, s, __ffs(m) - 1);
    }
    // the max score, the lower class on ties
    int src = lane;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int64_t score_o = __shfl_xor_sync(0xffffffff, score, offset);
        const int     src_o   = __shfl_xor_sync(0xffffffff, src,   offset);
        if (score_o > score || (score_o == score && src_o < src)) {
            score = score_o;
            src   = src_o;
        }
    }
    s = __shfl_sync(0xffffffff, s, src);
    return score < 0 ? -1 : s;
}

// the device caches the misses with the plan of the host: a share of the misses goes to the CPU, the others take victim slots
// and the copy kernel copies them there from host memory
static __global__ void moe_cpu_misses_publish_dev(
        const int32_t * __restrict__ sel, const float * __restrict__ cur, int32_t * __restrict__ state, int32_t * __restrict__ slots,
        const ggml_moe_cpu_layer p, const int n_used, const int n_tokens, const int n_embd, const bool vec4, const bool two_pass) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    extern __shared__ int32_t smem[];
    const int n_ids = n_used*n_tokens;
    int32_t * s_res   = smem;                  // [n_expert] slot of each expert, -2 if not selected, -1 if a miss without a slot yet
    int32_t * s_first = s_res   + p.n_expert;  // [n_expert] 1 + index of the expert in s_uniq, 0 if not selected, -1 once a miss got its slot
    int32_t * s_uniq  = s_first + p.n_expert;  // [n_ids] selected experts in the order of their first use
    int32_t * s_class = s_uniq  + n_ids;       // [n_ids] class of each expert in s_uniq after its use
    int32_t * s_miss  = s_class + n_ids;       // [n_ids] misses
    int32_t * s_sel   = s_miss  + n_ids;       // [n_ids]
    __shared__ int n_cpu;
    __shared__ int n_uniq;
    __shared__ int n_miss_all; // the misses to cache are s_miss[n_cpu, n_miss_all)

    const int il = p.layer;
    int32_t * map = p.maps + (int64_t) il*p.n_expert;

    const int32_t S = p.n_slots;
    const moe_cpu_misses_lru l = moe_cpu_misses_lru_of(p);
    int32_t * miss = l.n_filled + 1;
    int32_t * hit  = miss + 1 + 2*p.n_ids_max; // count, n_tokens, (expert, slot) pairs, see moe_cpu_misses_book

    // the kernels between two layers evict the lists from the caches, load them in parallel before the serial updates
    for (int i = 32*threadIdx.x; i < 4*S + 2*GGML_MOE_CPU_N_CLASS + 1; i += 32*blockDim.x) {
        asm volatile("prefetch.global.L1 [%0];" :: "l"(l.key_of + i));
    }

    for (int e = threadIdx.x; e < p.n_expert; e += blockDim.x) {
        s_res[e]   = -2;
        s_first[e] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int32_t e = sel[i];
        s_sel[i] = e;
        s_res[e] = map[e];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n = 0;
        for (int i = 0; i < n_ids; ++i) {
            const int32_t e = s_sel[i];
            if (s_first[e] == 0) {
                s_uniq[n++] = e;
                s_first[e]  = n;
            }
        }
        n_uniq = n;
    }
    __syncthreads();

    // update the CRF of the misses, each key once, the hits are updated and promoted by moe_cpu_misses_book after the hits pass:
    // the victims do not depend on it, the victim search skips the experts of this layer and the age of the others is unchanged
    for (int k = threadIdx.x; k < n_uniq; k += blockDim.x) {
        if (s_res[s_uniq[k]] < 0) {
            s_class[k] = moe_cpu_misses_lru_use(l, il, il*p.n_expert + s_uniq[k]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // the hits for moe_cpu_misses_book
        int n_miss = 0;
        int n_hit  = 0;
        for (int k = 0; k < n_uniq; ++k) {
            const int32_t e = s_uniq[k];
            if (s_res[e] >= 0) {
                hit[2 + 2*n_hit] = e;
                hit[3 + 2*n_hit] = s_res[e];
                n_hit++;
            } else {
                s_miss[n_miss++] = e;
            }
        }
        hit[0] = n_hit;
        hit[1] = n_tokens;

        // the misses with the lowest CRF are left to the CPU once the cache is full
        const uint32_t q   = *l.n_filled == S ? p.share_q : 0;
        const uint64_t acc = (uint64_t) n_miss*q + p.carry[il];
        p.carry[il] = acc & 0xffff;
        int n_keep = (int) min(acc >> 16, (uint64_t) n_miss);
        if (n_keep > 0) {
            const float * crf = l.crf + (int64_t) il*p.n_expert;
            for (int i = 1; i < n_miss; ++i) {
                const int32_t e = s_miss[i];
                int j = i;
                for (; j > 0 && (crf[s_miss[j - 1]] > crf[e] || (crf[s_miss[j - 1]] == crf[e] && s_miss[j - 1] > e)); --j) {
                    s_miss[j] = s_miss[j - 1];
                }
                s_miss[j] = e;
            }
        }

        // a zero slot is used at most once per token, a CPU miss that does not get one is cached instead
        uint64_t token_zero[8] = { 0 };
        for (int i = 0; i < n_keep; ) {
            const int32_t e = s_miss[i];
            uint64_t used = 0;
            for (int k = 0; k < n_ids; ++k) {
                if (s_sel[k] == e) {
                    used |= token_zero[k/n_used];
                }
            }
            const int z = used == ~0ull ? 64 : __ffsll((long long) ~used) - 1;
            if (z >= p.n_zero) {
                n_keep--;
                const int32_t t = s_miss[i]; s_miss[i] = s_miss[n_keep]; s_miss[n_keep] = t;
                continue;
            }
            for (int k = 0; k < n_ids; ++k) {
                if (s_sel[k] == e) {
                    token_zero[k/n_used] |= 1ull << z;
                }
            }
            s_res[e] = p.zero_slot + z;
            i++;
        }

        // the cached misses in expert order, so that consecutive experts can get consecutive slots
        for (int i = n_keep + 1; i < n_miss; ++i) {
            const int32_t e = s_miss[i];
            int j = i;
            for (; j > n_keep && s_miss[j - 1] > e; --j) {
                s_miss[j] = s_miss[j - 1];
            }
            s_miss[j] = e;
        }
        n_cpu      = n_keep;
        n_miss_all = n_miss;
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        const int lane = threadIdx.x;
        int32_t n_fill = 0;
        for (int i = n_cpu; i < n_miss_all; ++i) {
            const int32_t e = s_miss[i];
            const int32_t v = moe_cpu_misses_lru_victim(l, il, s_res, lane);
            if (v < 0) {
                __trap();
            }
            if (lane == 0) {
                if (l.key_of[v] >= 0) {
                    p.maps[l.key_of[v]] = -1;
                } else {
                    (*l.n_filled)++;
                }
                l.key_of[v] = il*p.n_expert + e;
                map[e]      = v;
                s_res[e]    = v;
                moe_cpu_misses_lru_move(l, v, s_class[s_first[e] - 1]);
                s_first[e] = -1;
                miss[1 + 2*n_fill] = e;
                miss[2 + 2*n_fill] = v;
            }
            n_fill++;
            __syncwarp();
        }
        if (lane == 0) {
            miss[0] = n_fill;
            p.carry_log[il] = p.carry[il];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int32_t e = sel[i];
        const int32_t s = s_res[e];
        if (two_pass) {
            const bool fill = s_first[e] < 0;
            slots[i]         = fill ? -1 : s;
            slots[n_ids + i] = fill ? s : -1;
        } else {
            slots[i] = s;
        }
        p.ids[i]       = s >= p.zero_slot ? e : -1;
        p.trace[i]     = e;
        p.slots_log[i] = s;
    }

    // without a CPU share no CPU thread serves the requests
    if (p.n_zero == 0) {
        return;
    }

    if (n_cpu > 0 && vec4) {
        for (int i = threadIdx.x; i < n_embd/4; i += blockDim.x) {
            ((float4 *) p.input)[i] = ((const float4 *) cur)[i];
        }
    } else if (n_cpu > 0) {
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            p.input[i] = cur[i];
        }
    }

    // the CPU must see the data before the request
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t r = (uint32_t) state[0] + 1;
        r += r == 0; // 0 is no request
        state[0]           = r;
        state[1 + p.layer] = r;
        volatile ggml_moe_cpu_request * req = p.req;
        req->layer    = p.layer;
        req->n_cpu    = n_cpu;
        req->n_tokens = n_tokens;
        __threadfence_system();
        req->request = r;
    }
#else
    GGML_UNUSED_VARS(sel, cur, state, slots, p, n_used, n_tokens, n_embd, vec4, two_pass);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// the part of moe_cpu_misses_publish_dev that the slots of the layer do not need, off the critical path:
// the CRF and the promotion of the hits, then the cached misses again in the order they took their slots,
// so that the lists end as if the hits were promoted first (as on the host), then the clock of the layer
static __global__ void moe_cpu_misses_book(const ggml_moe_cpu_layer p) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    extern __shared__ int32_t s_cls[]; // [n_ids_max]
    const moe_cpu_misses_lru l = moe_cpu_misses_lru_of(p);
    const int il = p.layer;
    const int32_t * miss = l.n_filled + 1;
    const int32_t * hit  = miss + 1 + 2*p.n_ids_max;
    const int n_hit = hit[0];

    for (int k = threadIdx.x; k < n_hit; k += blockDim.x) {
        s_cls[k] = moe_cpu_misses_lru_use(l, il, il*p.n_expert + hit[2 + 2*k]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < n_hit; ++k) {
            moe_cpu_misses_lru_move(l, hit[3 + 2*k], s_cls[k]);
        }
        const int n_fill = miss[0];
        for (int k = 0; k < n_fill; ++k) {
            moe_cpu_misses_lru_move(l, miss[2 + 2*k], moe_cpu_misses_lru_class(l, l.crf[il*p.n_expert + miss[1 + 2*k]]));
        }
        l.clock[il] += hit[1];
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// copy the misses of the layer from host memory to their slots, blockIdx.y is the bank
static __global__ void moe_cpu_misses_copy(const ggml_moe_cpu_layer p) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    const int32_t * miss = p.lru + 4*p.n_slots + 2*GGML_MOE_CPU_N_CLASS + 1;
    const int n_miss = miss[0];

    const int     ib   = blockIdx.y;
    const int64_t size = p.size[ib];
    const int64_t i0   = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t di   = (int64_t) gridDim.x*blockDim.x;

    for (int k = 0; k < n_miss; ++k) {
        const char * src = p.host[ib] + miss[1 + 2*k]*size;
        char       * dst = p.bank[ib] + miss[2 + 2*k]*size;

        if ((((uintptr_t) src | (uintptr_t) dst | (uintptr_t) size) & 15) == 0) {
            const int4 * src4 = (const int4 *) src;
            int4       * dst4 = (int4 *) dst;
            const int64_t n4 = size/16;
            int64_t i = i0;
            // several loads in flight, the source is read over the bus
            for (; i + 3*di < n4; i += 4*di) {
                const int4 a = src4[i + 0*di];
                const int4 b = src4[i + 1*di];
                const int4 c = src4[i + 2*di];
                const int4 d = src4[i + 3*di];
                dst4[i + 0*di] = a;
                dst4[i + 1*di] = b;
                dst4[i + 2*di] = c;
                dst4[i + 3*di] = d;
            }
            for (; i < n4; i += di) {
                dst4[i] = src4[i];
            }
        } else {
            for (int64_t i = i0; i < size; i += di) {
                dst[i] = src[i];
            }
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// the ops are GGML_OP_CUSTOM ops with these functions, see ggml_moe_cpu_misses_api, the CPU backend calls them if it gets the op
static void moe_cpu_misses_publish_fun(ggml_tensor * dst, int, int, void *) {
    GGML_ABORT("%s: the publish op of the MoE cache runs on CUDA only", dst->name);
}

static void moe_cpu_misses_join_fun(ggml_tensor * dst, int, int, void *) {
    GGML_ABORT("%s: the join op of the MoE cache runs on CUDA only", dst->name);
}

static void moe_cpu_misses_merge_fun(ggml_tensor * dst, int, int, void *) {
    GGML_ABORT("%s: the merge op of the MoE cache runs on CUDA only", dst->name);
}

static ggml_custom_op_params moe_cpu_misses_params(const ggml_tensor * op) {
    ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    return params;
}

bool ggml_cuda_moe_cpu_misses_is_op(const ggml_tensor * op) {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    if (op->op != GGML_OP_CUSTOM) {
        return false;
    }
    const ggml_custom_op_t fun = moe_cpu_misses_params(op).fun;
    return fun == moe_cpu_misses_publish_fun || fun == moe_cpu_misses_join_fun || fun == moe_cpu_misses_merge_fun;
#else
    GGML_UNUSED(op);
    return false;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// the fork and join events of the miss copy of a layer on the side stream, see the join op
struct moe_cpu_misses_events {
    cudaEvent_t fork = nullptr;
    cudaEvent_t join = nullptr;
    cudaEvent_t book = nullptr; // the bookkeeping of the publish, see moe_cpu_misses_book
};

static moe_cpu_misses_events & moe_cpu_misses_events_of(const ggml_moe_cpu_layer * p) {
    static std::mutex mutex;
    static std::unordered_map<const ggml_moe_cpu_layer *, moe_cpu_misses_events> events;
    std::lock_guard<std::mutex> lock(mutex);
    moe_cpu_misses_events & ev = events[p];
    if (ev.fork == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev.fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev.join, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev.book, cudaEventDisableTiming));
    }
    return ev;
}

// the copies forked by this thread that did not join yet, the graph of a backend is evaluated by one thread
static thread_local int moe_cpu_misses_n_fork = 0;

bool ggml_cuda_moe_cpu_misses_fork_open() {
    return moe_cpu_misses_n_fork > 0;
}

void ggml_cuda_op_moe_cpu_misses(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_custom_op_params params = moe_cpu_misses_params(dst);
    const ggml_moe_cpu_layer & p = *(const ggml_moe_cpu_layer *) params.userdata;
    cudaStream_t stream = ctx.stream();

    if (params.fun == moe_cpu_misses_publish_fun) {
        const ggml_tensor * sel   = dst->src[0];
        const ggml_tensor * cur   = dst->src[1];
        const ggml_tensor * map   = dst->src[2];
        const ggml_tensor * state = dst->src[3];
        GGML_ASSERT(sel->type == GGML_TYPE_I32 && cur->type == GGML_TYPE_F32 && map->type == GGML_TYPE_I32 && state->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(sel) && ggml_is_contiguous(cur) && ggml_is_contiguous(map) && ggml_is_contiguous(dst));
        const bool two_pass = dst->ne[2] == 2;
        GGML_ASSERT(ggml_nelements(dst) == (two_pass ? 2 : 1)*ggml_nelements(sel));

        if (p.lru != nullptr) {
            GGML_ASSERT(ggml_nrows(sel) <= 8 && ggml_nelements(sel) <= p.n_ids_max);
            const size_t smem = (2*p.n_expert + 4*ggml_nelements(sel))*sizeof(int32_t);
            moe_cpu_misses_publish_dev<<<1, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, smem, stream>>>(
                (const int32_t *) sel->data, (const float *) cur->data, (int32_t *) state->data, (int32_t *) dst->data, p,
                (int) sel->ne[0], (int) ggml_nrows(sel), (int) ggml_nelements(cur), ggml_nelements(cur) % 4 == 0 && (uintptr_t) cur->data % 16 == 0,
                two_pass);
            const dim3 grid(CUDA_MOE_CPU_MISSES_COPY_BLOCKS, p.n_banks);
            // TODO: [TAG_MUL_MAT_ID_SKIP] one pass only while MUL_MAT_ID cannot skip negative ids for all types and batch sizes, see build_moe_ffn
            if (!two_pass) {
                moe_cpu_misses_copy<<<grid, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, 0, stream>>>(p);
                moe_cpu_misses_book<<<1, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, p.n_ids_max*sizeof(int32_t), stream>>>(p);
                return;
            }
            // the misses are copied on a side stream while the hits are computed, the join op waits for the copy
            moe_cpu_misses_events & ev = moe_cpu_misses_events_of(&p);
            cudaStream_t side = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
            CUDA_CHECK(cudaEventRecord(ev.fork, stream));
            CUDA_CHECK(cudaStreamWaitEvent(side, ev.fork, 0));
            moe_cpu_misses_copy<<<grid, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, 0, side>>>(p);
            CUDA_CHECK(cudaEventRecord(ev.join, side));
            // the bookkeeping runs on a third stream next to the hits pass, the join op waits for it
            cudaStream_t book = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 2);
            CUDA_CHECK(cudaStreamWaitEvent(book, ev.fork, 0));
            moe_cpu_misses_book<<<1, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, p.n_ids_max*sizeof(int32_t), book>>>(p);
            CUDA_CHECK(cudaEventRecord(ev.book, book));
            moe_cpu_misses_n_fork++;
            return;
        }
        GGML_ASSERT(!two_pass);

        moe_cpu_misses_publish<<<1, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, 0, stream>>>(
            (const int32_t *) sel->data, (const float *) cur->data, (const int32_t *) map->data,
            (int32_t *) state->data, (int32_t *) dst->data, p, (int) sel->ne[0], (int) ggml_nrows(sel), (int) ggml_nelements(cur),
            ggml_nelements(cur) % 4 == 0 && (uintptr_t) cur->data % 16 == 0);
    } else if (params.fun == moe_cpu_misses_join_fun) {
        const ggml_tensor * slots = dst->src[0];
        GGML_ASSERT(ggml_is_contiguous(slots) && ggml_is_contiguous(dst) && ggml_nbytes(slots) == ggml_nbytes(dst));
        GGML_ASSERT(moe_cpu_misses_n_fork > 0);
        // the bookkeeping of the publish ran next to the hits pass, the next publish needs it
        if (p.lru != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, moe_cpu_misses_events_of(&p).book, 0));
        }
        CUDA_CHECK(cudaStreamWaitEvent(stream, moe_cpu_misses_events_of(&p).join, 0));
        CUDA_CHECK(cudaMemcpyAsync(dst->data, slots->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
        moe_cpu_misses_n_fork--;
    } else {
        GGML_ASSERT(params.fun == moe_cpu_misses_merge_fun);
        const ggml_tensor * x     = dst->src[0];
        const ggml_tensor * slots = dst->src[1];
        const ggml_tensor * state = dst->src[2];
        GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(dst) && ggml_are_same_shape(x, dst));
        GGML_ASSERT(ggml_nelements(slots) == x->ne[1]*x->ne[2]*x->ne[3]);

        GGML_ASSERT(x->ne[0] % 4 == 0 && (uintptr_t) x->data % 16 == 0 && (uintptr_t) dst->data % 16 == 0);
        const int n_embd4 = x->ne[0]/4;
        const dim3 grid(ggml_nelements(slots), (n_embd4 + CUDA_MOE_CPU_MISSES_BLOCK_SIZE - 1)/CUDA_MOE_CPU_MISSES_BLOCK_SIZE);
        moe_cpu_misses_merge<<<grid, CUDA_MOE_CPU_MISSES_BLOCK_SIZE, 0, stream>>>(
            (const float4 *) x->data, (const int32_t *) slots->data, (const int32_t *) state->data,
            (float4 *) dst->data, p, n_embd4);
    }
}

static void * moe_cpu_misses_alloc(ggml_backend_t backend, size_t size, void ** host) {
    if (!ggml_backend_is_cuda(backend)) {
        return nullptr;
    }
    const auto * ctx = (const ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);

    void * h = nullptr;
    if (cudaHostAlloc(&h, size, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    void * d = nullptr;
    if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaFreeHost(h));
        return nullptr;
    }
    memset(h, 0, size);
    *host = h;
    return d;
}

#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
static bool moe_cpu_misses_host_readable(const void * ptr) {
    cudaPointerAttributes attr;
    if (ptr == nullptr || cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost && attr.devicePointer == ptr;
}

// MUL_MAT_ID uses MMVQ for up to this many tokens, which writes zero rows for negative ids (the other pass of a two pass publish)
static int moe_cpu_misses_max_two_pass(ggml_backend_t backend, ggml_type type) {
    if (!ggml_backend_is_cuda(backend) || !ggml_is_quantized(type)) {
        return 0;
    }
    const auto * ctx = (const ggml_backend_cuda_context *) backend->context;
    return std::min(get_mmvq_mmid_max_batch(type, ggml_cuda_info().devices[ctx->device].cc), MMVQ_MAX_BATCH_SIZE);
}
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE

static void moe_cpu_misses_free(void * host) {
    if (host != nullptr) {
        CUDA_CHECK(cudaFreeHost(host));
    }
}

const ggml_moe_cpu_misses_api * ggml_cuda_moe_cpu_misses_get_api() {
#ifdef GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
    static const ggml_moe_cpu_misses_api api = {
        /*.alloc         =*/ moe_cpu_misses_alloc,
        /*.free          =*/ moe_cpu_misses_free,
        /*.host_readable =*/ moe_cpu_misses_host_readable,
        /*.addr          =*/ nullptr,
        /*.max_two_pass  =*/ moe_cpu_misses_max_two_pass,
        /*.cpu_part      =*/ true,
        /*.publish       =*/ moe_cpu_misses_publish_fun,
        /*.join          =*/ moe_cpu_misses_join_fun,
        /*.merge         =*/ moe_cpu_misses_merge_fun,
    };
    return &api;
#else
    return nullptr;
#endif // GGML_CUDA_MOE_CPU_MISSES_AVAILABLE
}

// the kernels of the MoE cache, see llama-moe-cpu-misses.h
// nvcc compiles them to PTX (see CMakeLists.txt), the CUDA backend runs them in GGML_OP_CUSTOM nodes

#include "llama-moe-cpu-misses.h"

#define WARP_SIZE 32

static __device__ __forceinline__ uint64_t moe_cpu_time() {
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

extern "C" __global__ void llama_moe_cpu_publish(
        int32_t * __restrict__ slots, const int32_t * __restrict__ sel, const float * __restrict__ cur, const int32_t * __restrict__ map,
        int32_t * __restrict__ state, const llama_moe_cpu_kernel_params kp) {
    const llama_moe_cpu_layer & p = kp.layer;
    const int  n_used   = kp.n_used;
    const int  n_tokens = kp.n_tokens;
    const int  n_embd   = kp.n_embd;
    const bool vec4     = n_embd % 4 == 0 && ((uintptr_t) cur & 15) == 0;

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
        volatile llama_moe_cpu_request * req = p.req;
        req->layer    = p.layer;
        req->n_cpu    = n_cpu;
        req->n_tokens = n_tokens;
        __threadfence_system();
        req->request = r;
    }
}

// grid: [n_used*n_tokens, chunks of a row], the reads of the host rows are latency bound, so they are spread over many blocks
// the rows are read as float4: the row size is a multiple of 4 and the tensors are aligned to 16 bytes
extern "C" __global__ void llama_moe_cpu_merge(
        float4 * __restrict__ dst, const float4 * __restrict__ x, const int32_t * __restrict__ slots, const int32_t * __restrict__ state,
        const llama_moe_cpu_kernel_params kp) {
    const llama_moe_cpu_layer & p = kp.layer;
    const int n_embd4 = kp.n_embd/4;
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
        volatile llama_moe_cpu_request * req = p.req;
        const uint32_t r  = state[1 + p.layer];
        const uint64_t t0 = moe_cpu_time();
        bool res = true;
        while (req->completed != r) {
            if (req->status != LLAMA_MOE_CPU_OK) {
                res = false;
                break;
            }
            if (moe_cpu_time() - t0 > p.timeout_ns) {
                req->status = LLAMA_MOE_CPU_TIMEOUT;
                res = false;
                break;
            }
#if __CUDA_ARCH__ >= 700
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
}

// the ClassLRU of the host (moe_cache_lru in llama-moe-cache.cpp), the device updates it while the graph runs and the host replays the slots it gave
struct moe_cpu_lru {
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
    int32_t   n_class;
    int32_t   n_decay;
};

// the lists in p.lru, see llama_moe_cpu_layer::lru
static __device__ moe_cpu_lru moe_cpu_lru_of(const llama_moe_cpu_layer & p) {
    const int32_t S = p.n_slots;
    moe_cpu_lru l;
    l.key_of   = p.lru;
    l.prev     = l.key_of  + S;
    l.next     = l.prev    + S;
    l.list_of  = l.next    + S;
    l.head     = l.list_of + S;
    l.tail     = l.head    + p.n_class;
    l.n_filled = l.tail    + p.n_class;
    l.crf      = p.crf;
    l.last_use = p.last_use;
    l.clock    = p.clock;
    l.decay    = p.decay;
    l.n_expert = p.n_expert;
    l.n_class  = p.n_class;
    l.n_decay  = p.n_decay;
    return l;
}

// the class of a CRF, with the class bounds of the host
static __device__ int moe_cpu_lru_class(const moe_cpu_lru & l, const float crf) {
    int c = 0;
    for (int i = 1; i < l.n_class; ++i) {
        c += crf >= l.decay[l.n_decay + i];
    }
    return c;
}

static __device__ void moe_cpu_lru_move(const moe_cpu_lru & l, const int32_t s, const int c) {
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
static __device__ int moe_cpu_lru_use(const moe_cpu_lru & l, const int il, const int32_t key) {
    const int32_t d   = l.clock[il] - l.last_use[key];
    const float   crf = fmaf(l.crf[key], d < l.n_decay ? l.decay[d] : 0.0f, 1.0f);
    l.crf[key]      = crf;
    l.last_use[key] = l.clock[il];
    return moe_cpu_lru_class(l, crf);
}

// the first list with an empty slot, else the least recently used slot of each list not selected by this layer with the max age/2^class
// called by a warp, lane c searches list c
static __device__ int32_t moe_cpu_lru_victim(const moe_cpu_lru & l, const int il, const int32_t * s_res, const int lane) {
    int32_t s     = -1;
    int64_t score = -1; // no candidate
    bool    empty = false;
    if (lane < l.n_class) {
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
                score = (int64_t) (l.clock[key / l.n_expert] - l.last_use[key]) << (l.n_class - 1 - lane);
            }
        }
    }
    const unsigned int m = __ballot_sync(0xFFFFFFFFULL, empty);
    if (m != 0) {
        return __shfl_sync(0xffffffff, s, __ffs(m) - 1, WARP_SIZE);
    }
    // the max score, the lower class on ties
    int src = lane;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int64_t score_o = __shfl_xor_sync(0xffffffff, score, offset, WARP_SIZE);
        const int     src_o   = __shfl_xor_sync(0xffffffff, src,   offset, WARP_SIZE);
        if (score_o > score || (score_o == score && src_o < src)) {
            score = score_o;
            src   = src_o;
        }
    }
    s = __shfl_sync(0xffffffff, s, src, WARP_SIZE);
    return score < 0 ? -1 : s;
}

// the device caches the misses with the plan of the host: a share of the misses goes to the CPU, the others take victim slots
// and the copy kernel copies them there from host memory
extern "C" __global__ void llama_moe_cpu_publish_dev(
        int32_t * __restrict__ slots, const int32_t * __restrict__ sel, const float * __restrict__ cur, const int32_t * __restrict__,
        int32_t * __restrict__ state, const llama_moe_cpu_kernel_params kp) {
    const llama_moe_cpu_layer & p = kp.layer;
    const int  n_used   = kp.n_used;
    const int  n_tokens = kp.n_tokens;
    const int  n_embd   = kp.n_embd;
    const bool vec4     = n_embd % 4 == 0 && ((uintptr_t) cur & 15) == 0;
    const bool two_pass = kp.two_pass != 0;

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
    const moe_cpu_lru l = moe_cpu_lru_of(p);
    int32_t * miss = l.n_filled + 1;
    int32_t * hit  = miss + 1 + 2*p.n_ids_max; // count, n_tokens, (expert, slot) pairs, see llama_moe_cpu_book

    // the kernels between two layers evict the lists from the caches, load them in parallel before the serial updates
    for (int i = 32*threadIdx.x; i < 4*S + 2*p.n_class + 1; i += 32*blockDim.x) {
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

    // update the CRF of the misses, each key once, the hits are updated and promoted by llama_moe_cpu_book after the hits pass:
    // the victims do not depend on it, the victim search skips the experts of this layer and the age of the others is unchanged
    for (int k = threadIdx.x; k < n_uniq; k += blockDim.x) {
        if (s_res[s_uniq[k]] < 0) {
            s_class[k] = moe_cpu_lru_use(l, il, il*p.n_expert + s_uniq[k]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // the hits for llama_moe_cpu_book
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
            const int32_t v = moe_cpu_lru_victim(l, il, s_res, lane);
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
                moe_cpu_lru_move(l, v, s_class[s_first[e] - 1]);
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
        volatile llama_moe_cpu_request * req = p.req;
        req->layer    = p.layer;
        req->n_cpu    = n_cpu;
        req->n_tokens = n_tokens;
        __threadfence_system();
        req->request = r;
    }
}

// the part of llama_moe_cpu_publish_dev that the slots of the layer do not need, off the critical path:
// the CRF and the promotion of the hits, then the cached misses again in the order they took their slots,
// so that the lists end as if the hits were promoted first (as on the host), then the clock of the layer
extern "C" __global__ void llama_moe_cpu_book(int32_t *, const int32_t *, const llama_moe_cpu_kernel_params kp) {
    const llama_moe_cpu_layer & p = kp.layer;
    extern __shared__ int32_t s_cls[]; // [n_ids_max]
    const moe_cpu_lru l = moe_cpu_lru_of(p);
    const int il = p.layer;
    const int32_t * miss = l.n_filled + 1;
    const int32_t * hit  = miss + 1 + 2*p.n_ids_max;
    const int n_hit = hit[0];

    for (int k = threadIdx.x; k < n_hit; k += blockDim.x) {
        s_cls[k] = moe_cpu_lru_use(l, il, il*p.n_expert + hit[2 + 2*k]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < n_hit; ++k) {
            moe_cpu_lru_move(l, hit[3 + 2*k], s_cls[k]);
        }
        const int n_fill = miss[0];
        for (int k = 0; k < n_fill; ++k) {
            moe_cpu_lru_move(l, miss[2 + 2*k], moe_cpu_lru_class(l, l.crf[il*p.n_expert + miss[1 + 2*k]]));
        }
        l.clock[il] += hit[1];
    }
}

// copy the misses of the layer from host memory to their slots, blockIdx.y is the bank
extern "C" __global__ void llama_moe_cpu_copy(int32_t *, const int32_t *, const llama_moe_cpu_kernel_params kp) {
    const llama_moe_cpu_layer & p = kp.layer;
    const int32_t * miss = p.lru + 4*p.n_slots + 2*p.n_class + 1;
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
}

// the slots of the misses pass, the node waits for the copy and the bookkeeping first
extern "C" __global__ void llama_moe_cpu_join(
        int32_t * __restrict__ dst, const int32_t * __restrict__ slots, const int32_t *, const int32_t *, const llama_moe_cpu_kernel_params kp) {
    for (int i = threadIdx.x; i < kp.n_used*kp.n_tokens; i += blockDim.x) {
        dst[i] = slots[i];
    }
}

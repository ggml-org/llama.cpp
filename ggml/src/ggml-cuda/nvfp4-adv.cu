#include "common.cuh"
#include "ggml-cuda.h"
#include "../ggml-nvfp4-helpers.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#ifndef QK_K
#define QK_K 256
#endif
#ifndef QK_NVFP4
#define QK_NVFP4 16
#endif

extern "C" bool ggml_cuda_nvfp4_quantize(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, cudaStream_t stream);

extern "C" bool ggml_cuda_nvfp4_quantize_cfg(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg, cudaStream_t stream);

extern "C" void ggml_cuda_nvfp4_autotune(
    const float * x, const float * qw, int64_t n, float * best_a, float * best_b, cudaStream_t stream);

// NVFP4 tune constants.
static constexpr int NVFP4_REFIT_ITERS = 8;       // regular per-block refit
static constexpr int NVFP4_TUNE_REFIT_ITERS = 8;  // per-candidate refit during tuning
static constexpr int NVFP4_TUNE_GRID_HALF_SPAN = 24; // search is [-span, +span] around (A0,B0)
static constexpr int NVFP4_TUNE_GRID_SIZE = 2 * NVFP4_TUNE_GRID_HALF_SPAN + 1;
static constexpr int NVFP4_TUNE_GRID_CENTER = NVFP4_TUNE_GRID_HALF_SPAN;
static constexpr int NVFP4_TUNE_GRID_EDGE_SKIP = 2;
static constexpr int NVFP4_TUNE_POOL_SIZE = 16;
static constexpr int NVFP4_TUNE_FIXED_POOL_SIZE = 14;
static constexpr int NVFP4_COMPAND_TOPK = 6; // q98 for 256 and q68 for 16
static constexpr int NVFP4_TUNE_INITIAL_SAMPLE_BLOCKS = 256;
static constexpr int NVFP4_TUNE_MAX_SAMPLE_BLOCKS = 1024;
static constexpr int NVFP4_TUNE_CAP_COMPARE_NBS[] = { 6, 16, 24, 64, 256, 512, 1024 };
static constexpr int NVFP4_TUNE_GUARD_TOPK = 64;
static constexpr int NVFP4_TUNE_GUARD_TOPK_FAST = 8;
static constexpr int NVFP4_TUNE_RESAMPLE_CAND_EXTRA = 32;

static constexpr float NVFP4_TUNE_GUARD_MAX_OBJ_FRAC = 1.02f;
static constexpr float NVFP4_TUNE_GUARD_MAX_REL_OBJ_FRAC = 1.12f;
static constexpr float NVFP4_TUNE_GUARD_TAIL_REL_OBJ_FRAC = 1.10f;
static constexpr float NVFP4_TUNE_CAP_OBJ_NORM_FRAC = 1.20f;
static constexpr float NVFP4_TUNE_CAP_REL_RMSE_FRAC = 1.10f;
static constexpr float NVFP4_TUNE_CAP_REL_RMSE_ABS = 1.00f;
static constexpr float NVFP4_TUNE_CAP_P95_REL_OBJ_FRAC = 1.10f;
static constexpr float NVFP4_TUNE_CAP_MAX_REL_OBJ_FRAC = 1.16f;
static constexpr float NVFP4_TUNE_CAP_TAIL_REL_OBJ_FRAC = 1.10f;
static constexpr float NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL = 0.015f;
static constexpr float NVFP4_TUNE_MIN_IMPROVE_FRAC = 0.0015f;

static constexpr float NVFP4_TUNE_SCORE_W_OBJ_NORM = 0.40f;
static constexpr float NVFP4_TUNE_SCORE_W_P95_REL_OBJ = 0.25f;
static constexpr float NVFP4_TUNE_SCORE_W_TAIL_REL_OBJ = 0.20f;
static constexpr float NVFP4_TUNE_SCORE_W_MAX_REL_OBJ = 0.10f;
static constexpr float NVFP4_TUNE_SCORE_W_ABS_MEAN_ERR_REL = 0.05f;

static constexpr int NVFP4_TUNE_ADAPT_NB_MID = 512;
static constexpr int NVFP4_TUNE_ADAPT_NB_HI = 1024;
static constexpr double NVFP4_TUNE_ADAPT_OBJ_NORM_MID = 0.010;
static constexpr double NVFP4_TUNE_ADAPT_OBJ_NORM_HI = 0.015;
static constexpr double NVFP4_TUNE_ADAPT_P95_REL_OBJ_MID = 0.020;
static constexpr double NVFP4_TUNE_ADAPT_P95_REL_OBJ_HI = 0.045;
static constexpr double NVFP4_TUNE_ADAPT_MAX_REL_OBJ_MID = 0.090;
static constexpr double NVFP4_TUNE_ADAPT_MAX_REL_OBJ_HI = 0.140;
static constexpr double NVFP4_TUNE_ADAPT_TAIL_REL_OBJ_MID = 0.010;
static constexpr double NVFP4_TUNE_ADAPT_TAIL_REL_OBJ_HI = 0.015;

static constexpr float NVFP4_TUNE_OUTLIER_W = 0.22f;
static constexpr float NVFP4_TUNE_AB_STEP_SCALE = 1.0f;
static constexpr float NVFP4_MAX_FP8_TENSOR_SCALE_6 = GGML_NVFP4_MAX_FP8_TENSOR_SCALE;
static constexpr float NVFP4_MAX_FP8_TENSOR_SCALE_4 = 256.0f;

static inline nvfp4_cuda_runtime_cfg nvfp4_cuda_cfg_resolve(const nvfp4_cuda_runtime_cfg * cfg) {
    nvfp4_cuda_runtime_cfg out{};
    out.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
    out.refit_iters = NVFP4_REFIT_ITERS;
    out.use_compand_sat = 1;
    out.reserved_i32 = 0;
    out.cap_m6 = NVFP4_MAX_FP8_TENSOR_SCALE_6;
    out.cap_m4 = NVFP4_MAX_FP8_TENSOR_SCALE_4;

    if (!cfg) {
        return out;
    }

    if (cfg->choose46_mode >= NVFP4_CUDA_CHOOSE46_ADAPTIVE && cfg->choose46_mode <= NVFP4_CUDA_CHOOSE46_FORCE_M4) {
        out.choose46_mode = cfg->choose46_mode;
    }

    if (cfg->refit_iters >= 0) {
        out.refit_iters = cfg->refit_iters;
    }
    if (out.refit_iters > 64) {
        out.refit_iters = 64;
    }

    if (cfg->use_compand_sat == 0 || cfg->use_compand_sat == 1) {
        out.use_compand_sat = cfg->use_compand_sat;
    }

    if (isfinite(cfg->cap_m6) && cfg->cap_m6 > 0.0f) {
        out.cap_m6 = cfg->cap_m6;
    }
    if (isfinite(cfg->cap_m4) && cfg->cap_m4 > 0.0f) {
        out.cap_m4 = cfg->cap_m4;
    }

    if (!(isfinite(out.cap_m6) && out.cap_m6 > 0.0f)) {
        out.cap_m6 = NVFP4_MAX_FP8_TENSOR_SCALE_6;
    }
    if (!(isfinite(out.cap_m4) && out.cap_m4 > 0.0f)) {
        out.cap_m4 = NVFP4_MAX_FP8_TENSOR_SCALE_4;
    }

    if (out.cap_m6 < out.cap_m4) {
        out.cap_m4 = out.cap_m6;
    }

    return out;
}

static __device__ __forceinline__ uint8_t nvfp4_pick_sub_scale_choose46(
    const float * x16,
    const float * qw16,
    float d_fp32,
    int choose46_mode) {
    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) {
        return 0;
    }

    if (choose46_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
        return adaptive_block_scale_4_or_6(x16, qw16, d_fp32);
    }

    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        max_abs = fmaxf(max_abs, fabsf(x16[i]));
    }
    if (!(max_abs > 0.0f)) {
        return 0;
    }

    const float inv_anchor = (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) ? 0.25f : (1.0f / 6.0f);
    const float sb_ideal = (max_abs * inv_anchor) / d_fp32;
    return ggml_fp8_ue4m3_from_fp32(sb_ideal);
}

static __device__ __forceinline__ float nvfp4_pick_subblock_cap_mode(
    const float * x16,
    const float * qw16,
    float d_fp32,
    float max_fp8_high,
    float max_fp8_low,
    int choose46_mode) {
    float high = (isfinite(max_fp8_high) && max_fp8_high > 0.0f) ? max_fp8_high : NVFP4_MAX_FP8_TENSOR_SCALE_6;
    float low = (isfinite(max_fp8_low) && max_fp8_low > 0.0f) ? fminf(max_fp8_low, high) : high;

    if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M6) {
        return high;
    }
    if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
        return low;
    }

    return ggml_nvfp4_pick_subblock_fp8_cap(x16, qw16, d_fp32, high, low);
}

static __device__ __forceinline__ float warp_max_f32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

static inline bool nvfp4_is_device_ptr(const void * ptr) {
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
    return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

static __device__ __forceinline__ void nvfp4_fill_tune_pool(float * pool, float a, float b) {
    pool[0]  = 0.9918823242f;
    pool[1]  = 0.9864501953f;
    pool[2]  = 0.875f;
    pool[3]  = 0.9375f;
    pool[4]  = 0.96875f;
    pool[5]  = 1.0f;
    pool[6]  = 1.015625f;
    pool[7]  = 1.03125f;
    pool[8]  = 1.046875f;
    pool[9]  = 1.0625f;
    pool[10] = 1.09375f;
    pool[11] = 1.125f;
    pool[12] = 1.1875f;
    pool[13] = 1.25f;
    pool[0] = a;
    pool[1] = b;
}

static __device__ __forceinline__ float nvfp4_topk_abs_threshold(const float * x, int n, int topk) {
    float top[NVFP4_COMPAND_TOPK] = { 0.0f };
    const int k = topk > NVFP4_COMPAND_TOPK ? NVFP4_COMPAND_TOPK : topk;

    #pragma unroll 1
    for (int i = 0; i < n; ++i) {
        float v = fabsf(x[i]);
        if (!isfinite(v)) {
            v = 0.0f;
        }
        if (!(v > top[k - 1])) {
            continue;
        }

        int pos = k - 1;
        while (pos > 0 && v > top[pos - 1]) {
            top[pos] = top[pos - 1];
            --pos;
        }
        top[pos] = v;
    }

    return top[k - 1];
}

static __device__ __forceinline__ uint8_t nvfp4_pick_sub_scale_compand(
    const float * x16,
    const float * qw16,
    float d_fp32,
    float max_fp8_high,
    float max_fp8_low,
    int choose46_mode) {
    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) {
        return 0;
    }

    const float abs_q = nvfp4_topk_abs_threshold(x16, 16, NVFP4_COMPAND_TOPK);
    if (!(abs_q > 0.0f) || !isfinite(abs_q)) {
        return 0;
    }

    uint8_t b = nvfp4_pick_sub_scale_choose46(x16, qw16, d_fp32, choose46_mode);
    if (b > 0x7eu) b = 0x7eu;

    const float max_fp8_sub = nvfp4_pick_subblock_cap_mode(
        x16, qw16, d_fp32, max_fp8_high, max_fp8_low, choose46_mode);
    const float sb = fminf(ggml_fp8_ue4m3_to_fp32(b), max_fp8_sub);
    const uint8_t b_init = ggml_fp8_ue4m3_from_fp32(sb);
    uint8_t cap_b = ggml_fp8_ue4m3_from_fp32(max_fp8_sub);
    if (cap_b > 0x7eu) cap_b = 0x7eu;
    uint8_t b0 = b_init;
    if (b0 > cap_b) b0 = cap_b;

    uint8_t best_b = b0;
    float best_sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_fp32 * ggml_fp8_ue4m3_to_fp32(best_b));
    for (int delta = 1; delta <= 2; ++delta) {
        const int cand0 = (int) b0 - delta;
        const int cand1 = (int) b0 + delta;
        if (cand0 >= 0) {
            const uint8_t bc = (uint8_t) cand0;
            const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_fp32 * ggml_fp8_ue4m3_to_fp32(bc));
            if (sse < best_sse) {
                best_sse = sse;
                best_b = bc;
            }
        }
        if (cand1 <= (int) cap_b) {
            const uint8_t bc = (uint8_t) cand1;
            const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_fp32 * ggml_fp8_ue4m3_to_fp32(bc));
            if (sse < best_sse) {
                best_sse = sse;
                best_b = bc;
            }
        }
    }
    return best_b;
}

static __device__ __forceinline__ float nvfp4_effective_cap_for_dreq(
    const float * x256,
    const float * qw256,
    float d_probe,
    float max_fp8_high,
    float max_fp8_low,
    int choose46_mode) {
    float high = (isfinite(max_fp8_high) && max_fp8_high > 0.0f) ? max_fp8_high : NVFP4_MAX_FP8_TENSOR_SCALE_6;
    float low = (isfinite(max_fp8_low) && max_fp8_low > 0.0f) ? fminf(max_fp8_low, high) : high;

    if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M6) {
        return high;
    }
    if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
        return low;
    }

    if (high <= low || !(d_probe > 0.0f) || !isfinite(d_probe)) {
        return high;
    }

    float sum_cap = 0.0f;
    int n_cap = 0;
    #pragma unroll
    for (int sub = 0; sub < 16; ++sub) {
        const float * x16 = x256 + sub * 16;
        const float * qw16 = qw256 ? (qw256 + sub * 16) : nullptr;
        const float cap_sub = nvfp4_pick_subblock_cap_mode(
            x16, qw16, d_probe, high, low, choose46_mode);
        sum_cap += cap_sub;
        ++n_cap;
    }

    float cap_eff = (n_cap > 0) ? (sum_cap / (float) n_cap) : high;
    if (!(cap_eff > 0.0f) || !isfinite(cap_eff)) {
        cap_eff = high;
    }
    return cap_eff;
}

static __device__ __forceinline__ void nvfp4_refit_scales_256_dev(
    const float * x256, const float * qw256, uint8_t * codes256,
    float d_min, float * d_out, uint8_t * sb_scales,
    const float * pool, int pool_n, float max_fp8, float max_fp8_low, int choose46_mode) {

    float t[16];
    float max_t = 0.0f;

    #pragma unroll 1
    for (int sub = 0; sub < 16; ++sub) {
        float num = 0.0f, den = 0.0f;

        #pragma unroll 1
        for (int j = 0; j < 16; ++j) {
            const int idx = sub*16 + j;
            const float w = ggml_nvfp4_qw(qw256, idx);
            const float v = kvalues_fp4_float[codes256[idx] & 0x0F];
            num += w * x256[idx] * v;
            den += w * v * v;
        }

        float ti = (den > 0.0f) ? (num / den) : 0.0f;

        if (ti < 0.0f) {
            ti = -ti;
            #pragma unroll 1
            for (int j = 0; j < 16; ++j) codes256[sub*16 + j] ^= 0x08;
        }

        if (!isfinite(ti) || ti < 0.0f) ti = 0.0f;

        t[sub] = ti;
        max_t = fmaxf(max_t, ti);
    }

    const float d_min_eff = (isfinite(d_min) && d_min > 0.0f) ? d_min : 0.0f;

    float d_req = 0.0f;
    if (max_t > 0.0f && isfinite(max_t)) {
        float cap_req = max_fp8;
        const float d_probe = d_min_eff > 0.0f ? d_min_eff : (max_t / max_fp8);
        cap_req = nvfp4_effective_cap_for_dreq(x256, qw256, d_probe, max_fp8, max_fp8_low, choose46_mode);
        d_req = max_t / cap_req;
        if (!isfinite(d_req) || d_req < 0.0f) d_req = 0.0f;
    }

    const float d0 = fmaxf(d_min_eff, d_req);
    if (!(d0 > 0.0f) || !isfinite(d0)) {
        *d_out = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 16; ++i) sb_scales[i] = 0;
        return;
    }

    float best_d = d0;
    float best_obj = 1e30f;
    uint8_t best_sb[16] = { 0 };

    #pragma unroll 1
    for (int mi = 0; mi < 16; ++mi) {
        if (mi >= pool_n) break;

        const float d_eff = d0 * pool[mi];
        if (!(d_eff > 0.0f) || !isfinite(d_eff)) continue;

        uint8_t sb_tmp[16];
        #pragma unroll 1
        for (int sub = 0; sub < 16; ++sub) {
            const float * x16 = x256 + sub * 16;
            const float * qw16 = qw256 ? (qw256 + sub * 16) : nullptr;
            const float max_fp8_sub = nvfp4_pick_subblock_cap_mode(
                x16, qw16, d_eff, max_fp8, max_fp8_low, choose46_mode);
            float sb = t[sub] / d_eff;
            if (!isfinite(sb) || sb < 0.0f) sb = 0.0f;
            else if (sb > max_fp8_sub) sb = max_fp8_sub;

            const uint8_t b_init = ggml_fp8_ue4m3_from_fp32(sb);
            uint8_t cap_b = ggml_fp8_ue4m3_from_fp32(max_fp8_sub);
            if (cap_b > 0x7eu) cap_b = 0x7eu;
            uint8_t b0 = b_init;
            if (b0 > cap_b) b0 = cap_b;
            uint8_t best_b = b0;
            float best_sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(best_b));
            for (int delta = 1; delta <= 2; ++delta) {
                const int cand0 = (int) b0 - delta;
                const int cand1 = (int) b0 + delta;
                if (cand0 >= 0) {
                    const uint8_t bc = (uint8_t) cand0;
                    const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(bc));
                    if (sse < best_sse) {
                        best_sse = sse;
                        best_b = bc;
                    }
                }
                if (cand1 <= (int) cap_b) {
                    const uint8_t bc = (uint8_t) cand1;
                    const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(bc));
                    if (sse < best_sse) {
                        best_sse = sse;
                        best_b = bc;
                    }
                }
            }
            sb_tmp[sub] = best_b;
        }

        const float obj = ggml_nvfp4_obj_256_codes(x256, qw256, codes256, d_eff, sb_tmp);

        if (obj < best_obj) {
            best_obj = obj;
            best_d = d_eff;
            #pragma unroll 1
            for (int i = 0; i < 16; ++i) best_sb[i] = sb_tmp[i];
        }
    }

    *d_out = best_d;
    #pragma unroll 1
    for (int i = 0; i < 16; ++i) sb_scales[i] = best_sb[i];
}


static __device__ __forceinline__ void nvfp4_refit_scales_256_dev_coop(
    const float * x256,
    const float * qw256,
    uint8_t * codes256,
    float d_min,
    float * d_out,
    uint8_t * sb_scales,
    const float * pool,
    int pool_n,
    float max_fp8,
    float max_fp8_low,
    int choose46_mode,
    float * red0,
    float * red1,
    float * red2,
    float * t16,
    uint8_t * sb_tmp,
    uint8_t * best_sb,
    int * flip16,
    int tid) {

    const int sub = tid >> 4;
    const int lane = tid & 15;

    const float w = ggml_nvfp4_qw(qw256, tid);
    const float v = kvalues_fp4_float[codes256[tid] & 0x0F];
    red0[tid] = w * x256[tid] * v;
    red1[tid] = w * v * v;
    __syncthreads();

    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) {
            red0[tid] += red0[tid + stride];
            red1[tid] += red1[tid + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        float ti = (red1[tid] > 0.0f) ? (red0[tid] / red1[tid]) : 0.0f;
        int flip = 0;
        if (ti < 0.0f) {
            ti = -ti;
            flip = 1;
        }
        if (!isfinite(ti) || ti < 0.0f) {
            ti = 0.0f;
        }
        t16[sub] = ti;
        flip16[sub] = flip;
    }
    __syncthreads();

    if (flip16[sub]) {
        codes256[tid] ^= 0x08;
    }
    __syncthreads();

    if (tid < 16) {
        red2[tid] = t16[tid];
    }
    __syncthreads();

    for (int stride = 8; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red2[tid] = fmaxf(red2[tid], red2[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float max_t = red2[0];
        const float d_min_eff = (isfinite(d_min) && d_min > 0.0f) ? d_min : 0.0f;
        float d_req = 0.0f;
        if (max_t > 0.0f && isfinite(max_t)) {
            float cap_req = max_fp8;
            const float d_probe = d_min_eff > 0.0f ? d_min_eff : (max_t / max_fp8);
            cap_req = nvfp4_effective_cap_for_dreq(x256, qw256, d_probe, max_fp8, max_fp8_low, choose46_mode);
            d_req = max_t / cap_req;
            if (!isfinite(d_req) || d_req < 0.0f) {
                d_req = 0.0f;
            }
        }

        red2[0] = fmaxf(d_min_eff, d_req); // d0
    }
    __syncthreads();

    const float d0 = red2[0];
    if (!(d0 > 0.0f) || !isfinite(d0)) {
        if (tid == 0) {
            *d_out = 0.0f;
        }
        if (tid < 16) {
            sb_scales[tid] = 0;
        }
        __syncthreads();
        return;
    }

    float best_d_local = d0;
    float best_obj_local = 1e30f;

    for (int mi = 0; mi < pool_n; ++mi) {
        const float d_eff = d0 * pool[mi];
        if (!(d_eff > 0.0f) || !isfinite(d_eff)) {
            __syncthreads();
            continue;
        }

        if (tid < 16) {
            const float * x16 = x256 + tid * 16;
            const float * qw16 = qw256 ? (qw256 + tid * 16) : nullptr;
            const float max_fp8_sub = nvfp4_pick_subblock_cap_mode(
                x16, qw16, d_eff, max_fp8, max_fp8_low, choose46_mode);
            float sb = t16[tid] / d_eff;
            if (!isfinite(sb) || sb < 0.0f) {
                sb = 0.0f;
            } else if (sb > max_fp8_sub) {
                sb = max_fp8_sub;
            }

            const uint8_t b_init = ggml_fp8_ue4m3_from_fp32(sb);
            uint8_t cap_b = ggml_fp8_ue4m3_from_fp32(max_fp8_sub);
            if (cap_b > 0x7eu) cap_b = 0x7eu;
            uint8_t b0 = b_init;
            if (b0 > cap_b) b0 = cap_b;
            uint8_t best_b = b0;
            float best_sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(best_b));
            for (int delta = 1; delta <= 2; ++delta) {
                const int cand0 = (int) b0 - delta;
                const int cand1 = (int) b0 + delta;
                if (cand0 >= 0) {
                    const uint8_t bc = (uint8_t) cand0;
                    const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(bc));
                    if (sse < best_sse) {
                        best_sse = sse;
                        best_b = bc;
                    }
                }
                if (cand1 <= (int) cap_b) {
                    const uint8_t bc = (uint8_t) cand1;
                    const float sse = ggml_nvfp4_subblock_sse_w_best(x16, qw16, d_eff * ggml_fp8_ue4m3_to_fp32(bc));
                    if (sse < best_sse) {
                        best_sse = sse;
                        best_b = bc;
                    }
                }
            }
            sb_tmp[tid] = best_b;
        }
        __syncthreads();

        const float scale = d_eff * ggml_fp8_ue4m3_to_fp32(sb_tmp[sub]);
        float pe_sse = 0.0f;
        float pe_err = 0.0f;
        float pe_w = 0.0f;
        if (scale > 0.0f && isfinite(scale)) {
            const float wi = ggml_nvfp4_qw(qw256, tid);
            const uint8_t qi = codes256[tid] & 0x0F;
            const float q = scale * kvalues_nvfp4_float(qi);
            const float e = x256[tid] - q;
            pe_sse = wi * e * e;
            pe_err = wi * e;
            pe_w = wi;
        }

        red0[tid] = pe_sse;
        red1[tid] = pe_err;
        red2[tid] = pe_w;
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red0[tid] += red0[tid + stride];
                red1[tid] += red1[tid + stride];
                red2[tid] += red2[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float sse = red0[0];
            const float obj = sse;

            if (obj < best_obj_local) {
                best_obj_local = obj;
                best_d_local = d_eff;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    best_sb[i] = sb_tmp[i];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_out = best_d_local;
    }
    if (tid < 16) {
        sb_scales[tid] = best_sb[tid];
    }
    __syncthreads();
}

static __device__ __forceinline__ float nvfp4_quantize_block_shared(
    float * s_x, const float * s_qw, bool has_qw,
    uint8_t * s_codes, uint8_t * s_scales, float * s_block_scale,
    float * s_pool, float * warp_max, float * s_obj,
    int tid, int sub,
    float max_fp8_high, float max_fp8_low,
    int choose46_mode, int refit_iters, int use_compand_sat,
    float a, float b) {

    const float max_fp8_high_eff = (isfinite(max_fp8_high) && max_fp8_high > 0.0f) ? max_fp8_high : NVFP4_MAX_FP8_TENSOR_SCALE_6;
    const float max_fp8_low_eff = (isfinite(max_fp8_low) && max_fp8_low > 0.0f) ? fminf(max_fp8_low, max_fp8_high_eff) : max_fp8_high_eff;
    const int choose_mode = (choose46_mode >= NVFP4_CUDA_CHOOSE46_ADAPTIVE && choose46_mode <= NVFP4_CUDA_CHOOSE46_FORCE_M4)
        ? choose46_mode
        : NVFP4_CUDA_CHOOSE46_ADAPTIVE;
    int refit_n = refit_iters;
    if (refit_n < 0) refit_n = NVFP4_REFIT_ITERS;
    if (refit_n > 64) refit_n = 64;
    const bool use_compand = use_compand_sat != 0;

    float ax   = fabsf(s_x[tid]);
    float gmax = warp_reduce_max<32>(ax);

    if ((tid & 31) == 0) warp_max[tid >> 5] = gmax;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < 8) ? warp_max[tid] : 0.0f;
        v = warp_reduce_max<32>(v);
        if (tid == 0) warp_max[0] = v;
    }
    __syncthreads();

    gmax = warp_max[0];

    if (gmax < 1e-15f) {
        if (tid == 0) *s_block_scale = 0.0f;
        if (tid < 16) s_scales[tid] = 0;
        s_codes[tid] = 0;
        __syncthreads();
        return 0.0f;
    }

    if (tid == 0) {
        float cap = max_fp8_high_eff;
        if (choose_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
            cap = max_fp8_low_eff;
        } else if (choose_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
            cap = ggml_nvfp4_pick_max_fp8_tensor_scale(
                s_x, has_qw ? s_qw : nullptr, gmax, max_fp8_low_eff);
            if (cap > max_fp8_high_eff) {
                cap = max_fp8_high_eff;
            }
        }
        warp_max[1] = cap;
    }
    __syncthreads();

    const float max_fp8_eff = warp_max[1];
    const float d_min = (gmax * 0.25f) / max_fp8_eff;

    if (tid == 0) {
        float d = d_min;
        if (use_compand) {
            const float abs_q = nvfp4_topk_abs_threshold(s_x, 256, NVFP4_COMPAND_TOPK);
            d = abs_q * (1.0f / NVFP4_E2M1_MAX_VALUE);
            if (!isfinite(d) || d < 0.0f) {
                d = 0.0f;
            }
            if (d < d_min) {
                d = d_min;
            }
        }
        *s_block_scale = d;
    }
    __syncthreads();

    if (tid < 16) {
        const float * x16 = &s_x[tid * 16];
        const float * qw16 = has_qw ? &s_qw[tid * 16] : nullptr;
        s_scales[tid] = nvfp4_pick_sub_scale_compand(x16, qw16, *s_block_scale, max_fp8_eff, max_fp8_low_eff, choose_mode);
    }
    __syncthreads();

    {
        const float scale = (*s_block_scale) * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (scale > 0.0f && isfinite(scale)) ? best_index_nvfp4(s_x[tid], scale) : 0;
    }
    __syncthreads();

    const int pool_n = NVFP4_TUNE_POOL_SIZE;
    if (tid == 0) {
        s_pool[0]  = 0.9918823242f;
        s_pool[1]  = 0.9864501953f;
        s_pool[2]  = 0.875f;
        s_pool[3]  = 0.9375f;
        s_pool[4]  = 0.96875f;
        s_pool[5]  = 1.0f;
        s_pool[6]  = 1.015625f;
        s_pool[7]  = 1.03125f;
        s_pool[8]  = 1.046875f;
        s_pool[9]  = 1.0625f;
        s_pool[10] = 1.09375f;
        s_pool[11] = 1.125f;
        s_pool[12] = 1.1875f;
        s_pool[13] = 1.25f;
        s_pool[NVFP4_TUNE_FIXED_POOL_SIZE + 0] = a;
        s_pool[NVFP4_TUNE_FIXED_POOL_SIZE + 1] = b;
    }
    __syncthreads();

    for (int it = 0; it < refit_n; ++it) {
        if (tid == 0) {
            float block_scale = *s_block_scale;
            nvfp4_refit_scales_256_dev(
                s_x, has_qw ? s_qw : nullptr, s_codes,
                d_min, &block_scale, s_scales,
                s_pool, pool_n, max_fp8_eff, max_fp8_low_eff, choose_mode);
            *s_block_scale = block_scale;
        }
        __syncthreads();
        const float scale = (*s_block_scale) * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (scale > 0.0f && isfinite(scale)) ? best_index_nvfp4(s_x[tid], scale) : 0;
        __syncthreads();
    }

    if (tid == 0) {
        float block_scale = *s_block_scale;
        nvfp4_refit_scales_256_dev(
            s_x, has_qw ? s_qw : nullptr, s_codes,
            d_min, &block_scale, s_scales,
            s_pool, pool_n, max_fp8_eff, max_fp8_low_eff, choose_mode);
        *s_block_scale = block_scale;
    }
    __syncthreads();

    {
        const float scale = (*s_block_scale) * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (scale > 0.0f && isfinite(scale)) ? best_index_nvfp4(s_x[tid], scale) : 0;
    }
    __syncthreads();

    if (tid == 0) {
        *s_obj = ggml_nvfp4_obj_256_codes(s_x, has_qw ? s_qw : nullptr, s_codes, *s_block_scale, s_scales);
    }
    __syncthreads();
    return *s_obj;
}

// <<< nb blocks, 256 threads >>>
__global__ void quantize_blocks(
    const void * x, bool x_bf16, float x_scale,
    const float * qw, int nb_per_row,
    block_nvfp4 * y, int nb,
    float a, float b, float max_fp8_high, float max_fp8_low, int choose46_mode, int refit_iters, int use_compand_sat) {

    const int ib = (int) blockIdx.x;
    if (ib >= nb) return;

    const int ip = ib >> 2;
    const int il = ib & 3;

    const float * qw256 = qw ? (qw + (size_t) (ib % nb_per_row) * 256) : nullptr;

    __shared__ float s_x[256];
    __shared__ float s_qw[256];
    __shared__ uint8_t s_codes[256];
    __shared__ uint8_t s_scales[16];
    __shared__ float s_block_scale;
    __shared__ uint8_t s_scales_best[16];
    __shared__ float s_block_scale_best;
    __shared__ uint8_t s_codes_best[256];
    const int tid = (int) threadIdx.x;
    const int sub = tid >> 4;

    const size_t off = (size_t) ib * 256 + (size_t) tid;
    float xv;
    if (x_bf16) {
        const __nv_bfloat16 * xb = (const __nv_bfloat16 *) x;
        xv = __bfloat162float(xb[off]);
    } else {
        const float * xf = (const float *) x;
        xv = xf[off];
    }
    if (x_scale != 1.0f) xv = xv * (1.0f / x_scale);

    s_x[tid] = xv;
    if (qw256) s_qw[tid] = qw256[tid];
    __syncthreads();

    __shared__ float s_pool[NVFP4_TUNE_POOL_SIZE];
    __shared__ float warp_max[8];
    __shared__ float s_obj;

    const bool has_qw = qw256 != nullptr;

    nvfp4_quantize_block_shared(s_x, s_qw, has_qw, s_codes, s_scales, &s_block_scale, s_pool, warp_max, &s_obj, tid, sub, max_fp8_high, max_fp8_low, choose46_mode, refit_iters, use_compand_sat, a, b);

    s_codes_best[tid] = s_codes[tid];
    if (tid < 16) s_scales_best[tid] = s_scales[tid];
    if (tid == 0) {
        s_block_scale_best = s_block_scale;
    }
    __syncthreads();

    if (tid == 0) {
        uint8_t scale_bytes[16];

        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            const float s_rel = ggml_fp8_ue4m3_to_fp32(s_scales_best[s]);
            const float s_abs = s_block_scale_best * s_rel;
            if (!(s_abs > 0.0f) || !isfinite(s_abs)) {
                scale_bytes[s] = 0;
                continue;
            }
            uint8_t b = ggml_fp8_ue4m3_from_fp32(s_abs);
            if (b > 0x7eu) b = 0x7eu;
            scale_bytes[s] = b;
        }

        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            y[ip].scales[il][s] = scale_bytes[s];
        }

    }

    __syncthreads();

    if (tid == 0) {
        ggml_nvfp4_pack_codes_256(s_codes_best, y[ip].qs[il]);
    }
}
static bool ggml_cuda_nvfp4_quantize_impl(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg, cudaStream_t stream) {

    const nvfp4_cuda_runtime_cfg cfg_res = nvfp4_cuda_cfg_resolve(cfg);

    const bool trace_env = std::getenv("LLAMA_NVFP4_TRACE") != nullptr;
    static std::atomic<int> trace_count{0};
    const bool trace = trace_env && trace_count.fetch_add(1, std::memory_order_relaxed) == 0;

    if (nrow <= 0 || n_per_row <= 0) return false;
    if ((n_per_row % 256) != 0) return false;

    const int nb_per_row = (int) (n_per_row / 256);
    if ((nb_per_row % 4) != 0) return false;

    const int64_t nb_total = (int64_t) nrow * (int64_t) nb_per_row;
    const int64_t np_per_row = nb_per_row / 4;
    const int64_t np_total = (int64_t) nrow * np_per_row;

    cudaStream_t st = stream ? stream : 0;

    const size_t bytes_x  = (size_t) nrow * (size_t) n_per_row * (x_bf16 ? sizeof(uint16_t) : sizeof(float));
    const size_t bytes_qw = qw ? ((size_t) n_per_row * sizeof(float)) : 0; // ONE-ROW imatrix (nb_per_row * 256)
    const size_t bytes_y  = (size_t) np_total * sizeof(block_nvfp4);

    static thread_local void *         d_x_buf  = nullptr;
    static thread_local float *        d_qw_buf = nullptr;
    static thread_local block_nvfp4 *  d_y_buf  = nullptr;
    static thread_local size_t cap_x = 0, cap_qw = 0, cap_y = 0;

    const bool x_is_device  = nvfp4_is_device_ptr(x);
    const bool qw_is_device = qw && nvfp4_is_device_ptr(qw);
    const bool y_is_device  = nvfp4_is_device_ptr(vy);

    if (trace) {
        std::fprintf(stderr,
            "nvfp4 cuda: nrow=%lld n_per_row=%lld x_bf16=%d x_scale=%.6g bytes_x=%zu bytes_qw=%zu bytes_y=%zu x_dev=%d qw_dev=%d y_dev=%d cfg={choose46=%d,refit=%d,compand=%d,cap6=%.3f,cap4=%.3f}\n",
            (long long) nrow, (long long) n_per_row, x_bf16 ? 1 : 0, (double) x_scale,
            bytes_x, bytes_qw, bytes_y,
            x_is_device ? 1 : 0, qw_is_device ? 1 : 0, y_is_device ? 1 : 0,
            cfg_res.choose46_mode, cfg_res.refit_iters, cfg_res.use_compand_sat,
            (double) cfg_res.cap_m6, (double) cfg_res.cap_m4);
    }

    // ---- X ----
    void * d_x = nullptr;
    if (x_is_device) {
        d_x = const_cast<void *>(x);
    } else {
        if (!d_x_buf || cap_x < bytes_x) {
            if (d_x_buf) cudaFree(d_x_buf);
            CUDA_CHECK(cudaMalloc(&d_x_buf, bytes_x));
            cap_x = bytes_x;
        }
        d_x = d_x_buf;
        CUDA_CHECK(cudaMemcpyAsync(d_x, x, bytes_x, cudaMemcpyHostToDevice, st));
    }

    // ---- QW (ONE ROW) ----
    float * d_qw = nullptr;
    if (!qw) {
        d_qw = nullptr;
    } else if (qw_is_device) {
        d_qw = const_cast<float *>(qw);
    } else {
        if (!d_qw_buf || cap_qw < bytes_qw) {
            if (d_qw_buf) cudaFree(d_qw_buf);
            CUDA_CHECK(cudaMalloc(&d_qw_buf, bytes_qw));
            cap_qw = bytes_qw;
        }
        d_qw = d_qw_buf;
        CUDA_CHECK(cudaMemcpyAsync(d_qw, qw, bytes_qw, cudaMemcpyHostToDevice, st));
    }

    // ---- Y ----
    block_nvfp4 * d_y = nullptr;
    if (y_is_device) {
        d_y = reinterpret_cast<block_nvfp4 *>(vy);
    } else {
        if (!d_y_buf || cap_y < bytes_y) {
            if (d_y_buf) cudaFree(d_y_buf);
            CUDA_CHECK(cudaMalloc(&d_y_buf, bytes_y));
            cap_y = bytes_y;
        }
        d_y = d_y_buf;
    }

    const float x_scale_eff = (isfinite(x_scale) && x_scale > 0.0f) ? x_scale : 1.0f;

    cudaGetLastError();
    quantize_blocks<<<(int) nb_total, 256, 0, st>>>(
        d_x, x_bf16, x_scale_eff,
        d_qw, nb_per_row,
        d_y, (int) nb_total,
        a, b,
        cfg_res.cap_m6, cfg_res.cap_m4,
        cfg_res.choose46_mode, cfg_res.refit_iters, cfg_res.use_compand_sat);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (!y_is_device) {
        CUDA_CHECK(cudaMemcpyAsync(vy, d_y, bytes_y, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
    }

    return true;
}

extern "C" bool ggml_cuda_nvfp4_quantize(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, nullptr, stream);
}

extern "C" bool ggml_cuda_nvfp4_quantize_cfg(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg, cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, cfg, stream);
}


__global__ void autotune(
    const float * __restrict__ x, const float * __restrict__ qw, int64_t sample_nb,
    float * __restrict__ results_obj,
    int choose46_mode, int refit_iters, int use_compand_sat,
    float cap_m6, float cap_m4) {

    int ai = (int) blockIdx.x - NVFP4_TUNE_GRID_CENTER;
    int bi = (int) blockIdx.y - NVFP4_TUNE_GRID_CENTER;

    float a = NVFP4_A0 + ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
    float b = NVFP4_B0 + bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
    const int choose_mode = (choose46_mode >= NVFP4_CUDA_CHOOSE46_ADAPTIVE && choose46_mode <= NVFP4_CUDA_CHOOSE46_FORCE_M4)
        ? choose46_mode
        : NVFP4_CUDA_CHOOSE46_ADAPTIVE;
    int refit_n = refit_iters;
    if (refit_n < 0) refit_n = NVFP4_TUNE_REFIT_ITERS;
    if (refit_n > 64) refit_n = 64;
    const bool use_compand = use_compand_sat != 0;
    const float max_fp8_high_eff = (isfinite(cap_m6) && cap_m6 > 0.0f) ? cap_m6 : NVFP4_MAX_FP8_TENSOR_SCALE_6;
    float max_fp8_low_eff = (isfinite(cap_m4) && cap_m4 > 0.0f) ? cap_m4 : NVFP4_MAX_FP8_TENSOR_SCALE_4;
    if (max_fp8_low_eff > max_fp8_high_eff) {
        max_fp8_low_eff = max_fp8_high_eff;
    }

    __shared__ float local_pool[NVFP4_TUNE_POOL_SIZE];
    if (threadIdx.x == 0) {
        nvfp4_fill_tune_pool(local_pool, a, b);
    }
    __syncthreads();
    const int pool_n = NVFP4_TUNE_POOL_SIZE;

    __shared__ float s_red_sse[256];
    __shared__ float s_sum_obj;
    __shared__ float s_refit_red0[256];
    __shared__ float s_refit_red1[256];
    __shared__ float s_refit_red2[256];
    __shared__ float s_refit_t16[16];
    __shared__ uint8_t s_refit_sb_tmp[16];
    __shared__ uint8_t s_refit_best_sb[16];
    __shared__ int s_refit_flip16[16];
    __shared__ float s_sum_obj2;
    __shared__ uint8_t s_codes[256];
    __shared__ uint8_t s_scales[16];
    __shared__ float s_d;

    const int tid = (int) threadIdx.x;

    if (tid == 0) {
        s_sum_obj = 0.0f;
        s_sum_obj2 = 0.0f;
    }
    __syncthreads();

    for (int ib = 0; ib < sample_nb; ++ib) {
        const float * x256  = x  + (int64_t) ib * 256;
        const float * qw256 = qw ? (qw + (int64_t) ib * 256) : NULL;

        __shared__ float s_warpmax[8];
        __shared__ float s_gmax;
        const int lane = tid & 31;
        const int wid  = tid >> 5;

        float v = fabsf(x256[tid]);
        v = warp_max_f32(v);

        if (lane == 0) s_warpmax[wid] = v;
        __syncthreads();

        float v2 = (tid < 8) ? s_warpmax[tid] : 0.0f;
        if (wid == 0) v2 = warp_max_f32(v2);

        if (tid == 0) s_gmax = v2;
        __syncthreads();

        float gmax = s_gmax;
        if (gmax < 1e-15f) continue;

        float max_fp8_eff = max_fp8_high_eff;
        if (choose_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
            max_fp8_eff = max_fp8_low_eff;
        } else if (choose_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
            max_fp8_eff = ggml_nvfp4_pick_max_fp8_tensor_scale(
                x256, qw256, gmax, max_fp8_low_eff);
            if (max_fp8_eff > max_fp8_high_eff) {
                max_fp8_eff = max_fp8_high_eff;
            }
        }
        if (!(max_fp8_eff > 0.0f) || !isfinite(max_fp8_eff)) {
            max_fp8_eff = max_fp8_high_eff;
        }
        const float d_min = (gmax * 0.25f) / max_fp8_eff;
        if (tid == 0) {
            float d = d_min;
            if (use_compand) {
                const float abs_q = nvfp4_topk_abs_threshold(x256, 256, NVFP4_COMPAND_TOPK);
                d = abs_q * (1.0f / NVFP4_E2M1_MAX_VALUE);
                if (!isfinite(d) || d < 0.0f) {
                    d = 0.0f;
                }
                if (d < d_min) {
                    d = d_min;
                }
            }
            s_d = d;
        }
        __syncthreads();

        if (tid < 16) {
            const float * x16 = x256 + tid * 16;
            const float * qw16 = qw256 ? qw256 + tid * 16 : NULL;
            s_scales[tid] = nvfp4_pick_sub_scale_compand(
                x16, qw16, s_d, max_fp8_eff, max_fp8_low_eff, choose_mode);
        }
        __syncthreads();

        const int sub = tid >> 4;
        const float sub_scale0 = s_d * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (sub_scale0 > 0.0f && isfinite(sub_scale0)) ? best_index_nvfp4(x256[tid], sub_scale0) : 0;
        __syncthreads();

        #pragma unroll
        for (int it = 0; it < refit_n; ++it) {
            nvfp4_refit_scales_256_dev_coop(
                x256, qw256, s_codes, d_min, &s_d, s_scales, local_pool, pool_n, max_fp8_eff, max_fp8_low_eff, choose_mode,
                s_refit_red0, s_refit_red1, s_refit_red2, s_refit_t16,
                s_refit_sb_tmp, s_refit_best_sb, s_refit_flip16, tid);
            __syncthreads();

            const float sub_scale_it = s_d * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
            s_codes[tid] = (sub_scale_it > 0.0f && isfinite(sub_scale_it)) ? best_index_nvfp4(x256[tid], sub_scale_it) : 0;
            __syncthreads();
        }

        nvfp4_refit_scales_256_dev_coop(
            x256, qw256, s_codes, d_min, &s_d, s_scales, local_pool, pool_n, max_fp8_eff, max_fp8_low_eff, choose_mode,
                s_refit_red0, s_refit_red1, s_refit_red2, s_refit_t16,
            s_refit_sb_tmp, s_refit_best_sb, s_refit_flip16, tid);
        __syncthreads();

        const float scale = s_d * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        float pe_sse = 0.0f;

        if (scale > 0.0f && isfinite(scale)) {
            const float w = ggml_nvfp4_qw(qw256, tid);
            const uint8_t qi = s_codes[tid] & 0xF;
            const float q = scale * kvalues_nvfp4_float(qi);
            const float e = x256[tid] - q;
            pe_sse = w * e * e;
        } else {
            // Prevent objective-grid collapse when a candidate lands in an invalid scale state.
            const float xi = x256[tid];
            const float w  = ggml_nvfp4_qw(qw256, tid);
            const bool bad_scale = !isfinite(scale) || scale < 0.0f;
            pe_sse = w * xi * xi * (bad_scale ? 4.0f : 1.0f);
        }

        s_red_sse[tid] = pe_sse;
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_red_sse[tid] += s_red_sse[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float sse = s_red_sse[0];
            const float obj_block = sse;
            s_sum_obj += obj_block;
            s_sum_obj2 += obj_block * obj_block;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float tail_term = (sample_nb > 0) ? sqrtf(s_sum_obj2 / (float) sample_nb) : 0.0f;
        const float score = s_sum_obj + NVFP4_TUNE_OUTLIER_W * (float) sample_nb * tail_term;
        results_obj[blockIdx.y * NVFP4_TUNE_GRID_SIZE + blockIdx.x] = score;
    }
}

__global__ void metrics(
    const float * __restrict__ x, const float * __restrict__ qw, int64_t sample_nb,
    float a, float b,
    int choose46_mode, int refit_iters, int use_compand_sat,
    float cap_m6, float cap_m4,
    float * __restrict__ out_sse,
    float * __restrict__ out_sum_err,
    float * __restrict__ out_sum_w,
    float * __restrict__ out_sum_x2) {

    const int tid = (int) threadIdx.x;
    const int sub = tid >> 4;
    const int choose_mode = (choose46_mode >= NVFP4_CUDA_CHOOSE46_ADAPTIVE && choose46_mode <= NVFP4_CUDA_CHOOSE46_FORCE_M4)
        ? choose46_mode
        : NVFP4_CUDA_CHOOSE46_ADAPTIVE;
    int refit_n = refit_iters;
    if (refit_n < 0) refit_n = NVFP4_TUNE_REFIT_ITERS;
    if (refit_n > 64) refit_n = 64;
    const bool use_compand = use_compand_sat != 0;
    const float max_fp8_high_eff = (isfinite(cap_m6) && cap_m6 > 0.0f) ? cap_m6 : NVFP4_MAX_FP8_TENSOR_SCALE_6;
    float max_fp8_low_eff = (isfinite(cap_m4) && cap_m4 > 0.0f) ? cap_m4 : NVFP4_MAX_FP8_TENSOR_SCALE_4;
    if (max_fp8_low_eff > max_fp8_high_eff) {
        max_fp8_low_eff = max_fp8_high_eff;
    }

    const int ib = (int) blockIdx.x;
    if (ib >= sample_nb) return;
    const float * x256  = x  + (int64_t) ib * 256;
    const float * qw256 = qw ? (qw + (int64_t) ib * 256) : NULL;

    __shared__ uint8_t s_codes[256];
    __shared__ uint8_t s_scales[16];
    __shared__ float s_block_scale;
    __shared__ float s_refit_red0[256];
    __shared__ float s_refit_red1[256];
    __shared__ float s_refit_red2[256];
    __shared__ float s_refit_t16[16];
    __shared__ uint8_t s_refit_sb_tmp[16];
    __shared__ uint8_t s_refit_best_sb[16];
    __shared__ int s_refit_flip16[16];

    float ax = fabsf(x256[tid]);
    float gmax = warp_max_f32(ax);

    __shared__ float warp_max[8];
    if ((tid & 31) == 0) warp_max[tid >> 5] = gmax;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < 8) ? warp_max[tid] : 0.0f;
        v = warp_max_f32(v);
        if (tid == 0) warp_max[0] = v;
    }
    __syncthreads();

    gmax = warp_max[0];
    if (gmax < 1e-15f) {
        if (tid == 0) {
            out_sse[ib] = 0.0f;
            out_sum_err[ib] = 0.0f;
            out_sum_w[ib] = 0.0f;
            out_sum_x2[ib] = 0.0f;
        }
        return;
    }

    float max_fp8_eff = max_fp8_high_eff;
    if (choose_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
        max_fp8_eff = max_fp8_low_eff;
    } else if (choose_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
        max_fp8_eff = ggml_nvfp4_pick_max_fp8_tensor_scale(
            x256, qw256, gmax, max_fp8_low_eff);
        if (max_fp8_eff > max_fp8_high_eff) {
            max_fp8_eff = max_fp8_high_eff;
        }
    }
    if (!(max_fp8_eff > 0.0f) || !isfinite(max_fp8_eff)) {
        max_fp8_eff = max_fp8_high_eff;
    }
    const float d_min = (gmax * 0.25f) / max_fp8_eff;
    if (tid == 0) {
        float d = d_min;
        if (use_compand) {
            const float abs_q = nvfp4_topk_abs_threshold(x256, 256, NVFP4_COMPAND_TOPK);
            d = abs_q * (1.0f / NVFP4_E2M1_MAX_VALUE);
            if (!isfinite(d) || d < 0.0f) {
                d = 0.0f;
            }
            if (d < d_min) {
                d = d_min;
            }
        }
        s_block_scale = d;
    }
    __syncthreads();

    if (tid < 16) {
        const float * x16  = x256 + tid * 16;
        const float * qw16 = qw256 ? (qw256 + tid * 16) : NULL;
        s_scales[tid] = nvfp4_pick_sub_scale_compand(
            x16, qw16, s_block_scale, max_fp8_eff, max_fp8_low_eff, choose_mode);
    }
    __syncthreads();

    {
        const float scale = s_block_scale * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (scale > 0.0f && isfinite(scale)) ? best_index_nvfp4(x256[tid], scale) : 0;
    }
    __syncthreads();

    __shared__ float pool[NVFP4_TUNE_POOL_SIZE];
    if (tid == 0) {
        nvfp4_fill_tune_pool(pool, a, b);
    }
    __syncthreads();
    const int pool_n = NVFP4_TUNE_POOL_SIZE;

    #pragma unroll
    for (int it = 0; it < refit_n; ++it) {
        nvfp4_refit_scales_256_dev_coop(
            x256, qw256, s_codes, d_min, &s_block_scale, s_scales, pool, pool_n, max_fp8_eff, max_fp8_low_eff, choose_mode,
                s_refit_red0, s_refit_red1, s_refit_red2, s_refit_t16,
            s_refit_sb_tmp, s_refit_best_sb, s_refit_flip16, tid);
        __syncthreads();

        const float scale = s_block_scale * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
        s_codes[tid] = (scale > 0.0f && isfinite(scale)) ? best_index_nvfp4(x256[tid], scale) : 0;
        __syncthreads();
    }

    __shared__ float red_sse[256];
    __shared__ float red_err[256];
    __shared__ float red_w[256];
    __shared__ float red_x2[256];

    const float scale = s_block_scale * ggml_fp8_ue4m3_to_fp32(s_scales[sub]);
    float pe_sse = 0.0f;
    float pe_err = 0.0f;
    float pe_w = 0.0f;
    float pe_x2 = 0.0f;

    if (scale > 0.0f && isfinite(scale)) {
        const float xi = x256[tid];
        const float w = ggml_nvfp4_qw(qw256, tid);
        const float q = scale * kvalues_fp4_float[s_codes[tid] & 0x0F];
        const float e = xi - q;
        pe_sse = w * e * e;
        pe_err = w * e;
        pe_w = w;
        pe_x2 = w * xi * xi;
    } else {
        // Maintain finite tuner metrics for invalid candidate scales.
        // NaN/Inf/negative scales receive a stronger SSE penalty.
        const float xi = x256[tid];
        const float w = ggml_nvfp4_qw(qw256, tid);
        const bool bad_scale = !isfinite(scale) || scale < 0.0f;
        pe_sse = w * xi * xi * (bad_scale ? 4.0f : 1.0f);
        pe_err = w * xi;
        pe_w = w;
        pe_x2 = w * xi * xi;
    }

    red_sse[tid] = pe_sse;
    red_err[tid] = pe_err;
    red_w[tid] = pe_w;
    red_x2[tid] = pe_x2;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_sse[tid] += red_sse[tid + stride];
            red_err[tid] += red_err[tid + stride];
            red_w[tid] += red_w[tid + stride];
            red_x2[tid] += red_x2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_sse[ib] = red_sse[0];
        out_sum_err[ib] = red_err[0];
        out_sum_w[ib] = red_w[0];
        out_sum_x2[ib] = red_x2[0];
    }
}

__global__ void nvfp4_gather_blocks_256(
    const float * __restrict__ src,
    float * __restrict__ dst,
    const int32_t * __restrict__ block_idx,
    int64_t n_blocks) {

    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_blocks * 256;
    if (tid >= n) {
        return;
    }
    const int64_t ib = tid >> 8;
    const int64_t j = tid & 255;
    dst[tid] = src[(int64_t) block_idx[ib] * 256 + j];
}

extern "C" void ggml_cuda_nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, cudaStream_t stream) {
    const bool trace = std::getenv("LLAMA_NVFP4_TRACE") != nullptr;
    const auto tune_t0 = std::chrono::steady_clock::now();

    int64_t nb = n / 256;
    int64_t sample_nb = nb < NVFP4_TUNE_INITIAL_SAMPLE_BLOCKS ? nb : NVFP4_TUNE_INITIAL_SAMPLE_BLOCKS;

    fprintf(stderr,
        "NVFP4_TUNE (CUDA start) blocks=%lld sample0=%lld sample_max=%d\n",
        (long long) nb, (long long) sample_nb, NVFP4_TUNE_MAX_SAMPLE_BLOCKS);

    if (sample_nb == 0) {
        *best_a = NVFP4_A0;
        *best_b = NVFP4_B0;
        return;
    }

    cudaStream_t st = stream ? stream : 0;
    const int64_t sample_elems = sample_nb * 256;

    static thread_local float * d_x_buf = nullptr;
    static thread_local float * d_qw_buf = nullptr;
    static thread_local float * d_x_low_buf = nullptr;
    static thread_local float * d_qw_low_buf = nullptr;
    static thread_local float * d_results = nullptr;
    static thread_local float * d_sse = nullptr;
    static thread_local float * d_sum_err = nullptr;
    static thread_local float * d_sum_w = nullptr;
    static thread_local float * d_sum_x2 = nullptr;
    static thread_local float * d_sse_batch = nullptr;
    static thread_local float * d_sum_err_batch = nullptr;
    static thread_local float * d_sum_w_batch = nullptr;
    static thread_local float * d_sum_x2_batch = nullptr;

    // metrics buffers are sized by sample_nb. they must be reallocated if sample_nb grows.
    static thread_local size_t cap_metrics = 0;

    static thread_local size_t cap_x = 0;
    static thread_local size_t cap_qw = 0;
    static thread_local size_t cap_x_low = 0;
    static thread_local size_t cap_qw_low = 0;
    static thread_local size_t cap_h_results = 0;
    static thread_local float * h_results_pinned = nullptr;
    static thread_local size_t cap_h_metric_vec = 0;
    static thread_local float * h_sse_pinned = nullptr;
    static thread_local float * h_sum_err_pinned = nullptr;
    static thread_local float * h_sum_w_pinned = nullptr;
    static thread_local float * h_sum_x2_pinned = nullptr;
    static thread_local size_t cap_metrics_batch = 0;
    static thread_local size_t cap_h_metric_batch = 0;
    static thread_local float * h_sse_batch_pinned = nullptr;
    static thread_local float * h_sum_err_batch_pinned = nullptr;
    static thread_local float * h_sum_w_batch_pinned = nullptr;
    static thread_local float * h_sum_x2_batch_pinned = nullptr;
    static thread_local int32_t * d_sample_idx = nullptr;
    static thread_local size_t cap_sample_idx = 0;
    static thread_local std::vector<int32_t> h_sample_idx;
    static thread_local float * h_sample_blocks_pinned = nullptr;
    static thread_local size_t cap_h_sample_blocks = 0;

    struct tune_policy_cfg {
        const char * name;
        int choose46_mode;
        int refit_iters;
        int use_compand_sat;
        float cap_m6;
        float cap_m4;
    };

    auto normalize_policy = [](tune_policy_cfg cfg) -> tune_policy_cfg {
        if (cfg.choose46_mode < NVFP4_CUDA_CHOOSE46_ADAPTIVE || cfg.choose46_mode > NVFP4_CUDA_CHOOSE46_FORCE_M4) {
            cfg.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
        }
        if (cfg.refit_iters < 0) cfg.refit_iters = NVFP4_TUNE_REFIT_ITERS;
        if (cfg.refit_iters > 64) cfg.refit_iters = 64;
        if (cfg.use_compand_sat != 0 && cfg.use_compand_sat != 1) {
            cfg.use_compand_sat = 1;
        }
        if (!(isfinite(cfg.cap_m6) && cfg.cap_m6 > 0.0f)) {
            cfg.cap_m6 = NVFP4_MAX_FP8_TENSOR_SCALE_6;
        }
        if (!(isfinite(cfg.cap_m4) && cfg.cap_m4 > 0.0f)) {
            cfg.cap_m4 = NVFP4_MAX_FP8_TENSOR_SCALE_4;
        }
        if (cfg.cap_m4 > cfg.cap_m6) {
            cfg.cap_m4 = cfg.cap_m6;
        }
        if (!cfg.name) {
            cfg.name = "unnamed";
        }
        return cfg;
    };

    auto same_policy = [](const tune_policy_cfg & a, const tune_policy_cfg & b) -> bool {
        return a.choose46_mode == b.choose46_mode &&
               a.refit_iters == b.refit_iters &&
               a.use_compand_sat == b.use_compand_sat &&
               a.cap_m6 == b.cap_m6 &&
               a.cap_m4 == b.cap_m4;
    };

    tune_policy_cfg active_policy = normalize_policy({
        "baseline_auto",
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        NVFP4_TUNE_REFIT_ITERS,
        1,
        NVFP4_MAX_FP8_TENSOR_SCALE_6,
        NVFP4_MAX_FP8_TENSOR_SCALE_4,
    });

    auto sample_block_index = [](int64_t src_nb, int64_t is, int64_t use_nb) -> int64_t {
        if (src_nb <= 1 || use_nb <= 1) {
            return 0;
        }
        return (is * (src_nb - 1)) / (use_nb - 1);
    };

    auto ensure_sample_idx = [&](int64_t src_nb, int64_t use_nb) {
        if ((int64_t) h_sample_idx.size() < use_nb) {
            h_sample_idx.resize((size_t) use_nb);
        }
        for (int64_t is = 0; is < use_nb; ++is) {
            h_sample_idx[(size_t) is] = (int32_t) sample_block_index(src_nb, is, use_nb);
        }
        if (!d_sample_idx || cap_sample_idx < (size_t) use_nb) {
            if (d_sample_idx) cudaFree(d_sample_idx);
            CUDA_CHECK(cudaMalloc(&d_sample_idx, (size_t) use_nb * sizeof(int32_t)));
            cap_sample_idx = (size_t) use_nb;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_sample_idx, h_sample_idx.data(), (size_t) use_nb * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    };

    auto ensure_host_sample_blocks = [&](size_t n_elems) {
        if (!h_sample_blocks_pinned || cap_h_sample_blocks < n_elems) {
            if (h_sample_blocks_pinned) {
                CUDA_CHECK(cudaFreeHost(h_sample_blocks_pinned));
            }
            CUDA_CHECK(cudaHostAlloc((void **) &h_sample_blocks_pinned, n_elems * sizeof(float), cudaHostAllocDefault));
            cap_h_sample_blocks = n_elems;
        }
    };

    auto copy_sampled_blocks = [&](const float * src, bool src_is_device, float * dst, int64_t use_nb) {
        const size_t bytes = (size_t) use_nb * 256 * sizeof(float);
        if (use_nb == nb) {
            const cudaMemcpyKind kind = src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, st));
            return;
        }

        ensure_sample_idx(nb, use_nb);
        if (src_is_device) {
            const int threads_g = 256;
            const int blocks_g = (int) (((int64_t) use_nb * 256 + threads_g - 1) / threads_g);
            nvfp4_gather_blocks_256<<<blocks_g, threads_g, 0, st>>>(src, dst, d_sample_idx, use_nb);
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        ensure_host_sample_blocks((size_t) use_nb * 256);
        for (int64_t is = 0; is < use_nb; ++is) {
            const int64_t ib = h_sample_idx[(size_t) is];
            memcpy(h_sample_blocks_pinned + (size_t) is * 256,
                   src + (size_t) ib * 256,
                   (size_t) 256 * sizeof(float));
        }
        CUDA_CHECK(cudaMemcpyAsync(dst, h_sample_blocks_pinned, bytes, cudaMemcpyHostToDevice, st));
    };

    auto downsample_device_blocks = [&](const float * src, float * dst, int64_t src_nb, int64_t use_nb) {
        const size_t bytes = (size_t) use_nb * 256 * sizeof(float);
        if (use_nb == src_nb) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st));
            return;
        }
        ensure_sample_idx(src_nb, use_nb);
        const int threads_g = 256;
        const int blocks_g = (int) (((int64_t) use_nb * 256 + threads_g - 1) / threads_g);
        nvfp4_gather_blocks_256<<<blocks_g, threads_g, 0, st>>>(src, dst, d_sample_idx, use_nb);
        CUDA_CHECK(cudaGetLastError());
    };

    const size_t bytes_x = (size_t) sample_elems * sizeof(float);
    if (!d_x_buf || cap_x < bytes_x) {
        if (d_x_buf) cudaFree(d_x_buf);
        CUDA_CHECK(cudaMalloc(&d_x_buf, bytes_x));
        cap_x = bytes_x;
    }

    const bool x_is_device = nvfp4_is_device_ptr(x);
    copy_sampled_blocks(x, x_is_device, d_x_buf, sample_nb);

    // Sample qw over the same block subset as x.
    float * d_qw_use = nullptr;
    if (qw) {
        const size_t bytes_qw = (size_t) sample_elems * sizeof(float);
        if (!d_qw_buf || cap_qw < bytes_qw) {
            if (d_qw_buf) cudaFree(d_qw_buf);
            CUDA_CHECK(cudaMalloc(&d_qw_buf, bytes_qw));
            cap_qw = bytes_qw;
        }

        copy_sampled_blocks(qw, nvfp4_is_device_ptr(qw), d_qw_buf, sample_nb);
        d_qw_use = d_qw_buf;
    }

    if (!d_results) CUDA_CHECK(cudaMalloc(&d_results, NVFP4_TUNE_GRID_SIZE * NVFP4_TUNE_GRID_SIZE * sizeof(float)));

    dim3 blocks(NVFP4_TUNE_GRID_SIZE, NVFP4_TUNE_GRID_SIZE);
    dim3 threads(256);
    autotune<<<blocks, threads, 0, st>>>(
        d_x_buf, d_qw_use, sample_nb, d_results,
        active_policy.choose46_mode, active_policy.refit_iters, active_policy.use_compand_sat,
        active_policy.cap_m6, active_policy.cap_m4);
    CUDA_CHECK(cudaGetLastError());

    const int n_scores = NVFP4_TUNE_GRID_SIZE * NVFP4_TUNE_GRID_SIZE;
    const size_t bytes_scores = (size_t) n_scores * sizeof(float);
    int best_lin = 0;

    if (!h_results_pinned || cap_h_results < (size_t) n_scores) {
        if (h_results_pinned) {
            CUDA_CHECK(cudaFreeHost(h_results_pinned));
        }
        CUDA_CHECK(cudaHostAlloc((void **) &h_results_pinned, bytes_scores, cudaHostAllocDefault));
        cap_h_results = (size_t) n_scores;
    }
    CUDA_CHECK(cudaMemcpyAsync(h_results_pinned, d_results, bytes_scores, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    float min_obj = 1e30f;
    float max_obj = -1e30f;
    int min_lin = -1;
    int max_lin = -1;
    for (int i = 0; i < n_scores; ++i) {
        const float v = h_results_pinned[i];
        if (v < min_obj) {
            min_obj = v;
            min_lin = i;
        }
        if (v > max_obj) {
            max_obj = v;
            max_lin = i;
        }
    }
    const bool score_flat = std::isfinite(min_obj) && std::isfinite(max_obj) && max_obj <= min_obj * 1.000001f;

    if (trace) {
        fprintf(stderr, "NVFP4_TUNE (CUDA score-spread) min_obj=%g min_lin=%d max_obj=%g max_lin=%d\n",
            min_obj, min_lin, max_obj, max_lin);
    }

    const int   base_lin = NVFP4_TUNE_GRID_CENTER * NVFP4_TUNE_GRID_SIZE + NVFP4_TUNE_GRID_CENTER;
    const float base_obj = h_results_pinned[base_lin];

    // Grid-selection guard parameters.
    const int   EDGE_SKIP   = NVFP4_TUNE_GRID_EDGE_SKIP; // skip outer ring
    const float MIN_IMPROVE = NVFP4_TUNE_MIN_IMPROVE_FRAC;

    float best_obj = 1e30f;
    int   best_i   = base_lin;

    for (int i = 0; i < n_scores; ++i) {
        const int bi_i = i / NVFP4_TUNE_GRID_SIZE;
        const int ai_i = i - bi_i * NVFP4_TUNE_GRID_SIZE;
        const int ai   = ai_i - NVFP4_TUNE_GRID_CENTER;
        const int bi   = bi_i - NVFP4_TUNE_GRID_CENTER;

        // Rule A: avoid boundary solutions (often unstable / overfit)
        if (abs(ai) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP) {
            continue;
        }
        if (abs(bi) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP) {
            continue;
        }

        const float obj = h_results_pinned[i];
        if (obj < best_obj) {
            best_obj = obj;
            best_i   = i;
        }
    }

    // Rule B: only accept if materially better than baseline
    if (!(best_obj < base_obj * (1.0f - MIN_IMPROVE))) {
        best_i   = base_lin;
        best_obj = base_obj;
    }

    best_lin = best_i;

    // Guard stage:
    // 1) keep a larger top-K candidate pool by primary objective,
    // 2) validate each candidate on a held-out slice,
    // 3) reject outliers before falling back.
    {
        const size_t want = (size_t) sample_nb;
        if (cap_metrics < want || !d_sse || !d_sum_err || !d_sum_w || !d_sum_x2) {
            if (d_sse)     { cudaFree(d_sse);     d_sse     = nullptr; }
            if (d_sum_err) { cudaFree(d_sum_err); d_sum_err = nullptr; }
            if (d_sum_w)   { cudaFree(d_sum_w);   d_sum_w   = nullptr; }
            if (d_sum_x2)  { cudaFree(d_sum_x2);  d_sum_x2  = nullptr; }

            CUDA_CHECK(cudaMalloc(&d_sse,     want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_err, want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_w,   want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_x2,  want * sizeof(float)));
            cap_metrics = want;
        }
    }
    {
        const size_t want = (size_t) sample_nb;
        const size_t bytes = want * sizeof(float);
        if (cap_h_metric_vec < want || !h_sse_pinned || !h_sum_err_pinned || !h_sum_w_pinned || !h_sum_x2_pinned) {
            if (h_sse_pinned)     CUDA_CHECK(cudaFreeHost(h_sse_pinned));
            if (h_sum_err_pinned) CUDA_CHECK(cudaFreeHost(h_sum_err_pinned));
            if (h_sum_w_pinned)   CUDA_CHECK(cudaFreeHost(h_sum_w_pinned));
            if (h_sum_x2_pinned)  CUDA_CHECK(cudaFreeHost(h_sum_x2_pinned));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sse_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_err_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_w_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_x2_pinned, bytes, cudaHostAllocDefault));
            cap_h_metric_vec = want;
        }
    }

    const bool run_robust = !score_flat;
    const int guard_topk = run_robust ? std::min(NVFP4_TUNE_GUARD_TOPK, n_scores) : 0;
    int cand_lin[NVFP4_TUNE_GUARD_TOPK];
    float cand_obj[NVFP4_TUNE_GUARD_TOPK];
    for (int i = 0; i < NVFP4_TUNE_GUARD_TOPK; ++i) {
        cand_lin[i] = -1;
        cand_obj[i] = 1e30f;
    }

    auto insert_topk = [&](int lin, float obj) {
        int pos = -1;
        for (int i = 0; i < guard_topk; ++i) {
            if (obj < cand_obj[i]) {
                pos = i;
                break;
            }
        }
        if (pos < 0) {
            return;
        }
        for (int j = guard_topk - 1; j > pos; --j) {
            cand_obj[j] = cand_obj[j - 1];
            cand_lin[j] = cand_lin[j - 1];
        }
        cand_obj[pos] = obj;
        cand_lin[pos] = lin;
    };

    for (int i = 0; i < n_scores; ++i) {
        const int bi_i = i / NVFP4_TUNE_GRID_SIZE;
        const int ai_i = i - bi_i * NVFP4_TUNE_GRID_SIZE;
        const int ai   = ai_i - NVFP4_TUNE_GRID_CENTER;
        const int bi   = bi_i - NVFP4_TUNE_GRID_CENTER;
        if (abs(ai) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP || abs(bi) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP) {
            continue;
        }
        insert_topk(i, h_results_pinned[i]);
    }

    const int base_lin_guard = NVFP4_TUNE_GRID_CENTER * NVFP4_TUNE_GRID_SIZE + NVFP4_TUNE_GRID_CENTER;
    insert_topk(base_lin_guard, h_results_pinned[base_lin_guard]);
    bool base_present = false;
    for (int i = 0; i < guard_topk; ++i) {
        if (cand_lin[i] == base_lin_guard) {
            base_present = true;
            break;
        }
    }
    if (!base_present) {
        cand_lin[guard_topk - 1] = base_lin_guard;
        cand_obj[guard_topk - 1] = h_results_pinned[base_lin_guard];
    }

    const int val_nb = (int) sample_nb;
    struct tune_eval_stats {
        double max_obj = 1e30;
        double max_rel_obj = 1e30;
        double mean_obj = 1e30;
        double obj_norm = 1e30;
        double rel_rmse = 1e30;
        double abs_mean_err = 1e30;
        double abs_mean_err_rel = 1e30;
        double avg_obj = 1e30;
        double p95_rel_obj = 1e30;
        double tail_rel_obj = 1e30;
        double sum_sse = 0.0;
        double sum_err = 0.0;
        double sum_w = 0.0;
        double sum_x2 = 0.0;
        double rmse = 1e30;
        double x_rms = 1e30;
        int valid_blocks = 0;
    };

    auto p95_inplace = [](double * vals, int n) -> double {
        if (n <= 0) {
            return 1e30;
        }
        const int idx = (95 * (n - 1)) / 100;
        std::nth_element(vals, vals + idx, vals + n);
        return vals[idx];
    };

    auto reduce_candidate_metrics = [&](const float * sse_h, const float * sum_err_h, const float * sum_w_h, const float * sum_x2_h, int eval_nb, tune_eval_stats & out) {
        out = tune_eval_stats{};
        if (eval_nb <= 0) {
            return;
        }

        double rel_obj_vals[NVFP4_TUNE_MAX_SAMPLE_BLOCKS];
        int rel_count = 0;
        double max_obj = 0.0;
        double max_rel_obj = 0.0;
        double sum_obj = 0.0;
        double sum_rel_obj2 = 0.0;
        double sum_err_all = 0.0;
        double sum_w_all = 0.0;
        double sum_x2_all = 0.0;
        int cnt = 0;

        for (int i = 0; i < eval_nb; ++i) {
            const double sw  = (double) sum_w_h[i];
            const double sse = (double) sse_h[i];
            const double se  = (double) sum_err_h[i];
            const double sx2 = (double) sum_x2_h[i];
            if (!(sw > 0.0) || !(sx2 > 0.0) || !std::isfinite(sw) || !std::isfinite(sse) || !std::isfinite(se) || !std::isfinite(sx2)) {
                continue;
            }

            const double obj = sse;
            const double rel_obj = sse / sx2;
            if (!std::isfinite(obj) || !std::isfinite(rel_obj) || obj < 0.0 || rel_obj < 0.0) {
                continue;
            }

            sum_obj += obj;
            sum_rel_obj2 += rel_obj * rel_obj;
            if (rel_count < NVFP4_TUNE_MAX_SAMPLE_BLOCKS) {
                rel_obj_vals[rel_count++] = rel_obj;
            }
            cnt += 1;
            if (obj > max_obj) {
                max_obj = obj;
            }
            if (rel_obj > max_rel_obj) {
                max_rel_obj = rel_obj;
            }
            sum_err_all += se;
            sum_w_all += sw;
            sum_x2_all += sx2;
        }

        if (cnt > 0 && sum_w_all > 0.0 && sum_x2_all > 0.0 && rel_count > 0) {
            const double mean_err = sum_err_all / sum_w_all;
            const double rmse = sqrt(sum_obj / sum_w_all);
            const double x_rms = sqrt(sum_x2_all / sum_w_all);
            out.max_obj = max_obj;
            out.max_rel_obj = max_rel_obj;
            out.mean_obj = sum_obj / (double) cnt;
            out.obj_norm = sum_obj / sum_x2_all;
            out.rel_rmse = (x_rms > 0.0 && std::isfinite(rmse)) ? (rmse / x_rms) : 1e30;
            out.abs_mean_err = std::isfinite(mean_err) ? fabs(mean_err) : 1e30;
            out.abs_mean_err_rel = (x_rms > 0.0 && std::isfinite(mean_err)) ? (fabs(mean_err) / x_rms) : 1e30;
            out.avg_obj = out.mean_obj;
            out.p95_rel_obj = p95_inplace(rel_obj_vals, rel_count);
            out.tail_rel_obj = sqrt(sum_rel_obj2 / (double) cnt);
            out.sum_sse = sum_obj;
            out.sum_err = sum_err_all;
            out.sum_w = sum_w_all;
            out.sum_x2 = sum_x2_all;
            out.rmse = rmse;
            out.x_rms = x_rms;
            out.valid_blocks = cnt;
        }
    };

    auto ensure_metric_batch_capacity = [&](size_t want_total) {
        if (cap_metrics_batch < want_total || !d_sse_batch || !d_sum_err_batch || !d_sum_w_batch || !d_sum_x2_batch) {
            if (d_sse_batch)     { cudaFree(d_sse_batch);     d_sse_batch = nullptr; }
            if (d_sum_err_batch) { cudaFree(d_sum_err_batch); d_sum_err_batch = nullptr; }
            if (d_sum_w_batch)   { cudaFree(d_sum_w_batch);   d_sum_w_batch = nullptr; }
            if (d_sum_x2_batch)  { cudaFree(d_sum_x2_batch);  d_sum_x2_batch = nullptr; }

            CUDA_CHECK(cudaMalloc(&d_sse_batch,     want_total * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_err_batch, want_total * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_w_batch,   want_total * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_x2_batch,  want_total * sizeof(float)));
            cap_metrics_batch = want_total;
        }

        if (cap_h_metric_batch < want_total || !h_sse_batch_pinned || !h_sum_err_batch_pinned || !h_sum_w_batch_pinned || !h_sum_x2_batch_pinned) {
            const size_t bytes = want_total * sizeof(float);
            if (h_sse_batch_pinned)     CUDA_CHECK(cudaFreeHost(h_sse_batch_pinned));
            if (h_sum_err_batch_pinned) CUDA_CHECK(cudaFreeHost(h_sum_err_batch_pinned));
            if (h_sum_w_batch_pinned)   CUDA_CHECK(cudaFreeHost(h_sum_w_batch_pinned));
            if (h_sum_x2_batch_pinned)  CUDA_CHECK(cudaFreeHost(h_sum_x2_batch_pinned));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sse_batch_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_err_batch_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_w_batch_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_x2_batch_pinned, bytes, cudaHostAllocDefault));
            cap_h_metric_batch = want_total;
        }
    };

    auto eval_candidates_batch = [&](const float * x_eval, const float * qw_eval, int eval_nb, const float * a_vals, const float * b_vals, int n_cands, const tune_policy_cfg & policy, tune_eval_stats * out_stats) {
        for (int i = 0; i < n_cands; ++i) {
            out_stats[i] = tune_eval_stats{};
        }
        if (eval_nb <= 0 || n_cands <= 0) {
            return;
        }

        const size_t total = (size_t) eval_nb * (size_t) n_cands;
        ensure_metric_batch_capacity(total);
        for (int ci = 0; ci < n_cands; ++ci) {
            const size_t off = (size_t) ci * (size_t) eval_nb;
            metrics<<<eval_nb, 256, 0, st>>>(
                x_eval, qw_eval, eval_nb, a_vals[ci], b_vals[ci],
                policy.choose46_mode, policy.refit_iters, policy.use_compand_sat,
                policy.cap_m6, policy.cap_m4,
                d_sse_batch + off, d_sum_err_batch + off, d_sum_w_batch + off, d_sum_x2_batch + off);
            CUDA_CHECK(cudaGetLastError());
        }

        const size_t bytes = total * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(h_sse_batch_pinned, d_sse_batch, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_err_batch_pinned, d_sum_err_batch, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_w_batch_pinned, d_sum_w_batch, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_x2_batch_pinned, d_sum_x2_batch, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        for (int ci = 0; ci < n_cands; ++ci) {
            const size_t off = (size_t) ci * (size_t) eval_nb;
            reduce_candidate_metrics(
                h_sse_batch_pinned + off,
                h_sum_err_batch_pinned + off,
                h_sum_w_batch_pinned + off,
                h_sum_x2_batch_pinned + off,
                eval_nb,
                out_stats[ci]);
        }
    };

    auto eval_candidate = [&](const float * x_eval, const float * qw_eval, int eval_nb, float a, float b, const tune_policy_cfg & policy, tune_eval_stats & out) {
        out = tune_eval_stats{};
        if (eval_nb <= 0) {
            return;
        }

        metrics<<<eval_nb, 256, 0, st>>>(
            x_eval, qw_eval, eval_nb, a, b,
            policy.choose46_mode, policy.refit_iters, policy.use_compand_sat,
            policy.cap_m6, policy.cap_m4,
            d_sse, d_sum_err, d_sum_w, d_sum_x2);
        CUDA_CHECK(cudaGetLastError());

        const size_t bytes = (size_t) eval_nb * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(h_sse_pinned, d_sse, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_err_pinned, d_sum_err, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_w_pinned, d_sum_w, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_sum_x2_pinned, d_sum_x2, bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        reduce_candidate_metrics(h_sse_pinned, h_sum_err_pinned, h_sum_w_pinned, h_sum_x2_pinned, eval_nb, out);
    };

    tune_eval_stats base_eval_guard;
    eval_candidate(d_x_buf, d_qw_use, val_nb, NVFP4_A0, NVFP4_B0, active_policy, base_eval_guard);
    const double base_max_obj = base_eval_guard.max_obj;
    const double base_max_rel_obj = base_eval_guard.max_rel_obj;
    const double base_abs_mean_err_rel = base_eval_guard.abs_mean_err_rel;
    const double base_tail_rel_obj = base_eval_guard.tail_rel_obj;
    const double base_obj_norm_guard = base_eval_guard.obj_norm;
    const double base_p95_rel_obj_guard = base_eval_guard.p95_rel_obj;

    double guard_max_obj =
        (std::isfinite(base_max_obj) && base_max_obj > 0.0)
            ? (base_max_obj * (double) NVFP4_TUNE_GUARD_MAX_OBJ_FRAC)
            : 1e30;
    double guard_max_rel_obj =
        (std::isfinite(base_max_rel_obj) && base_max_rel_obj > 0.0)
            ? (base_max_rel_obj * (double) NVFP4_TUNE_GUARD_MAX_REL_OBJ_FRAC)
            : 1e30;
    double guard_tail_rel_obj =
        (std::isfinite(base_tail_rel_obj) && base_tail_rel_obj > 0.0)
            ? (base_tail_rel_obj * (double) NVFP4_TUNE_GUARD_TAIL_REL_OBJ_FRAC)
            : 1e30;
    if (std::isfinite(base_max_obj) && base_max_obj > 0.0) {
        guard_max_obj = fmax(guard_max_obj, base_max_obj);
    }
    if (std::isfinite(base_max_rel_obj) && base_max_rel_obj > 0.0) {
        guard_max_rel_obj = fmax(guard_max_rel_obj, base_max_rel_obj);
    }
    if (std::isfinite(base_tail_rel_obj) && base_tail_rel_obj > 0.0) {
        guard_tail_rel_obj = fmax(guard_tail_rel_obj, base_tail_rel_obj);
    }
    double guard_abs_mean_err_rel = 1e30;
    if (std::isfinite(base_abs_mean_err_rel) && base_abs_mean_err_rel > 0.0) {
        double g = fmax(base_abs_mean_err_rel * 1.25, base_abs_mean_err_rel + 1e-4);
        if (NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL > 0.0f) {
            const double cap = (double) NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL;
            g = fmin(g, fmax(cap, base_abs_mean_err_rel));
        }
        guard_abs_mean_err_rel = g;
    } else if (NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL > 0.0f) {
        guard_abs_mean_err_rel = (double) NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL;
    }

    const bool guard_risky =
        !std::isfinite(base_eval_guard.obj_norm) || !std::isfinite(base_eval_guard.p95_rel_obj) ||
        !std::isfinite(base_eval_guard.max_rel_obj) || !std::isfinite(base_eval_guard.tail_rel_obj) ||
        base_eval_guard.obj_norm >= NVFP4_TUNE_ADAPT_OBJ_NORM_MID ||
        base_eval_guard.p95_rel_obj >= NVFP4_TUNE_ADAPT_P95_REL_OBJ_MID ||
        base_eval_guard.max_rel_obj >= NVFP4_TUNE_ADAPT_MAX_REL_OBJ_MID ||
        base_eval_guard.tail_rel_obj >= NVFP4_TUNE_ADAPT_TAIL_REL_OBJ_MID;
    // Evaluate the full guard candidate set when the baseline is risky.
    const int guard_eval_cap = guard_risky ? guard_topk : std::min(guard_topk, NVFP4_TUNE_GUARD_TOPK_FAST);

    int best2_lin_pass = base_lin;
    int best2_rank_pass = -1;
    double best2_max_obj_pass = base_obj_norm_guard;
    double best2_max_rel_obj_pass = base_p95_rel_obj_guard;
    double best2_tail_rel_obj_pass = base_tail_rel_obj;
    double best2_abs_mean_err_rel_pass = base_abs_mean_err_rel;
    double best2_score_pass = 1e30;

    int best2_lin = base_lin;
    int best2_rank = -1;
    double best2_max_obj = base_obj_norm_guard;
    double best2_max_rel_obj = base_p95_rel_obj_guard;
    double best2_tail_rel_obj = base_tail_rel_obj;
    double best2_abs_mean_err_rel = base_abs_mean_err_rel;
    bool guard_fallback = false;

    int guard_eval_lin[NVFP4_TUNE_GUARD_TOPK];
    float guard_eval_a[NVFP4_TUNE_GUARD_TOPK];
    float guard_eval_b[NVFP4_TUNE_GUARD_TOPK];
    int guard_eval_n = 0;
    bool guard_eval_has_base = false;
    for (int ci = 0; ci < guard_topk; ++ci) {
        const int lin = cand_lin[ci];
        if (lin < 0 || guard_eval_n >= guard_eval_cap) {
            continue;
        }
        guard_eval_has_base = guard_eval_has_base || (lin == base_lin_guard);
        const int bi_i = lin / NVFP4_TUNE_GRID_SIZE;
        const int ai_i = lin - bi_i * NVFP4_TUNE_GRID_SIZE;
        const int ai = ai_i - NVFP4_TUNE_GRID_CENTER;
        const int bi = bi_i - NVFP4_TUNE_GRID_CENTER;
        guard_eval_lin[guard_eval_n] = lin;
        guard_eval_a[guard_eval_n] = NVFP4_A0 + (float) ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        guard_eval_b[guard_eval_n] = NVFP4_B0 + (float) bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        ++guard_eval_n;
    }
    if (!guard_eval_has_base && guard_eval_cap > 0) {
        const int lin = base_lin_guard;
        const int bi_i = lin / NVFP4_TUNE_GRID_SIZE;
        const int ai_i = lin - bi_i * NVFP4_TUNE_GRID_SIZE;
        const int ai = ai_i - NVFP4_TUNE_GRID_CENTER;
        const int bi = bi_i - NVFP4_TUNE_GRID_CENTER;
        if (guard_eval_n < guard_eval_cap) {
            guard_eval_lin[guard_eval_n] = lin;
            guard_eval_a[guard_eval_n] = NVFP4_A0 + (float) ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
            guard_eval_b[guard_eval_n] = NVFP4_B0 + (float) bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
            ++guard_eval_n;
        } else if (guard_eval_n > 0) {
            const int last = guard_eval_n - 1;
            guard_eval_lin[last] = lin;
            guard_eval_a[last] = NVFP4_A0 + (float) ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
            guard_eval_b[last] = NVFP4_B0 + (float) bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        }
    }
    tune_eval_stats guard_eval_stats[NVFP4_TUNE_GUARD_TOPK];
    eval_candidates_batch(d_x_buf, d_qw_use, val_nb, guard_eval_a, guard_eval_b, guard_eval_n, active_policy, guard_eval_stats);

    for (int gi = 0; gi < guard_eval_n; ++gi) {
        const int lin = guard_eval_lin[gi];
        const tune_eval_stats & cand_eval = guard_eval_stats[gi];

        if (!std::isfinite(cand_eval.max_obj) ||
            !std::isfinite(cand_eval.max_rel_obj) ||
            !std::isfinite(cand_eval.obj_norm) ||
            !std::isfinite(cand_eval.abs_mean_err_rel) ||
            !std::isfinite(cand_eval.p95_rel_obj) ||
            !std::isfinite(cand_eval.tail_rel_obj)) {
            continue;
        }
        const double score =
            (double) NVFP4_TUNE_SCORE_W_OBJ_NORM * cand_eval.obj_norm +
            (double) NVFP4_TUNE_SCORE_W_P95_REL_OBJ * cand_eval.p95_rel_obj +
            (double) NVFP4_TUNE_SCORE_W_TAIL_REL_OBJ * cand_eval.tail_rel_obj +
            (double) NVFP4_TUNE_SCORE_W_MAX_REL_OBJ * cand_eval.max_rel_obj +
            (double) NVFP4_TUNE_SCORE_W_ABS_MEAN_ERR_REL * cand_eval.abs_mean_err_rel;

        const bool pass =
            cand_eval.max_obj <= guard_max_obj &&
            cand_eval.max_rel_obj <= guard_max_rel_obj &&
            cand_eval.tail_rel_obj <= guard_tail_rel_obj &&
            cand_eval.abs_mean_err_rel <= guard_abs_mean_err_rel;
        if (pass && (score < best2_score_pass ||
            (score == best2_score_pass && cand_eval.obj_norm < best2_max_obj_pass) ||
            (score == best2_score_pass && cand_eval.obj_norm == best2_max_obj_pass && cand_eval.p95_rel_obj < best2_max_rel_obj_pass) ||
            (score == best2_score_pass && cand_eval.obj_norm == best2_max_obj_pass && cand_eval.p95_rel_obj == best2_max_rel_obj_pass && cand_eval.tail_rel_obj < best2_tail_rel_obj_pass) ||
            (score == best2_score_pass && cand_eval.obj_norm == best2_max_obj_pass && cand_eval.p95_rel_obj == best2_max_rel_obj_pass && cand_eval.tail_rel_obj == best2_tail_rel_obj_pass && cand_eval.abs_mean_err_rel < best2_abs_mean_err_rel_pass))) {
            best2_lin_pass = lin;
            best2_rank_pass = gi;
            best2_max_obj_pass = cand_eval.obj_norm;
            best2_max_rel_obj_pass = cand_eval.p95_rel_obj;
            best2_tail_rel_obj_pass = cand_eval.tail_rel_obj;
            best2_abs_mean_err_rel_pass = cand_eval.abs_mean_err_rel;
            best2_score_pass = score;
        }
    }

    if (best2_rank_pass >= 0) {
        best2_lin = best2_lin_pass;
        best2_rank = best2_rank_pass;
        best2_max_obj = best2_max_obj_pass;
        best2_max_rel_obj = best2_max_rel_obj_pass;
        best2_tail_rel_obj = best2_tail_rel_obj_pass;
        best2_abs_mean_err_rel = best2_abs_mean_err_rel_pass;
        guard_fallback = false;
    } else {
        best2_lin = base_lin;
        best2_rank = -1;
        best2_max_obj = base_obj_norm_guard;
        best2_max_rel_obj = base_p95_rel_obj_guard;
        best2_tail_rel_obj = base_tail_rel_obj;
        best2_abs_mean_err_rel = base_abs_mean_err_rel;
        guard_fallback = true;
    }

    best_lin = best2_lin;
    const int best_bi_i0 = best_lin / NVFP4_TUNE_GRID_SIZE;
    const int best_ai_i0 = best_lin - best_bi_i0 * NVFP4_TUNE_GRID_SIZE;
    const int best_ai0 = best_ai_i0 - NVFP4_TUNE_GRID_CENTER;
    const int best_bi0 = best_bi_i0 - NVFP4_TUNE_GRID_CENTER;
    float best_a_now = NVFP4_A0 + (float) best_ai0 * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
    float best_b_now = NVFP4_B0 + (float) best_bi0 * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);

    bool cap_compare_enabled = false;
    int cap_selected_nb = -1;
    int cap_selected_lin = -1;
    double cap_selected_obj_norm = 1e30;
    double cap_selected_rel_rmse = 1e30;
    double cap_selected_tail_rel_obj = 1e30;
    double cap_selected_p95_rel_obj = 1e30;
    double cap_selected_max_rel_obj = 1e30;
    double cap_selected_abs_mean_err = 1e30;
    double cap_selected_abs_mean_err_rel = 1e30;
    double cap_selected_avg_obj = 1e30;
    double cap_selected_agg_score = 1e30;
    int cap_kept_count = 0;

    const double base_obj_norm = base_eval_guard.obj_norm;
    const double base_rel_rmse = base_eval_guard.rel_rmse;
    const double base_p95_rel_obj = base_eval_guard.p95_rel_obj;
    const double base_max_rel_obj_cap = base_eval_guard.max_rel_obj;
    const double base_tail_rel_obj_cap = base_eval_guard.tail_rel_obj;

    const double cap_gate_obj_norm =
        (std::isfinite(base_obj_norm) && base_obj_norm > 0.0)
            ? (base_obj_norm * (double) NVFP4_TUNE_CAP_OBJ_NORM_FRAC)
            : 1e30;
    double cap_gate_rel_rmse =
        (std::isfinite(base_rel_rmse) && base_rel_rmse > 0.0)
            ? fmin(base_rel_rmse * (double) NVFP4_TUNE_CAP_REL_RMSE_FRAC, (double) NVFP4_TUNE_CAP_REL_RMSE_ABS)
            : (double) NVFP4_TUNE_CAP_REL_RMSE_ABS;
    if (std::isfinite(base_rel_rmse) && base_rel_rmse > 0.0) {
        cap_gate_rel_rmse = fmax(cap_gate_rel_rmse, base_rel_rmse);
    }
    double cap_gate_p95_rel_obj =
        (std::isfinite(base_p95_rel_obj) && base_p95_rel_obj > 0.0)
            ? fmax(base_p95_rel_obj, base_p95_rel_obj * (double) NVFP4_TUNE_CAP_P95_REL_OBJ_FRAC)
            : 1e30;
    double cap_gate_max_rel_obj =
        (std::isfinite(base_max_rel_obj_cap) && base_max_rel_obj_cap > 0.0)
            ? fmax(base_max_rel_obj_cap, base_max_rel_obj_cap * (double) NVFP4_TUNE_CAP_MAX_REL_OBJ_FRAC)
            : 1e30;
    double cap_gate_tail_rel_obj =
        (std::isfinite(base_tail_rel_obj_cap) && base_tail_rel_obj_cap > 0.0)
            ? fmax(base_tail_rel_obj_cap, base_tail_rel_obj_cap * (double) NVFP4_TUNE_CAP_TAIL_REL_OBJ_FRAC)
            : 1e30;
    double cap_gate_abs_mean_err_rel = 1e30;
    if (NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL > 0.0f) {
        cap_gate_abs_mean_err_rel = (double) NVFP4_TUNE_CAP_ABS_MEAN_ERR_REL;
    }
    if (std::isfinite(base_abs_mean_err_rel) && base_abs_mean_err_rel > 0.0) {
        cap_gate_abs_mean_err_rel = fmax(cap_gate_abs_mean_err_rel, base_abs_mean_err_rel);
    }

    const bool run_cap_compare = run_robust && guard_risky;

    constexpr int k_cap_compare_n = (int) (sizeof(NVFP4_TUNE_CAP_COMPARE_NBS) / sizeof(NVFP4_TUNE_CAP_COMPARE_NBS[0]));
    int cap_try_n = 0;
    int cap_try_nb[k_cap_compare_n];
    int cap_try_lin[k_cap_compare_n];
    float cap_try_a[k_cap_compare_n];
    float cap_try_b[k_cap_compare_n];

    for (int ci = 0; run_cap_compare && ci < k_cap_compare_n; ++ci) {
        const int cap_nb = NVFP4_TUNE_CAP_COMPARE_NBS[ci];
        if (sample_nb < cap_nb) {
            continue;
        }
        cap_compare_enabled = true;

        const size_t bytes_cap = (size_t) cap_nb * 256 * sizeof(float);

        if (!d_x_low_buf || cap_x_low < bytes_cap) {
            if (d_x_low_buf) cudaFree(d_x_low_buf);
            CUDA_CHECK(cudaMalloc(&d_x_low_buf, bytes_cap));
            cap_x_low = bytes_cap;
        }

        downsample_device_blocks(d_x_buf, d_x_low_buf, sample_nb, cap_nb);

        float * d_qw_cap_use = nullptr;
        if (d_qw_use) {
            if (!d_qw_low_buf || cap_qw_low < bytes_cap) {
                if (d_qw_low_buf) cudaFree(d_qw_low_buf);
                CUDA_CHECK(cudaMalloc(&d_qw_low_buf, bytes_cap));
                cap_qw_low = bytes_cap;
            }

            downsample_device_blocks(d_qw_use, d_qw_low_buf, sample_nb, cap_nb);
            d_qw_cap_use = d_qw_low_buf;
        }

        autotune<<<blocks, threads, 0, st>>>(
            d_x_low_buf, d_qw_cap_use, cap_nb, d_results,
            active_policy.choose46_mode, active_policy.refit_iters, active_policy.use_compand_sat,
            active_policy.cap_m6, active_policy.cap_m4);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_results_pinned, d_results, bytes_scores, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        int cap_lin = base_lin;
        float cap_best_obj = 1e30f;
        for (int i = 0; i < n_scores; ++i) {
            const int bi_i = i / NVFP4_TUNE_GRID_SIZE;
            const int ai_i = i - bi_i * NVFP4_TUNE_GRID_SIZE;
            const int ai   = ai_i - NVFP4_TUNE_GRID_CENTER;
            const int bi   = bi_i - NVFP4_TUNE_GRID_CENTER;
            if (abs(ai) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP || abs(bi) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP) {
                continue;
            }
            const float v = h_results_pinned[i];
            if (v < cap_best_obj) {
                cap_best_obj = v;
                cap_lin = i;
            }
        }

        const int cap_bi_i = cap_lin / NVFP4_TUNE_GRID_SIZE;
        const int cap_ai_i = cap_lin - cap_bi_i * NVFP4_TUNE_GRID_SIZE;
        const int cap_ai = cap_ai_i - NVFP4_TUNE_GRID_CENTER;
        const int cap_bi = cap_bi_i - NVFP4_TUNE_GRID_CENTER;
        const float cap_a = NVFP4_A0 + (float) cap_ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        const float cap_b = NVFP4_B0 + (float) cap_bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);

        if (cap_try_n < k_cap_compare_n) {
            cap_try_nb[cap_try_n] = cap_nb;
            cap_try_lin[cap_try_n] = cap_lin;
            cap_try_a[cap_try_n] = cap_a;
            cap_try_b[cap_try_n] = cap_b;
            ++cap_try_n;
        }
    }

    if (cap_try_n > 0) {
        tune_eval_stats cap_try_eval[k_cap_compare_n];
        eval_candidates_batch(d_x_buf, d_qw_use, (int) sample_nb, cap_try_a, cap_try_b, cap_try_n, active_policy, cap_try_eval);

        for (int ti = 0; ti < cap_try_n; ++ti) {
            const int cap_nb = cap_try_nb[ti];
            const int cap_lin = cap_try_lin[ti];
            const float cap_a = cap_try_a[ti];
            const float cap_b = cap_try_b[ti];
            const tune_eval_stats & cap_eval = cap_try_eval[ti];

            const double cap_eval_obj_norm = cap_eval.obj_norm;
            const double cap_eval_rel_rmse = cap_eval.rel_rmse;
            const double cap_eval_tail_rel_obj = cap_eval.tail_rel_obj;
            const double cap_eval_p95_rel_obj = cap_eval.p95_rel_obj;
            const double cap_eval_max_rel_obj = cap_eval.max_rel_obj;
            const double cap_eval_abs_mean_err = cap_eval.abs_mean_err;
            const double cap_eval_avg_obj = cap_eval.avg_obj;
            const double cap_eval_abs_mean_err_rel = cap_eval.abs_mean_err_rel;

            const bool cap_pass =
                std::isfinite(cap_eval.max_obj) && std::isfinite(cap_eval.max_rel_obj) &&
                cap_eval.max_obj <= guard_max_obj &&
                cap_eval.max_rel_obj <= guard_max_rel_obj &&
                cap_eval.tail_rel_obj <= guard_tail_rel_obj &&
                std::isfinite(cap_eval_obj_norm) && std::isfinite(cap_eval_rel_rmse) &&
                std::isfinite(cap_eval_abs_mean_err) && std::isfinite(cap_eval_abs_mean_err_rel) &&
                std::isfinite(cap_eval_tail_rel_obj) && std::isfinite(cap_eval_p95_rel_obj) && std::isfinite(cap_eval_max_rel_obj) &&
                cap_eval_obj_norm <= cap_gate_obj_norm &&
                cap_eval_rel_rmse <= cap_gate_rel_rmse &&
                cap_eval_tail_rel_obj <= cap_gate_tail_rel_obj &&
                cap_eval_p95_rel_obj <= cap_gate_p95_rel_obj &&
                cap_eval_max_rel_obj <= cap_gate_max_rel_obj &&
                cap_eval_abs_mean_err_rel <= cap_gate_abs_mean_err_rel;

            bool cap_better = false;
            double cap_agg_score = 1e30;
            if (cap_pass) {
                const double obj_ratio =
                    (std::isfinite(base_obj_norm) && base_obj_norm > 0.0)
                        ? (cap_eval_obj_norm / base_obj_norm)
                        : cap_eval_obj_norm;
                const double rel_rmse_ratio =
                    (std::isfinite(base_rel_rmse) && base_rel_rmse > 0.0)
                        ? (cap_eval_rel_rmse / base_rel_rmse)
                        : cap_eval_rel_rmse;
                const double abs_mean_err_rel_ratio =
                    (std::isfinite(base_abs_mean_err_rel) && base_abs_mean_err_rel > 0.0)
                        ? (cap_eval_abs_mean_err_rel / base_abs_mean_err_rel)
                        : cap_eval_abs_mean_err_rel;
                const double p95_rel_obj_ratio =
                    (std::isfinite(base_p95_rel_obj) && base_p95_rel_obj > 0.0)
                        ? (cap_eval_p95_rel_obj / base_p95_rel_obj)
                        : cap_eval_p95_rel_obj;
                const double tail_rel_obj_ratio =
                    (std::isfinite(base_tail_rel_obj_cap) && base_tail_rel_obj_cap > 0.0)
                        ? (cap_eval_tail_rel_obj / base_tail_rel_obj_cap)
                        : cap_eval_tail_rel_obj;
                const double max_rel_obj_ratio =
                    (std::isfinite(base_max_rel_obj_cap) && base_max_rel_obj_cap > 0.0)
                        ? (cap_eval_max_rel_obj / base_max_rel_obj_cap)
                        : cap_eval_max_rel_obj;
                cap_agg_score =
                    (double) NVFP4_TUNE_SCORE_W_OBJ_NORM * obj_ratio +
                    (double) NVFP4_TUNE_SCORE_W_P95_REL_OBJ * p95_rel_obj_ratio +
                    (double) NVFP4_TUNE_SCORE_W_TAIL_REL_OBJ * tail_rel_obj_ratio +
                    (double) NVFP4_TUNE_SCORE_W_MAX_REL_OBJ * max_rel_obj_ratio +
                    (double) NVFP4_TUNE_SCORE_W_ABS_MEAN_ERR_REL * abs_mean_err_rel_ratio;
                if (cap_selected_nb < 0) {
                    cap_better = true;
                } else if (cap_agg_score < cap_selected_agg_score) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm < cap_selected_obj_norm) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse < cap_selected_rel_rmse) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse == cap_selected_rel_rmse &&
                           cap_eval_tail_rel_obj < cap_selected_tail_rel_obj) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse == cap_selected_rel_rmse &&
                           cap_eval_tail_rel_obj == cap_selected_tail_rel_obj &&
                           cap_eval_p95_rel_obj < cap_selected_p95_rel_obj) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse == cap_selected_rel_rmse &&
                           cap_eval_tail_rel_obj == cap_selected_tail_rel_obj &&
                           cap_eval_p95_rel_obj == cap_selected_p95_rel_obj &&
                           cap_eval_max_rel_obj < cap_selected_max_rel_obj) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse == cap_selected_rel_rmse &&
                           cap_eval_tail_rel_obj == cap_selected_tail_rel_obj &&
                           cap_eval_p95_rel_obj == cap_selected_p95_rel_obj &&
                           cap_eval_max_rel_obj == cap_selected_max_rel_obj &&
                           cap_eval_abs_mean_err_rel < cap_selected_abs_mean_err_rel) {
                    cap_better = true;
                } else if (cap_agg_score == cap_selected_agg_score &&
                           cap_eval_obj_norm == cap_selected_obj_norm &&
                           cap_eval_rel_rmse == cap_selected_rel_rmse &&
                           cap_eval_tail_rel_obj == cap_selected_tail_rel_obj &&
                           cap_eval_p95_rel_obj == cap_selected_p95_rel_obj &&
                           cap_eval_max_rel_obj == cap_selected_max_rel_obj &&
                           cap_eval_abs_mean_err_rel == cap_selected_abs_mean_err_rel &&
                           cap_nb < cap_selected_nb) {
                    // Prefer smaller sample cap only on exact aggregate ties.
                    cap_better = true;
                }
            }

            if (cap_better) {
                cap_selected_nb = cap_nb;
                cap_selected_lin = cap_lin;
                cap_selected_obj_norm = cap_eval_obj_norm;
                cap_selected_rel_rmse = cap_eval_rel_rmse;
                cap_selected_tail_rel_obj = cap_eval_tail_rel_obj;
                cap_selected_p95_rel_obj = cap_eval_p95_rel_obj;
                cap_selected_max_rel_obj = cap_eval_max_rel_obj;
                cap_selected_abs_mean_err = cap_eval_abs_mean_err;
                cap_selected_abs_mean_err_rel = cap_eval_abs_mean_err_rel;
                cap_selected_avg_obj = cap_eval_avg_obj;
                cap_selected_agg_score = cap_agg_score;
                best_lin = cap_lin;
                best_a_now = cap_a;
                best_b_now = cap_b;
                cap_kept_count += 1;
            }

            if (trace) {
                fprintf(stderr,
                    "NVFP4_TUNE (CUDA cap-compare step) nb=%d obj_norm=%g rel_rmse=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g abs_mean_err=%g abs_mean_err_rel=%g avg_obj=%g val_obj_max=%g val_rel_obj_max=%g pass=%d lin=%d better=%d\n",
                    cap_nb, cap_eval_obj_norm, cap_eval_rel_rmse, cap_eval_p95_rel_obj, cap_eval_tail_rel_obj, cap_eval_max_rel_obj, cap_eval_abs_mean_err, cap_eval_abs_mean_err_rel, cap_eval_avg_obj,
                    cap_eval.max_obj, cap_eval.max_rel_obj, cap_pass ? 1 : 0, cap_lin, cap_better ? 1 : 0);
            }
        }
    }

    if (cap_compare_enabled && cap_selected_nb < 0) {
        // Safe-closed fallback: keep guarded choice; if guard had no pass-set winner, keep baseline.
        best_lin = guard_fallback ? base_lin : best2_lin;
        const int safe_bi_i = best_lin / NVFP4_TUNE_GRID_SIZE;
        const int safe_ai_i = best_lin - safe_bi_i * NVFP4_TUNE_GRID_SIZE;
        const int safe_ai = safe_ai_i - NVFP4_TUNE_GRID_CENTER;
        const int safe_bi = safe_bi_i - NVFP4_TUNE_GRID_CENTER;
        best_a_now = NVFP4_A0 + (float) safe_ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        best_b_now = NVFP4_B0 + (float) safe_bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
    }

    if (trace) {
        fprintf(stderr,
            "NVFP4_TUNE (CUDA guard) topk=%d val=%d base_obj_max=%g base_rel_obj_max=%g base_tail_rel_obj=%g base_abs_mean_err_rel=%g guard_obj=%g guard_rel_obj=%g guard_tail_rel_obj=%g guard_abs_mean_err_rel=%g chosen_rank=%d chosen_obj_norm=%g chosen_p95_rel_obj=%g chosen_tail_rel_obj=%g chosen_abs_mean_err_rel=%g fallback=%d\n",
            guard_topk, val_nb,
            base_max_obj, base_max_rel_obj, base_tail_rel_obj, base_abs_mean_err_rel,
            guard_max_obj, guard_max_rel_obj, guard_tail_rel_obj, guard_abs_mean_err_rel,
            best2_rank, best2_max_obj, best2_max_rel_obj, best2_tail_rel_obj, best2_abs_mean_err_rel,
            guard_fallback ? 1 : 0);
        if (cap_compare_enabled) {
            fprintf(stderr,
                "NVFP4_TUNE (CUDA cap-compare final) selected_nb=%d selected_obj_norm=%g selected_rel_rmse=%g selected_p95_rel_obj=%g selected_tail_rel_obj=%g selected_max_rel_obj=%g selected_abs_mean_err=%g selected_avg_obj=%g selected_lin=%d kept=%d gates=(obj_norm<=%g rel_rmse<=%g p95_rel_obj<=%g tail_rel_obj<=%g max_rel_obj<=%g abs_mean_err_rel<=%g min_improve=%g)\n",
                cap_selected_nb, cap_selected_obj_norm, cap_selected_rel_rmse, cap_selected_p95_rel_obj, cap_selected_tail_rel_obj, cap_selected_max_rel_obj, cap_selected_abs_mean_err, cap_selected_avg_obj, cap_selected_lin, cap_kept_count,
                cap_gate_obj_norm, cap_gate_rel_rmse, cap_gate_p95_rel_obj, cap_gate_tail_rel_obj, cap_gate_max_rel_obj, cap_gate_abs_mean_err_rel, (double) NVFP4_TUNE_MIN_IMPROVE_FRAC);
        }
    }

    // Metrics buffers (cached, capacity-managed)
    auto ensure_metric_capacity = [&](int64_t want_nb) {
        const size_t want = (size_t) want_nb;
        if (cap_metrics < want || !d_sse || !d_sum_err || !d_sum_w || !d_sum_x2) {
            if (d_sse)     { cudaFree(d_sse);     d_sse     = nullptr; }
            if (d_sum_err) { cudaFree(d_sum_err); d_sum_err = nullptr; }
            if (d_sum_w)   { cudaFree(d_sum_w);   d_sum_w   = nullptr; }
            if (d_sum_x2)  { cudaFree(d_sum_x2);  d_sum_x2  = nullptr; }

            CUDA_CHECK(cudaMalloc(&d_sse,     want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_err, want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_w,   want * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum_x2,  want * sizeof(float)));
            cap_metrics = want;
        }
        if (cap_h_metric_vec < want || !h_sse_pinned || !h_sum_err_pinned || !h_sum_w_pinned || !h_sum_x2_pinned) {
            const size_t bytes = want * sizeof(float);
            if (h_sse_pinned)     CUDA_CHECK(cudaFreeHost(h_sse_pinned));
            if (h_sum_err_pinned) CUDA_CHECK(cudaFreeHost(h_sum_err_pinned));
            if (h_sum_w_pinned)   CUDA_CHECK(cudaFreeHost(h_sum_w_pinned));
            if (h_sum_x2_pinned)  CUDA_CHECK(cudaFreeHost(h_sum_x2_pinned));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sse_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_err_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_w_pinned, bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc((void **) &h_sum_x2_pinned, bytes, cudaHostAllocDefault));
            cap_h_metric_vec = want;
        }
    };
    ensure_metric_capacity(sample_nb);

    auto lin_to_ab = [&](int lin, float & a, float & b) {
        const int bi_i = lin / NVFP4_TUNE_GRID_SIZE;
        const int ai_i = lin - bi_i * NVFP4_TUNE_GRID_SIZE;
        const int ai = ai_i - NVFP4_TUNE_GRID_CENTER;
        const int bi = bi_i - NVFP4_TUNE_GRID_CENTER;
        a = NVFP4_A0 + (float) ai * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
        b = NVFP4_B0 + (float) bi * (NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE);
    };

    float final_a = best_a_now;
    float final_b = best_b_now;
    tune_eval_stats final_eval;
    eval_candidate(d_x_buf, d_qw_use, (int) sample_nb, final_a, final_b, active_policy, final_eval);
    int64_t final_sample_nb = sample_nb;

    int64_t adaptive_nb = sample_nb;
    if (nb > sample_nb) {
        const bool risk_hi =
            !std::isfinite(final_eval.obj_norm) || !std::isfinite(final_eval.max_rel_obj) || !std::isfinite(final_eval.p95_rel_obj) ||
            final_eval.obj_norm >= NVFP4_TUNE_ADAPT_OBJ_NORM_HI ||
            final_eval.p95_rel_obj >= NVFP4_TUNE_ADAPT_P95_REL_OBJ_HI ||
            final_eval.max_rel_obj >= NVFP4_TUNE_ADAPT_MAX_REL_OBJ_HI ||
            final_eval.tail_rel_obj >= NVFP4_TUNE_ADAPT_TAIL_REL_OBJ_HI;
        const bool risk_mid =
            guard_fallback ||
            final_eval.obj_norm >= NVFP4_TUNE_ADAPT_OBJ_NORM_MID ||
            final_eval.p95_rel_obj >= NVFP4_TUNE_ADAPT_P95_REL_OBJ_MID ||
            final_eval.max_rel_obj >= NVFP4_TUNE_ADAPT_MAX_REL_OBJ_MID ||
            final_eval.tail_rel_obj >= NVFP4_TUNE_ADAPT_TAIL_REL_OBJ_MID;

        if (risk_hi) {
            adaptive_nb = std::min<int64_t>(nb, (int64_t) NVFP4_TUNE_ADAPT_NB_HI);
        } else if (risk_mid) {
            adaptive_nb = std::min<int64_t>(nb, (int64_t) NVFP4_TUNE_ADAPT_NB_MID);
        }
    }

    if (adaptive_nb > sample_nb) {
        const size_t bytes_x_adapt = (size_t) adaptive_nb * 256 * sizeof(float);
        if (!d_x_buf || cap_x < bytes_x_adapt) {
            if (d_x_buf) cudaFree(d_x_buf);
            CUDA_CHECK(cudaMalloc(&d_x_buf, bytes_x_adapt));
            cap_x = bytes_x_adapt;
        }
        copy_sampled_blocks(x, x_is_device, d_x_buf, adaptive_nb);

        if (qw) {
            const size_t bytes_qw_adapt = (size_t) adaptive_nb * 256 * sizeof(float);
            if (!d_qw_buf || cap_qw < bytes_qw_adapt) {
                if (d_qw_buf) cudaFree(d_qw_buf);
                CUDA_CHECK(cudaMalloc(&d_qw_buf, bytes_qw_adapt));
                cap_qw = bytes_qw_adapt;
            }
            copy_sampled_blocks(qw, nvfp4_is_device_ptr(qw), d_qw_buf, adaptive_nb);
            d_qw_use = d_qw_buf;
        } else {
            d_qw_use = nullptr;
        }
        ensure_metric_capacity(adaptive_nb);

        // Re-run objective grid on the larger sample pool so adaptive selection
        // is not constrained to only candidates from the initial 256-block pass.
        autotune<<<blocks, threads, 0, st>>>(
            d_x_buf, d_qw_use, adaptive_nb, d_results,
            active_policy.choose46_mode, active_policy.refit_iters, active_policy.use_compand_sat,
            active_policy.cap_m6, active_policy.cap_m4);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_results_pinned, d_results, bytes_scores, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        tune_eval_stats resample_base;
        eval_candidate(d_x_buf, d_qw_use, (int) adaptive_nb, NVFP4_A0, NVFP4_B0, active_policy, resample_base);

        auto safe_ratio = [](double v, double b) {
            if (std::isfinite(v) && std::isfinite(b) && b > 0.0) {
                return v / b;
            }
            return v;
        };
        auto robust_score = [&](const tune_eval_stats & s, const tune_eval_stats & b) {
            const double obj_ratio = safe_ratio(s.obj_norm, b.obj_norm);
            const double p95_ratio = safe_ratio(s.p95_rel_obj, b.p95_rel_obj);
            const double tail_ratio = safe_ratio(s.tail_rel_obj, b.tail_rel_obj);
            const double max_ratio = safe_ratio(s.max_rel_obj, b.max_rel_obj);
            const double mean_ratio = safe_ratio(s.abs_mean_err_rel, b.abs_mean_err_rel);
            return
                (double) NVFP4_TUNE_SCORE_W_OBJ_NORM * obj_ratio +
                (double) NVFP4_TUNE_SCORE_W_P95_REL_OBJ * p95_ratio +
                (double) NVFP4_TUNE_SCORE_W_TAIL_REL_OBJ * tail_ratio +
                (double) NVFP4_TUNE_SCORE_W_MAX_REL_OBJ * max_ratio +
                (double) NVFP4_TUNE_SCORE_W_ABS_MEAN_ERR_REL * mean_ratio;
        };
        auto nearly_eq = [](double a, double b) {
            const double s = fmax(1.0, fmax(fabs(a), fabs(b)));
            return fabs(a - b) <= 1e-12 * s;
        };

        constexpr int resample_cand_cap = NVFP4_TUNE_GUARD_TOPK + NVFP4_TUNE_RESAMPLE_CAND_EXTRA;
        int resample_lin[resample_cand_cap];
        int resample_n = 0;
        auto add_resample_cand = [&](int lin) {
            if (lin < 0 || resample_n >= resample_cand_cap) {
                return;
            }
            for (int i = 0; i < resample_n; ++i) {
                if (resample_lin[i] == lin) {
                    return;
                }
            }
            resample_lin[resample_n++] = lin;
        };

        // Adaptive-grid top-K seed from the enlarged sample set.
        int adapt_cand_lin[NVFP4_TUNE_GUARD_TOPK];
        float adapt_cand_obj[NVFP4_TUNE_GUARD_TOPK];
        for (int i = 0; i < NVFP4_TUNE_GUARD_TOPK; ++i) {
            adapt_cand_lin[i] = -1;
            adapt_cand_obj[i] = 1e30f;
        }
        const int adapt_topk = std::min(NVFP4_TUNE_GUARD_TOPK, n_scores);
        auto adapt_insert_topk = [&](int lin, float obj) {
            int pos = -1;
            for (int i = 0; i < adapt_topk; ++i) {
                if (obj < adapt_cand_obj[i]) {
                    pos = i;
                    break;
                }
            }
            if (pos < 0) {
                return;
            }
            for (int j = adapt_topk - 1; j > pos; --j) {
                adapt_cand_obj[j] = adapt_cand_obj[j - 1];
                adapt_cand_lin[j] = adapt_cand_lin[j - 1];
            }
            adapt_cand_obj[pos] = obj;
            adapt_cand_lin[pos] = lin;
        };
        for (int i = 0; i < n_scores; ++i) {
            const int bi_i = i / NVFP4_TUNE_GRID_SIZE;
            const int ai_i = i - bi_i * NVFP4_TUNE_GRID_SIZE;
            const int ai = ai_i - NVFP4_TUNE_GRID_CENTER;
            const int bi = bi_i - NVFP4_TUNE_GRID_CENTER;
            if (abs(ai) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP || abs(bi) >= NVFP4_TUNE_GRID_HALF_SPAN - EDGE_SKIP) {
                continue;
            }
            adapt_insert_topk(i, h_results_pinned[i]);
        }
        adapt_insert_topk(base_lin, h_results_pinned[base_lin]);

        add_resample_cand(base_lin);
        add_resample_cand(best_lin);
        add_resample_cand(best2_lin);
        if (cap_selected_lin >= 0) {
            add_resample_cand(cap_selected_lin);
        }
        for (int i = 0; i < adapt_topk && resample_n < resample_cand_cap; ++i) {
            add_resample_cand(adapt_cand_lin[i]);
        }
        for (int i = 0; i < guard_topk && resample_n < resample_cand_cap; ++i) {
            add_resample_cand(cand_lin[i]);
        }

        float resample_a[resample_cand_cap];
        float resample_b[resample_cand_cap];
        for (int ci = 0; ci < resample_n; ++ci) {
            lin_to_ab(resample_lin[ci], resample_a[ci], resample_b[ci]);
        }
        tune_eval_stats resample_eval[resample_cand_cap];
        eval_candidates_batch(d_x_buf, d_qw_use, (int) adaptive_nb, resample_a, resample_b, resample_n, active_policy, resample_eval);

        int best_resample_lin = base_lin;
        float best_resample_a = NVFP4_A0;
        float best_resample_b = NVFP4_B0;
        tune_eval_stats best_resample_eval = resample_base;
        double best_resample_score = robust_score(resample_base, resample_base);

        for (int ci = 0; ci < resample_n; ++ci) {
            const int lin = resample_lin[ci];
            const float a_c = resample_a[ci];
            const float b_c = resample_b[ci];
            const tune_eval_stats & cand_eval = resample_eval[ci];
            if (!std::isfinite(cand_eval.obj_norm) ||
                !std::isfinite(cand_eval.p95_rel_obj) ||
                !std::isfinite(cand_eval.tail_rel_obj) ||
                !std::isfinite(cand_eval.max_rel_obj) ||
                !std::isfinite(cand_eval.abs_mean_err_rel)) {
                continue;
            }

            const double score = robust_score(cand_eval, resample_base);
            bool better = false;
            if (score < best_resample_score && !nearly_eq(score, best_resample_score)) {
                better = true;
            } else if (nearly_eq(score, best_resample_score) &&
                       (cand_eval.obj_norm < best_resample_eval.obj_norm ||
                        (cand_eval.obj_norm == best_resample_eval.obj_norm && cand_eval.p95_rel_obj < best_resample_eval.p95_rel_obj) ||
                        (cand_eval.obj_norm == best_resample_eval.obj_norm && cand_eval.p95_rel_obj == best_resample_eval.p95_rel_obj && cand_eval.tail_rel_obj < best_resample_eval.tail_rel_obj) ||
                        (cand_eval.obj_norm == best_resample_eval.obj_norm && cand_eval.p95_rel_obj == best_resample_eval.p95_rel_obj && cand_eval.tail_rel_obj == best_resample_eval.tail_rel_obj && cand_eval.max_rel_obj < best_resample_eval.max_rel_obj) ||
                        (cand_eval.obj_norm == best_resample_eval.obj_norm && cand_eval.p95_rel_obj == best_resample_eval.p95_rel_obj && cand_eval.tail_rel_obj == best_resample_eval.tail_rel_obj && cand_eval.max_rel_obj == best_resample_eval.max_rel_obj && cand_eval.abs_mean_err_rel < best_resample_eval.abs_mean_err_rel))) {
                better = true;
            }
            if (!better) {
                continue;
            }

            best_resample_score = score;
            best_resample_lin = lin;
            best_resample_a = a_c;
            best_resample_b = b_c;
            best_resample_eval = cand_eval;
        }

        best_lin = best_resample_lin;
        best_a_now = best_resample_a;
        best_b_now = best_resample_b;
        final_a = best_resample_a;
        final_b = best_resample_b;
        final_eval = best_resample_eval;
        final_sample_nb = adaptive_nb;

        if (trace) {
            fprintf(stderr,
                "NVFP4_TUNE (CUDA adaptive-resample) samples0=%lld samples1=%lld cands=%d selected_lin=%d selected_score=%g obj_norm=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g\n",
                (long long) sample_nb, (long long) adaptive_nb, resample_n, best_resample_lin, best_resample_score,
                final_eval.obj_norm, final_eval.p95_rel_obj, final_eval.tail_rel_obj, final_eval.max_rel_obj);
        }
    }

    tune_policy_cfg final_policy = active_policy;
    int selector_total = 0;
    int selector_accept = 0;
    const bool run_selector = guard_risky || guard_fallback || final_sample_nb > sample_nb;

    auto selector_score = [](const tune_eval_stats & s) {
        return
            0.26 * s.obj_norm +
            0.24 * s.p95_rel_obj +
            0.16 * s.tail_rel_obj +
            0.28 * s.max_rel_obj +
            0.06 * s.abs_mean_err_rel;
    };
    auto selector_better = [](const tune_eval_stats & cand, double cand_score, const tune_eval_stats & best, double best_score) {
        const double eps = 1e-12;
        if (cand_score + eps < best_score) {
            return true;
        }
        if (fabs(cand_score - best_score) <= eps) {
            if (cand.max_rel_obj + eps < best.max_rel_obj) return true;
            if (fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && cand.p95_rel_obj + eps < best.p95_rel_obj) return true;
            if (fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && fabs(cand.p95_rel_obj - best.p95_rel_obj) <= eps && cand.obj_norm + eps < best.obj_norm) return true;
            if (fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && fabs(cand.p95_rel_obj - best.p95_rel_obj) <= eps && fabs(cand.obj_norm - best.obj_norm) <= eps && cand.tail_rel_obj + eps < best.tail_rel_obj) return true;
        }
        return false;
    };

    constexpr tune_policy_cfg selector_policies_raw[] = {
        { "baseline_recover_320",          NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 1, 320.0f, 256.0f },
        { "baseline_recover_384_256",      NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 1, 384.0f, 256.0f },
        { "baseline_recover_448_refit4",   NVFP4_CUDA_CHOOSE46_ADAPTIVE, 4,                     1, 448.0f, 256.0f },
        { "force_m6_384_256",              NVFP4_CUDA_CHOOSE46_FORCE_M6, NVFP4_TUNE_REFIT_ITERS, 1, 384.0f, 256.0f },
        { "force_m4_320_256",              NVFP4_CUDA_CHOOSE46_FORCE_M4, NVFP4_TUNE_REFIT_ITERS, 1, 320.0f, 256.0f },
        { "adaptive_nocompand_384_256",    NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 0, 384.0f, 256.0f },
    };

    constexpr int selector_policy_n = (int) (sizeof(selector_policies_raw) / sizeof(selector_policies_raw[0]));
    if (run_selector) {
        for (int pi = 0; pi < selector_policy_n; ++pi) {
            const tune_policy_cfg policy = normalize_policy(selector_policies_raw[pi]);
            if (same_policy(policy, final_policy)) {
                continue;
            }
            ++selector_total;

            constexpr int k_sel_max = 32;
            float cand_a[k_sel_max];
            float cand_b[k_sel_max];
            int cand_n = 0;
            auto add_ab = [&](float aa, float bb) {
                if (cand_n >= k_sel_max) {
                    return;
                }
                for (int i = 0; i < cand_n; ++i) {
                    if (fabsf(cand_a[i] - aa) <= 1e-12f && fabsf(cand_b[i] - bb) <= 1e-12f) {
                        return;
                    }
                }
                cand_a[cand_n] = aa;
                cand_b[cand_n] = bb;
                ++cand_n;
            };

            const float step = NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE;
            const float half = 0.5f * step;

            add_ab(final_a, final_b);
            add_ab(NVFP4_A0, NVFP4_B0);
            add_ab(final_a + step, final_b);
            add_ab(final_a - step, final_b);
            add_ab(final_a, final_b + step);
            add_ab(final_a, final_b - step);
            add_ab(final_a + step, final_b + step);
            add_ab(final_a + step, final_b - step);
            add_ab(final_a - step, final_b + step);
            add_ab(final_a - step, final_b - step);
            add_ab(final_a + 2.0f * step, final_b);
            add_ab(final_a - 2.0f * step, final_b);
            add_ab(final_a, final_b + 2.0f * step);
            add_ab(final_a, final_b - 2.0f * step);

            tune_eval_stats coarse_eval[k_sel_max];
            eval_candidates_batch(d_x_buf, d_qw_use, (int) final_sample_nb, cand_a, cand_b, cand_n, policy, coarse_eval);

            int coarse_best = -1;
            double coarse_best_score = 1e30;
            for (int ci = 0; ci < cand_n; ++ci) {
                const tune_eval_stats & ce = coarse_eval[ci];
                if (!std::isfinite(ce.obj_norm) || !std::isfinite(ce.p95_rel_obj) || !std::isfinite(ce.tail_rel_obj) || !std::isfinite(ce.max_rel_obj) || !std::isfinite(ce.abs_mean_err_rel)) {
                    continue;
                }
                const double sc = selector_score(ce);
                if (coarse_best < 0 || selector_better(ce, sc, coarse_eval[coarse_best], coarse_best_score)) {
                    coarse_best = ci;
                    coarse_best_score = sc;
                }
            }
            if (coarse_best < 0) {
                continue;
            }

            float local_best_a = cand_a[coarse_best];
            float local_best_b = cand_b[coarse_best];
            tune_eval_stats local_best_eval = coarse_eval[coarse_best];
            double local_best_score = coarse_best_score;

            float ref_a[16];
            float ref_b[16];
            int ref_n = 0;
            auto add_ref = [&](float aa, float bb) {
                if (ref_n >= 16) {
                    return;
                }
                for (int i = 0; i < ref_n; ++i) {
                    if (fabsf(ref_a[i] - aa) <= 1e-12f && fabsf(ref_b[i] - bb) <= 1e-12f) {
                        return;
                    }
                }
                ref_a[ref_n] = aa;
                ref_b[ref_n] = bb;
                ++ref_n;
            };
            add_ref(local_best_a, local_best_b);
            add_ref(local_best_a + half, local_best_b);
            add_ref(local_best_a - half, local_best_b);
            add_ref(local_best_a, local_best_b + half);
            add_ref(local_best_a, local_best_b - half);
            add_ref(local_best_a + half, local_best_b + half);
            add_ref(local_best_a + half, local_best_b - half);
            add_ref(local_best_a - half, local_best_b + half);
            add_ref(local_best_a - half, local_best_b - half);

            tune_eval_stats ref_eval[16];
            eval_candidates_batch(d_x_buf, d_qw_use, (int) final_sample_nb, ref_a, ref_b, ref_n, policy, ref_eval);
            for (int ri = 0; ri < ref_n; ++ri) {
                const tune_eval_stats & ce = ref_eval[ri];
                if (!std::isfinite(ce.obj_norm) || !std::isfinite(ce.p95_rel_obj) || !std::isfinite(ce.tail_rel_obj) || !std::isfinite(ce.max_rel_obj) || !std::isfinite(ce.abs_mean_err_rel)) {
                    continue;
                }
                const double sc = selector_score(ce);
                if (selector_better(ce, sc, local_best_eval, local_best_score)) {
                    local_best_a = ref_a[ri];
                    local_best_b = ref_b[ri];
                    local_best_eval = ce;
                    local_best_score = sc;
                }
            }

            const bool accepted = selector_better(local_best_eval, local_best_score, final_eval, selector_score(final_eval));
            if (accepted) {
                ++selector_accept;
                final_a = local_best_a;
                final_b = local_best_b;
                final_eval = local_best_eval;
                final_policy = policy;
            }

            if (trace) {
                fprintf(stderr,
                    "NVFP4_TUNE (CUDA selector) cand=%d/%d policy=%s choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f score=%g obj_norm=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g accepted=%d\n",
                    selector_total,
                    selector_policy_n,
                    policy.name,
                    policy.choose46_mode,
                    policy.refit_iters,
                    policy.use_compand_sat,
                    (double) policy.cap_m6,
                    (double) policy.cap_m4,
                    local_best_score,
                    local_best_eval.obj_norm,
                    local_best_eval.p95_rel_obj,
                    local_best_eval.tail_rel_obj,
                    local_best_eval.max_rel_obj,
                    accepted ? 1 : 0);
            }
        }
    }

    const double obj = final_eval.sum_sse;
    const double avg_obj = final_eval.avg_obj;
    const double obj_norm = final_eval.obj_norm;
    const double rmse = final_eval.rmse;
    const double x_rms = final_eval.x_rms;
    const double rel_rmse = final_eval.rel_rmse;
    const double mean_err = (final_eval.sum_w > 0.0) ? (final_eval.sum_err / final_eval.sum_w) : 0.0;
    const double sum_w = final_eval.sum_w;
    const double sum_err = final_eval.sum_err;
    const double p95_rel_obj = final_eval.p95_rel_obj;
    const double tail_rel_obj = final_eval.tail_rel_obj;
    const double max_rel_obj = final_eval.max_rel_obj;
    const int64_t used_blocks = final_eval.valid_blocks;

    *best_a = final_a;
    *best_b = final_b;

    const auto tune_t1 = std::chrono::steady_clock::now();
    const double tune_s = std::chrono::duration<double>(tune_t1 - tune_t0).count();

    /*if (trace) {*/
        fprintf(stderr,
            "NVFP4_TUNE (CUDA) a=%.10f b=%.10f obj=%g avg_obj=%g obj_norm=%g rmse=%g x_rms=%g rel_rmse=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g mean_err=%g sum_w=%g sum_err=%g used=%lld samples=%lld tune_s=%.4f guard_risky=%d guard_eval=%d/%d cap_compare=%d selector_accept=%d/%d policy=%s choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f\n",
            (double) *best_a, (double) *best_b,
            obj, avg_obj, obj_norm, rmse, x_rms, rel_rmse, p95_rel_obj, tail_rel_obj, max_rel_obj, mean_err, sum_w, sum_err,
            (long long) used_blocks, (long long) final_sample_nb,
            tune_s,
            guard_risky ? 1 : 0,
            guard_eval_n,
            guard_topk,
            run_cap_compare ? 1 : 0,
            selector_accept,
            selector_total,
            final_policy.name,
            final_policy.choose46_mode,
            final_policy.refit_iters,
            final_policy.use_compand_sat,
            (double) final_policy.cap_m6,
            (double) final_policy.cap_m4);
    //}
}

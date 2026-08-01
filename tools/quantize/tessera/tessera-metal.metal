//
// tessera-metal.metal
//
// Metal compute kernels for the Tessera quantize pipeline. These are
// quantize-time fitness kernels, NOT the T640 inference kernels in
// ggml-metal.metal. The layouts and scales consumed here come from the
// tessera-quant.cpp CPU path; the kernels below must reproduce those
// semantics bit-for-bit (within fp32 rounding) so the Metal and CPU paths
// are interchangeable.
//
// Three kernels:
//   1. ts_metal_sct_reduce + ts_metal_sct_ternarize
//        - fuses scale+clip+ternarize+meanabs (replaces FUSE A).
//   2. ts_metal_dmr
//        - fuses dequant+outlier-restore+MSE+recon (replaces FUSE B).
//   3. ts_metal_awq_grid
//        - batched AWQ alpha grid search (the big win).
//
// Conventions (mirror ggml-metal.metal):
//   - threadgroup shared memory for per-row / per-alpha reductions.
//   - simd_sum / simd_max for intra-warp reductions.
//   - float throughout (the CPU path is float; matching it avoids a precision
//     mismatch we would otherwise have to reconcile).
//

#include <metal_stdlib>
using namespace metal;

#define TS_PAGE_SIZE      640
#define TS_LANE_SIZE      20
#define TS_LANES_PER_PAGE 32

// Max row width the dequant/recon and awq-grid kernels handle directly.
// The host falls back to CPU for wider rows. 8192 covers essentially all
// real hidden_dim values; the scratch is threadgroup memory so this also
// keeps the per-threadgroup footprint under the 32KB Metal limit.
#ifndef TS_DMR_MAX_ROW
#define TS_DMR_MAX_ROW 8192
#endif
#ifndef TS_AWQ_MAX_PAGES
#define TS_AWQ_MAX_PAGES 32
#endif

// ---------------------------------------------------------------------------
// f16 <-> f32 helper (matches tessera-quant.cpp's round-to-nearest-even decode)
// ---------------------------------------------------------------------------

static inline float ts_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int e = -1;
            do { mant <<= 1; e--; } while ((mant & 0x400u) == 0u);
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(e + 127 + 1) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return as_type<float>(bits);
}

// ---------------------------------------------------------------------------
// Kernel argument structs (mirrored on the host side; keep in sync)
// ---------------------------------------------------------------------------

struct ts_sct_args {
    uint32_t out_dim;
    uint32_t in_dim;
    float    clip;        // 0 (or >= 1) disables clipping
    uint32_t do_clip;     // 1 if clip in (0,1)
};

struct ts_dmr_args {
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t n_outliers;
    uint32_t _pad;
};

struct ts_awq_args {
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t n_grid;
    uint32_t _pad;
};

// Per-row partial reductions.
struct ts_sct_partial {
    float abs_sum;   // sum |ws| over this row
    float maxabs;    // max |ws| over this row (for clipping)
};
struct ts_dmr_partial {
    float mse_sum;   // sum of (ws-deq)^2 over this row
    float n_count;   // number of elements accumulated
};
struct ts_awq_partial {
    float err_sum;   // sum of diff^2 * act2 over this row
    float n_count;
};

// ===========================================================================
// Kernel 1: scale + clip + ternarize (replaces FUSE A)
// ===========================================================================
// Split into two dispatches because the threshold is a global reduction:
//   phase 0 (ts_metal_sct_reduce): per row, ws[c]=W[r,c]*wscale[c], core=copy,
//     reduce sum|ws| and max|ws| into a device partial. Host sums partials to
//     get the global mean(|ws|) threshold and per-row clip limits.
//   phase 1 (ts_metal_sct_ternarize): per row, clip core in place at
//     limit=row_maxabs*clip, then ternarize against the global threshold.
//
// Threadgroup layout: 256 threads per row. Each thread strides through the
// row; rows up to 8192 cols are one threadgroup. The host tiles wider rows.

#define TS_SCT_THREADS 256

kernel void ts_metal_sct_reduce(
    constant ts_sct_args & args   [[buffer(0)]],
    device const float *   W      [[buffer(1)]],   // [out_dim * in_dim]
    device const float *   wscale [[buffer(2)]],   // [in_dim]
    device       float *   ws     [[buffer(3)]],   // [out_dim * in_dim]
    device       float *   core   [[buffer(4)]],   // [out_dim * in_dim]
    device ts_sct_partial * partials [[buffer(5)]], // [out_dim]
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    if (row >= args.out_dim) return;
    const uint in_dim = args.in_dim;
    device const float * Wrow    = W    + (uint64_t)row * in_dim;
    device       float * wsrow   = ws   + (uint64_t)row * in_dim;
    device       float * corerow = core + (uint64_t)row * in_dim;

    float local_abs_sum = 0.0f;
    float local_maxabs  = 0.0f;
    for (uint c = tid; c < in_dim; c += nt) {
        float v = Wrow[c] * wscale[c];
        wsrow[c]   = v;
        corerow[c] = v;
        float a = fabs(v);
        local_abs_sum += a;
        local_maxabs  = max(local_maxabs, a);
    }

    // simd reduction then cross-simdgroup via shared memory.
    local_abs_sum = simd_sum(local_abs_sum);
    local_maxabs  = simd_max(local_maxabs);

    threadgroup float tg_sum[32];
    threadgroup float tg_max[32];
    const uint sg = tid / 32u;
    const uint sl = tid & 31u;
    if (sl == 0u) {
        tg_sum[sg] = local_abs_sum;
        tg_max[sg] = local_maxabs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_simd = nt / 32u;
    if (tid == 0u) {
        float gsum = 0.0f, gmax = 0.0f;
        for (uint i = 0u; i < n_simd; i++) {
            gsum += tg_sum[i];
            gmax  = max(gmax, tg_max[i]);
        }
        partials[row].abs_sum = gsum;
        partials[row].maxabs  = gmax;
    }
}

kernel void ts_metal_sct_ternarize(
    constant ts_sct_args & args       [[buffer(0)]],
    device       float *   core       [[buffer(1)]],   // [out_dim * in_dim], clipped in place
    device       int8_t *  ternary    [[buffer(2)]],   // [out_dim * in_dim]
    device const float *   clip_limits[[buffer(3)]],   // [out_dim], row maxabs*clip
    constant float &       threshold  [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    if (row >= args.out_dim) return;
    const uint in_dim = args.in_dim;
    device float  * corerow = core     + (uint64_t)row * in_dim;
    device int8_t * terow   = ternary  + (uint64_t)row * in_dim;

    const float limit = args.do_clip ? clip_limits[row] : INFINITY;

    for (uint c = tid; c < in_dim; c += nt) {
        float v = corerow[c];
        if (args.do_clip) {
            v = clamp(v, -limit, limit);
            corerow[c] = v;
        }
        int8_t t = 0;
        if (fabs(v) >= threshold) {
            t = (v > 0.0f) ? (int8_t)1 : (v < 0.0f ? (int8_t)(-1) : (int8_t)0);
        }
        terow[c] = t;
    }
}

// ===========================================================================
// Kernel 2: dequant + outlier restore + MSE + recon (replaces FUSE B)
// ===========================================================================
// One threadgroup per row. A threadgroup scratch of in_dim floats holds the
// dequantized row so outlier restore and MSE can both read it without
// re-dequantizing. The scratch is sized to in_dim by the host encoder via
// setThreadgroupMemoryLength (unsized array bound at [[threadgroup(0)]]); this
// keeps the per-threadgroup footprint exactly in_dim*4 bytes (not the static
// max), so any in_dim up to the device's 32KB threadgroup limit (~8192 cols)
// works without exceeding it. Wider rows fall back to CPU.

kernel void ts_metal_dmr(
    constant ts_dmr_args & args         [[buffer(0)]],
    device const int8_t *  ternary      [[buffer(1)]],   // [out_dim * in_dim]
    device const uint16_t* page_scales  [[buffer(2)]],   // f16 [out_dim * pages_per_row]
    device const int8_t *  lane_scales  [[buffer(3)]],   // [out_dim * pages_per_row * 32]
    device const int32_t * outlier_idx  [[buffer(4)]],   // flat, sorted by row
    device const uint32_t* row_starts   [[buffer(5)]],   // [out_dim+1]
    device const float *   ws           [[buffer(6)]],   // [out_dim * in_dim]
    device const float *   input_scale  [[buffer(7)]],   // [in_dim]
    device       float *   recon        [[buffer(8)]],   // [out_dim * in_dim]
    device ts_dmr_partial * partials    [[buffer(9)]],  // [out_dim]
    threadgroup float *    deq_scratch  [[threadgroup(0)]],  // [in_dim], sized by host
    uint3   tgp [[threadgroup_position_in_grid]],
    uint    tid [[thread_index_in_threadgroup]],
    uint3   tpt [[threads_per_threadgroup]])
{
    const uint row = tgp.x;
    if (row >= args.out_dim) return;
    const uint in_dim = args.in_dim;
    const uint nt = tpt.x;
    const uint pages_per_row = (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;

    device const int8_t *  terow   = ternary     + (uint64_t)row * in_dim;
    device const float *   wsrow   = ws          + (uint64_t)row * in_dim;
    device       float *   reconrow= recon       + (uint64_t)row * in_dim;
    device const uint16_t* row_ps  = page_scales + (uint64_t)row * pages_per_row;
    device const int8_t *  row_ls  = lane_scales + (uint64_t)row * pages_per_row * TS_LANES_PER_PAGE;

    // 1. dequant into threadgroup scratch.
    for (uint c = tid; c < in_dim; c += nt) {
        const uint p    = c / TS_PAGE_SIZE;
        const uint lane = (c % TS_PAGE_SIZE) / TS_LANE_SIZE;
        const float amp = ts_f16_to_f32(row_ps[p]) *
                          (float)row_ls[p * TS_LANES_PER_PAGE + lane] * (1.0f / 127.0f);
        deq_scratch[c] = (float)terow[c] * amp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. outlier restore: deq[col] = ws[col]. outlier_idx entries are absolute
    //    (row*in_dim + col) indices, already filtered to this row's range by
    //    row_starts.
    const uint lo = row_starts[row];
    const uint hi = row_starts[row + 1];
    for (uint k = lo + tid; k < hi; k += nt) {
        const uint col = (uint)outlier_idx[k] % in_dim;
        deq_scratch[col] = wsrow[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. MSE = mean((ws-deq)^2) over the row (outliers now contribute 0) and
    //    recon[col] = deq[col] * input_scale[col].
    float local_mse = 0.0f;
    for (uint c = tid; c < in_dim; c += nt) {
        const float dq = deq_scratch[c];
        const float d  = wsrow[c] - dq;
        local_mse += d * d;
        reconrow[c] = dq * input_scale[c];
    }

    // Per-threadgroup MSE reduction. Reuse the tail of deq_scratch (after the
    // last live column) as scratch for the cross-simdgroup sum so we don't
    // need a second static threadgroup allocation. n_simd <= 8 for nt<=256.
    local_mse = simd_sum(local_mse);
    const uint sg = tid / 32u;
    const uint sl = tid & 31u;
    // park the partial sums just past deq_scratch's live range (in_dim floats).
    // The host allocates in_dim*4 + 256 bytes to leave room for this.
    threadgroup float * tg_sum = deq_scratch + in_dim;
    if (sl == 0u) tg_sum[sg] = local_mse;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_simd = nt / 32u;
    if (tid == 0u) {
        float gsum = 0.0f;
        for (uint i = 0u; i < n_simd; i++) gsum += tg_sum[i];
        partials[row].mse_sum = gsum;
        partials[row].n_count = (float)in_dim;
    }
}

// ===========================================================================
// Kernel 3: batched AWQ grid search (the big win)
// ===========================================================================
// Two-phase dispatch because the ternarize threshold is a GLOBAL reduction
// (mean(|ws|) over ALL rows for a given alpha), not per-row:
//
//   phase 1 (ts_metal_awq_threshold): per (row, alpha) accumulate sum|ws|;
//     host reduces across rows to get threshold[g] = sum / (out_dim*in_dim).
//   phase 2 (ts_metal_awq_grid): per (row, alpha) ternarize against
//     threshold[g], fit per-page/lane scales, dequant, accumulate
//     importance-weighted err; host reduces to one MSE per alpha.
//
// `inv_median` is 1/median(|act_scales|) on the finite positives, precomputed
// by the host (matches ts_median_finite_positive in tessera-quant.cpp).

// ---- phase 1: per-(row, alpha) abs_sum partials ----
kernel void ts_metal_awq_threshold(
    constant ts_awq_args & args    [[buffer(0)]],
    device const float *   W       [[buffer(1)]],   // [out_dim * in_dim]
    device const float *   act     [[buffer(2)]],   // [in_dim]
    device const float *   grid    [[buffer(3)]],   // [n_grid]
    constant float &       inv_median [[buffer(4)]],
    device float *         abs_partials [[buffer(5)]], // [out_dim * n_grid]
    uint3 tgp [[threadgroup_position_in_grid]],     // .x=row, .y=alpha
    uint  tid [[thread_index_in_threadgroup]],
    uint3 tpt [[threads_per_threadgroup]])
{
    const uint row  = tgp.x;
    const uint gidx = tgp.y;
    const uint nt   = tpt.x;
    if (row >= args.out_dim || gidx >= args.n_grid) return;
    const uint  in_dim = args.in_dim;
    const float alpha  = grid[gidx];

    device const float * Wrow = W + (uint64_t)row * in_dim;

    float abs_sum = 0.0f;
    for (uint c = tid; c < in_dim; c += nt) {
        float rel = clamp(act[c] * inv_median, 1.0f / 256.0f, 256.0f);
        abs_sum += fabs(Wrow[c] * pow(rel, alpha));
    }
    abs_sum = simd_sum(abs_sum);
    threadgroup float tg_sum[32];
    const uint sg = tid / 32u;
    const uint sl = tid & 31u;
    const uint n_simd = nt / 32u;
    if (sl == 0u) tg_sum[sg] = abs_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (uint i = 0u; i < n_simd; i++) g += tg_sum[i];
        abs_partials[(uint64_t)row * args.n_grid + gidx] = g;
    }
}

// ---- phase 2: ternarize + dequant + err, using the resolved thresholds ----
kernel void ts_metal_awq_grid(
    constant ts_awq_args & args    [[buffer(0)]],
    device const float *   W       [[buffer(1)]],   // [out_dim * in_dim]
    device const float *   act     [[buffer(2)]],   // [in_dim]
    device const float *   act2    [[buffer(3)]],   // [in_dim] = act^2
    device const float *   grid    [[buffer(4)]],   // [n_grid]
    constant float *       thresholds [[buffer(5)]], // [n_grid] resolved mean(|ws|)
    constant float &       inv_median [[buffer(6)]], // 1/median(|act|)
    device ts_awq_partial * partials [[buffer(7)]], // [out_dim * n_grid]
    uint3 tgp [[threadgroup_position_in_grid]],     // .x=row, .y=alpha idx
    uint  tid [[thread_index_in_threadgroup]],
    uint3 tpt [[threads_per_threadgroup]])
{
    const uint row  = tgp.x;
    const uint gidx = tgp.y;
    const uint nt   = tpt.x;
    if (row >= args.out_dim || gidx >= args.n_grid) return;
    const uint  in_dim = args.in_dim;
    const float alpha  = grid[gidx];
    const float threshold = thresholds[gidx];

    device const float * Wrow = W + (uint64_t)row * in_dim;

    threadgroup float tg_sum[32];
    const uint sg = tid / 32u;
    const uint sl = tid & 31u;
    const uint n_simd = nt / 32u;

    // ---- per-page lane fit + dequant + err ----
    const uint pages_per_row = (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;
    threadgroup float lane_tg[TS_AWQ_MAX_PAGES * TS_LANES_PER_PAGE];

    float local_err = 0.0f;
    for (uint p = 0; p < pages_per_row; p++) {
        // lane targets: one thread per lane (32 threads).
        if (tid < TS_LANES_PER_PAGE) {
            float s_abs = 0.0f;
            int   cnt   = 0;
            for (uint k = 0; k < TS_LANE_SIZE; k++) {
                const uint c = p * TS_PAGE_SIZE + tid * TS_LANE_SIZE + k;
                if (c >= in_dim) break;
                const float rel = clamp(act[c] * inv_median,
                                        1.0f / 256.0f, 256.0f);
                const float wsv = Wrow[c] * pow(rel, alpha);
                if (fabs(wsv) >= threshold) { s_abs += fabs(wsv); cnt++; }
            }
            lane_tg[p * TS_LANES_PER_PAGE + tid] = (cnt > 0) ? (s_abs / (float)cnt) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // page_max = max lane target (reduced across the 32 lane-threads).
        float page_max = (tid < TS_LANES_PER_PAGE)
                         ? lane_tg[p * TS_LANES_PER_PAGE + tid] : 0.0f;
        page_max = simd_max(page_max);
        if (tid == 0u) lane_tg[p * TS_LANES_PER_PAGE] = page_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        page_max = lane_tg[p * TS_LANES_PER_PAGE];
        if (page_max < 1e-30f) page_max = 1.0f;

        // dequant + err. Each thread strides the 640 page columns.
        for (uint c = tid; c < TS_PAGE_SIZE; c += nt) {
            const uint col = p * TS_PAGE_SIZE + c;
            if (col >= in_dim) break;
            const float rel = clamp(act[col] * inv_median, 1.0f / 256.0f, 256.0f);
            const float s   = pow(rel, alpha);
            const float wsv = Wrow[col] * s;
            int t = 0;
            if (fabs(wsv) >= threshold) {
                t = (wsv > 0.0f) ? 1 : (wsv < 0.0f ? -1 : 0);
            }
            const uint  lane = c / TS_LANE_SIZE;
            const float lt   = lane_tg[p * TS_LANES_PER_PAGE + lane];
            // Match ts_compute_scales + ts_dequant exactly: the stored lane
            // scale is round(lt/page_max*127) clamped to [1,127], and the
            // dequant amplitude is page_max * lane_q / 127.
            float raw_q = (lt / page_max) * 127.0f;
            int   q = (int) rint(raw_q);
            q = max(1, min(127, q));
            const float amp = page_max * (float)q * (1.0f / 127.0f);
            const float dq       = (float)t * amp;
            const float dq_orig  = dq / s;       // back to original weight space
            const float d        = dq_orig - Wrow[col];
            local_err += d * d * act2[col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // reduce err across threadgroup, write per-(row, alpha) partial.
    local_err = simd_sum(local_err);
    if (sl == 0u) tg_sum[sg] = local_err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (uint i = 0u; i < n_simd; i++) g += tg_sum[i];
        const uint64_t pidx = (uint64_t)row * args.n_grid + gidx;
        partials[pidx].err_sum = g;
        partials[pidx].n_count = (float)in_dim;
    }
}

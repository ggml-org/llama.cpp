#include "common.h"

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
kernel void kernel_ssm_conv_f32_f32(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = args.ne10;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = args.ne10;
  //const int64_t ncs = args.ne00;
  //const int64_t nr  = args.ne01;
  //const int64_t n_t = args.ne1;
  //const int64_t n_s = args.ne2;

    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);
    device       float  * x = (device       float  *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = sumf;
}

constant short FC_ssm_conv_bs   [[function_constant(FC_SSM_CONV + 0)]];

// Batched version: each threadgroup processes multiple tokens for better efficiency
// Thread layout: each thread handles one token, threadgroup covers BATCH_SIZE tokens
kernel void kernel_ssm_conv_f32_f32_batched(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_batched_4(
        constant ggml_metal_kargs_ssm_conv & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    // tgpig.x = row index (ir)
    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)
    // tgpig.z = sequence index (i3)
    // tpitg.x = thread within batch (0..BATCH_SIZE-1)
    const short BATCH_SIZE = FC_ssm_conv_bs;

    const int64_t ir      = tgpig.x;
    const int64_t i2_base = tgpig.y * BATCH_SIZE;
    const int64_t i3      = tgpig.z;
    const int64_t i2_off  = tpitg.x;
    const int64_t i2      = i2_base + i2_off;

    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)
    const int64_t n_t = args.ne1;   // number of tokens

    // Bounds check for partial batches at the end
    if (i2 >= n_t) {
        return;
    }

    // Load conv weights (shared across all tokens for this row)
    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);

    // Load source for this specific token
    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);

    // Output location for this token
    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32, Mamba-2 part
// Optimized version: reduces redundant memory loads by having one thread load shared values
// TAIL == false is the whole-sequence / decode path: token_offset folds away at compile time.
template<bool TAIL>
kernel void kernel_ssm_scan_impl(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgptg[[simdgroups_per_threadgroup]],
        uint3    tgpg[[threadgroups_per_grid]]) {
    constexpr short NW = N_SIMDWIDTH;

    // Shared memory layout:
    // [0..sgptg*NW-1]: partial sums for reduction (existing)
    // [sgptg*NW..sgptg*NW+sgptg-1]: pre-computed x_dt values for each token in batch
    // [sgptg*NW+sgptg..sgptg*NW+2*sgptg-1]: pre-computed dA values for each token in batch
    threadgroup float * shared_sums = shared;
    threadgroup float * shared_x_dt = shared + sgptg * NW;
    threadgroup float * shared_dA   = shared + sgptg * NW + sgptg;

    shared_sums[tpitg.x] = 0.0f;

    const int32_t i0 = tpitg.x;
    const int32_t i1 = tgpig.x;
    const int32_t ir = tgpig.y; // current head
    const int32_t i3 = tgpig.z; // current seq

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;
    const int32_t n_s = args.n_seqs;
    const int32_t K   = args.K;
    const int32_t n_t_total = TAIL ? args.n_seq_tokens_total : n_t;
    const int32_t t_off     = TAIL ? args.token_offset       : 0;

    const int32_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);
    device const float * s0_buff = t_off != 0 ?
        s_buff :
        (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);

    const int32_t i = i0 + i1*nc;
    const int32_t g = ir / (nh / ng); // repeat_interleave

    float s0 = s0_buff[i];
    float s  = 0.0f;

    device const float * A = (device const float *) ((device const char *) src3 + ir*args.nb31); // {ne30, nh}

    const float A0 = A[i0%args.ne30];

    device const float * x  = (device const float *)((device const char *) src1 + i1*args.nb10 + ir*args.nb11 + t_off*args.nb12 + i3*args.nb13); // {dim, nh, nt, ns}
    device const float * dt = (device const float *)((device const char *) src2 + ir*args.nb20 + t_off*args.nb21 + i3*args.nb22);                 // {nh, nt, ns}
    device const float * B  = (device const float *)((device const char *) src4 + g*args.nb41 + t_off*args.nb42 + i3*args.nb43);                  // {d_state, ng, nt, ns}
    device const float * C  = (device const float *)((device const char *) src5 + g*args.nb51 + t_off*args.nb52 + i3*args.nb53);                  // {d_state, ng, nt, ns}

    device float * y = dst + (i1 + ir*nr + t_off*nh*nr + i3*(n_t_total*nh*nr)); // {dim, nh, nt, ns}

    for (int i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-compute x_dt and dA for this batch of tokens
        // Only first sgptg threads do the loads and expensive math
        if (i0 < sgptg && i2 + i0 < n_t) {
            // ns12 and ns21 are element strides (nb12/nb10, nb21/nb20)
            device const float * x_t  = x  + i0 * args.ns12;
            device const float * dt_t = dt + i0 * args.ns21;

            const float dt0  = dt_t[0];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_x_dt[i0] = x_t[0] * dtsp;
            shared_dA[i0]   = dtsp;  // Store dtsp, compute exp(dtsp * A0) per-thread since A0 varies
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            const float x_dt = shared_x_dt[t];
            const float dA   = exp(shared_dA[t] * A0);

            s = (s0 * dA) + (B[i0] * x_dt);

            const float sumf = simd_sum(s * C[i0]);

            if (tiisg == 0) {
                shared_sums[t*NW + sgitg] = sumf;
            }

            // recurse
            s0 = s;

            const int32_t slot = n_t - 1 - (i2 + t);
            if (slot > 0 && slot < K) {
                device float * s_snapshot = (device float *) ((device char *) s_buff + (int64_t) slot*n_s*args.nb03);
                s_snapshot[i] = s;
            }

            B  += args.ns42;
            C  += args.ns52;
        }

        // Advance pointers for next batch
        x  += sgptg * args.ns12;
        dt += sgptg * args.ns21;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sumf = simd_sum(shared_sums[sgitg*NW + tiisg]);

        if (tiisg == 0 && i2 + sgitg < n_t) {
            y[sgitg*nh*nr] = sumf;
        }

        y += sgptg*nh*nr;
    }

    s_buff[i] = s;
}

typedef decltype(kernel_ssm_scan_impl<false>) kernel_ssm_scan_t;

template [[host_name("kernel_ssm_scan_f32")]]      kernel kernel_ssm_scan_t kernel_ssm_scan_impl<false>;
template [[host_name("kernel_ssm_scan_f32_tail")]] kernel kernel_ssm_scan_t kernel_ssm_scan_impl<true>;

// Chunked SSD SSM scan via Metal simdgroup MMatrix Multiply-Accumulate (simdgroup_float8x8) fast path.
// One threadgroup per (head, sequence) and tokens are processed in chunks.
// C*B^T computed in each chunk one time and reused across the head_dim channel tiles.
kernel void kernel_ssm_scan_ssd_mma_f32(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr short CS  = OP_SSM_SCAN_SSD_CS;
    constexpr short TC  = 8; // Tile Count of each edge in a simdgroup 8x8 tile
    constexpr short HD  = OP_SSM_SCAN_SSD_HD;
    constexpr short NSG = OP_SSM_SCAN_SSD_NSG;

    // acs/exp(acs)/state-decay vectors, dtX[CS][HD], four private SAM row tiles [8][CS],
    // and two 8x8 scratch tiles per simdgroup. Total: 26.75 KiB.
    threadgroup float * shared_acs         = shared;
    threadgroup float * shared_exp_acs     = shared + CS;
    threadgroup float * shared_state_decay = shared + 2*CS;
    threadgroup float * shared_dtx         = shared + 3*CS;
    threadgroup float * shared_sam         = shared + 3*CS + CS*HD;
    threadgroup float * sam_rows    = shared_sam + sgitg*TC*CS;
    threadgroup float * shared_tile = shared_sam + NSG*TC*CS;
    threadgroup float * tile0       = shared_tile + sgitg*2*TC*TC;
    threadgroup float * tile1       = tile0 + TC*TC;

    const int32_t ir = tgpig.y; // current head
    const int32_t i3 = tgpig.z; // current seq

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;
    const int32_t n_t_total = args.n_seq_tokens_total;
    const int32_t g   = ir / (nh / ng);

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + args.s_off);

    device const float * A  = (device const float *) ((device const char *) src3 + ir*args.nb31);
    device const float * x  = (device const float *) ((device const char *) src1 + ir*args.nb11 + i3*args.nb13);
    device const float * dt = (device const float *) ((device const char *) src2 + ir*args.nb20 + i3*args.nb22);
    device const float * B  = (device const float *) ((device const char *) src4 +  g*args.nb41 + i3*args.nb43);
    device const float * C  = (device const float *) ((device const char *) src5 +  g*args.nb51 + i3*args.nb53);

    device float * y = dst + (ir*nr + i3*(n_t_total*nh*nr));

    for (int32_t t0 = 0; t0 < n_t; t0 += CS) {
        for (int32_t idx = tiitg; idx < CS*HD; idx += NSG*N_SIMDWIDTH) {
            const int32_t t = idx / HD;
            const int32_t c = idx % HD;
            const float dt0  = dt[(t0 + t) * (int32_t) args.ns21];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_dtx[idx] = x[(t0 + t) * (int32_t) args.ns12 + c] * dtsp;
        }
        if (tiitg < CS) {
            const float dt0  = dt[(t0 + tiitg) * (int32_t) args.ns21];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_acs[tiitg] = dtsp * A[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg == 0) {
            float acc = 0.0f;
            for (short t = 0; t < CS; ++t) {
                acc += shared_acs[t];
                shared_acs[t] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < CS) {
            shared_exp_acs[tiitg] = exp(shared_acs[tiitg]);
            shared_state_decay[tiitg] = exp(shared_acs[CS - 1] - shared_acs[tiitg]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float * state = t0 == 0 ? s0_buff : s_buff;

        // Build one 8x64 row tile of SAM per simdgroup, then reuse it across every channel tile.
        for (short ib = sgitg; ib < CS/TC; ib += NSG) {
            for (short jb = 0; jb <= ib; ++jb) {
                simdgroup_float8x8 cb = make_filled_simdgroup_matrix<float, 8>(0.0f);

                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc;
                    simdgroup_float8x8 mb;
                    simdgroup_load(mc, C + (t0 + ib*TC)*(int32_t) args.ns52 + k0, args.ns52);
                    simdgroup_load(mb, B + (t0 + jb*TC)*(int32_t) args.ns42 + k0, args.ns42, 0, true);
                    simdgroup_multiply_accumulate(cb, mc, mb, cb);
                }

                threadgroup float * sam = sam_rows + jb*TC;
                simdgroup_store(cb, sam, CS);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (short e = tiisg; e < TC*TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short rj = e % TC;
                    const short i  = ib*TC + ri;
                    const short j  = jb*TC + rj;
                    sam[ri*CS + rj] = j <= i ?
                        sam[ri*CS + rj] * exp(shared_acs[i] - shared_acs[j]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (short ch = 0; ch < HD/TC; ++ch) {
                simdgroup_float8x8 y_diag  = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_float8x8 y_inter = make_filled_simdgroup_matrix<float, 8>(0.0f);

                for (short jb = 0; jb <= ib; ++jb) {
                    simdgroup_float8x8 sam;
                    simdgroup_float8x8 mdtx;
                    simdgroup_load(sam,  sam_rows + jb*TC,                   CS);
                    simdgroup_load(mdtx, shared_dtx + jb*TC*HD + ch*TC,     HD);
                    simdgroup_multiply_accumulate(y_diag, sam, mdtx, y_diag);
                }

                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc;
                    simdgroup_float8x8 ms;
                    simdgroup_load(mc, C + (t0 + ib*TC)*(int32_t) args.ns52 + k0, args.ns52);
                    simdgroup_load(ms, state + ch*TC*nc + k0, nc, 0, true);
                    simdgroup_multiply_accumulate(y_inter, mc, ms, y_inter);
                }

                simdgroup_store(y_diag,  tile0, TC);
                simdgroup_store(y_inter, tile1, TC);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (short e = tiisg; e < TC*TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short ci = e % TC;
                    const int32_t token = t0 + ib*TC + ri;
                    y[token*nh*nr + ch*TC + ci] =
                        tile0[e] + shared_exp_acs[ib*TC + ri] * tile1[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // All simdgroups must finish reading s_buff before any thread overwrites it.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // Keep the carried-state reduction in token order. Reassociating this particular product
        // with MMA compounds rounding differences at every chunk boundary; CB, y_diag, and C*S
        // remain on the matrix unit.
        const float chunk_decay = exp(shared_acs[CS - 1]);
        for (int32_t idx = tiitg; idx < nc*HD; idx += NSG*N_SIMDWIDTH) {
            const int32_t ci = idx / nc;
            const int32_t si = idx % nc;
            float state_c = 0.0f;
            for (short t = 0; t < CS; ++t) {
                state_c += shared_state_decay[t] *
                    B[(t0 + t)*(int32_t) args.ns42 + si] *
                    shared_dtx[t*HD + ci];
            }
            s_buff[idx] = chunk_decay * state[idx] + state_c;
        }

        // All state tiles must be visible before the next chunk consumes s_buff as S_prev.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

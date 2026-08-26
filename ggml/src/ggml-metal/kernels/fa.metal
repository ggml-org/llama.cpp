#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

//constant float FC_flash_attn_ext_scale         [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
//constant float FC_flash_attn_ext_max_bias      [[function_constant(FC_FLASH_ATTN_EXT + 11)]];
//constant float FC_flash_attn_ext_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT + 12)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
    short NSG>        // number of simd groups
void kernel_flash_attn_ext_impl(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    // note: I had some concerns that using this instead of the ugly macros above was affecting performance
    //       need to re-check carefully and if no regressions are observerd - remove the macros
    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler
    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC
    //const short NS10 = FC_flash_attn_ext_ns10;
    //const short NS20 = FC_flash_attn_ext_ns20;

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
  //constexpr short DV8  = DV/8;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
  //constexpr short PV16 = PV/16;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t

    // mask storage in shared mem
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    // per-query mask pointers
    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            // read the mask into shared mem
            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }

#if 0
                // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // used to detect blocks full of -INF
                // skip only when the entire threadgroup is masked
                half2 smax2(-MAXHALF/2, -MAXHALF/2);

                FOR_UNROLL (short j = 0; j < Q; ++j) {
                    smax2 = max(smax2, sm2[j*SH + tiisg]);
                }

                smax2 = simd_max(smax2);

                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
                    // this barrier is important
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    continue;
                }
#endif
            }

            // Q*K^T
            // this is compile-time check, so it does not have runtime overhead
            if (is_same<kd4x4_t, k4x4_t>::value) {
                // we can read directly from global memory
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        // note: too much unroll can tank the performance for large heads
                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                // TODO: this is the quantized K cache branch - not optimized yet
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                // scale and apply the logitcap / mask
                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                // mqk = mqk + slope*mask
                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                // the P matrix from the paper (Q rows, C columns)
                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                // we can read directly from global memory
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    // TODO: this is the quantized V cache branch - not optimized yet

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    // store to global memory
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q  = OP_FLASH_ATTN_EXT_NQPSG, // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_NCPSG> // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
      // note: disabled cases to reduce library load time
      //case 1: kernel_flash_attn_ext_impl<FWD_TMPL, 1>(FWD_ARGS); break;
      //case 2: kernel_flash_attn_ext_impl<FWD_TMPL, 2>(FWD_ARGS); break;
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f32_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f32_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f32_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f32_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f32_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f32_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f32_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f32_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  576, 512>;

template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

template [[host_name("kernel_flash_attn_ext_q4_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q4_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q8_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];

//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];

constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*Q*C; // shared memory per simdgroup

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

  //const short T = PK + NSG*SH; // shared memory size per query in (half)

  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                            0*PK); // holds the query data
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                            0*PK); // same as above but in q4_t
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // same as above but in s4_t
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*Q*C + Q*NSG*PK); // scratch buffer for mask
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*Q*PV       + Q*NSG*PK + NSG*SH); // scratch buffer for the results

    // store the result for all queries in shared memory (the O matrix from the paper)
    so4 += tiisg;

    {
        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load Q query rows to shared memory
    {
        for (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
            if (iq1_q < args.ne01) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[qq*PK4 + i] = (q4_t) q4[i];
                    } else {
                        sq4[qq*PK4 + i] = (q4_t) 0.0f;
                    }
                }
            } else {
                for (short i = tiisg; i < PK4; i += NW) {
                    sq4[qq*PK4 + i] = (q4_t) 0.0f;
                }
            }
        }
    }

    // zero out so
    for (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq*DV4 + i*NL] = (o4_t) 0.0f;
        }
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q];
        float M[Q];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            S[qq] = 0.0f;
            M[qq] = -FLT_MAX/2;
        }

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // pointer to the mask
        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            device const half * pm[Q];
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                // padded query rows clamp to row 0 of the mask to avoid OOB; their scores
                // are forced to -inf below, so the values never affect the result.
                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (ic + tiisg >= args.ne11) {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        pm[qq] = (device const half *) (mask) +
                            (iq1*Q + qq)*C +
                            (iq2%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    }
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) < args.ne01) {
                        sm[qq*C + tiisg] = pm[qq][ic + tiisg];
                    } else {
                        sm[qq*C + tiisg] = -MAXHALF;
                    }
                }
            } else {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) >= args.ne01) {
                        sm[qq*C + tiisg] = -MAXHALF;
                    }
                }
            }

            {
                bool any_finite = false;
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (simd_max(sm[qq*C + tiisg]) > -MAXHALF) {
                        any_finite = true;
                    }
                }
                if (!any_finite) {
                    continue;
                }
            }

            // Q*K^T
            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);

                pk4 += ty*NS10/4 + tx;

                qk_t mqk[Q][C/NE];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        mqk[qq][cc] = 0.0f;
                    }
                }

                // each simdgroup processes Q queries and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                            }
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                            }
                        }
                    }

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (NE == 1) {
                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                        } else {
                            // simdgroup reduce (NE = 4)
                            // [ 0 ..  7] -> [ 0]
                            // [ 8 .. 15] -> [ 8]
                            // [16 .. 23] -> [16]
                            // [24 .. 31] -> [24]
                            if (NE <= 1) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
                            }

                            // broadcast
                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (FC_flash_attn_ext_vec_has_mask &&
                       !FC_flash_attn_ext_vec_has_scap &&
                       !FC_flash_attn_ext_vec_has_bias) {
                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (qk_t) sm[qq*C + NE*tx + ty]);
                    } else {
                        mqk[qq][tx] *= args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            mqk[qq][tx] = args.logit_softcap*precise::tanh(mqk[qq][tx]);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty]*slope;
                        } else {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty];
                        }

                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const float m = M[qq];
                    const float s = ss[qq*C + tiisg];

                    M[qq] = simd_max(max(M[qq], s));

                    const float ms = exp(m - M[qq]);
                    const float vs = exp(s - M[qq]);

                    S[qq] = S[qq]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[qq*C + tiisg] = vs;

                    // O = diag(ms)*O
                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[Q][DV4/NL];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[qq][ii] = 0.0f;
                    }
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + NE*cc + ty]));
                            }
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float m = M[qq];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[qq] = simd_max(max(M[qq], s));

                const float ms = exp(m - M[qq]);
                const float vs = exp(s - M[qq]);

                S[qq] = S[qq]*ms + simd_sum(vs);

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[qq*DV4 + ii*NL] *= ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                ss[2*qq + 0] = (s_t) S[qq];
                ss[2*qq + 1] = (s_t) M[qq];
            }
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;

            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            // interleave the workgroup data
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
            }

            // store S and M
            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 32, 32, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;


#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    constexpr short NH    = OP_LIGHTNING_INDEXER_NH;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                sq4[ih*DK4 + i4] = half4(q4[i4]);
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = ((device const float *) pw)[i_head + tiitg];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;

// ============================================================================
// Tensor API (MPP tensor_ops) Flash Attention prefill kernel
//
// One simdgroup per threadgroup (32 threads); each matmul op runs with
// execution scope = the whole (single) simdgroup (metal::execution_simdgroup).
// Each threadgroup handles QPSG = 8 queries over the full KV range (same
// "one threadgroup per Q tile, full KV loop" model as the matrix kernel above
// - no split-KV across simdgroups).
//
// IMPORTANT: the MPP coop destination tile layout is implementation-defined
// (it depends on the op, the dtypes, the opscope AND the surrounding code /
// register pressure).  Two verified landmines (do not "simplify" these away):
//   - the output write must be ELEMENT pointer arithmetic on a float*
//     ((device float *) dst + (uint64_t)elems * DV); doing the offset in BYTES
//     (char* + elems*DV*4) corrupts the QK^T (C, QPSG) coop tile layout
//   - dispatching 4 simdgroups with per-lane execution scope (the former
//     design) only produced correct results in the presence of extra
//     dead-code "bloat" - the 1-simdgroup design is the stable one
//
// Verified design notes (see the Phase-0 spike, 2026-07):
//   - QK^T: descriptor(m=C, n=QPSG, k=DK, transL=false, transR=true);
//     left = K tile (DK, C) f16 device, right = Q tile (DK, QPSG) f32 device
//     (half x float -> float is in the dtype table; Q is f32 in ggml)
//   - the QK destination coop tensor (f32) has idx0 = query, idx1 = kv; each
//     thread owns a (2 queries x 8 kv) sub-tile
//   - per-query max/sum: in-register partials + threadgroup exchange
//     (the hardware reduce_rows/reduce_columns reduce the wrong axis for this
//     tile shape and their ownership is not what we need)
//   - P stays in registers (f32) and is fed directly to the PV matmul as the
//     right input: get_right_input_cooperative_tensor<half, float, float>(P)
//   - PV: descriptor(m=PVM, n=QPSG, k=C, transL=true, transR=false),
//     left = V tile (PVM, C) f16 device, m dimension capped at 128 -> for
//     dv > 128 the PV is split into d blocks (PVM = 128, or 64 for dv % 128
//     != 0); each block re-runs the chunk loop with a self-contained f32
//     accumulator (a coop destination tile is a compiler-managed register
//     tile: distinct named objects do not share storage)
//   - the PV destination has idx0 = query, idx1 = d; the output is written
//     element-wise (store() only works for the transposed natural layout)
//   - the last partial KV chunk (ne11 % C != 0) is read from the pad buffer
//     produced by kernel_flash_attn_ext_pad (padded rows are zero; the padded
//     scores are clobbered to -FLT_MAX/2 in registers before the softmax)
//
// fast-path gate (host-side): f16 K/V, no mask/sinks/ALiBi/softcap,
// dk % 16 == 0, dv % 64 == 0 && dv <= 512, ne01 % (NLANES*QPSG) == 0 (== 8)
// ============================================================================
#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// note: file-scope constants are not allowed in MSL, so these are defined in the kernel
//       (the host-side gate mirrors the values: NLANES*QPSG = 8 queries per threadgroup)
template <short DK, short DV>
void kernel_flash_attn_ext_tensor_impl(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiitg,
        ushort  sgitg) {
    using namespace mpp::tensor_ops;

    // NOTE: 1 simdgroup per threadgroup (32 threads), execution scope = 1
    // simdgroup.  Using 4 simdgroups with execution_simdgroups<1> leaves the
    // coop destination tile layout implementation-defined and the compiler
    // can collapse the (C, QPSG) tile (verified: n dim shrinks, results wrong).
    constexpr short NLANES = 1;                    // simdgroups per threadgroup
    // NOTE: 16 queries per threadgroup (QPSG = 16) is 2x faster in isolated
    // microbenchmarks (K/V bandwidth bound per query: 12.6 -> 24.8 TFLOPS),
    // but in the full kernel the larger tiles (P: (C,16), O: (PVM,16)) spill
    // and it is ~30% SLOWER (8.6 -> 6.1 TFLOPS).  8 queries is the sweet spot.
    constexpr short QPSG   = 8;                   // queries per threadgroup
    constexpr short C      = OP_FLASH_ATTN_EXT_VEC_NCPSG; // 32 kv items per chunk
    // (32-wide chunks: smaller K/V operand and P tiles fit better in registers
    // than 64-wide; the pad buffer is written with matching 32-wide chunks)

    // the tensor-core destination tile caps the m dimension at 128; for
    // dv > 128 split the PV into d blocks (128-wide, or 64-wide when dv is
    // not a multiple of 128 - both verified)
    constexpr short PVM  = ((DV % 128) == 0) ? ((DV < 128) ? DV : 128) : 64;
    constexpr short NBLK = (DV + PVM - 1) / PVM;
    static_assert(DV % PVM == 0, "dv must be a multiple of the PV block size");

    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const int    iq1 = tgpig[0]*QPSG;              // 8 queries per threadgroup

    // GQA: Q head -> KV head
    const short ikv2 = iq2/(args.ne02/args.ne_12_2);
    const short ikv3 = iq3/(args.ne03/args.ne_12_3);

    // row strides in elements
    const int sq = args.nb01/4;  // Q is f32
    const int sk = args.ns10;
    const int sv = args.ns20;

    // NOTE: MPP requires non-const element types in the operand decltypes
    device float * qp = (device float *) (q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03) + (sgitg*QPSG)*sq;
    device half  * kp = (device half  *) (k + ikv2*args.nb12 + ikv3*args.nb13);
    device half  * vp = (device half  *) (v + ikv2*args.nb22 + ikv3*args.nb23);

    // ---- Q tile for this lane: (DK, QPSG) f32, strided ----
    auto tQ = tensor(qp, dextents<int, 2>(DK, QPSG), array<int, 2>{ 1, sq });

    // ---- matmul ops (execution scope: 1 simdgroup = this lane) ----
    // QK^T: P (C, QPSG) = K^T (DK, C) x Q (DK, QPSG)
    // NOTE: k must be dynamic_extent (k is taken from the left operand extent);
    // using a compile-time constant here breaks the coop destination tile layout
    constexpr auto desc_qk = matmul2d_descriptor(
            C, QPSG, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> mm_qk;

    // PV: O (PVM, QPSG) += V^T (PVM, C) x P (C, QPSG)
    constexpr auto desc_pvb = matmul2d_descriptor(
            PVM, QPSG, C, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pvb, execution_simdgroup> mm_pvb;

    auto tK0 = tensor(kp, dextents<int, 2>(DK, C), array<int, 2>{ 1, sk });
    auto tV0 = tensor(vp, dextents<int, 2>(PVM, C), array<int, 2>{ 1, sv });

    auto cT_qk0 = mm_qk.template get_destination_cooperative_tensor<decltype(tK0), decltype(tQ), float>();

    // ---- shared memory per lane: partial max/sum per (query, thread) + M/S/alpha ----
    constexpr short SHM_FLOATS = NLANES*(QPSG*32 + QPSG*32 + 3*QPSG);
    threadgroup float * sh_qmax  = (threadgroup float *) shmem_f16;
    threadgroup float * sh_qsum  = sh_qmax  + NLANES*QPSG*32;
    threadgroup float * sh_M     = sh_qsum  + NLANES*QPSG*32;
    threadgroup float * sh_S     = sh_M     + NLANES*QPSG;
    threadgroup float * sh_alpha = sh_S     + NLANES*QPSG;

    (void) SHM_FLOATS;

    const int jb  = sgitg*QPSG;          // query offset within lane (sgitg == 0)
    const int jqb = sgitg*QPSG*32;       // (query, thread) offset (sgitg == 0)
    const int t   = tiitg % 32;          // thread within the (only) simdgroup

    const int kv      = args.ne11;
    const int nchunks = (kv + C - 1)/C;
    const bool has_kvpad = kv % C != 0;

    // mask: (kv, nq, ne32, ne33) f16, kv contiguous; ne31 == 0 means no mask
    const bool has_mask  = args.ne31 != 0;
    const bool has_sinks = (args.flags & 1u) != 0;
    const bool has_scap  = args.logit_softcap != 0.0f;

    // ALiBi: slope per Q head (the mask holds the position distances)
    float slope = 1.0f;
    if (args.max_bias != 0.0f) {
        const int h = iq2;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
        slope = pow(base, (float) exph);
    }
    const float mscale = args.max_bias != 0.0f ? slope : 1.0f;

    // mask head/batch offset (bytes); the query offset is applied per element
    const uint64_t mbase = (uint64_t) (iq2 % args.ne32) * args.nb32 + (uint64_t) (iq3 % args.ne33) * args.nb33;

    // pad buffer layout (kernel_flash_attn_ext_pad): the last C kv items per
    // (KV head, batch), K then V, zero-padded beyond kv; the mask section is
    // (nq, C) f16 per (Q head, batch) with -MAXHALF beyond kv
    const uint64_t pad_k_offs = (uint64_t) (ikv2 + ikv3*args.ne_12_2) * args.nb11*C;
    const uint64_t pad_v_offs = (uint64_t) args.nb11*C*args.ne_12_2*args.ne_12_3 +
                                (uint64_t) (ikv2 + ikv3*args.ne_12_2) * args.nb21*C;
    const uint64_t pad_mask_offs = (uint64_t) (args.nb11 + args.nb21) * C * args.ne_12_2 * args.ne_12_3 +
                                   2u * C * args.ne31 * ((uint64_t) (iq2 % args.ne32) + (uint64_t) (iq3 % args.ne33) * args.ne32);

    auto make_cT_pv = [&]() {
        auto tVb = tensor(vp, dextents<int, 2>(PVM, C), array<int, 2>{ 1, sv });
        return mm_pvb.template get_destination_cooperative_tensor<decltype(tVb), decltype(cT_qk0), float>();
    };

    if constexpr (NBLK == 2) {
        // hoisted form: QK^T + online softmax run once per chunk; the P tile
        // (f32, in registers) is reused for both PV d blocks.  (The per-block
        // form below recomputes QK^T + softmax per d block: NBLKx the QK work.)
        auto cT_pv0 = make_cT_pv();
        auto cT_pv1 = make_cT_pv();

        {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < cT_pv0.get_capacity(); ++i) {
                if (cT_pv0.is_valid_element(i)) { cT_pv0[i] = 0.0f; }
            }
        }
        {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < cT_pv1.get_capacity(); ++i) {
                if (cT_pv1.is_valid_element(i)) { cT_pv1[i] = 0.0f; }
            }
        }
        // online-softmax state in PER-THREAD REGISTERS: the chunk reductions
        // use simd_max/simd_sum (warp-level), so the state is uniform across
        // the simdgroup and NO shared memory / barriers are needed.  Barriers
        // drain the tensor-core pipeline between the QK^T and PV matmuls
        // (microbench on this GPU: ~10.5 TFLOPS with the smem + 4-barrier
        // reduction vs ~12.6 without).
        // the running max may be ANY constant >= max score seen so far: it
        // cancels in O/S.  Use ONE GLOBAL max (scalar) for the O rescale so
        // the per-chunk rescale is a trivial scalar multiply (no per-element
        // index decode / per-query alpha array).  Per-row maxes are tracked
        // only for the sinks correction at the end.
        float M = -FLT_MAX / 2;
        float S[QPSG];
        for (int j = 0; j < QPSG; ++j) {
            S[j] = 0.0f;
        }
        float alpha = 1.0f; // exp(M_old - M_new), uniform scalar

        // ---- chunk interleave: QK^T of chunk ic+1 is issued before PV of
        // chunk ic, so the two tensor-core matmuls pipeline.  (Serial
        // per-chunk QK->PV starves the tensor core between matmuls;
        // microbench on this GPU: serial 12.5 vs interleaved 22.2 TFLOPS.)
        // Two P tiles alternate (coop tiles cannot be swapped by assignment);
        // the QK^T destination type is chunk-independent, so both tiles share
        // the chunk-0 operand decltypes.
        auto tK0c = tensor(kp, dextents<int, 2>(DK, C), array<int, 2>{ 1, sk });
        auto cT_qkA = mm_qk.template get_destination_cooperative_tensor<decltype(tK0c), decltype(tQ), float>();
        auto cT_qkB = mm_qk.template get_destination_cooperative_tensor<decltype(tK0c), decltype(tQ), float>();
        using cT_qk_t = decltype(cT_qkA);

        // ---- prologue: QK^T(0) + score + online-softmax reduction -> cT_qkA ----
        {
            const int kc = min((int) C, kv);
            const bool use_pad = has_kvpad && C > kv;
            device half * kp_c = use_pad ? (device half *) (pad + pad_k_offs) : kp;

            // mask row base for this chunk: real mask (kv contiguous, query
            // stride nb31) or the pad section (C per query, indexed from 0)
            device const half * mp = nullptr;
            int mstride = 0; // in halfs, per local query j
            if (has_mask) {
                if (use_pad) {
                    mp      = (device const half *) (pad + pad_mask_offs) + (iq1 + sgitg*QPSG) * (int) C;
                    mstride = C;
                } else {
                    // global query index: iq1 + sgitg*QPSG + j (j is local, 0..QPSG-1)
                    mp      = (device const half *) (mask + mbase) + (uint64_t) (iq1 + sgitg*QPSG) * (args.nb31 / 2);
                    mstride = (int) (args.nb31 / 2);
                }
            }

            auto tK = tensor(kp_c, dextents<int, 2>(DK, C), array<int, 2>{ 1, sk });
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qkA.get_capacity(); ++i) {
                    if (cT_qkA.is_valid_element(i)) { cT_qkA[i] = 0.0f; }
                }
            }
            mm_qk.run(tK, tQ, cT_qkA);

            // scale in registers; clobber padded kv (idx1 >= kc) to -inf
            float lmax[QPSG], lsum[QPSG];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPSG; ++j) { lmax[j] = -FLT_MAX / 2; lsum[j] = 0.0f; }
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qkA.get_capacity(); ++i) {
                    if (!cT_qkA.is_valid_element(i)) { continue; }
                    auto idx = cT_qkA.get_multidimensional_index(i);
                    const int q = (int) idx[0];
                    float s;
                    if ((int) idx[1] >= kc) {
                        s = -FLT_MAX / 2;
                    } else {
                        s = cT_qkA[i]*args.scale;
                        if (has_scap) { s = args.logit_softcap * tanh(s); }
                        if (has_mask) { s += (float) mp[(uint) q * mstride + (uint) idx[1]] * mscale; }
                    }
                    cT_qkA[i] = s;
                    if (s > lmax[q]) { lmax[q] = s; }
                }
            }

            // warp-level reduction: ONE global max (scalar) + per-row sums.
            // (The max cancels in O/S; a global max keeps the per-chunk O
            // rescale a trivial scalar multiply.)
            float lmax_g = -FLT_MAX / 2;
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPSG; ++j) {
                lmax_g = max(lmax_g, simd_max(lmax[j]));
            }
            {
                const float m_old = M;
                M = max(M, lmax_g);
                alpha = exp(m_old - M);
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    S[j] = S[j]*alpha;
                }
            }

            // ---- exp in registers, partial sums ----
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qkA.get_capacity(); ++i) {
                    if (!cT_qkA.is_valid_element(i)) { continue; }
                    auto idx = cT_qkA.get_multidimensional_index(i);
                    const int q = (int) idx[0];
                    const float p = exp(cT_qkA[i] - M);
                    cT_qkA[i] = p; // P in registers (f32), later used as PV right input
                    lsum[q] += p;
                }
            }
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPSG; ++j) {
                S[j] += simd_sum(lsum[j]);
            }
        }

        for (int ic = 0; ic < nchunks; ++ic) {
            // ---- current-chunk context (PV only) ----
            const int k0 = ic*C;
            const bool use_pad = has_kvpad && k0 + C > kv;
            device half * vp_c = use_pad ? (device half *) (pad + pad_v_offs) : vp + (uint) k0*sv;
            auto tV0 = tensor(vp_c, dextents<int, 2>(PVM, C), array<int, 2>{ 1, sv });
            auto tV1 = tensor(vp_c + PVM, dextents<int, 2>(PVM, C), array<int, 2>{ 1, sv });

            // ---- rescale both O accumulators (per element: query = idx0) ----
            // reads alpha_ic (written by the reduction of chunk ic); skip the
            // first chunk: alpha_0 = exp(-FLT_MAX/2 - m_new) = 0
            // rescale only when the running max actually changed (alpha < 1):
            // with random/causal scores M stabilizes after a few chunks and
            // the pass is skipped for the rest (uniform scalar branch - no
            // divergence).  The pass itself does not overlap with the tensor
            // core (it serializes between QK^T and PV), so skipping matters.
            if (ic > 0 && alpha != 1.0f) {
                // scalar alpha (global max): a trivial multiply, no index decode
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv0.get_capacity(); ++i) {
                    if (cT_pv0.is_valid_element(i)) { cT_pv0[i] *= alpha; }
                }
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv1.get_capacity(); ++i) {
                    if (cT_pv1.is_valid_element(i)) { cT_pv1[i] *= alpha; }
                }
            }

            // ---- early QK^T(ic+1) + score + reduction -> the other P tile ----
            // issued before PV(ic) so the two matmuls pipeline on the tensor core
            if (ic + 1 < nchunks) {
                const int k1 = (ic + 1)*C;
                const int kc = min((int) C, kv - k1);
                const bool use_pad1 = has_kvpad && k1 + C > kv;
                device half * kp_c = use_pad1 ? (device half *) (pad + pad_k_offs) : kp + (uint) k1*sk;

                device const half * mp = nullptr;
                int mstride = 0; // in halfs, per local query j
                if (has_mask) {
                    if (use_pad1) {
                        mp      = (device const half *) (pad + pad_mask_offs) + (iq1 + sgitg*QPSG) * (int) C;
                        mstride = C;
                    } else {
                        // global query index: iq1 + sgitg*QPSG + j (j is local, 0..QPSG-1)
                        mp      = (device const half *) (mask + (uint64_t) k1 * 2 + mbase)
                                + (uint64_t) (iq1 + sgitg*QPSG) * (args.nb31 / 2);
                        mstride = (int) (args.nb31 / 2);
                    }
                }

                thread cT_qk_t & cT_qkN = (ic & 1) ? cT_qkA : cT_qkB;
                auto tK = tensor(kp_c, dextents<int, 2>(DK, C), array<int, 2>{ 1, sk });
                {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < cT_qkN.get_capacity(); ++i) {
                        if (cT_qkN.is_valid_element(i)) { cT_qkN[i] = 0.0f; }
                    }
                }
                mm_qk.run(tK, tQ, cT_qkN);

                // scale in registers; clobber padded kv (idx1 >= kc) to -inf
                float lmax[QPSG], lsum[QPSG];
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) { lmax[j] = -FLT_MAX / 2; lsum[j] = 0.0f; }
                {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < cT_qkN.get_capacity(); ++i) {
                        if (!cT_qkN.is_valid_element(i)) { continue; }
                        auto idx = cT_qkN.get_multidimensional_index(i);
                        const int q = (int) idx[0];
                        float s;
                        if ((int) idx[1] >= kc) {
                            s = -FLT_MAX / 2;
                        } else {
                            s = cT_qkN[i]*args.scale;
                            if (has_scap) { s = args.logit_softcap * tanh(s); }
                            if (has_mask) { s += (float) mp[(uint) q * mstride + (uint) idx[1]] * mscale; }
                        }
                        cT_qkN[i] = s;
                        if (s > lmax[q]) { lmax[q] = s; }
                    }
                }

                // warp-level reduction: one global max (scalar) + per-row sums
                float lmax_g = -FLT_MAX / 2;
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    lmax_g = max(lmax_g, simd_max(lmax[j]));
                }
                {
                    const float m_old = M;
                    M = max(M, lmax_g);
                    alpha = exp(m_old - M);
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < QPSG; ++j) {
                        S[j] = S[j]*alpha;
                    }
                }

                // ---- exp in registers, partial sums ----
                {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < cT_qkN.get_capacity(); ++i) {
                        if (!cT_qkN.is_valid_element(i)) { continue; }
                        auto idx = cT_qkN.get_multidimensional_index(i);
                        const int q = (int) idx[0];
                        const float p = exp(cT_qkN[i] - M);
                        cT_qkN[i] = p; // P in registers (f32), later used as PV right input
                        lsum[q] += p;
                    }
                }
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    S[j] += simd_sum(lsum[j]);
                }
            }

            // ---- PV(ic): P(ic) (in the current P tile) as right input, both d blocks ----
            {
                thread cT_qk_t & cT_qkC = (ic & 1) ? cT_qkB : cT_qkA;
                auto cT_pr = mm_pvb.template get_right_input_cooperative_tensor<half, float, float>(cT_qkC);
                mm_pvb.run(tV0, cT_pr, cT_pv0);
                mm_pvb.run(tV1, cT_pr, cT_pv1);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- final: sinks, O /= S, element-wise write for both d blocks ----
        {
            // sinks: a virtual sink position with score sinks[iq2] and no O
            // contribution.  O and S are both relative to the final global
            // max M, so the correction uses M: M' = max(M, sink),
            // S' = S*exp(M - M') + exp(sink - M'), O' = O*exp(M - M').
            // (M' is uniform across queries: the sink is per head, not per
            // query, but the per-query loop keeps the write form simple.)
            float alpha_final[QPSG];
            float s_sink = 0.0f;
            if (has_sinks) {
                s_sink = ((device const float *) sinks)[iq2];
            }
            const float m2 = has_sinks ? max(M, s_sink) : M;
            const float a_final = has_sinks ? exp(M - m2) : 1.0f;
            for (int j = 0; j < QPSG; ++j) {
                alpha_final[j] = a_final;
                if (has_sinks) {
                    S[j] = S[j]*a_final + exp(s_sink - m2);
                }
            }

            // output layout is (DV, heads, batch, batch3) with DV innermost:
            //   offset = (iq3*ne2*ne1 + iq2 + (iq1 + sgitg*QPSG + j)*ne1)*DV + d
            // NOTE: the write must use ELEMENT pointer arithmetic on a float*
            // (cast dst first, then add element offsets).  Doing the offset in
            // BYTES (char* + element*DV) corrupts the coop tile register layout
            // of the matmul ops above (verified: QK^T (C,QPSG) tile collapses).
            device float * op = (device float *) dst +
                    (uint64_t) (iq3*args.ne2*args.ne1 + iq2 + (iq1 + sgitg*QPSG)*args.ne1) * DV;
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv0.get_capacity(); ++i) {
                    if (!cT_pv0.is_valid_element(i)) { continue; }
                    auto idx = cT_pv0.get_multidimensional_index(i);
                    const int j = (int) idx[0]; // query
                    const int d = (int) idx[1]; // head dim (block 0)
                    if (j >= 0 && j < QPSG && d >= 0 && d < (int) PVM && iq1 + j < args.ne01) {
                        const float s = S[j];
                        op[(uint64_t) j*args.ne1*DV + (uint) d] = (s == 0.0f) ? 0.0f : cT_pv0[i]*alpha_final[j]/s;
                    }
                }
            }
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv1.get_capacity(); ++i) {
                    if (!cT_pv1.is_valid_element(i)) { continue; }
                    auto idx = cT_pv1.get_multidimensional_index(i);
                    const int j = (int) idx[0]; // query
                    const int d = PVM + (int) idx[1]; // head dim (block 1)
                    if (j >= 0 && j < QPSG && d >= 0 && d < (int) DV && iq1 + j < args.ne01) {
                        const float s = S[j];
                        op[(uint64_t) j*args.ne1*DV + (uint) d] = (s == 0.0f) ? 0.0f : cT_pv1[i]*alpha_final[j]/s;
                    }
                }
            }

        }
    } else {
        // NBLK != 2 (NBLK == 1, or NBLK == 4 for dv = 512): per-d-block form.
        // For NBLK == 4, carrying 4 live PV destination tiles would overflow
        // the register file, so QK^T + softmax are recomputed per d block
        // (correct, but NBLKx the QK work).
        for (int b = 0; b < NBLK; ++b) {
            auto cT_pv = make_cT_pv();

            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv.get_capacity(); ++i) {
                    if (cT_pv.is_valid_element(i)) { cT_pv[i] = 0.0f; }
                }
            }
        for (int j = t; j < QPSG; j += 32) {
            sh_M[jb + j]      = -FLT_MAX / 2;
            sh_S[jb + j]      = 0.0f;
            sh_alpha[jb + j]  = 1.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (int ic = 0; ic < nchunks; ++ic) {
            const int k0 = ic*C;
            const int kc = min((int) C, kv - k0);

            // reset this thread's (q, t) column
            {
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    sh_qmax[jqb + j*32 + t] = -FLT_MAX / 2;
                    sh_qsum[jqb + j*32 + t] = 0.0f;
                }
            }

            // the last partial chunk is read from the pad buffer (padded with 0);
            // the pad buffer holds only the last C kv items per (KV head, batch),
            // so its rows are indexed from 0 (not k0)
            const bool use_pad = has_kvpad && k0 + C > kv;
            device half * kp_c = use_pad ? (device half *) (pad + pad_k_offs) : kp + (uint) k0*sk;
            device half * vp_c = use_pad ? (device half *) (pad + pad_v_offs) : vp + (uint) k0*sv;

            // mask row base for this chunk: real mask (kv contiguous, query
            // stride nb31) or the pad section (C per query, indexed from 0)
            device const half * mp = nullptr;
            int mstride = 0; // in halfs, per local query j
            if (has_mask) {
                if (use_pad) {
                    mp      = (device const half *) (pad + pad_mask_offs) + (iq1 + sgitg*QPSG) * (int) C;
                    mstride = C;
                } else {
                    // global query index: iq1 + sgitg*QPSG + j (j is local, 0..QPSG-1)
                    mp      = (device const half *) (mask + (uint64_t) k0 * 2 + mbase)
                            + (uint64_t) (iq1 + sgitg*QPSG) * (args.nb31 / 2);
                    mstride = (int) (args.nb31 / 2);
                }
            }

            auto tK  = tensor(kp_c, dextents<int, 2>(DK, C), array<int, 2>{ 1, sk });
            // block b covers d = b*PVM .. b*PVM + PVM - 1 (d is the innermost dim of V)
            auto tVb = tensor(vp_c + (uint) b*PVM, dextents<int, 2>(PVM, C), array<int, 2>{ 1, sv });

            // ---- QK^T ----
            auto cT_qk = mm_qk.template get_destination_cooperative_tensor<decltype(tK), decltype(tQ), float>();
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qk.get_capacity(); ++i) {
                    if (cT_qk.is_valid_element(i)) { cT_qk[i] = 0.0f; }
                }
            }
            mm_qk.run(tK, tQ, cT_qk);

            // scale in registers; clobber padded kv (idx1 >= kc) to -inf
            float lmax[QPSG], lsum[QPSG];
            #pragma clang loop unroll(full)
            for (int j = 0; j < QPSG; ++j) { lmax[j] = -FLT_MAX / 2; lsum[j] = 0.0f; }
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qk.get_capacity(); ++i) {
                    if (!cT_qk.is_valid_element(i)) { continue; }
                    auto idx = cT_qk.get_multidimensional_index(i);
                    const int q = (int) idx[0];
                    float s;
                    if ((int) idx[1] >= kc) {
                        s = -FLT_MAX / 2;
                    } else {
                        s = cT_qk[i]*args.scale;
                        if (has_scap) { s = args.logit_softcap * tanh(s); }
                        if (has_mask) { s += (float) mp[(uint) q * mstride + (uint) idx[1]] * mscale; }
                    }
                    cT_qk[i] = s;
                    if (s > lmax[q]) { lmax[q] = s; }
                }
            }

            // partial max -> shared (non-owned queries hold -FLT_MAX/2 -> harmless)
            {
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    sh_qmax[jqb + j*32 + t] = lmax[j];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // M / alpha update: thread t (< QPSG) finalizes query t
            if (t < QPSG) {
                const int j = t;
                float m_new = sh_M[jb + j];
                #pragma clang loop unroll(full)
                for (int tt = 0; tt < 32; ++tt) {
                    m_new = max(m_new, sh_qmax[jqb + j*32 + tt]);
                }
                sh_alpha[jb + j] = exp(sh_M[jb + j] - m_new);
                sh_M[jb + j]     = m_new;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // ---- exp in registers, partial sums ----
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_qk.get_capacity(); ++i) {
                    if (!cT_qk.is_valid_element(i)) { continue; }
                    auto idx = cT_qk.get_multidimensional_index(i);
                    const int q = (int) idx[0];
                    const float p = exp(cT_qk[i] - sh_M[jb + q]);
                    cT_qk[i] = p; // P in registers (f32), later used as PV right input
                    lsum[q] += p;
                }
            }
            {
                #pragma clang loop unroll(full)
                for (int j = 0; j < QPSG; ++j) {
                    sh_qsum[jqb + j*32 + t] = lsum[j];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // S update: thread t (< QPSG) finalizes query t
            if (t < QPSG) {
                const int j = t;
                float s_new = 0.0f;
                #pragma clang loop unroll(full)
                for (int tt = 0; tt < 32; ++tt) {
                    s_new += sh_qsum[jqb + j*32 + tt];
                }
                sh_S[jb + j] = sh_S[jb + j]*sh_alpha[jb + j] + s_new;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // ---- rescale O accumulator (per element: query = idx0) ----
            // NOTE: skip the first chunk: alpha_1 = exp(-FLT_MAX/2 - m_new) = 0
            if (ic > 0) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv.get_capacity(); ++i) {
                    if (!cT_pv.is_valid_element(i)) { continue; }
                    auto idx = cT_pv.get_multidimensional_index(i);
                    cT_pv[i] *= sh_alpha[jb + (int) idx[0]];
                }
            }

            // ---- PV: P (in registers) as right input ----
            {
                auto cT_pr = mm_pvb.template get_right_input_cooperative_tensor<half, float, float>(cT_qk);
                mm_pvb.run(tVb, cT_pr, cT_pv);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- final: sinks, O /= S, element-wise write for this d block ----
        {
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // sinks: a virtual sink position with score sinks[iq2] and no O
            // contribution: M' = max(M, sink), S' = S*exp(M - M') + exp(sink - M'),
            // O' = O*exp(M - M'). The per-query O factor is stashed in sh_alpha
            // (reused; the last-chunk alpha is not needed at this point).
            if (has_sinks) {
                const float s_sink = ((device const float *) sinks)[iq2];
                for (int j = t; j < QPSG; j += 32) {
                    const float m  = sh_M[jb + j];
                    const float m2 = max(m, s_sink);
                    sh_alpha[jb + j] = exp(m - m2);
                    sh_S[jb + j]     = sh_S[jb + j]*sh_alpha[jb + j] + exp(s_sink - m2);
                }
            } else {
                for (int j = t; j < QPSG; j += 32) {
                    sh_alpha[jb + j] = 1.0f;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // output layout is (DV, heads, batch, batch3) with DV innermost:
            //   offset = (iq3*ne2*ne1 + iq2 + (iq1 + sgitg*QPSG + j)*ne1)*DV + d
            // (this lane handles queries iq1 + sgitg*QPSG .. + QPSG-1)
            // NOTE: the write must use ELEMENT pointer arithmetic on a float*
            // (cast dst first, then add element offsets).  Doing the offset in
            // BYTES (char* + element*DV) corrupts the coop tile register layout
            // of the matmul ops above (verified: QK^T (C,QPSG) tile collapses).
            device float * op = (device float *) dst +
                    (uint64_t) (iq3*args.ne2*args.ne1 + iq2 + (iq1 + sgitg*QPSG)*args.ne1) * DV;
            {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < cT_pv.get_capacity(); ++i) {
                    if (!cT_pv.is_valid_element(i)) { continue; }
                    auto idx = cT_pv.get_multidimensional_index(i);
                    const int j = (int) idx[0]; // query
                    const int d = b*PVM + (int) idx[1]; // head dim
                    if (j >= 0 && j < QPSG && d >= 0 && d < (int) DV && iq1 + j < args.ne01) {
                        const float s = sh_S[jb + j];
                        op[(uint64_t) j*args.ne1*DV + (uint) d] = (s == 0.0f) ? 0.0f : cT_pv[i]*sh_alpha[jb + j]/s;
                    }
                }
            }

        }
        }
    }
}

template <short DK, short DV>
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    (void) blk;

    kernel_flash_attn_ext_tensor_impl<DK, DV>(args, q, k, v, mask, sinks, pad, dst, shmem_f16, tgpig, tiitg, sgitg);
}

// head size instantiations (f16 K/V only - see the host-side fast-path gate)
template [[host_name("kernel_flash_attn_ext_tensor_dk64_dv64"  )]]  kernel void kernel_flash_attn_ext_tensor<  64,  64>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk128_dv128")]]  kernel void kernel_flash_attn_ext_tensor< 128, 128>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk192_dv128")]]  kernel void kernel_flash_attn_ext_tensor< 192, 128>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk192_dv192")]]  kernel void kernel_flash_attn_ext_tensor< 192, 192>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk256_dv256")]]  kernel void kernel_flash_attn_ext_tensor< 256, 256>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk320_dv256")]]  kernel void kernel_flash_attn_ext_tensor< 320, 256>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk512_dv512")]]  kernel void kernel_flash_attn_ext_tensor< 512, 512>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
template [[host_name("kernel_flash_attn_ext_tensor_dk576_dv512")]]  kernel void kernel_flash_attn_ext_tensor< 576, 512>(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q, device const char * k, device const char * v, device const char * mask,
        device const char * sinks, device const char * pad, device const char * blk,
        device       char * dst, threadgroup half * shmem_f16, uint3 tgpig, ushort tiitg, ushort sgitg);
#endif // GGML_METAL_HAS_TENSOR

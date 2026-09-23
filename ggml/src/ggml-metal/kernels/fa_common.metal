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
constant int32_t FC_flash_attn_ext_nsg   [[function_constant(FC_FLASH_ATTN_EXT + 22)]];
constant int32_t FC_flash_attn_ext_nqptg [[function_constant(FC_FLASH_ATTN_EXT + 23)]];

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
    constexpr short NQB = Q/8; // 8x8 query blocks per K/V load
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    static_assert(Q % 8 == 0, "");
    static_assert(NQB == 1 || (is_same<kd4x4_t, k4x4_t>::value && is_same<vd4x4_t, v4x4_t>::value), "the quantized K/V branches handle 8 queries only");

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
                    qk8x8_t mqk[NQB];

                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                        mqk[qb] = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);
                    }

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);

                            FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                simdgroup_load(mq, pq + qb*(8*DK) + 8*i, DK);

                                simdgroup_barrier(mem_flags::mem_none);

                                simdgroup_multiply_accumulate(mqk[qb], mq, mk, mqk[qb]);
                            }
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        // note: too much unroll can tank the performance for large heads
                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                simdgroup_load(mq[0], pq + qb*(8*DK) + 0*8 + 16*i, DK);
                                simdgroup_load(mq[1], pq + qb*(8*DK) + 1*8 + 16*i, DK);

                                simdgroup_barrier(mem_flags::mem_none);

                                simdgroup_multiply_accumulate(mqk[qb], mq[0], mk[0], mqk[qb]);
                                simdgroup_multiply_accumulate(mqk[qb], mq[1], mk[1], mqk[qb]);
                            }
                        }
                    }

                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                        simdgroup_store(mqk[qb], ps + qb*(8*SH), SH, 0, false);
                    }

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

                    o8x8_t lo[NQB][NO];

                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                        auto sot = so + qb*(8*PV) + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[qb][ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs[NQB];

                                FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                    simdgroup_load(vs[qb], ss + qb*(8*SH) + 8*cc, SH, 0, false);
                                }

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 0], vs[qb], mv[0], lo[qb][2*ii + 0]);
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 1], vs[qb], mv[1], lo[qb][2*ii + 1]);
                                    }
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[NQB][2];

                                FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                    simdgroup_load(vs[qb][0], ss + qb*(8*SH) + 16*cc + 0, SH, 0, false);
                                    simdgroup_load(vs[qb][1], ss + qb*(8*SH) + 16*cc + 8, SH, 0, false);
                                }

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 0], vs[qb][0], mv[0], lo[qb][2*ii + 0]);
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 1], vs[qb][0], mv[1], lo[qb][2*ii + 1]);
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 0], vs[qb][1], mv[2], lo[qb][2*ii + 0]);
                                        simdgroup_multiply_accumulate(lo[qb][2*ii + 1], vs[qb][1], mv[3], lo[qb][2*ii + 1]);
                                    }
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    FOR_UNROLL (short qb = 0; qb < NQB; ++qb) {
                        auto sot = so + qb*(8*PV) + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[qb][ii], sot, PV, 0, false);

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
#define FWD_TMPL(nq) q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, nq, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    // note: the wide tile is instantiated for F16 K/V only to reduce library load time
    constexpr short QW = is_same<kd4x4_t, half4x4>::value ? 2*Q : Q;
#define FWD_IMPL(nsg) \
    if (FC_flash_attn_ext_nqptg == QW) { \
        kernel_flash_attn_ext_impl<FWD_TMPL(QW), nsg>(FWD_ARGS); \
    } else { \
        kernel_flash_attn_ext_impl<FWD_TMPL(Q), nsg>(FWD_ARGS); \
    }
    switch (FC_flash_attn_ext_nsg) {
      // note: disabled cases to reduce library load time
      //case 1: kernel_flash_attn_ext_impl<FWD_TMPL(Q), 1>(FWD_ARGS); break;
      //case 2: kernel_flash_attn_ext_impl<FWD_TMPL(Q), 2>(FWD_ARGS); break;
        case 4: FWD_IMPL(4); break;
        case 8: FWD_IMPL(8); break;
    }
#undef FWD_IMPL
#undef FWD_TMPL
#undef FWD_ARGS
}

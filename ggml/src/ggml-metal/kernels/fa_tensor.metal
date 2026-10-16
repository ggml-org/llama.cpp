#include "common.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename T,                                 // Q, K, V and P types
    typename T4,
    short DK,                                   // K head size
    short DV,                                   // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_NQPSG, // queries per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG, // cache items per threadgroup
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup T     * sq = (threadgroup T     *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup T     * sp = (threadgroup T     *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }

        ((threadgroup T4 *) sq)[i] = (T4) (q4*qscale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = (device const half *) (mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int nblk1 = (args.ne01 + Q - 1)/Q;
        const int nblk0 = (args.ne11 + C - 1)/C;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    float slope = 1.0f;

    // ALiBi
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exph);
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device T *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the host guarantees ne11 % C == 0
    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += C) {
        char blk_cur = 1;

        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];

            if (blk_cur == 0) {
                continue;
            }
        }

        // Q*K^T
        {
            auto tk = tensor((device T *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = args.logit_softcap*precise::tanh(s[ii]*args.scale);
                }
            }

            if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += slope*(float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const T p = (T) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device T *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        // the sink only adds to the denominator - its rescale of O is folded into the final scale
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks) {
            const float s = ((device const float *) sinks)[iq2];
            const float m = max(M[jj], s);

            ms = exp(M[jj] - m);

            S[jj] = S[jj]*ms + exp(s - m);
        }

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : ms/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past ne01 are clipped by the tensor extents
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2 + (uint64_t) iq1*args.ne1)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1*DV}));

    co.store(td);
}

typedef decltype(kernel_flash_attn_ext_tensor<half, half4, 64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

constant int32_t FC_flash_attn_ext_tensor_gqa_nwg [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 23)]];

// small batches with GQA: up to Q rows of (token, query head) that share a KV head are packed into one threadgroup, so each K/V block is read once for them
// the partial results of the NWG KV splits are merged by kernel_flash_attn_ext_vec_reduce
template<
    short DK,                                       // K head size
    short DV,                                       // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_GQA_NQPSG, // packed rows per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG,     // cache items per iteration
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_GQA_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor_gqa(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int NWG = FC_flash_attn_ext_tensor_gqa_nwg;

    // packed row r -> token r/ng, head ikv*ng + r%ng
    const int ng = args.ne02/args.ne_12_2;
    const int nr = args.ne01*ng;

    const int ir0 = tgpig[0]*Q;
    const int ikv = tgpig[1];
    const int iwg = tgpig[2];

    const short tiitg = sgitg*NW + tiisg;

    threadgroup half  * sq = (threadgroup half  *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    k += ikv*args.nb12;
    v += ikv*args.nb22;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int r = ir0 + i/(DK/4);

        float4 q4 = 0.0f;
        if (r < nr) {
            q4 = ((device const float4 *) (q + (r/ng)*args.nb01 + (ikv*ng + r%ng)*args.nb02))[i%(DK/4)];
        }

        ((threadgroup half4 *) sq)[i] = (half4) (q4*args.scale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const int r = min(ir0 + jj*NSG + sgitg, nr - 1);

        pm[jj] = (device const half *) (mask + (r/ng)*args.nb31 + ((ikv*ng + r%ng)%args.ne32)*args.nb32);
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // contiguous range of KV blocks of this workgroup - the host guarantees ne11 % C == 0
    const int nblk = args.ne11/C;
    const int ib0  = (iwg*nblk)/NWG;
    const int ib1  = ((iwg + 1)*nblk)/NWG;

    // tokens of the packed rows of this threadgroup (blk has one row per token and mask head)
    const int it0 = ir0/ng;
    const int it1 = (min(ir0 + Q, nr) - 1)/ng;

    for (int ic0 = ib0; ic0 < ib1; ++ic0) {
        const int ic = ic0*C;

        // skip the block if it is fully masked for all tokens of this threadgroup (checks every mask head)
        if (FC_flash_attn_ext_tensor_has_mask) {
            char blk_cur = 0;

            for (int i2 = 0; i2 < args.ne32; ++i2) {
                for (int it = it0; it <= it1; ++it) {
                    blk_cur |= blk[(i2*args.ne01 + it)*nblk + ic0];
                }
            }

            if (blk_cur == 0) {
                continue;
            }
        }

        // Q*K^T
        {
            auto tk = tensor((device half *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_mask) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += (float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const half p = (half) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write the unnormalized O, S and M in the layout of kernel_flash_attn_ext_vec_reduce
    const uint64_t nrows = (uint64_t) args.ne3*args.ne2*args.ne1;

    device float * dst_o  = (device float *) dst;
    device float * dst_sm = (device float *) dst + nrows*DV*NWG;

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const int r = ir0 + jj*NSG + sgitg;

        if (r < nr && tiisg == 0) {
            const uint64_t rid = (uint64_t) (r/ng)*args.ne1 + ikv*ng + r%ng;

            dst_sm[rid*(2*NWG) + 2*iwg + 0] = S[jj];
            dst_sm[rid*(2*NWG) + 2*iwg + 1] = M[jj];
        }
    }

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            const auto idx = co.get_multidimensional_index(i);

            const int r = ir0 + idx[1];

            if (r < nr) {
                const uint64_t rid = (uint64_t) (r/ng)*args.ne1 + ikv*ng + r%ng;

                dst_o[rid*DV*NWG + (NWG*(idx[0]/4) + iwg)*4 + idx[0]%4] = co[i];
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_tensor_gqa<128, 128>) flash_attn_ext_tensor_gqa_t;

template [[host_name("kernel_flash_attn_ext_tensor_gqa_f16_dk128_dv128")]] kernel flash_attn_ext_tensor_gqa_t kernel_flash_attn_ext_tensor_gqa<128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_gqa_f16_dk256_dv256")]] kernel flash_attn_ext_tensor_gqa_t kernel_flash_attn_ext_tensor_gqa<256, 256>;

#endif // GGML_METAL_HAS_TENSOR

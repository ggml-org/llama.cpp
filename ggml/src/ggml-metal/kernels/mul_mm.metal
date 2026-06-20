#include "common.h"
#include "dequantize.h"

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, N_MM_NK_TOTAL, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);

        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // cT.store handles bounds checking via tD's extents (M, N)
    device float * dstBatch = (device float *)dst + im * N * M;

    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    cT.store(tD.slice(ra, rb));
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
#else
    auto tA = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
#else
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mm.run(sB, sA, cT);
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
#else
    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;

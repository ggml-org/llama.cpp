#include "common.h"
#include "dequantize.h"

constant bool    FC_mul_mm_bc_inp       [[function_constant(FC_MUL_MM + 0)]];
constant bool    FC_mul_mm_bc_out       [[function_constant(FC_MUL_MM + 1)]];
constant short   FC_mul_mm_ne12         [[function_constant(FC_MUL_MM + 2)]];
constant short   FC_mul_mm_ne13         [[function_constant(FC_MUL_MM + 3)]];
constant short   FC_mul_mm_r2           [[function_constant(FC_MUL_MM + 4)]];
constant short   FC_mul_mm_r3           [[function_constant(FC_MUL_MM + 5)]];
constant int32_t FC_mul_mm_id_map0_ne02 [[function_constant(FC_MUL_MM + 7)]];

// each block_q contains 16*nl weights

template<
    typename S0, typename S1,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &),
    typename T0, typename T1>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T0_4x4 = typename value_type_4x4<T0>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));

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

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
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
            (uint64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (uint64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
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
                device float  * D  = (device float  *) dst + r0 + (uint64_t) (r1 + j)*args.ne0 + (uint64_t) im*args.ne1*args.ne0;
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

    // per expert: entry count (pass 1), replaced by the base offset of the slice (pass 2)
    threadgroup uint32_t * cnt = (threadgroup uint32_t *) (shmem);
    threadgroup uint32_t * bas = cnt + FC_mul_mm_id_map0_ne02;

    // staging of the ids of ntg tokens
    threadgroup uint16_t * sids = (threadgroup uint16_t *) (bas + FC_mul_mm_id_map0_ne02);

    // pass 1: count the entries of this expert - every occurrence of this expert within the
    // ids is an entry (duplicate ids produce multiple entries for the same token)
    uint32_t n_all = 0;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            // 0xFFFF marks a skipped slot, it never matches an expert since ne02 fits in a threadgroup
            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const int32_t id = src2_i32[i20];
                sids[tpitg*ne20 + i20] = id == -1 ? 0xFFFF : (uint16_t) id;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                n_all += sids_t[i20] == ide;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    cnt[ide] = n_all;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exclusive prefix sum over the counts: the entries of this expert start at bas[ide] -
    // every slice is padded by one entry for the uniform dummy store of pass 2 - and emit the
    // entry tile descriptors of kernel_mul_mm_id: (expert, first entry, valid rows) per tile
    if (tpitg == 0) {
        constexpr uint32_t NTILE = 32; // must match NR1 of kernel_mul_mm_id

        uint32_t base = 0;
        uint32_t tile = 0;

        device uint32_t * tpe_u32 = (device uint32_t *) (htpe);

        for (int e = 0; e < FC_mul_mm_id_map0_ne02; ++e) {
            const uint32_t n = cnt[e];

            bas[e] = base;

            for (uint32_t k = 0; k < n; k += NTILE) {
                tpe_u32[1 + 3*tile    ] = e;
                tpe_u32[1 + 3*tile + 1] = base + k;
                tpe_u32[1 + 3*tile + 2] = n - k < NTILE ? n - k : NTILE;

                tile++;
            }

            base += n + 1;
        }

        tpe_u32[0] = tile;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 2: write the entries of this expert into its slice of the compact list
    device int32_t * ids_i32 = (device int32_t *) (hids);

    uint32_t ient = bas[ide];

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                sids[tpitg*ne20 + i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            short first = -1;
            short n_ents = 0;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const short hit = sids_t[i20] == ide;

                first = hit && first < 0 ? i20 : first;
                n_ents += hit;
            }

            // one store per token for every thread, regardless of the match - conditional
            // stores of the divergent lanes are very slow - the dummy value of the tokens
            // without a match lands on the padding slot of this slice and is never read
            ids_i32[ient] = first >= 0 ? (i21 + t)*ne20 + first : 0;

            if (n_ents > 1) {
                // duplicate ids: the remaining matches of this token
                short j = 1;

                FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                    if (sids_t[i20] == ide && i20 != first) {
                        ids_i32[ient + j] = (i21 + t)*ne20 + i20;
                        j += 1;
                    }
                }
            }

            ient += n_ents;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// kernel_mul_mm_id only writes the rows that an expert owns, so the skipped rows are zeroed here
kernel void kernel_mul_mm_id_zero(
        constant ggml_metal_kargs_mul_mm_id_zero & args,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i20 = tgpig.x; // slot
    const int i21 = tgpig.y; // token

    const int32_t id = ((device const int32_t *) (src2 + i21*args.nb21))[i20];

    if (id != -1) {
        return;
    }

    device float * dst_row = (device float *) (dst + i20*args.nb1 + i21*args.nb2);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = 0.0f;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_9" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<9>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

template<typename S0, typename S1, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &), typename T0, typename T1>
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
    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T0_4x4 = typename value_type_4x4<T0>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));


    const int tile = tgpig.x; // entry tile, described by the prepass
    const int r0   = tgpig.y*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    // the prepass emits one (expert, first entry, valid rows) descriptor per tile that has
    // work - the grid is sized by a static upper bound, so the tail early exits
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = tpe_u32[1 + 3*tile];     // expert
    const int idb = tpe_u32[1 + 3*tile + 1]; // first entry of this tile in hids

    // number of entries in this tile, always <= NR1
    const short nr1 = (short) ((tpe_u32[1 + 3*tile + 2] < NR1) ? tpe_u32[1 + 3*tile + 2] : NR1);

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63

    const short il0 = (tiitg % NL0);

    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    const short iy = 8*(tiitg % NL1);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    short il = il0;

    const int id = ids_i32[idb + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const int   i12 = (id / args.ne20);

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    const bool has_hi = nr1 > NR1H;

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;

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

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
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
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[idb + j];

        const short ide = id % args.ne20;
        const int   idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
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

// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half, float4x4, 1, dequantize_f32, float, float>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]] kernel mul_mm_t kernel_mul_mm<half, half, float4x4, 1, dequantize_f32, float, float>;
template [[host_name("kernel_mul_mm_f16_f32")]] kernel mul_mm_t kernel_mul_mm<half, half, half4x4,  1, dequantize_f16, half,  float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_f32_f32_prec_bf16")]] kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, float4x4, 1, dequantize_f32, float, float>;
#endif

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_f16_f32_prec_bf16")]]     kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, half4x4,       1,     dequantize_f16,     half,  float>;
template [[host_name("kernel_mul_mm_q1_0_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_q2_0_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_0_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_1_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_q5_0_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_q5_1_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_mxfp4_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_mxfp4,   2,     dequantize_mxfp4,   float, float>;
template [[host_name("kernel_mul_mm_q2_K_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_q3_K_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_q4_K_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_q5_K_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_q6_K_f32_prec_bf16")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_prec_bf16")]] kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_prec_bf16")]]  kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_prec_bf16")]] kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq3_s_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_iq2_s_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_s_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_m_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_prec_bf16")]]  kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_prec_bf16")]]  kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_tq2_0_f32_prec_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;
#endif
template [[host_name("kernel_mul_mm_f32_f32_prec_f32")]]     kernel mul_mm_t kernel_mul_mm<float, float, float4x4,      1,     dequantize_f32,     float, float>;
template [[host_name("kernel_mul_mm_f16_f32_prec_f32")]]     kernel mul_mm_t kernel_mul_mm<float, float, half4x4,       1,     dequantize_f16,     half,  float>;
template [[host_name("kernel_mul_mm_q1_0_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_q2_0_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_0_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_1_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_q5_0_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_q5_1_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_mxfp4_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_mxfp4,   2,     dequantize_mxfp4,   float, float>;
template [[host_name("kernel_mul_mm_q2_K_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_q3_K_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_q4_K_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_q5_K_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_q6_K_f32_prec_f32")]]    kernel mul_mm_t kernel_mul_mm<float, float, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_prec_f32")]] kernel mul_mm_t kernel_mul_mm<float, float, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_prec_f32")]]  kernel mul_mm_t kernel_mul_mm<float, float, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_prec_f32")]] kernel mul_mm_t kernel_mul_mm<float, float, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq3_s_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_iq2_s_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_s_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_m_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_prec_f32")]]  kernel mul_mm_t kernel_mul_mm<float, float, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_prec_f32")]]  kernel mul_mm_t kernel_mul_mm<float, float, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_tq2_0_f32_prec_f32")]]   kernel mul_mm_t kernel_mul_mm<float, float, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32_prec_f32")]] kernel mul_mm_t kernel_mul_mm<float,  float,  bfloat4x4, 1, dequantize_bf16, bfloat, float>;
template [[host_name("kernel_mul_mm_bf16_f32")]]          kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, float>;
template [[host_name("kernel_mul_mm_bf16_bf16")]]         kernel mul_mm_t kernel_mul_mm<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, bfloat>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_mxfp4,   2,     dequantize_mxfp4,   float, float>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half, half, float4x4,      1,     dequantize_f32,     float, half>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half, half, half4x4,       1,     dequantize_f16,     half,  half>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q1_0,    8,     dequantize_q1_0,    float, half>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q2_0,    4,     dequantize_q2_0,    float, half>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_0,    2,     dequantize_q4_0,    float, half>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_1,    2,     dequantize_q4_1,    float, half>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_0,    2,     dequantize_q5_0,    float, half>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_1,    2,     dequantize_q5_1,    float, half>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q8_0,    2,     dequantize_q8_0,    float, half>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_mxfp4,   2,     dequantize_mxfp4,   float, half>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, half>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, half>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, half>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, half>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, half>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, half>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, half>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, half>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, half>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, half>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, half>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, half>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, half>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, half>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, half>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half, float4x4, 1, dequantize_f32, float, float>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half, half, float4x4, 1, dequantize_f32, float, float>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half, half, half4x4,  1, dequantize_f16, half,  float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, float>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q1_0, 8, dequantize_q1_0, float, float>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q2_0, 4, dequantize_q2_0, float, float>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_0, 2, dequantize_q4_0, float, float>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_1, 2, dequantize_q4_1, float, float>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_0, 2, dequantize_q5_0, float, float>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_1, 2, dequantize_q5_1, float, float>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q8_0, 2, dequantize_q8_0, float, float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_mxfp4, 2, dequantize_mxfp4, float, float>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_mxfp4, 2, dequantize_mxfp4, float, float>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_f32_f32_prec_bf16")]] kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, float4x4, 1, dequantize_f32, float, float>;
#endif

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_f16_f32_prec_bf16")]]     kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, half4x4,       1,     dequantize_f16,     half,  float>;
template [[host_name("kernel_mul_mm_id_q1_0_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q2_0_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_0_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_1_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_0_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_1_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q3_K_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_K_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_prec_bf16")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_prec_bf16")]] kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32_prec_bf16")]]  kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32_prec_bf16")]] kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32_prec_bf16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32_prec_bf16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32_prec_bf16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32_prec_bf16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32_prec_bf16")]]  kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32_prec_bf16")]]  kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32_prec_bf16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;
#endif

template [[host_name("kernel_mul_mm_id_f32_f32_prec_f32")]]    kernel mul_mm_id kernel_mul_mm_id<float, float, float4x4,      1,     dequantize_f32,     float, float>;
template [[host_name("kernel_mul_mm_id_f16_f32_prec_f32")]]    kernel mul_mm_id kernel_mul_mm_id<float, float, half4x4,       1,     dequantize_f16,     half,  float>;
template [[host_name("kernel_mul_mm_id_q1_0_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q2_0_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_0_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_1_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_0_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_1_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_mxfp4,   2,     dequantize_mxfp4,   float, float>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q3_K_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q5_K_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_prec_f32")]]   kernel mul_mm_id kernel_mul_mm_id<float, float, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_prec_f32")]]kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32_prec_f32")]] kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32_prec_f32")]]kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32_prec_f32")]] kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32_prec_f32")]] kernel mul_mm_id kernel_mul_mm_id<float, float, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32_prec_f32")]]  kernel mul_mm_id kernel_mul_mm_id<float, float, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32_prec_f32")]] kernel mul_mm_id kernel_mul_mm_id<float, float, bfloat4x4, 1, dequantize_bf16, bfloat, float>;
#endif

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half, half, float4x4,   1, dequantize_f32,  float, half>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half, half, half4x4,    1, dequantize_f16,  half,  half>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q1_0, 8, dequantize_q1_0, float, half>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q2_0, 4, dequantize_q2_0, float, half>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_0, 2, dequantize_q4_0, float, half>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_1, 2, dequantize_q4_1, float, half>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_0, 2, dequantize_q5_0, float, half>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_1, 2, dequantize_q5_1, float, half>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q8_0, 2, dequantize_q8_0, float, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat, block_mxfp4, 2, dequantize_mxfp4, float, half>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_mxfp4, 2, dequantize_mxfp4, float, half>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, half>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, half>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, half>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, half>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, half>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, half>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, half>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, half>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, half>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, half>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, half>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, half>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, half>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, half>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, half>;

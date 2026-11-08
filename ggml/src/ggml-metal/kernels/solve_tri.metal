#include "common.h"

constant short FC_solve_tri_nsg [[function_constant(FC_SOLVE_TRI + 0)]];
constant short FC_solve_tri_n   [[function_constant(FC_SOLVE_TRI + 1)]];
constant short FC_solve_tri_k   [[function_constant(FC_SOLVE_TRI + 2)]];

kernel void kernel_solve_tri_f32(
        constant ggml_metal_kargs_solve_tri & args,
        device   const char * src0,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        ushort3 tgpig[[threadgroup_position_in_grid]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const short NSG = FC_solve_tri_nsg;
    const short N   = FC_solve_tri_n;
    const short K   = FC_solve_tri_k;
    const short NP  = PAD2(N, NW);

    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x*NSG + sgitg;

    threadgroup float * sh0 = (threadgroup float *) shmem;

    device const float * src0_ptr = (device const float *)(src0 + i02 * args.nb02 + i03 * args.nb03) + sgitg*N;
    device const float * src1_ptr = (device const float *)(src1 + i02 * args.nb12 + i03 * args.nb13) + i01;
    device       float * dst_ptr  = (device       float *)(dst  + i02 * args.nb2  + i03 * args.nb3)  + i01;

    for (short rr = 0; rr < N; rr += NSG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            threadgroup float * sh0_cur = sh0 + sgitg*NP;

            for (short t = 0; t*NW < N; ++t) {
                const short idx = t*NW + tiisg;
                sh0_cur[idx] = src0_ptr[idx];
            }

            src0_ptr += NSG*N;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i01 >= args.ne10) {
            continue;
        }

        for (short ir = 0; ir < NSG && rr + ir < N; ++ir) {
            const short r = rr + ir;

            threadgroup float * sh0_cur = sh0 + ir*NP;

            float sum = 0.0f;

            for (short t = 0; t*NW < r; ++t) {
                const short idx = t*NW + tiisg;
                sum += sh0_cur[idx] * dst_ptr[idx*K] * (idx < r);
            }

            sum = simd_sum(sum);

            if (tiisg == 0) {
                const float diag = sh0_cur[r];

                dst_ptr[r*K] = (src1_ptr[r*K] - sum) / diag;
            }
        }
    }
}

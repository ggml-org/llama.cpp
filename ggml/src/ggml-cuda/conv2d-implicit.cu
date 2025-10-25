// #include <cuda_runtime.h>
#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "conv2d-implicit.cuh"


typedef unsigned int uint;
constexpr uint WARPSIZE = 32;

static __global__ void reduce_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    if (row * blockDim.x + col < ncols) {
        for (int i = 0; i < nrows; ++i){
            sum += x[i * ncols + row * blockDim.x + col];
        }
        dst[row * blockDim.x + col] = sum;
    }
}


template<typename T, const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          // layout: 0, NHWC; 1, NCHW
          const int layout, const bool vec_load, const int ksplit, const int PAD=4>
static __global__ void conv2d_implicit_kernel(const float * __restrict__ input,
                                              const T * __restrict__ kernel,
                                              float * __restrict__ output,
                                              const param_t param) {

    // __shared__ char smem[4 * (TM*TN*NUM_THREADS <= (BM * BK +  BK * (BN+PAD)) ? (BM * BK +  BK * (BN+PAD)) : (TM*TN*NUM_THREADS))];
    __shared__ char smem[sizeof(float) * (TM*TN*NUM_THREADS) <= sizeof(float) * 2 * (BM+PAD) * BK +  sizeof(T)*2*BK * (BN+PAD) ?
         sizeof(float)*2*(BM+PAD)*BK + sizeof(T)*2*BK*(BN+PAD) : sizeof(float) * (TM*TN*NUM_THREADS)];
    // __shared__ float smeminput[2 * BM * BK];
    // __shared__ float smemweight[2 * BK * (BN+PAD)];
    T *smemweight = reinterpret_cast<T *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 2 * BK * (BN+PAD) * sizeof(T));

    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    const uint PQ = param.Oh * param.Ow;

    // Warp tile
    const uint lane_id = tx % WARPSIZE;
    const uint warp_id = tx / WARPSIZE;
    const int mma_tid_x = warp_id / (BN / WN); //(lane_id / 2) % 8;
    const int mma_tid_y = warp_id % (BN / WN); //(lane_id / 16) * 2 + (lane_id % 2);

    // lds addr
    // int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    // int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    // const uint threadIdxInWarp = tx % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = lane_id % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = lane_id / (WSUBN / TN); // i/4

    // int x = bx * BM + input_lds_addr;
    // int y = by * BN + weight_lds_addr;
    int z = blockIdx.z;


    // float weight_ldg_reg[4];
    // float input_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    // int posh_ori = ((bx * 128 + tx / 2 ) / param.Ow) * param.u - param.p;
    // int posw_ori = ((bx * 128 + tx / 2 ) % param.Ow) * param.v - param.q;
    // int posh_ori = fastdiv(bx * BM + tx / 2, param.OW_fastdiv) * param.u - param.p;
    // int posw_ori = fastmodulo(bx * BM + tx / 2, param.OW_fastdiv) * param.v - param.q;


    // int inOffset = (ksplit > 0):  z * param.c * param.h * param.w ;
    // int weiOffset = (by * BN + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = layout == 0 ? param.c * param.w : param.h * param.w;
    // int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // uint ks, start_k;

    // if constexpr (ksplit > 0){
    //     const uint ks =  (weightKOffset + ksplit - 1) / ksplit;
    //     const uint start_k = z * ks;
    // } else {
    //     const uint ks = weightKOffset;
    //     const uint start_k = 0;
    // }
    const uint ks =  (ksplit > 0) ? (weightKOffset + ksplit - 1) / ksplit : weightKOffset;
    const uint start_k = (ksplit > 0)? z * ks: 0;
    const uint end_k = min(start_k + ks, weightKOffset);

    // sts addr
    // int weight_sts_addr = (tx % 8) * 132 +
    //                       (tx / 8) * 4;
    int write_flag = 1;
    T weight_frag[2][WNITER * TN];
    float input_frag[2][WMITER * TM] = {0.f};
    float output_frag[WMITER * TM * WNITER * TN] = {0.f};
// #pragma unroll
//     for (int i = 0; i < 8; ++i)
//     {
// #pragma unroll
//         for (int j = 0; j < 8; ++j)
//         {
//             output_frag[i][j] = 0;
//         }
//     }

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = tx / (BK / 4);
    const uint innerColA = tx % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    // const uint innerRowB = tx / (BN / 4);
    // const uint innerColB = tx % (BN / 4);
    // constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

// ldg
    const uint weight_sts_addr = innerRowA + innerColA * (BN+PAD) * 4;
#pragma unroll
    for (uint offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
        if(vec_load){
            // if (by * BN  + innerRowA + offset < param.k &&  start_k + innerColA * 4 < param.c * param.r * param.s){
            if (by * BN  + innerRowA + offset < param.k &&   start_k + innerColA * 4 < end_k){
                if constexpr (std::is_same_v<T, float>){
                    float4 tmp = reinterpret_cast<const float4 *>(&kernel[(by * BN + innerRowA + offset) * weightKOffset + start_k + innerColA * 4])[0];
                    smemweight[weight_sts_addr + offset +          0] = tmp.x;
                    smemweight[weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                    smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                    smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
                }else{ // read 4 halves
                    // half val[4];
                    float2 tmp = reinterpret_cast<const float2 *>(&kernel[(by * BN + innerRowA + offset) * weightKOffset + start_k + innerColA * 4])[0];
                    const half *val = reinterpret_cast<const half *>(&tmp);
                    // val[1] = reinterpret_cast<half2 *>(&tmp.y);
                    smemweight[weight_sts_addr + offset +          0] = val[0];
                    smemweight[weight_sts_addr + offset +   (BN+PAD)] = val[1];
                    smemweight[weight_sts_addr + offset + 2*(BN+PAD)] = val[2];
                    smemweight[weight_sts_addr + offset + 3*(BN+PAD)] = val[3];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                }
            }
        }else{
            #pragma unroll
            for (int i = 0; i < 4; ++i){
                if (by * BN  + innerRowA + offset < param.k &&  start_k + innerColA * 4 + i < end_k){
                    // float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4])[0];
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = kernel[(by * BN + innerRowA + offset) * weightKOffset + start_k + innerColA * 4 + i];
                } else {
                    smemweight[weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                }
            }
        }
    }


    // int curC = (tx / 32) / (param.r * param.s);             // channel offset
    // int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    // int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

    // int curR = (tx % 2) * 4 / (param.s * param.c);             // channel offset
    // int curS = ((tx % 2) * 4 % (param.s * param.c)) / param.c; // kernel r offset
    // int curC = ((tx % 2) * 4 % (param.s * param.c)) % param.c; // kernel s offset

    const uint input_sts_addr = innerRowA + innerColA * (BM+PAD) * 4;
#pragma unroll
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        int n = (ksplit > 0) ? (bx * BM + innerRowA + offset) / PQ : z;
        const unsigned int npq_res = (bx * BM + innerRowA + offset) % PQ;
        const int posh_ori = fastdiv((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.u - param.p;
        const int posw_ori = fastmodulo((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.v - param.q;
        int inOffset = n * param.c * param.h * param.w ;
        if(vec_load){
            // const uint curR = fastdiv(start_k + innerColA * 4,  param.SC_fastdiv);             // channel offset
            // const uint curS = fastdiv(fastmodulo(start_k + innerColA * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
            // const uint curC = fastmodulo(fastmodulo(start_k + innerColA * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
            const uint cur0 = fastdiv(start_k + innerColA * 4,
                   layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
            const uint cur1 = fastdiv(fastmodulo(start_k + innerColA * 4,
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const uint cur2 = fastmodulo(fastmodulo(start_k + innerColA * 4,
                layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
            const uint curC = layout == 0 ? cur2 : cur0;
            const uint curR = layout == 0 ? cur0 : cur1;
            const uint curS = layout == 0 ? cur1 : cur2;
            const int curH = posh_ori + curR * param.d_h; // input h
            const int curW = posw_ori + curS * param.d_w; // input w
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && start_k + innerColA * 4 < end_k){
                // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                float4 tmp = reinterpret_cast<const float4 *>(&input[inOffset + inOffsetTmp])[0];
                smeminput[input_sts_addr + offset +           0] = tmp.x;
                smeminput[input_sts_addr + offset +      BM+PAD] = tmp.y;
                smeminput[input_sts_addr + offset +  2*(BM+PAD)] = tmp.z;
                smeminput[input_sts_addr + offset +  3*(BM+PAD)] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i){
                // const uint curR = fastdiv(start_k + innerColA * 4 + i,  param.SC_fastdiv);             // channel offset
                // const uint curS = fastdiv(fastmodulo(start_k + innerColA * 4 + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                // const uint curC = fastmodulo(fastmodulo(start_k + innerColA * 4 + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                const uint cur0 = fastdiv(start_k + innerColA * 4 + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                const uint cur1 = fastdiv(fastmodulo(start_k + innerColA * 4 + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint cur2 = fastmodulo(fastmodulo(start_k + innerColA * 4 + i,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint curC = layout == 0 ? cur2 : cur0;
                const uint curR = layout == 0 ? cur0 : cur1;
                const uint curS = layout == 0 ? cur1 : cur2;
                const int curH = posh_ori + curR * param.d_h; // input h
                const int curW = posw_ori + curS * param.d_w; // input w
                if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && start_k + innerColA * 4 + i < end_k){
                    // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                    int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = input[inOffset + inOffsetTmp];
                } else {
                    smeminput[input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                }
            }
        }
    }

    // sts
    // for (int i = 0; i < 4; ++i)
    // {
    //     smemweight[weight_sts_addr + i*132] = weight_ldg_reg[i];
    // }
    // for (int i = 0; i < 4; ++i)
    // {
    //     smeminput[input_sts_addr + i * 128] = input_ldg_reg[i];
    // }

    __syncthreads();

    if(tx == 0 && bx == 0 && by == 0 && z == 0){
        printf("non tensor \n");
    }

    // if(tx == 0 && bx == 0 && by == 0 && z == 0){
    //     for(int i=0; i < 128; ++i)
    //         printf("%.2f,",  smeminput[i]);
    //     printf("\n");
    //     for(int i=128; i < 256; ++i)
    //         printf("%.2f,",  smeminput[i]);
    //     printf("\n");
    // }

    // if(tx == 0 && bx == 0 && by == 0 && z == 0){
    //     printf("%u, %u, %u, %u \n",  innerRowA, innerColA, rowStrideA, weight_sts_addr);
    //     for(int i=0; i < 16; ++i)
    //         printf("%f,",  smemweight[i]);
    //     printf("\n");
    //     for(int i=0; i < 16; ++i)
    //         printf("%f,",  param.weight[i*param.c*param.r*param.s]);
    //     printf("\n");
    // }

    // lds
    // int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;
    const uint input_lds_addr =  mma_tid_x * WM;
#pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        input_frag[0][wSubRowIdx * TM + i] = smeminput[input_lds_addr + wSubRowIdx * WSUBM +
                               threadRowInWarp * TM + i];

    // int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    const uint weight_lds_addr = mma_tid_y * WN;
#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
      for (uint i = 0; i < TN; ++i)
        weight_frag[0][wSubColIdx * TN + i] = smemweight[weight_lds_addr + wSubColIdx * WSUBN +
                             threadColInWarp * TN + i];

// #pragma unroll
//     for (int i = 0; i < 4; ++i)
//     {
//         weight_frag[0][i] = smemweight[weight_lds_addr + i];
//         weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
//     }
    // if(tx == 0 && bx == 0 && by == 0 && z == 0)
    // {
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][0], weight_frag[0][1], weight_frag[0][2], weight_frag[0][3]);
    //     printf("weight_ldg_reg:%f,%f,%f,%f\n",  weight_frag[0][4], weight_frag[0][5], weight_frag[0][6], weight_frag[0][7]);
    // }
// #pragma unroll
//     for (int i = 0; i < 4; ++i)
//     {
//         input_frag[0][i] = smeminput[input_lds_addr + i];
//         input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
//     }


    for (int crs = start_k; crs < end_k; crs += BK)
    {
        // ldg
//         if (by * BN + tx / 2 < param.k && tx % 2 * 4 < param.c * param.r * param.s){
//             float4 tmp = reinterpret_cast<float4 *>(&param.weight[by * BN + tx / 2 * weightKOffset + tx % 2 * 4 + crs + 8])[0];
//             weight_ldg_reg[0] = tmp.x;
//             weight_ldg_reg[1] = tmp.y;
//             weight_ldg_reg[2] = tmp.z;
//             weight_ldg_reg[3] = tmp.w;
//         } else {
//  #pragma unroll
//             for (int i = 0; i < 4; ++i)
//                 weight_ldg_reg[i] = 0.0;
//         }
        // curR = (crs + 8 + tx % 2 * 4) / (param.s * param.c);             // channel offset
        // curS = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) / param.c; // kernel r offset
        // curC = ((crs + 8 + tx % 2 * 4) % (param.s * param.c)) % param.c; // kernel s offset
//         curR = fastdiv(crs + 8 + (tx % 2) * 4,  param.SC_fastdiv);             // channel offset
//         curS = fastdiv(fastmodulo(crs + 8 + (tx % 2) * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
//         curC = fastmodulo(fastmodulo(crs + 8 + (tx % 2) * 4, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset

//         int curH = posh_ori + curR; // input h
//         int curW = posw_ori + curS; // input w
//         if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h){
//             int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;

//             // float4 tmp = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];
//             // input_ldg_reg[0] = tmp.x;
//             // input_ldg_reg[1] = tmp.y;
//             // input_ldg_reg[2] = tmp.z;
//             // input_ldg_reg[3] = tmp.w;
//             reinterpret_cast<float4 *>(&input_ldg_reg[0])[0] = reinterpret_cast<float4 *>(&param.input[inOffset + inOffsetTmp])[0];        } else {
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//                 input_ldg_reg[i] = 0.0;
//         }

        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < BK - 1; ++subcrs)
        {
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//             {
//                 weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * (BN+4) * 8 + weight_lds_addr + (subcrs + 1) * (BN+4) + i];
//                 weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * (BN+4) * 8 + weight_lds_addr + (subcrs + 1) * (BN+4) + i + 16];
//             }
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
                for (uint i = 0; i < TN; ++i)
                    weight_frag[(subcrs + 1) % 2][wSubColIdx * TN + i] = smemweight[load_flag * (BN+PAD) * BK +
                        (subcrs + 1) * (BN+PAD) + weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            // float* base_ptr = smemweight + load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132;

            // // first 4 values -> weight_frag[...][0..3]
            // float4 v0 = *reinterpret_cast<const float4*>(base_ptr);

            // // next 4 values (offset +16) -> weight_frag[...][4..7]
            // float4 v1 = *reinterpret_cast<const float4*>(base_ptr + 16);

            // // unpack into weight_frag
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][0]) = v0;
            // *reinterpret_cast<float4*>(&weight_frag[(subcrs + 1) % 2][4]) = v1;
// #pragma unroll
//             for (int i = 0; i < 4; ++i)
//             {
//                 input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * BM * 8 + input_lds_addr + (subcrs + 1) * BM + i];
//                 input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * BM * 8 + input_lds_addr + (subcrs + 1) * BM + i + 32];
//             }
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
                for (uint i = 0; i < TM; ++i)
                    input_frag[(subcrs + 1) % 2][wSubRowIdx * TM + i] = smeminput[load_flag * (BM+PAD) * BK +
                        (subcrs + 1) * (BM+PAD) + input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];

// #pragma unroll
//             for (int i = 0; i < 8; ++i)
//             {
// #pragma unroll
//                 for (int j = 0; j < 8; ++j)
//                 {
//                     output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
//                 }
//             }
            // execute warptile matmul
#pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
#pragma unroll
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                        (wSubColIdx * TN) + resIdxN] +=
                                input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM] *
                                ggml_cuda_cast<float>(weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN]);
                            // if(tx == 0 && bx == 0 && by == 0 && z == 0){
                            //     printf("subcrs:%d, i:%d, j:%d, %f * %f = %f, acc = %f\n", subcrs, wSubRowIdx * TM + resIdxM, wSubColIdx * TN + resIdxN,
                            //         input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM],
                            //         weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN],
                            //         input_frag[subcrs % 2][wSubRowIdx * TM + resIdxM] *
                            //         weight_frag[subcrs % 2][wSubColIdx * TN + resIdxN],
                            //         output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            //             (wSubColIdx * TN) + resIdxN]);
                            // }
                        }
                    }
                }
            }
        }
        // ldg
#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BN; offset += rowStrideA) {
            if(vec_load){
                if (by * BN  + innerRowA + offset < param.k &&  innerColA * 4 + crs + BK < end_k){
                    if constexpr (std::is_same_v<T, float>){
                        float4 tmp = reinterpret_cast<const float4 *>(&kernel[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK])[0];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +          0] = tmp.x;
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +   (BN+PAD)] = tmp.y;
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 2*(BN+PAD)] = tmp.z;
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 3*(BN+PAD)] = tmp.w;
                    } else {
                        float2 tmp = reinterpret_cast<const float2 *>(&kernel[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK])[0];
                        const half *val = reinterpret_cast<const half *>(&tmp);
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +          0] = val[0];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset +   (BN+PAD)] = val[1];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 2*(BN+PAD)] = val[2];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + 3*(BN+PAD)] = val[3];
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                }
            }else{
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    if (by * BN  + innerRowA + offset < param.k &&  innerColA * 4 + crs + BK + i < end_k){
                        // float4 tmp = reinterpret_cast<float4 *>(&param.weight[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK + i])[0];
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = kernel[(by * BN + innerRowA + offset) * weightKOffset + innerColA * 4 + crs + BK + i];
                    } else {
                        smemweight[write_flag * (BN+PAD) * BK + weight_sts_addr + offset + i*(BN+PAD)] = (T)0.f;
                    }
                }
            }
        }
#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            int n = (ksplit > 0) ? (bx * BM + innerRowA + offset) / PQ : z;
            const unsigned int npq_res = (bx * BM + innerRowA + offset) % PQ;
            const int posh_ori = fastdiv((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.u - param.p;
            const int posw_ori = fastmodulo((ksplit > 0) ? npq_res: bx * BM + innerRowA + offset, param.OW_fastdiv) * param.v - param.q;
            int inOffset = n * param.c * param.h * param.w ;
            if(vec_load){
                // const uint curR = fastdiv(innerColA * 4 + crs + BK,  param.SC_fastdiv);             // channel offset
                // const uint curS = fastdiv(fastmodulo(innerColA * 4 + crs + BK, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                // const uint curC = fastmodulo(fastmodulo(innerColA * 4 + crs + BK, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                const uint cur0 = fastdiv(innerColA * 4 + crs + BK,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                const uint cur1 = fastdiv(fastmodulo(innerColA * 4 + crs + BK,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint cur2 = fastmodulo(fastmodulo(innerColA * 4 + crs + BK,
                    layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                    layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                const uint curC = layout == 0 ? cur2 : cur0;
                const uint curR = layout == 0 ? cur0 : cur1;
                const uint curS = layout == 0 ? cur1 : cur2;

                const int curH = posh_ori + curR * param.d_h; // input h
                const int curW = posw_ori + curS * param.d_w; // input w
                if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && innerColA * 4 + crs + BK < end_k){
                    // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                    int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                    float4 tmp = reinterpret_cast<const float4 *>(&input[inOffset + inOffsetTmp])[0];
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +           0] = tmp.x;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +      BM+PAD] = tmp.y;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +  2*(BM+PAD)] = tmp.z;
                    smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset +  3*(BM+PAD)] = tmp.w;
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i){
                    // const uint curR = fastdiv(innerColA * 4 + crs + BK + i,  param.SC_fastdiv);             // channel offset
                    // const uint curS = fastdiv(fastmodulo(innerColA * 4 + crs + BK + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                    // const uint curC = fastmodulo(fastmodulo(innerColA * 4 + crs + BK + i, param.SC_fastdiv),  param.C_fastdiv); // kernel r offset
                    const uint cur0 = fastdiv(innerColA * 4 + crs + BK + i,
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv);             // channel offset
                    const uint cur1 = fastdiv(fastmodulo(innerColA * 4 + crs + BK + i,
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                        layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                    const uint cur2 = fastmodulo(fastmodulo(innerColA * 4 + crs + BK + i,
                        layout == 0 ? param.SC_fastdiv : param.RS_fastdiv),
                        layout == 0 ? param.C_fastdiv  : param.S_fastdiv); // kernel r offset
                    const uint curC = layout == 0 ? cur2 : cur0;
                    const uint curR = layout == 0 ? cur0 : cur1;
                    const uint curS = layout == 0 ? cur1 : cur2;

                    const int curH = posh_ori + curR * param.d_h; // input h
                    const int curW = posw_ori + curS * param.d_w; // input w
                    if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && innerColA * 4 + crs + BK + i < end_k){
                        // int inOffsetTmp = curH * inChannelOffset + curW * param.c + curC;
                        int inOffsetTmp = layout == 0 ?
                                curH * inChannelOffset + curW * param.c + curC:
                                curC * inChannelOffset + curH * param.w + curW;
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = input[inOffset + inOffsetTmp];
                    } else {
                        smeminput[write_flag * (BM+PAD) * BK + input_sts_addr + offset + i*(BM+PAD)] = 0.f;
                    }
                }
            }
        }
        // sts
        // for (int i = 0; i < 4; ++i)
        // {
        //     smemweight[write_flag * (BN+4) * 8 + weight_sts_addr + i * (BN+4)] = weight_ldg_reg[i];
        // }
        // for (int i = 0; i < 4; ++i)
        // {
        //     smeminput[write_flag * BM * 8 + input_sts_addr + i * BM] = input_ldg_reg[i];
        // }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
#pragma unroll
            for (uint i = 0; i < TM; ++i)
                input_frag[0][wSubRowIdx * TM + i] = smeminput[(load_flag ^ 1) * (BM+PAD) * BK +
                    input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
#pragma unroll
            for (uint i = 0; i < TN; ++i)
                weight_frag[0][wSubColIdx * TN + i] = smemweight[(load_flag ^ 1) * (BN+PAD) * BK +
                    weight_lds_addr + wSubColIdx * WSUBN + threadColInWarp * TN + i];
// #pragma unroll
//         for (int i = 0; i < 4; ++i)
//         {
//             weight_frag[0][i] = smemweight[(load_flag ^ 1) * (BN+4) * 8 + weight_lds_addr + i];
//             weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * (BN+4) * 8 + weight_lds_addr + i + 16];
//         }
// #pragma unroll
//         for (int i = 0; i < 4; ++i)
//         {
//             input_frag[0][i] = smeminput[(load_flag ^ 1) * BM * 8 + input_lds_addr + i];
//             input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * BM * 8 + input_lds_addr + i + 32];
//         }
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
#pragma unroll
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        output_frag[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                    (wSubColIdx * TN) + resIdxN] +=
                            input_frag[1][wSubRowIdx * TM + resIdxM] *
                            ggml_cuda_cast<float>(weight_frag[1][wSubColIdx * TN + resIdxN]);
                    }
                }
            }
        }
// #pragma unroll
//         for (int i = 0; i < 8; ++i)
//         {
// #pragma unroll
//             for (int j = 0; j < 8; ++j)
//             {
//                 output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
//             }
//         }
    }

    // if(tx == 59 && bx == 0 && by == 0 && z == 0){
    //     for (int i = 0; i < WMITER * TM * WNITER * TN; ++i){
    //         printf("%f,",  output_frag[i]);
    //         if((i+1) % (WNITER * TN) == 0)
    //             printf("\n");
    //     }
    //     printf("\n");
    // }
    // if(tx == 59 && bx == 0 && by == 0 && z == 0){
    //     int cnt[3] = {0};
    //     float values[3] = {-1.f};
    //     for (int i = 0; i < WMITER * TM * WNITER * TN; ++i){
    //         for(int j = 0; j < 3; j++){
    //             if (output_frag[i] == values[j]){                    
    //                 cnt[j]++;
    //                 break;                    
    //             } else{
    //                 if (cnt[j] == 0){
    //                     values[j] = output_frag[i];
    //                     cnt[j]++;
    //                     break;
    //                 }
    //             }          
    //         }
    //     }
    //     for(int j = 0; j < 3; j++){
    //         if(values[j] != -1.f)
    //             printf("value: %f, cnt: %d \n", values[j], cnt[j]);
    //     }
    // }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);
    // float *smembias = reinterpret_cast<float *>(smem + 16 * 1024);

    // bias ldg/sts
    // if (tx < BN)
    // {
    //     smembias[tx] = param.bias[by * BN + tx];
    // }

    // constexpr uint OUTMITER = (TM * TN * WNITER * WMITER * NUM_THREADS) / (2 * BK * (BM + BN)) / OUTNITER;
    // const uint WMITER_TM_OUTMITER = WMITER * TM / OUTMITER;
    // const uint WNITER_TN_OUTNITER = WNITER * TN / OUTNITER;



//     // uint32_t bias_lds_addr = warp_id / 2 * 32;

// #pragma unroll
//     for (int i = 0; i < 2; ++i)
//     {
// #pragma unroll
//         for (int j = 0; j < 2; ++j)
//         {
//             __syncthreads();

// #pragma unroll
//             for (int subi = 0; subi < 4; ++subi)
//             {
// #pragma unroll
//                 for (int subj = 0; subj < 4; ++subj)
//                 {
//                     // output sts
//                     smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
//                 }
//             }
//             __syncthreads();

// #pragma unroll
//             for (int subk = 0; subk < 16; ++subk)
//             {
//                 int outOffset = z * param.k * param.Oh * param.Ow + (m_idx + i * 16 + subk) * param.Oh * param.Ow + n_idx + j * 32;
//                 if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
//                     param.output[outOffset] = smemoutput[output_lds_addr + subk * 32];
//             }
//         }
//     }
    const uint output_lds_addr = warp_id * WSUBM * WSUBN + lane_id;
    // const uint m_idx = by * BN + mma_tid_y * WN + threadColInWarp * WNITER_TN_OUTNITER;
    // const uint n_idx = bx * BM + mma_tid_x * WM + threadRowInWarp * WMITER_TM_OUTMITER;
    // const uint output_sts_addr = warp_id * WMITER_TM_OUTMITER * WNITER_TN_OUTNITER * WARPSIZE +
    //                     (threadRowInWarp * (WSUBN / TN)  + threadColInWarp) * WMITER_TM_OUTMITER * WNITER_TN_OUTNITER;
    const uint output_sts_addr = mma_tid_x * BN / WN * TM * TN * WARPSIZE + mma_tid_y * TM * TN * WARPSIZE +
                         threadColInWarp * TN * WSUBM + threadRowInWarp * TM;
    const uint m_idx = by * BN + mma_tid_y * WN;
    const uint n_idx = bx * BM + mma_tid_x * WM;

#pragma unroll
    for (int i = 0; i < WMITER; ++i)
    {
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < TM; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < TN; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subj * WSUBM + subi] =
                        output_frag[(i * TM + subi) * (WNITER * TN) + j * TN + subj];
                }
            }
            __syncthreads();
#pragma unroll
            for (int subk = 0; subk < TM * TN; ++subk){
                const uint row =  m_idx + j * WSUBN + (lane_id + subk * WARPSIZE) / WSUBM;
                const uint gemm_i =  n_idx + i * WSUBM + (lane_id + subk * WARPSIZE) % WSUBM;
                const int n = (ksplit > 0) ? gemm_i / PQ : z;
                const int col = (ksplit > 0) ? gemm_i % PQ : gemm_i;
                if (n < param.n && row < param.k && col < param.Oh * param.Ow){
                //     int outOffset = z * param.n * param.k * param.Oh * param.Ow +  n * param.k * param.Oh * param.Ow  + (m_idx + i * 16 + subk) * param.Oh * param.Ow + (n_idx + j * 32);
                // if (n < param.n && (m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                //     param.interm[outOffset] = smemoutput[output_lds_addr + subk * 32];
                    const uint outOffset = ksplit > 0 ? 
                                z * param.n * param.k * param.Oh * param.Ow + n * param.k * param.Oh * param.Ow +
                                row * param.Oh * param.Ow + col :
                                z * param.k * param.Oh * param.Ow + row * param.Oh * param.Ow + col;
                    output[outOffset] = smemoutput[output_lds_addr + subk * WARPSIZE];
                }
            }
        }
    }
}



template <unsigned int mma_tiles_per_warp_m, unsigned int mma_tiles_per_warp_k, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_a(
  const half* src,
  half (&reg)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]
)
{
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  static_assert(mma_tiles_per_warp_m == 8, "mma_tiles_per_warp_m must be 4");
  static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");

  uint32_t (&reg_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][2] = reinterpret_cast<uint32_t(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2]>(reg);
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
  swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
  constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes
    
    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0][0]), "=r"(reg_[0][0][1]), "=r"(reg_[1][0][0]), "=r"(reg_[1][0][1])
      : "r"(src_addr)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][0][0]), "=r"(reg_[2][0][1]), "=r"(reg_[3][0][0]), "=r"(reg_[3][0][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][0][0]), "=r"(reg_[4][0][1]), "=r"(reg_[5][0][0]), "=r"(reg_[5][0][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 0
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][0][0]), "=r"(reg_[6][0][1]), "=r"(reg_[7][0][0]), "=r"(reg_[7][0][1])
      : "r"(src_addr + 96 * smem_stride_)
    );

    src_addr ^= 0b10000;
    
    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][1][0]), "=r"(reg_[0][1][1]), "=r"(reg_[1][1][0]), "=r"(reg_[1][1][1])
      : "r"(src_addr)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][1][0]), "=r"(reg_[2][1][1]), "=r"(reg_[3][1][0]), "=r"(reg_[3][1][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][1][0]), "=r"(reg_[4][1][1]), "=r"(reg_[5][1][0]), "=r"(reg_[5][1][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 1
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][1][0]), "=r"(reg_[6][1][1]), "=r"(reg_[7][1][0]), "=r"(reg_[7][1][1])
      : "r"(src_addr + 96 * smem_stride_)
    );
    
    src_addr ^= 0b110000;

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][2][0]), "=r"(reg_[0][2][1]), "=r"(reg_[1][2][0]), "=r"(reg_[1][2][1])
      : "r"(src_addr)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][2][0]), "=r"(reg_[2][2][1]), "=r"(reg_[3][2][0]), "=r"(reg_[3][2][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][2][0]), "=r"(reg_[4][2][1]), "=r"(reg_[5][2][0]), "=r"(reg_[5][2][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 2
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][2][0]), "=r"(reg_[6][2][1]), "=r"(reg_[7][2][0]), "=r"(reg_[7][2][1])
      : "r"(src_addr + 96 * smem_stride_)
    );
    src_addr ^= 0b10000;

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][3][0]), "=r"(reg_[0][3][1]), "=r"(reg_[1][3][0]), "=r"(reg_[1][3][1])
      : "r"(src_addr)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[2][3][0]), "=r"(reg_[2][3][1]), "=r"(reg_[3][3][0]), "=r"(reg_[3][3][1])
      : "r"(src_addr + 32 * smem_stride_)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[4][3][0]), "=r"(reg_[4][3][1]), "=r"(reg_[5][3][0]), "=r"(reg_[5][3][1])
      : "r"(src_addr + 64 * smem_stride_)
    );

    // 3
    asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[6][3][0]), "=r"(reg_[6][3][1]), "=r"(reg_[7][3][0]), "=r"(reg_[7][3][1])
      : "r"(src_addr + 96 * smem_stride_)
    );
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

template <unsigned int mma_tiles_per_warp_k, unsigned int mma_tiles_per_warp_n, unsigned int smem_stride>
__device__ __forceinline__ void ldmatrix_b(
  const half* src,
  half (&reg)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]
)
{
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  static_assert(mma_tiles_per_warp_k == 4, "mma_tiles_per_warp_k must be 4");
  static_assert(mma_tiles_per_warp_n == 8, "mma_tiles_per_warp_n must be 8");
  
  uint32_t (&reg_) [4][8] = reinterpret_cast<uint32_t(&)[4][8]>(reg);
//   const unsigned int logical_offset = ((threadIdx.x % 8) * smem_stride) +  (((threadIdx.x % 32) / 8) * 8);
//   unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b11100000000) >> 5);
//   uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
//   constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes
  unsigned int logical_offset = (threadIdx.x % 32) * smem_stride;
  unsigned int swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4);
  swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2);
  uint32_t src_addr = cvta_to_shared_u32(src + swizzled_offset);
  constexpr unsigned int smem_stride_ = smem_stride * sizeof(half); // convert stride to bytes


//   asm volatile (
//     "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
//     "{%0, %1, %2, %3}, [%4];"
//     : "=r"(reg_[0][0]), "=r"(reg_[0][1]), "=r"(reg_[0][2]), "=r"(reg_[0][3])
//     : "r"(src_addr)
//   );

    // 0
  asm volatile (
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(reg_[0][0]), "=r"(reg_[0][1]), "=r"(reg_[0][2]), "=r"(reg_[0][3])
      : "r"(src_addr)
    );


  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[0][4]), "=r"(reg_[0][5]), "=r"(reg_[0][6]), "=r"(reg_[0][7])
    // : "r"(src_addr ^ 0b1000000)
    : "r"(src_addr + 32 * smem_stride_)
  );

  src_addr ^= 0b10000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[1][0]), "=r"(reg_[1][1]), "=r"(reg_[1][2]), "=r"(reg_[1][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[1][4]), "=r"(reg_[1][5]), "=r"(reg_[1][6]), "=r"(reg_[1][7])
    // : "r"(src_addr ^ 0b1000000)
    : "r"(src_addr + 32 * smem_stride_)
  );

//   src_addr += 8 * smem_stride_;
  src_addr ^= 0b110000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[2][0]), "=r"(reg_[2][1]), "=r"(reg_[2][2]), "=r"(reg_[2][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[2][4]), "=r"(reg_[2][5]), "=r"(reg_[2][6]), "=r"(reg_[2][7])
    // : "r"(src_addr ^ 0b1000000)
    : "r"(src_addr + 32 * smem_stride_)
  );

  src_addr ^= 0b10000;

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[3][0]), "=r"(reg_[3][1]), "=r"(reg_[3][2]), "=r"(reg_[3][3])
    : "r"(src_addr)
  );

  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(reg_[3][4]), "=r"(reg_[3][5]), "=r"(reg_[3][6]), "=r"(reg_[3][7])
    // : "r"(src_addr ^ 0b1000000)
    : "r"(src_addr + 32 * smem_stride_)
  );
#else
    GGML_UNUSED(src);
    GGML_UNUSED(reg);
    NO_DEVICE_CODE;
#endif
}

template<const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WK,  const int NUM_THREADS>
static __global__ void conv2d_implicit_kernel(const half * __restrict__ input,
                                              const half * __restrict__ kernel,
                                              half * __restrict__ output,
                                              const param_t param) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
  constexpr unsigned int MMA_M = 16;
  constexpr unsigned int MMA_N = 8;

// if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y ==0)
//      printf("conv2d_implicit_kernel launch BM:%d, BN:%d, BK:%d, WM:%d, WN:%d, WK:%d, NUM_THREADS:%d \n", BM, BN, BK, WM, WN, WK, NUM_THREADS);

  const unsigned int K = param.c * param.r * param.s;
//   const uint PQ = param.Oh * param.Ow;
  const uint inChannelOffset = param.c * param.w;
  const uint weightKOffset = param.c * param.r * param.s;

  // for convenience/readability in index calculations
//   const unsigned int A_stride = K;
//   const unsigned int B_stride = N;
//   const unsigned int CD_stride = N;

  // calculate how many bits of shared memory indices are going to be swizzled, and create masks
//   constexpr unsigned int SWIZZLE_BITS_B = int_log2(BN / 8);

  // loop bounds, constexpr where possible allows for loop unrolling
  constexpr unsigned int mma_tiles_per_warp_k = 4;
  constexpr unsigned int mma_tiles_per_warp_m = WM / MMA_M;
  constexpr unsigned int mma_tiles_per_warp_n = WN / MMA_N;
  const unsigned int num_block_tiles_k = (K + (BK-1)) / BK;
  
  // calculate block/warp indices
  const unsigned int block_m = blockIdx.y;
  const unsigned int block_n = blockIdx.x;
  const unsigned int warp_m = threadIdx.y;
  const unsigned int warp_n = threadIdx.x / 32;
  
  // double buffering
  extern __shared__ half shmem[];
  half* A_block_smem = shmem;
  half* B_block_smem = &shmem[BM * BK];
  constexpr int BUFFER_SIZE = BM * BK + BK * BN;

  // declare register storage
  // ptx instructions expect uint32_t registers, where each uint32_t is 2 halfs packed together  
  uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
  uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
  uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];
  
  // convenience cast to half for register storage
  half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);
  half (&A_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]>(A_register);
  half (&B_register_) [mma_tiles_per_warp_k][mma_tiles_per_warp_n][2] = reinterpret_cast<half(&)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]>(B_register);

  // accumulators start at 0
  for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
  {
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        acc_register_[mma_m][mma_n][0] = 0;
        acc_register_[mma_m][mma_n][1] = 0;
        acc_register_[mma_m][mma_n][2] = 0;
        acc_register_[mma_m][mma_n][3] = 0;
      }
  }

  // these register arrays are used to cache values pre-fetched from global memory during the inner loop of the kernel
  // the code is nicer if we hard code it for these tile dimensions and number of threads
  // since we performing this copy with float4 pointers, for these tile dimensions it works out to be 8 float4s for A and 4 float4s for B
  static_assert(BM == 256);
  static_assert(BN == 256);
  static_assert(BK == 32);
  static_assert(NUM_THREADS == 256);
  float4 A_gmem_cache_reg[4];
  float4 B_gmem_cache_reg[4];

  // prefetch the first block tile of A,B into shared memory
//   half* A_block_gmem = input + (block_m * BM * A_stride);
  const half* A_block_gmem = input;
  const half* B_block_gmem = kernel + (block_n * weightKOffset);
  tileMemcpySwizzleA<BM, NUM_THREADS>(A_block_gmem, A_block_smem, inChannelOffset, param);
  tileMemcpySwizzleB<BN, NUM_THREADS>(B_block_gmem, B_block_smem, weightKOffset, param);

  // construct const pointers to warp tiles for use inside the inner loop
//   if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x ==0 && blockIdx.y ==0){
//     for(int i = 0; i < 32; ++i)
//         printf("%.2f,", __half2float(A_block_smem[i]));
//     printf("\n");
//   }

  int offset_direction = 1;

  for (unsigned int block_k = 1; block_k <= num_block_tiles_k; block_k++)
  {
    __syncthreads();

    if (block_k != num_block_tiles_k)
    {
    //   half* A_block_gmem = A + (block_m * BM * A_stride) + (block_k * BK);
      const half* A_block_gmem = input;
      const half* B_block_gmem = kernel + (block_n * weightKOffset);
      tileMemcpyLoadA<BM, BK, NUM_THREADS, 4>(A_block_gmem, A_gmem_cache_reg, block_k * BK, inChannelOffset, param);
      tileMemcpyLoadB<BN, BK, NUM_THREADS, 4>(B_block_gmem, B_gmem_cache_reg, block_k * BK, weightKOffset, param);
    }
    half* A_warp_tile = A_block_smem + (warp_m * WM * BK);
    half* B_warp_tile = B_block_smem + (warp_n * WN * BK);

    ldmatrix_a<mma_tiles_per_warp_m, mma_tiles_per_warp_k, BK>(A_warp_tile, A_register_);
    ldmatrix_b<mma_tiles_per_warp_k, mma_tiles_per_warp_n, BK>(B_warp_tile, B_register_);

    // outer product between mma tiles
    #pragma unroll
    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++)
    {
      #pragma unroll
      for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
      {
        #pragma unroll
        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
        {
          asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, "
            "{%2, %3}, "
            "{%4}, "
            "{%5, %6};"
            : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
            : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
              "r"(B_register[mma_k][mma_n])
              "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
          );
        }
      }
    //   if(threadIdx.x == 0 && threadIdx.y ==0 && blockIdx.x ==0 && blockIdx.y ==0){
    //     printf(" %d, %d: %f, %f, %f, %f \n", block_k, mma_k, __half2float(acc_register_[3][0][0]), __half2float(acc_register_[3][0][1]), 
    //     __half2float(acc_register_[3][0][2]), __half2float(acc_register_[3][0][3]));
    //     printf(" %d, %d: %f, %f, %f, %f \n", block_k, mma_k, __half2float(A_register_[3][mma_k][0]), __half2float(A_register_[3][mma_k][1]), 
    //     __half2float(A_register_[3][mma_k][2]), __half2float(A_register_[3][mma_k][3]));
    //     printf(" %d, %d: %f, %f, %f, %f \n", block_k, mma_k, __half2float(B_register_[mma_k][0][0]), __half2float(B_register_[mma_k][0][1]), 
    //     __half2float(B_register_[mma_k][0][2]), __half2float(B_register_[mma_k][0][3]));
    //   }
    //   if(threadIdx.x < 4 && threadIdx.y ==0 && blockIdx.x ==0 && blockIdx.y ==0){        
    //     printf("A %d, %d, %d: %f, %f \n", block_k, mma_k, threadIdx.x, __half2float(A_register_[3][mma_k][0]), __half2float(A_register_[3][mma_k][1]));
    //     printf("B %d, %d, %d: %f, %f \n", block_k, mma_k, threadIdx.x, __half2float(B_register_[mma_k][0][0]), __half2float(B_register_[mma_k][0][1]));
    //   }
    }
    // if(threadIdx.x == 0 && threadIdx.y ==0 && blockIdx.x ==0 && blockIdx.y ==0){
    //  printf(" %d: %f, %f, %f, %f \n", block_k, __half2float(acc_register_[3][0][0]), __half2float(acc_register_[3][0][1]), 
    //  __half2float(acc_register_[3][0][2]), __half2float(acc_register_[3][0][3]));
    //  printf(" %d: %f, %f, %f, %f \n", block_k, __half2float(A_register_[3][0][0]), __half2float(A_register_[3][0][1]), 
    //  __half2float(A_register_[3][0][2]), __half2float(A_register_[3][0][3]));
    //  printf(" %d: %f, %f, %f, %f \n", block_k, __half2float(B_register_[3][0][0]), __half2float(B_register_[3][0][1]), 
    //  __half2float(B_register_[3][0][2]), __half2float(B_register_[3][0][3]));
    // }


    if (block_k != num_block_tiles_k)
    {
      // switch smem buffers each iteration
      A_block_smem = A_block_smem + BUFFER_SIZE * offset_direction;
      B_block_smem = B_block_smem + BUFFER_SIZE * offset_direction;
      offset_direction = -1 * offset_direction;

      tileMemcpySwizzleStore<BM, NUM_THREADS, 4>(A_gmem_cache_reg, A_block_smem);
      tileMemcpySwizzleStore<BN, NUM_THREADS, 4>(B_gmem_cache_reg, B_block_smem);
    }
  }



    // reuse smem
    half *smemoutput = shmem;
    const uint lane_id = threadIdx.x % WARPSIZE;  
    const uint mma_row = lane_id / 4;
    const uint mma_col = lane_id % 4;
    const uint output_lds_addr = warp_m * WM * BN/2 + lane_id * BN/2 + warp_n * WN/2;
    const uint output_sts_addr = warp_m * WM * BN/2 + mma_row * BN/2 + warp_n * WN/2  + mma_col * 2;
    const uint m_idx = block_n * BN + warp_n * WN;
    const uint n_idx = block_m * BM + warp_m * WM + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        __syncthreads();

        for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
        {
            for (unsigned int mma_n = i * mma_tiles_per_warp_n/2; mma_n < (i+1)*mma_tiles_per_warp_n/2; mma_n++)
            {
                    // output sts
                uint32_t (&reg_)[2] = reinterpret_cast<uint32_t(&)[2]>(acc_register_[mma_m][mma_n]);
                const uint idx = output_sts_addr + 
                            mma_m * MMA_M * BN / 2 + (mma_n - i * mma_tiles_per_warp_n/2) * MMA_N;
                uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(&smemoutput[idx]);
                dst_ptr[0] = reg_[0];
                dst_ptr = reinterpret_cast<uint32_t*>(&smemoutput[idx + 8 * BN / 2]);
                dst_ptr[0] = reg_[1];
            }
        }
        __syncthreads();
        // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x ==0 && blockIdx.y ==0){
        //     for(int ii = 0; ii < 128; ++ii)
        //         printf("%.2f,", __half2float(smemoutput[ii]));
        //     printf("\n");
        //     for(int ii = 128; ii < 256; ++ii)
        //         printf("%.2f,", __half2float(smemoutput[ii]));
        //     printf("\n");
        //     for(int ii = 0; ii < 128; ++ii)
        //         printf("%.2f,", __half2float(smemoutput[ii*128]));
        //     printf("\n");
        //     for(int ii = 128; ii < 256; ++ii)
        //         printf("%.2f,", __half2float(smemoutput[ii*128]));
        //     printf("\n");
        // }

#pragma unroll
        for (int subk = 0; subk < WN / 2; ++subk){
            for (int j = 0; j < 4; ++j){
                const uint row =  m_idx + subk + i * WN / 2;
                const uint gemm_i =  n_idx + j*32;
                const int n = fastdiv(gemm_i, param.OHOW_fastdiv);
                const int col = fastmodulo(gemm_i, param.OHOW_fastdiv);
                if(n < param.n && row < param.k && col < param.Oh * param.Ow){
                //     int outOffset = z * param.n * param.k * param.Oh * param.Ow +  n * param.k * param.Oh * param.Ow  + (m_idx + i * 16 + subk) * param.Oh * param.Ow + (n_idx + j * 32);
                // if (n < param.n && (m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                //     param.interm[outOffset] = smemoutput[output_lds_addr + subk * 32];
                    const uint outOffset = n * param.k * param.Oh * param.Ow + row * param.Oh * param.Ow + col;
                    output[outOffset] = smemoutput[output_lds_addr + subk + j*32*BN/2];
                    // if(outOffset == 32){
                    //     printf("(%u, %u, %u, %u), output[%d,%d,%d]=%f \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
                    //          n, row, col, __half2float(output[outOffset]));
                    // }
                }
            }
        }
    }

#else
    GGML_UNUSED(input);
    GGML_UNUSED(kernel);
    GGML_UNUSED(output);
    GGML_UNUSED(param);
    NO_DEVICE_CODE;
#endif
}


#define NUM_VARIANTS 6

/*
  conv_shapes[][0]: ne_input=[384,512,256,1],ne_kernel=[3,3,256,256]
  conv_shapes[][1]: ne_input=[96,128,512,1],ne_kernel=[3,3,512,512]
  conv_shapes[][2]: ne_input=[192,256,512,1git diff],ne_kernel=[3,3,512,512]
*/
constexpr static int conv_shapes[][NUM_VARIANTS] = {
    { 128, 128,  128, 256 }, // BM
    { 256,  128,  256, 128 }, // BN
    { 8, 8, 8, 8 }, // BK
    { 128, 64,  32, 128   }, // WM
    { 32,  32 ,  256, 32   }, // WN
    { 2,   2,  1, 1   }, // WNITER
    { 8,   4,  4, 4  }, // TM
    { 8,   4,  8, 8   }, // TN
    { 256,  256, 128, 256}	    //  NUM_THREADS
};

template <typename T, unsigned int CONV_SHAPE>
static void conv2d_implicit_cuda(const float * X_D, const T * K_D, float * Y_D, const param_t P, cudaStream_t st) {    

    const uint BM = conv_shapes[0][CONV_SHAPE];
    const uint BN = conv_shapes[1][CONV_SHAPE];
    const uint BK = conv_shapes[2][CONV_SHAPE];
    const uint WM = conv_shapes[3][CONV_SHAPE];
    const uint WN = conv_shapes[4][CONV_SHAPE];
    const uint WNITER = conv_shapes[5][CONV_SHAPE];
    const uint TM = conv_shapes[6][CONV_SHAPE];
    const uint TN = conv_shapes[7][CONV_SHAPE];
    const uint NUM_THREADS = conv_shapes[8][CONV_SHAPE];
    int blockx = ((P.Oh * P.Ow + BM - 1) / BM); // blockx  number
    int blocky = (P.k + BN-1) / BN;             // blocky  number
    int blockz = P.n;                           // blockz  number
    // int threadx = NUM; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 thblock(NUM_THREADS, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    // int smem_size = 24 * 1024;
    if(P.c % 4 == 0){
        if(P.layout == 0)
            conv2d_implicit_kernel<T, BM, BN, BK, WM, WN,
                WNITER, TM, TN, NUM_THREADS, 0, true, 0><<<grid, thblock, 0, st>>>(X_D, K_D, Y_D, P);
        else if(P.layout == 1)
            conv2d_implicit_kernel<T, BM, BN, BK, WM, WN,
                WNITER, TM, TN, NUM_THREADS, 1, false, 0><<<grid, thblock, 0, st>>>(X_D, K_D, Y_D, P);
    } else{
        if(P.layout == 0)
            conv2d_implicit_kernel<T, BM, BN, BK, WM, WN,
                WNITER, TM, TN, NUM_THREADS, 0, false, 0><<<grid, thblock, 0, st>>>(X_D, K_D, Y_D, P);
        else if(P.layout == 1)
            conv2d_implicit_kernel<T, BM, BN, BK, WM, WN,
                WNITER, TM, TN, NUM_THREADS, 1, false, 0><<<grid, thblock, 0, st>>>(X_D, K_D, Y_D, P);
    }
}

static void conv2d_implicit_cuda_f16(ggml_backend_cuda_context & ctx, const float * X_D, const half * K_D, float * Y_D, int cc, const param_t P, cudaStream_t st) {
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) && P.layout == 0 && P.c % 8 == 0) {
// #if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        // printf("tensor core path called\n");
        constexpr unsigned int BM_dim = 256;
        constexpr unsigned int BN_dim = 256;
        constexpr unsigned int BK_dim = 32;

        constexpr unsigned int WARPS_PER_BLOCK_M = 2;
        constexpr unsigned int WARPS_PER_BLOCK_N = 4;
        constexpr unsigned int WARPS_PER_BLOCK_K = 4;

        constexpr unsigned int WM_dim = BM_dim / WARPS_PER_BLOCK_M;
        constexpr unsigned int WN_dim = BN_dim / WARPS_PER_BLOCK_N;
        constexpr unsigned int WK_dim = BK_dim / WARPS_PER_BLOCK_K;
        const unsigned int BlocksM =  (P.n * P.Oh * P.Ow + BM_dim - 1) / BM_dim;
        const unsigned int BlocksN =  (P.k + BN_dim - 1) / BN_dim;
        constexpr unsigned int ThreadsM = WARPS_PER_BLOCK_M;
        constexpr unsigned int ThreadsN = WARPSIZE * WARPS_PER_BLOCK_N;
        constexpr unsigned int NumThreads = ThreadsM * ThreadsN;
        const unsigned int shmem_bytes = (BM_dim * BK_dim + BK_dim * BN_dim) * 2 * sizeof(half);

        cudaFuncSetAttribute(conv2d_implicit_kernel<BM_dim, BN_dim, BK_dim, WM_dim, WN_dim, WK_dim, NumThreads>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,    65536); // set shared memory limit to 64KB which is maximum for sm_75
        dim3 gridDim(BlocksN, BlocksM);
        dim3 blockDim(ThreadsN, ThreadsM);

        int id = ggml_cuda_get_device();
        ggml_cuda_pool_alloc<half> x_f16(ctx.pool(id));

        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        size_t ne = P.c * P.h * P.w * P.n;
        x_f16.alloc(ne);
        to_fp16_cuda(X_D, x_f16.get(), ne, st);
        const half *X_H = x_f16.get();
        ggml_cuda_pool_alloc<half> Y_H(ctx.pool(id), P.k * P.Oh * P.Ow * P.n);
        conv2d_implicit_kernel<BM_dim, BN_dim, BK_dim,
            WM_dim, WN_dim, WK_dim, NumThreads>
            <<<gridDim, blockDim, shmem_bytes, st>>>(X_H, K_D, Y_H.get(), P);
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(Y_H.get(), Y_D, P.k * P.Oh * P.Ow * P.n, st);
// #else
//     printf("non tensor path called\n");
//     conv2d_implicit_cuda<half, 1>(X_D, K_D, Y_D, P, st);
// #endif
    } else{
       conv2d_implicit_cuda<half, 1>(X_D, K_D, Y_D, P, st);
    }

}

static void conv2d_implicit_cuda_f32(ggml_backend_cuda_context & ctx, const float * X_D, const float * K_D, float * Y_D, int cc, const param_t P, cudaStream_t st) {
    conv2d_implicit_cuda<float, 1>(X_D, K_D, Y_D, P, st);
}

void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);


    cudaStream_t st = ctx.stream();
    const int cc            = ggml_cuda_info().devices[ctx.device].cc;

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y
    const int       LT   = p[6];  // layout

    GGML_ASSERT(LT == 0 || LT == 1);

    // same number of input channels
    GGML_ASSERT(LT == 0 ? input->ne[0] == kernel->ne[0] : input->ne[2] == kernel->ne[2]);
    // No cwhn
    GGML_ASSERT(p[7] == false);

    // const int IW = input->ne[0];   // input_w
    // const int IH = input->ne[1];   // input_h
    // const int OW = dst->ne[0];     // output_w
    // const int OH = dst->ne[1];     // output_h
    // const int KW = kernel->ne[0];  // kernel_w
    // const int KH = kernel->ne[1];  // kernel_h
    // const int IC = input->ne[2];   // input_channels
    const int IW = input->ne[LT == 0 ? 1 : 0];   // input_w
    const int IH = input->ne[LT == 0 ? 2 : 1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[LT == 0 ? 1 : 0];  // kernel_w
    const int KH = kernel->ne[LT == 0 ? 2 : 1];  // kernel_h
    const int IC = input->ne[LT == 0 ? 0: 2];   // input_channels

    const int OC = kernel->ne[3];  // ouptut_chanles
    const int B  = input->ne[3];   // n_batches
    
    const int64_t total  = B * OC * OH * OW;
    
    param_t params = { B, IC, IH, IW, OC, KH, KW, ST_Y, ST_X, PD_Y, PD_X, DL_Y, DL_X, OH, OW };
    params.SC_fastdiv = init_fastdiv_values(KW*IC);
    params.OW_fastdiv = init_fastdiv_values(OW);
    params.OHOW_fastdiv = init_fastdiv_values(OW*OH);
    params.C_fastdiv = init_fastdiv_values(IC);
    params.RS_fastdiv = init_fastdiv_values(KW*KH);
    params.S_fastdiv = init_fastdiv_values(KW);
    params.layout = LT;

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_implicit_cuda_f16(ctx, X_D, (half *) K_D, Y_D, cc, params, st);
    } else {
        conv2d_implicit_cuda_f32(ctx, X_D, K_D, Y_D, cc, params, st);
    }
}

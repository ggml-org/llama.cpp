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
    __shared__ char smem[sizeof(float) * (TM*TN*NUM_THREADS) <= sizeof(float) * 2 * BM * BK +  sizeof(T)*2*BK * (BN+PAD) ?
         sizeof(float)*2*BM*BK + sizeof(T)*2*BK*(BN+PAD) : sizeof(float) * (TM*TN*NUM_THREADS)];
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

    const uint input_sts_addr = innerRowA + innerColA * BM * 4;
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
                smeminput[input_sts_addr + offset +     0] = tmp.x;
                smeminput[input_sts_addr + offset +    BM] = tmp.y;
                smeminput[input_sts_addr + offset +  2*BM] = tmp.z;
                smeminput[input_sts_addr + offset +  3*BM] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    smeminput[input_sts_addr + offset + i*BM] = 0.f;
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
                    smeminput[input_sts_addr + offset + i*BM] = input[inOffset + inOffsetTmp];
                } else {
                    smeminput[input_sts_addr + offset + i*BM] = 0.f;
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
                    input_frag[(subcrs + 1) % 2][wSubRowIdx * TM + i] = smeminput[load_flag * BM * BK +
                        (subcrs + 1) * BM + input_lds_addr + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];

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
                    smeminput[write_flag * BM * BK + input_sts_addr + offset +     0] = tmp.x;
                    smeminput[write_flag * BM * BK + input_sts_addr + offset +    BM] = tmp.y;
                    smeminput[write_flag * BM * BK + input_sts_addr + offset +  2*BM] = tmp.z;
                    smeminput[write_flag * BM * BK + input_sts_addr + offset +  3*BM] = tmp.w;
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        smeminput[write_flag * BM * BK + input_sts_addr + offset + i*BM] = 0.f;
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
                        smeminput[write_flag * BM * BK + input_sts_addr + offset + i*BM] = input[inOffset + inOffsetTmp];
                    } else {
                        smeminput[write_flag * BM * BK + input_sts_addr + offset + i*BM] = 0.f;
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
                input_frag[0][wSubRowIdx * TM + i] = smeminput[(load_flag ^ 1) * BM * BK +
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

static void conv2d_implicit_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const param_t P, cudaStream_t st) {
    conv2d_implicit_cuda<half, 1>(X_D, K_D, Y_D, P, st);
}

static void conv2d_implicit_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const param_t P, cudaStream_t st) {
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
    params.C_fastdiv = init_fastdiv_values(IC);
    params.RS_fastdiv = init_fastdiv_values(KW*KH);
    params.S_fastdiv = init_fastdiv_values(KW);
    params.layout = LT;

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_implicit_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_implicit_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}

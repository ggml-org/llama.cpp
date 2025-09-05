#include "conv2d-implicit.cuh"
#include "convert.cuh"

typedef struct{    
    unsigned int      n;                              //batch size              
    unsigned int      c;                              //number if channels          
    unsigned int      h;                              //height
    unsigned int      w;                              //width                 
    unsigned int      k;                              //number of filters
    unsigned int      r;                              //filter height
    unsigned int      s;                              //filter width
    unsigned int      u;                              //stride height
    unsigned int      v;                              //stride width
    unsigned int      p;                              //padding height
    unsigned int      q;                              //padding width
    unsigned int      d_h;                            //dilation height
    unsigned int      d_w;                            //dilation width
    unsigned int      Oh;                             //output height
    unsigned int      Ow;                             //output width
} param_t;



template <typename T>
static __global__ void conv2d_implicit_kernel(const float * __restrict__ input,
                                              const T * __restrict__ kernel,
                                              float * __restrict__ output,
                                              const param_t param) {

    extern __shared__ unsigned char smem[];
    T *smemweight = reinterpret_cast<T *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    
    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    T weight_ldg_reg[4];
    float input_ldg_reg[4];
    
    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i){
        posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr
    int weight_sts_addr = (tx % 8) * 132 +
                          (tx / 8) * 4;
    int input_sts_addr = (tx / 32) * 128 + (tx % 32);

    int write_flag = 1;
    T weight_frag[2][8];
    float input_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i){
#pragma unroll
        for (int j = 0; j < 8; ++j){
            output_frag[i][j] = 0;
        }
    }
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i){
        if (tx % 8 < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k){
            weight_ldg_reg[i] = kernel[weiOffset + tx % 8 + i * weightKOffset];
        }
        else{
            weight_ldg_reg[i] = (T)0.f;
        }
    }
    int curC = (tx / 32) / (param.r * param.s);             // channel offset
    int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset
#pragma unroll
    for (int i = 0; i < 4; ++i){
        int curH = posh_ori[i] + curR * param.d_h; // input h
        int curW = posw_ori[i] + curS * param.d_w; // input w
        int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
        if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && curC < param.c){
            input_ldg_reg[i] = input[inOffset + inOffsetTmp];
        }
        else{
            input_ldg_reg[i] = 0.0;
        }
    }
    // sts
    for (int i = 0; i < 4; ++i){
        smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
    }
    for (int i = 0; i < 4; ++i){
        smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
    }

    __syncthreads();
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i){
        weight_frag[0][i] = smemweight[weight_lds_addr + i];
        weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i){
        input_frag[0][i] = smeminput[input_lds_addr + i];
        input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
    }
    for (int crs = 0; crs < param.r * param.s * param.c; crs += 8){
        // ldg
        int weiOffsetTmp = crs + 8 + tx % 8;
#pragma unroll
        for (int i = 0; i < 4; ++i){
            if (weiOffsetTmp < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k){
                weight_ldg_reg[i] = kernel[weiOffset + weiOffsetTmp + i * weightKOffset];
            }
            else{
                weight_ldg_reg[i] = (T)0.f;
            }
        }
        curC = (crs + 8 + tx / 32) / (param.r * param.s);             // channel offset
        curR = ((crs + 8 + tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
        curS = ((crs + 8 + tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

#pragma unroll
        for (int i = 0; i < 4; ++i){
            int curH = posh_ori[i] + curR * param.d_h; // input h
            int curW = posw_ori[i] + curS * param.d_w; // input w
            int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h && curC < param.c){
                input_ldg_reg[i] = input[inOffset + inOffsetTmp];
            }
            else{
                input_ldg_reg[i] = 0.f;
            }
        }
        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; ++subcrs){
#pragma unroll
            for (int i = 0; i < 4; ++i){
                weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i){
                input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i){
#pragma unroll
                for (int j = 0; j < 8; ++j){
                    output_frag[i][j] += ggml_cuda_cast<float>(weight_frag[subcrs % 2][i]) * input_frag[subcrs % 2][j];
                }
            }
        }
        // sts
        for (int i = 0; i < 4; ++i){
            smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i){
            smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i){
            weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
            weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i){
            input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
            input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i){
#pragma unroll
            for (int j = 0; j < 8; ++j){
                output_frag[i][j] += ggml_cuda_cast<float>(weight_frag[1][i]) * input_frag[1][j];
            }
        }
    }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);


    uint32_t output_sts_addr = warp_id * 512 + mma_tid_y * 4 * 8 * 4 + mma_tid_x * 4;
    uint32_t output_lds_addr = warp_id * 512 + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i){
#pragma unroll
        for (int j = 0; j < 2; ++j){
            __syncthreads();
#pragma unroll
            for (int subi = 0; subi < 4; ++subi){
#pragma unroll
                for (int subj = 0; subj < 4; ++subj){
                    // output sts
                    smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                }
            }
            __syncthreads();

#pragma unroll
            for (int subk = 0; subk < 16; ++subk){
                int outOffset = z * param.k * param.Oh * param.Ow + (m_idx + i * 16 + subk) * param.Oh * param.Ow + n_idx + j * 32;
                if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                    output[outOffset] = smemoutput[output_lds_addr + subk * 32];
            }
        }
    }
}

template <typename T>
static void conv2d_implicit_cuda(const float * X_D, const T * K_D, float * Y_D, const param_t P, cudaStream_t st) {    
    int blockx = ((P.Oh * P.Ow + 127) / 128); // blockx  number
    int blocky = (P.k + 127) / 128;             // blocky  number
    int blockz = P.n;                           // blockz  number
    int threadx = CUDA_CONV2D_IMPLICT_BLOCK_SIZE; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 thblock(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    int smem_size = 24 * 1024;
    conv2d_implicit_kernel<T><<<grid, thblock, smem_size, st>>>(X_D, K_D, Y_D, P);
}

static void conv2d_implicit_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const param_t P, cudaStream_t st) {
    conv2d_implicit_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_implicit_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const param_t P, cudaStream_t st) {
    conv2d_implicit_cuda<float>(X_D, K_D, Y_D, P, st);
}

void ggml_cuda_op_conv2d_implicit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int IW = input->ne[0];   // input_w
    const int IH = input->ne[1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[0];  // kernel_w
    const int KH = kernel->ne[1];  // kernel_h
    const int IC = input->ne[2];   // input_channels
    const int OC = kernel->ne[3];  // ouptut_chanles
    const int B  = input->ne[3];   // n_batches
    
    const int64_t total  = B * OC * OH * OW;
    
    param_t params = { B, IC, IH, IW, OC, KH, KW, ST_Y, ST_X, PD_Y, PD_X, DL_Y, DL_X, OH, OW };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_implicit_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_implicit_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}

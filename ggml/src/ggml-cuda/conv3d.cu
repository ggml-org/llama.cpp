// CUDA Conv3D (GGML_OP_CONV_3D)
// One thread per output element; X/Y are float, K is float|half. nb[]-stride indexing.

#include "conv3d.cuh"
#include "convert.cuh"

// kernel block size
#define CUDA_CONV3D_BLOCK_SIZE 256

struct conv3d_params {
    int64_t IW, IH, ID; // input dims
    int64_t OW, OH, OD; // output dims
    int64_t KW, KH, KD; // kernel dims
    int64_t ST_X, ST_Y, ST_Z; // stride
    int64_t PD_X, PD_Y, PD_Z; // pad
    int64_t DL_X, DL_Y, DL_Z; // dilation
    int64_t IC, OC; // channels
    int64_t B; // batch
    int64_t TOTAL; // B * OC * OD * OH * OW

    int64_t input_nb[4];  // src1 nb[]
    int64_t kernel_nb[4]; // src0 nb[]
    int64_t dst_nb[4];    // dst nb[]
};

// Unpack idx -> (b, oc, oz, oy, ox) with ox fastest
static inline __device__ void conv3d_unpack_index(
        int64_t idx,
        const conv3d_params & p,
        int64_t & b,
        int64_t & oc,
        int64_t & oz,
        int64_t & oy,
        int64_t & ox) {
    ox = idx % p.OW;
    idx /= p.OW;
    oy = idx % p.OH;
    idx /= p.OH;
    oz = idx % p.OD;
    idx /= p.OD;
    oc = idx % p.OC;
    b  = idx / p.OC;
}

// nb[]-stride addressing with fused dims

template<typename TK>
static __global__ void conv3d_kernel(
        const float * __restrict__ X,   // input data  (src1)
        const TK    * __restrict__ K,   // kernel data (src0)
        float       * __restrict__ Y,   // output data (dst)
        conv3d_params params,
        const int64_t c,   // input channels
        const int64_t n,   // batch size
        const int64_t oc   // output channels
    ) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.TOTAL) {
        return;
    }

    int64_t b, oc_idx, oz, oy, ox;
    conv3d_unpack_index(idx, params, b, oc_idx, oz, oy, ox);

    float sum = 0.0f;

    for (int64_t ic = 0; ic < c; ++ic) {
        for (int64_t kz = 0; kz < params.KD; ++kz) {
            for (int64_t ky = 0; ky < params.KH; ++ky) {
                for (int64_t kx = 0; kx < params.KW; ++kx) {
                    const int64_t sz = oz * params.ST_Z + kz * params.DL_Z - params.PD_Z;
                    const int64_t sy = oy * params.ST_Y + ky * params.DL_Y - params.PD_Y;
                    const int64_t sx = ox * params.ST_X + kx * params.DL_X - params.PD_X;

                    if (sz < 0 || sz >= params.ID || sy < 0 || sy >= params.IH || sx < 0 || sx >= params.IW) {
                        continue;
                    }

                    const int64_t cn_idx  = b * params.IC + ic;        // input fused
                    const int64_t kch_idx = oc_idx * params.IC + ic;   // kernel fused

                    const int64_t x_offset = sx * params.input_nb[0]
                                           + sy * params.input_nb[1]
                                           + sz * params.input_nb[2]
                                           + cn_idx * params.input_nb[3];
                    const int64_t k_offset = kx * params.kernel_nb[0]
                                           + ky * params.kernel_nb[1]
                                           + kz * params.kernel_nb[2]
                                           + kch_idx * params.kernel_nb[3];

                    const float x_val = X[x_offset / (int64_t)sizeof(float)];
                    const TK    k_val_raw = K[k_offset / (int64_t)sizeof(TK)];
                    const float k_val = ggml_cuda_cast<float>(k_val_raw);
                    sum += x_val * k_val;
                }
            }
        }
    }
    const int64_t ocn_idx = b * params.OC + oc_idx;
    const int64_t y_offset = ox * params.dst_nb[0]
                           + oy * params.dst_nb[1]
                           + oz * params.dst_nb[2]
                           + ocn_idx * params.dst_nb[3];
    Y[y_offset / (int64_t)sizeof(float)] = sum;
}

// f32 kernel
static void conv3d_cuda_f32(
        const float * X_D,
        const float * K_D,
        float       * Y_D,
        const conv3d_params & params,
        const int64_t c,
        const int64_t n,
        const int64_t oc,
        cudaStream_t st) {
    const int blocks = (int) ((params.TOTAL + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE);
    conv3d_kernel<float><<<blocks, CUDA_CONV3D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, params, c, n, oc);
    CUDA_CHECK(cudaGetLastError());
}

// f16 kernel
static void conv3d_cuda_f16(
        const float * X_D,
        const half  * K_D,
        float       * Y_D,
        const conv3d_params & params,
        const int64_t c,
        const int64_t n,
        const int64_t oc,
        cudaStream_t st) {
    const int blocks = (int) ((params.TOTAL + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE);
    conv3d_kernel<half><<<blocks, CUDA_CONV3D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, params, c, n, oc);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    const int32_t * p = (const int32_t *) dst->op_params;
    const int32_t s0 = p[0];
    const int32_t s1 = p[1];
    const int32_t s2 = p[2];
    const int32_t p0 = p[3];
    const int32_t p1 = p[4];
    const int32_t p2 = p[5];
    const int32_t d0 = p[6];
    const int32_t d1 = p[7];
    const int32_t d2 = p[8];
    const int32_t c  = p[9];  // input channels
    const int32_t n  = p[10]; // batch size
    const int32_t oc = p[11]; // output channels

    const int64_t IW = input->ne[0];
    const int64_t IH = input->ne[1];
    const int64_t ID = input->ne[2];
    const int64_t OW = dst->ne[0];
    const int64_t OH = dst->ne[1];
    const int64_t OD = dst->ne[2];
    const int64_t KW = kernel->ne[0];
    const int64_t KH = kernel->ne[1];
    const int64_t KD = kernel->ne[2];

    conv3d_params params;
    params.IW = IW; params.IH = IH; params.ID = ID;
    params.OW = OW; params.OH = OH; params.OD = OD;
    params.KW = KW; params.KH = KH; params.KD = KD;
    params.ST_X = s0; params.ST_Y = s1; params.ST_Z = s2;
    params.PD_X = p0; params.PD_Y = p1; params.PD_Z = p2;
    params.DL_X = d0; params.DL_Y = d1; params.DL_Z = d2;
    params.IC = (int64_t) c; params.OC = (int64_t) oc;
    params.B  = (int64_t) n;
    params.TOTAL = params.B * params.OC * params.OD * params.OH * params.OW;
    for (int i = 0; i < 4; ++i) {
        params.input_nb[i]  = input->nb[i];
        params.kernel_nb[i] = kernel->nb[i];
        params.dst_nb[i]    = dst->nb[i];
    }

    GGML_ASSERT(kernel->ne[3] == params.IC * params.OC);
    GGML_ASSERT(input->ne[3]  == params.IC * params.B);

    cudaStream_t st = ctx.stream();

    const float * X_D = (const float *) input->data;
    float *       Y_D = (float *) dst->data;

    if (kernel->type == GGML_TYPE_F32) {
        const float * K_D = (const float *) kernel->data;
        conv3d_cuda_f32(X_D, K_D, Y_D, params, (int64_t) c, (int64_t) n, (int64_t) oc, st);
    } else if (kernel->type == GGML_TYPE_F16) {
        const half * K_D = (const half *) kernel->data;
        conv3d_cuda_f16(X_D, K_D, Y_D, params, (int64_t) c, (int64_t) n, (int64_t) oc, st);
    } else {
        GGML_ASSERT(false && "unsupported kernel type");
    }
}


#include "conv3d.cuh"
#include "convert.cuh"
#include "ggml.h"

struct conv_3d_params {
    const int64_t IW, IH, ID;
    const int64_t OW, OH, OD;
    const int64_t KW, KH, KD;
    const int64_t ST_X, ST_Y, ST_Z;
    const int64_t PD_X, PD_Y, PD_Z;
    const int64_t DL_X, DL_Y, DL_Z;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
    int64_t z_min, z_max;
};

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, int out_z, const conv_3d_params & P) {
    kernel_bounds bounds;

    bounds.y_min = max((int64_t)0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);

    bounds.x_min = max((int64_t)0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);

    bounds.z_min = max((int64_t)0, (P.PD_Z - out_z * P.ST_Z + P.DL_Z - 1) / P.DL_Z);
    bounds.z_max = min(P.KD, (P.ID + P.PD_Z - out_z * P.ST_Z + P.DL_Z - 1) / P.DL_Z);

    return bounds;
}

template<typename T>
static __global__ void conv_3d_kernel(const float* __restrict__ input,
                               const T* __restrict__ kernel,
                               float* __restrict__ output,
                               conv_3d_params P)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(output_index >= P.TOTAL)
    {
        return;
    }

    size_t out_el_column = output_index % P.OW;
    size_t out_el_row = output_index / (P.OW) % P.OH;
    size_t out_el_depth = output_index / (P.OH * P.OW ) % P.OD;
    size_t out_el_channel = output_index / (P.OW * P.OH * P.OD) % P.OC;
    size_t out_el_batch = output_index / (P.OW * P.OH * P.OD * P.OC);

    float acc = 0;
    {
        for(size_t channel = 0; channel < P.IC; ++channel)
        {
	        kernel_bounds box = calculate_kernel_bounds(out_el_column, out_el_row, out_el_depth, P);

            for(size_t k_z = box.z_min; k_z < box.z_max; ++k_z)
            {
                size_t in_d = out_el_depth * P.ST_Z + k_z * P.DL_Z - P.PD_Z;
                for(size_t k_y = box.y_min; k_y < box.y_max; ++k_y)
                {
                    size_t in_row = out_el_row * P.ST_Y + k_y * P.DL_Y - P.PD_Y;

                    for(size_t k_x = box.x_min; k_x < box.x_max; ++k_x)
                    {
                        size_t in_col = out_el_column * P.ST_X + k_x * P.DL_X - P.PD_X;

                        size_t input_index = out_el_batch * (P.IC * P.ID * P.IH * P.IW) +
                                             channel * (P.ID * P.IH * P.IW ) +
                                             in_d * (P.IH * P.IW) +
            	                             in_row * P.IW +
            	    		                 in_col;

            	        size_t kernel_index =  out_el_channel * (P.IC * P.KD * P.KH * P.KW) +
            	                               channel * (P.KD * P.KH * P.KW) +
            	                               k_z * (P.KH * P.KW) +
                                               k_y * P.KW +
            	    		                   k_x;

                        acc += input[input_index] * ggml_cuda_cast<float>(kernel[kernel_index]);
            	    }
                }
            }
        }
    }

    output[output_index] = acc;
}

template <typename T>
static void conv3d_cuda(const float * input_data, const T * kernel_data, float * output_data, const conv_3d_params P, cudaStream_t st) {
    const int block = (P.TOTAL + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE;
    conv_3d_kernel<T><<<block, CUDA_CONV3D_BLOCK_SIZE, 0, st>>>(input_data, kernel_data, output_data, P);
}

static void conv3d_cuda_f16(const float* input_data, const half* kernel_data, float* output, const conv_3d_params P, cudaStream_t st)
{
    conv3d_cuda<half>(input_data, kernel_data, output, P, st);
}

static void conv3d_cuda_f32(const float* input_data, const float* kernel_data, float* output, const conv_3d_params P, cudaStream_t st)
{
    conv3d_cuda<float>(input_data, kernel_data, output, P, st);
}

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
{
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    float* kernel_data = (float*)kernel->data;
    const float* input_data = (float*)input->data;
    float* output = (float*)dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;

    const int ST_X = p[0];  // stride_width
    const int ST_Y = p[1];  // stride_height
    const int ST_Z = p[2];  // stride_depth
    const int PD_X = p[3];  // padding_width
    const int PD_Y = p[4];  // padding_height
    const int PD_Z = p[5];  // padding_depth
    const int DL_X = p[6];  // dilation_width
    const int DL_Y = p[7];  // dilation_height
    const int DL_Z = p[8];  // dilation_depth
    const int IC   = p[9];  // input_channels
    const int B    = p[10]; // batches
    const int OC   = p[11]; // output_channels

    const int IW = input->ne[0];  // input_weight
    const int IH = input->ne[1];  // input_height
    const int ID = input->ne[2];  // input_depth
    const int OW = dst->ne[0];    // output_width
    const int OH = dst->ne[1];    // output_height
    const int OD = dst->ne[2];    // output_depth
    const int KW = kernel->ne[0]; // kernel_width
    const int KH = kernel->ne[1]; // kernel_height
    const int KD = kernel->ne[2]; // kernel_depth

    const int total = B * OC * OD * OH * OW;
    conv_3d_params params = {IW, IH, ID, OW, OH, OD, KW, KH, KD, ST_X, ST_Y, ST_Z, PD_X, PD_Y, PD_Z, DL_X, DL_Y, DL_Z, IC, OC, B, total};

    if (kernel->type == GGML_TYPE_F16) {
        conv3d_cuda_f16(input_data, (half *) kernel_data, output, params, st);
    } else {
        conv3d_cuda_f32(input_data, kernel_data, output, params, st);
    }
}

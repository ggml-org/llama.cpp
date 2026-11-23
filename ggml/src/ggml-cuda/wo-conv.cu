#include "wo-conv.cuh"
#include "convert.cuh"

namespace {

// the block size mul_mat_vec_f picks for ncols
int mmvf_block_size(int64_t ncols, int warp_size, int cc) {
    int64_t best = warp_size;
    int64_t niter_best = (ncols + 2*warp_size - 1)/(2*warp_size);
    int64_t max_bs = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_bs = 128;
    }
    for (int64_t bs = 2*warp_size; bs <= max_bs; bs += warp_size) {
        const int64_t niter = (ncols + 2*bs - 1)/(2*bs);
        if (niter < niter_best) {
            niter_best = niter;
            best = bs;
        }
    }
    return (int) best;
}

struct wo_conv_params {
    const nv_bfloat16 * x; int64_t x_s1; const float * y; int ncols2; int nrows;
    float * mm_dst;
    const float * past; int64_t past_s1; const float * w; int64_t w_s1; float scale;
    float * state; float * dst; const float * res2; float * dst2;
};

template <int block_size, int d_conv>
__global__ void wo_conv_f32(const wo_conv_params p) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x;
    if (row >= p.nrows) {
        return; // the idle blocks of the RDNA4 grid padding
    }
    const int tid = threadIdx.x;

    // mul_mat_vec_f, one BF16 row and one F32 vector (the HIP path)
    const int    * x2 = (const int *) (p.x + row*p.x_s1);
    const float2 * y2 = (const float2 *) p.y;

    __shared__ float buf_iw[warp_size];
    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }
    float sumf = 0.0f;
    for (int col2 = tid; col2 < p.ncols2; col2 += block_size) {
        const int    tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
        const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
        ggml_cuda_mad(sumf, tmpx0, tmpy.x);
        ggml_cuda_mad(sumf, tmpx1, tmpy.y);
    }
    sumf = warp_reduce_sum<warp_size>(sumf);
    if (block_size > warp_size) {
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid < warp_size) {
            sumf = buf_iw[tid];
            sumf = warp_reduce_sum<warp_size>(sumf);
        }
    }
    if (tid != 0) {
        return;
    }

    // ssm_conv_step_f32 for this channel, the product as its new column and its residual
    const int   c = row;
    const float m = sumf;
    p.mm_dst[c] = m;

    float win[d_conv];
#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        win[k] = p.past[c*p.past_s1 + k];
    }
    win[d_conv - 1] = m;

    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < d_conv; ++k) {
        s += win[k] * p.w[c*p.w_s1 + k];
    }
    const float out = (s + m) * p.scale;

#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        p.state[c*(d_conv - 1) + k] = win[k + 1];
    }
    if (p.dst2) {
        p.dst2[c] = p.res2[c] + out;
    } else {
        p.dst[c] = out;
    }
}

template <int block_size>
void launch_bs(const wo_conv_params & p, int d_conv, dim3 grid, cudaStream_t stream) {
    switch (d_conv) {
        case 2: wo_conv_f32<block_size, 2><<<grid, block_size, 0, stream>>>(p); break;
        case 3: wo_conv_f32<block_size, 3><<<grid, block_size, 0, stream>>>(p); break;
        case 4: wo_conv_f32<block_size, 4><<<grid, block_size, 0, stream>>>(p); break;
        default: GGML_ABORT("wo_conv: conv width %d", d_conv);
    }
}

} // namespace

void ggml_cuda_op_wo_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
        const ggml_tensor * past, const ggml_tensor * conv_w, float conv_scale, ggml_tensor * state_cell,
        ggml_tensor * dst, const ggml_tensor * res2, ggml_tensor * dst2) {
    const ggml_tensor * x = mm->src[0];
    const ggml_tensor * y = mm->src[1];

    wo_conv_params p;
    p.x       = (const nv_bfloat16 *) x->data;
    p.x_s1    = x->nb[1]/ggml_type_size(x->type);
    p.y       = (const float *) y->data;
    p.ncols2  = (int) (x->ne[0]/2);
    p.nrows   = (int) x->ne[1];
    p.mm_dst  = (float *) mm->data;
    p.past    = (const float *) past->data;
    p.past_s1 = past->nb[1]/sizeof(float);
    p.w       = (const float *) conv_w->data;
    p.w_s1    = conv_w->nb[1]/sizeof(float);
    p.scale   = conv_scale;
    p.state   = (float *) state_cell->data;
    p.dst     = (float *) dst->data;
    p.res2    = dst2 ? (const float *) res2->data : nullptr;
    p.dst2    = dst2 ? (float *) dst2->data : nullptr;

    const int dev = ctx.device;
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const int bs  = mmvf_block_size(x->ne[0], ggml_cuda_info().devices[dev].warp_size, cc);
    dim3 grid(p.nrows);
    // mul_mat_vec_f's RDNA4 padding: a grid of a multiple of 256 blocks runs about twice as slow without it
    if (p.nrows % 256 == 0 && GGML_CUDA_CC_IS_RDNA4(cc)) {
        grid.x += 2;
    }
    const int d_conv = (int) conv_w->ne[0];
    cudaStream_t stream = ctx.stream();
    switch (bs) {
        case  32: launch_bs< 32>(p, d_conv, grid, stream); break;
        case  64: launch_bs< 64>(p, d_conv, grid, stream); break;
        case  96: launch_bs< 96>(p, d_conv, grid, stream); break;
        case 128: launch_bs<128>(p, d_conv, grid, stream); break;
        case 160: launch_bs<160>(p, d_conv, grid, stream); break;
        case 192: launch_bs<192>(p, d_conv, grid, stream); break;
        case 224: launch_bs<224>(p, d_conv, grid, stream); break;
        case 256: launch_bs<256>(p, d_conv, grid, stream); break;
        default: GGML_ABORT("wo_conv: block size %d", bs);
    }
}

#pragma once

#include "vecdotq.cuh"
#include "unary.cuh"

struct spark_q6_activation {
    int u[2];
    float d[2];
    __device__ spark_q6_activation(const block_q8_1 * y, int iqs) {
        const int off = 4*(iqs/16)+(iqs%16)/8;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = get_int_b4(y[off+2*i].qs, iqs%8);
            d[i] = __low2float(y[off+2*i].ds);
        }
    }
    __device__ float dot(const void * weights, int b, int iqs) const {
        const block_q6_K & w = ((const block_q6_K *) weights)[b];
        const int vl = get_int_b2(w.ql, iqs);
        const int vh = get_int_b2(w.qh, 8*(iqs/16)+iqs%8) >> (2*((iqs%16)/8));
        const int8_t * scales = w.scales+8*(iqs/16)+(iqs%16)/4;
        float sum = 0.0f;
        // Keep one packed chunk per lane; the generic Q6 helper now uses two.
        #pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            const int low = (vl >> (4*i)) & 0x0f0f0f0f;
            const int high = ((vh >> (4*i)) << 4) & 0x30303030;
            int dot = ggml_cuda_dp4a(low | high, u[i], 0);
            dot = ggml_cuda_dp4a(0xe0e0e0e0, u[i], dot);
            sum += d[i]*(dot*scales[4*i]);
        }
        return float(w.d)*sum;
    }
};

template<int width> struct spark_q4_wide {
    int u[2][2*width], sum[2];
    float d[2];
    __device__ spark_q4_wide(const block_q8_1 * y, int lane) {
        const int j = lane/(4/width), sub = width*(lane%(4/width));
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const auto & a = y[2*j+i];
            d[i] = __low2float(a.ds);
            sum[i] = 0;
            #pragma unroll
            for (int q = 0; q < 2*width; ++q) {
                u[i][q] = get_int_b4(a.qs,sub+(q < width ? q : q+4-width));
                sum[i] = ggml_cuda_dp4a(0x01010101,u[i][q],sum[i]);
            }
        }
    }
    __device__ float dot(const void * weights, int b, int lane) const {
        const auto & w = ((const block_q4_K *) weights)[b];
        const int j = lane/(4/width), sub = width*(lane%(4/width));
        const int * qs = (const int *) w.qs + 8*j+sub;
        int v[2*width];
        #pragma unroll
        for (int q = 0; q < 2*width; ++q) v[q] = qs[q < width ? q : q+4-width];
        const auto * s = (const uint16_t *) w.scales;
        uint16_t aux[2];
        if (j < 2) {
            aux[0] = s[j] & 0x3f3f;
            aux[1] = s[j+2] & 0x3f3f;
        } else {
            aux[0] = (s[j+2] & 0x0f0f) | ((s[j-2] & 0xc0c0) >> 2);
            aux[1] = ((s[j+2] >> 4) & 0x0f0f) | ((s[j] & 0xc0c0) >> 2);
        }
        const auto * sc = (const uint8_t *) aux;
        float sd = 0, sm = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int dot = 0;
            #pragma unroll
            for (int q = 0; q < 2*width; ++q) dot = ggml_cuda_dp4a((v[q] >> (4*i)) & 0x0f0f0f0f,u[i][q],dot);
            sd += d[i]*(dot*sc[i]);
            sm += d[i]*(sum[i]*sc[i+2]);
        }
        const float2 dm = __half22float2(w.dm);
        return dm.x*sd - dm.y*sm;
    }
};

template<ggml_type type, int width, int nw, int rows, int groups, int k, bool gated, bool quant = false, int min_blocks = 1, bool fuse_norm = false>
__launch_bounds__(32*nw*groups,min_blocks)
static __global__ void spark_gemv(const void * weights, const void * gate, const block_q8_1 * y, float * dst, const float * bias = nullptr, block_q8_1 * qdst = nullptr,
        const float * norm_input = nullptr, const float * norm_weight = nullptr, float norm_eps = 0.0f) {
    static_assert(!quant || (nw == 1 && rows*groups == 32), "quant tile");
    constexpr int qk = type == GGML_TYPE_Q8_0 ? 32 : 256;
    constexpr int lanes = type == GGML_TYPE_Q8_0 ? 4/width : type == GGML_TYPE_Q4_K ? 16/width : 32;
    constexpr int vdr = type == GGML_TYPE_Q6_K ? 1 : 2;
    constexpr int nb = k/qk;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y % nw;
    const int group = threadIdx.y/nw;
    const int tid = 32*warp + lane;
    if constexpr (fuse_norm) {
        static_assert(k == 2048 && nw*groups == 8, "norm tile");
        const int index = threadIdx.y*32+lane;
        const float4 a = ((const float4 *) norm_input)[index];
        const float4 b = ((const float4 *) norm_input)[index+256];
        float sa = a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;
        float sb = b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w;
        sa = warp_reduce_sum<32>(sa);
        sb = warp_reduce_sum<32>(sb);
        __shared__ float sums[16];
        __shared__ block_q8_1 normalized[k/32];
        if (lane == 0) {
            sums[threadIdx.y] = sa;
            sums[threadIdx.y+8] = sb;
        }
        __syncthreads();
        float sum = lane < 16 ? sums[lane] : 0.0f;
        sum = warp_reduce_sum<32>(sum);
        const float scale = rsqrtf(sum/float(k)+norm_eps);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int col = index+256*i;
            const float4 v = i == 0 ? a : b;
            const float4 w = ((const float4 *) norm_weight)[col];
            const float4 out = make_float4(scale*v.x*w.x,scale*v.y*w.y,scale*v.z*w.z,scale*v.w*w.w);
            const float amax = warp_reduce_max<8>(fmaxf(fmaxf(fabsf(out.x),fabsf(out.y)),fmaxf(fabsf(out.z),fabsf(out.w))));
            const float total = warp_reduce_sum<8>(out.x+out.y+out.z+out.w);
            const float d = amax/127.0f;
            const float inv = amax > 0 ? 1.0f/d : 0.0f;
            auto & q = normalized[col/8];
            const int offset = (col%8)*4;
            q.qs[offset] = roundf(out.x*inv);
            q.qs[offset+1] = roundf(out.y*inv);
            q.qs[offset+2] = roundf(out.z*inv);
            q.qs[offset+3] = roundf(out.w*inv);
            if (col%8 == 0) q.ds = make_half2(d,total);
        }
        __syncthreads();
        y = normalized;
    }
    const int row = (blockIdx.x*groups+group)*rows;
    float a[rows] = {}, g[rows] = {};
    #pragma unroll 1
    for (int b = tid/lanes; b < nb; b += 32*nw/lanes) {
        const int iqs = vdr*(tid%lanes);
        const block_q8_1 * act = y+b*(qk/32);
        if constexpr (type == GGML_TYPE_Q4_K) {
            const spark_q4_wide<width> activation(act,tid%lanes);
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                a[r] += activation.dot(weights,(row+r)*nb+b,tid%lanes);
                if constexpr (gated) g[r] += activation.dot(gate,(row+r)*nb+b,tid%lanes);
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            int u[2*width];
            #pragma unroll
            for (int q = 0; q < 2*width; ++q) u[q] = get_int_b4(act->qs,width*iqs+q);
            const float d = __low2float(act->ds);
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                const auto & w = ((const block_q8_0 *) weights)[(row+r)*nb+b];
                int dot = 0;
                #pragma unroll
                for (int q = 0; q < 2*width; ++q) dot = ggml_cuda_dp4a(get_int_b2(w.qs,width*iqs+q),u[q],dot);
                a[r] += float(w.d)*d*float(dot);
            }
        } else {
            const spark_q6_activation activation(act,iqs);
            #pragma unroll
            for (int r = 0; r < rows; ++r) a[r] += activation.dot(weights,(row+r)*nb+b,iqs);
        }
    }
    __shared__ float sa[groups][nw > 1 ? nw-1 : 1][rows][32];
    __shared__ float sg[groups][nw > 1 ? nw-1 : 1][rows][32];
    if constexpr (nw > 1) {
        if (warp > 0) {
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                sa[group][warp-1][r][lane] = a[r];
                if constexpr (gated) sg[group][warp-1][r][lane] = g[r];
            }
        }
        __syncthreads();
        if (warp > 0) return;
        #pragma unroll
        for (int w = 0; w < nw-1; ++w) {
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                a[r] += sa[group][w][r][lane];
                if constexpr (gated) g[r] += sg[group][w][r][lane];
            }
        }
    }
    __shared__ float output[rows*groups];
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        a[r] = warp_reduce_sum<32>(a[r]);
        if constexpr (gated) {
            g[r] = warp_reduce_sum<32>(g[r]);
            a[r] *= ggml_cuda_op_gelu_single(g[r]);
        }
        if (lane == 0) {
            const float result = a[r] + (bias ? bias[row+r] : 0.0f);
            if constexpr (!quant) dst[row+r] = result;
            if constexpr (quant) output[group*rows+r] = result;
        }
    }
    if constexpr (quant) {
        __syncthreads();
        if (threadIdx.y == 0) {
            const float v = output[lane];
            const float amax = warp_reduce_max<32>(fabsf(v));
            const float sum = warp_reduce_sum<32>(v);
            const float d = amax/127.0f;
            qdst[blockIdx.x].qs[lane] = amax == 0 ? 0 : roundf(v/d);
            if (lane == 0) qdst[blockIdx.x].ds = make_half2(d,sum);
        }
    }
}

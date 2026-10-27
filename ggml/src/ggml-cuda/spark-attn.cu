#include "spark-attn.cuh"

template<int heads>
__launch_bounds__(128, 1)
static __global__ void spark_attention_scores(
        const char * q, const char * k, const char * v, const half * mask,
        float * partial, float2 * metadata, int length, int q_stride,
        int k_row, int k_head, int v_row, int v_head, float scale) {
    const int tid = threadIdx.x;
    const int lane = tid%8;
    const int head0 = blockIdx.y*heads;
    const int chunk = (length+gridDim.x-1)/gridDim.x;
    const int begin = blockIdx.x*chunk;
    const int end = min(begin+chunk,length);
    float queries[heads][32];
    __shared__ float scores[heads][1024];
    __shared__ float scratch[4];
#pragma unroll
    for (int h = 0; h < heads; ++h) {
        const float * query = (const float *) (q+(head0+h)*q_stride);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                queries[h][8*d+j] = query[(d*8+lane)*8+j]*scale;
            }
        }
    }
    float maximum[heads];
#pragma unroll
    for (int h = 0; h < heads; ++h) {
        maximum[h] = -INFINITY;
    }
    for (int offset = 0; offset < chunk; offset += 16) {
        const int pos = begin+offset+tid/8;
        const bool valid = pos < end;
        const half * key = (const half *) (k+(head0/4)*k_head+pos*k_row);
        float dots[heads] = {};
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int4 packed = valid ? ((const int4 *) key)[d*8+lane] : make_int4(0,0,0,0);
            const half * values = (const half *) &packed;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float value = __half2float(values[j]);
#pragma unroll
                for (int h = 0; h < heads; ++h) {
                    dots[h] += value*queries[h][d*8+j];
                }
            }
        }
#pragma unroll
        for (int h = 0; h < heads; ++h) {
            const float dot = warp_reduce_sum<8>(dots[h]);
            const float score = valid ? dot+(mask ? __half2float(mask[pos]) : 0.0f) : -INFINITY;
            maximum[h] = fmaxf(maximum[h],score);
            if (lane == 0 && valid) {
                scores[h][pos-begin] = score;
            }
        }
    }
    float maxval[heads];
    float total[heads];
#pragma unroll
    for (int h = 0; h < heads; ++h) {
        maxval[h] = block_reduce<block_reduce_method::MAX,128>(maximum[h],scratch);
        __syncthreads();
        float sum = 0;
        for (int pos = tid; pos < end-begin; pos += 128) {
            const float probability = isfinite(maxval[h]) ? expf(scores[h][pos]-maxval[h]) : 0.0f;
            scores[h][pos] = probability;
            sum += probability;
        }
        total[h] = block_reduce<block_reduce_method::SUM,128>(sum,scratch);
        __syncthreads();
    }
    float result[heads][2] = {};
    for (int pos = begin; pos < end; ++pos) {
        const half * value = (const half *) (v+(head0/4)*v_head+pos*v_row);
        const float2 pair = __half22float2(((const half2 *) value)[tid]);
        const float a = pair.x;
        const float b = pair.y;
#pragma unroll
        for (int h = 0; h < heads; ++h) {
            const float probability = scores[h][pos-begin];
            result[h][0] += probability*a;
            result[h][1] += probability*b;
        }
    }
#pragma unroll
    for (int h = 0; h < heads; ++h) {
        const int offset = ((head0+h)*gridDim.x+blockIdx.x)*256;
        ((float2 *) (partial+offset))[tid] = make_float2(result[h][0],result[h][1]);
        if (tid == 0) {
            metadata[(head0+h)*gridDim.x+blockIdx.x] = make_float2(maxval[h],total[h]);
        }
    }
}

template<bool quantize = false>
static __global__ void spark_attention_combine(const float * partial, const float2 * metadata, float * dst, int parts,
        block_q8_1 * quantized = nullptr, const float * gate = nullptr) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    __shared__ float factors[64];
    __shared__ float denominator;
    if (dim < 32) {
        float maximum = -INFINITY;
        for (int p = dim; p < parts; p += 32) {
            maximum = fmaxf(maximum,metadata[head*parts+p].x);
        }
        maximum = warp_reduce_max<32>(maximum);
        float sum = 0;
        for (int p = dim; p < parts; p += 32) {
            const float2 meta = metadata[head*parts+p];
            const float factor = isfinite(meta.x) ? expf(meta.x-maximum) : 0;
            factors[p] = factor;
            sum += meta.y*factor;
        }
        sum = warp_reduce_sum<32>(sum);
        if (dim == 0) {
            denominator = sum;
        }
    }
    __syncthreads();
    float numerator = 0;
    for (int p = 0; p < parts; ++p) {
        numerator += partial[(head*parts+p)*256+dim]*factors[p];
    }
    const float value = denominator > 0 ? numerator/denominator : 0;
    if constexpr (quantize) {
        const float gated = value*gate[head];
        const float amax = warp_reduce_max<32>(fabsf(gated));
        const float sum = warp_reduce_sum<32>(gated);
        const float d = amax/127.0f;
        auto & out = quantized[head*8+dim/32];
        out.qs[dim%32] = amax == 0 ? 0 : roundf(gated/d);
        if (dim%32 == 0) out.ds = make_half2(d,sum);
    } else {
        dst[head*256+dim] = value;
    }

}

bool ggml_cuda_spark_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        void * quantized, const float * gate) {
    static const int enabled = []() {
        const char * value = getenv("GGML_CUDA_SPARK_ATTN");
        return value ? std::atoi(value) : 0;
    }();
    if (!enabled || ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    float scale, max_bias, softcap;
    memcpy(&scale,dst->op_params,sizeof(float));
    memcpy(&max_bias,(const float *) dst->op_params+1,sizeof(float));
    memcpy(&softcap,(const float *) dst->op_params+2,sizeof(float));
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
            q->ne[0] != 256 || q->ne[1] != 1 || (q->ne[2] != 16 && q->ne[2] != 8) || q->ne[3] != 1 ||
            k->ne[0] != 256 || k->ne[2] != q->ne[2]/4 || k->ne[3] != 1 || k->ne[1] > 8192 ||
            v->ne[0] != 256 || v->ne[1] != k->ne[1] || v->ne[2] != q->ne[2]/4 || v->ne[3] != 1 ||
            q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 || k->ne[1] <= 0 ||
            dst->type != GGML_TYPE_F32 || ggml_nelements(dst) != q->ne[2]*256 || !ggml_is_contiguous(dst) ||
            ((uintptr_t(k->data) | k->nb[1] | k->nb[2]) & 15) != 0 ||
            ((uintptr_t(v->data) | v->nb[1] | v->nb[2]) & 3) != 0 ||
            dst->src[4] || max_bias != 0 || softcap != 0 ||
            (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1))) {
        return false;
    }
    const bool short_context = k->ne[1] <= 1024;
    if ((enabled == 2 && !short_context) || (enabled == 3 && short_context)) {
        return false;
    }
    static const int heads = []() {
        const char * value = getenv("GGML_CUDA_SPARK_ATTN_HEADS");
        return value ? std::atoi(value) : 1;
    }();
    static const int requested_parts = []() {
        const char * value = getenv("GGML_CUDA_SPARK_ATTN_PARTS");
        return value ? std::atoi(value) : 0;
    }();
    const int chosen_parts = requested_parts > 0 ? requested_parts : short_context ? 16 : 32;
    const int parts = std::min<int>(std::max<int>((k->ne[1]+1023)/1024,chosen_parts),64);
    ggml_cuda_pool_alloc<float> partial(ctx.pool(),q->ne[2]*256*parts);
    ggml_cuda_pool_alloc<float2> metadata(ctx.pool(),q->ne[2]*parts);
    const auto launch = [&](auto group) {
        constexpr int nh = decltype(group)::value;
        spark_attention_scores<nh><<<dim3(parts,q->ne[2]/nh),128,0,ctx.stream()>>>(
            (const char *) q->data,(const char *) k->data,(const char *) v->data,mask ? (const half *) mask->data : nullptr,
            partial.get(),metadata.get(),k->ne[1],q->nb[2],k->nb[1],k->nb[2],v->nb[1],v->nb[2],scale);
    };
    switch (heads) {
        case 2: launch(std::integral_constant<int,2>{}); break;
        case 4: launch(std::integral_constant<int,4>{}); break;
        default: launch(std::integral_constant<int,1>{}); break;
    }
    if (quantized && gate) {
        spark_attention_combine<true><<<q->ne[2],256,0,ctx.stream()>>>(partial.get(),metadata.get(),nullptr,parts,(block_q8_1 *) quantized,gate);
    } else {
        spark_attention_combine<false><<<q->ne[2],256,0,ctx.stream()>>>(partial.get(),metadata.get(),(float *) dst->data,parts);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

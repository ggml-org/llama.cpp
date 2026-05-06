#include "fwht.cuh"

template <int N>
static __global__ void fwht_f32_kernel(
    const char * src1_data, char * dst_data,
    int64_t ne11, int64_t ne12,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3,
    float scale)
{
    __shared__ float smem[N];

    const int64_t r = blockIdx.x;
    const int tid = threadIdx.x;

    const int64_t i13 = r / (ne11 * ne12);
    const int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
    const int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

    const float * src_row = (const float *) (src1_data + i11 * nb11 + i12 * nb12 + i13 * nb13);
    float * dst_row       = (float *)       (dst_data  + i11 * nb1  + i12 * nb2  + i13 * nb3);

    smem[tid]         = src_row[tid] * scale;
    smem[tid + N / 2] = src_row[tid + N / 2] * scale;
    __syncthreads();

    #pragma unroll
    for (int len = 1; len < N; len <<= 1) {
        int pair_dist = len;
        int group = tid / pair_dist;
        int local = tid % pair_dist;

        int i = group * pair_dist * 2 + local;
        int j = i + pair_dist;

        float u = smem[i];
        float v = smem[j];

        smem[i] = u + v;
        smem[j] = u - v;

        __syncthreads();
    }

    dst_row[tid]         = smem[tid];
    dst_row[tid + N / 2] = smem[tid + N / 2];
}

void ggml_cuda_forward_fwht_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t n = src1->ne[0];
    GGML_ASSERT((n & (n - 1)) == 0);
    GGML_ASSERT(n <= 1024);

    const int64_t nr = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const float scale = 1.0f / sqrtf((float)n);

    const char * src1_data = (const char *) src1->data;
    char * dst_data = (char *) dst->data;

    cudaStream_t stream = ctx.stream();

    dim3 blocks(nr);
    dim3 threads(n / 2);

    switch (n) {
        case 64:
            fwht_f32_kernel<64><<<blocks, threads, 0, stream>>>(
                src1_data, dst_data, src1->ne[1], src1->ne[2],
                src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], scale);
            break;
        case 128:
            fwht_f32_kernel<128><<<blocks, threads, 0, stream>>>(
                src1_data, dst_data, src1->ne[1], src1->ne[2],
                src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], scale);
            break;
        case 256:
            fwht_f32_kernel<256><<<blocks, threads, 0, stream>>>(
                src1_data, dst_data, src1->ne[1], src1->ne[2],
                src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3], scale);
            break;
        default:
            GGML_ABORT("FWHT CUDA: unsupported n %lld\n", (long long)n);
    }
}

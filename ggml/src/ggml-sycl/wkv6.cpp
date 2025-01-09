#include <sycl/sycl.hpp>
#include "wkv6.hpp"

constexpr int WKV_BLOCK_SIZE = 64;  // Matching CUDA_WKV_BLOCK_SIZE

// Helper function for the main kernel
static void rwkv_wkv_f32_kernel(
    const int B, const int T, const int C, const int H,
    const float* k, const float* v, const float* r,
    const float* tf, const float* td, const float* s,
    float* dst, const sycl::nd_item<3>& item_ct1, float* shared_mem) {

    const int tid = item_ct1.get_local_id(2);
    const int bid = item_ct1.get_group(2);

    const int head_size = WKV_BLOCK_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    // Set up shared memory pointers
    float* _k = shared_mem;
    float* _r = _k + head_size;
    float* _tf = _r + head_size;
    float* _td = _tf + head_size;

    // Local state array
    float state[WKV_BLOCK_SIZE];

    // Load initial state
    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        const float _v = v[t];
        float y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            // Load data in vec4 chunks
            k4 = sycl::float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            r4 = sycl::float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            tf4 = sycl::float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            td4 = sycl::float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            // Compute key-value product
            sycl::float4 kv4 = k4 * _v;

            // Accumulate weighted sum
            y += sycl::dot(r4, tf4 * kv4 + s4);

            // Update state
            s4 = s4 * td4 + kv4;

            // Store updated state
            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    // Save final state
    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {

    const float* k_d = (const float*)dst->src[0]->data;
    const float* v_d = (const float*)dst->src[1]->data;
    const float* r_d = (const float*)dst->src[2]->data;
    const float* tf_d = (const float*)dst->src[3]->data;
    const float* td_d = (const float*)dst->src[4]->data;
    const float* s_d = (const float*)dst->src[5]->data;
    float* dst_d = (float*)dst->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[3];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[2];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    const size_t shared_mem_size = WKV_BLOCK_SIZE * 4 * sizeof(float); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shared_mem_acc(shared_mem_size, cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rwkv_wkv_f32_kernel(
                    B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                    item_ct1, (float*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
    });

    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
}

#include "ssm_conv.hpp"
#include "common.hpp"

using namespace sycl;

// SSM_CONV kernel: State Space Model Convolution 1D
// This implements a sliding window convolution with history context
static void kernel_ssm_conv(
    queue &q,
    const float *src_data,    // input sequence [d_conv-1+n_t, d_inner, n_s]
    const float *weights,     // convolution weights [d_conv, d_inner]
    float *dst_data,          // output [d_inner, n_t, n_s]
    int d_conv,               // convolution window size
    int d_inner,              // number of inner channels
    int n_t,                  // number of tokens to process
    int n_s,                  // batch size (number of sequences)
    int src_stride_inner,     // stride between channels in src
    int src_stride_seq,       // stride between sequences in src
    int dst_stride_token,     // stride between tokens in dst
    int dst_stride_seq        // stride between sequences in dst
) {
    // Each work item handles one (channel, token, sequence) combination
    const size_t total_work = d_inner * n_t * n_s;
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;
    
    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(global_range, local_range), [=](nd_item<1> item) {
            const size_t idx = item.get_global_id(0);
            
            if (idx >= total_work) return;
            
            // Decode indices: idx = seq * (d_inner * n_t) + token * d_inner + channel
            const int channel = idx % d_inner;
            const int token = (idx / d_inner) % n_t;
            const int seq = idx / (d_inner * n_t);
            
            // Calculate input starting position for this token and channel
            // Input layout: [d_conv-1+n_t, d_inner, n_s]
            // We start from token position and take d_conv elements in dim 0
            const float *input_base = src_data + seq * src_stride_seq + channel * src_stride_inner;
            
            // Get weights for this channel
            // Weights layout: [d_conv, d_inner]
            const float *channel_weights = weights + channel * d_conv;
            
            // Perform dot product: sum(input_window * weights)
            float sum = 0.0f;
            for (int i = 0; i < d_conv; i++) {
                // Access input at position (token + i, channel, seq)
                sum += input_base[token + i] * channel_weights[i];
            }
            
            // Write result to output
            const size_t dst_idx = seq * dst_stride_seq + 
                                  token * dst_stride_token + 
                                  channel;
            dst_data[dst_idx] = sum;
        });
    });
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // Extract dimensions
    const int d_conv = src1->ne[0];         // convolution window size
    const int d_inner = src1->ne[1];        // number of inner channels
    const int n_t = dst->ne[1];             // number of tokens to process
    const int n_s = dst->ne[2];             // batch size
    
    // Verify dimensions match expectations
    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t); // input length
    GGML_ASSERT(src0->ne[1] == d_inner);           // channels match
    GGML_ASSERT(dst->ne[0] == d_inner);            // output channels
    
    // Calculate strides based on tensor layout
    // src0: [d_conv-1+n_t, d_inner, n_s] - input sequence
    const int src_stride_inner = src0->ne[0];                    // stride between channels in elements
    const int src_stride_seq = src0->ne[0] * src0->ne[1];       // stride between sequences in elements
    
    // dst: [d_inner, n_t, n_s] - output
    const int dst_stride_token = dst->ne[0];                    // stride between tokens in elements  
    const int dst_stride_seq = dst->ne[0] * dst->ne[1];         // stride between sequences in elements

    try {
        queue *q = ctx.stream();

        const float *src_data = (const float *) src0->data;
        const float *weights = (const float *) src1->data;
        float *dst_data = (float *) dst->data;
        
        GGML_ASSERT(src_data && weights && dst_data);

        // Launch kernel
        kernel_ssm_conv(
            *q, src_data, weights, dst_data,
            d_conv, d_inner, n_t, n_s,
            src_stride_inner, src_stride_seq,
            dst_stride_token, dst_stride_seq
        );
        
        // Wait for completion
        q->wait();
        
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}
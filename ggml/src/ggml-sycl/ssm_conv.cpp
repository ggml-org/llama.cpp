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
    int ncs __attribute__((unused)),                  // input sequence length (d_conv-1+n_t)
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
            // Following CPU implementation: s[i0 + i1*ncs] where i0 is conv position, i1 is channel
            // Note: s pointer is offset by token position for sliding window
            const float *s = src_data + seq * src_stride_seq + channel * src_stride_inner + token;
            
            // Get weights for this channel
            // Weights layout: [d_conv, d_inner]  
            // Following CPU implementation: c[i0 + i1*nc] where i0 is conv position, i1 is channel
            const float *c = weights + channel * d_conv;
            
            // Perform dot product: sum(input_window * weights)
            // Following CPU implementation exactly
            float sumf = 0.0f;
            for (int i0 = 0; i0 < d_conv; ++i0) {
                sumf += s[i0] * c[i0];  // s[i0 + i1*ncs] * c[i0 + i1*nc] 
            }
            
            // Write result to output
            // Output layout: [d_inner, n_t, n_s]
            const size_t dst_idx = seq * dst_stride_seq + 
                                  token * dst_stride_token + 
                                  channel;
            dst_data[dst_idx] = sumf;
        });
    });
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // conv_x: input sequence  
    ggml_tensor * src1 = dst->src[1];  // conv1d.weight: convolution weights

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // Extract dimensions following CPU implementation
    const int d_conv = src1->ne[0];         // convolution window size
    const int ncs = src0->ne[0];            // d_conv - 1 + n_t (input sequence length)  
    const int d_inner = src0->ne[1];        // number of inner channels
    const int n_t = dst->ne[1];             // number of tokens to process
    const int n_s = dst->ne[2];             // batch size (number of sequences)
    
    // Verify dimensions match CPU implementation exactly
    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t); // input length
    GGML_ASSERT(src0->ne[1] == d_inner);           // channels match
    GGML_ASSERT(src1->ne[1] == d_inner);           // weight channels match
    GGML_ASSERT(dst->ne[0] == d_inner);            // output channels
    GGML_ASSERT(dst->ne[1] == n_t);                // output tokens
    GGML_ASSERT(dst->ne[2] == n_s);                // output sequences
    
    // Verify stride assumptions (from CPU implementation)
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));  
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    
    // Calculate strides based on tensor layout (in elements, not bytes)
    // src0: [d_conv-1+n_t, d_inner, n_s] - input sequence
    const int src_stride_inner = ncs;                            // stride between channels in elements
    const int src_stride_seq = ncs * d_inner;                   // stride between sequences in elements
    
    // dst: [d_inner, n_t, n_s] - output  
    const int dst_stride_token = d_inner;                       // stride between tokens in elements
    const int dst_stride_seq = d_inner * n_t;                   // stride between sequences in elements

    try {
        queue *q = ctx.stream();

        const float *src_data = (const float *) src0->data;
        const float *weights = (const float *) src1->data;
        float *dst_data = (float *) dst->data;
        
        GGML_ASSERT(src_data && weights && dst_data);
        
        // Launch kernel
        kernel_ssm_conv(
            *q, src_data, weights, dst_data,
            d_conv, d_inner, n_t, n_s, ncs,
            src_stride_inner, src_stride_seq,
            dst_stride_token, dst_stride_seq
        );
        
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}
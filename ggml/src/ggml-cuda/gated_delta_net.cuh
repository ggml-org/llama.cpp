#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

#define GDN_AB_FUSION_MAX_TOKENS 8 // Max. number of tokens for which to fuse the GDN alpha/beta projections.

// the alpha/beta projections that feed gated_delta_net
struct ggml_cuda_gated_delta_net_ab {
    const ggml_tensor * alpha_mm;
    const ggml_tensor * beta_mm;
    const ggml_tensor * dt;
    const ggml_tensor * ssm_a;
    ggml_tensor *       gate_out;
    ggml_tensor *       beta_out;
};

bool ggml_cuda_gated_delta_net_ab_supported_type(ggml_type type);

// runs both projections and their activation tails in one kernel
void ggml_cuda_op_gated_delta_net_ab(ggml_backend_cuda_context & ctx, const ggml_cuda_gated_delta_net_ab & ab);

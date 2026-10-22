#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
    // with state_ids, the initial state is row state_ids[seq] of state_rows (row stride in floats) instead of src[5]:
    // the gather build_rs makes of it is skipped (see ggml_cuda_gdn_state_rows_plan)
    const float *   state_rows       = nullptr;
    const int32_t * state_ids        = nullptr;
    int64_t         state_row_stride = 0;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst plus the scratch of the MMA path that takes it, reserved with the graph
size_t ggml_cuda_gated_delta_net_get_alloc_size(int device, const ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

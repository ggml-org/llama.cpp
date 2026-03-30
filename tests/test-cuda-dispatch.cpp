#include <cassert>

#include <ggml.h>

#include "../ggml/src/ggml-cuda/mmvq.cuh"

int main() {
    assert(ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, 1, false));
    assert(ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F32, MMVQ_MAX_BATCH_SIZE, false));

    assert(ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_TQ3_0, GGML_TYPE_F32, GGML_TYPE_F32, 1, false));
    assert(!ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q4_0, GGML_TYPE_F16, GGML_TYPE_F32, 1, false));
    assert(!ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F16, 1, false));
    assert(!ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, MMVQ_MAX_BATCH_SIZE + 1, false));
    assert(!ggml_cuda_can_use_mul_mat_vec_q(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, 1, true));

    return 0;
}

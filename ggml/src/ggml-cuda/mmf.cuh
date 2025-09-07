#include "common.cuh"

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * scr0_ne, const int src1_ncols);

template <ggml_type type>
void mul_mat_f_case(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == type);
    ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
}

#define DECL_MMF_CASE(type)                                                        \
    template void mul_mat_f_case<type>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)

extern DECL_MMF_CASE(GGML_TYPE_F32);
extern DECL_MMF_CASE(GGML_TYPE_F16);
extern DECL_MMF_CASE(GGML_TYPE_BF16);

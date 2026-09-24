#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// per-element body shared by the single and the fused quantized kernels
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __device__ __forceinline__ void set_rows_quant_one(const float * __restrict__ src0,
                                                          const idx_t * __restrict__ src1,
                                                          block_type * __restrict__ dst,
                                                          const int64_t i,
                                                          const int64_t s01,
                                                          const int64_t s02,
                                                          const int64_t s03,
                                                          const int64_t s10,
                                                          const int64_t s11,
                                                          const int64_t s12,
                                                          const int64_t s1,
                                                          const int64_t s2,
                                                          const int64_t s3,
                                                          const uint3   ne00,
                                                          const uint3   ne01,
                                                          const uint3   ne02,
                                                          const uint3   ne11_fd,
                                                          const uint3   ne12_fd) {
    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);
}

// Generic quantized set_rows kernel template
// lanes_per_qblock > 1 maps that many threads to one quant block for a warp-cooperative quantize_func
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *), int lanes_per_qblock = 1>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = (int64_t(blockDim.x) * blockIdx.x + threadIdx.x) / lanes_per_qblock;

    // warp-uniform when lanes_per_qblock is a whole number of warps, so full-mask shuffles below stay safe
    if (i >= ne_total) {
        return;
    }

    set_rows_quant_one<idx_t, block_type, qk, quantize_func>(src0, src1, dst, i,
        s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00, ne01, ne02, ne11_fd, ne12_fd);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// pointers and strides for two same-shape, same-type SET_ROWS done in one launch
template <typename idx_t, typename block_type>
struct set_rows_fused_args {
    const float * src0[2];
    const idx_t * src1[2];
    block_type  * dst[2];
    int64_t       s01[2], s02[2], s03[2];
    int64_t       s10[2], s11[2], s12[2];
    int64_t       s1[2], s2[2], s3[2];
};

template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *), int lanes_per_qblock = 1>
static __global__ void k_set_rows_quant_fused(const set_rows_fused_args<idx_t, block_type> args,
                                              const int64_t ne_total,
                                              const uint3   ne00,
                                              const uint3   ne01,
                                              const uint3   ne02,
                                              const uint3   ne11_fd,
                                              const uint3   ne12_fd) {
    const int64_t i = (int64_t(blockDim.x) * blockIdx.x + threadIdx.x) / lanes_per_qblock;

    // warp-uniform when lanes_per_qblock is a whole number of warps, so full-mask shuffles below stay safe
    if (i >= 2*ne_total) {
        return;
    }

    // both jobs have the same shape, so the job split is also warp-uniform
    const int job = i >= ne_total;

    set_rows_quant_one<idx_t, block_type, qk, quantize_func>(
        args.src0[job], args.src1[job], args.dst[job], i - job*ne_total,
        args.s01[job], args.s02[job], args.s03[job],
        args.s10[job], args.s11[job], args.s12[job],
        args.s1[job], args.s2[job], args.s3[job],
        ne00, ne01, ne02, ne11_fd, ne12_fd);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*), int lanes_per_qblock = 1>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total*lanes_per_qblock + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func, lanes_per_qblock><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*), int lanes_per_qblock = 1>
static void set_rows_cuda_quant_fused(const ggml_tensor * dst0, const ggml_tensor * dst1, cudaStream_t stream) {
    const ggml_tensor * src0 = dst0->src[0];
    const ggml_tensor * src1 = dst0->src[1];

    GGML_ASSERT(src0->ne[0] % qk == 0);
    const int64_t ne_total   = ggml_nelements(src0) / qk;
    const int     num_blocks = (2*ne_total*lanes_per_qblock + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;

    set_rows_fused_args<idx_t, block_type> args;

    const ggml_tensor * dsts[2] = { dst0, dst1 };
    for (int j = 0; j < 2; ++j) {
        const ggml_tensor * s0 = dsts[j]->src[0];
        const ggml_tensor * s1 = dsts[j]->src[1];

        args.src0[j] = (const float *) s0->data;
        args.src1[j] = (const idx_t *) s1->data;
        args.dst[j]  = (block_type  *) dsts[j]->data;

        args.s01[j] = s0->nb[1]/sizeof(float);
        args.s02[j] = s0->nb[2]/sizeof(float);
        args.s03[j] = s0->nb[3]/sizeof(float);
        args.s10[j] = s1->nb[0]/sizeof(idx_t);
        args.s11[j] = s1->nb[1]/sizeof(idx_t);
        args.s12[j] = s1->nb[2]/sizeof(idx_t);
        args.s1[j]  = dsts[j]->nb[1];
        args.s2[j]  = dsts[j]->nb[2];
        args.s3[j]  = dsts[j]->nb[3];
    }

    if (ne_total > 0 && src0->ne[0] > 0 && src0->ne[1] > 0 && src0->ne[2] > 0 && src1->ne[1] > 0 && src1->ne[2] > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) src0->ne[0]);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) src0->ne[1]);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) src0->ne[2]);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) src1->ne[1]);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) src1->ne[2]);

        k_set_rows_quant_fused<idx_t, block_type, qk, quantize_func, lanes_per_qblock>
            <<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                args, ne_total, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template<typename idx_t>
static void set_rows_cuda_fused(const ggml_tensor * dst0, const ggml_tensor * dst1, cudaStream_t stream) {
    switch (dst0->type) {
        case GGML_TYPE_Q4_0:
            set_rows_cuda_quant_fused<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(dst0, dst1, stream);
            break;
        case GGML_TYPE_Q4_1:
            set_rows_cuda_quant_fused<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(dst0, dst1, stream);
            break;
        case GGML_TYPE_Q4_H:
            // one warp per block, same as the single-node path
            set_rows_cuda_quant_fused<idx_t, block_q4_h, QK4_H, quantize_f32_q4_h_block_warp, WARP_SIZE>(dst0, dst1, stream);
            break;
        case GGML_TYPE_Q5_0:
            set_rows_cuda_quant_fused<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(dst0, dst1, stream);
            break;
        case GGML_TYPE_Q5_1:
            set_rows_cuda_quant_fused<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(dst0, dst1, stream);
            break;
        case GGML_TYPE_Q8_0:
            set_rows_cuda_quant_fused<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(dst0, dst1, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            set_rows_cuda_quant_fused<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(dst0, dst1, stream);
            break;
        default:
            GGML_ABORT("unsupported type %s", ggml_type_name(dst0->type));
    }
}

static bool set_rows_fused_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_H:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_should_fuse_set_rows_pair(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_SET_ROWS || b->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // the fused kernel runs one dst type and one shape; anything else keeps two launches
    if (a->type != b->type || !set_rows_fused_type_supported(a->type)) {
        return false;
    }
    if (a->src[0]->type != GGML_TYPE_F32 || b->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (a->src[1]->type != b->src[1]->type) {
        return false;
    }
    // both nodes are fusion outputs, so ggml_can_fuse_subgraph_ext skips its use-count check for them
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (b->src[i] == a) {
            return false;
        }
    }
    // the two writes run concurrently, so they must not touch the same memory
    const int64_t a_start = (int64_t) a->data;
    const int64_t b_start = (int64_t) b->data;
    if (a_start < b_start + (int64_t) ggml_nbytes(b) && b_start < a_start + (int64_t) ggml_nbytes(a)) {
        return false;
    }
    return ggml_are_same_shape(a->src[0], b->src[0]) && ggml_are_same_shape(a->src[1], b->src[1]);
}

void ggml_cuda_op_set_rows_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst0, ggml_tensor * dst1) {
    cudaStream_t stream = ctx.stream();

    if (dst0->src[1]->type == GGML_TYPE_I64) {
        set_rows_cuda_fused<int64_t>(dst0, dst1, stream);
    } else {
        set_rows_cuda_fused<int32_t>(dst0, dst1, stream);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_H) {
        // one warp per block: one thread per block starves the GPU at decode and lane stragglers cancel the solver's early exit
        set_rows_cuda_quant<idx_t, block_q4_h, QK4_H, quantize_f32_q4_h_block_warp, WARP_SIZE>(
            src0_d, src1_d, (block_q4_h*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

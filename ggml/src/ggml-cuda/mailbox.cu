#include "mailbox.cuh"

#include "ggml-mailbox.h"

#define MAILBOX_BLOCK 256

// one block: store the payload into the host memory, make the stores visible to the host, then raise the flag, so a host
// that sees the flag sees the payload
static __global__ void mailbox_send(const float * __restrict__ x, const int32_t * __restrict__ ids,
        float * __restrict__ mb_f32, int32_t * __restrict__ mb_i32, volatile uint32_t * flag, const volatile uint32_t * epoch,
        const int n_f32, const int n_i32) {
    for (int i = threadIdx.x; i < n_f32; i += blockDim.x) {
        mb_f32[i] = x[i];
    }
    for (int i = threadIdx.x; i < n_i32; i += blockDim.x) {
        mb_i32[i] = ids[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = *epoch;
        __threadfence_system();
    }
}

// the host wrote the payload before this kernel was launched: plain volatile loads see it
static __global__ void mailbox_recv(const volatile float * __restrict__ mb_f32, float * __restrict__ dst, const int n) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        dst[i] = mb_f32[i];
    }
}

static const ggml_mailbox_op * mailbox_op_of(const ggml_tensor * op) {
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) op->op_params;
    if (p->fun != ggml_mailbox_cpu_stub || p->userdata == nullptr) {
        return nullptr;
    }
    const ggml_mailbox_op * m = (const ggml_mailbox_op *) p->userdata;
    return m->magic == GGML_MAILBOX_MAGIC ? m : nullptr;
}

bool ggml_cuda_mailbox_op_supported(const ggml_tensor * op) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(op);
    return false;
#else
    const ggml_mailbox_op * m = mailbox_op_of(op);
    if (m == nullptr) {
        return false;
    }
    switch (m->kind) {
        case GGML_MAILBOX_SEND:
            return op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                   op->src[0]->ne[0]*op->src[0]->ne[1]*op->src[0]->ne[2]*op->src[0]->ne[3] == m->n_f32 &&
                   op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[1]) &&
                   ggml_nelements(op->src[1]) == m->n_i32;
        case GGML_MAILBOX_RECV:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) == m->n_f32;
        default:
            return false;
    }
#endif
}

void ggml_cuda_op_mailbox(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("mailbox ops need the CUDA backend");
#else
    const ggml_mailbox_op * m = mailbox_op_of(dst);
    GGML_ASSERT(m != nullptr);
    cudaStream_t stream = ctx.stream();

    switch (m->kind) {
        case GGML_MAILBOX_SEND:
            mailbox_send<<<1, MAILBOX_BLOCK, 0, stream>>>((const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
                    (float *) m->dev_f32, (int32_t *) m->dev_i32, m->dev_flag, m->dev_epoch, m->n_f32, m->n_i32);
            break;
        case GGML_MAILBOX_RECV:
            mailbox_recv<<<(m->n_f32 + MAILBOX_BLOCK - 1)/MAILBOX_BLOCK, MAILBOX_BLOCK, 0, stream>>>(
                    (const volatile float *) m->dev_f32, (float *) dst->data, m->n_f32);
            break;
        default:
            GGML_ABORT("unknown mailbox op");
    }
#endif
}

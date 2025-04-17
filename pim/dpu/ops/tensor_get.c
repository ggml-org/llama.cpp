#include "tensor_get.h"

#define OP_TENSOR_GET_DEBUG_PRINT 0

static pim_tensor_des *src0 = NULL;

void tensor_get_prepare(msg_block_header *header_ptr)
{
    src0 = &header_ptr->src0;

#if OP_TENSOR_GET_DEBUG_PRINT
    printf("src0.dpu_addr: %d, ne0: %d, ne1: %d\n", header_ptr->src0.ptr.dpu_addr,
           header_ptr->src0.ne[0], header_ptr->src0.ne[1]);
#endif
}

void tensor_get_tasklets_run()
{
}

void tensor_get_merge()
{
    __mram_ptr int32_t *src0_addr = (__mram_ptr int32_t *)(DPU_MRAM_HEAP_POINTER + src0->ptr.dpu_addr);
    __mram_ptr int32_t *dst = (__mram_ptr int32_t *)RESULT_BUFFER_ADDR;
    for (int i = 0; i < src0->ne[0] * src0->ne[1]; i++)
    {
        dst[i] = src0_addr[i];
    }
}
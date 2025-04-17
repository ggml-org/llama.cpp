#include "tensor_add.h"

#define OP_TENSOR_ADD_DEBUG_PRINT 0

static pim_tensor_des *src0 = NULL;
static int32_t *num = 0;

void tensor_add_prepare(msg_block_header *header_ptr)
{
    src0 = &header_ptr->src0;
    num = (int32_t *)((char *)header_ptr + sizeof(msg_block_header));

#if OP_TENSOR_ADD_DEBUG_PRINT
    printf("src0.dpu_addr: %d, ne0: %d, ne1: %d, num: %d\n", header_ptr->src0.ptr.dpu_addr,
           header_ptr->src0.ne[0], header_ptr->src0.ne[1], *num);
#endif

    __mram_ptr int32_t *src0_addr = (__mram_ptr int32_t *)(DPU_MRAM_HEAP_POINTER + header_ptr->src0.ptr.dpu_addr);
    for (int i = 0; i < header_ptr->src0.ne[0] * header_ptr->src0.ne[1]; i++)
    {
        src0_addr[i] += *num;
    }
}

void tensor_add_tasklets_run()
{
}

void tensor_add_merge()
{
}
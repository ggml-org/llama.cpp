#include "msg_comm.h"
#include <assert.h>

void msg_buffer_send(msg_buffer *builder, struct dpu_set_t dpu_set)
{
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, MESSAGE_BUFFER_OFFSET, builder->buffer,
                                builder->header_ptr->total_size, DPU_XFER_DEFAULT));
}

void msg_buffer_recv(struct dpu_set_t dpu_set, void **dst, uint32_t size_per_dpu)
{
    assert((size_per_dpu % 8 == 0) && (size_per_dpu <= RESULT_BUFFER_SIZE));
    struct dpu_set_t dpu;
    uint32_t each_dpu;

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, dst[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, RESULT_BUFFER_OFFSET, size_per_dpu, DPU_XFER_DEFAULT));
}
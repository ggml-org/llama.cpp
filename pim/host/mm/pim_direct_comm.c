#include "pim_direct_comm.h"
#include <assert.h>

void dpu_send_direct(struct dpu_set_t dpu_set, remote_ptr dst, void **src, uint32_t size_per_dpu)
{
    assert(size_per_dpu % 8 == 0);
    struct dpu_set_t dpu;
    uint32_t each_dpu;

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, src[each_dpu]));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                             dst.dpu_addr, size_per_dpu, DPU_XFER_DEFAULT));
}

void dpu_recv_direct(struct dpu_set_t dpu_set, remote_ptr src, void **dst, uint32_t size_per_dpu)
{
    assert(size_per_dpu % 8 == 0);
    struct dpu_set_t dpu;
    uint32_t each_dpu;

    DPU_FOREACH(dpu_set, dpu, each_dpu)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, dst[each_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, src.dpu_addr, size_per_dpu, DPU_XFER_DEFAULT));
}

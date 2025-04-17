#include "pim_mm.h"
#include <assert.h>

static uint32_t dpu_free_storage_offset = FREE_STORAGE_OFFSET;

void all_dpu_mm_reset()
{
    dpu_free_storage_offset = FREE_STORAGE_OFFSET;
}

remote_ptr all_dpu_alloc(uint32_t size_per_dpu)
{
    remote_ptr ptr = {ALL_DPU, dpu_free_storage_offset};
    dpu_free_storage_offset += align8(size_per_dpu);
    assert(dpu_free_storage_offset < 63 * 1024 * 1024 && "Memory alloc fault: insufficient free memory.");
    return ptr;
}
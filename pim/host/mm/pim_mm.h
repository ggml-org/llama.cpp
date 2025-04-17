#ifndef _PIM_MM_H
#define _PIM_MM_H
#include <stdint.h>
#include "../util/util.h"

#define MESSAGE_BUFFER_OFFSET 0
#define MESSAGE_BUFFER_SIZE 0x100000 // 1MB

#define RESULT_BUFFER_OFFSET (MESSAGE_BUFFER_OFFSET + MESSAGE_BUFFER_SIZE)
#define RESULT_BUFFER_SIZE 0x100000 // 1MB

#define FREE_STORAGE_OFFSET (RESULT_BUFFER_OFFSET + RESULT_BUFFER_SIZE)

#define ALIGN8 __attribute__((aligned(8)))

#define ALL_DPU -1
#define DPU_NULL_PTR 0

typedef struct
{
    int32_t dpu_id;
    uint32_t dpu_addr;
} remote_ptr;

void all_dpu_mm_reset();
remote_ptr all_dpu_alloc(uint32_t size_per_dpu);

#endif
#ifndef _MRAM_MM_H
#define _MRAM_MM_H

#include <mram.h>
#include "../../host/mm/pim_mm.h"

#define MESSAGE_BUFFER_ADDR (DPU_MRAM_HEAP_POINTER)
#define RESULT_BUFFER_ADDR (MESSAGE_BUFFER_ADDR + MESSAGE_BUFFER_SIZE)
#define FREE_STORAGE_ADDR (RESULT_BUFFER_ADDR + RESULT_BUFFER_SIZE)

int mram2wram(__mram_ptr void *pmram,void *pwram,uint32_t size);
int wram2mram(__mram_ptr void *pmram,void *pwram,uint32_t size);

#endif
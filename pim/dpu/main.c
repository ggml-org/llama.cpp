#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#include "./sto/g_vars.h"
#include "./sto/mram_mm.h"
#include "./ops/gemv.h"
#include "./ops/tensor_add.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#define DPU_MAIN_DEBUG_PRINT 0

BARRIER_INIT(my_barrier, NR_TASKLETS);

static void init()
{
    mem_reset(); // reset heap in wram
    mram_read((__mram_ptr void const *)(MESSAGE_BUFFER_ADDR), &g_msg_header, sizeof(msg_buffer_header));

#if DPU_MAIN_DEBUG_PRINT
    printf("block_cnt: %d, total_size: %d\n", g_msg_header.block_cnt, g_msg_header.total_size);
#endif

    assert(g_msg_header.total_size <= 16 * 1024);
    g_msg_cache_ptr = (char *)mem_alloc(g_msg_header.total_size);
    mram2wram((__mram_ptr void const *)(MESSAGE_BUFFER_ADDR), g_msg_cache_ptr, g_msg_header.total_size);
    g_blocks_offset_ptr = (uint32_t *)(g_msg_cache_ptr + g_msg_header.total_size - align8(sizeof(uint32_t) * g_msg_header.block_cnt));

#if DPU_MAIN_DEBUG_PRINT
    for (int i = 0; i < g_msg_header.block_cnt; i++)
    {
        printf("blocks_offset: %d ", g_blocks_offset_ptr[i]);
    }
    printf("\n");
#endif
}

int main()
{
    unsigned int tasklet_id = me();

    if (tasklet_id == 0)
    {
        init();
    }

    barrier_wait(&my_barrier);

    for (int i = 0; i < g_msg_header.block_cnt; i++)
    {
        msg_block_header *header_ptr = (msg_block_header *)(g_msg_cache_ptr + g_blocks_offset_ptr[i]);
#if DPU_MAIN_DEBUG_PRINT
        if (tasklet_id == 0)
        {
            printf("header_ptr->op: %d, src0.dpu_addr: %d, ne0: %d, ne1: %d\n", header_ptr->op, header_ptr->src0.ptr.dpu_addr, 
            header_ptr->src0.ne[0], header_ptr->src0.ne[1]);
        }
#endif
        switch (header_ptr->op)
        {
        case PIM_OP_GEMV:
            if (tasklet_id == 0)
            {
                gemv_prepare();
            }
            barrier_wait(&my_barrier);
            gemv_tasklets_run();
            barrier_wait(&my_barrier);
            if (tasklet_id == 0)
            {
                gemv_merge();
            }
            break;

        case PIM_OP_TENSOR_ADD_FOR_TEST:
            if (tasklet_id == 0)
            {
                tensor_add_prepare(header_ptr);
            }
            barrier_wait(&my_barrier);
            tensor_add_tasklets_run();
            barrier_wait(&my_barrier);
            if (tasklet_id == 0)
            {
                tensor_add_merge();
            }
            break;

        case PIM_OP_TENSOR_GET_FOR_TEST:
            if (tasklet_id == 0)
            {
                tensor_get_prepare(header_ptr);
            }
            barrier_wait(&my_barrier);
            tensor_get_tasklets_run();
            barrier_wait(&my_barrier);
            if (tasklet_id == 0)
            {
                tensor_get_merge();
            }
            break;

        default:
            break;
        }
    }
    return 0;
}

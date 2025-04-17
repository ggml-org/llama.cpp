#include "test_case.h"
#include "./mm/pim_direct_comm.h"
#include <assert.h>
#include <stdio.h>

#define TEST_CASE_NR_DPUS 64

#ifndef TEST_CASE_DPU_BINARY
#define TEST_CASE_DPU_BINARY "./dpu_task"
#endif

#define TEST_CASE_DEBUG_PRINT 0

void test_case_mm(void)
{
    all_dpu_mm_reset();
    remote_ptr ptr0 = all_dpu_alloc(91);
    assert(ptr0.dpu_id == ALL_DPU && ptr0.dpu_addr == FREE_STORAGE_OFFSET);
    remote_ptr ptr1 = all_dpu_alloc(1024);
    assert(ptr1.dpu_id == ALL_DPU && ptr1.dpu_addr == FREE_STORAGE_OFFSET + align8(91));
    all_dpu_mm_reset();
    ptr0 = all_dpu_alloc(1024);
    assert(ptr0.dpu_id == ALL_DPU && ptr0.dpu_addr == FREE_STORAGE_OFFSET);
    ptr1 = all_dpu_alloc(92);
    assert(ptr1.dpu_id == ALL_DPU && ptr1.dpu_addr == FREE_STORAGE_OFFSET + 1024);

    // push a matrix to dpu set
    int32_t nums[TEST_CASE_NR_DPUS][TEST_CASE_NR_DPUS];
    int cnt = 0;
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        for (int j = 0; j < TEST_CASE_NR_DPUS; j++)
        {
            nums[i][j] = cnt;
            cnt++;
        }
    }

    struct dpu_set_t dpu_set;
    DPU_ASSERT(dpu_alloc(TEST_CASE_NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, TEST_CASE_DPU_BINARY, NULL));

    void *src_ptrs[TEST_CASE_NR_DPUS];
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        src_ptrs[i] = nums[i];
    }

    dpu_send_direct(dpu_set, ptr0, src_ptrs, sizeof(int32_t) * TEST_CASE_NR_DPUS);

    int32_t *nums_copy = malloc(sizeof(int32_t) * TEST_CASE_NR_DPUS * TEST_CASE_NR_DPUS);
    void *dst_ptrs[TEST_CASE_NR_DPUS];
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        dst_ptrs[i] = nums_copy + i * TEST_CASE_NR_DPUS;
    }
    dpu_recv_direct(dpu_set, ptr0, dst_ptrs, TEST_CASE_NR_DPUS * sizeof(int32_t));
    cnt = 0;
    for (int i = 0; i < TEST_CASE_NR_DPUS * TEST_CASE_NR_DPUS; i++)
    {
        assert(nums_copy[i] == cnt);
        cnt++;
    }

    free(nums_copy);

    DPU_ASSERT(dpu_free(dpu_set));
}

void test_case_msg(void)
{
    struct dpu_set_t dpu_set;
    DPU_ASSERT(dpu_alloc(TEST_CASE_NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, TEST_CASE_DPU_BINARY, NULL));

    all_dpu_mm_reset();
    remote_ptr ptr0 = all_dpu_alloc(1024);
    assert(ptr0.dpu_id == ALL_DPU && ptr0.dpu_addr == FREE_STORAGE_OFFSET);

    // push a matrix to dpu set
    int32_t nums[TEST_CASE_NR_DPUS][TEST_CASE_NR_DPUS];
    int cnt = 0;
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        for (int j = 0; j < TEST_CASE_NR_DPUS; j++)
        {
            nums[i][j] = cnt;
            cnt++;
        }
    }

    void *src_ptrs[TEST_CASE_NR_DPUS];
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        src_ptrs[i] = nums[i];
    }

    dpu_send_direct(dpu_set, ptr0, src_ptrs, sizeof(int32_t) * TEST_CASE_NR_DPUS);

    msg_block_des msg_add, msg_get;
    int32_t num = 100;
    msg_block_builder_op_tensor_add_for_test(&msg_add, ptr0, TEST_CASE_NR_DPUS, 1, num);
    msg_block_builder_op_tensor_get_for_test(&msg_get, ptr0, TEST_CASE_NR_DPUS, 1);

    msg_buffer buffer;
    msg_buffer_init(&buffer);
    msg_buffer_clear(&buffer);
    msg_buffer_append(&buffer, &msg_add);
    msg_buffer_append(&buffer, &msg_get);
    msg_buffer_finish(&buffer);

#if TEST_CASE_DEBUG_PRINT
    msg_buffer_dump_int32(&buffer);
#endif

    msg_buffer_send(&buffer, dpu_set);

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

#if TEST_CASE_DEBUG_PRINT
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_set, dpu)
    {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
#endif

    int32_t *nums_copy = malloc(sizeof(int32_t) * TEST_CASE_NR_DPUS * TEST_CASE_NR_DPUS);
    void *dst_ptrs[TEST_CASE_NR_DPUS];
    for (int i = 0; i < TEST_CASE_NR_DPUS; i++)
    {
        dst_ptrs[i] = nums_copy + i * TEST_CASE_NR_DPUS;
    }
    msg_buffer_recv(dpu_set, dst_ptrs, sizeof(int32_t) * TEST_CASE_NR_DPUS);
    cnt = 0;
    for (int i = 0; i < TEST_CASE_NR_DPUS * TEST_CASE_NR_DPUS; i++)
    {
        assert(nums_copy[i] == cnt + num);
        cnt++;
    }

    free(nums_copy);
    msg_block_free(&msg_add);
    msg_block_free(&msg_get);
    DPU_ASSERT(dpu_free(dpu_set));
}
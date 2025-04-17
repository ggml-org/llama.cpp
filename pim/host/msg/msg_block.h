#ifndef _MSG_BLOCK_H
#define _MSG_BLOCK_H

#include "../mm/pim_mm.h"
#include <string.h>
#include <stdlib.h>

enum pim_op
{
    PIM_OP_GEMV = 0,
    PIM_OP_TENSOR_ADD_FOR_TEST,
    PIM_OP_TENSOR_GET_FOR_TEST,

    PIM_OP_COUNT,
};

typedef ALIGN8 struct
{
    int32_t type;
    int32_t ne[2];
    remote_ptr ptr;
} pim_tensor_des;

typedef ALIGN8 struct
{
    uint32_t op;
    pim_tensor_des src0; // weight
    pim_tensor_des src1; // input
} msg_block_header;

typedef struct
{
    msg_block_header header;
    void *extra;
    uint32_t extra_size;
} msg_block_des;

void msg_block_builder_op_tensor_add_for_test(msg_block_des* msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1, int32_t num);
void msg_block_builder_op_tensor_get_for_test(msg_block_des* msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1);

void msg_block_free(msg_block_des* msg);

#endif
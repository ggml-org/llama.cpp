#include "msg_block.h"

void msg_block_builder_op_tensor_add_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1, int32_t num)
{
    msg->header.op = PIM_OP_TENSOR_ADD_FOR_TEST;
    msg->header.src0.type = 0; // int32_t
    msg->header.src0.ne[0] = ne0;
    msg->header.src0.ne[1] = ne1;
    msg->header.src0.ptr = target_tensor;

    msg->extra = malloc(sizeof(int));

    memcpy(msg->extra, &num, sizeof(num));
    msg->extra_size = sizeof(num);
}

void msg_block_builder_op_tensor_get_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1)
{
    msg->header.op = PIM_OP_TENSOR_GET_FOR_TEST;
    msg->header.src0.type = 0; // int32_t
    msg->header.src0.ne[0] = ne0;
    msg->header.src0.ne[1] = ne1;
    msg->header.src0.ptr = target_tensor;

    msg->extra = NULL;
    msg->extra_size = 0;
}

void msg_block_free(msg_block_des *msg)
{
    if (msg && msg->extra)
    {
        free(msg->extra);
    }
}
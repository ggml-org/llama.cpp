#ifndef _TENSOR_ADD_H
#define _TENSOR_ADD_H

#include <mram.h>
#include "../../host/msg/msg_block.h"

void tensor_add_prepare(msg_block_header *header_ptr);
void tensor_add_tasklets_run();
void tensor_add_merge();

#endif
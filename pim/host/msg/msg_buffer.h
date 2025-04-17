#ifndef _MSG_BUFFER_H
#define _MSG_BUFFER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "msg_block.h"

#define MSG_MAX_SIZE (1024 * 1024) // 1MB
#define BLOCK_MAX_COUNT 16

typedef ALIGN8 struct
{
    uint32_t block_cnt;
    uint32_t total_size;
} msg_buffer_header;

typedef struct
{
    msg_buffer_header *header_ptr;
    char *buffer;
    uint32_t block_offsets[BLOCK_MAX_COUNT];
} msg_buffer;

void msg_buffer_init(msg_buffer *builder);
void msg_buffer_append(msg_buffer *builder, msg_block_des* msg_block);
void msg_buffer_finish(msg_buffer *builder);
void msg_buffer_clear(msg_buffer *builder);
void msg_buffer_dump_int32(msg_buffer *builder);

#endif
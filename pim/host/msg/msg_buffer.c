#include "msg_buffer.h"
#include "../util/util.h"
#include <assert.h>
#include <string.h>

void msg_buffer_clear(msg_buffer *builder)
{
    builder->header_ptr = (msg_buffer_header *)builder->buffer;
    builder->header_ptr->block_cnt = 0;
    builder->header_ptr->total_size = sizeof(msg_buffer_header);
}

void msg_buffer_init(msg_buffer *builder)
{
    builder->buffer = malloc(MSG_MAX_SIZE);
    msg_buffer_clear(builder);
}

void msg_buffer_append(msg_buffer *builder, msg_block_des *msg_block)
{
    assert(builder->header_ptr->block_cnt < BLOCK_MAX_COUNT);
    assert(builder->header_ptr->total_size + sizeof(msg_block_header) +
               align8(msg_block->extra_size) <=
           MSG_MAX_SIZE);

    builder->block_offsets[builder->header_ptr->block_cnt] = builder->header_ptr->total_size;
    builder->header_ptr->block_cnt++;
    memcpy(builder->buffer + builder->header_ptr->total_size, &msg_block->header, sizeof(msg_block_header));
    builder->header_ptr->total_size += sizeof(msg_block_header);
    memcpy(builder->buffer + builder->header_ptr->total_size, msg_block->extra, msg_block->extra_size);
    builder->header_ptr->total_size += align8(msg_block->extra_size);
}

void msg_buffer_finish(msg_buffer *builder)
{
    assert(builder->header_ptr->total_size + align8(sizeof(uint32_t) * builder->header_ptr->block_cnt) <= MSG_MAX_SIZE);
    memcpy(builder->buffer + builder->header_ptr->total_size, builder->block_offsets,
           sizeof(uint32_t) * builder->header_ptr->block_cnt);
    builder->header_ptr->total_size += align8(sizeof(uint32_t) * builder->header_ptr->block_cnt);
}

void msg_buffer_dump_int32(msg_buffer *builder)
{
    for (int i = 0; i < builder->header_ptr->total_size; i += sizeof(uint32_t))
    {
        uint32_t *buffer = (uint32_t *)(builder->buffer + i);
        printf("%d ", *buffer);
    }
    printf("\n");
}

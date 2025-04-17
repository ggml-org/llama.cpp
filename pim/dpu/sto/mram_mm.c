#include "mram_mm.h"

#define DATA_BLOCK_SIZE 2048

int mram2wram(__mram_ptr void *pmram, void *pwram, uint32_t size)
{
    uint32_t rest_size = size;
    uint32_t index = 0;
    __mram_ptr void *from;
    void *to;
    while (rest_size >= DATA_BLOCK_SIZE)
    {
        from = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        to = (void *)(((unsigned char *)pwram) + index);
        mram_read(from, to, DATA_BLOCK_SIZE);
        rest_size -= DATA_BLOCK_SIZE;
        index += DATA_BLOCK_SIZE;
    }

    if (rest_size)
    {
        from = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        to = (void *)(((unsigned char *)pwram) + index);
        mram_read(from, to, rest_size);
    }
    return 0;
}

int wram2mram(__mram_ptr void *pmram, void *pwram, uint32_t size)
{
    uint32_t rest_size = size;
    uint32_t index = 0;
    __mram_ptr void *to;
    void *from;
    while (rest_size >= DATA_BLOCK_SIZE)
    {
        to = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        from = (void *)(((unsigned char *)pwram) + index);
        mram_write(from, to, DATA_BLOCK_SIZE);
        rest_size -= DATA_BLOCK_SIZE;
        index += DATA_BLOCK_SIZE;
    }

    if (rest_size)
    {
        to = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        from = (void *)(((unsigned char *)pwram) + index);
        mram_write(from, to, rest_size);
    }
    return 0;
}
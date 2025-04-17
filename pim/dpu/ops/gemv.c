#include "gemv.h"

__host int16_t mul_table_int4_int8[1 << 4][1 << 8];
__mram_noinit float table_f32_f16[1 << 16];

static float lookup_fp16_to_fp32(uint16_t f)
{
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    uint16_t alignedOffset;
    float temp[8];

    alignedOffset = s & 0xfff8;
    mram_read((__mram_ptr void const *)(table_f32_f16 + alignedOffset), temp, sizeof(float) * 8);
    return temp[s & 0x7];
}

#define FP16_TO_FP32(x) lookup_fp16_to_fp32(x)

void gemv_prepare()
{
}

void gemv_tasklets_run()
{
}

void gemv_merge()
{
}
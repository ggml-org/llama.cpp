#include "util.h"

uint32_t align8(uint32_t s)
{
    if ((s & 0x7) == 0)
        return s;
    return ((s >> 3) + 1) << 3;
}


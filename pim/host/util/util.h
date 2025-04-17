#ifndef _UTIL_H
#define _UTIL_H

#include <stdint.h>

// Find the lowest index for the rank-th group
#define BLOCK_LOW(rank, size, n) ((rank) * (n) / (size))

// Find the highest index for the rank-th group
#define BLOCK_HIGH(rank, size, n) (BLOCK_LOW((rank) + 1, (size), (n)) - 1)

uint32_t align8(uint32_t s);

#endif
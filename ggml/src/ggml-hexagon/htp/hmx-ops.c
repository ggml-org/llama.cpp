// HMX operations compiled as a single translation unit.
// This allows interprocedural optimizations within HMX ops without requiring global HTP LTO.

#include "hmx-queue.c"
#include "hmx-matmul-ops.c"
#include "hmx-flash-attn-ops.c"

// Q8 kernel template instantiation

#include "../fattn-q8.cuh"

// Phase 5: Temporarily disabled - DKQ=112 is not multiple of 32 (Q8_0 block size)
// TODO: Re-enable after adding support for non-32-multiple head sizes
// DECL_FATTN_TILE_CASE(112, 112);

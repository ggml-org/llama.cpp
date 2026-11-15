// Verifies that the element order of the CUDA PTQ1_0 accessor (ptq1_0_trit in
// ggml-cuda/common.cuh) matches the CPU codec. The accessor is pure integer logic, so
// it is checked on the host; keep the copy below in sync with common.cuh.
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#define QK_PTQ1_0 128
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };

// --- copy of ptq1_0_trit from ggml-cuda/common.cuh ---
static int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

int main(void) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_PTQ1_0);
    if (traits->blck_size != QK_PTQ1_0 || traits->type_size != sizeof(block_ptq1_0)) {
        return 1;
    }
    unsigned seed = 12345;
    long bad = 0, total = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        block_ptq1_0 blk;
        for (int i = 0; i < 24; ++i) { seed = seed*1103515245u+12345u; blk.qs[i] = (seed>>16)&0xFF; }
        for (int i = 0; i <  2; ++i) { seed = seed*1103515245u+12345u; blk.qh[i] = (seed>>16)&0xFF; }
        blk.d = 0x3c00; // fp16 1.0: compare integer accessor to the actual CPU dequantizer.
        float ref[QK_PTQ1_0];
        traits->to_float(&blk, ref, QK_PTQ1_0);
        for (int e = 0; e < QK_PTQ1_0; ++e) {
            ++total;
            if (ptq1_0_trit(&blk, e) != ref[e]) {
                if (bad < 5) printf("  MISMATCH blk%d e=%d accessor=%d cpu=%g\n", trial, e, ptq1_0_trit(&blk,e), ref[e]);
                ++bad;
            }
        }
    }
    printf("  checked %ld element positions across 20000 random blocks\n", total);
    printf("  mismatches: %ld\n", bad);
    printf("  %s\n", bad==0 ? "Host accessor replica matches production CPU codec; device execution not tested" : "MAPPING BUG");
    return bad != 0;
}

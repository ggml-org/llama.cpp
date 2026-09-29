// Verifies that the element order of the CUDA PTQ1_0 accessor (ptq1_0_trit in
// ggml-cuda/common.cuh) matches the CPU codec. The accessor is pure integer logic, so
// it is checked on the host; keep the copy below in sync with common.cuh.
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

// --- the CPU reference traversal from ggml-quants.c dequantize_row_ptq1_0 ---
static void cpu_ref(const block_ptq1_0 * x, int * out) {
    const uint8_t pow3[6] = {1,3,9,27,81,243};
    const size_t stages[3] = {32,16,8};
    int o = 0; size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
        const size_t c = stages[s];
        for (; j + c <= sizeof(x->qs); j += c)
            for (size_t n = 0; n < 5; ++n)
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = x->qs[j+m] * pow3[n];
                    out[o++] = (int)(((uint16_t)q * 3) >> 8) - 1;
                }
    }
    for (size_t n = 0; n < 4; ++n)
        for (size_t h = 0; h < sizeof(x->qh); ++h) {
            uint8_t q = x->qh[h] * pow3[n];
            out[o++] = (int)(((uint16_t)q * 3) >> 8) - 1;
        }
}

int main(void) {
    unsigned seed = 12345;
    long bad = 0, total = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        block_ptq1_0 blk;
        for (int i = 0; i < 24; ++i) { seed = seed*1103515245u+12345u; blk.qs[i] = (seed>>16)&0xFF; }
        for (int i = 0; i <  2; ++i) { seed = seed*1103515245u+12345u; blk.qh[i] = (seed>>16)&0xFF; }
        int ref[QK_PTQ1_0]; cpu_ref(&blk, ref);
        for (int e = 0; e < QK_PTQ1_0; ++e) {
            ++total;
            if (ptq1_0_trit(&blk, e) != ref[e]) {
                if (bad < 5) printf("  MISMATCH blk%d e=%d cuda=%d cpu=%d\n", trial, e, ptq1_0_trit(&blk,e), ref[e]);
                ++bad;
            }
        }
    }
    printf("  checked %ld element positions across 20000 random blocks\n", total);
    printf("  mismatches: %ld\n", bad);
    printf("  %s\n", bad==0 ? "CUDA element mapping MATCHES the CPU codec exactly" : "MAPPING BUG");
    return bad != 0;
}

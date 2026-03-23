#ifndef TQ_UTILS_IMPL
#define TQ_UTILS_IMPL

#if defined(DATA_A_TQ2_0)
int tq2_dequantize(uint ib, uint iqs) {
    const uint upper = iqs / 128;

    const uint byte = (upper * 32) + (iqs % 32);
    const uint shift = ((iqs % 128) / 32) * 2;

    const int c = (int(data_a[ib].qs[byte]) >> shift) & 3;

    return c - 1;
}
#endif

#if defined(DATA_A_TQ1_0)
int tq1_dequantize(uint ib, uint iqs) {
    const uint pow3[6] = uint[6](1, 3, 9, 27, 81, 243);

    if (iqs < 160) {
        const uint trit = iqs / 32;
        const uint byte = iqs % 32;
        const uint q = uint(data_a[ib].qs[byte]);
        const uint val = (((q * pow3[trit]) & 255) * 3) / 256;
        return int(val) - 1;
    } else if (iqs < 240) {
        const uint relative_idx = iqs - 160;
        const uint trit = relative_idx / 16;
        const uint byte = relative_idx % 16;
        const uint q = uint(data_a[ib].qs[32 + byte]);
        const uint val = (((q * pow3[trit]) & 255) * 3) / 256;
        return int(val) - 1;
    } else {
        const uint relative_idx = iqs - 240;
        const uint trit = relative_idx / 4;
        const uint byte = relative_idx % 4;
        const uint q = uint(data_a[ib].qh[byte]);
        const uint val = (((q * pow3[trit]) & 255) * 3) / 256;
        return int(val) - 1;
    }
}
#endif

#endif
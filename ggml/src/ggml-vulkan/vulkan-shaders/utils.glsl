#ifndef UTILS_COMP
#define UTILS_COMP

// mod and div are expensive and coordinates/dimensions are often power of 2 or equal to 1
uint fastmod(uint a, uint b) {
    if ((b & (b-1)) == 0) {
        return a & (b-1);
    }
    return a % b;
}

// see init_fastdiv_values in ggml-vulkan.cpp
uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;
    // msbs = mulhi(n, mp)
    umulExtended(n, mp, msbs, lsbs);
    return (msbs + n) >> L;
}

uint fastdiv_L(uint packed, uint slot) {
    return (packed >> (slot * 8)) & 0x3Fu;
}

uint fastdiv_small(uint a, uint b) {
    return (a < b) ? 0 : (a / b);
}

// log(1 + x); the atanh series 2*y*(1 + y^2/3 + ... + y^12/13), y = x/(2+x),
// keeps the precision that log(1 + x) loses for x in [-0.5, 1].
float log1p(float x) {
    float y = x / (2.0f + x);
    float y2 = y * y;
    float p = 1.0f / 13.0f;
    p = p * y2 + 1.0f / 11.0f;
    p = p * y2 + 1.0f / 9.0f;
    p = p * y2 + 1.0f / 7.0f;
    p = p * y2 + 1.0f / 5.0f;
    p = p * y2 + 1.0f / 3.0f;
    p = p * y2 + 1.0f;
    float s = 2.0f * y * p;
    float l = log(1.0f + x);
    bool in_series = (x >= -0.5f) && (x <= 1.0f);
    bool tiny = abs(x) < 2.5e-38f;
    float r = in_series ? s : l;
    return tiny ? x : r;
}

void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03, uint ne00, uint ne01, uint ne02, uint ne03) {
    i03 = fastdiv_small(idx, (ne02*ne01*ne00));
    const uint i03_offset = i03 * ne02*ne01*ne00;
    i02 = fastdiv_small((idx - i03_offset), (ne01*ne00));
    const uint i02_offset = i02*ne01*ne00;
    i01 = (idx - i03_offset - i02_offset) / ne00;
    i00 = idx - i03_offset - i02_offset - i01*ne00;
}

#endif // UTILS_COMP

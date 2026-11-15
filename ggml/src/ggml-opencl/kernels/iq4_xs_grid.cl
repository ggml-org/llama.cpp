constant char iq4_xs_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

inline uint iq4_xs_code(uint n) {
    const uint lo = (n & 4u) ? 0xF6EADDCFu : 0xBFAD9881u;
    const uint hi = (n & 4u) ? 0x71594535u : 0x26190D01u;
    const uint t = (n & 8u) ? hi : lo;
    return (t >> (8*(n & 3u))) & 255u;
}

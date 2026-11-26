#ifndef TILEDK_LAYOUT_GLSL
#define TILEDK_LAYOUT_GLSL

// Store each TILEK-wide K strip contiguously across all rows.
// K must be divisible by TILEK.

#ifndef TILEK
#define TILEK 16
#endif

#if (TILEK == 0) || ((TILEK & (TILEK - 1)) != 0)
#error "TILEK must be a power of two"
#endif

const uint TILEK_U = uint(TILEK);

uint tiledk_first_idx(const uint row, const uint kidx, const uint nrows) {
    return TILEK_U * ((kidx / TILEK_U) * nrows + row) + (kidx % TILEK_U);
}

#endif

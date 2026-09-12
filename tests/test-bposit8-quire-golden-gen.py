# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Emit an integer (M, E) LUT for every b-posit8 code plus random dot-product
test cases with an EXACT golden quire value, computed from open-bposit's
rational reference. Used to verify the C exact-quire vec_dot in isolation
before wiring it into llama.cpp/ggml.

Each b-posit8 value is an exact dyadic rational -> value = M * 2^E (M,E ints).
The golden dot is Sum_j (Mx*My) * 2^(Ex+Ey+sx+sy), computed in Python
Fraction, then expressed as an exact 256-bit / 96-frac fixed-point quire
integer (the canonical Anomly quire) for bit-exact comparison with C.
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, "reference")
from bposit_ref import decode_bposit8, decoded_to_fraction_8, ZERO_8, NAR_8  # noqa: E402

QUIRE_FRAC_BITS = 96
QUIRE_BITS = 256  # signed two's-complement, 8 x 32-bit limbs

# ---- integer (sign,M,E) form for every code -------------------------------
# value = M * 2^E exactly (M signed, E int). zero -> M=0. NAR flagged.
lut = []  # (kind, M, E)  kind: 0 normal, 1 zero, 2 nar
for code in range(256):
    if code == ZERO_8:
        lut.append((1, 0, 0)); continue
    if code == NAR_8:
        lut.append((2, 0, 0)); continue
    fr = decoded_to_fraction_8(decode_bposit8(code))
    num, den = fr.numerator, fr.denominator
    assert den & (den - 1) == 0, f"denominator not power of two for code {code}: {den}"
    e = -(den.bit_length() - 1)  # den == 2^(bit_length-1)
    lut.append((0, num, e))       # value = num * 2^e


def frac_of(code):
    kind, M, E = lut[code]
    if kind == 2:
        raise ValueError("NaR")
    return Fraction(M) * (Fraction(1, 1 << -E) if E < 0 else Fraction(1 << E))


def quire_int(fr):
    """Exact 256-bit/96-frac fixed-point image of a dyadic Fraction.
    Requires fr to be exactly representable (no bits below 2^-96)."""
    scaled = fr * (1 << QUIRE_FRAC_BITS)
    assert scaled.denominator == 1, "golden underflows the 96-frac quire; pick milder scales"
    v = scaled.numerator
    lim = 1 << (QUIRE_BITS - 1)
    assert -lim <= v < lim, "golden overflows 256-bit quire; pick milder scales"
    return v & ((1 << QUIRE_BITS) - 1)  # two's complement image


# ---- deterministic pseudo-random test cases (no external RNG) --------------
# simple LCG so the generator is reproducible and self-contained
_state = 0x2026_07_22
def rnd():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return _state >> 33  # 31 bits

QK = 32
cases = []  # (sx, sy, xs[QK], ys[QK], golden_quire_hexlimbs)
for _ in range(64):
    # modest block scales so the exact dot fits the 96-frac quire exactly
    sx = (rnd() % 13) - 6
    sy = (rnd() % 13) - 6
    xs = [rnd() & 0xFF for _ in range(QK)]
    ys = [rnd() & 0xFF for _ in range(QK)]
    # avoid NaR codes in test data (0x80); remap to zero
    xs = [0x00 if c == NAR_8 else c for c in xs]
    ys = [0x00 if c == NAR_8 else c for c in ys]
    acc = Fraction(0)
    for a, b in zip(xs, ys):
        acc += frac_of(a) * frac_of(b)
    acc *= (Fraction(1 << sx) if sx >= 0 else Fraction(1, 1 << -sx))
    acc *= (Fraction(1 << sy) if sy >= 0 else Fraction(1, 1 << -sy))
    q = quire_int(acc)
    limbs = [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
    cases.append((sx, sy, xs, ys, limbs, acc))

# ---- multi-block streaming cases (K spans several blocks -> one quire) -----
# ggml_vec_dot handles n = nb*QK; the quire streams across blocks. Verify.
stream_cases = []  # (nb, sx_list, sy_list, xs_flat, ys_flat, limbs, golden_double)
for nb in (2, 4, 8):
    acc = Fraction(0)
    sxs, sys, xf, yf = [], [], [], []
    for _b in range(nb):
        sx = (rnd() % 9) - 4
        sy = (rnd() % 9) - 4
        sxs.append(sx); sys.append(sy)
        def rc():
            v = rnd() & 0xFF
            return 0x00 if v == NAR_8 else v
        xs = [rc() for _ in range(QK)]
        ys = [rc() for _ in range(QK)]
        xf += xs; yf += ys
        blk = Fraction(0)
        for a, b in zip(xs, ys):
            blk += frac_of(a) * frac_of(b)
        blk *= (Fraction(1 << sx) if sx >= 0 else Fraction(1, 1 << -sx))
        blk *= (Fraction(1 << sy) if sy >= 0 else Fraction(1, 1 << -sy))
        acc += blk
    q = quire_int(acc)
    limbs = [(q >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
    stream_cases.append((nb, sxs, sys, xf, yf, limbs, float(acc)))

# ---- catastrophic-cancellation case (exact quire must nail it) -------------
# build x,y so products are [+big, -big(x62), ...] that cancel to a small exact
# residue -- pick codes for +1 and construct via scales. Use code for value 1.0.
one_code = None
for code in range(256):
    kind, M, E = lut[code]
    if kind == 0 and Fraction(M) * (Fraction(1, 1 << -E) if E < 0 else Fraction(1 << E)) == 1:
        one_code = code; break
assert one_code is not None, "no code decodes to exactly 1.0"

# ---- emit C header --------------------------------------------------------
out = []
out.append("/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. */")
out.append("/* AUTO-GENERATED by gen_bp8_dot_golden.py — do not edit. */")
out.append("#include <stdint.h>")
out.append(f"#define BP8_QK {QK}")
out.append(f"#define BP8_QUIRE_FRAC_BITS {QUIRE_FRAC_BITS}")
out.append("typedef struct { int kind; int64_t M; int E; } bp8_lut_t;")
out.append("static const bp8_lut_t BP8_LUT[256] = {")
for kind, M, E in lut:
    out.append(f"  {{ {kind}, {M}LL, {E} }},")
out.append("};")
out.append(f"#define BP8_NCASES {len(cases)}")
out.append("typedef struct { int sx, sy; uint8_t xs[BP8_QK], ys[BP8_QK]; uint32_t golden[8]; } bp8_case_t;")
out.append("static const bp8_case_t BP8_CASES[BP8_NCASES] = {")
for sx, sy, xs, ys, limbs, _acc in cases:
    xsl = ",".join(str(c) for c in xs)
    ysl = ",".join(str(c) for c in ys)
    gl = ",".join(f"0x{l:08x}u" for l in limbs)
    out.append(f"  {{ {sx}, {sy}, {{{xsl}}}, {{{ysl}}}, {{{gl}}} }},")
out.append("};")

out.append(f"#define BP8_NSTREAM {len(stream_cases)}")
out.append("typedef struct { int nb; int sx[8]; int sy[8]; "
           "uint8_t xs[8*BP8_QK]; uint8_t ys[8*BP8_QK]; uint32_t golden[8]; double gd; } bp8_stream_t;")
out.append("static const bp8_stream_t BP8_STREAM[BP8_NSTREAM] = {")
for nb, sxs, sys, xf, yf, limbs, gd in stream_cases:
    sxl = ",".join(str(v) for v in (sxs + [0]*(8-nb)))
    syl = ",".join(str(v) for v in (sys + [0]*(8-nb)))
    xsl = ",".join(str(c) for c in xf)
    ysl = ",".join(str(c) for c in yf)
    gl = ",".join(f"0x{l:08x}u" for l in limbs)
    out.append(f"  {{ {nb}, {{{sxl}}}, {{{syl}}}, {{{xsl}}}, {{{ysl}}}, {{{gl}}}, {gd!r} }},")
out.append("};")
out.append(f"#define BP8_ONE_CODE {one_code}")

hdr = "\n".join(out) + "\n"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-bposit8-quire-golden.h"), "w") as f:
    f.write(hdr)

print(f"LUT: 256 codes, {len(cases)} dot cases emitted.")
print("sample case[0] acc =", float(cases[0][5]), " scales", cases[0][0], cases[0][1])
print("wrote test-bposit8-quire-golden.h")

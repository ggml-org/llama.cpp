#!/usr/bin/env python3
"""
Offline enumeration for iFairy complex 2-bit 3-weight LUT (TL1).

This script corresponds to section "10.1 步骤一：规格验证与离线枚举"
in IFAIRY_ARM_3W_LUT_DESIGN.md.

It:
  - Enumerates all 3×2-bit combinations (c0, c1, c2), ci ∈ {0,1,2,3}.
  - Uses the weight semantics from the design doc:
        0 -> -1, 1 -> +1, 2 -> -i, 3 -> +i.
  - For each triple, decomposes (w0, w1, w2) as
        (w0, w1, w2) = factor * (1, u1, u2),
    where factor ∈ {1, i, -1, -i} and (1, u1, u2) is one of 16 canonical
    patterns.
  - Builds:
        canonical_idx[64] : raw 6-bit index -> 4-bit canonical index
        factor_exp[64]    : raw 6-bit index -> exponent e, i^e = factor
                           (e ∈ {0, 1, 2, 3} → {1, i, -1, -i})
  - Verifies that for a fixed test activation (x0, x1, x2),
    direct scalar accumulation equals factor * canonical accumulation.
  - Emits baseline LUT values for the 16 canonical patterns for
    that activation triple.

Usage
=====
Run from the project root:

    python scripts/ifairy_3w_lut_enum.py

The script prints C-style arrays and a small human-readable summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# Weight code -> exponent e such that weight(c) = i^e
# Design doc: 0 -> -1, 1 -> +1, 2 -> -i, 3 -> +i
#
# i^0 =  1
# i^1 =  i
# i^2 = -1
# i^3 = -i
#
# So:
#   0 (-1)  -> e = 2
#   1 (+1)  -> e = 0
#   2 (-i)  -> e = 3
#   3 (+i)  -> e = 1
CODE_TO_EXP: Dict[int, int] = {0: 2, 1: 0, 2: 3, 3: 1}

# Exponent -> complex unit i^e
EXP_TO_COMPLEX: Dict[int, complex] = {
    0: 1.0 + 0.0j,
    1: 0.0 + 1.0j,
    2: -1.0 + 0.0j,
    3: 0.0 - 1.0j,
}

# Exponent -> symbolic name (for printing)
EXP_TO_NAME: Dict[int, str] = {
    0: " 1",
    1: " i",
    2: "-1",
    3: "-i",
}


@dataclass(frozen=True)
class CanonicalPattern:
    """Canonical (1, u1, u2) pattern, encoded by exponents of i."""

    u1_exp: int  # exponent e, i^e = u1
    u2_exp: int  # exponent e, i^e = u2

    @property
    def idx(self) -> int:
        """
        Canonical index in [0, 15]:

            idx = (u1_exp << 2) | u2_exp

        This is a pure encoding choice for 16 patterns.
        """

        return ((self.u1_exp & 0x3) << 2) | (self.u2_exp & 0x3)


def raw_index(c0: int, c1: int, c2: int) -> int:
    """
    Raw 6-bit index as defined in the design doc:

        idx_raw = (c0 << 4) | (c1 << 2) | c2
    """

    return ((c0 & 0x3) << 4) | ((c1 & 0x3) << 2) | (c2 & 0x3)


def decompose_triple(c0: int, c1: int, c2: int) -> Tuple[CanonicalPattern, int]:
    """
    Decompose a triple (c0, c1, c2) into:

        (w0, w1, w2) = factor * (1, u1, u2)

    with:
        - factor = w0  (so exponent f_exp = e0)
        - u1 = w1 / w0, u2 = w2 / w0

    In exponent form (mod 4):
        e0, e1, e2 : CODE_TO_EXP[ci]
        factor_exp = e0
        u1_exp     = (e1 - e0) mod 4
        u2_exp     = (e2 - e0) mod 4
    """

    e0 = CODE_TO_EXP[c0]
    e1 = CODE_TO_EXP[c1]
    e2 = CODE_TO_EXP[c2]

    factor_exp = e0
    u1_exp = (e1 - e0) & 0x3
    u2_exp = (e2 - e0) & 0x3

    pattern = CanonicalPattern(u1_exp=u1_exp, u2_exp=u2_exp)
    return pattern, factor_exp


def compute_baseline_lut_values(
    canonical_patterns: List[CanonicalPattern],
    x: Tuple[complex, complex, complex],
) -> Dict[int, complex]:
    """
    For each canonical pattern, compute baseline LUT value:

        S_base(idx') = 1 * x0 + u1 * x1 + u2 * x2

    using the provided test activation triple x = (x0, x1, x2).
    Returns a mapping idx' -> complex S_base.
    """

    x0, x1, x2 = x
    baseline: Dict[int, complex] = {}

    for pattern in canonical_patterns:
        u1 = EXP_TO_COMPLEX[pattern.u1_exp]
        u2 = EXP_TO_COMPLEX[pattern.u2_exp]
        s_base = (1.0 + 0.0j) * x0 + u1 * x1 + u2 * x2
        baseline[pattern.idx] = s_base

    return baseline


def main() -> None:
    # Fixed test activation triple (integer components, but stored as complex).
    #
    # These values are only for spec checking / offline enumeration.
    # They do not appear in runtime kernels.
    x0 = 1.0 + 2.0j
    x1 = 3.0 - 4.0j
    x2 = -5.0 + 6.0j
    x = (x0, x1, x2)

    canonical_idx: List[int] = [0] * 64
    factor_exp: List[int] = [0] * 64

    # Collect all canonical patterns encountered, indexed by (u1_exp, u2_exp).
    patterns_map: Dict[Tuple[int, int], CanonicalPattern] = {}

    # For consistency checks: for each pattern, all S_base across different
    # (c0, c1, c2) that map to it should match.
    s_base_check: Dict[int, complex] = {}

    for c0 in range(4):
        for c1 in range(4):
            for c2 in range(4):
                idx_raw = raw_index(c0, c1, c2)

                pattern, f_exp = decompose_triple(c0, c1, c2)
                canonical_idx[idx_raw] = pattern.idx
                factor_exp[idx_raw] = f_exp

                patterns_map.setdefault(
                    (pattern.u1_exp, pattern.u2_exp),
                    pattern,
                )

                # Verify scalar equality:
                #   direct: w0*x0 + w1*x1 + w2*x2
                #   via canonical:
                #       S_base = 1*x0 + u1*x1 + u2*x2
                #       S      = factor * S_base
                e0 = CODE_TO_EXP[c0]
                e1 = CODE_TO_EXP[c1]
                e2 = CODE_TO_EXP[c2]
                w0 = EXP_TO_COMPLEX[e0]
                w1 = EXP_TO_COMPLEX[e1]
                w2 = EXP_TO_COMPLEX[e2]

                s_direct = w0 * x0 + w1 * x1 + w2 * x2

                u1 = EXP_TO_COMPLEX[pattern.u1_exp]
                u2 = EXP_TO_COMPLEX[pattern.u2_exp]
                s_base = (1.0 + 0.0j) * x0 + u1 * x1 + u2 * x2
                factor = EXP_TO_COMPLEX[f_exp]
                s_via_canonical = factor * s_base

                if abs(s_direct - s_via_canonical) > 1e-6:
                    raise RuntimeError(
                        f"Mismatch for (c0,c1,c2)=({c0},{c1},{c2}) "
                        f"raw idx={idx_raw}: "
                        f"direct={s_direct}, via_canonical={s_via_canonical}"
                    )

                # Check that all S_base for the same canonical idx agree.
                idx_c = pattern.idx
                if idx_c in s_base_check:
                    if abs(s_base_check[idx_c] - s_base) > 1e-6:
                        raise RuntimeError(
                            f"Inconsistent S_base for idx'={idx_c}: "
                            f"existing={s_base_check[idx_c]}, new={s_base}"
                        )
                else:
                    s_base_check[idx_c] = s_base

    # Sanity checks on patterns.
    if len(patterns_map) != 16:
        raise RuntimeError(
            f"Expected 16 canonical patterns, got {len(patterns_map)}"
        )

    canonical_patterns = sorted(
        patterns_map.values(),
        key=lambda p: p.idx,
    )

    baseline = compute_baseline_lut_values(canonical_patterns, x)

    print("// Auto-generated by scripts/ifairy_3w_lut_enum.py")
    print("// canonical_idx[64] : raw 6-bit index -> 4-bit canonical index (idx' ∈ [0,15])")
    print("// factor_exp[64]    : raw 6-bit index -> exponent e, where i^e = factor")
    print("//                    (e ∈ {0,1,2,3} corresponds to {1, i, -1, -i})")
    print()

    # Print canonical_idx as a C-style uint8_t array.
    print("static const uint8_t ifairy_canonical_idx[64] = {")
    for i in range(0, 64, 8):
        line = ", ".join(f"{canonical_idx[j]:2d}" for j in range(i, i + 8))
        print(f"    {line},  // [{i:2d}..{i+7:2d}]")
    print("};")
    print()

    # Print factor_exp as a C-style uint8_t array, plus comments with symbolic names.
    print("static const uint8_t ifairy_factor_exp[64] = {")
    for i in range(0, 64, 8):
        exps = factor_exp[i : i + 8]
        line = ", ".join(f"{e:2d}" for e in exps)
        names = ", ".join(EXP_TO_NAME[e] for e in exps)
        print(f"    {line},  // [{i:2d}..{i+7:2d}] -> ({names})")
    print("};")
    print()

    # Print baseline LUT values for the 16 canonical patterns.
    print("// Baseline LUT values S_base[idx'] for canonical patterns (for test x0,x1,x2)")
    print(
        f"// Test activation: x0 = {x0.real:+.1f}{x0.imag:+.1f}i, "
        f"x1 = {x1.real:+.1f}{x1.imag:+.1f}i, "
        f"x2 = {x2.real:+.1f}{x2.imag:+.1f}i"
    )
    print("struct ifairy_lut_entry { int idx; const char *u1; const char *u2; float r; float i; };")
    print("static const struct ifairy_lut_entry ifairy_lut_baseline[16] = {")
    for pattern in canonical_patterns:
        idx_c = pattern.idx
        s_base = baseline[idx_c]
        u1_name = EXP_TO_NAME[pattern.u1_exp]
        u2_name = EXP_TO_NAME[pattern.u2_exp]
        print(
            "    {"
            f" .idx = {idx_c:2d}, "
            f'.u1 = "{u1_name.strip()}", '
            f'.u2 = "{u2_name.strip()}", '
            f".r = {s_base.real:.1f}f, "
            f".i = {s_base.imag:.1f}f"
            " },"
        )
    print("};")


if __name__ == "__main__":
    main()


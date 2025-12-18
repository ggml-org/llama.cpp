from __future__ import annotations

import array
import unicodedata
import requests


MAX_CODEPOINTS = 0x110000

UNICODE_DATA_URL = "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt"


# see https://www.unicode.org/L2/L1999/UnicodeData.html
def unicode_data_iter():
    res = requests.get(UNICODE_DATA_URL)
    res.raise_for_status()
    data = res.content.decode()

    prev = []

    for line in data.splitlines():
        # ej: 0000;<control>;Cc;0;BN;;;;;N;NULL;;;;
        line = line.split(";")

        cpt = int(line[0], base=16)
        assert cpt < MAX_CODEPOINTS

        cpt_lower = int(line[-2] or "0", base=16)
        assert cpt_lower < MAX_CODEPOINTS

        cpt_upper = int(line[-3] or "0", base=16)
        assert cpt_upper < MAX_CODEPOINTS

        categ = line[2].strip()
        assert len(categ) == 2

        bidir = line[4].strip()
        assert len(categ) == 2

        name = line[1]
        if name.endswith(", First>"):
            prev = (cpt, cpt_lower, cpt_upper, categ, bidir)
            continue
        if name.endswith(", Last>"):
            assert prev[1:] == (0, 0, categ, bidir)
            for c in range(prev[0], cpt):
                yield (c, cpt_lower, cpt_upper, categ, bidir)

        yield (cpt, cpt_lower, cpt_upper, categ, bidir)


# see definition in unicode.h
CODEPOINT_FLAG_UNDEFINED   = 0x0001  #
CODEPOINT_FLAG_NUMBER      = 0x0002  # \p{N}
CODEPOINT_FLAG_LETTER      = 0x0004  # \p{L}
CODEPOINT_FLAG_SEPARATOR   = 0x0008  # \p{Z}
CODEPOINT_FLAG_MARK        = 0x0010  # \p{M}
CODEPOINT_FLAG_PUNCTUATION = 0x0020  # \p{P}
CODEPOINT_FLAG_SYMBOL      = 0x0040  # \p{S}
CODEPOINT_FLAG_CONTROL     = 0x0080  # \p{C}

UNICODE_CATEGORY_TO_FLAG = {
    "Cn": CODEPOINT_FLAG_UNDEFINED,    # Undefined
    "Cc": CODEPOINT_FLAG_CONTROL,      # Control
    "Cf": CODEPOINT_FLAG_CONTROL,      # Format
    "Co": CODEPOINT_FLAG_CONTROL,      # Private Use
    "Cs": CODEPOINT_FLAG_CONTROL,      # Surrrogate
    "Ll": CODEPOINT_FLAG_LETTER,       # Lowercase Letter
    "Lm": CODEPOINT_FLAG_LETTER,       # Modifier Letter
    "Lo": CODEPOINT_FLAG_LETTER,       # Other Letter
    "Lt": CODEPOINT_FLAG_LETTER,       # Titlecase Letter
    "Lu": CODEPOINT_FLAG_LETTER,       # Uppercase Letter
    "L&": CODEPOINT_FLAG_LETTER,       # Cased Letter
    "Mc": CODEPOINT_FLAG_MARK,         # Spacing Mark
    "Me": CODEPOINT_FLAG_MARK,         # Enclosing Mark
    "Mn": CODEPOINT_FLAG_MARK,         # Nonspacing Mark
    "Nd": CODEPOINT_FLAG_NUMBER,       # Decimal Number
    "Nl": CODEPOINT_FLAG_NUMBER,       # Letter Number
    "No": CODEPOINT_FLAG_NUMBER,       # Other Number
    "Pc": CODEPOINT_FLAG_PUNCTUATION,  # Connector Punctuation
    "Pd": CODEPOINT_FLAG_PUNCTUATION,  # Dash Punctuation
    "Pe": CODEPOINT_FLAG_PUNCTUATION,  # Close Punctuation
    "Pf": CODEPOINT_FLAG_PUNCTUATION,  # Final Punctuation
    "Pi": CODEPOINT_FLAG_PUNCTUATION,  # Initial Punctuation
    "Po": CODEPOINT_FLAG_PUNCTUATION,  # Other Punctuation
    "Ps": CODEPOINT_FLAG_PUNCTUATION,  # Open Punctuation
    "Sc": CODEPOINT_FLAG_SYMBOL,       # Currency Symbol
    "Sk": CODEPOINT_FLAG_SYMBOL,       # Modifier Symbol
    "Sm": CODEPOINT_FLAG_SYMBOL,       # Math Symbol
    "So": CODEPOINT_FLAG_SYMBOL,       # Other Symbol
    "Zl": CODEPOINT_FLAG_SEPARATOR,    # Line Separator
    "Zp": CODEPOINT_FLAG_SEPARATOR,    # Paragraph Separator
    "Zs": CODEPOINT_FLAG_SEPARATOR,    # Space Separator
}


codepoint_flags = array.array('H', [CODEPOINT_FLAG_UNDEFINED]) * MAX_CODEPOINTS
table_whitespace = []
table_lowercase = []
table_uppercase = []
table_nfd = []

for (cpt, cpt_lower, cpt_upper, categ, bidir) in unicode_data_iter():
    # convert codepoint to unicode character
    char = chr(cpt)

    # codepoint category flags
    codepoint_flags[cpt] = UNICODE_CATEGORY_TO_FLAG[categ]

    # lowercase conversion
    if cpt_lower:
        table_lowercase.append((cpt, cpt_lower))

    # uppercase conversion
    if cpt_upper:
        table_uppercase.append((cpt, cpt_upper))

    # NFD normalization
    norm = ord(unicodedata.normalize('NFD', char)[0])
    if cpt != norm:
        table_nfd.append((cpt, norm))


# whitespaces, see "<White_Space>" https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
table_whitespace.extend(range(0x0009, 0x000D + 1))
table_whitespace.extend(range(0x2000, 0x200A + 1))
table_whitespace.extend([0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


# sort by codepoint
table_whitespace.sort()
table_lowercase.sort()
table_uppercase.sort()
table_nfd.sort()


# group ranges with same flags
ranges_flags: list[tuple[int, int]] = [(0, codepoint_flags[0])]  # start, flags
for codepoint, flags in enumerate(codepoint_flags):
    if flags != ranges_flags[-1][1]:
        ranges_flags.append((codepoint, flags))
ranges_flags.append((MAX_CODEPOINTS, 0x0000))


# group ranges with same nfd
ranges_nfd: list[tuple[int, int, int]] = [(0, 0, 0)]  # start, last, nfd
for codepoint, norm in table_nfd:
    start = ranges_nfd[-1][0]
    if ranges_nfd[-1] != (start, codepoint - 1, norm):
        ranges_nfd.append(None)  # type: ignore[arg-type]  # dummy, will be replaced below
        start = codepoint
    ranges_nfd[-1] = (start, codepoint, norm)


# Generate 'unicode-data.cpp':
#   python ./scripts//gen-unicode-data.py > unicode-data.cpp

def out(line=""):
    print(line, end='\n')  # noqa


out("""\
// generated with scripts/gen-unicode-data.py

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
""")

out("const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1")
for codepoint, flags in ranges_flags:
    out("{0x%06X, 0x%04X}," % (codepoint, flags))
out("};\n")

out("const std::unordered_set<uint32_t> unicode_set_whitespace = {")
for codepoint in table_whitespace:
    out("0x%06X," % codepoint)
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {")
for tuple_lw in table_lowercase:
    out("{0x%06X, 0x%06X}," % tuple_lw)
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {")
for tuple_up in table_uppercase:
    out("{0x%06X, 0x%06X}," % tuple_up)
out("};\n")

out("const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd")
for triple in ranges_nfd:
    out("{0x%06X, 0x%06X, 0x%06X}," % triple)
out("};\n")

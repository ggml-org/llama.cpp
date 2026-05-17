# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
# Usage: cmake -DINPUT=build/tools/ui/dist/index.html -DOUTPUT=build/tools/ui/dist/index.html.hpp -P scripts/xxd.cmake

SET(INPUT "" CACHE STRING "Input File")
SET(OUTPUT "" CACHE STRING "Output File")

get_filename_component(filename "${INPUT}" NAME)
string(REGEX REPLACE "\\.|-" "_" name "${filename}")

file(READ "${INPUT}" hex_data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")

# Quote the variable so an empty file produces a 0-length result instead of
# cmake's cryptic "string sub-command LENGTH requires two arguments" error.
# This is what nix-sandbox builds hit when both npm and HF download fail
# and the UI source ends up empty (upstream PR #23190 also addresses the
# upstream-side flag-honoring bug; this defensive quote is belt-and-suspenders).
string(LENGTH "${hex_data}" hex_len)
math(EXPR len "${hex_len} / 2")

file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")

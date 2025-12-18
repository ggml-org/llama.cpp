# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
# Usage: cmake -DINPUT=tools/server/public/index.html -DOUTPUT=tools/server/index.html.hpp -P scripts/xxd.cmake

SET(INPUT "" CACHE STRING "Input File")
SET(OUTPUT "" CACHE STRING "Output File")

get_filename_component(filename "${INPUT}" NAME)
string(REGEX REPLACE "\\.|-" "_" name "${filename}")

file(READ "${INPUT}" hex_data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")

string(LENGTH ${hex_data} hex_len)
math(EXPR len "${hex_len} / 2")

file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")

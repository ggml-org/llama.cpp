# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
# Usage: cmake -DINPUT=build/tools/ui/dist/index.html -DOUTPUT=build/tools/ui/dist/index.html.hpp -P scripts/xxd.cmake
# When INPUT does not exist, generates an empty placeholder to allow build continuation.

SET(INPUT "" CACHE STRING "Input File")
SET(OUTPUT "" CACHE STRING "Output File")

get_filename_component(filename "${INPUT}" NAME)
string(REGEX REPLACE "\\.|-" "_" name "${filename}")

# Handle missing input file: generate empty placeholder to allow build to continue
# (e.g., when UI provisioning fails due to offline environment)
if(NOT EXISTS "${INPUT}")
    message(WARNING "xxd: input file \"${INPUT}\" not found, generating empty placeholder in \"${OUTPUT}\"")
    file(WRITE "${OUTPUT}" "// placeholder: UI asset not available (\"${INPUT}\" missing)\nunsigned char ${name}[] = {};\nunsigned int ${name}_len = 0;\n")
    return()
endif()

file(READ "${INPUT}" hex_data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")

string(LENGTH ${hex_data} hex_len)
math(EXPR len "${hex_len} / 2")

file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")

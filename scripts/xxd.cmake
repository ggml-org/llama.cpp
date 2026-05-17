# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
# Usage: cmake -DINPUT=build/tools/ui/dist/index.html -DOUTPUT=build/tools/ui/dist/index.html.hpp -P scripts/xxd.cmake

SET(INPUT "" CACHE STRING "Input File")
SET(OUTPUT "" CACHE STRING "Output File")

get_filename_component(filename "${INPUT}" NAME)
string(REGEX REPLACE "\\.|-" "_" name "${filename}")

# Robust handling for the offline / nix-sandbox build path where the UI
# provisioning flow (npm build → HF Bucket fallback) can leave assets either
# missing entirely or present-but-empty. Without these guards xxd.cmake hits
# either "file failed to open for reading" (missing) or "string sub-command
# LENGTH requires two arguments" (empty). Emit a valid 0-byte symbol in both
# cases so the build completes; the server side already handles a server
# build with LLAMA_UI_DEFAULT_ENABLED=0 cleanly.
if(NOT EXISTS "${INPUT}")
    file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {0};\nunsigned int ${name}_len = 0;\n")
    return()
endif()

file(READ "${INPUT}" hex_data HEX)
string(LENGTH "${hex_data}" hex_len)
math(EXPR len "${hex_len} / 2")

if(len EQUAL 0)
    file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {0};\nunsigned int ${name}_len = 0;\n")
else()
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")
    file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")
endif()

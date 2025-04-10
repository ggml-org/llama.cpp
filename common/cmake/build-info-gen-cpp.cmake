include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp")

# Only write the build info if it changed
if(EXISTS ${OUTPUT_FILE})
    file(READ ${OUTPUT_FILE} CONTENTS)
    string(REGEX MATCH "LLAMA_COMMIT = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMMIT ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_COMPILER = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_COMPILER ${CMAKE_MATCH_1})
    string(REGEX MATCH "LLAMA_BUILD_TARGET = \"([^\"]*)\";" _ ${CONTENTS})
    set(OLD_TARGET ${CMAKE_MATCH_1})
    if (
        NOT OLD_COMMIT   STREQUAL BUILD_COMMIT   OR
        NOT OLD_COMPILER STREQUAL BUILD_COMPILER OR
        NOT OLD_TARGET   STREQUAL BUILD_TARGET
    )
        configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
    endif()
else()
    configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})
endif()

define_property(GLOBAL PROPERTY LICENSE_TEXT
    BRIEF_DOCS "Embedded licenses"
    FULL_DOCS  "Global string containing all aggregated licenses"
)

function(license_add_file NAME FILE)
    if(NOT IS_ABSOLUTE "${FILE}")
        set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
    endif()
    if(EXISTS "${FILE}")
        set(TITLE "License for ${NAME}")
        string(REGEX REPLACE "." "=" UNDERLINE "${TITLE}")
        file(READ "${FILE}" TEXT)
        get_property(TMP GLOBAL PROPERTY LICENSE_TEXT)
        string(APPEND TMP "${TITLE}\n${UNDERLINE}\n\n${TEXT}\n")
        set_property(GLOBAL PROPERTY LICENSE_TEXT "${TMP}")
    else()
        message(WARNING "License file '${FILE}' not found")
    endif()
endfunction()

function(license_generate TARGET_NAME)
    message(STATUS "Generating embedded license file for target: ${TARGET_NAME}")
    get_property(TEXT GLOBAL PROPERTY LICENSE_TEXT)

    # Convert to hex because MSVC cannot handle long literal strings...
    string(HEX "${TEXT}" TEXT_HEX)
    string(REGEX REPLACE "(..)" "0x\\1," TEXT_BYTES "${TEXT_HEX}")

    set(CPP_FILE "${CMAKE_BINARY_DIR}/license.cpp")
    file(WRITE "${CPP_FILE}" "unsigned char LICENSES[] = { ${TEXT_BYTES} 0x00 };\n")

    if(TARGET ${TARGET_NAME})
        target_sources(${TARGET_NAME} PRIVATE "${CPP_FILE}")
    else()
        message(FATAL_ERROR "Target '${TARGET_NAME}' does not exist")
    endif()
endfunction()

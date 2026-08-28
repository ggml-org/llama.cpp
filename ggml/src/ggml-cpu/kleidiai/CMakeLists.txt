set(BUILD_SHARED_LIBS OFF)
set(CMAKE_SKIP_INSTALL_RULES TRUE)

add_subdirectory("${KLEIDIAI_SRC}" "${KLEIDIAI_BIN}" EXCLUDE_FROM_ALL)

if (NOT TARGET kleidiai)
    message(FATAL_ERROR "KleidiAI target was not created")
endif()

if (MSVC)
    target_compile_options(kleidiai PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/WX->)
else()
    target_compile_options(kleidiai PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-error>)
endif()

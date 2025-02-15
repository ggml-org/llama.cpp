set(TARGET llama-tts)
add_library(anyascii
    anyascii.cpp
    anyascii.h
)
add_executable(${TARGET} tts.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE anyascii)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

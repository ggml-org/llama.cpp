set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_library(mimi-model STATIC mimi-model.h mimi-model.cpp)
target_link_libraries(mimi-model PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
# for using C++ designated initializers, TODO: can be changed back to C++17 in the future
target_compile_features(mimi-model PRIVATE cxx_std_20)

set(TARGET llama-mimi)
add_executable(${TARGET} mimi.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common mimi-model ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-tts-csm)
add_executable(${TARGET} tts-csm.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common mimi-model ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

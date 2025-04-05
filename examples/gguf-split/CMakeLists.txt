cmake_minimum_required(VERSION 3.10)
project(llama-gguf)

# On Linux force non-PIE linking (this flag is applied per target below as well)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE REQUIRED fuse3)
include_directories(${FUSE_INCLUDE_DIRS})
link_directories(${FUSE_LIBRARY_DIRS})

# Original target for gguf-split
set(TARGET llama-gguf-split)
add_executable(${TARGET} gguf-split.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_options(${TARGET} PRIVATE "-no-pie")
target_link_libraries(${TARGET} PRIVATE 
    common 
    llama 
    ${CMAKE_THREAD_LIBS_INIT} 
    dl 
    pthread 
    gcc_s
)
target_compile_features(${TARGET} PRIVATE cxx_std_20)

# FUSE-based merge target
set(MERGE_TARGET llama-gguf-merge)
add_executable(${MERGE_TARGET} gguf-fuse.cpp)
install(TARGETS ${MERGE_TARGET} RUNTIME)
target_include_directories(${MERGE_TARGET} PRIVATE ${FUSE_INCLUDE_DIRS})
target_link_options(${MERGE_TARGET} PRIVATE "-no-pie")
target_link_libraries(${MERGE_TARGET} PRIVATE 
    common 
    llama 
    ${CMAKE_THREAD_LIBS_INIT} 
    dl 
    pthread 
    gcc_s 
    ${FUSE_LIBRARIES}
)
target_compile_features(${MERGE_TARGET} PRIVATE cxx_std_20)

# FUSE filesystem target
set(FUSE_TARGET llama-gguf-fuse)
add_executable(${FUSE_TARGET} gguf-fuse.cpp)
install(TARGETS ${FUSE_TARGET} RUNTIME)
target_include_directories(${FUSE_TARGET} PRIVATE ${FUSE_INCLUDE_DIRS})
target_link_options(${FUSE_TARGET} PRIVATE "-no-pie")
target_link_libraries(${FUSE_TARGET} PRIVATE 
    common 
    llama 
    ${CMAKE_THREAD_LIBS_INIT} 
    dl 
    pthread 
    gcc_s 
    ${FUSE_LIBRARIES}
)
target_compile_features(${FUSE_TARGET} PRIVATE cxx_std_20)

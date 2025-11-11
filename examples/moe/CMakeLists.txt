add_executable(moe-loader main.cpp)
target_link_libraries(moe-loader PRIVATE llama Threads::Threads)
target_include_directories(moe-loader PRIVATE ${CMAKE_SOURCE_DIR}/src)

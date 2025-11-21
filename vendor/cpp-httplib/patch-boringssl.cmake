# Remove bssl
file(READ "CMakeLists.txt" content)
string(REPLACE "add_executable(bssl" "#add_executable(bssl" content "${content}")
string(REPLACE "target_link_libraries(bssl" "#target_link_libraries(bssl" content "${content}")
string(REPLACE "install(TARGETS bssl" "#install(TARGETS bssl" content "${content}")
file(WRITE "CMakeLists.txt" "${content}")

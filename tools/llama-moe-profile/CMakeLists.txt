set(TARGET llama-moe-profile)

add_executable(${TARGET} llama-moe-profile.cpp)

# standalone: C++ stdlib only, no llama libs (reads raw trace bytes)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

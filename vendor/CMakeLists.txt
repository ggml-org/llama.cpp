# mtmd needs these even when common is not built
add_subdirectory(hash)
add_subdirectory(miniaudio)
add_subdirectory(nlohmann)
add_subdirectory(sheredom)
add_subdirectory(stb)

# used by the server and download helpers
if (LLAMA_BUILD_SERVER OR LLAMA_DOWNLOAD)
    add_subdirectory(cpp-httplib)
endif()

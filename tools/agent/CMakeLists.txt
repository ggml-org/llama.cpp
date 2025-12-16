set(TARGET llama-agent)

set(AGENT_SOURCES
    agent.cpp
    agent-loop.cpp
    tool-registry.cpp
    permission.cpp
    tools/tool-bash.cpp
    tools/tool-read.cpp
    tools/tool-write.cpp
    tools/tool-edit.cpp
    tools/tool-glob.cpp
    mcp/mcp-client.cpp
    mcp/mcp-server-manager.cpp
    mcp/mcp-tool-wrapper.cpp
)

add_executable(${TARGET} ${AGENT_SOURCES})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

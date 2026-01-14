set(TARGET llama-agent)

set(AGENT_SOURCES
    agent.cpp
    agent-loop.cpp
    tool-registry.cpp
    permission.cpp
    permission-async.cpp
    skills/skills-manager.cpp
    agents-md/agents-md-manager.cpp
    subagent/subagent-types.cpp
    subagent/subagent-display.cpp
    subagent/subagent-runner.cpp
    subagent/subagent-output.cpp
    tools/tool-bash.cpp
    tools/tool-read.cpp
    tools/tool-write.cpp
    tools/tool-edit.cpp
    tools/tool-glob.cpp
    tools/tool-task.cpp
)

# MCP support only available on Unix (uses fork, pipe, poll)
if(NOT WIN32)
    list(APPEND AGENT_SOURCES
        mcp/mcp-client.cpp
        mcp/mcp-server-manager.cpp
        mcp/mcp-tool-wrapper.cpp
    )
endif()

add_executable(${TARGET} ${AGENT_SOURCES})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# llama-agent-server (HTTP API for agent)
if(LLAMA_HTTPLIB)
    set(AGENT_SERVER_TARGET llama-agent-server)

    set(AGENT_SERVER_SOURCES
        server/agent-server.cpp
        server/agent-session.cpp
        server/agent-routes.cpp
        agent-loop.cpp
        tool-registry.cpp
        permission.cpp
        permission-async.cpp
        skills/skills-manager.cpp
        agents-md/agents-md-manager.cpp
        subagent/subagent-types.cpp
        subagent/subagent-display.cpp
        subagent/subagent-runner.cpp
        subagent/subagent-output.cpp
        tools/tool-bash.cpp
        tools/tool-read.cpp
        tools/tool-write.cpp
        tools/tool-edit.cpp
        tools/tool-glob.cpp
        tools/tool-task.cpp
        ../server/server-http.cpp
    )

    # MCP support only available on Unix (uses fork, pipe, poll)
    if(NOT WIN32)
        list(APPEND AGENT_SERVER_SOURCES
            mcp/mcp-client.cpp
            mcp/mcp-server-manager.cpp
            mcp/mcp-tool-wrapper.cpp
        )
    endif()

    add_executable(${AGENT_SERVER_TARGET} ${AGENT_SERVER_SOURCES})
    target_link_libraries(${AGENT_SERVER_TARGET} PRIVATE server-context PUBLIC common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${AGENT_SERVER_TARGET} PRIVATE cxx_std_17)
    target_include_directories(${AGENT_SERVER_TARGET} PRIVATE ../server)
    target_include_directories(${AGENT_SERVER_TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
    # Include directory for generated server assets (index.html.gz.hpp, loading.html.hpp)
    target_include_directories(${AGENT_SERVER_TARGET} PRIVATE ${CMAKE_BINARY_DIR}/tools/server)
    # Depend on llama-server to ensure generated assets exist
    add_dependencies(${AGENT_SERVER_TARGET} llama-server)

    if(WIN32)
        target_link_libraries(${AGENT_SERVER_TARGET} PRIVATE ws2_32)
    endif()

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${AGENT_SERVER_TARGET} RUNTIME)
    endif()
endif()

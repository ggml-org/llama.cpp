# Specialized chat template parsers, listed explicitly so that adding or removing one re-runs CMake instead of leaving an incremental build stale.

set(LLAMA_CHAT_PARSERS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/parsers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parsers.h
    ${CMAKE_CURRENT_LIST_DIR}/cohere2moe.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deepseek.cpp
    ${CMAKE_CURRENT_LIST_DIR}/functionary-v3-2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemma4.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gigachat-v3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpt-oss.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kimi-k2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kimi-k3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lfm2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/minicpm5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/minimax-m3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ministral3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/muse-glimmer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qwen3-coder.cpp
)

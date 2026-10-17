# llama-engine: inference engine (context, task queue, common utils), used by llama-server and CLI

set(TARGET llama-engine)

add_library(${TARGET}
    engine-chat.cpp
    engine-chat.h
    engine-task.cpp
    engine-task.h
    engine-queue.cpp
    engine-queue.h
    engine-common.cpp
    engine-common.h
    engine-decision.cpp
    engine-decision.h
    engine-context.cpp
    engine-context.h
    engine-schema.cpp
    engine-schema.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

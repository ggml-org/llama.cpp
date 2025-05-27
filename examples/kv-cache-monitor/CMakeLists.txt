set(KV_TARGET llama-kv-cache-monitor)
add_executable(${KV_TARGET} kv-cache-monitor.cpp)
install(TARGETS ${KV_TARGET} RUNTIME)
target_link_libraries(${KV_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${KV_TARGET} PRIVATE cxx_std_17)

# KQV Trace Monitor
set(KQV_TRACE_TARGET llama-kqv-trace-monitor)
add_executable(${KQV_TRACE_TARGET} kqv-trace-monitor.cpp)
install(TARGETS ${KQV_TRACE_TARGET} RUNTIME)
target_link_libraries(${KQV_TRACE_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${KQV_TRACE_TARGET} PRIVATE cxx_std_17)


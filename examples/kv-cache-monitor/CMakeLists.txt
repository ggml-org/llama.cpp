# KQV Trace Monitor
set(TARGET_KV_CACHE_MONITOR kqv-trace-monitor)
set(TARGET_KQV_READER kqv-tensor-reader)

add_executable(${TARGET_KV_CACHE_MONITOR} kqv-trace-monitor.cpp)
install(TARGETS ${TARGET_KV_CACHE_MONITOR} RUNTIME DESTINATION bin)
target_link_libraries(${TARGET_KV_CACHE_MONITOR} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_KV_CACHE_MONITOR} PRIVATE cxx_std_11)

# KV Quant Monitor
add_executable(llama-kv-quant-monitor kv-quant-monitor.cpp)
install(TARGETS llama-kv-quant-monitor RUNTIME)
target_link_libraries(llama-kv-quant-monitor PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-kv-quant-monitor PRIVATE cxx_std_17)

# KQV Tensor Reader (specialized for reading KQV tensors and their sources)
add_executable(${TARGET_KQV_READER} kqv-tensor-reader.cpp)
install(TARGETS ${TARGET_KQV_READER} RUNTIME DESTINATION bin)
target_link_libraries(${TARGET_KQV_READER} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_KQV_READER} PRIVATE cxx_std_11)

# Tensor Difference Analyzer for comparing current tensors with saved reference tensors
add_executable(llama-tensor-diff-analyzer tensor-diff-analyzer.cpp)
install(TARGETS llama-tensor-diff-analyzer RUNTIME)
target_link_libraries(llama-tensor-diff-analyzer PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-tensor-diff-analyzer PRIVATE cxx_std_17)

# Flash Attention Mixed KV Cache Verification Tool
add_executable(llama-flash-attn-mixed-verify flash-attn-mixed-verify.cpp)
install(TARGETS llama-flash-attn-mixed-verify RUNTIME)
target_link_libraries(llama-flash-attn-mixed-verify PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-flash-attn-mixed-verify PRIVATE cxx_std_17)


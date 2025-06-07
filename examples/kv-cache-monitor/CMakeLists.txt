# KQV Trace Monitor
set(KQV_TRACE_TARGET llama-kqv-trace-monitor)
add_executable(${KQV_TRACE_TARGET} kqv-trace-monitor.cpp)
install(TARGETS ${KQV_TRACE_TARGET} RUNTIME)
target_link_libraries(${KQV_TRACE_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
 target_compile_features(${KQV_TRACE_TARGET} PRIVATE cxx_std_17)

 # KV Quant Monitor
 add_executable(llama-kv-quant-monitor kv-quant-monitor.cpp)
 install(TARGETS llama-kv-quant-monitor RUNTIME)
 target_link_libraries(llama-kv-quant-monitor PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
 target_compile_features(llama-kv-quant-monitor PRIVATE cxx_std_17)

# GGUF Reader for verifying saved tensor files
add_executable(llama-kqv-gguf-reader gguf-reader.cpp)
install(TARGETS llama-kqv-gguf-reader RUNTIME)
target_link_libraries(llama-kqv-gguf-reader PRIVATE ggml)
target_compile_features(llama-kqv-gguf-reader PRIVATE cxx_std_17)

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


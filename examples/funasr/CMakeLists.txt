# Fun-ASR-Nano llama.cpp runtime (single-binary ASR, CUDA/CPU)
#   ported from FunAudioLLM/FunASR runtime-llamacpp:
#   - encoder (SAN-M) + adaptor run on ggml (CUDA backend when available)
#   - LLM (Qwen3-0.6B) runs via llama (offloaded to GPU)
#   - built-in FSMN-VAD front end

set(FUNASR_COMMON ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(nano-cli)

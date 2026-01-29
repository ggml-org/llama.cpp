llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            models/afmoe.cpp
            models/apertus.cpp
            models/arcee.cpp
            models/arctic.cpp
            models/arwkv7.cpp
            models/baichuan.cpp
            models/bailingmoe.cpp
            models/bailingmoe2.cpp
            models/bert.cpp
            models/bitnet.cpp
            models/bloom.cpp
            models/chameleon.cpp
            models/chatglm.cpp
            models/codeshell.cpp
            models/cogvlm.cpp
            models/cohere2-iswa.cpp
            models/command-r.cpp
            models/dbrx.cpp
            models/deci.cpp
            models/deepseek.cpp
            models/deepseek2.cpp
            models/dots1.cpp
            models/dream.cpp
            models/ernie4-5-moe.cpp
            models/ernie4-5.cpp
            models/exaone.cpp
            models/exaone4.cpp
            models/exaone-moe.cpp
            models/falcon-h1.cpp
            models/falcon.cpp
            models/gemma-embedding.cpp
            models/gemma.cpp
            models/gemma2-iswa.cpp
            models/gemma3.cpp
            models/gemma3n-iswa.cpp
            models/glm4-moe.cpp
            models/glm4.cpp
            models/gpt2.cpp
            models/gptneox.cpp
            models/granite-hybrid.cpp
            models/granite.cpp
            models/grok.cpp
            models/grovemoe.cpp
            models/hunyuan-dense.cpp
            models/hunyuan-moe.cpp
            models/internlm2.cpp
            models/jais.cpp
            models/jamba.cpp
            models/lfm2.cpp
            models/llada-moe.cpp
            models/llada.cpp
            models/llama-iswa.cpp
            models/llama.cpp
            models/maincoder.cpp
            models/mamba.cpp
            models/mimo2-iswa.cpp
            models/minicpm3.cpp
            models/minimax-m2.cpp
            models/modern-bert.cpp
            models/mpt.cpp
            models/nemotron-h.cpp
            models/nemotron.cpp
            models/neo-bert.cpp
            models/olmo.cpp
            models/olmo2.cpp
            models/olmoe.cpp
            models/openai-moe-iswa.cpp
            models/openelm.cpp
            models/orion.cpp
            models/pangu-embedded.cpp
            models/phi2.cpp
            models/phi3.cpp
            models/plamo.cpp
            models/plamo2.cpp
            models/plamo3.cpp
            models/plm.cpp
            models/qwen.cpp
            models/qwen2.cpp
            models/qwen2moe.cpp
            models/qwen2vl.cpp
            models/qwen3.cpp
            models/qwen3vl.cpp
            models/qwen3vl-moe.cpp
            models/qwen3moe.cpp
            models/qwen3next.cpp
            models/refact.cpp
            models/rnd1.cpp
            models/rwkv6-base.cpp
            models/rwkv6.cpp
            models/rwkv6qwen2.cpp
            models/rwkv7-base.cpp
            models/rwkv7.cpp
            models/seed-oss.cpp
            models/smallthinker.cpp
            models/smollm3.cpp
            models/stablelm.cpp
            models/starcoder.cpp
            models/starcoder2.cpp
            models/t5-dec.cpp
            models/t5-enc.cpp
            models/wavtokenizer-dec.cpp
            models/xverse.cpp
            models/mistral3.cpp
            models/graph-context-mamba.cpp
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

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
            models/llm_build_arcee.cpp
            models/llm_build_arctic.cpp
            models/llm_build_arwkv7.cpp
            models/llm_build_baichuan.cpp
            models/llm_build_bailingmoe.cpp
            models/llm_build_bert.cpp
            models/llm_build_bitnet.cpp
            models/llm_build_bloom.cpp
            models/llm_build_chameleon.cpp
            models/llm_build_chatglm.cpp
            models/llm_build_codeshell.cpp
            models/llm_build_cohere2_iswa.cpp
            models/llm_build_command_r.cpp
            models/llm_build_dbrx.cpp
            models/llm_build_deci.cpp
            models/llm_build_deepseek.cpp
            models/llm_build_deepseek2.cpp
            models/llm_build_dots1.cpp
            models/llm_build_dream.cpp
            models/llm_build_ernie4_5_moe.cpp
            models/llm_build_ernie4_5.cpp
            models/llm_build_exaone.cpp
            models/llm_build_exaone4.cpp
            models/llm_build_falcon_h1.cpp
            models/llm_build_falcon.cpp
            models/llm_build_gemma_embedding_iswa.cpp
            models/llm_build_gemma.cpp
            models/llm_build_gemma2_iswa.cpp
            models/llm_build_gemma3_iswa.cpp
            models/llm_build_gemma3n_iswa.cpp
            models/llm_build_glm4_moe.cpp
            models/llm_build_glm4.cpp
            models/llm_build_gpt2.cpp
            models/llm_build_gptneox.cpp
            models/llm_build_granite_hybrid.cpp
            models/llm_build_granite.cpp
            models/llm_build_grok.cpp
            models/llm_build_hunyuan_dense.cpp
            models/llm_build_hunyuan_moe.cpp
            models/llm_build_internlm2.cpp
            models/llm_build_jais.cpp
            models/llm_build_jamba.cpp
            models/llm_build_lfm2.cpp
            models/llm_build_llada_moe.cpp
            models/llm_build_llada.cpp
            models/llm_build_llama_iswa.cpp
            models/llm_build_llama.cpp
            models/llm_build_mamba.cpp
            models/llm_build_minicpm3.cpp
            models/llm_build_mpt.cpp
            models/llm_build_nemotron_h.cpp
            models/llm_build_nemotron.cpp
            models/llm_build_neo_bert.cpp
            models/llm_build_olmo.cpp
            models/llm_build_olmo2.cpp
            models/llm_build_olmoe.cpp
            models/llm_build_openai_moe_iswa.cpp
            models/llm_build_openelm.cpp
            models/llm_build_orion.cpp
            models/llm_build_phi2.cpp
            models/llm_build_phi3.cpp
            models/llm_build_plamo.cpp
            models/llm_build_plamo2.cpp
            models/llm_build_plm.cpp
            models/llm_build_qwen.cpp
            models/llm_build_qwen2.cpp
            models/llm_build_qwen2moe.cpp
            models/llm_build_qwen2vl.cpp
            models/llm_build_qwen3.cpp
            models/llm_build_qwen3moe.cpp
            models/llm_build_refact.cpp
            models/llm_build_rwkv6_base.cpp
            models/llm_build_rwkv6.cpp
            models/llm_build_rwkv6qwen2.cpp
            models/llm_build_rwkv7_base.cpp
            models/llm_build_rwkv7.cpp
            models/llm_build_seed_oss.cpp
            models/llm_build_smallthinker.cpp
            models/llm_build_smollm3.cpp
            models/llm_build_stablelm.cpp
            models/llm_build_starcoder.cpp
            models/llm_build_starcoder2.cpp
            models/llm_build_t5_dec.cpp
            models/llm_build_t5_enc.cpp
            models/llm_build_wavtokenizer_dec.cpp
            models/llm_build_xverse.cpp
            models/llm_graph_context_mamba.cpp
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

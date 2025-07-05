# llama.cpp/examples/training

This directory contains examples related to language model training using llama.cpp/GGML.
So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP.
Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory.
**For CPU training, compile llama.cpp without any additional backends such as CUDA.**
**For CUDA training, use the maximum number of GPU layers.**

Proof of concept:

With load data from common file:

``` sh
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune --file wikitext-2-raw/wiki.test.raw -ngl 999 --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
./build/bin/llama-perplexity --file wikitext-2-raw/wiki.test.raw -ngl 999 --model finetuned-model.gguf
```

With load data from parquet (without batching):

You need install arrow package and build with LLAMA_PARQUET=ON

``` sh
mkdir build; cmake -DLLAMA_PARQUET=ON .. ; make
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune -ngl 999 --dataset-format parquet --parquet-path parquet.parquet --tokens-column tokens --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
```
The perplexity value of the finetuned model should be lower after training on the test set for 2 epochs.

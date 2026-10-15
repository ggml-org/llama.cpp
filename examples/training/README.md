# llama.cpp/examples/training

## finetune
This directory contains examples related to language model training using llama.cpp/GGML.
So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP.
Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory.
**For CPU training, compile llama.cpp without any additional backends such as CUDA.**
**For CUDA training, use the maximum number of GPU layers.**

Flash attention is disabled during training because `FLASH_ATTN_EXT` has no backward pass.

Proof of concept:

``` sh
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune --file wikitext-2-raw/wiki.test.raw -ngl 999 --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
./build/bin/llama-perplexity --file wikitext-2-raw/wiki.test.raw -ngl 999 --model finetuned-model.gguf
```

The perplexity value of the finetuned model should be lower after training on the test set for 2 epochs.

## finetune-lora

LoRA fine-tuning trains adapter weights next to the frozen base model, so it needs comparatively
less memory than training the whole model, and the result is a regular adapter file that `llama-cli`,
`llama-server` and `llama-perplexity` load with `--lora`.

Besides LLaMA, the following dense & hybrid architectures have been tested:

| Family   | `general.architecture` |
| -------- | ---------------------- |
| Qwen3    | `qwen3`                |
| Gemma 3  | `gemma3`               |
| Qwen 3.5 | `qwen35`               |

The training options are the same as `llama-finetune` (`-lr`, `-lr-min`, `-wd`, `-epochs`, `-val-split`,
`-opt`) plus the LoRA and checkpoint options below. LoRA finetuning on CUDA is not yet supported.
Flash attention is disabled since `FLASH_ATTN_EXT` has no backward pass.
**Compile llama.cpp without any additional backends such as CUDA.**

``` sh
# train a new adapter: rank 8, alpha 16, attention proj
./build/bin/llama-finetune-lora -m model.gguf -f wiki.test.raw -c 512 -b 512 -ub 512 -o adapter.gguf

# rank 16 on all linear layers, cosine lr schedule with warmup
./build/bin/llama-finetune-lora -m model.gguf -f wiki.test.raw -c 512 -b 512 -ub 512 \
    --lora-rank 16 --lora-alpha 32 --lora-modules all --lr-scheduler cosine --warmup-steps 100 -o adapter.gguf

# continue training an existing adapter
./build/bin/llama-finetune-lora -m model.gguf -f wiki.test.raw -c 512 -b 512 -ub 512 \
    --lora adapter.gguf -o adapter-2.gguf

# eval
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw -c 512 --lora adapter.gguf
```

Options:

- `--lora-rank N`: rank of the adapter (default 8)
- `--lora-alpha A`: scaling factor (default 16)
- `--lora-modules LIST`: comma-separated modules, from `attn_q`, `attn_k`, `attn_v`, `attn_o`,
  `ffn_gate`, `ffn_up`, `ffn_down`, `output`, or `all` (default `attn_q,attn_k,attn_v,attn_o`)
- `--lora-seed N`: seed of the adapter initialization, 0 for a random one (default 42)
- `-o FILENAME`: output adapter file (default `trained-lora-adapter.gguf`)
- `--lr-scheduler constant|cosine|linear`: learning rate over the training steps, cosine and linear decay
  from `-lr` to `-lr-min` (default constant)
- `--warmup-steps N`, `--warmup-ratio F`: linear learning rate warmup over the first steps, given as a count
  or as a fraction of all steps (default none)

### Checkpoints

`--checkpoint-save-steps N` writes a checkpoint every N optimizer steps into `--checkpoint-save-dir`
(default `./checkpoints`). A checkpoint dir has the adapter (`model.gguf`), the optimizer state
(`optimizer.gguf`) and the LoRA params (`metadata.txt`). `--resume-from DIR` continues training
from a ckpt and `--auto-resume` picks the latest one in the checkpoint dir

### Supervised fine-tuning on chat data

With `--assistant-loss-only` the input file is read as JSONL, one conversation per line
`{"messages": [{"role": ..., "content": ...}, ...]}` form. Each conversation is rendered with the model's
chat template, or one given with by `--chat-template-file`, and only the assistant turns contribute to
the loss. ChatML, Gemma and Llama 3 format turn markers are recognized. Conversations longer
than the context are skipped.

``` sh
./build/bin/llama-finetune-lora -m model.gguf -f conversations.jsonl -c 512 -b 512 -ub 512 \
    --assistant-loss-only -o adapter.gguf
```

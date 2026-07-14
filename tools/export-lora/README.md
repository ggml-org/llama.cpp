# export-lora

Apply LORA adapters to base model and export the resulting model.

```
usage: llama-export-lora [options]

options:
  -m,    --model FNAME                  model path from which to load base model
         --lora FNAME                   path to LoRA adapter (use comma-separated values to load multiple adapters)
         --lora-scaled FNAME:SCALE,...  path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)
  -o,    --output, --output-file FNAME  output file (default: 'ggml-lora-merged-f16.gguf')
```

For example:

```bash
./bin/llama-export-lora \
    -m open-llama-3b-v2.gguf \
    -o open-llama-3b-v2-english2tokipona-chat.gguf \
    --lora lora-open-llama-3b-v2-english2tokipona-chat-LATEST.gguf
```

Multiple LORA adapters can be applied by passing comma-separated values to `--lora FNAME` or `--lora-scaled FNAME:SCALE,...`:

```bash
./bin/llama-export-lora \
    -m your_base_model.gguf \
    -o your_merged_model.gguf \
    --lora-scaled lora_task_A.gguf:0.5,lora_task_B.gguf:0.5
```

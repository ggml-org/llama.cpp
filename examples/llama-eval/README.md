# llama.cpp/example/llama-eval

`llama-eval.py` is a single-script evaluation runner that sends prompt/response pairs to any OpenAI-compatible HTTP server (the default `llama-server`).

```bash
./llama-server -m model.gguf --port 8033
python examples/llama-eval/llama-eval.py --path_server http://localhost:8033 --n_prompts 100 --prompt_source arc
```

The supported tasks are:

- **GSM8K** — grade-school math
- **AIME** — competition math (integer answers)
- **MMLU** — multi-domain multiple choice
- **HellaSwag** — commonsense reasoning multiple choice
- **ARC** — grade-school science multiple choice
- **WinoGrande** — commonsense coreference multiple choice

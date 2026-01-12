# llama.cpp/example/llama-eval

The purpose of this example is to to run evaluations metrics against a an openapi api compatible LLM via http (llama-server).

```bash
./llama-server -m model.gguf --port 8033
```

```bash
python examples/llama-eval/llama-eval.py --path_server http://localhost:8033 --n_prompt 100  --prompt_source arc
```

## Supported tasks (MVP)

- **GSM8K** — grade-school math (final-answer only)
- **AIME** — competition math (final-answer only)
- **MMLU** — multi-domain knowledge (multiple choice)
- **HellaSwag** — commonsense reasoning (multiple choice)
- **ARC** — grade-school science reasoning (multiple choice)
- **WinoGrande** — commonsense coreference resolution (multiple choice)
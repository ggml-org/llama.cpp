# llama.cpp/example/parallel

Simplified simulation of serving incoming requests in parallel

## Example

Generate 128 client requests (`-ns 128`), simulating 8 concurrent clients (`-np 8`). The system prompt is shared (`-pps`), meaning that it is computed once at the start. The client requests consist of up to 10 junk questions (`--junk 10`) followed by the actual question.

```bash
llama-parallel -m model.gguf -np 8 -ns 128 --top-k 1 -pps --junk 10 -c 16384
```

> [!NOTE]
> It's recommended to use base models with this example. Instruction tuned models might not be able to properly follow the custom chat template specified here, so the results might not be as expected.

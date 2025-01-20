# llama.cpp/example/run

The purpose of this example is to demonstrate a minimal usage of llama.cpp for running models.

```bash
llama-run granite-code
```

```bash
llama-run -h
Description:
  Runs a llm

Usage:
  llama-run [options] model [prompt]

Options:
  -c, --context-size <value>
      Context size (default: 2048)
  -n, --ngl <value>
      Number of GPU layers (default: 0)
  --temp <value>
      Temperature (default: 0.8)
  -v, --verbose, --log-verbose
      Set verbosity level to infinity (i.e. log all messages, useful for debugging)
  -h, --help
      Show help message

Commands:
  model
      Model is a string with an optional prefix of
      huggingface:// (hf://), ollama://, https:// or file://.
      If no protocol is specified and a file exists in the specified
      path, file:// is assumed, otherwise if a file does not exist in
      the specified path, ollama:// is assumed. Models that are being
      pulled are downloaded with .partial extension while being
      downloaded and then renamed as the file without the .partial
      extension when complete.

Examples:
  llama-run llama3
  llama-run ollama://granite-code
  llama-run ollama://smollm:135m
  llama-run hf://QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf
  llama-run huggingface://bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF/SmolLM-1.7B-Instruct-v0.2-IQ3_M.gguf
  llama-run https://example.com/some-file1.gguf
  llama-run some-file2.gguf
  llama-run file://some-file3.gguf
  llama-run --ngl 999 some-file4.gguf
  llama-run --ngl 999 some-file5.gguf Hello World
```

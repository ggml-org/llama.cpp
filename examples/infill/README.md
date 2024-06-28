# llama.cpp/example/infill

This example shows how to use the infill mode with Code Llama models supporting infill mode.
Currently the 7B and 13B models support infill mode.

Infill supports most of the options available in the main example.

For further information have a look at the main README.md in llama.cpp/example/main/README.md

## Common Options

In this section, we cover the most commonly used options for running the `infill` program with the LLaMA models:

-   `-m FNAME, --model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-i, --interactive`: Run the program in interactive mode, allowing you to provide input directly and receive real-time responses.
-   `-n N, --n-predict N`: Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text.
-   `-c N, --ctx-size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference.
-   `--spm-infill`: Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.

## Input Prompts

The `infill` program provides several ways to interact with the LLaMA models using input prompts:

-   `--in-prefix PROMPT_BEFORE_CURSOR`: Provide the prefix directly as a command-line option.
-   `--in-suffix PROMPT_AFTER_CURSOR`: Provide the suffix directly as a command-line option.
-   `--interactive-first`: Run the program in interactive mode and wait for input right away. (More on this below.)

## Interaction

The `infill` program offers a seamless way to interact with LLaMA models, allowing users to receive real-time infill suggestions. The interactive mode can be triggered using `--interactive`, and `--interactive-first`

### Interaction Options

-   `-i, --interactive`: Run the program in interactive mode, allowing users to get real time code suggestions from model.
-   `--interactive-first`: Run the program in interactive mode and immediately wait for user input before starting the text generation.
-   `--color`: Enable colorized output to differentiate visually distinguishing between prompts, user input, and generated text.

### Example

Download a model that supports infill, for example CodeLlama:
```console
scripts/hf.sh --repo TheBloke/CodeLlama-13B-GGUF --file codellama-13b.Q5_K_S.gguf --outdir models
```

```bash
./llama-infill -t 10 -ngl 0 -m models/codellama-13b.Q5_K_S.gguf -c 4096 --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "
```

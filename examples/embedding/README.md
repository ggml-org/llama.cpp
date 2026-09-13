# llama.cpp/example/embedding

This example demonstrates generate high-dimensional embedding vector of a given text with llama.cpp.

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./llama-embedding -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>/dev/null
```

### Windows:

```powershell
llama-embedding.exe -m ./path/to/model --pooling mean --log-disable -p "Hello World!" 2>$null
```

The above command will output space-separated float values.

## extra parameters
### --embd-normalize $integer$
| $integer$ | description         | formula |
|-----------|---------------------|---------|
| $-1$      | none                |
| $0$       | max absolute int16  | $\Large{{32760 * x_i} \over\max \lvert x_i\rvert}$
| $1$       | taxicab             | $\Large{x_i \over\sum \lvert x_i\rvert}$
| $2$       | euclidean (default) | $\Large{x_i \over\sqrt{\sum x_i^2}}$
| $>2$      | p-norm              | $\Large{x_i \over\sqrt[p]{\sum \lvert x_i\rvert^p}}$

### --embd-output-format $'string'$
| $'string'$ | description                  |  |
|------------|------------------------------|--|
| ''         | same as before               | (default)
| 'array'    | single embeddings            | $[[x_1,...,x_n]]$
|            | multiple embeddings          | $[[x_1,...,x_n],[x_1,...,x_n],...,[x_1,...,x_n]]$
| 'json'     | openai style                 |
| 'json+'    | add cosine similarity matrix |
| 'raw'      | plain text output            |

### --embd-separator $"string"$
| $"string"$   | |
|--------------|-|
| "\n"         | (default)
| "<#embSep#>" | for example
| "<#sep#>"    | other example

## examples
### Unix-based systems (Linux, macOS, etc.):

```bash
./llama-embedding -p 'Castle<#sep#>Stronghold<#sep#>Dog<#sep#>Cat' --pooling mean --embd-separator '<#sep#>' --embd-normalize 2  --embd-output-format '' -m './path/to/model.gguf' --n-gpu-layers 99 --log-disable 2>/dev/null
```

### Windows:

```powershell
llama-embedding.exe -p 'Castle<#sep#>Stronghold<#sep#>Dog<#sep#>Cat' --pooling mean --embd-separator '<#sep#>' --embd-normalize 2  --embd-output-format '' -m './path/to/model.gguf' --n-gpu-layers 99 --log-disable 2>/dev/null
```

## late-interaction (ColBERT) models

ColBERT-style models emit one embedding vector **per token** instead of a single pooled vector. Convert and run them with `--pooling none`:

```bash
./llama-embedding -m ./path/to/colbert-model.gguf --pooling none --embd-normalize -1 -p "Hello World!" 2>/dev/null
```

For sentence-transformers exports that store the final projection as a Dense module directly after the transformer (e.g. [PyLate](https://github.com/lightonai/pylate) models such as [GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)), `convert_hf_to_gguf.py` detects the Dense module via `modules.json` automatically and the projection is applied in-graph: each token row is emitted at the projected width (`llama_model_n_embd_out()`, e.g. 128) rather than the hidden size.

Query/document marker tokens, query expansion, and MaxSim scoring are the responsibility of the client application.

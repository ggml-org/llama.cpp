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

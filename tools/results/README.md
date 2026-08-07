# Results

The `llama-results` tool can be used to `--check` the outputs of a model vs. a previous commit to detect whether they have changed.
Example usage:

``` sh
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed."  # writes results to file
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed." --check  # compares results vs file
```

The metric by which the results are compared is the normalized mean squared error (NMSE) with a tolerance of $10^{-6}$.

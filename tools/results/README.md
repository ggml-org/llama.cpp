# Results

The `llama-results` tool can be used to `--check` the outputs of a model vs. a previous commit to detect whether they have changed.
Example usage:

``` sh
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed."  # writes results to file
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed." --check  # compares results vs file
```

The metric by which the results are compared is the normalized mean squared error (NMSE) with a tolerance of $10^{-6}$.

Use `--decode-steps 32` on both invocations to also compare fixed-input decoding, clear/refill, and a saved-state restore followed by another decode. Each decode input is taken cyclically from the prompt tokens, so sampling does not change the workload between runs. The context must fit the prompt plus all decode steps and one final token. All recorded logits must be finite; the comparison applies the NMSE limit separately to every token and reports the worst value.

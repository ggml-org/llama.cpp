# llama.cpp/examples/debug

This is a utility intended to help debug a model by registering a callback that
logs GGML operations and tensor data. It can also store the generated logits or
embeddings as well as the prompt and token ids for comparision with the original
model.

### Usage

```shell
llama-debug \
  --hf-repo ggml-org/models \
  --hf-file phi-2/ggml-model-q4_0.gguf \
  --model phi-2-q4_0.gguf \
  --prompt hello \
  --save-logits \
  --verbose
```
The tensor data is logged as debug and required the --verbose flag. The reason
for this is that while useful for a model with many layers there can be a lot of
output. You can filter the tensor names using the `--tensor-filter` option.

A recommended approach is to first run without `--verbose` and see if the
generated logits/embeddings are close to the original model. If they are not,
then it might be required to inspect tensor by tensor and in that case it is
useful to enable the `--verbose` flag along with `--tensor-filter` to focus on
specific tensors.

### Options
This example supports all standard `llama.cpp` options and also accepts the
following options:
```console
$ llama-debug --help
...

----- example-specific params -----

--save-logits                           save final logits to files for verification (default: false)
--logits-output-dir PATH                directory for saving logits output files (default: data)
--tensor-filter REGEX                   filter tensor names for debug output (regex pattern, can be specified multiple times)
```

### Output Files

When `--save-logits` is enabled, the following files are created in the output
directory:

* `llamacpp-<model>[-embeddings].bin`        - Binary output (logits or embeddings)
* `llamacpp-<model>[-embeddings].txt`        - Text output (logits or embeddings, one per line)
* `llamacpp-<model>[-embeddings]-prompt.txt` - Prompt text and token IDs
* `llamacpp-<model>[-embeddings]-tokens.bin` - Binary token IDs for programmatic comparison

These files can be compared against the original model's output to verify the
converted model.

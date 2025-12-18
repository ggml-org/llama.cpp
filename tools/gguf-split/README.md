## GGUF split Example

CLI to split / merge GGUF files.

**Command line options:**

- `--split`: split GGUF to multiple GGUF, default operation.
- `--split-max-size`: max size per split in `M` or `G`, f.ex. `500M` or `2G`.
- `--split-max-tensors`: maximum tensors in each split: default(128)
- `--merge`: merge multiple GGUF to a single GGUF.

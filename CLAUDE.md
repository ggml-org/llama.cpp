IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# AI Policy
- AI is assistive only; AI-generated PRs are restricted per AGENTS.md
- Contributor reviews and writes code themselves

# Code Style & Conventions
- snake_case naming; optimize for longest common prefix
- 4 spaces indentation, brackets on same line
- `void * ptr`, `int & a` (space around pointer/reference)
- Avoid templates and fancy STL
- Use sized integer types (`int32_t`) in public API
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines, naming, and PR process

# ggml Tensor Conventions
- Data stored in row-major order
- Dimension 0 = columns, dimension 1 = rows, dimension 2 = matrices
- **Matrix multiply is unconventional**: `C = ggml_mul_mat(ctx, A, B)` means `C^T = A * B^T`

# Quantization
- See [docs/quantization/](docs/quantization/) for comprehensive documentation
- See [docs/quantization/09-adding-new-types.md](docs/quantization/09-adding-new-types.md) for adding new types

## Key Files
- `ggml/include/ggml.h`: type enums (`ggml_type`)
- `ggml/src/ggml-common.h`: block structures
- `ggml/src/ggml-quants.c`: reference quantize/dequantize implementations
- `tools/quantize/quantize.cpp`: CLI tool
- `src/llama-quant.cpp`: core quantization engine

## Quantization Families
- **Q**: simple uniform quantization
- **K**: super-block quantization (multiple sub-blocks per super-block)
- **IQ**: importance-weighted quantization
- **T**: ternary quantization
- **MXFP**: Microsoft floating-point quantization

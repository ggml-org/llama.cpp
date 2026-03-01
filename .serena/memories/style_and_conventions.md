# Code Style and Conventions

## Naming
- `snake_case` for all functions, variables, types
- Optimize for longest common prefix: `number_small`, `number_big` (not `small_number`)
- Pattern: `<class>_<method>` with method being `<action>_<noun>`
  - `llama_model_init()` - class: llama_model, method: init
  - `llama_sampler_get_seed()` - class: llama_sampler, method: get_seed

## Formatting
- 4 spaces indentation
- Brackets on same line
- `void * ptr`, `int & a` (space around pointer/reference)
- Vertical alignment for readability

## Design Principles
- Minimal dependencies (avoid third-party libs)
- Cross-platform (Linux, macOS, Windows)
- Simple STL usage: basic for loops, minimize templates
- Public API types: use int32_t etc., size_t for allocation sizes

## Matrix Multiplication Convention (IMPORTANT)
`C = ggml_mul_mat(ctx, A, B)` computes: C^T = A * B^T ⟺ C = B * A^T

## Tensor Storage
- Row-major order
- Dimension 0 = columns, Dimension 1 = rows, Dimension 2 = matrices

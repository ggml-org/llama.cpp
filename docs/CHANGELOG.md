# Changelog

## Unreleased

### Breaking changes
- `ggml_tensor` gained a `layout` pointer and updated padding, which changes struct size and
  layout. Rebuild any code that depends on `ggml_tensor` or uses `sizeof(ggml_tensor)`.
  You can define `GGML_TENSOR_STRUCT_VERSION_EXPECTED` before including `ggml.h` to
  enforce the expected ABI version at compile time.

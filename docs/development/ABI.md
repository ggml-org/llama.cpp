# ABI Compatibility Notes

## ggml_tensor layout field (2026-01)

The `ggml_tensor` struct now includes a `layout` pointer and updated padding aligned to
`GGML_MEM_ALIGN`. This changes the struct size and memory layout, so any binary compiled
against older `ggml.h` headers is ABI-incompatible.

### Impact
- Code using `sizeof(ggml_tensor)` or pointer arithmetic over `ggml_tensor` arrays will break.
- Libraries compiled with old headers may read or write the wrong fields at runtime.

### Migration
1. Rebuild any dependent binaries and shared libraries against the updated headers.
2. Avoid mixing older `ggml` libraries with newer headers (or vice versa).
3. If you ship a prebuilt integration, bump your package version to reflect the ABI break.

### Optional compile-time guard
Define `GGML_TENSOR_STRUCT_VERSION_EXPECTED` before including `ggml.h` to ensure you are
building against the expected ABI:

```c
#define GGML_TENSOR_STRUCT_VERSION_EXPECTED 2
#include "ggml.h"
```

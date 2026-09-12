# Initial fork integration validation

Base revision: `d230ddd763ffe27781c7ffd237ea78b639b36b6d`.

The documented static AVX2 configuration was built from this checkout with GCC 15.1.0 on Windows x86-64. No prebuilt experimental library was used as a reference.

| Check | Result |
| --- | --- |
| `test-q4kp-kernels`, `test-q4kp-runtime` | Built |
| `llama-completion`, `llama-bench` | Built; version/help smoke passed |
| CTest `q4kp-*` | 6/6 passed |
| Python synthetic kernel suite | 17/17 passed, no ISA skips on the test machine |
| Runtime modes off/p6/vnni/wide/invalid | 231 checks per mode passed |
| Separate `GGML_CPU_Q4KP=OFF` CPU backend | Built |

The Python suite compared the original AVX2 kernels with P6 and independent scalar outputs (45 GEMV cases, 116 GEMM cases), VNNI (104 GEMV cases), and wide (122 GEMV cases), including output guards and input immutability. Runtime checks used canonical Q4_K data, repeated tensor loading, aligned and rejected views, and two-thread graphs with 1, 4, 5 and 20 activation rows.

Unrelated upstream MinGW warnings remain in the subprocess helper and benchmark date formatting. The current server UI embedding tool failed with a non-ASCII build path; the documented build uses completion and benchmark tools with server/app disabled. Neither warning-free upstream builds nor server/UI support is claimed here.

Linux PIC requirements were checked in source and addressed for the shared synthetic test library; Linux and Clang execution were not performed. Unsupported ISA fallback was reviewed in source, but no separate machine lacking VNNI was used. This record is not a model accuracy certification or an end-to-end speedup claim. Re-run the tests after source, compiler or CPU changes.

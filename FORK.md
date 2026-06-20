# Nueramarcos/llama.cpp

Personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

Scope: local validation and smoke checks only — not a distribution fork.

## Quick validate

```bash
git clone https://github.com/Nueramarcos/llama.cpp.git
cd llama.cpp
cmake --version
test -f CMakeLists.txt && test -f ggml/CMakeLists.txt && echo ok
```

## Build hint

```bash
cmake -B build
cmake --build build -j
```

Upstream: https://github.com/ggml-org/llama.cpp

Maintained by [Nueramarcos](https://github.com/Nueramarcos).
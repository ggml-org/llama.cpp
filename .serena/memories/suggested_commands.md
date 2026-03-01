# Suggested Commands

## Environment Setup (REQUIRED before build/run)
```bash
source /opt/intel/oneapi/setvars.sh --force
```

## Build
```bash
cmake -B build -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
ninja -C build -j $(nproc)
```

## Testing
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R <test-name> -V   # single test
./build/bin/test-backend-ops               # ggml operator tests
```

## Run Inference
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

## Benchmarking
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

## Code Formatting
```bash
clang-format-19 -i <file.cpp>
clang-format-19 --dry-run -Werror <file.cpp>  # dry-run check
```

## Device Selection
```bash
sycl-ls                                    # list devices
ONEAPI_DEVICE_SELECTOR=level_zero:0 ...    # select Arc B580
```

## Debug Environment Variables
- GGML_SYCL_UNIFIED_FORCE_LEGACY=1 - Force legacy kernels
- GGML_SYCL_ONEDNN_PP=0 - Disable oneDNN for prompt processing
- GGML_SYCL_DEBUG=1 - Enable kernel dispatch logging

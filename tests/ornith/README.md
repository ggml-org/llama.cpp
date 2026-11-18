# Standalone checks for the repack-backend changes

Not wired into CMake: they link the static ggml libraries of a finished build
and include the CPU backend's internal headers.

```sh
L=/path/to/llama.cpp; B=$L/build            # a build with BUILD_SHARED_LIBS=OFF and GGML_NATIVE=ON
I="-I$L/ggml/include -I$L/ggml/src -I$L/ggml/src/ggml-cpu"
LIBS="-Wl,--start-group $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a -Wl,--end-group -lpthread -lm -ldl"
# add -fopenmp to LIBS when the build used GGML_OPENMP=ON
g++ -O2 -march=native -std=c++17 $I test_q6k_repack.cpp -o test_q6k_repack $LIBS && ./test_q6k_repack
g++ -O2 -march=native -std=c++17 $I bench_moe_ops.cpp  -o bench_moe_ops  $LIBS && ./bench_moe_ops 16
```

- `test_q6k_repack.cpp`: the AVX-512 Q6_K 8x8 GEMV/GEMM against the generic
  kernels and an f32 reference over K=256..2048, N=8..512, M=1..20.
- `bench_moe_ops.cpp`: the model's hot matmuls (MoE experts at M=1/512/2048,
  dense Q6_K projections, the output head) through the plain CPU buffer and the
  repack buffer, with timing and the maximum relative difference. `BENCH_TAILS=1`
  uses heavy-tailed activations; a second argument filters cases by substring.

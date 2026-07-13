#!/bin/bash
# Build libppu_fa3.so: FlashAttention-3 (hopper/) sm90 fwd kernels + torch-free shim. Links only CUDA runtime.
set -e
HOP=/root/llama.cpp/thirdparty/flash-attention/hopper
CUT=/root/llama.cpp/thirdparty/flash-attention/csrc/cutlass/include
TI=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__)+'/include')")
OUT=/root/llama.cpp/ppu_so/fa3/build; mkdir -p $OUT
FLAGS="-gencode arch=compute_90a,code=sm_90a -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler -fPIC -I$HOP -I$CUT -I$TI -I$TI/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1 -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_FP8 -DFLASHATTENTION_DISABLE_DROPOUT --threads 2"
# instances (fwd sm90 fp16/bf16, non-split/paged/packgqa) + shim
SRCS=""
for hd in 64 96 128 192 256; do for dt in fp16 bf16; do SRCS="$SRCS $HOP/instantiations/flash_fwd_hdim${hd}_${dt}_sm90.cu"; done; done
echo "[fa3] compiling 10 instances (parallel)..."
pids=""
for s in $SRCS; do
    o=$OUT/$(basename $s .cu).o
    nvcc -c "$s" -o "$o" $FLAGS >$OUT/$(basename $s).log 2>&1 &
    pids="$pids $!"
done
nvcc -c /root/llama.cpp/ppu_so/fa3/flash3_api_c.cpp -o $OUT/shim.o $FLAGS >$OUT/shim.log 2>&1 &
pids="$pids $!"
fail=0; for p in $pids; do wait $p || fail=1; done
[ $fail -ne 0 ] && { echo "[fa3] COMPILE FAILED"; grep -l -iE "error" $OUT/*.log; exit 1; }
echo "[fa3] linking libppu_fa3.so..."
nvcc -shared -o /root/llama.cpp/ppu_so/fa3/libppu_fa3.so $OUT/*.o -Xcompiler -fPIC -lcudart
echo "[fa3] done. torch check:"; ldd /root/llama.cpp/ppu_so/fa3/libppu_fa3.so | grep -iE "torch|c10" || echo "  torch-free (only cudart)"
nm -D /root/llama.cpp/ppu_so/fa3/libppu_fa3.so | grep ppu_flash_attn_fwd

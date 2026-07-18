# ROCm/AMD Fixes — Moltes94 fork

> **Branch**: `fix/rocm-recurrent-checkpoints`
> **PR**: https://github.com/ggml-org/llama.cpp/pull/24785
> **Hardware**: AMD Radeon RX 7900 XTX (24 GB), ROCm 7.2, CachyOS (Arch)

---

## Issues fixed

| Issue | Symptom | Impact |
|---|---|---|
| **[#22746](https://github.com/ggml-org/llama.cpp/issues/22746)** | Forced full prompt re-processing on Gated DeltaNet models (Qwen3.6) | **207s prefill per turn** at 38K context — unusable for agents |
| **[#20176](https://github.com/ggml-org/llama.cpp/issues/20176)** | Context checkpoints crash AMD GPUs | `--ctx-checkpoints 0` required, worsens re-processing |
| **[#23322](https://github.com/ggml-org/llama.cpp/issues/23322)** | Low MTP acceptance on RX 7900 XTX | Corrupted recurrent state degrades speculative decoding |

## What this fork does

Backports the `recurrent_shrink` / `recurrent_expand` mechanism from
[BeeLlama](https://github.com/Anbeeld/beellama.cpp) into mainline llama.cpp
to properly handle the recurrent state during prompt cache operations.

- **Shrink** recurrent state to 1 cell before prompt cache save/load
- **Expand** back to N cells after, allowing context checkpoints to work
- **Auto-detect** AMD GPUs and disable checkpoints on non-recurrent models

## Results (Qwen3.6-35B-A3B MoE MTP, n=2, 96K ctx)

| Turn | Context | tg | MTP accept. | Re-processing |
|------|---------|----|-------------|---------------|
| 1 | 21K | — | 100% | No |
| 2 | 22K | — | 60% | No |
| 3 | 22K | 63.7 t/s | 68.5% | No |
| 4 | 23K | 93.5 t/s | 52.4% | No |
| 5 | 24K | — | 46.2% | No |
| ... | ... | ... | ... | ... |
| 15 | 42K | 95 t/s | 67.2% | No |

Longest tested context: **83K tokens**. Sustained tg: **84-105 t/s**.

## Build

```bash
# Requires ROCm 7.x installed at /opt/rocm
cat > /tmp/gcc-wrap.sh << 'EOF'
#!/bin/bash
args=()
for arg in "$@"; do
    case "$arg" in
        -Wunreachable-code-break|-Wunreachable-code-return) ;;
        *) args+=("$arg") ;;
    esac
done
exec /usr/bin/gcc "${args[@]}"
EOF
chmod +x /tmp/gcc-wrap.sh

CC=/tmp/gcc-wrap.sh CXX=/opt/rocm/bin/hipcc cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build -j8 --target llama-server
```

## Launch

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 ROCR_VISIBLE_DEVICES=0 \
./build/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 999 --n-cpu-moe 0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -c 98304 -b 4096 -ub 1024 \
  -ctk q4_0 -ctv q4_0 -fa on --jinja \
  --no-host --cache-ram 10240 \
  --temp 0.7 --repeat-penalty 1.15 \
  --reasoning off \
  --host 127.0.0.1 --port 1234 --metrics -np 1
```

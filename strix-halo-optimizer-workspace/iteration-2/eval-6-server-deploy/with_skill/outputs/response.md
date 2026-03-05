# Optimal llama-server Configuration for Strix Halo (Llama 3 8B Q4_K_M, 4-5 Concurrent Users)

## Prerequisites (Do These First)

### 1. Increase TTM Memory Limits

This is critical. Without it, the GPU can only access a small fraction of system RAM and you will hit OOM errors under sustained multi-user load.

Create `/etc/modprobe.d/increase_amd_memory.conf`:
```
options ttm pages_limit=25600000
options ttm page_pool_size=25600000
```

Apply and reboot:
```bash
sudo update-initramfs -u -k all
sudo reboot
```

Verify after reboot:
```bash
cat /sys/module/ttm/parameters/pages_limit
# Should show: 25600000
```

### 2. Install ROCm 6.1+

Verify your installation:
```bash
rocminfo | grep -i "Name:" | grep gfx
# Should show gfx1150 or gfx1151
```

Make sure your user is in the `video` and `render` groups:
```bash
sudo usermod -a -G video,render $USER
```

### 3. Remove HSA_OVERRIDE_GFX_VERSION If Set

If you followed older guides, you may have this set. It forces generic kernels and hurts performance. Remove it from your `.bashrc` or shell profile.

## Build Configuration

Use the optimized HIP/ROCm build targeting gfx1150 specifically:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

Key flags explained:
- **`GPU_TARGETS="gfx1150"`**: Compiles kernels specifically for RDNA 3.5 instead of a bloated multi-arch build. Cuts compile time from 30+ minutes to under 10 and produces better-optimized code.
- **`GGML_HIP_GRAPHS=ON`**: HIP graph capture reduces kernel dispatch overhead, which matters under concurrent load.
- **`GGML_CUDA_FORCE_MMQ=ON`**: Forces quantized matmul kernels, which are often faster than hipBLAS on RDNA 3.5 for quantized models like Q4_K_M.

If you have rocWMMA v2.0+ installed, also add `-DGGML_HIP_ROCWMMA_FATTN=ON` for optimized flash attention kernels.

Verify the build:
```bash
./build/bin/llama-bench --list-devices
# Should show: AMD Radeon Graphics (gfx1150 or gfx1151)
```

## Server Launch Configuration

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-server \
  -m /path/to/llama-3-8b-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -np 5 \
  -b 2048 -ub 512 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
```

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-ngl 99` | Full GPU offload | The Radeon 890M handles all layers well for an 8B model. Unified memory means zero transfer penalty. |
| `--no-mmap` | Disable memory mapping | Critical for stable operation. HIP's `hipMemcpy()` has severe overhead with mmap'd pages under sustained load. |
| `-fa on` | Flash attention | Reduces memory usage for long contexts and is WMMA-accelerated on RDNA 3.5. |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | With 5 parallel slots, each needing its own KV cache, this cuts memory usage roughly in half vs f16 with negligible quality loss. |
| `-c 8192` | 8K context length | Good balance for most use cases. Each slot gets up to 8K tokens of context. |
| `-np 5` | 5 parallel slots | Matches your team size (4-5 users). Each user gets a dedicated slot so requests don't queue behind each other. |
| `-b 2048 -ub 512` | Batch sizes | Good defaults for prompt processing throughput on Strix Halo. |
| `-t 4` | 4 CPU threads | Keep this low. CPU and GPU share the ~256 GB/s memory bus, so excess CPU threads compete with the GPU for bandwidth and hurt token generation speed. |
| `ROCBLAS_USE_HIPBLASLT=1` | hipBLASLt kernels | Without this, prompt processing is 2-3x slower (~350 t/s vs ~880 t/s for 7B Q4). Always set this. |

### Memory Budget Estimate

For Llama 3 8B Q4_K_M with this configuration:
- **Model weights**: ~4.9 GB (8B params at ~4.8 bits/weight for Q4_K_M)
- **KV cache (q8_0, 5 slots, 8K context)**: ~2.5 GB approximately
- **Runtime overhead**: ~1-2 GB
- **Total**: ~8-9 GB -- well within any Strix Halo system's unified memory

This leaves plenty of headroom. If your team needs longer contexts, you can safely increase `-c` to 16384 or even 32768.

## Expected Performance

With this configuration, expect approximately:
- **Prompt processing**: ~800-900 tokens/s (with hipBLASLt enabled)
- **Token generation (single user)**: ~40-50 tokens/s
- **Token generation (5 concurrent)**: Throughput is shared across slots, so individual users will see lower per-user speeds under full concurrent load, but aggregate throughput remains high

## Setting Up as a Systemd Service

For production stability, run llama-server as a systemd service so it auto-restarts on failure and starts on boot.

Create `/etc/systemd/system/llama-server.service`:
```ini
[Unit]
Description=llama.cpp Inference Server
After=network.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama.cpp
Environment=ROCBLAS_USE_HIPBLASLT=1
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /opt/models/llama-3-8b-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  -np 5 \
  -b 2048 -ub 512 \
  -t 4 \
  --host 0.0.0.0 \
  --port 8080
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server
sudo systemctl status llama-server
```

Key service settings:
- **`Restart=on-failure`** with **`RestartSec=10`**: Auto-restarts the server if it crashes, with a 10-second cooldown.
- **`LimitMEMLOCK=infinity`**: Allows HIP to lock memory pages as needed.
- **`LimitNOFILE=65536`**: Prevents file descriptor exhaustion under concurrent connections.
- **`Environment=ROCBLAS_USE_HIPBLASLT=1`**: The hipBLASLt variable must be set in the service environment, not just your shell profile.

## Monitoring

### Health Check

The server exposes a `/health` endpoint:
```bash
curl http://localhost:8080/health
```

You can use this for load balancer health checks or a simple watchdog script.

### Metrics

The `/metrics` endpoint provides Prometheus-compatible metrics for monitoring request throughput, latency, and slot utilization.

### GPU Monitoring

Monitor GPU memory and utilization with:
```bash
watch -n 2 rocm-smi
```

Watch for TTM-related OOM under sustained multi-user load. If you see errors, verify your TTM `pages_limit` is correctly set.

## Tuning for Your Workload

### If users need longer contexts

Increase `-c` (e.g., `-c 16384` or `-c 32768`). With q8_0 KV cache and a relatively small 8B model, you have plenty of memory headroom. If memory becomes tight at very large contexts, switch to `-ctk q4_0 -ctv q4_0` for additional savings at the cost of slight quality degradation on very long contexts.

### If you add more concurrent users

Increase `-np` to match. Each additional slot adds KV cache memory proportional to context length. For an 8B Q4_K_M model, each 8K-context slot with q8_0 KV cache costs roughly 0.5 GB, so scaling to 8-10 users is feasible.

### If prompt processing feels slow

Verify `ROCBLAS_USE_HIPBLASLT=1` is set in the server's environment (check `systemctl show llama-server | grep Environment`). This is the most common cause of unexpectedly slow prompt ingestion.

### Benchmarking your specific setup

Before going to production, run a quick benchmark to validate performance:
```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench \
  -m /path/to/llama-3-8b-Q4_K_M.gguf \
  -ngl 99 \
  --no-mmap \
  -t 4 \
  -p 512 -n 128 \
  -r 3
```

You should see pp512 in the 800-900 t/s range and tg128 around 40-50 t/s. If prompt processing is below 400 t/s, hipBLASLt is not active.

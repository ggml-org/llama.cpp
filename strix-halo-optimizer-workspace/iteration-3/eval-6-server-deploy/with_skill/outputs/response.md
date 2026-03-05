# Optimal llama-server Configuration for Strix Halo: Llama 3 8B Q4_K_M with 4-5 Concurrent Users

## Prerequisites (Do These First)

### 1. Increase TTM Memory Limits (Critical)

Without this, the GPU can only access a small fraction of system RAM and you will hit OOM errors even with free memory available.

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

### 2. Remove HSA_OVERRIDE_GFX_VERSION (If Set)

Older guides recommended overriding the GFX version, but llama.cpp now has native RDNA 3.5 support. Using the override forces slower generic kernels. Check and remove if present:

```bash
echo $HSA_OVERRIDE_GFX_VERSION
grep -n HSA_OVERRIDE_GFX_VERSION ~/.bashrc
```

### 3. Verify ROCm Installation

```bash
rocminfo | grep -i "Name:" | grep gfx
# Should show: gfx1150 or gfx1151

rocm-smi --showid
```

Ensure your user is in the required groups:
```bash
sudo usermod -a -G video,render $USER
```

## Build Configuration

Use the optimized HIP build targeting your exact GPU architecture:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1150" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3"

cmake --build build --config Release -j $(nproc)
```

Key flags explained:

| Flag | Why |
|------|-----|
| `GPU_TARGETS="gfx1150"` | Compiles kernels specifically for RDNA 3.5 instead of a bloated multi-arch build. Reduces compile time from 30+ minutes to under 10 and produces better-optimized code. |
| `GGML_HIP_GRAPHS=ON` | Enables HIP graph capture, reducing kernel dispatch overhead -- important for server workloads with sustained throughput. |
| `GGML_CUDA_FORCE_MMQ=ON` | Forces quantized matmul kernels instead of hipBLAS. Often faster on RDNA 3.5 for quantized models like Q4_K_M. |
| `GGML_NATIVE=ON` | Auto-detects and enables all CPU features on the build machine (AVX-512, BF16, etc.) for any CPU-side work. |

If you have rocWMMA v2.0+ installed, add `-DGGML_HIP_ROCWMMA_FATTN=ON` for optimized flash attention kernels via WMMA instructions.

Verify the build:
```bash
./build/bin/llama-bench --list-devices
# Should show: AMD Radeon Graphics (gfx1150 or gfx1151)
```

## Production Server Configuration

### Environment Variables

Add these to your shell profile or systemd service file for persistence:

```bash
# Critical: 2-3x improvement in prompt processing speed
export ROCBLAS_USE_HIPBLASLT=1

# Pin to the iGPU if needed
export HIP_VISIBLE_DEVICES=0
```

The `ROCBLAS_USE_HIPBLASLT=1` variable is essential. Without it, prompt processing on gfx1150 runs at roughly 350 t/s instead of ~880 t/s for a 7-8B Q4 model -- a 2.5x penalty.

### Server Launch Command

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-server \
  -m /path/to/llama-3-8b-q4_k_m.gguf \
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
| `-ngl 99` | Full GPU offload | The Radeon 890M handles token generation well, and unified memory means zero PCIe transfer penalty. For an 8B Q4_K_M model (~4.9 GB), this fits easily in the unified memory pool. |
| `--no-mmap` | Disable memory mapping | HIP's `hipMemcpy()` must lock/unlock mmap pages, causing severe overhead during model loading. With `--no-mmap`, loading is fast and clean. |
| `-fa on` | Flash attention enabled | WMMA-accelerated on RDNA 3.5. Reduces memory usage and improves attention computation speed. |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV cache memory compared to the f16 default with negligible quality impact. With 5 parallel slots, each user needs their own KV cache, so this is especially important for multi-user serving. |
| `-c 8192` | Context size | Llama 3 8B supports up to 8192 tokens natively. Set this to match your workload. If your users send shorter prompts, you can reduce this to 4096 to save memory and improve throughput. |
| `-np 5` | 5 parallel slots | One slot per concurrent user. Matches your team size of 4-5 users. Each slot maintains its own KV cache and can process requests independently via continuous batching. |
| `-b 2048 -ub 512` | Batch sizes | Good defaults for Strix Halo. The logical batch size of 2048 allows efficient prompt processing; the physical batch size of 512 controls actual GPU computation granularity. |
| `-t 4` | 4 CPU threads | Critical for Strix Halo. CPU and GPU share the ~256 GB/s memory bus. High thread counts compete with the GPU for bandwidth, degrading token generation speed. 2-4 threads is optimal for GPU-primary inference. |

### Memory Budget Estimate

For Llama 3 8B Q4_K_M with 5 parallel users:

| Component | Approximate Memory |
|-----------|-------------------|
| Model weights (Q4_K_M, ~4.8 bits/weight) | ~5 GB |
| KV cache per slot (8192 ctx, q8_0, 8B model) | ~0.5 GB |
| KV cache total (5 slots) | ~2.5 GB |
| Runtime overhead | ~1 GB |
| **Total** | **~8.5 GB** |

This fits very comfortably in the Strix Halo's unified memory pool (up to 128 GB), leaving plenty of headroom for the OS and other applications.

## Expected Performance

Based on the Strix Halo architecture specs and benchmark data from the skill references:

| Metric | Expected Range |
|--------|---------------|
| Prompt processing (pp512) | ~700-900 t/s (with hipBLASLt) |
| Token generation (single user) | ~40-50 t/s |
| Token generation (5 concurrent users) | ~8-10 t/s per user (shared bandwidth) |

Token generation speed scales inversely with concurrent users because it is memory-bandwidth bound. With 5 users generating simultaneously, the ~256 GB/s bandwidth is shared across all active sequences.

## Production Hardening

### systemd Service File

Create `/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=llama.cpp Inference Server
After=network.target

[Service]
Type=simple
User=llama
Group=llama
Environment="ROCBLAS_USE_HIPBLASLT=1"
Environment="HIP_VISIBLE_DEVICES=0"
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /opt/models/llama-3-8b-q4_k_m.gguf \
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

### Health Monitoring

llama-server exposes monitoring endpoints. You can poll these for health checks:

```bash
# Basic health check
curl http://localhost:8080/health

# Slot utilization and metrics
curl http://localhost:8080/metrics
```

### Reverse Proxy (Optional)

For TLS termination and basic auth, place nginx or caddy in front:

```nginx
server {
    listen 443 ssl;
    server_name llm.internal.example.com;

    ssl_certificate /etc/ssl/certs/llm.crt;
    ssl_certificate_key /etc/ssl/private/llm.key;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;  # LLM responses can take time
        proxy_send_timeout 300s;
    }
}
```

## Benchmarking Before Going Live

Before deploying, validate performance on your specific hardware:

```bash
export ROCBLAS_USE_HIPBLASLT=1

./build/bin/llama-bench \
  -m /path/to/llama-3-8b-q4_k_m.gguf \
  -ngl 99 \
  --no-mmap \
  -t 4 \
  -p 512 -n 128 \
  -r 3
```

Expected results for 8B Q4_K_M:
- **pp512**: 700-900 t/s (prompt processing -- confirms hipBLASLt is working)
- **tg128**: 40-50 t/s (token generation -- confirms GPU offload is working)

If pp512 is below 400 t/s, `ROCBLAS_USE_HIPBLASLT=1` is not taking effect. Double-check the environment variable is set in the shell running the benchmark.

If tg128 is below 30 t/s, check that `-ngl 99` is actually offloading all layers (look for GPU layer count in the startup logs) and that no other applications are consuming GPU bandwidth.

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| OOM with plenty of free RAM | Increase TTM `pages_limit` (see Prerequisites above) |
| Slow prompt processing (~350 t/s instead of ~880 t/s) | Set `ROCBLAS_USE_HIPBLASLT=1` |
| Model loading hangs or is very slow | Use `--no-mmap` (already in recommended config) |
| "No HIP devices found" | Install ROCm 6.1+, add user to video/render groups, verify with `rocminfo` |
| Poor token generation speed | Rebuild with `GGML_CUDA_FORCE_MMQ=ON`, reduce thread count to `-t 4` |
| `rocm-smi` shows tiny VRAM | Normal for unified memory -- the GPU uses system RAM via the unified pool, not dedicated VRAM |
| Performance regression after update | Pin to a known-good commit; try toggling `GGML_CUDA_FORCE_MMQ` on/off |

## Tuning for Your Workload

If your team's usage patterns differ from the defaults:

- **Short prompts, fast responses**: Reduce `-c` to 4096 to save KV cache memory and slightly improve throughput.
- **Long document processing**: Increase `-c` up to 8192 (Llama 3 native limit). Consider `-ctk q4_0 -ctv q4_0` if memory gets tight, though q8_0 is preferred for quality.
- **Fewer concurrent users**: Reduce `-np` to match actual concurrency. Each unused slot still reserves KV cache memory.
- **Batch processing (not interactive)**: Increase `-b 4096` for better prompt processing throughput on long inputs.

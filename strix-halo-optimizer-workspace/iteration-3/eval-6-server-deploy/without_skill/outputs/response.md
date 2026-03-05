# Deploying llama-server on AMD Strix Halo for Production Use

## Hardware Context: AMD Strix Halo (Ryzen AI 300 Series)

- **CPU**: Zen 5 cores with AVX-512 support, large L3 cache, CCX topology
- **iGPU**: RDNA 3.5 (Radeon 890M) with Vulkan compute support, shared unified memory with CPU
- **Memory**: Unified LPDDR5X (typically 32-64 GB), shared between CPU and iGPU
- **Key advantage**: Unified memory architecture means CPU and iGPU share the same physical memory, eliminating PCIe transfer bottlenecks

For Llama 3 8B Q4_K_M (~4.9 GB model size), this hardware is well-suited -- the model fits entirely in RAM with ample headroom for KV caches serving 4-5 concurrent users.

---

## 1. Building llama.cpp for Strix Halo

Build with Vulkan backend support to leverage the RDNA 3.5 iGPU:

```bash
# Install prerequisites (Ubuntu/Debian)
sudo apt-get install libvulkan-dev glslc libssl-dev

# Build with Vulkan support
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Verify the iGPU is detected:

```bash
./build/bin/llama-server --list-devices
# Should show the Radeon 890M (RDNA 3.5) as an available Vulkan device
```

The CPU backend will automatically use AVX-512 instructions available on Zen 5. No special build flags are needed for that; the build system auto-detects CPU capabilities.

---

## 2. Optimal Server Launch Configuration

```bash
./build/bin/llama-server \
    --model /path/to/llama-3-8b-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 5 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 16 \
    --threads-http 4 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --cache-prompt \
    --cache-reuse 256 \
    --mlock \
    --n-gpu-layers 99 \
    --metrics \
    --api-key-file /etc/llama-server/api-keys.txt \
    --timeout 600 \
    --no-webui
```

### Parameter Rationale

#### Slots and Context

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--parallel 5` | 5 slots | One slot per concurrent user. The server defaults to 4 when set to auto (`-1`), but since you have 4-5 users, setting 5 ensures no one queues unnecessarily. |
| `--ctx-size 8192` | 8192 tokens | Llama 3 supports 8192 natively. Total KV memory = 8192 * 5 slots. With q8_0 cache types, this is roughly 1.2-1.5 GB total for all slots. |
| `--kv-unified` | (enabled by default when parallel is auto; explicitly set if needed) | Uses a single shared KV buffer across all slots for better memory efficiency. Note: when `--parallel` is explicitly set (not auto/-1), you may want to add `--kv-unified` explicitly. |

#### Batch Processing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--batch-size 2048` | 2048 (default) | Logical batch size for prompt processing. The default is appropriate for the available memory bandwidth on LPDDR5X. |
| `--ubatch-size 512` | 512 (default) | Physical batch size. Balances between throughput and memory usage during prompt processing. |
| `--cont-batching` | enabled (default) | Continuous batching is critical for multi-user throughput -- it allows new prompts to begin processing while other slots are generating tokens. |

#### Threading

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--threads 8` | 8 | Generation threads. Zen 5 Strix Halo typically has 12 cores (12C/24T). Using 8 threads for generation leaves cores available for batch processing and HTTP handling. Over-subscribing threads hurts latency. |
| `--threads-batch 16` | 16 | Prompt processing is more parallelizable and benefits from more threads. Using 16 threads (of 24 hardware threads) for batch processing maximizes prompt eval throughput. |
| `--threads-http 4` | 4 | Dedicated HTTP threads for handling connections from your 4-5 users, separate from inference threads. |

#### GPU Offloading

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--n-gpu-layers 99` | All layers | Offloads all 32 layers of Llama 3 8B to the RDNA 3.5 iGPU via Vulkan. On Strix Halo, the unified memory architecture means this does not require data copies -- the iGPU accesses the same physical memory as the CPU. This significantly accelerates matrix multiplications. |
| `--flash-attn on` | Enabled | Flash Attention reduces memory usage and improves throughput for the attention computation. Explicitly enable rather than relying on auto-detection. |

#### KV Cache Optimization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--cache-type-k q8_0` | q8_0 | Quantizing KV cache keys to 8-bit reduces memory by 50% vs f16 default, with negligible quality loss. This is important for fitting 5 concurrent contexts. |
| `--cache-type-v q8_0` | q8_0 | Same for values. Together, q8_0 K+V roughly halves KV cache memory compared to f16. |
| `--cache-prompt` | enabled (default) | Prompt caching avoids reprocessing common prefixes (e.g., system prompts shared across users). |
| `--cache-reuse 256` | 256 tokens | Enables KV cache reuse via shifting when at least 256 tokens of prefix match. Particularly valuable when users share a common system prompt. |
| `--mlock` | enabled | Locks model weights in RAM, preventing the OS from swapping them to disk. Essential for production stability and consistent latency. |

#### Security and Monitoring

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--api-key-file` | File-based keys | Use a file containing one API key per line. Users authenticate with the `Authorization: Bearer <key>` header. More secure than passing keys on the command line. |
| `--metrics` | enabled | Exposes Prometheus-compatible metrics at `/metrics` for monitoring throughput, latency, and slot utilization. |
| `--no-webui` | disabled UI | For production, disable the web UI to reduce attack surface. Users should interact via the API only. |
| `--timeout 600` | 10 minutes | Read/write timeout for HTTP connections. 600 seconds is the default and appropriate for long-running generation requests. |

---

## 3. Memory Budget Estimate

For Llama 3 8B Q4_K_M with 5 slots at 8192 context:

| Component | Estimated Memory |
|-----------|-----------------|
| Model weights (Q4_K_M) | ~4.9 GB |
| KV cache (5 slots * 8192 ctx, q8_0) | ~1.2-1.5 GB |
| Working memory / overhead | ~0.5-1.0 GB |
| **Total** | **~6.6-7.4 GB** |

This fits comfortably within the 32+ GB of unified LPDDR5X on Strix Halo, leaving substantial headroom for the OS and other processes.

---

## 4. Systemd Service for Production

Create a systemd service file for automatic startup and restart:

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp Inference Server
After=network.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama-server
ExecStart=/opt/llama-server/llama-server \
    --model /opt/llama-server/models/llama-3-8b-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 5 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 16 \
    --threads-http 4 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --cache-prompt \
    --cache-reuse 256 \
    --mlock \
    --n-gpu-layers 99 \
    --metrics \
    --api-key-file /etc/llama-server/api-keys.txt \
    --timeout 600 \
    --no-webui
Restart=always
RestartSec=5
LimitMEMLOCK=infinity
LimitNOFILE=65536

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/opt/llama-server
ReadWritePaths=/var/log/llama-server

# Environment variables (alternative to CLI args)
# Environment=LLAMA_ARG_MODEL=/opt/llama-server/models/llama-3-8b-q4_k_m.gguf
# Environment=LLAMA_ARG_HOST=0.0.0.0
# Environment=LLAMA_ARG_N_PARALLEL=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server
sudo journalctl -u llama-server -f  # Monitor logs
```

Note: `LimitMEMLOCK=infinity` is required when using `--mlock` so the kernel allows the process to lock all its memory.

---

## 5. Reverse Proxy with nginx

For TLS termination and rate limiting in front of the server:

```nginx
upstream llama_backend {
    server 127.0.0.1:8080;
    keepalive 16;
}

server {
    listen 443 ssl;
    server_name llm.internal.example.com;

    ssl_certificate     /etc/ssl/certs/llm.crt;
    ssl_certificate_key /etc/ssl/private/llm.key;

    # Increase timeouts for long-running generation
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    proxy_connect_timeout 10s;

    # SSE (Server-Sent Events) support for streaming
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://llama_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Alternatively, llama-server has built-in SSL support via `--ssl-key-file` and `--ssl-cert-file` if you want to avoid running nginx.

---

## 6. Monitoring and Health Checks

### Health Endpoint

The server exposes `GET /health` which returns:

- `200 OK` when the server is ready and accepting requests
- `503 Service Unavailable` when the server is loading or all slots are busy (with `?fail_on_no_slot=1`)

Use this for load balancer health checks or systemd watchdog integration.

### Prometheus Metrics

With `--metrics` enabled, `GET /metrics` exposes:

- Request count and latency histograms
- Token generation rates (prompt eval and generation)
- Slot utilization (busy vs. idle)
- KV cache usage

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'llama-server'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
```

### Slot Monitoring

The `GET /slots` endpoint (enabled by default, controllable with `--slots`/`--no-slots`) shows the real-time status of each processing slot, including:

- Current state (idle, processing prompt, generating)
- Context usage per slot
- Tokens processed and generated

---

## 7. Performance Tuning Tips

### Strix Halo Specific

1. **Unified memory advantage**: With `--n-gpu-layers 99`, the Vulkan backend uses the iGPU for compute while the data stays in the same physical memory. No PCIe bottleneck exists. This is the primary advantage of Strix Halo for inference.

2. **LPDDR5X bandwidth**: The unified memory bandwidth (~120+ GB/s) is shared between CPU and iGPU. During token generation (which is memory-bandwidth-bound), avoid running other memory-intensive workloads on the system.

3. **Thread pinning (optional)**: If you observe inconsistent latency, consider using `--cpu-range` to pin inference threads to specific CCX cores, keeping threads within the same L3 cache domain:
   ```
   --cpu-range 0-7 --cpu-range-batch 0-15
   ```

4. **Priority boost**: For a dedicated inference server, elevate process priority:
   ```
   --prio 2
   ```
   This sets "high" priority for inference threads.

### General Throughput Tips

5. **If latency spikes occur under load**: Reduce `--parallel` to 4. Fewer slots means less KV cache contention and faster per-request generation at the cost of queuing the 5th user.

6. **For longer contexts**: If users regularly send long prompts (>4K tokens), consider reducing `--parallel` to 4 and increasing `--ctx-size` to 16384. This trades concurrency for context length.

7. **KV cache quantization tradeoff**: If you observe quality degradation with `q8_0` caches, switch to `f16` (the default). The memory cost is roughly doubled for KV cache, but on a 32+ GB system this is still manageable.

8. **Prompt caching**: The `--cache-prompt` and `--cache-reuse 256` settings are particularly valuable when your team uses a shared system prompt. The common prefix is processed once and reused across requests.

---

## 8. Quick Validation Checklist

After starting the server, verify it is working correctly:

```bash
# 1. Check health
curl http://localhost:8080/health

# 2. Test a completion (with API key)
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'

# 3. Check metrics
curl http://localhost:8080/metrics

# 4. Check slot status
curl http://localhost:8080/slots
```

---

## Summary

The key configuration choices for your Strix Halo + Llama 3 8B Q4_K_M + 4-5 users scenario:

| Setting | Value | Why |
|---------|-------|-----|
| `--parallel` | 5 | One slot per user |
| `--ctx-size` | 8192 | Full native context |
| `--n-gpu-layers` | 99 | Full iGPU offload via Vulkan (unified memory, no copy overhead) |
| `--flash-attn` | on | Memory efficiency + throughput |
| `--cache-type-k/v` | q8_0 | Halves KV cache memory with minimal quality impact |
| `--threads` | 8 | Generation threads (subset of available cores) |
| `--threads-batch` | 16 | Prompt processing threads (more parallelism) |
| `--mlock` | enabled | Prevents swapping, ensures stable latency |
| `--cont-batching` | enabled | Concurrent request processing |
| `--cache-prompt` | enabled | Reuses common prompt prefixes |
| `--metrics` | enabled | Production observability |

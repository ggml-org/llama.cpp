# Optimal llama-server Configuration for Strix Halo with Llama 3 8B Q4_K_M

## Hardware Context

AMD Strix Halo (Ryzen AI 300 series) features:
- **CPU**: Zen 5 cores with AVX-512 support, large L3 cache, CCX (Core Complex) topology
- **iGPU**: RDNA 3.5 (Radeon 890M) with shared unified memory
- **Memory**: LPDDR5X unified memory (typically 32-64 GB, shared between CPU and iGPU)
- **Key advantage**: Unified memory architecture eliminates CPU-GPU data copies

Llama 3 8B Q4_K_M model characteristics:
- ~4.9 GB model file size
- ~5-6 GB RAM for model weights in memory
- KV cache memory depends on context size and number of parallel slots

## Recommended Launch Configuration

```bash
./llama-server \
    --model /path/to/Meta-Llama-3-8B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 5 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 16 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --predict -1 \
    --slots
```

## Parameter Rationale

### Context and Slots

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--ctx-size 8192` | 8192 tokens | Llama 3 supports up to 8192 natively. This is the total context budget divided across all slots. Each of the 5 slots gets ~1638 tokens. If your users need longer contexts, increase this (e.g., 16384 or 32768) but monitor memory usage. |
| `--parallel 5` | 5 slots | One slot per concurrent user. Match this to your expected peak concurrency (4-5 users). The server defaults to 4 when set to auto. Setting to 5 covers your upper bound. |
| `--cont-batching` | enabled | Continuous batching (enabled by default) allows the server to process prompt tokens for one request while generating tokens for another, dramatically improving throughput for concurrent users. |

### Threading

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--threads 8` | 8 threads | For token generation. Zen 5 cores are powerful individually; using too many threads adds synchronization overhead. For an 8B model, 8 threads typically hits the sweet spot on Zen 5. |
| `--threads-batch 16` | 16 threads | For prompt processing (batch/prefill). This is more parallelizable, so using more threads improves prompt ingestion speed. Use up to the number of physical cores available. |

Strix Halo SKUs vary (up to 16 cores). Adjust these based on your specific SKU:
- **12-core SKU**: Use `--threads 6 --threads-batch 12`
- **16-core SKU**: Use `--threads 8 --threads-batch 16`

### Batching

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--batch-size 2048` | 2048 tokens | Logical batch size. This is the maximum number of tokens processed in a single batch. The default (2048) works well for 5 concurrent users. |
| `--ubatch-size 512` | 512 tokens | Physical batch size. Controls actual compute granularity. The default (512) is well-tuned for most hardware. |

### Memory Optimization

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `--flash-attn on` | enabled | Flash Attention reduces KV cache memory traffic and improves throughput, especially important with multiple concurrent slots. Zen 5 and RDNA 3.5 both benefit. |
| `--cache-type-k q8_0` | q8_0 quantized | Quantizing the KV cache K values to q8_0 roughly halves KV cache memory vs f16 with negligible quality loss. Critical for fitting 5 slots in memory. |
| `--cache-type-v q8_0` | q8_0 quantized | Same for V values. Together these reduce KV cache from ~1 GB to ~500 MB for 5 slots at 8K context. |

### Memory Budget Estimate (8192 ctx, 5 parallel slots)

| Component | Approximate Memory |
|-----------|-------------------|
| Model weights (Q4_K_M) | ~5.0 GB |
| KV cache (q8_0 K+V, 5 slots) | ~0.5 GB |
| Working memory / overhead | ~1.0 GB |
| **Total** | **~6.5 GB** |

This leaves significant headroom on a 32 GB or 64 GB Strix Halo system.

## Optional: Vulkan iGPU Offload

Strix Halo's RDNA 3.5 iGPU shares unified memory with the CPU. You can offload some layers to the iGPU for additional throughput:

```bash
# Build with Vulkan support first:
# cmake -B build -DGGML_VULKAN=ON && cmake --build build

./llama-server \
    --model /path/to/Meta-Llama-3-8B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 5 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 16 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --n-gpu-layers 33 \
    --slots
```

Setting `--n-gpu-layers 33` offloads all 32 transformer layers plus the output layer to the Vulkan iGPU. Because Strix Halo uses unified memory, this avoids the usual PCIe transfer bottleneck. Benchmark both CPU-only and Vulkan configurations to determine which gives better throughput for your specific SKU -- the iGPU can be faster for batch processing while the Zen 5 CPU excels at parallel slot handling.

## NUMA Considerations

If your Strix Halo SKU has multiple CCX clusters (some higher-end models), consider:

```bash
--numa distribute
```

This spreads work across NUMA nodes. However, most Strix Halo mobile/workstation chips are single-socket with unified memory, so this may not be necessary. Test with and without.

## Production Hardening

### Systemd Service File

Create `/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=llama.cpp Server
After=network.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    --model /opt/models/Meta-Llama-3-8B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 8192 \
    --parallel 5 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 16 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --slots
Restart=always
RestartSec=5
LimitNOFILE=65536
Environment="LLAMA_CACHE=/opt/llama.cpp/cache"

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl status llama-server
```

### Reverse Proxy (Nginx)

For TLS termination and basic access control:

```nginx
upstream llama_backend {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl;
    server_name llama.internal.example.com;

    ssl_certificate     /etc/ssl/certs/llama.crt;
    ssl_certificate_key /etc/ssl/private/llama.key;

    # Basic auth for team access
    auth_basic "LLM Service";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Increase timeouts for long generations
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    # SSE support for streaming
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://llama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

### Monitoring

The `--slots` endpoint (enabled by default) exposes slot utilization at `GET /slots`. Use it to monitor:

- Active vs idle slots
- Tokens processed per slot
- Prompt cache hit rates

You can also use the `/health` endpoint for load balancer health checks and `/metrics` for Prometheus-compatible metrics.

### API Key Protection

llama-server supports API key authentication:

```bash
--api-key "your-secret-key-here"
```

Clients must then include `Authorization: Bearer your-secret-key-here` in requests. For multiple keys, use `--api-key-file /path/to/keys.txt` with one key per line.

## Performance Tuning Checklist

1. **Benchmark first**: Run `llama-bench` with your model to establish baseline tok/s
2. **Thread count sweep**: Try `--threads` values from 4 to your physical core count; measure tok/s at each
3. **Test Vulkan vs CPU**: Compare throughput with and without `--n-gpu-layers 33`
4. **Monitor memory**: Watch `htop` / `free -h` under full 5-user concurrent load
5. **Stress test**: Use the server's built-in bench endpoint or `tools/server/bench` to simulate concurrent users before going live
6. **Context scaling**: If users need longer conversations, increase `--ctx-size` but re-verify memory fits

## Scaling Considerations

If 5 concurrent users is not enough in the future:
- **More slots**: Increase `--parallel` to 8 (increase `--ctx-size` proportionally to maintain per-slot context)
- **Quantized KV cache**: Drop to `q4_0` for K and V cache types to further reduce memory per slot
- **Smaller context**: If your use case allows, reduce per-slot context by lowering `--ctx-size`
- **Model swap**: Consider Q4_K_S quantization for slightly less memory, or Q5_K_M if you have headroom and want better quality

# Data-Local Compute: CPU on Unified Cache Host Pinned Memory

## Can SYCL CPU compute directly on unified cache host pinned memory?

**Yes.** SYCL absolutely supports CPU devices — the oneAPI CPU runtime can execute SYCL kernels on the CPU via oneTBB + oneDNN. And since the unified cache's host pinned memory (`sycl::malloc_host()`) is just regular host-accessible memory, a CPU device can compute on it with **zero data movement**.

## Two implementation paths

### Path 1: SYCL CPU Device (compute follows data)

The unified cache already knows which weights are in VRAM vs pinned host. During graph compute:
- Weight in VRAM -> dispatch to GPU device (current behavior)
- Weight in pinned host -> dispatch to SYCL CPU device (new)
- Zero copy either way — compute goes to where data lives

This works because `sycl::malloc_host()` returns USM that's directly accessible to the CPU device. oneDNN matmul on CPU is well-optimized (it's Intel's own library).

### Path 2: ggml CPU backend on same memory

Even simpler — `sycl::malloc_host()` returns a normal host pointer. The ggml CPU backend's hand-tuned AVX-512 kernels can operate on it directly. No SYCL CPU device needed at all. Just expose the pointer to the existing CPU backend.

## The real challenge: activation transfers

The catch isn't the weight computation — it's the **activations between layers**:

```
Layer 5 (GPU, weights in VRAM)  -> activations on GPU
Layer 6 (CPU, weights on host)  -> need activations copied GPU->CPU
Layer 7 (GPU, weights in VRAM)  -> need activations copied CPU->GPU
```

Every GPU<->CPU boundary requires an activation transfer. But:
- Activations are tiny compared to weights (a few MB vs hundreds of MB per layer)
- With pinned host memory, GPU<->CPU transfers are DMA-accelerated
- This is still much cheaper than streaming entire weight matrices

## Why this is better than both fit_params AND weight streaming

| Approach | Weight movement | Compute engine | Activation movement |
|----------|----------------|---------------|-------------------|
| fit_params (current) | None (layers pre-assigned) | CPU backend for offloaded layers | Scheduler handles |
| Weight streaming | 117 MB/layer GPU<-Host per token | GPU only | None |
| **Data-local compute** | **None** | **CPU for host layers, GPU for VRAM layers** | **Small activation transfers** |

Weight streaming moves ~117 MB per layer per token. Data-local compute moves ~8 MB of activations at layer boundaries. That's a **14x reduction** in data movement.

## Implementation sketch

The cleanest path would be:

1. **Unified cache already tracks tier per tensor** (DEVICE, PINNED_HOST, MMAP)
2. **At graph_compute time**: check each mul_mat's weight tier
3. **If PINNED_HOST**: submit to CPU queue instead of GPU queue (or let ggml CPU backend handle it)
4. **Insert async copy nodes** at tier boundaries for activations
5. **No fit_params** — the cache's tiering IS the offload decision

The ggml scheduler already supports multi-backend dispatch (it's how fit_params works today — assigning layers to GPU vs CPU backend). The difference is that instead of fit_params deciding placement at load time, the **unified cache's runtime tiering** decides it dynamically.

## Open question

Which path is more practical?
- **SYCL CPU device**: Stays within the SYCL backend, unified kernel dispatch, but needs multi-queue management
- **ggml CPU backend**: Uses existing optimized AVX-512 kernels, scheduler already handles multi-backend, but needs pointer sharing across backends

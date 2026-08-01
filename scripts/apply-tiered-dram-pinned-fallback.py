#!/usr/bin/env python3
from pathlib import Path

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text(encoding="utf-8")

if (
    "bool host_ptr_owned = false;" in text
    and "using a mapped pinned copy" in text
    and "cudaFreeHost(state->host_ptr)" in text
):
    print(f"already patched {path}")
    raise SystemExit(0)

old = """    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    size_t size = 0;
    size_t alloc_size = 0;
"""
new = """    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    bool host_ptr_owned = false;
    size_t size = 0;
    size_t alloc_size = 0;
"""
if old not in text:
    raise SystemExit("tensor_state layout did not match expected source")
text = text.replace(old, new, 1)

old = """        if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM && state->device_ptr) {
            (void) cudaFree(state->device_ptr);
            state->device_ptr = nullptr;
        }
        if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
            free_sparse_tensor(state);
        }
"""
new = """        if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM && state->device_ptr) {
            (void) cudaFree(state->device_ptr);
            state->device_ptr = nullptr;
        }
        if (state->host_ptr_owned && state->host_ptr) {
            (void) cudaFreeHost(state->host_ptr);
            state->host_ptr = nullptr;
            state->host_ptr_owned = false;
        }
        if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
            free_sparse_tensor(state);
        }
"""
if old not in text:
    raise SystemExit("tiered_buffer_free block did not match expected source")
text = text.replace(old, new, 1)

old = """        case GGML_CUDA_TIERED_MEMORY_DRAM: {
            cudaDeviceProp properties = {};
            TIERED_CUDA_CHECK(cudaGetDeviceProperties(&properties, ctx->device));
            if (!properties.canMapHostMemory) {
                throw std::runtime_error("CUDA device cannot map host memory");
            }
            ensure_registered(ctx, state->host_ptr, state->size);
            TIERED_CUDA_CHECK(cudaHostGetDevicePointer(
                    &state->device_ptr, state->host_ptr, 0));
            tensor->data = state->device_ptr;
        } break;
"""
new = r"""        case GGML_CUDA_TIERED_MEMORY_DRAM: {
            cudaDeviceProp properties = {};
            TIERED_CUDA_CHECK(cudaGetDeviceProperties(&properties, ctx->device));
            if (!properties.canMapHostMemory) {
                throw std::runtime_error("CUDA device cannot map host memory");
            }

            // GGUF weights normally come from a read-only file mmap. Some
            // pre-Ampere GPUs/drivers reject cudaHostRegisterMapped for that
            // range. Fall back to an owned mapped pinned copy.
            try {
                ensure_registered(ctx, state->host_ptr, state->size);
            } catch (const std::exception & error) {
                (void) cudaGetLastError();
                GGML_LOG_WARN(
                        "tiered-memory: direct DRAM registration failed for %s (%s); "
                        "using a mapped pinned copy\n",
                        tensor->name, error.what());

                void * pinned = nullptr;
                TIERED_CUDA_CHECK(cudaHostAlloc(
                        &pinned, state->size,
                        cudaHostAllocPortable | cudaHostAllocMapped));
                std::memcpy(pinned, data, state->size);
                state->host_ptr = pinned;
                state->host_ptr_owned = true;
            }

            TIERED_CUDA_CHECK(cudaHostGetDevicePointer(
                    &state->device_ptr, state->host_ptr, 0));
            tensor->data = state->device_ptr;
        } break;
"""
if old not in text:
    raise SystemExit("DRAM tier block did not match expected source")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print(f"patched {path}")

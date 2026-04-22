#pragma once

// AllReduce provider for multi-GPU tensor parallelism.
//
// The meta backend splits each transformer layer's PARTIAL-axis subgraph across
// N GPUs and requires an AllReduce after each segment to sum the partial results.
// This enum selects which implementation performs that reduction.
//
// The active provider is chosen once at communicator init time by
// ggml_cuda_select_allreduce_provider() and stored in
// ggml_backend_cuda_comm_context::provider.
enum ggml_cuda_allreduce_provider {
    // NVIDIA/AMD Collective Communications Library (NCCL/RCCL).
    // Optimal on NVLink/NVSwitch topologies; auto-selects the best transport.
    // Requires GGML_USE_NCCL at compile time.
    GGML_CUDA_ALLREDUCE_NCCL   = 0,

    // Internal host/CUDA staged reduction built into llama.cpp.
    // Works on any interconnect (PCIe, NVLink) without an external library.
    // Can outperform NCCL on PCIe-only systems for latency-sensitive tensor sizes.
    GGML_CUDA_ALLREDUCE_INTERNAL = 1,
};

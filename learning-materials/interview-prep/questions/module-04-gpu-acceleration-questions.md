# Module 4: GPU Acceleration - Interview Questions

**Purpose**: Interview preparation for GPU acceleration and parallel computing
**Target Level**: Senior to Staff Engineers
**Module Coverage**: Module 4 - CUDA, Metal, ROCm, Performance Optimization
**Question Count**: 25 (distributed across 4 categories)
**Last Updated**: 2025-11-18
**Created By**: Agent 8 (Integration Coordinator)

---

## Table of Contents

1. [Conceptual Questions](#conceptual-questions) (7 questions)
2. [Technical Questions](#technical-questions) (7 questions)
3. [System Design Questions](#system-design-questions) (6 questions)
4. [Debugging Questions](#debugging-questions) (5 questions)

---

## Conceptual Questions

### Question 1: CUDA Fundamentals for LLM Inference

**Category**: Conceptual
**Difficulty**: Senior (L5/L6)
**Companies**: NVIDIA, OpenAI, Meta AI
**Time Allotted**: 20 minutes

**Question**: Explain how CUDA accelerates LLM inference. What are threads, blocks, and grids? How does memory hierarchy (global, shared, registers) impact performance?

**Key Points to Cover**:
- Thread hierarchy and parallelism model
- Memory bandwidth vs compute
- Kernel fusion and optimization strategies
- Tensor Core utilization for matrix multiplication

**Rubric Score**: 16/28 (Senior), 22/28 (Staff)

---

### Question 2: Metal vs CUDA Architecture

**Category**: Conceptual
**Difficulty**: Senior (L5/L6)
**Companies**: Apple, Meta AI
**Time Allotted**: 15 minutes

**Question**: Compare Metal and CUDA for LLM inference. What are the architectural differences? When would you choose each?

**Key Points**: Unified memory, command buffers, threadgroups vs blocks, performance characteristics

---

### Question 3: GPU Memory Management

**Category**: Conceptual
**Difficulty**: Mid-Senior (L4/L5)
**Time Allotted**: 15 minutes

**Question**: Explain GPU memory management for large models. How do you handle models larger than VRAM? What is layer offloading?

**Key Points**: VRAM limitations, CPU-GPU memory transfers, layer splitting, unified memory on Apple Silicon

---

### Question 4: Tensor Cores and Mixed Precision

**Category**: Conceptual
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 20 minutes

**Question**: What are Tensor Cores? How do they accelerate matrix multiplication? What are the requirements for using them effectively?

**Key Points**: WMMA API, matrix dimensions, INT8/FP16 support, throughput improvements

---

### Question 5: Batch Processing on GPU

**Category**: Conceptual
**Difficulty**: Mid-Senior (L4/L5)
**Time Allotted**: 15 minutes

**Question**: Why is batching important for GPU utilization? How does batch size affect latency and throughput?

**Key Points**: Parallelism exploitation, memory bandwidth amortization, optimal batch size selection

---

### Question 6: Flash Attention

**Category**: Conceptual
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 20 minutes

**Question**: Explain Flash Attention. Why is it faster than standard attention? How does it reduce memory usage?

**Key Points**: Tiling, memory hierarchy exploitation, IO-awareness, recomputation strategy

---

### Question 7: Multi-GPU Strategies

**Category**: Conceptual
**Difficulty**: Staff (L6/L7)
**Time Allotted**: 25 minutes

**Question**: Design a multi-GPU inference strategy. Compare tensor parallelism, pipeline parallelism, and data parallelism.

**Key Points**: Communication overhead, load balancing, NVLink vs PCIe, scaling efficiency

---

## Technical Questions

### Question 8: Implementing CUDA Kernel for Matrix Multiplication

**Category**: Technical
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 40 minutes

**Question**: Write a CUDA kernel for matrix multiplication with tiling optimization. Explain shared memory usage.

**Implementation Required**: Working CUDA code with tiling, shared memory, thread synchronization

---

### Question 9: GPU Profiling and Optimization

**Category**: Technical
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 20 minutes

**Question**: How would you profile a slow CUDA kernel? What tools and metrics would you use? Walk through an optimization process.

**Key Tools**: nvprof, Nsight Compute, occupancy calculator, roofline analysis

---

### Question 10: Memory Coalescing

**Category**: Technical
**Difficulty**: Mid-Senior (L4/L5)
**Time Allotted**: 15 minutes

**Question**: What is memory coalescing? How do you ensure coalesced memory access? What's the performance impact?

**Key Points**: 128-byte transactions, alignment, strided access patterns, bandwidth utilization

---

### Question 11: Implementing Layer Offloading

**Category**: Technical
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 30 minutes

**Question**: Implement a layer offloading strategy for a model that doesn't fit in VRAM. Handle CPU-GPU transfers efficiently.

**Implementation**: Code showing layer splitting, async transfers, pipeline overlap

---

### Question 12: CUDA Streams and Concurrency

**Category**: Technical
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 25 minutes

**Question**: Explain CUDA streams. How would you use multiple streams to overlap computation and memory transfers?

**Key Points**: Async operations, stream synchronization, kernel concurrency, pinned memory

---

### Question 13: Fused Kernels for Attention

**Category**: Technical
**Difficulty**: Staff (L6/L7)
**Time Allotted**: 45 minutes

**Question**: Design and implement a fused kernel for attention computation (QKV projection + softmax + output).

**Implementation**: CUDA code with kernel fusion, shared memory optimization, warp-level primitives

---

### Question 14: GPU-Aware MPI for Multi-Node

**Category**: Technical
**Difficulty**: Staff (L6/L7)
**Time Allotted**: 30 minutes

**Question**: Implement multi-node GPU communication using GPU-Direct RDMA. How does it improve over CPU-mediated transfers?

**Key Points**: NCCL, InfiniBand, GPUDirect, bandwidth analysis

---

## System Design Questions

### Question 15: High-Throughput GPU Serving Architecture

**Category**: System Design
**Difficulty**: Staff (L6/L7)
**Time Allotted**: 60 minutes

**Question**: Design a GPU serving system for 10,000 req/sec. Consider batching, multi-GPU, scheduling, and failure handling.

**Key Components**: Dynamic batching, GPU pooling, request scheduling, monitoring

---

### Question 16: GPU Resource Allocation Strategy

**Category**: System Design
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 45 minutes

**Question**: Design a multi-tenant GPU allocation system. How do you ensure fairness and prevent resource starvation?

**Key Points**: Time slicing, MPS, GPU virtualization, QoS guarantees

---

### Question 17: Cost-Optimized GPU Selection

**Category**: System Design
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 30 minutes

**Question**: You need to serve a 70B model. Compare A100, H100, V100, and consumer GPUs. What would you choose and why?

**Analysis**: Cost per token, throughput, memory capacity, TCO calculation

---

### Question 18: GPU Monitoring and Auto-Scaling

**Category**: System Design
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 40 minutes

**Question**: Design a GPU monitoring and auto-scaling system for dynamic workloads.

**Key Components**: Metrics collection, scaling policies, warmup handling, cost optimization

---

### Question 19: Mixed Hardware Deployment

**Category**: System Design
**Difficulty**: Staff (L6/L7)
**Time Allotted**: 45 minutes

**Question**: Design a system supporting NVIDIA, AMD, and Apple GPUs. How do you abstract hardware differences?

**Architecture**: Backend abstraction, capability detection, routing strategy

---

### Question 20: GPU Memory Optimization for Large Models

**Category**: System Design
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 35 minutes

**Question**: Design memory optimization strategies for fitting a 180B model on 8×80GB GPUs.

**Techniques**: Tensor parallelism, quantization, activation checkpointing, memory planning

---

## Debugging Questions

### Question 21: CUDA Out of Memory Error

**Category**: Debugging
**Difficulty**: Mid-Senior (L4/L5)
**Time Allotted**: 20 minutes

**Question**: Your model runs fine with batch=1 but OOM with batch=2. How do you debug and fix this?

**Process**: Memory profiling, leak detection, fragmentation analysis, solutions

---

### Question 22: Performance Regression After Update

**Category**: Debugging
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 25 minutes

**Question**: After updating CUDA drivers, inference is 50% slower. How do you investigate?

**Tools**: Profiling, kernel timing, driver version comparison, rollback strategy

---

### Question 23: Numerical Instability on GPU

**Category**: Debugging
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 25 minutes

**Question**: Model outputs differ between CPU and GPU. How do you debug this?

**Investigation**: Precision differences, reduction order, atomic operations, reproducibility

---

### Question 24: Multi-GPU Communication Bottleneck

**Category**: Debugging
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 30 minutes

**Question**: Multi-GPU setup scaling is poor (2×GPU = 1.3×speed). Find and fix the bottleneck.

**Analysis**: Communication profiling, topology, synchronization overhead, optimization

---

### Question 25: GPU Hang or Timeout

**Category**: Debugging
**Difficulty**: Senior (L5/L6)
**Time Allotted**: 25 minutes

**Question**: GPU inference occasionally hangs. How do you debug and prevent this?

**Tools**: cuda-gdb, timeout detection, deadlock analysis, recovery strategies

---

## Summary

**Module 4 Coverage**:
- CUDA programming fundamentals
- Metal and ROCm backends
- GPU memory management
- Kernel optimization techniques
- Multi-GPU strategies
- Performance profiling
- Production GPU serving
- Hardware selection and cost optimization

**Difficulty Distribution**:
- Mid-Senior: 4 questions
- Senior: 17 questions
- Staff: 4 questions

**Interview Company Alignment**:
- ✅ NVIDIA (all levels)
- ✅ OpenAI L5-L7
- ✅ Anthropic L5-L7
- ✅ Meta AI E5-E7
- ✅ Apple (Metal-specific)
- ✅ GPU-focused startups

---

**Maintained by**: Agent 8 (Integration Coordinator)
**Last Updated**: 2025-11-18

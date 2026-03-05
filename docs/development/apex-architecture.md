# APEX Runtime Scheduling — Architecture

## System Overview

```mermaid
graph TB
    subgraph "Model Load Time"
        ML[llama_model_load] --> LT[load_tensors]
        LT --> WA[Weight Allocation<br/>tensors → GPU/CPU buffers]
        LT --> AF[Auto-Fit<br/>llama_params_fit_impl]
        AF --> LF[LAYER_FRACTION_FFN<br/>UMA bandwidth-aware]
    end

    subgraph "Context Creation"
        CC[llama_init_from_model] --> BE[Backend Init<br/>GPU + CPU backends]
        BE --> SC[Scheduler Creation<br/>ggml_backend_sched_new]
        SC --> EC[Eval Callback Setup<br/>uma_profiler_cb_eval]
    end

    subgraph "Runtime Inference"
        DEC[llama_decode] --> UB[process_ubatch]
        UB --> GB[Graph Build<br/>model.build_graph]
        GB --> GCB[graph_get_cb<br/>APEX routing]
        GCB --> |"attention ops"| CPU[CPU Backend]
        GCB --> |"FFN ops"| GPU[GPU Backend]
        UB --> GA[Graph Allocate<br/>sched_alloc_graph]
        GA --> GS[Graph Split<br/>3-pass backend assign]
        GS --> GC[Graph Compute<br/>compute_splits]
        GC --> |"op_overlap=true"| AO[Async Overlap<br/>event-based sync]
        GC --> |"op_overlap=false"| SEQ[Sequential<br/>split-by-split]
    end

    subgraph "APEX Decision Pipeline"
        EC --> |"N iterations"| PR[Profiler Data<br/>per-op timing]
        PR --> AL[Analyze Layers<br/>attn_us, ffn_us, AI]
        AL --> IE[APEX Inequality<br/>ratio < threshold?]
        IE --> |"yes"| POL[Create Policy<br/>apex_offload_policy]
        IE --> |"no"| GPU_ONLY[GPU-Only Mode]
        POL --> GCB
    end
```

## APEX Decision Flow

```mermaid
flowchart TD
    START[Profiling Complete] --> CALC[Calculate Per-Layer Averages<br/>avg_ffn_us, avg_attn_us]
    CALC --> EST[Estimate CPU Attention Time<br/>T_catt = T_gatt × 10]
    EST --> INEQ{APEX Inequality<br/>ratio < threshold?}

    INEQ --> |"ratio = T_gatt / T_catt<br/>threshold = 2×T_ffn/T_attn + 3 + T_attn/T_ffn"| CHECK{Hybrid Throughput<br/>> GPU-Only?}

    CHECK --> |"Yes"| OVERLAP[ASYNC_OVERLAP<br/>CPU attention ∥ GPU FFN]
    CHECK --> |"No"| GPUONLY[GPU_ONLY<br/>All ops on GPU]
    INEQ --> |"ratio ≥ threshold"| GPUONLY

    OVERLAP --> POLICY[Create Offload Policy<br/>layers 0 to n_layer-1]
    POLICY --> GCBHOOK[Hook into graph_get_cb<br/>Route attn → CPU]
    GCBHOOK --> SCHED[Scheduler splits graph<br/>GPU splits + CPU splits]
    SCHED --> ASYNC[Async split overlap<br/>event-based sync]
```

## Split Overlap Timing

```mermaid
gantt
    title Sequential vs Async Overlap Execution
    dateFormat X
    axisFormat %s

    section Sequential
    GPU FFN layer 0       :a1, 0, 10
    CPU Attn layer 0      :a2, 10, 18
    GPU FFN layer 1       :a3, 18, 28
    CPU Attn layer 1      :a4, 28, 36

    section Async Overlap
    GPU FFN layer 0       :b1, 0, 10
    CPU Attn layer 0      :b2, 5, 13
    GPU FFN layer 1       :b3, 10, 20
    CPU Attn layer 1      :b4, 15, 23
```

## Data Flow Through Scheduler

```mermaid
graph LR
    subgraph "Graph Build"
        T1[FFN tensors<br/>GPU backend] --> S1[Split 0<br/>GPU]
        T2[Attn tensors<br/>CPU backend] --> S2[Split 1<br/>CPU]
        T3[FFN tensors<br/>GPU backend] --> S3[Split 2<br/>GPU]
    end

    subgraph "Execution"
        S1 --> |"compute async"| E1[GPU Kernel]
        E1 --> |"record event"| EV1[Event 0]
        EV1 --> |"event wait"| S2
        S2 --> |"compute async"| E2[CPU Kernel]
        E2 --> |"record event"| EV2[Event 1]

        E1 --> |"op_overlap:<br/>no wait"| S3
        EV2 --> |"event wait<br/>when needed"| S3
        S3 --> |"compute async"| E3[GPU Kernel]
    end
```

## Component Dependency

```mermaid
graph BT
    UP[uma-profiler.h/cpp<br/>Per-op timing + roofline] --> AS[apex-scheduler.h/cpp<br/>Inequality gate + policy]
    AS --> LC[llama-context.h/cpp<br/>graph_get_cb + decode hook]
    LC --> BS[ggml-backend.cpp<br/>op_overlap split execution]

    UP --> |"profiling data"| AS
    AS --> |"offload policy"| LC
    LC --> |"set_tensor_backend"| BS
    BS --> |"event sync"| GPU_BE[GPU Backend]
    BS --> |"event sync"| CPU_BE[CPU Backend]
```

## UMA Memory Architecture (Strix Halo)

```mermaid
graph TB
    subgraph "Shared LPDDR5X Memory"
        W[Model Weights] --- KC[KV Cache]
        KC --- CB[Compute Buffers]
    end

    subgraph "Zen 5 CPU"
        CT[CPU Threads] --> |"~100 GB/s<br/>bandwidth"| W
        CT --> |"attention ops<br/>compute-bound"| KC
    end

    subgraph "RDNA 3.5 iGPU"
        GCU[GPU CUs] --> |"~200 GB/s<br/>effective bandwidth"| W
        GCU --> |"FFN matmuls<br/>bandwidth-bound"| CB
    end

    W --> |"zero-copy<br/>unified memory"| CT
    W --> |"zero-copy<br/>unified memory"| GCU
```

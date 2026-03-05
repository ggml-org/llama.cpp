# APEX Runtime Scheduling — User Stories

Reference: APEX (arXiv:2506.03296) — Asynchronous Parallel CPU-GPU Execution for Online LLM Inference

## Context

Our branch already implements **static** APEX-inspired optimizations (load-time layer splitting
with `LAYER_FRACTION_FFN`, UMA profiler with roofline classification). These user stories cover
making the APEX theoretical framework **practical at runtime** — dynamic scheduling decisions
during inference rather than only at model load time.

---

## US-1: Runtime Profiling-Informed Layer Reassignment

**As a** user running LLM inference on a UMA system (e.g., AMD Strix Halo),
**I want** the system to measure per-layer execution times during the first N iterations
and automatically adjust which layers run on GPU vs CPU,
**so that** the layer assignment reflects actual hardware performance rather than static estimates.

### Acceptance Criteria
- After `N` warmup iterations (configurable, default 3), the UMA profiler classifies each
  layer's attention and FFN ops as bandwidth-bound or compute-bound
- The profiler data feeds back into a `reassign_layers()` function that can update
  `tensor_buft_overrides` for subsequent inference
- Reassignment only triggers if the profiled data shows >10% potential throughput improvement
- A log message reports any layer reassignment decisions
- No regression in first-token latency (warmup overhead < 5%)

---

## US-2: APEX Critical Inequality Gate

**As a** developer or power user,
**I want** the system to evaluate APEX's critical throughput inequality at runtime
to decide whether CPU offload is profitable for the current workload,
**so that** CPU offload only activates when it actually improves throughput.

### Acceptance Criteria
- Implements the APEX inequality: `N_G/N_C < 2*(T_glinear/T_gatt) + 3 + (T_gatt/T_glinear)`
- `T_glinear` and `T_gatt` are measured from profiler data (not estimated)
- `N_G` and `N_C` are derived from measured throughput (tokens/sec on each device)
- The gate is evaluated after profiling completes and before any layer reassignment
- If the inequality does not hold, CPU offload is skipped (GPU-only execution)
- Decision is logged: "APEX gate: CPU offload profitable=yes/no (ratio=X, threshold=Y)"

---

## US-3: Asynchronous CPU-GPU Attention Overlap

**As a** user running decode-heavy workloads on memory-constrained hardware,
**I want** CPU attention computation to overlap with GPU FFN computation,
**so that** the CPU does useful work while the GPU processes the next layer's linear ops.

### Acceptance Criteria
- For layers where attention is offloaded to CPU (`LAYER_FRACTION_FFN`), the CPU attention
  for layer `i` executes concurrently with GPU FFN for layer `i+1`
- Uses the existing `ggml_backend_sched` pipeline parallelism infrastructure (`n_copies`)
- Synchronization occurs just before the GPU needs CPU attention results (lazy sync)
- If CPU attention is not ready when GPU needs it, GPU does not stall — it continues
  and checks again next iteration
- Throughput improvement of >10% on decode-heavy workloads vs sequential execution
- No correctness issues (results match sequential execution bit-for-bit)

---

## US-4: Dynamic Mixed-Workload Scheduling

**As a** user running the llama.cpp server with concurrent prefill and decode requests,
**I want** the scheduler to dynamically choose between GPU-only, asymmetric pipelining,
and asynchronous overlap strategies based on current workload mix,
**so that** throughput is maximized regardless of whether the workload is prefill-heavy,
decode-heavy, or mixed.

### Acceptance Criteria
- Implements APEX's three-strategy selection: GPU-only, asymmetric pipelining, async overlap
- Strategy selection uses measured `T_glinear`, `T_gatt`, and `T_overlap` from profiler
- For mixed workloads: `T_overlap = T_glinear_pref + T_gatt_pref + T_glinear + T_gatt`
- Strategy can change between batches (not locked at startup)
- Server mode (`llama-server`) benefits from dynamic strategy switching
- Logging shows strategy transitions: "APEX scheduler: switching to {strategy} (throughput est: X tok/s)"

---

## US-5: Per-Op Profiler Accuracy and Low Overhead

**As a** developer,
**I want** the UMA profiler to produce accurate per-op timings with minimal overhead,
**so that** scheduling decisions based on profiler data are reliable.

### Acceptance Criteria
- Profiler uses `ggml_time_us()` with proper synchronization (GPU ops must complete before timing)
- Overhead during profiling iterations < 5% of total inference time
- After profiling completes, callback overhead is zero (profiler disables itself)
- Profiler handles batch size changes gracefully (re-profiles if batch size changes >2x)
- Arithmetic intensity calculation is validated against known op characteristics
  (e.g., MUL_MAT during decode with batch=1 should be bandwidth-bound)

---

## Priority Order

1. **US-5** — Profiler accuracy (foundation for all other stories)
2. **US-2** — Critical inequality gate (determines if offload helps at all)
3. **US-1** — Runtime layer reassignment (acts on profiler + gate decisions)
4. **US-3** — Async overlap (the main throughput win from APEX)
5. **US-4** — Dynamic mixed-workload scheduling (server-mode optimization)

## Non-Goals (for now)

- CPU attention task pool with layer-wise prioritization (APEX future work)
- Multi-GPU scheduling (we target single iGPU UMA systems)
- Prefill-only optimization (APEX shows minimal gains for prefill-dominated workloads)

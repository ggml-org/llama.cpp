# Cross-Backend Profiler

llama.cpp includes a built-in cross-backend profiler that captures per-operation timing, data transfer costs, and tensor shapes across all compute backends. It works with any application built on the ggml scheduler — no source changes needed.

## Supported Backends

| Backend | Status | Timing method |
|---------|--------|---------------|
| CPU     | Supported | Wall-clock (`CLOCK_MONOTONIC_RAW`) |
| CUDA    | Supported | `cudaEvent` GPU timestamps |
| Vulkan  | Supported | GPU timestamp queries |
| BLAS    | Supported | Wall-clock |
| Metal   | Not yet supported | — |
| OpenCL  | Not yet supported | — |

The scheduler also profiles **data copies** (H2D, D2H, D2D) between backends regardless of which backends have native profiler support.

## Enabling the Profiler

There are two independent ways to enable profiling. They can be used separately or together.

### CLI flags (`--profile`, `--profile-output`)

Available in `llama-cli`, `llama-completion`, `llama-server`, and `debug`:

```bash
# Print summary to stdout
llama-completion -m model.gguf --profile -p "Hello world"

# Export to JSON
llama-completion -m model.gguf --profile --profile-output profile.json -p "Hello world"

# Export to plain text
llama-completion -m model.gguf --profile --profile-output profile.txt -p "Hello world"
```

The output format is chosen by file extension: `.json` for JSON, `.txt` for plain text. Any other extension defaults to JSON.

### Environment variable (`GGML_PROFILE`)

The `GGML_PROFILE` environment variable enables profiling at the ggml scheduler level. This works with **any** application that uses the scheduler — including third-party tools like `sd.cpp` — without CLI flag support.

```bash
# Print summary to stdout
GGML_PROFILE=1 llama-completion -m model.gguf -p "Hello world"

# Export JSON
GGML_PROFILE=profile.json llama-completion -m model.gguf -p "Hello world"

# Export plain text
GGML_PROFILE=profile.txt llama-completion -m model.gguf -p "Hello world"

# Works with any ggml-based application
GGML_PROFILE=1 sd -m model.gguf -p "a cat"
```

| Value | Behavior |
|-------|----------|
| `1`, `stdout`, or empty | Print summary to stdout |
| `path.json` | Export JSON to file |
| `path.txt` | Export plain text to file |
| Any other path | Export JSON to file |

The export happens automatically when the scheduler is freed (typically at program exit).

## Output Formats

### Console summary (stdout)

The default when `--profile` is used without `--profile-output`, or `GGML_PROFILE=1`:

```
=== Profiling Summary ===
  [OP  ] backend 0 MUL_MAT                    45.2%  count=1200  total=  120.50 ms  avg=  100.42 us  ...  12.30 GB/s  [4096 x 4096]
  [OP  ] backend 1 MUL_MAT_ID                 30.1%  count= 600  total=   80.20 ms  avg=  133.67 us  ...   0.08 GB/s  [2688 x 1856 x 128]
  [COPY] backend 0 copy_H2D                    5.3%  count= 200  total=   14.10 ms  avg=   70.50 us  ...   2.50 GB/s
  ...
```

Each line shows: event type (OP or COPY), backend index, operation name, percentage of total time, call count, timing stats, bandwidth, and representative tensor shape.

### Plain text (`.txt`)

A more detailed report with three sections:

1. **Profiling Summary** — total time, record count, unique ops
2. **Per-Backend Summary** — ops and copies per backend with aggregate bandwidth
3. **Operations table** — full breakdown with bandwidth and tensor shapes for all source tensors

### JSON (`.json`)

Machine-readable format suitable for the Python analysis tool. Contains:

- `version`: Format version (currently `2`)
- `backends[]`: Backend metadata (name, device, device type)
- `records[]`: Every profiling event with:
  - `type`: `0` = OP, `1` = COPY
  - `name`: Operation name (e.g. `"MUL_MAT"`, `"copy_H2D"`)
  - `backend_id`, `split_id`: Scheduler indices
  - `start_ns`, `duration_ns`: Timing in nanoseconds
  - `bytes`: Output tensor size (OPs) or transfer size (COPYs)
  - `extra`: Fusion name for fused ops, or `null`
  - `ne_src0`, `ne_src1`, `ne_src2`: Source tensor dimensions (4-element arrays)

`ne_src2` is populated only for `MUL_MAT_ID` (expert selection indices); it is `[0,0,0,0]` for all other ops.

## Python Analysis Tool

The `tools/profiler/profiler.py` script reads JSON exports and produces analysis reports and visualizations.

### Basic usage

```bash
# Print summary
python -m tools.profiler.profiler profile.json

# Show top 10 operations by time
python -m tools.profiler.profiler profile.json --top-ops 10

# Show top 10 longest individual kernels
python -m tools.profiler.profiler profile.json --top-kernels 10

# Show inefficiency ranking (highest time-per-byte)
python -m tools.profiler.profiler profile.json --inefficiency
```

### Export visualizations

```bash
# Interactive HTML timeline (self-contained, no dependencies)
python -m tools.profiler.profiler profile.json --html-viewer timeline.html

# Chrome Trace format (open in chrome://tracing or Perfetto)
python -m tools.profiler.profiler profile.json --chrome-trace trace.json

# Downsample large traces for the HTML viewer
python -m tools.profiler.profiler profile.json --html-viewer timeline.html --html-max-records 50000
```

Multiple exports can be combined in a single invocation:

```bash
python -m tools.profiler.profiler profile.json --html-viewer timeline.html --chrome-trace trace.json --top-ops 20
```

### CLI reference

| Argument | Description |
|----------|-------------|
| `profile` (positional) | Path to profiler JSON file |
| `--chrome-trace FILE` | Export Chrome Trace Event format |
| `--html-viewer FILE` | Export interactive HTML timeline |
| `--html-max-records N` | Limit records in HTML output (0 = unlimited) |
| `--top-ops N` | Show top N operations by total time |
| `--top-kernels N` | Show top N longest individual kernels |
| `--inefficiency` | Rank operations by time per byte (higher = worse) |

### HTML viewer features

The HTML viewer is a self-contained file with no external dependencies:

- **Canvas timeline** with per-backend lanes and color-coded operations
- **Zoom controls** (1s / 100ms / 1ms / 100us) and mouse drag navigation
- **Minimap** showing the full trace with a viewport indicator
- **Hover tooltips** with operation name, duration, shape, and bytes
- **Stats table** with collapsible tree: Operation → Backend → Tensor shape, showing % time, count, avg/min/max, and bandwidth
- **Legend** showing the most frequent operation types

## What Gets Measured

### OP events

Every tensor operation (MUL_MAT, ADD, UNARY, FLASH_ATTN_EXT, etc.) is recorded with:

- **Timing**: Start/end timestamps (nanosecond precision)
- **Bytes**: Output tensor size (`ggml_nbytes(node)`)
- **Tensor shapes**: Dimensions of `src[0]`, `src[1]`, and `src[2]` (when applicable)
- **Bandwidth**: Computed as `bytes / duration` — useful for identifying memory-bound vs compute-bound operations

### COPY events

Data transfers between backends:

- **Direction**: `copy_H2D` (host→device), `copy_D2H` (device→host), `copy_D2D` (device→device)
- **Bytes**: Exact transfer size
- **Bandwidth**: Transfer throughput

### MoE weight copies

When `--cpu-moe` is used, the scheduler selectively copies only the active experts. These partial copies are recorded as individual COPY events with the actual bytes transferred.

## Programmatic API

For custom applications, the profiler can be controlled through the C API defined in `ggml/include/ggml-profiler.h`:

```c
// Enable profiling on a scheduler
ggml_backend_sched_set_profiling(sched, true);

// ... run inference ...

// Get raw records
const ggml_profile_record * records;
int n = ggml_backend_sched_get_profiling_records(sched, &records);

// Or export directly
ggml_backend_sched_print_profiling(sched);                         // stdout
ggml_backend_sched_export_profiling_json(sched, "profile.json");   // JSON file
ggml_backend_sched_export_profiling_text(sched, "profile.txt");    // text file
ggml_backend_sched_write_profiling_json(sched, fp);                // JSON to FILE*
ggml_backend_sched_write_profiling_text(sched, fp);                // text to FILE*

// Reset for next measurement window
ggml_backend_sched_reset_profiling(sched);
```

Records accumulate across multiple `graph_compute` calls until explicitly reset or the scheduler is freed.

## Tips

- **Prompt eval vs generation**: The profiler captures all graph computes. During prompt evaluation you'll see larger batch sizes in tensor shapes; during generation, batch size is typically 1-2.
- **Vulkan concurrent mode**: When Vulkan dispatches multiple operations concurrently, they are reported as a single combined record spanning the full GPU time interval.
- **Bandwidth interpretation**: For compute ops, bandwidth = `output_bytes / duration`. This is not memory bandwidth — it's a proxy for throughput. MUL_MAT with low bandwidth typically indicates compute-bound behavior; high bandwidth indicates memory-bound.
- **Large traces**: For long inference runs, the JSON can be large. Use `--html-max-records` to downsample the HTML viewer, or use Chrome Trace format which handles large files well.
- **Multiple backends**: Backend IDs in the output correspond to the scheduler's priority order (0 = highest priority, typically GPU; last = CPU).

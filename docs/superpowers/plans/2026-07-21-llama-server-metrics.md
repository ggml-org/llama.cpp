# llama-server Metrics Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add KV-cache byte/type, prompt/context-size, and latency metrics to llama-server's `/metrics` endpoint, plus an OTel Collector config to ship them (with host/GPU metrics) to Mimir.

**Architecture:** New fields on the existing `server_metrics` struct (`tools/server/server-context.cpp:787`) are populated in the existing `on_prompt_eval`/`on_prediction` hooks, copied into `server_task_result_metrics` (`tools/server/server-task.h:512`), and rendered in the exposition block (`server-context.cpp:4404`). Histograms are new: a small render helper emits cumulative `_bucket`/`_sum`/`_count` series. KV-cache byte sizes come from a minimal new accessor over llama.cpp's existing internal `size_k_bytes()`/`size_v_bytes()`. Hardware metrics are collected out-of-process by an OTel Collector agent on the GPU host.

**Tech Stack:** C++17 (llama.cpp / ggml), Prometheus text exposition format 0.0.4, OpenTelemetry Collector (YAML), Mimir remote_write.

## Global Constraints

- **Local fork only** — pragmatic, llama.cpp-specific code is acceptable; not for upstream. Still follow existing conventions and keep the change minimal.
- **Namespace:** all server metric names use the `llamacpp:` prefix (existing convention at `server-context.cpp:4505`).
- **Labels:** every server series keeps the existing `{model="…"}` label built at `server-context.cpp:4479-4492`. New series MUST route through the same rendering path so they inherit it.
- **Histogram semantics:** new histograms are **cumulative (monotonic)** — they are NOT added to `reset_bucket()` (`server-context.cpp:855`). Only the existing bucket-scoped gauges reset per scrape. This divergence is intentional (see design doc §"New exposition machinery").
- **Test sandbox limitation:** server pytest cannot download models in this environment. Tests here are written against a **local gguf** via `model_file`; the HF-download presets will not run offline. Verify locally with an explicit `--model` path.
- **Build:** `cmake --build build -j` from repo root; server binary at `build/bin/llama-server`.
- **Bucket ladders (verbatim from spec):**
  - token histograms: `512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144`
  - `time_to_first_token_seconds`: `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64`
  - `generation_latency_seconds`: `0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `include/llama.h` | Public C API | Add `llama_memory_kv_size_bytes()` declaration |
| `src/llama.cpp` | Public C API impl | Implement accessor over `llama_get_memory` → `llama_kv_cache` |
| `tools/server/server-task.h` | Metrics result DTO | Add new fields to `server_task_result_metrics` |
| `tools/server/server-context.cpp` | Metric collection, population, exposition | New `server_metrics` fields + hooks; new histogram render helper; populate result; emit series |
| `tools/server/tests/unit/test_metrics.py` | Server test | New test file scraping `/metrics` |
| `deploy/otel-collector-llama.yaml` | Collector agent config | New config file |
| `deploy/README-metrics.md` | Deployment notes | New doc |

---

## Task 1: Expose KV-cache byte size via public API

**Files:**
- Modify: `include/llama.h` (near `llama_get_memory`, line 559)
- Modify: `src/llama.cpp` (implement alongside other `llama_memory_*` wrappers)

**Interfaces:**
- Produces: `void llama_memory_kv_size_bytes(llama_memory_t mem, size_t * size_k, size_t * size_v);` — writes total K and V cache bytes. Writes `0` to both if `mem` is null or not a `llama_kv_cache` (e.g. recurrent-only). Safe to call any time after context creation.

- [ ] **Step 1: Add the declaration to the public header**

In `include/llama.h`, immediately after the `llama_get_memory` declaration (line 559):

```c
    // Total bytes currently allocated for the K and V cache tensors across all
    // layers. Writes 0 to both outputs when the memory is not a standard KV
    // cache (e.g. recurrent). Either output pointer may be NULL.
    LLAMA_API void llama_memory_kv_size_bytes(
                    llama_memory_t   mem,
                          size_t *   size_k,
                          size_t *   size_v);
```

- [ ] **Step 2: Implement it in src/llama.cpp**

Find where `llama_get_memory` is implemented in `src/llama.cpp` (search `llama_get_memory`). Add nearby. It must include `src/llama-kv-cache.h` (check the top of the file; add the include if absent):

```cpp
void llama_memory_kv_size_bytes(llama_memory_t mem, size_t * size_k, size_t * size_v) {
    size_t k = 0;
    size_t v = 0;
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        k = kv->size_k_bytes();
        v = kv->size_v_bytes();
    }
    if (size_k) { *size_k = k; }
    if (size_v) { *size_v = v; }
}
```

- [ ] **Step 3: Build**

Run: `cmake --build build -j --target llama`
Expected: compiles clean. (`size_k_bytes()`/`size_v_bytes()` are declared at `src/llama-kv-cache.h:292-293`; `dynamic_cast` works because `llama_kv_cache : public llama_memory_i`.)

- [ ] **Step 4: Commit**

```bash
git add include/llama.h src/llama.cpp
git commit -m "llama: expose KV cache byte size via llama_memory_kv_size_bytes"
```

---

## Task 2: Add KV-byte and cache-type fields to server metrics result

**Files:**
- Modify: `tools/server/server-task.h:512-551` (`server_task_result_metrics`)
- Modify: `tools/server/server-context.cpp:2502-2533` (populate the result)

**Interfaces:**
- Consumes: `llama_memory_kv_size_bytes` (Task 1); `params_base.cache_type_k` / `params_base.cache_type_v` (`common_params`, type `ggml_type`); `ctx_tgt` (member, `server-context.cpp:910`).
- Produces: `server_task_result_metrics` gains `kv_cache_k_bytes`, `kv_cache_v_bytes` (`uint64_t`), and `kv_cache_type_k`, `kv_cache_type_v` (`std::string`).

- [ ] **Step 1: Add fields to the result struct**

In `tools/server/server-task.h`, inside `server_task_result_metrics` after the existing `kv_cache_cells` field (line 544):

```cpp
    // KV cache byte footprint and live quantization types
    uint64_t    kv_cache_k_bytes = 0;
    uint64_t    kv_cache_v_bytes = 0;
    std::string kv_cache_type_k;   // e.g. "f16", "q8_0"
    std::string kv_cache_type_v;
```

- [ ] **Step 2: Populate them where the metrics result is built**

In `tools/server/server-context.cpp`, in the `SERVER_TASK_TYPE_METRICS` handler, immediately after `res->kv_cache_cells = (uint64_t) n_ctx;` (line 2533):

```cpp
                    {
                        size_t k_bytes = 0;
                        size_t v_bytes = 0;
                        llama_memory_kv_size_bytes(llama_get_memory(ctx_tgt), &k_bytes, &v_bytes);
                        res->kv_cache_k_bytes = (uint64_t) k_bytes;
                        res->kv_cache_v_bytes = (uint64_t) v_bytes;
                    }
                    res->kv_cache_type_k = ggml_type_name(params_base.cache_type_k);
                    res->kv_cache_type_v = ggml_type_name(params_base.cache_type_v);
```

- [ ] **Step 3: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean. (`ggml_type_name` is already used in this file, e.g. `arg.cpp`; it is declared in `ggml.h` which the server includes transitively.)

- [ ] **Step 4: Commit**

```bash
git add tools/server/server-task.h tools/server/server-context.cpp
git commit -m "server: collect KV cache bytes and live cache types for /metrics"
```

---

## Task 3: Emit KV-byte and cache-type series in the exposition block

**Files:**
- Modify: `tools/server/server-context.cpp:4442-4474` (gauge definitions) and `:4494-4509` (render loop)

**Interfaces:**
- Consumes: `res_task->kv_cache_k_bytes`, `kv_cache_v_bytes`, `kv_cache_type_k`, `kv_cache_type_v` (Task 2).
- Produces: Prometheus series `llamacpp:kv_cache_k_bytes`, `llamacpp:kv_cache_v_bytes` (gauges), and `llamacpp:kv_cache_type{cache="k"|"v",type="…"}` (gauge, value 1).

- [ ] **Step 1: Add the two byte gauges to the gauge array**

In `tools/server/server-context.cpp`, inside the `"gauge"` array, after the `kv_cache_cells` entry (ends line 4473, `}}` closes the gauge list — insert before that closing):

```cpp
            },{
                    {"name",  "kv_cache_k_bytes"},
                    {"help",  "Bytes allocated for the K cache across all layers."},
                    {"value",  res_task->kv_cache_k_bytes}
            },{
                    {"name",  "kv_cache_v_bytes"},
                    {"help",  "Bytes allocated for the V cache across all layers."},
                    {"value",  res_task->kv_cache_v_bytes}
```

- [ ] **Step 2: Emit the labelled cache-type series after the main render loop**

The existing render loop ends at line 4509 (`}` closing the outer `for`). Immediately after it, before `res->headers[...]` (line 4511), append the type series manually (they need a custom label, so they bypass the scalar loop). Reuse the already-built `model_label` string but merge the extra label:

```cpp
        // cache-type series: value is always 1; the type is carried as a label so
        // Grafana can display which quantization is live per model. model_label is
        // "{model=\"...\"}" — splice the cache/type labels in before the closing brace.
        auto emit_type = [&](const char * cache, const std::string & type) {
            std::string labels = model_label;
            labels.pop_back(); // drop trailing '}'
            labels += ",cache=\"";
            labels += cache;
            labels += "\",type=\"";
            labels += type;
            labels += "\"}";
            prometheus << "# HELP llamacpp:kv_cache_type Live KV cache quantization type (value is always 1).\n"
                       << "# TYPE llamacpp:kv_cache_type gauge\n"
                       << "llamacpp:kv_cache_type" << labels << " 1\n";
        };
        emit_type("k", res_task->kv_cache_type_k);
        emit_type("v", res_task->kv_cache_type_v);
```

- [ ] **Step 3: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean.

- [ ] **Step 4: Manual smoke check**

Run (substitute a local gguf path):
```bash
./build/bin/llama-server --model /path/to/model.gguf --metrics --port 8099 &
sleep 8 && curl -s localhost:8099/metrics | grep -E "kv_cache_(k_bytes|v_bytes|type)"
kill %1
```
Expected: three metric families printed; `kv_cache_k_bytes` > 0; `kv_cache_type{cache="k",type="f16"} 1` (or the configured type).

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "server: expose kv_cache_{k,v}_bytes and kv_cache_type in /metrics"
```

---

## Task 4: Add a cumulative-histogram render helper

**Files:**
- Modify: `tools/server/server-context.cpp` — add a free helper function above the `get_metrics` lambda (before line 4371).

**Interfaces:**
- Produces: `static void render_histogram(std::stringstream & out, const std::string & name, const std::string & help, const std::string & model_label, const std::vector<double> & bounds, const std::vector<uint64_t> & bucket_counts, uint64_t count, double sum);`
  - `bounds` are the upper bounds (the ladder). `bucket_counts` has the same length as `bounds`, each the **cumulative** count `≤ bound`. Emits `_bucket{le="…"}` for each bound plus `le="+Inf"`, then `_sum` and `_count`. Splices `le` into `model_label` the same way Task 3 splices `cache`/`type`.

- [ ] **Step 1: Add the helper**

In `tools/server/server-context.cpp`, immediately before `this->get_metrics = ...` (line 4371 is inside a method; place the helper at file scope near the other `static` helpers at the top of the anonymous/impl section — search for an existing `static` free function in this file and co-locate). Use this exact implementation:

```cpp
static void render_histogram(std::stringstream & out,
                             const std::string & name,
                             const std::string & help,
                             const std::string & model_label,
                             const std::vector<double> & bounds,
                             const std::vector<uint64_t> & bucket_counts,
                             uint64_t count,
                             double sum) {
    out << "# HELP llamacpp:" << name << " " << help << "\n";
    out << "# TYPE llamacpp:" << name << " histogram\n";

    // model_label is "{model=\"...\"}"; splice le=... before the closing brace.
    auto with_le = [&](const std::string & le) {
        std::string labels = model_label;
        labels.pop_back();
        labels += ",le=\"";
        labels += le;
        labels += "\"}";
        return labels;
    };

    for (size_t i = 0; i < bounds.size(); ++i) {
        std::ostringstream le;
        le << bounds[i];
        out << "llamacpp:" << name << "_bucket" << with_le(le.str())
            << " " << bucket_counts[i] << "\n";
    }
    out << "llamacpp:" << name << "_bucket" << with_le("+Inf") << " " << count << "\n";
    out << "llamacpp:" << name << "_sum"   << model_label << " " << sum   << "\n";
    out << "llamacpp:" << name << "_count" << model_label << " " << count << "\n";
}
```

- [ ] **Step 2: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean (helper is unused for now — acceptable; it is consumed in Task 6). If the compiler errors on unused-function with `-Werror`, proceed to Task 6 in the same commit; otherwise commit now.

- [ ] **Step 3: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "server: add cumulative-histogram render helper for /metrics"
```

---

## Task 5: Collect prompt/context-size and latency histogram data

**Files:**
- Modify: `tools/server/server-context.cpp:787-862` (`server_metrics` struct + hooks)
- Modify: `tools/server/server-task.h:512` (result struct)
- Modify: `tools/server/server-context.cpp:2502` (populate result)

**Interfaces:**
- Consumes: `server_slot` fields — `n_prompt_tokens_processed` (used at `on_prompt_eval`), `prompt.n_tokens()`, `t_prompt_processing` (ms, `server-context.cpp:283`), `t_token_generation` (ms, `:284`), `n_decoded`.
- Produces: four cumulative histogram accumulators on `server_metrics`, mirrored on `server_task_result_metrics` as `std::vector<uint64_t>` bucket arrays + `uint64_t` counts + `double` sums, named:
  `hist_prompt_tokens_*`, `hist_context_tokens_*`, `hist_ttft_*`, `hist_gen_latency_*` (suffixes `_buckets`, `_count`, `_sum`).

- [ ] **Step 1: Add a small histogram accumulator type and the four instances to `server_metrics`**

In `tools/server/server-context.cpp`, just above `struct server_metrics` (line 787):

```cpp
// minimal cumulative histogram: bounds are fixed upper edges; observe() bumps
// every bucket whose bound >= value (cumulative form Prometheus expects).
struct metric_histogram {
    std::vector<double>   bounds;
    std::vector<uint64_t> buckets; // cumulative counts, same length as bounds
    uint64_t              count = 0;
    double                sum   = 0.0;

    explicit metric_histogram(std::vector<double> b)
        : bounds(std::move(b)), buckets(bounds.size(), 0) {}

    void observe(double value) {
        count++;
        sum += value;
        for (size_t i = 0; i < bounds.size(); ++i) {
            if (value <= bounds[i]) {
                buckets[i]++;
            }
        }
    }
};
```

- [ ] **Step 2: Add the four histograms as members of `server_metrics`**

Inside `struct server_metrics`, after `n_busy_slots_total` (line 814):

```cpp
    metric_histogram hist_prompt_tokens {
        {512,1024,2048,4096,6144,8192,12288,16384,24576,32768,49152,65536,98304,131072,196608,262144}};
    metric_histogram hist_context_tokens {
        {512,1024,2048,4096,6144,8192,12288,16384,24576,32768,49152,65536,98304,131072,196608,262144}};
    metric_histogram hist_ttft_seconds {
        {0.05,0.1,0.25,0.5,1,2,4,8,16,32,64}};
    metric_histogram hist_gen_latency_seconds {
        {0.1,0.25,0.5,1,2,5,10,20,40,80}};
```

- [ ] **Step 3: Record observations in the existing hooks**

In `on_prompt_eval` (line 820), after the existing `n_tokens_max = std::max(...)` line (828):

```cpp
        hist_prompt_tokens.observe((double) slot.n_prompt_tokens_processed);
        hist_ttft_seconds.observe(slot.t_prompt_processing / 1.e3); // ms -> s
```

In `on_prediction` (line 831), after the draft counters (838):

```cpp
        hist_context_tokens.observe((double) slot.prompt.n_tokens());
        hist_gen_latency_seconds.observe(slot.t_token_generation / 1.e3); // ms -> s
```

(These are cumulative — deliberately NOT touched by `reset_bucket()`.)

- [ ] **Step 4: Mirror the four histograms onto the result struct**

In `tools/server/server-task.h`, after the fields added in Task 2:

```cpp
    // cumulative latency/size histograms (bucket arrays are cumulative counts)
    std::vector<uint64_t> hist_prompt_tokens_buckets;
    uint64_t              hist_prompt_tokens_count = 0;
    double                hist_prompt_tokens_sum   = 0.0;
    std::vector<uint64_t> hist_context_tokens_buckets;
    uint64_t              hist_context_tokens_count = 0;
    double                hist_context_tokens_sum   = 0.0;
    std::vector<uint64_t> hist_ttft_buckets;
    uint64_t              hist_ttft_count = 0;
    double                hist_ttft_sum   = 0.0;
    std::vector<uint64_t> hist_gen_latency_buckets;
    uint64_t              hist_gen_latency_count = 0;
    double                hist_gen_latency_sum   = 0.0;
```

- [ ] **Step 5: Copy the histograms into the result**

In `server-context.cpp`, in the metrics handler after the KV-byte population from Task 2 (i.e. after `res->kv_cache_type_v = ...`):

```cpp
                    res->hist_prompt_tokens_buckets = metrics.hist_prompt_tokens.buckets;
                    res->hist_prompt_tokens_count   = metrics.hist_prompt_tokens.count;
                    res->hist_prompt_tokens_sum     = metrics.hist_prompt_tokens.sum;
                    res->hist_context_tokens_buckets = metrics.hist_context_tokens.buckets;
                    res->hist_context_tokens_count   = metrics.hist_context_tokens.count;
                    res->hist_context_tokens_sum     = metrics.hist_context_tokens.sum;
                    res->hist_ttft_buckets = metrics.hist_ttft_seconds.buckets;
                    res->hist_ttft_count   = metrics.hist_ttft_seconds.count;
                    res->hist_ttft_sum     = metrics.hist_ttft_seconds.sum;
                    res->hist_gen_latency_buckets = metrics.hist_gen_latency_seconds.buckets;
                    res->hist_gen_latency_count   = metrics.hist_gen_latency_seconds.count;
                    res->hist_gen_latency_sum     = metrics.hist_gen_latency_seconds.sum;
```

- [ ] **Step 6: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean.

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-context.cpp tools/server/server-task.h
git commit -m "server: accumulate prompt/context-size and latency histograms"
```

---

## Task 6: Render the histograms in /metrics

**Files:**
- Modify: `tools/server/server-context.cpp` — after the cache-type emission from Task 3 (before `res->headers[...]`, line 4511).

**Interfaces:**
- Consumes: `render_histogram` (Task 4); histogram fields on `res_task` (Task 5); the fixed bounds vectors.

- [ ] **Step 1: Emit the four histograms**

After the `emit_type("v", ...)` line from Task 3:

```cpp
        const std::vector<double> token_bounds = {512,1024,2048,4096,6144,8192,12288,16384,24576,32768,49152,65536,98304,131072,196608,262144};
        const std::vector<double> ttft_bounds  = {0.05,0.1,0.25,0.5,1,2,4,8,16,32,64};
        const std::vector<double> gen_bounds   = {0.1,0.25,0.5,1,2,5,10,20,40,80};

        render_histogram(prometheus, "prompt_tokens_size", "Distribution of prompt sizes in tokens.",
            model_label, token_bounds, res_task->hist_prompt_tokens_buckets,
            res_task->hist_prompt_tokens_count, res_task->hist_prompt_tokens_sum);
        render_histogram(prometheus, "context_used_tokens", "Distribution of context used in tokens.",
            model_label, token_bounds, res_task->hist_context_tokens_buckets,
            res_task->hist_context_tokens_count, res_task->hist_context_tokens_sum);
        render_histogram(prometheus, "time_to_first_token_seconds", "Distribution of prompt-eval (TTFT) latency in seconds.",
            model_label, ttft_bounds, res_task->hist_ttft_buckets,
            res_task->hist_ttft_count, res_task->hist_ttft_sum);
        render_histogram(prometheus, "generation_latency_seconds", "Distribution of generation latency in seconds.",
            model_label, gen_bounds, res_task->hist_gen_latency_buckets,
            res_task->hist_gen_latency_count, res_task->hist_gen_latency_sum);
```

- [ ] **Step 2: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean; `render_histogram` no longer unused.

- [ ] **Step 3: Manual smoke check**

```bash
./build/bin/llama-server --model /path/to/model.gguf --metrics --port 8099 &
sleep 8
curl -s -X POST localhost:8099/completion -d '{"prompt":"hello world","n_predict":8}' >/dev/null
curl -s localhost:8099/metrics | grep -E "prompt_tokens_size_(bucket|sum|count)|time_to_first_token_seconds_bucket"
kill %1
```
Expected: `_bucket{...,le="512"}`, ascending cumulative counts, a `le="+Inf"` line equal to `_count`, and a `_sum`.

- [ ] **Step 4: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "server: render prompt/context and latency histograms in /metrics"
```

---

## Task 7: Add per-device VRAM gauges

**Files:**
- Modify: `tools/server/server-task.h:512` (result struct)
- Modify: `tools/server/server-context.cpp` — populate in metrics handler; emit after histograms.

**Interfaces:**
- Consumes: `ggml_backend_dev_memory(dev, &free, &total)` (`ggml/include/ggml-backend.h:181`); device enumeration `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)` (already used at `server-context.cpp:1087`); `ggml_backend_dev_name(dev)`.
- Produces: result field `std::vector<std::tuple<std::string,uint64_t,uint64_t>> vram_devices;` (name, free, total), rendered as `llamacpp:vram_free_bytes{device="…"}` / `llamacpp:vram_total_bytes{device="…"}`.

- [ ] **Step 1: Add result field**

In `tools/server/server-task.h`, after the histogram fields:

```cpp
    // per-device VRAM (device name, free bytes, total bytes)
    std::vector<std::tuple<std::string, uint64_t, uint64_t>> vram_devices;
```

Ensure `<tuple>` is included at the top of `server-task.h` (add `#include <tuple>` if absent).

- [ ] **Step 2: Populate in the metrics handler**

In `server-context.cpp`, after the histogram copy from Task 5 Step 5:

```cpp
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                            continue;
                        }
                        size_t free_b = 0, total_b = 0;
                        ggml_backend_dev_memory(dev, &free_b, &total_b);
                        res->vram_devices.emplace_back(ggml_backend_dev_name(dev),
                                                       (uint64_t) free_b, (uint64_t) total_b);
                    }
```

- [ ] **Step 3: Emit after the histograms (Task 6)**

```cpp
        for (const auto & d : res_task->vram_devices) {
            std::string dev_label = model_label;
            dev_label.pop_back();
            dev_label += ",device=\"" + std::get<0>(d) + "\"}";
            prometheus << "# HELP llamacpp:vram_free_bytes Free VRAM on the device.\n"
                       << "# TYPE llamacpp:vram_free_bytes gauge\n"
                       << "llamacpp:vram_free_bytes"  << dev_label << " " << std::get<1>(d) << "\n"
                       << "# HELP llamacpp:vram_total_bytes Total VRAM on the device.\n"
                       << "# TYPE llamacpp:vram_total_bytes gauge\n"
                       << "llamacpp:vram_total_bytes" << dev_label << " " << std::get<2>(d) << "\n";
        }
```

- [ ] **Step 4: Build**

Run: `cmake --build build -j --target llama-server`
Expected: compiles clean. (`GGML_BACKEND_DEVICE_TYPE_GPU` and `ggml_backend_dev_type` are in `ggml-backend.h`.)

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-task.h tools/server/server-context.cpp
git commit -m "server: expose per-device VRAM free/total gauges in /metrics"
```

---

## Task 8: Python test for the new metrics

**Files:**
- Create: `tools/server/tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: the running server's `/metrics` endpoint. Uses the `ServerPreset`/`ServerProcess` fixtures from `tools/server/tests/utils.py`.

> **Sandbox note (Global Constraints):** the default presets download from HF and will NOT work offline here. To run locally, set `server.model_file` to a local gguf and clear `model_hf_repo`/`model_hf_file`. The test below uses the standard preset so it runs in CI; verify locally by editing the fixture as noted.

- [ ] **Step 1: Write the test**

Create `tools/server/tests/unit/test_metrics.py`:

```python
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 1
    server.server_metrics = True  # appends --metrics (utils.py:192)
    # To run OFFLINE: uncomment and point at a local gguf:
    # server.model_hf_repo = None
    # server.model_hf_file = None
    # server.model_file = "/path/to/local.gguf"


def _metrics_text():
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    return res.body if isinstance(res.body, str) else res.body.decode()


def test_kv_cache_bytes_and_type_present():
    global server
    server.start()
    text = _metrics_text()
    assert "llamacpp:kv_cache_k_bytes" in text
    assert "llamacpp:kv_cache_v_bytes" in text
    assert 'llamacpp:kv_cache_type{' in text and 'cache="k"' in text


def test_histograms_have_bucket_sum_count():
    global server
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "hello world", "n_predict": 8})
    text = _metrics_text()
    for name in ["prompt_tokens_size", "time_to_first_token_seconds", "generation_latency_seconds"]:
        assert f'llamacpp:{name}_bucket{{' in text, name
        assert f'llamacpp:{name}_sum' in text, name
        assert f'llamacpp:{name}_count' in text, name
    # +Inf bucket must equal _count for prompt_tokens_size
    inf_lines = [l for l in text.splitlines() if l.startswith("llamacpp:prompt_tokens_size_bucket") and 'le="+Inf"' in l]
    count_lines = [l for l in text.splitlines() if l.startswith("llamacpp:prompt_tokens_size_count")]
    assert inf_lines and count_lines
    assert inf_lines[0].split()[-1] == count_lines[0].split()[-1]


def test_vram_gauges_present_when_gpu():
    global server
    server.start()
    text = _metrics_text()
    # VRAM series only emitted when a GPU backend is present; assert well-formed if so.
    if "llamacpp:vram_total_bytes" in text:
        assert "llamacpp:vram_free_bytes" in text
        assert 'device="' in text
```

- [ ] **Step 2: Check the `server_metrics` fixture flag name**

Run: `grep -n "server_metrics\|--metrics" tools/server/tests/utils.py`
Expected: confirms the `ServerProcess` attribute that appends `--metrics` (line ~192). If the attribute differs from `server_metrics`, update the fixture line to match.

- [ ] **Step 3: Run the test (requires network for model download, or a local gguf per the note)**

Run: `cd tools/server/tests && python -m pytest unit/test_metrics.py -v`
Expected: PASS. If offline, edit the fixture per the sandbox note first; otherwise the model download will time out (this is the known sandbox limitation, not a test failure).

- [ ] **Step 4: Commit**

```bash
git add tools/server/tests/unit/test_metrics.py
git commit -m "server: add /metrics test for KV bytes, histograms, and VRAM"
```

---

## Task 9: OTel Collector agent config + deployment notes

**Files:**
- Create: `deploy/otel-collector-llama.yaml`
- Create: `deploy/README-metrics.md`

**Interfaces:** none (ops artifacts). Placeholders `<mimir-host>`, `<gpuhost>`, `<tenant>`, and rung ports are environment facts filled at deploy time.

- [ ] **Step 1: Write the Collector config**

Create `deploy/otel-collector-llama.yaml`:

```yaml
# OTel Collector agent — runs on the Fedora GPU host, ships directly to Mimir.
# Deploy as a systemd service alongside llama-server. See README-metrics.md.
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: llama-server
          scrape_interval: 15s
          static_configs:
            - targets: ['127.0.0.1:8081', '127.0.0.1:8082']  # one per rung — set to your ports
  hostmetrics:
    collection_interval: 15s
    scrapers: { cpu: {}, memory: {}, load: {}, filesystem: {}, network: {} }
  dcgm:
    collection_interval: 15s
    # Requires the nvidia-dcgm service AND an otelcol-contrib build.
    # Fallback (core distro / no DCGM): delete this receiver and add a prometheus
    # scrape job for nvidia_gpu_exporter instead.

processors:
  resourcedetection:
    detectors: [system]
    system: { hostname_sources: [os] }
  batch: {}

exporters:
  prometheusremotewrite:
    endpoint: "http://<mimir-host>:9009/api/v1/push"
    external_labels: { host: "<gpuhost>" }
    # headers: { X-Scope-OrgID: "<tenant>" }   # required only if Mimir is multi-tenant

service:
  pipelines:
    metrics:
      receivers: [prometheus, hostmetrics, dcgm]
      processors: [resourcedetection, batch]
      exporters: [prometheusremotewrite]
```

- [ ] **Step 2: Write the deployment notes**

Create `deploy/README-metrics.md`:

```markdown
# llama-server metrics → Mimir

## Topology
OTel Collector agent on the Fedora GPU host scrapes localhost llama-server rungs,
host metrics, and GPU metrics, then remote_writes directly to Mimir. Grafana reads
Mimir. See `docs/superpowers/specs/2026-07-21-llama-server-metrics-design.md`.

## Prerequisites
- llama-server started with `--metrics` (bind metrics port to 127.0.0.1).
- `otelcol-contrib` (for the `dcgm` and `hostmetrics` receivers). Verify:
  `otelcol-contrib components | grep -E "dcgm|hostmetrics|prometheusremotewrite"`
- For GPU: either the `nvidia-dcgm` service running, OR `nvidia_gpu_exporter`
  (then swap the `dcgm` receiver for a prometheus scrape job — see config comment).

## Fill in before deploying
- `static_configs.targets`: one `127.0.0.1:<port>` per running rung.
- `<mimir-host>`: Mimir distributor host (default push path `/api/v1/push`, port 9009).
- `<gpuhost>`: label value identifying this host.
- `X-Scope-OrgID`: uncomment iff Mimir enforces multi-tenancy (else writes 401).

## Install (systemd sketch)
1. `cp deploy/otel-collector-llama.yaml /etc/otelcol-contrib/config.yaml`
2. Edit placeholders.
3. `systemctl restart otelcol-contrib && systemctl enable otelcol-contrib`
4. Verify: `curl -s localhost:8888/metrics | grep otelcol_exporter_sent` climbs.

## Grafana panels (PromQL)
- KV vs VRAM: `llamacpp:kv_cache_k_bytes + llamacpp:kv_cache_v_bytes` overlaid with
  `llamacpp:vram_free_bytes` (and DCGM `DCGM_FI_DEV_FB_USED`).
- Prompt p95: `histogram_quantile(0.95, sum by (le,model) (rate(llamacpp:prompt_tokens_size_bucket[$__rate_interval])))`
- TTFT p95: same over `llamacpp:time_to_first_token_seconds_bucket`.
- Spec-decode accept rate: `rate(llamacpp:draft_tokens_accepted_total[$__rate_interval]) / rate(llamacpp:draft_tokens_total[$__rate_interval])`
- Context-shift rate: `rate(llamacpp:n_ctx_shift_total[$__rate_interval])`
```

- [ ] **Step 3: Commit**

```bash
git add deploy/otel-collector-llama.yaml deploy/README-metrics.md
git commit -m "deploy: OTel Collector agent config and notes for llama-server metrics"
```

---

## Self-Review

**Spec coverage:**
- §2a KV bytes + type → Tasks 1–3 ✓
- §2b prompt/context histograms → Tasks 4–6 ✓
- §2c TTFT/gen-latency histograms → Tasks 4–6 ✓
- §2d accept-rate & ctx-shift Grafana-only → no server task; PromQL in Task 9 README ✓
- §2e VRAM gauges → Task 7 ✓
- New cumulative-histogram exposition helper → Task 4; non-reset semantics honored (not added to `reset_bucket()`) ✓
- Collector agent → Mimir → Task 9 ✓
- Testing (§Testing) → Task 8 + manual smoke checks in Tasks 3/6 ✓

**Placeholder scan:** Only intentional deployment placeholders (`<mimir-host>` etc.) in Task 9, explicitly flagged as environment facts. No TODO/TBD in code steps; all code shown in full.

**Type consistency:** `llama_memory_kv_size_bytes(mem, size_k, size_v)` — signature identical in Task 1 (decl+impl) and Task 2 (call). `render_histogram(...)` signature identical in Task 4 (def) and Task 6 (call). `metric_histogram::observe`/`.buckets`/`.count`/`.sum` used consistently in Task 5. Result-struct field names (`kv_cache_k_bytes`, `hist_*_buckets/_count/_sum`, `vram_devices`) match between definition (Tasks 2/5/7 in `server-task.h`) and use (exposition in Tasks 3/6/7).

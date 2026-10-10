# A770 llama.cpp interactivity: xe and i915

Research snapshot: 2026-09-29. Read-only host/source inspection and primary-source research; no configuration changes or GPU benchmarks performed for this note. Current repository source, service configuration and the latest research README supersede stale AGENTS.md descriptions.

## Correct workload baseline first

The research README was updated at 17:49-18:05: historical llama-bench tests defaulted to `--moe-cache auto`, including the i915 reference. Production explicitly uses `--moe-cache off`. These select different placements. The auto placement streams selected expert weights from host through the scheduler; the tracer identified a Q6_K expert upload. This is neither evidence that production uses that placement nor a pure activation-copy diagnosis.

Latest recorded xe/copy-engine-off benchmarks: auto 14.52 tg/s, soft 22.72, off 47.55; at 8k auto 13.70, soft 20.08, off 42.07. These are recorded experiments, not reproduced here. Production-like short real text with off placement gives 32.78 tg/s copy-off versus 32.38 copy-on. There is no matched i915 production-placement comparison. Consequently the established xe decode penalty applies to auto placement; it must not be generalized to production. The blanket claim that random tokens are worst for locality is also unsupported by these results.

Evidence: [latest campaign README](/home/svnbjrn/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/README.md), [copy trace](/home/svnbjrn/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/copy-path-trace.md). Current scheduler's `copy_experts` is in [ggml-backend.cpp](/mnt/mrgr/ggml-llama.cpp/ggml/src/ggml-backend.cpp:2436).

## Concrete starting profiles

These are defensible baselines, not measured optima. Keep model, binary, context, placement, prompt and speculation identical when comparing drivers.

| Setting | xe production baseline | i915 comparison baseline |
|---|---|---|
| Device | `ONEAPI_DEVICE_SELECTOR=level_zero:0`, verify maps to A770 | Same |
| Copy engine | Preserve `UR_L0_USE_COPY_ENGINE=0` | Initially preserve 0 to isolate KMD; separately compare default afterwards |
| Placement | Explicit `--moe-cache off --fit on --fit-target 1024` | Same, inspect resulting placement |
| Slots/threads | Existing `--parallel 1 --threads 12 --threads-batch 12` | Same |
| KV/attention | Existing `-fa on -ctk q8_0 -ctv q8_0` | Same |
| Context | Existing 131072 until requirements justify reducing it | Same for fair comparison |
| Speculation | Existing ngram-mod for production; disable in driver-isolation runs | Same |
| Graphs | Preserve existing `GGML_SYCL_ENABLE_GRAPH=1` and eviction timeout 300; verify actual replay | Same flags; actual device support may differ |
| Submission | Keep current immediate-list behavior | Same; do not force regular lists as a latency fix |
| Scheduler | Preserve finite recovery controls, current timeslice; no more timeout inflation | Use finite driver defaults/previously justified values; no disabling hangcheck |
| Frequency/power | Automatic scaling within existing 700-2400 MHz request range; 230 W existing power cap | Automatic scaling, read actual bounds after boot |

Local evidence for copy-off: four recorded benchmark rounds plus six server requests completed; default production config suffered BCS resets. This is a promising workaround with finite evidence, not proof of elimination. Immediate-list-off gave 2.17 tg/s and watchdog termination in the recorded contaminated run, so it is not a useful local workaround. Disabling driver in-order lists reset BCS; event-cache-off also stalled; counter-based-events-off only has one passing trial.

## What the driver knobs actually change

Xe's GuC SLPC requests clocks within min/max bounds, but PCODE decides actual frequency under thermal/power limits. `cur_freq` is a request, `act_freq` is achieved frequency. The live snapshot's `cur_freq=2400` and idle `act_freq=0` is not evidence the GPU is stuck at 2400 MHz. If cold requests show measured ramp latency, a temporary minimum at `rpe_freq` is a reasonable experiment; here min already equals rpe (700 MHz). Pinning min=max is not justified without p95 benefit and power/temperature evidence. [Xe frequency documentation](https://docs.kernel.org/gpu/xe/xe_gt_freq.html)

Xe engine-class settings use microseconds for `timeslice_duration_us` and `preempt_timeout_us`; i915 engine settings use milliseconds. A timeslice determines scheduling opportunity under contention; the preemption timeout is a recovery deadline after preemption fails, not the normal scheduling quantum. Larger preemption/job timeouts cannot accelerate a copy or fix an unsignaled dependency. Existing xe CCS/RCS preempt 7.5 s is a substantial recovery delay, not an interactivity optimization. Do not label it harmless. Existing timeslice is 1 ms; preserve it until there is a measured scheduling problem. Xe defaults affect subsequently created exec queues, so merely writing sysfs during a run is not a controlled A/B. [Xe sysfs implementation](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_hw_engine_class_sysfs.c), [i915 sysfs implementation](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/i915/gt/sysfs_engines.c)

Xe LR-mode VMs explicitly allow submissions without an upper execution-time limit. A larger ordinary job timeout cannot rescue a stalled LR compute queue. Keep process/request watchdogs and reset/error collection; do not disable kernel hang detection. CPU `Nice=-16` in the service is not GPU queue priority and can hurt other host tasks under contention. [Xe uAPI](https://github.com/torvalds/linux/blob/master/include/uapi/drm/xe_drm.h)

The historical merge plan calls already-supported i915 platforms permanently experimental under xe. This is not a blanket statement that xe is experimental on all Intel GPUs. Current upstream `dg2_desc` still has `require_force_probe=true`, and its probe diagnostic explicitly says the hardware is not officially supported by xe in that kernel. The practical implication is less assured DG2 coverage, not a reason to abandon debugging. [Historical plan](https://docs.kernel.org/6.5/gpu/rfc/xe.html), [current xe PCI table](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_pci.c)

## Runtime, memory and graphs

SYCL queue ordering, UR command lists/events and NEO engine selection sit above either KMD. Copy-engine-off changes where the runtime performs copies and their synchronization topology. It does not turn xe into i915, disable all kernel memory migration, or prove BCS hardware is defective.

Intel documents A-series as normally using the V1 Level Zero adapter and immediate lists; newer B-series uses V2 by default. Regular lists can amortize submission but can also serialize unrelated queues; immediate lists have different host overhead. Version-specific source wins over old internet environment-variable recipes. Do not force `SYCL_UR_USE_LEVEL_ZERO_V2` or pile legacy `SYCL_PI_*` variables onto current `UR_L0_*` controls as a generic A770 optimization. [Intel immediate-list guide](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html), [runtime variables](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md)

The current fork reads `GGML_SYCL_ENABLE_GRAPH`, default zero, and checks device graph support. `GGML_SYCL_DISABLE_GRAPHS` from old notes is absent from this source. Setting ENABLE_GRAPH=1 does not prove replay. `GGML_SYCL_GRAPH_PROFILE=1` can expose replay counts during a separate diagnostic run. Graph eviction 300 seconds can avoid rerecording after shorter chat pauses if graphs actually run; it retains state longer. Keep current setting pending evidence rather than claim graphs are always unsupported on DG2. [current graph gate](/mnt/mrgr/ggml-llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:7196)

Keep normal pinned-host allocation unless it is implicated by evidence. Pinned-off previously reset BCS; it is not a proven solution. Pinning avoids some paging/staging costs but consumes host-memory resources. The fork's `GGML_SYCL_HOST_PINNED_MEM_2G` is an allocation-layout experiment with possible startup cost, not a universal interactivity setting. Neither pinning nor `--mlock` eliminates GPU event dependencies. [fork runtime documentation](/mnt/mrgr/ggml-llama.cpp/docs/backend/SYCL.md:795)

## Interactivity above the driver

Measure TTFT after idle, warm TTFT, median and p95 inter-token gap, longest gap, and completion/error rate at actual context lengths. Average pp/tg throughput hides a ten-second pause. Compare the same real prompts; report prefix-cache hits and speculation acceptance. The existing ngram-mod configuration may help repeated code/text but repetitive synthetic prompts can exaggerate its benefit.

Keep one slot for single-user latency. More slots can raise aggregate throughput while worsening per-request latency and KV pressure. For concurrent users, continuous batching and smaller prompt batches are an explicit tradeoff; test batch/ubatch 256 against existing values only if prompt processing blocks other responses. Reducing batch sizes is not guaranteed faster TTFT. A 131k reserved context can consume memory that could hold weights; reducing it to the needed ceiling may change placement favorably, but must follow actual conversation requirements. Source exposes separate batch, ubatch, parallel, context and CPU-thread controls. [argument definitions](/mnt/mrgr/ggml-llama.cpp/common/arg.cpp:1745)

Keep model files on the already-used direct filesystem backing path: the unit deliberately resolves mergerfs to avoid FUSE variability. Avoid simultaneous memory-heavy scans and compilation during latency comparisons. Local topology has one NUMA node and two L3 groups: CPUs 0-5/12-17 and 6-11/18-23. Therefore `numactl --interleave=all` does not distribute across separate memory nodes here. A six-core CCD affinity experiment versus twelve physical cores could isolate cache/locality and submission contention, but neither CCD pinning nor all 24 threads is automatically optimal. Do not assign realtime scheduling to the server; CPU priority does not repair a GPU wait.

The A770 is not fully isolated just because no monitor is attached: prior trace recorded KWin as a DRM client. If GUI responsiveness or tail latency matters, keep desktop rendering on the Raphael iGPU and verify actual DRM clients. This is a scheduling-isolation hypothesis, not a claim KWin caused malformed BCS commands. Avoid switching display configuration during an inference comparison.

Verify ReBAR/Above-4G decoding and actual PCIe link capability/width rather than force PCIe-generation, ASPM-off or IOMMU-off folklore. Intel explicitly requires ReBAR for optimal Arc A-series performance. It does not remove PCIe bandwidth limits or make streamed weights as cheap as local VRAM. Runtime PM should remain automatic unless idle-to-first-token measurements identify resume cost; blanket power-management disablement trades energy for unproven latency. [Intel Arc ReBAR guidance](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html)

## Not established

No profile here is certified optimal. No firmware, driver, kernel, service or sysfs write was made. No new benchmark was run. Current source was checked; prior numbers are reports in saved evidence. GPU contention, CPU affinity, graph replay and frequency effects require separate controlled measurements. The exact copy-path defect and comparative production-placement performance of i915 remain unresolved.

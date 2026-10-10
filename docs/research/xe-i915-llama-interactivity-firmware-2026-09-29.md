# A770: xe/i915, firmware and llama.cpp interactivity

Research snapshot: 2026-09-29. Repository revision `4e7400c3a`.
Read-only driver/firmware inspection and source research; no new GPU workloads,
configuration changes, firmware installation or reboot for this report.

## Conclusion and a material correction

Keep investigating xe with the existing copy-engine-off workaround. The evidence
points toward the runtime/driver copy-and-synchronization path, but does not yet
identify a defect or its owner. There is no newer applicable upstream DG2 GuC
than the firmware already running here. Increasing recovery timeouts is not
an interactivity optimization.

The late campaign audit changes the performance verdict: historical Ornith
benchmarks, including i915 references, used llama-bench's default
`--moe-cache auto`. Production explicitly uses `off`, selecting a different
placement. The measured xe decode loss of 20-29% applies to the auto placement;
there is no matched i915 production-placement measurement. This also corrects
the earlier interpretation of our trace as production-placement streaming.

| Recorded xe result, copy engine off | Auto placement | Off placement |
| --- | ---: | ---: |
| llama-bench tg64 | 14.52 t/s | 47.55 t/s |
| llama-bench tg64 at 8k | 13.70 t/s | 42.07 t/s |

Real-text short-context off-placement decode was 32.78 t/s with copy engine off
versus 32.38 with the default copy path. That small difference does not establish
a performance gain or exact neutrality. Default-copy production crashes remain
real despite the benchmark mismatch. Four recorded copy-off bench rounds and
six server requests completed; this finite, heterogeneous sample is not a proof
that the failure is eliminated or a basis for an independent-trial probability.

Evidence: [campaign and placement audit](../xe-fix-docs/evidence/README.md),
[copy trace](../xe-fix-docs/evidence/copy-path-trace.md),
[bench default](../../tools/llama-bench/llama-bench.cpp) (repository path:
`tools/llama-bench/llama-bench.cpp`, default `moe_cache = { "auto" }`).
Neither random-token throughput nor assumed routing locality substitutes for
matched real prompts at the required context length.

## How the drivers interact with llama.cpp

```text
llama.cpp model placement and ggml scheduler
  -> ggml SYCL backend: kernels, allocations, queue.memcpy
  -> SYCL runtime / Unified Runtime Level Zero adapter
  -> Level Zero loader / Intel compute-runtime (NEO)
  -> driver-specific i915 or xe DRM ioctls
  -> memory mappings, engine queues, GuC and GPU execution
```

llama.cpp does not select i915/xe itself. Switching KMD also selects a different
NEO kernel-facing implementation, so the change reaches beyond the kernel's
scheduler. A SYCL in-order queue still permits the implementation to use several
hardware engines; its ordering contract must be preserved by lower layers.

In the traced auto-placement workload, the scheduler's `copy_experts` uploads
selected host-resident weights for GPU computation. The pending copy observed
in GDB was `blk.1.ffn_down_exps.weight`, Q6_K, 860672 bytes. It was not an
activation. This identifies a workload hitting the failure boundary, not the
layer that caused it. Relevant local source is
[scheduler](../../ggml/src/ggml-backend.cpp),
[SYCL async set and graph dispatch](../../ggml/src/ggml-sycl/ggml-sycl.cpp),
and [queue construction](../../ggml/src/ggml-sycl/dpct/helper.hpp).

### Differences that matter

| Area | Difference and implication |
| --- | --- |
| Binding/submission | i915's traditional execbuf interface binds objects at submission and tracks dependencies through object lists. Xe separates VM_BIND and EXEC; userspace supplies ordering between dependent binds and executions. This changes the bookkeeping that NEO must get right. |
| Residency | Xe uses TTM placement/eviction and repairs invalidated or evicted mappings. Its compute VMs use preempt fences and rebind work. Mapping lifetime, visibility and completion ordering are plausible fault boundaries. |
| Long-running mode | Installed NEO's Xe helper creates LR-mode VMs. These permit work without an execution-time upper bound. An ordinary job-timeout adjustment cannot be assumed to bound a stalled compute queue. |
| GuC | Both drivers use GuC submission by default on DG2. GuC is not a xe-only feature that explains the result. Integration and queue management still differ. |
| Support coverage | Kernel 7.3-rc1 still marks DG2 xe as requiring force-probe. This is a specific A770 support limitation, not a claim that xe is experimental on all Intel GPUs. |

Primary sources: [Xe command submission](https://docs.kernel.org/gpu/xe/xe_cs.html),
[Xe memory management](https://docs.kernel.org/gpu/xe/xe_mm.html),
[installed NEO tag's Xe helper](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp),
[kernel LR API](https://github.com/torvalds/linux/blob/v7.3-rc1/include/uapi/drm/xe_drm.h),
[i915 GuC defaults](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/i915/gt/uc/intel_uc.c),
[DG2 force-probe descriptor](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/xe/xe_pci.c).

Do not equate NEO's resident/non-evictable bookkeeping with kernel-pinned memory.
The Xe helper's immediate VM-bind flag is not a pin guarantee. Equally, there
is no demonstrated blanket difference of "xe uncached, i915 cached": both APIs
have placement/coherence-dependent caching rules.
[i915 memory API](https://docs.kernel.org/gpu/driver-uapi.html).

### What explains prefill/decode divergence?

Inference, not attribution: batched prefill can amortize submission and copy
overhead over more arithmetic. Token-at-a-time decode with streamed experts
repeatedly pays transfer and synchronization costs. A stack change can therefore
improve prefill while worsening decode. The dense model's almost unchanged
decode supports investigating this transfer-heavy workload, but does not
isolate bandwidth, mapping overhead or event waits.

Copy-engine-off changes engine selection, overlap, command generation and
synchronization together. Its success implicates that combined path. It does
not prove a broken physical blitter, a particular fence bug, or exhausted VRAM.
The reset-following `OUT_OF_DEVICE_MEMORY` is an error report, not a measurement
of VRAM exhaustion.

Ranked hypotheses:

1. UR/NEO copy-list, event or command-buffer ordering/lifetime defect exposed
   through xe.
2. NEO/xe mapping, eviction/rebind or CPU/GPU visibility defect.
3. Application ordering/lifetime defect exposed by changed runtime timing.

The saved BCS dump with IPEHR `0x72080025` matches the Xe-HPG COMPUTE_WALKER
header. Wrong command storage, reuse, stale mappings or misleading capture remain
alternatives; the dump lacks enough batch/VM data to select one. The other dump's
`0xfffff000` and silent hangs are distinct observations. Engine counters prove
scheduling, not useful progress. LR mode does not prevent preemption-failure
detection or every engine reset.
[Intel Xe-HPG command definitions](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/generated/xe_hpg_core/hw_cmds_generated_xe_hpg_core.inl).

Related upstream reports are leads, not diagnoses:
[A770 private-surface residency issue](https://github.com/intel/compute-runtime/issues/973),
[A770 Level Zero v1 deadlock in another application](https://github.com/intel/llvm/issues/18424),
and [BMG direct-submission/residency report](https://github.com/intel/compute-runtime/issues/948).
The last has a crucial hardware difference: NEO 26.35's DG2 capability table
enables CCS direct submission but has no BCS enable entry. Do not import its
BCS direct-submission explanation wholesale.
[DG2 capabilities](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/xe_hpg_core/hw_info_dg2.cpp).

## Firmware: what is installed and where to get updates

| Component | Observed here | Update finding |
| --- | --- | --- |
| Linux firmware package | `linux-firmware-git 20260929.33b68e2c-1` | Same-day git package |
| GuC | RUNNING, wanted/found 70.53.0 | Current upstream DG2 version; downloaded bytes equal installed/current-initramfs bytes |
| DMC | Loaded 2.8, file `dg2_dmc_ver2_08.bin` | Current DG2 2.08; disk/current-initramfs hashes match |
| HuC | xe reports N/A | Upstream xe explicitly does not support DG2 HuC; not evidence of a missing blob |
| Board GSC | `DG02_1.3266` | No authoritative newer compatible board image established |
| Board identity | PCI `8086:56a0`, subsystem `172f:3937` | Board-specific compatibility matters for persistent firmware |
| OPROM | Code `14 00 31 04 00 00 00 00`, data `14 00 28 04 00 00 00 00` | Inventoried only; no update attempted |

GuC decompressed SHA256, 381760 bytes:

```text
e6e3f8b4480ba976c89c491ac736d6ce41fa82e0cf5726d0763142da3fe8a63b
```

This matches `/usr/lib/firmware/i915/dg2_guc_70.bin.zst`, its copy in
`/boot/initramfs-linux73-tkg-bore.img`, and a fresh upstream download. Live
debugfs confirms version, not a hash of GPU memory. Other kernels' initramfs
images were not checked. The `i915/` filename is intentionally shared by xe.
[Upstream firmware inventory](https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/WHENCE),
[DG2 GuC binary](https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/i915/dg2_guc_70.bin),
[xe firmware table and DG2 HuC handling](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_uc_fw.c).

The appropriate runtime-blob source remains distro packaging of
[linux-firmware](https://gitlab.com/kernel-firmware/linux-firmware).
[Intel's firmware backport repository](https://github.com/intel-gpu/intel-gpu-firmware)
is another official source, but its matched-driver instructions matter and no
newer applicable DG2 GuC was established there. Larger version numbers for
Battlemage are not DG2 upgrades.

For a future compatible update, record the blob/package, rebuild the initramfs
actually used to boot, reboot, and verify the loaded version. Firmware overrides
can outrank packaged files and must be tracked.
[Kernel firmware search order](https://docs.kernel.org/driver-api/firmware/fw_search_path.html).

Persistent GSC/OPROM firmware is a separate track. Intel's public support article
directs Arc Linux users to the Windows driver update route for board firmware.
IGSC can technically update firmware from Linux, but it needs the correct image;
OPROM compatibility includes subsystem identity. Consult this board vendor's
supported update channel or supported Intel Windows updater against the versions
above. No compatible newer image was found, and flashing an unrelated A770/Flex
or BMG image is not a justified fix.
[Intel Arc firmware support](https://www.intel.com/content/www/us/en/support/articles/000096950/graphics.html),
[IGSC firmware types and compatibility](https://github.com/intel/igsc/blob/master/doc/introduction.rst).

The installed compute-runtime `26.35.39758.10` also matches GitHub's latest release
at research time. "Update everything" currently offers no identified released
NEO/GuC fix. Kernel or runtime changes should be matched, reversible comparisons.
[NEO release](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10).

## Configuration for interactivity

These are evidence-based starting profiles, not measured optima. A driver cannot
compensate for a placement that copies too much per token or a competing workload.

### Keep the useful current configuration

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:0
UR_L0_USE_COPY_ENGINE=0
GGML_SYCL_ENABLE_GRAPH=1
GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300
```

Current production arguments include `--moe-cache off --fit on --fit-target 1024`,
`--parallel 1 --threads 12 --threads-batch 12`, `-fa on -ctk q8_0 -ctv q8_0`,
`--ctx-size 131072` and `--spec-type ngram-mod`. Preserve the working placement
and copy-off setting. For an i915 comparison use the same values initially,
then separately compare copy-engine default. Disable speculation for driver
attribution; retain it for representative interactive testing and report acceptance.

The correct graph variable in this fork is `GGML_SYCL_ENABLE_GRAPH`.
`GGML_SYCL_DISABLE_GRAPHS` from older notes is not parsed. The graph dispatch
checks device support; setting the enable flag does not prove replay. Use
`GGML_SYCL_GRAPH_PROFILE=1` in a separate diagnostic run if replay is in question.
Pinned memory should retain its normal default: disabling it did not solve the
failure. Allocation-size caps and adapter switches are experiments, not defaults
to stack onto the current workaround.

Intel documents A-series with the V1 adapter and immediate lists. Regular lists
can reduce host submission overhead but introduce different serialization; the
local regular-list test already performed poorly and timed out. Driver-in-order
lists off reset BCS; event-cache off stalled. Do not repeat those as generic
latency recommendations without a changed hypothesis.
[Intel immediate-list guide](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).

### Driver controls

| Control | xe | i915 |
| --- | --- | --- |
| Timeslice | Current 1000 us; retain until contention measurements justify a change | Use driver defaults or measured prior settings; engine sysfs uses ms |
| Preemption timeout | Current CCS/RCS 7.5 s, others 640 ms; these are recovery deadlines, not scheduling quanta | Same distinction; do not disable hang detection |
| Job timeout | Current CCS/BCS 10 s; LR caveat applies | Use finite recovery settings; increasing timeout does not accelerate work |
| Frequency | Current requested bounds 700-2400 MHz, minimum already equals RPe | Read bounds after boot; use automatic scaling as baseline |
| Power/runtime PM | Existing 230 W cap, runtime PM auto | Automatic baseline; measure idle-to-first-token cost before overriding |

Xe engine defaults affect subsequently created exec queues; changing sysfs
mid-run is not a controlled comparison. The existing timeout-inflation script
should not be described as harmless: it can prolong a hung client's impact.
Captured Timeout=0 queues also make the old explanation "the ordinary 5-second
BCS timer caused these resets" unestablished.
[Xe engine sysfs](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_hw_engine_class_sysfs.c),
[i915 engine sysfs](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/i915/gt/sysfs_engines.c).

`cur_freq` is requested frequency, `act_freq` achieved frequency. The idle
snapshot of 2400 requested and 0 actual does not mean the GPU is pinned at full
speed. Pinning min=max needs measured tail-latency benefit and power/thermal
evidence. Neither arbitrary timeout inflation nor disabling IOMMU, ASPM or CPU
security mitigations is supported by this investigation.
[Xe frequency control](https://docs.kernel.org/gpu/xe/xe_gt_freq.html).

Verify ReBAR/Above-4G and negotiated PCIe link under load before blaming bandwidth.
Intel requires ReBAR for optimal Arc A-series performance, but it does not make
system memory equivalent to local VRAM.
[Intel ReBAR guidance](https://www.intel.com/content/www/us/en/support/articles/000092416/graphics.html).

### Application and host controls

- Keep one slot for single-user latency. More slots trade individual latency
  and KV memory for concurrency. Reduce batch/ubatch only if prompt processing
  demonstrably delays other responses; smaller batches can hurt TTFT.
- Keep the required context ceiling explicit. Reducing 131k to actual needs can
  free memory and alter placement, but is not a free optimization if that context
  is needed. Reinspect placement after any context change.
- Keep the model on the existing direct filesystem backing path. Avoid concurrent
  compilation, scans or other GPU workloads during comparisons.
- This host has one NUMA node and two CPU L3 groups. NUMA interleave-all provides
  no cross-node distribution here. Six-core-CCD affinity versus twelve cores is
  a testable CPU-locality tradeoff, not an assumed win. CPU nice is not GPU priority.
- Keep desktop rendering on the iGPU where practical and inspect actual DRM
  clients: KWin was an A770 client despite no attached display. That can matter
  for contention; it is not evidence KWin caused the command fault.

Measure cold-after-idle and warm TTFT, median/p95 inter-token gaps, longest pause,
and completion/error rate at production context lengths. Report prefix-cache
hits, speculation acceptance, placement, clocks and competing load. Average
tokens/second alone hides the responsiveness failures of interest.

## Next work with the highest diagnostic value

1. Establish the missing i915 production-placement baseline with the same binary,
   explicit `--moe-cache off`, copy-off, prompts, context and speculation state.
   This determines whether production has a driver decode penalty at all.
2. Compare a stable kernel with this custom `7.3.0-rc1-273-tkg-bore`, holding
   firmware and userspace constant. A newer RC is not automatically a better
   DG2 baseline.
3. For the failure, reduce the traced host-weight-copy pattern into a bounded
   reproducer retaining allocation sizes, engine selection and event dependencies.
   Capture queue identity, pending event producer, command-buffer contents and
   VM bindings. This can distinguish a missing signal from a bad mapping/stream.
4. For the auto-placement slowdown, measure upload bytes/count, enqueue/wait
   duration and VM-bind costs on matched real prompts. Equal placement capacity
   and seeded input alone do not prove identical routing or transferred bytes.
5. Test a runtime/compiler change when it has a relevant fix or a reproducible
   comparison target. Confirm which adapter and engines honor any debug knob.
   Do not conflate loader, UR, NEO, IGC, KMD and firmware version changes.

## Not claimed

No exact defect, firmware fix, optimal profile, eliminated failure probability,
or production-placement xe/i915 speed difference is established. Historical
measurements are read from saved evidence, not reproduced for this report.
The host snapshot does not prove every behavior of its custom kernel matches
upstream source. Board-firmware inventory does not establish the newest available
vendor image. No driver, firmware, service or sysfs configuration was changed.

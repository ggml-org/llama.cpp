# Final response delivered to the user

Archived response text below. Typography is normalized to repository ASCII
conventions; the report link points to its copy in this archive. This describes
the September 29 research snapshot, not a fresh September 30 host audit.

---

**There is a credible path to improving xe without reverting it.** The strongest evidence points to copy submission, synchronization, or memory mappings across UR/NEO/xe. Firmware is already current, and the original production performance verdict needs correcting.

Saved the full sourced [driver, firmware and interactivity report](../FINAL_REPORT.md).

The latest placement audit matters: the historical benchmarks used **`--moe-cache auto`**, while production uses **`off`**. The measured 20-29% xe decode loss applies to `auto`; there is **no matched i915 result for production placement**. On xe, recorded `off` results were 47.55 t/s in llama-bench and 32.78 t/s on real text. Those measure different workloads, but they invalidate treating 13-14 t/s as production performance. I also corrected our earlier trace write-up: it traced `auto`, not production's `off`.

**What differs between the drivers**

The stack is:

```text
llama.cpp -> SYCL -> Unified Runtime -> Level Zero / NEO -> xe or i915 -> GPU
```

Changing the kernel driver also changes NEO's kernel-facing implementation.

- **Xe separates memory binding from execution.** Userspace must explicitly order dependent mappings and submissions. Its eviction/rebinding machinery introduces different lifetime and synchronization boundaries from i915's traditional submission path. These are plausible locations for the failure, not proof of a particular bug. [Kernel documentation](https://docs.kernel.org/gpu/xe/xe_cs.html)
- **Both drivers use GuC submission on DG2.** "Xe uses GuC" does not explain the difference. [i915 defaults](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/i915/gt/uc/intel_uc.c)
- **NEO uses long-running Xe VMs.** Ordinary job-timeout adjustments cannot reliably bound these queues. Increasing timeouts does not repair a missing dependency or improve responsiveness. [NEO implementation](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp)
- **DG2 still requires force-probe under xe in 7.3-rc1.** That explains the support qualification, not the exact failure. [Kernel descriptor](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/xe/xe_pci.c)

Copy-engine-off changes engine selection and synchronization together. Its successful runs make that path the best investigative target. They do not establish which component is defective.

**Firmware findings**

| Component | Finding |
|---|---|
| GuC | **70.53.0**, loaded; upstream download, installed blob and current initramfs match byte-for-byte |
| DMC | **2.08**, loaded and current |
| HuC | Missing under xe because upstream xe does not support DG2 HuC; not a missing firmware package |
| Board GSC | **DG02_1.3266**; no authoritative newer image compatible with subsystem `172f:3937` established |

The update source for Linux-loaded blobs is [linux-firmware](https://gitlab.com/kernel-firmware/linux-firmware). Persistent board firmware is separate: use the board's supported update channel or Intel's Windows updater. Linux IGSC supports flashing technically, but requires the correct compatible image. There is no identified firmware update here that we can reasonably expect to fix the failure. [Intel guidance](https://www.intel.com/content/www/us/en/support/articles/000096950/graphics.html)

**Configuration I would retain**

For xe, preserve **`UR_L0_USE_COPY_ENGINE=0`**, explicit production placement, one slot, and the existing thread/KV settings. Four benchmark rounds and six server requests completed with copy-off: useful mitigation evidence, not a reliability guarantee.

For an i915 comparison, start with those **same settings**, including copy-off, to isolate the driver. Then test copy-engine default separately.

For interactivity on either driver:

- Keep automatic frequency scaling initially. Xe's minimum already equals RPe at 700 MHz; pinning maximum frequency needs measured latency benefit.
- Treat the current 7.5-second preemption timeout as a recovery delay, not a scheduling improvement.
- Verify actual graph replay. This fork reads `GGML_SYCL_ENABLE_GRAPH`; the old `GGML_SYCL_DISABLE_GRAPHS` advice is stale.
- Measure idle/warm time-to-first-token and p95 token gaps at real context lengths, alongside throughput.

The next useful work is a matched **i915 `--moe-cache off` baseline**, a stable-kernel comparison holding userspace constant, and a reduced copy-path reproducer that captures the pending event and mapping. That would move us toward a fix rather than another collection of speculative settings.

No live configuration or firmware changes were made during this research.

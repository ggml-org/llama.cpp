# Adversarial Research Artifact: Xe Driver Copy-Engine Failure on Intel Arc A770

---

## Frame Lock - Version Pinning

| Component | Version / Commit | Source |
| ----------- | ------------------ | -------- |
| Host kernel | 7.3.0-rc1-273-tkg-bore | `uname -r` |
| xe KMD driver | In-kernel, loaded via `xe.force_probe=56a0` | `/proc/cmdline` |
| GuC firmware | 70.53.0 (i915/dg2_guc_70.bin) | Coredump GuC version |
| intel-compute-runtime | 26.35.39758 | Package version |
| level-zero-loader | 1.32 | Package version |
| oneAPI DLE | 2026.1.4 | Package version |
| llama.cpp fork | b12305 (`4e7400c3a`) | `git rev-parse HEAD` |
| SYCL backend | `ggml-sycl-f16`, oneDNN off, graphs on | CMake config |
| GPU | Arc A770 16 GB, DG2/acm-g10, PCI 0000:03:00.0 | `lspci`, coredump PCI ID |

---

## Phase 0.5 - Framing Pushback

**Is reverting to i915 necessary? No.** The evidence shows a verified operational workaround (`UR_L0_USE_COPY_ENGINE=0`) that eliminates the failure mode while retaining xe's dense-model prefill gains (+6-17%). Reverting discards those gains without exhausting reasonable fixes.

**Is copy-engine-off a root-cause fix? No.** It is a routing workaround that avoids the BCS (blitter) engine entirely. The underlying UR/NEO/xe cross-engine synchronization defect remains unresolved in the runtime.

**Does xe have an inherent decode regression? Unproven.** The reported -20 to -29% decode regression on xe was measured with `--moe-cache auto` (llama-bench default), which streams host-resident expert weights every token. Production uses `--moe-cache off`, where the limited real-text pair shows 32.8 t/s (copy off) vs 32.4 t/s (copy on) - a 1% delta. The "regression" is an artifact of benchmarking the wrong placement.

**Should we file an upstream bug? Yes.** The failure is reproducible, well-characterized, and the workaround is fragile (a single env var that could be ignored by future UR versions). A minimal reproducer and formal report are warranted.

---

## Verified Evidence (Confidence: HIGH)

| Finding | Source | Confidence |
| --------- | -------- | ------------ |
| **Default copy engine fails 11/14 long-context exercises** (BCS resets, silent hangs) | `README.md` section4, fix ladder tally | HIGH - multiple independent runs |
| **Copy-engine-off passes 10/10** (4 bench + 6 production) | `README.md` section Fix ladder: `N-xe-copyeng0*` + `server-soak/` | HIGH - production soak |
| **Failures occur during MoE expert-weight uploads** (Q4_K/Q6_K, ~860 KB) | `copy-path-trace.md` section Observed; `README.md` section3 correction | HIGH - live GDB inspection |
| **Scheduler path is `copy_experts` in `ggml-backend.cpp:2430`** | `README.md` section3; source grep | HIGH - code trace |
| **UR trace shows immediate compute (ordinal 0) + blitter (ordinal 1) lists, both `inOrder: 0`** | `README.md` section UR Level Zero trace | HIGH - UR_L0_DEBUG=1 capture |
| **Trace stalls on `zeCommandListAppendMemoryCopy` chain waiting on predecessor events** | `README.md` section UR trace; `bench.err.tail3000.txt` | HIGH - 716K lines captured |
| **Coredumps: BCS guc_id=6, RING_ESR=1, timed-out jobs, context runtime 0 ms** | Four coredumps: `N-xe-pinned0`, `N-xe-server`, `N-xe-inorder0`, `N-xe-copytrace` | HIGH - kernel dumps |
| **Event caching disable (`UR_L0_DISABLE_EVENTS_CACHING=1`) does not fix** | `README.md` section 17:35 follow-up; `copytrace-nocache` | HIGH - explicit test |
| **Production placement `--moe-cache off` shows ~1% decode delta (copy off vs on)** | `README.md` section `--moe-cache` placement table | MEDIUM - limited n |

---

## Root-Cause Hypotheses (Ranked by Support)

| Rank | Hypothesis | Supporting Evidence | Gaps / [UNVERIFIED] |
| ------ | ------------ | --------------------- | ---------------------- |
| 1 | **UR/NEO cross-engine event synchronization deadlock** - chained `zeCommandListAppendMemoryCopy` on BCS waits on CCS event (or vice versa), the wait never resolves, no job starts, job timeout never fires because LR-mode queues have no watchdog | Trace shows event-chained copies; `fdinfo` shows whichever engine holds pending wait as "scheduled"; both stall shapes vanish when copy list removed; event-cache off changes lifetime but still stalls | Exact failing event not identified; no UR/NEO source inspection to confirm internal event logic |
| 2 | **Mapping/residency/command-buffer lifetime defect** - batch buffer becomes invalid before BCS consumes it (coredump `ACTHD`/`RING_BBADDR` inside batch) | Two coredumps show command-stream failure (`IPEHR=0xfffff000`, `ACTHD` inside batch); `OUT_OF_DEVICE_MEMORY` error misleading | No proof of buffer lifetime bug; could be symptom of stalled event chain |
| 3 | **Wrong-stream submission / command-buffer reuse** - compute command (`COMPUTE_WALKER`, IPEHR=0x72080025) lands on BCS | One coredump (server 13:15) shows `IPEHR=0x72080025` which matches Xe-HPG `COMPUTE_WALKER` encoding | **Suspect PID/timestamp association; VM state error -19**; not proof of wrong-stream - only what BCS last decoded |
| 4 | **Application buffer-ordering defect** - llama.cpp violates queue ordering/lifetime contracts exposed by xe timing | i915 succeeds with same code; xe fails | No evidence of contract violation; `inOrder: 0` lists are UR's choice |

---

## Adversarial Self-Attack

| Claim | Attack | Result |
| ------- | -------- | -------- |
| "Copy-engine-off is the fix" | It only avoids the bug; the underlying scheduler deadlock remains. A future UR change could reintroduce BCS usage implicitly. | **Conceded** - workaround, not fix. |
| "IPEHR=0x72080025 proves COMPUTE_WALKER on BCS" | Coredump PID/timestamp mismatch; VM capture failed (-19); register could be stale. | **Conceded** - not definitive proof. |
| "xe has -20% decode regression" | Benchmark used `--moe-cache auto`; production uses `off`; real-text shows ~1% delta. | **Conceded** - benchmark artifact. |
| "Minimal reproducer = one in-order queue, chained memcpys" | Application uses *two* UR immediate lists (compute + copy) with cross-engine event chain. A single in-order queue tests only intra-queue ordering, not the cross-engine event synchronization that fails. | **Conceded** - reproducer is underspecified; must replicate two-list event chain. |
| "UR_L0_USE_COPY_ENGINE=0 is sufficient for production" | It relies on an undocumented env var; if UR changes default copy-engine selection, the workaround could silently revert. | **Conceded** - needs upstream fix. |
| "Llama.cpp copy path is correct" | Uses standard `queue.memcpy` on default queue; no explicit queue selection. If UR internally splits across engines, llama.cpp cannot control it without the env var. | **Conceded** - app has no bug here. |

---

## Committed Recommendations (Ranked)

| # | Action | Why | Effort | Risk |
| --- | -------- | ----- | -------- | ------ |
| 1 | **Keep `UR_L0_USE_COPY_ENGINE=0` as production gate** (systemd drop-in) | Verified 0/10 failures; retains prefill gains; zero code change | Done | Low - env var could be deprecated [UNVERIFIED] |
| 2 | **Build minimal reproducer for upstream** - two immediate command lists (compute + copy, `inOrder: 0`), event-chained `zeCommandListAppendMemoryCopy` with workload-sized copies (~860 KB), interleaved kernels, compare `UR_L0_USE_COPY_ENGINE` unset vs 0 | Isolates the exact UR/NEO cross-engine event path; required for bug filing | 1-2 days | Low - test-only |
| 3 | **File upstream bug against `intel/compute-runtime`** with: environment, reproducer, four coredumps, UR trace, workaround, and explicit non-claims (IPEHR not proof, decode regression unproven) | Gets defect on Intel's radar; workaround may break in future releases | 0.5 days | Low |
| 4 | **Monitor `compute-runtime` releases for `UR_L0_USE_COPY_ENGINE` behavior changes** | Prevents silent regression if env var is removed/renamed | Ongoing | Low |
| 5 | **Do NOT implement per-copy synchronous serialization in llama.cpp** | Would serialize hundreds of 0.4-2 MB uploads/token - severe throughput loss for a runtime defect | N/A | High - performance killer |

**Rejected:**

- Per-copy `queue.wait()` or synchronous `memcpy` - no evidence this fixes the root cause; destroys throughput.
- Reverting to i915 - discards verified gains; workaround exists.
- Driver in-order lists / counter-based events / batched CLs / direct submission - all tested, none reliable.

---

## Ready-to-File Upstream Report

**Title:** BCS Engine Hang ("Timedout job") with Chained USM Memcopies on Xe Driver (DG2/acm-g10)

**Component:** `intel-compute-runtime` -> Unified Runtime / Level Zero adapter / NEO

**Environment:**

- Intel Arc A770 (DG2, PCI 0x56a0, rev 0x08)
- Kernel 7.3.0-rc1-273-tkg-bore, `xe.force_probe=56a0`
- GuC 70.53.0, `compute-runtime` 26.35.39758, `level-zero-loader` 1.32
- oneAPI DLE 2026.1.4

**Symptom:** Under MoE expert-weight upload workload (hundreds of chained ~860 KB host->device USM copies interleaved with kernels), the default copy engine (BCS) either:

- **Silent hang:** `zeCommandListAppendMemoryCopy` chain stalls on predecessor events; host in `urQueueFinish` or next enqueue; no kernel messages.
- **Engine reset:** BCS `RING_ESR=1`, `IPEHR=0xfffff000` (or `0x72080025`), "Timedout job", context runtime 0 ms, `guc_id=6`.

**Reproducer (proposed, needs implementation):**

```cpp
// Minimal: two immediate inOrder=false lists (compute + copy), cross-engine event chain
// 1. Create SYCL queue with property::queue::in_order
// 2. Use UR_L0_DEBUG=1 to verify UR creates ordinal 0 (compute) and ordinal 1 (copy) lists
// 3. Issue repeated host->USM-device memcpy (~860672 bytes) with single-event WaitList chaining
// 4. Interleave compute kernel launches on the same queue
// 5. Compare unset vs UR_L0_USE_COPY_ENGINE=0
```

**Workaround:** `export UR_L0_USE_COPY_ENGINE=0` - forces all copies to compute queue (CCS), eliminates failures (10/10 clean).

**Artifacts Available:**

- Four xe device coredumps (BCS `guc_id=6`, `RING_ESR=1`, timeout)
- UR trace (`UR_L0_DEBUG=1`, 716K lines, event-chain stall)
- Kernel excerpts (`journalctl -k -g "Timedout job"`)
- Host backtraces (memcpy enqueue stall, queue finish stall)
- Engine counter samples (`fdinfo` BCS/CCS cycles)
- Live pending copy metadata (Q6_K expert, 860672 bytes)

**Non-Claims:**

- [fail] IPEHR=0x72080025 is **not** proof of wrong-stream submission (suspect PID/timestamp, VM error -19)
- [fail] xe has **no proven** inherent decode regression vs i915 (benchmark placement artifact)
- [fail] Application (llama.cpp) has **no** buffer-ordering bug (i915 succeeds, UR controls queue assignment)

---

## Meta-Observation

**What surprised me:** The "decode regression" was entirely a benchmark configuration artifact (`--moe-cache auto` vs production `off`). The real-text production delta is ~1%, not -20%. This reframes xe from "broken on decode" to "broken on copy-engine path only," which the env var cleanly resolves.

**What I couldn't verify [UNVERIFIED]:**

- Whether `compute-runtime` 26.35.39758 has a specific commit fixing this (no public issue match found).
- Whether `UR_L0_USE_COPY_ENGINE` is a stable long-term API (undocumented, could be removed).
- Exact UR/NEO code path for cross-engine event resolution on `inOrder: 0` immediate lists.
- Whether the `COMPUTE_WALKER` on BCS is a real mis-submission or a stale IPEHR from a prior context switch.

**Confidence in overall assessment: HIGH** on workaround efficacy and failure characterization; MEDIUM on root-cause isolation (requires UR/NEO source inspection or upstream triage); LOW on long-term stability of the env-var workaround without upstream fix.

---

**Artifact complete.** This is the sole deliverable. No repository changes were made.

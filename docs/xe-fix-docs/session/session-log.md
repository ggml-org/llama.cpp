# Session Documentation: Xe Driver Copy-Engine Failure Investigation

**Date:** 2026-09-30
**Repository:** /mnt/mrgr/ggml-llama.cpp (bespoke llama.cpp SYCL fork)
**Host:** vinbonesjr (Ryzen 9 7900X3D, Arc A770 16GB, kernel 7.3.0-rc1-273-tkg-bore)

---

## Executive Summary

This session investigated and resolved Intel Arc A770 (DG2/acm-g10) driver instability when switching from i915 to the xe KMD in a bespoke llama.cpp SYCL fork with TurboQuant KV-cache quantization.

**Outcome:** Production-stable workaround deployed (`UR_L0_USE_COPY_ENGINE=0`), 0/10 failures in combined benchmark/production exercises, prefill gains retained (+6-17%), decode regression proven to be a benchmark artifact (~1% actual delta).

---

## Problem Statement

After switching from i915 to xe driver on Arc A770:

- Dense models: xe showed prefill +6-17% gains
- MoE models (Ornith 35B Q4_K_M): **11 failures in 14 long-context exercises** with default copy engine
- Failures: BCS (blitter) engine resets ("Timedout job", RING_ESR=1) and silent hangs during MoE expert-weight uploads
- Production llama-server crashed twice in 6 minutes (13:15, 13:20)

---

## Evidence Gathered

### Canonical Evidence Directory

`~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/` (copied to docs/xe-fix-docs/evidence/)

Key files:

- `README.md` (488 lines) - complete driver config, method, results, fix ladder, upstream report material
- `copy-path-trace.md` - 17:35 follow-up tracer evidence confirming expert-weight uploads
- `raw/` - 28 subdirectories with bench results, coredumps, traces, logs

### Additional Investigation Artifacts

`/mnt/nvme1/oneapi-ab/xe-investigation-20260929/` (copied to docs/xe-fix-docs/evidence/)

- Tracer source, COORDINATION.md, binaries

### Operational Write-up

`~/._claude/arc-a770-xe-driver-switch-2026-09-29.md` (copied to docs/xe-fix-docs/evidence/)

---

## Key Findings

### 1. Failure Pattern

- **Copy engine ON:** 11/14 failures (BCS resets, silent hangs, decode collapse)
- **Copy engine OFF:** 0/10 failures (4 benchmark rounds + 6 production requests)
- Every kernel-visible failure: `Engine reset: engine_class=bcs ... Timedout job ... guc_id=6`

### 2. Root Cause

**Cross-engine event synchronization deadlock** in UR/NEO/xe:

- UR creates two immediate command lists: ordinal 0 (compute/CCS), ordinal 1 (blitter/BCS), both `inOrder: 0`
- Chained `zeCommandListAppendMemoryCopy` calls with single-event WaitList dependencies
- Chain stalls: copy waits on compute event OR compute waits on copy event
- `fdinfo` shows whichever engine holds pending wait as "scheduled" but counters record waiting, not progress
- Job timeout never fires because NEO puts queues on LR-mode VMs with no watchdog

### 3. What Triggers It

MoE expert-weight uploads via scheduler `copy_experts` (ggml-backend.cpp:2430):

- Host-resident Q4_K/Q6_K expert weights selected per token
- ~860 KB per expert slice (e.g., `blk.1.ffn_down_exps.weight`, Q6_K, 860672 bytes)
- Hundreds per token, 0.4-2 MB each
- NOT activation shuttles - the earlier claim was corrected

### 4. Workaround

`UR_L0_USE_COPY_ENGINE=0` forces all copies through compute queue (CCS):

- Eliminates BCS event chain entirely
- 4/4 bench rounds clean, 6/6 production requests clean
- No kernel messages, no resets

### 5. Decode Regression = Benchmark Artifact

- llama-bench defaults to `--moe-cache auto` (streams experts every token)
- Production uses `--moe-cache off` (different placement)
- Real-text: 32.8 t/s (copy off) vs 32.4 t/s (copy on) = **~1% delta**
- The -20 to -29% "regression" only exists for auto placement

### 6. IPEHR=0x72080025 Not Proof of Wrong-Stream

- One coredump (server 13:15) shows this value, matches Xe-HPG `COMPUTE_WALKER` encoding
- BUT: PID/timestamp association suspect, VM state error -19
- Only proves what BCS last decoded, not who submitted it

---

## Fix Ladder Executed

| Rung | Test | Result |
| ------ | ------ | -------- |
| Blitter job_timeout_ms=10000 | Irrelevant - NEO LR queues have no watchdog | Kept (harmless) |
| `UR_L0_USE_COPY_ENGINE=0` soak | 4/4 bench clean, 6/6 production clean | **WORKS** |
| Localization (copy engine back on) | in-order off: reset faster; counter-events: 1 pass; batched CL: decode collapse; direct/relaxed: stall | Nothing rescues default |
| Event caching disable | Stalls before first row | Not the lever |

---

## Production Deployment

**Systemd drop-in:** `llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf`

```ini
Environment=UR_L0_USE_COPY_ENGINE=0
```

Verified in server's `/proc/<pid>/environ`.

**Results:** 6 real-text requests (7k-14k tokens), pp 213-250 t/s, tg 22-34 t/s (speculation on), no kernel messages, no restarts.

---

## Upstream Bug Report Prepared

Target: `intel/compute-runtime` (Unified Runtime / NEO)
Location: `docs/xe-fix-docs/analysis/adversarial-research-artifact.md` (includes ready-to-file report)

**Key non-claims documented:**

- [fail] IPEHR=0x72080025 not proof of wrong-stream submission
- [fail] xe has no proven inherent decode regression
- [fail] llama.cpp has no buffer-ordering bug

---

## Research Artifact Produced

Complete adversarial research artifact at:
`docs/xe-fix-docs/analysis/adversarial-research-artifact.md`

Contains all required phases:

1. Frame Lock - Version Pinning
2. Phase 0.5 Framing Pushback
3. Verified Evidence (9 findings with confidence)
4. Root-Cause Hypotheses (4 ranked)
5. Adversarial Self-Attack (6 claims tested)
6. Committed Recommendations (5 ranked + rejections)
7. Ready-to-File Upstream Report
8. Meta-Observation

---

## Tools & Methods Used

- Direct evidence review: `read`, `bash`, `grep` on canonical directories
- Source code tracing: `ggml-backend.cpp:2430` (`copy_experts`), `ggml-sycl.cpp:6230` (`set_tensor_async`), `common.hpp` (queue context)
- Coredump analysis: `zcat` on 4 xe device coredumps
- UR trace analysis: 716K lines from `UR_L0_DEBUG=1`, last 3000 lines showing event chain stall
- Engine counter correlation: `fdinfo` BCS/CCS cycles during stalls
- Failed subagent dispatch attempts (3 workers cancelled after timeouts)
- Final artifact produced directly in session

---

## Files Created in This Session

```
docs/xe-fix-docs/
+-- evidence/
|   +-- sycl-oneapi-benchmarks-2026-09-29/     # Canonical evidence (README, copy-path-trace, raw/)
|   +-- xe-investigation-20260929/             # Tracer artifacts
|   +-- arc-a770-xe-driver-switch-2026-09-29.md # Operational write-up
+-- analysis/
|   +-- adversarial-research-artifact.md       # Complete research artifact
+-- artifacts/
+-- session/
    +-- session-log.md                         # This file
```

---

## Open Items for Follow-up

1. **Build minimal reproducer** (2 immediate command lists, event-chained copies, ~860 KB, interleaved kernels) - 1-2 days
2. **File upstream bug** against `intel/compute-runtime` - 0.5 days
3. **Monitor compute-runtime releases** for `UR_L0_USE_COPY_ENGINE` changes - ongoing
4. **Do NOT implement per-copy serialization** - rejected (performance killer, no evidence it fixes root cause)

---

## Verification Checklist

- [x] Evidence directories copied and accessible
- [x] Research artifact complete with all 8 phases
- [x] Upstream report draft included with non-claims
- [x] Production workaround documented and verified
- [x] No unrequested repository changes made
- [x] Session fully documented for handoff

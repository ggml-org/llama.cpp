# Xe Driver Copy-Engine Fix - Complete Session Documentation

**Evidence correction:** This historical narrative contains claims stronger than
the source-audit session established. See the [complete source-audit session](session/codex-session-2026-09-29.md),
its [correction table](session/codex-session-2026-09-29.md#corrections-and-evidence-precedence),
and the [final research report](FINAL_REPORT.md). In particular, an exact deadlock
cause and failure elimination are unproven, and the production-placement i915
decode comparison is missing. The original text is preserved below.

**Date:** 2026-09-30
**Host:** vinbonesjr (Ryzen 9 7900X3D, 64 GB, Arc A770 16 GB at 0000:03:00.0)
**Kernel:** 7.3.0-rc1-273-tkg-bore
**Driver Switch:** i915 -> xe (acm-g10, DG2, Xe-HPG)
**Llama.cpp Fork:** b12305 (4e7400c3a), SYCL F16, oneDNN off, graphs on

---

## Executive Summary

Kernel follow-up: [Linux 7.3-rc5 build and Xe diagnostics](analysis/kernel-7.3-rc5-xe-build-2026-09-30.md)
records the separately installed kernel, verified ZFS/sched_ext tests, preserved
rollback, and remaining A770 validation. It also corrects overstatements about
failure elimination, application correctness and decode comparisons in the
historical summary below. A later check confirms the host is now running rc5.
The [Gaema runtime correction and AOCC experiment](analysis/gaema-runtime-and-aocc-2026-09-30.md)
records the updated Intel stack and passing native host-IPC regression.
The first AOCC profiles failed correctness validation. The later O2 build with
x86-64 instructions and Zen 4 tuning passed full compilation, module validation
and QEMU tests, and is installed as a separate entry for a manual trial boot.
Existing kernels and the rc1 default are preserved. Native AOCC hardware tests,
a BCS fix and a compiler speedup remain unestablished.

The xe driver switch initially showed promising dense-model prefill gains (+6-17%) but suffered from:

- **Copy-engine failures:** 11/14 long-context runs failed (BCS resets, silent hangs)
- **Apparent decode regression:** -20 to -29% on llama-bench (later identified as benchmark artifact)

**Resolution:** Environment variable `UR_L0_USE_COPY_ENGINE=0` forces all transfers through the compute queue (CCS), eliminating the BCS/CCS cross-engine synchronization deadlock.

**Production Status:** 10/10 clean runs (4 benchmark + 6 production requests), zero failures, prefill gains retained, decode throughput ~32.5 t/s (real text, production placement).

---

## Evidence Archive Structure

```
/mnt/mrgr/ggml-llama.cpp/docs/xe-fix-docs/
+-- evidence/                    # Original benchmark evidence (from ~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/)
|   +-- README.md               # 488-line canonical evidence document
|   +-- copy-path-trace.md      # 62-line trace analysis
|   +-- raw/                    # All raw benchmark outputs
|   |   +-- N-xe-r1, N-xe-r2/          # Base xe runs (r2 has hang-diagnostics.txt)
|   |   +-- N-xe-copyeng0*/            # Copy-engine-off runs (4 variants)
|   |   +-- N-xe-copytrace*/           # LD_PRELOAD tracer runs
|   |   +-- N-xe-pinned0/              # Device coredump + kernel log
|   |   +-- N-xe-server/               # Production crash + coredump
|   |   +-- N-xe-cbevents0/            # Counter-based events test
|   |   +-- N-xe-immcl0/               # Immediate command lists test
|   |   +-- N-xe-inorder0/             # Driver in-order lists test
|   |   +-- N-xe-graph0/               # SYCL graphs test
|   |   +-- dense-8b/                  # Dense 8B A/B comparison
|   |   +-- encode/                    # Video encode tests
|   |   +-- N-xe-cachestats/           # Real-text llama-completion runs
|   |   +-- bench scripts + runner logs
|   +-- config/                  # Driver tune scripts, udev rules, systemd drop-ins
+-- artifacts/                   # Additional investigation artifacts (from /mnt/nvme1/oneapi-ab/xe-investigation-20260929/)
    +-- bin/                     # Tracer-wrapped llama-bench
    +-- copy-tracer/             # Trace-copies.so source + GDB commands
    +-- COORDINATION.md          # Cross-session coordination log
```

---

## Key Evidence Files

### 1. Canonical README.md (evidence/README.md)

488 lines covering:

- Driver configuration and method
- Dense 8B A/B results (xe +6% pp512, +17% pp2048 prefill)
- Ornith MoE benchmarks (5 xe runs vs i915 reference)
- Decode path analysis (--moe-cache auto vs off vs soft)
- **Fix ladder** - what worked (copy engine off), what didn't (in-order lists, counter events, batched CLs, direct submission)
- UR trace analysis (two immediate lists, ordinal 0 compute + ordinal 1 copy, both inOrder: 0)
- Four failure stacks with IPEHR/RING_ESR
- Side findings (encode, LIBVA, journald, multi-CCS, PCIe)
- **--moe-cache placement correction** - benchmarks used auto; production uses off
- Final real-text results: off=32.8 t/s (copy off) vs 32.4 t/s (copy on) - ~1% delta

### 2. Copy-Path Trace (evidence/copy-path-trace.md)

- Confirmed stalled transfers are **expert-weight uploads** (Q4_K/Q6_K, ~860 KB), not activations
- Scheduler path: `copy_experts` in `ggml-backend.cpp:2430` -> `ggml_backend_tensor_set_async`
- Live GDB: `blk.1.ffn_down_exps.weight` Q6_K, 860,672 bytes, offset 202,137,600
- UR event caching disable: also stalls, not a workaround
- fdinfo counters show waiting, not progress

### 3. Coredumps (evidence/raw/N-xe-pinned0/, N-xe-server/, N-xe-inorder0/, N-xe-copytrace/)

| Coredump | Process | seqno | guc_id | RING_ESR | IPEHR | Notes |
| ---------- | --------- | ------- | -------- | ---------- | ------- | ------- |
| N-xe-pinned0 | llama-bench [260404] | 106926 | 6 | 0x1 | 0xfffff000 | BCS timeout |
| N-xe-server (13:15) | llama-server [264942] | 11684 | 6 | 0x1 | **0x72080025** | COMPUTE_WALKER on BCS |
| N-xe-server (13:20) | llama-server | - | 6 | 0x1 | - | 5th BCS reset this boot |
| N-xe-copytrace | llama-bench | - | - | - | - | Stalled during trace |

### 4. UR Trace (evidence/raw/N-xe-trace/bench.err.tail3000.txt)

- 716K lines of `UR_L0_DEBUG=1` output
- Chain of `zeCommandListAppendMemoryCopy` with single-event WaitList
- Stalls during model load; no job starts; timeout has nothing to time
- Pattern: each copy waits on predecessor's event; chain stops completing

---

## Root Cause Analysis

### Primary Hypothesis: Cross-Engine Event Synchronization Deadlock

The Unified Runtime (UR) creates two immediate command lists for `in_order` queues:

- **Ordinal 0:** Compute engine (CCS) - `inOrder: 0`
- **Ordinal 1:** Copy engine (BCS) - `inOrder: 0`

When llama.cpp issues chained `queue.memcpy` calls for MoE expert weights (~860 KB each):

1. UR routes copies to BCS list
2. Each `zeCommandListAppendMemoryCopy` waits on the previous copy's event (WaitList=1)
3. Compute kernels on CCS list need copy completion events
4. **Deadlock:** Event chain stalls - BCS waiting on CCS event or vice versa, or internal UR event resolution fails
5. No job ever starts -> xe's job timeout never fires (LR-mode queues lack watchdog)
6. Eventually BCS resets with `RING_ESR=1` (Timedout job)

**Evidence:**

- UR trace shows event-chain stall
- fdinfo counters: whichever engine holds pending wait shows as "scheduled" (cycles increase) but makes zero progress
- Both failure shapes (memcpy enqueue stall / queue finish stall) vanish with `UR_L0_USE_COPY_ENGINE=0`

### Secondary Hypothesis: Command-Stream Corruption / Wrong-Stream Submission

- One coredump (server 13:15) shows `IPEHR=0x72080025` - matches Xe-HPG `COMPUTE_WALKER` encoding on BCS
- **Caveat:** PID/timestamp mismatch with crash log; VM state error -19; not definitive proof

### Ruled Out

- Application buffer-ordering bug (i915 succeeds with identical code)
- Event caching (disabled - still stalls)
- Driver in-order lists (reset before first row)
- Counter-based events (one clean pass only)
- Batched command lists (no reset but 2.2 t/s decode)
- NEO direct submission / relaxed ordering (silent stalls)

---

## Fix Verification

### Workaround: `UR_L0_USE_COPY_ENGINE=0`

Forces all USM copies to compute queue (CCS), bypassing BCS entirely.

| Configuration | Failures/Total | Benchmark (tg64) | Real-Text (tg) | Status |
| --------------- | ---------------- | ------------------ | ---------------- | -------- |
| xe, copy engine **on**, --moe-cache auto | 11/14 | 14.5 | - | FAIL |
| xe, copy engine **off**, --moe-cache auto | 0/10 | 14.5 | - | PASS |
| xe, copy engine **off**, --moe-cache off | 0/6 prod | 47.6 (bench) / 32.8 (text) | 32.8 | PASS |
| xe, copy engine **on**, --moe-cache off | 2/2 prod crash | 47.6 / 32.8 | 32.4 | FAIL |

**Production deployment:** Systemd drop-in `/etc/systemd/system/llama-sycl.service.d/xe-copy-engine.conf`:

```ini
[Service]
Environment=UR_L0_USE_COPY_ENGINE=0
```

---

## Decode Regression Re-evaluation

**Original claim:** xe has -20 to -29% decode regression vs i915.

**Reality:** Benchmark used `--moe-cache auto` (llama-bench default), which streams host-resident expert weights every token. Production uses `--moe-cache off` (no optional cache, but scheduler still uploads selected experts).

| Placement | Copy Engine | llama-bench tg64 | Real-Text tg | Notes |
| ----------- | ------------- | ------------------ | -------------- | ------- |
| auto | off | 14.5 | - | All prior benchmarks |
| soft | off | 22.7 | 18.5 | Keeps 18/41 layers GPU-resident |
| **off** | **off** | **47.6** | **32.8** | **Production config** |
| off | on | 47.6 | 32.4 | ~1% delta |

**Conclusion:** No inherent xe decode regression. The "regression" was an artifact of benchmarking the wrong placement. Production on xe with copy engine off achieves ~32.5 t/s real-text decode.

---

## Upstream Bug Report (Ready to File)

**Target:** `intel/compute-runtime` (NEO / Unified Runtime)

**Title:** BCS Engine Hang ("Timedout job") with Chained USM Memcopies on Xe Driver (DG2/acm-g10)

**Environment:** Intel Arc A770, xe KMD, compute-runtime 26.35.39758, level-zero-loader 1.32

**Symptom:** Chained ~860 KB host->device USM copies on `in_order` queue stall or trigger BCS reset (`RING_ESR=1`, `IPEHR=0xfffff000` or `0x72080025`).

**Reproducer:** Two immediate `inOrder: false` command lists (compute + copy), event-chained `zeCommandListAppendMemoryCopy` (~860 KB), interleaved kernels. Compare `UR_L0_USE_COPY_ENGINE` unset vs 0.

**Workaround:** `export UR_L0_USE_COPY_ENGINE=0` - 10/10 clean runs.

**Artifacts Available:** 4 coredumps, UR trace (716K lines), kernel logs, host backtraces, engine counter samples, live copy metadata.

**Non-Claims:**

- [fail] IPEHR=0x72080025 != proof of wrong-stream (suspect PID/timestamp, VM error -19)
- [fail] No proven inherent decode regression (benchmark placement artifact)
- [fail] No llama.cpp buffer-ordering bug (i915 succeeds)

---

## Session Timeline

| Time | Event |
| ------ | ------- |
| 2026-09-29 13:15 | Production crash #1 (missed initially) |
| 2026-09-29 13:20 | Production crash #2 |
| 2026-09-29 15:58-17:44 | Fix ladder runs (evidence/raw/N-xe-*) |
| 2026-09-29 17:35 | Copy-path tracer runs (copytrace, copytrace-nocache) |
| 2026-09-29 17:44+ | Real-text --moe-cache comparison runs |
| 2026-09-29 17:59 | README updated to 488 lines |
| 2026-09-30 | Session documentation created |

---

## Files Modified During Session

### Production Configuration (Not in Repo)

- `/etc/systemd/system/llama-sycl.service.d/xe-copy-engine.conf` - `UR_L0_USE_COPY_ENGINE=0`
- `/etc/modprobe.d/intel-xe.conf` - `xe.force_probe=56a0`
- Kernel cmdline - `i915.force_probe=!56a0 xe.force_probe=56a0`

### Repository Documentation (This Directory)

- `docs/xe-fix-docs/SESSION_DOCUMENTATION.md` - This file
- `docs/xe-fix-docs/ADVERSARIAL_RESEARCH_ARTIFACT.md` - Complete research artifact
- `docs/xe-fix-docs/evidence/` - All benchmark evidence
- `docs/xe-fix-docs/artifacts/` - Tracer source, coordination logs

---

## Commands Reference

### Build (JIT - Development)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -GNinja \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DGGML_SYCL_F16=ON
ninja -C build-sycl
```

### Build (AOT - Production)

```bash
cmake -B build-aot -GNinja \
  -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DGGML_SYCL_F16=ON
ninja -C build-aot   # ~14 min
```

### Benchmark

```bash
# Product bench
./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -p 512 -n 128 -r 3

# Depth sweep
./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -p 0 -n 128 -d 0,4096,16384

# Real-text
./build-sycl/bin/llama-completion -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -f prompt.txt -n 128 -temp 0
```

### GPU Discipline (Mandatory Before Timing)

```bash
sudo systemctl stop llama-sycl.cpp.service
fuser -v /dev/dri/renderD128
dmesg | grep -iE 'xe.*(reset|hang|timeout|GuC)'
# ... run benchmark ...
sudo systemctl start llama-sycl.cpp.service
```

### Runtime Environment (Production)

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export GGML_SYCL_DISABLE_GRAPHS=1
export UR_L0_USE_COPY_ENGINE=0        # THE FIX
# Optional: TURBO_AUTO_ASYMMETRIC=0, TURBO_LAYER_ADAPTIVE=7
```

---

## Lessons Learned

1. **Benchmark configuration matters critically** - `--moe-cache auto` vs `off` changed decode numbers from 14.5 -> 47.6 t/s (bench) and 18.5 -> 32.8 t/s (text). Never trust llama-bench decode rows for MoE models.

2. **Environment variable workarounds are fragile** - `UR_L0_USE_COPY_ENGINE=0` is undocumented and could be removed. Upstream fix is essential.

3. **Coredump PID/timestamp association can be misleading** - The server coredump's IPEHR=0x72080025 didn't match its crash log timestamp.

4. **Cross-engine synchronization is the weak point** - UR's two immediate lists (compute + copy) with event chaining is the failure mode. Single-queue testing doesn't reproduce it.

5. **Production traffic reveals failures benchmarks miss** - The 13:15 crash occurred on traffic "nobody in this session sent." Soak testing with real workloads is irreplaceable.

---

## Next Steps (If Any)

1. **File upstream bug** at <https://github.com/intel/compute-runtime/issues> with the prepared report
2. **Monitor compute-runtime releases** for `UR_L0_USE_COPY_ENGINE` behavior changes
3. **Consider explicit queue selection in llama.cpp SYCL backend** if upstream fix stalls (would require code changes to use separate copy queue with `UR_L0_USE_COPY_ENGINE=0` semantics built-in)

---

## Appendix: Research Artifact (Verbatim)

See `ADVERSARIAL_RESEARCH_ARTIFACT.md` in this directory for the complete adversarial methodology output including framing pushback, evidence tables, self-attack, and committed recommendations.

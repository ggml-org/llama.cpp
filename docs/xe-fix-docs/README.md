# Xe Driver Fix Documentation Index

## September 29 source-audit and research session archive

Added September 30 at the user's request:

- [Full final report](FINAL_REPORT.md), with primary-source citations and portable evidence links.
- [Complete session account](session/codex-session-2026-09-29.md): requests,
  chronology, code audit, GPU coordination, tracer implementation/verification,
  two diagnostic runs, firmware checks, provenance, corrections and open work.
- [Final response delivered to the user](session/final-response-2026-09-29.md).
- [Evidence corrections and precedence](session/codex-session-2026-09-29.md#corrections-and-evidence-precedence).
- [Source manifest](session/source-manifest-2026-09-30.json): 173 committed targets
  with current hashes and original source hashes; 19 unbundled artifacts are
  listed separately under `omitted`. This is a partial archive.
- Text has been normalized to ASCII. JSON Unicode escapes preserve decoded
  values; prose and captured text use ASCII representations. Original byte
  counts and hashes are retained as `source_bytes` and `source_sha256`.
- `evidence/raw/` is canonical; the nested campaign's `raw` is a relative
  symlink to it. Checkouts without symlink support should use the canonical path.
- `artifacts/` is the canonical investigation tree; the duplicate
  `evidence/xe-investigation-20260929/` was removed. The manifest tracks `artifacts/`.
- Historical tracer wrappers require local paths and unbundled `.so` files;
  they are provenance records, not runnable tools from a fresh checkout.
- [Original i915 comparison campaign](evidence/baseline-2026-09-27/README.md).

**Historical-content notice:** The earlier index below and some companion
summaries overstate conclusions. The exact defect is not established; copy-off
has finite successful evidence, not proven failure elimination. The auto-placement
decode regression is measured, while the production off-placement i915 comparison
is missing. The recorded production drop-in belongs to
`llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d`, not `llama-sycl.service.d`.
Read the correction guide before using the historical claims as findings.
Later kernel/reproducer work remains separately attributed in `analysis/`.

## 2026-09-30 root cause (supersedes the cause ranking below)

The instrumented rc5 runs closed the question: the failure is NEO's blitter residency on
the xe KMD, not a UR event chain and not the kernel. Every userptr bind of the read-only
mmap'd model pages fails with `EPERM` on xe; NEO answers each failure with its
`evictUnusedAllocations()` sweep (compute-runtime #973/#1010 family), and the sweep unbinds
the blitter's KMD-submitted command buffer while its job is pending; the blitter parses
scratch zeros, halts on the next mapped allocation, the host stalls, and the next LR-mode
suspend produces the `Engine reset` after the 640 ms preempt timeout. Compute is on direct
submission and unaffected. Evidence, runs X1/X4/X5 and the fix ladder:
[`docs/research/xe-kmd-bcs-copy-engine-2026-09-30.md`](../research/xe-kmd-bcs-copy-engine-2026-09-30.md).
The fork now defaults `UR_L0_USE_COPY_ENGINE=0` on xe by itself (`ggml/src/ggml-sycl/xe-kmd.cpp`),
and `--load-mode none` removes the per-copy userptr imports (pinned `SYCL_Host` experts).
NEO master (`8ae033266e`) still fails the same way (X7A); an 11-line NEO patch that retries
the userptr bind read-only on `EPERM` fixes the trigger (X7B, 0 failed binds, clean with the
blitter on): `docs/research/patches/0001-neo-retry-userptr-bind-readonly-on-eperm.patch`.
The production drop-in path is `llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf`.

## Earlier index (preserved historical content)

**Session:** 2026-09-30
**Status:** Production stable with workaround (`UR_L0_USE_COPY_ENGINE=0`)
**Upstream Bug:** Ready to file against `intel/compute-runtime`

---

## Directory Structure

```
xe-fix-docs/
+-- README.md                          # This index
+-- SESSION_DOCUMENTATION.md           # Complete session narrative with timeline, commands, lessons
+-- ADVERSARIAL_RESEARCH_ARTIFACT.md   # Full adversarial methodology output (8 phases)
+-- evidence/                          # Original benchmark evidence (copied from ~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/)
|   +-- README.md                      # 488-line canonical evidence document
|   +-- copy-path-trace.md             # 62-line trace analysis
|   +-- raw/                           # All raw benchmark outputs (28 directories)
|   |   +-- N-xe-r1, N-xe-r2/                 # Base xe runs
|   |   +-- N-xe-copyeng0/                    # Copy-engine-off (base)
|   |   +-- N-xe-copyeng0-moeoff/             # Copy-off + --moe-cache off
|   |   +-- N-xe-copyeng0-moesoft/            # Copy-off + --moe-cache soft
|   |   +-- N-xe-copyeng0-r2, r3, r4/         # Additional soak rounds
|   |   +-- N-xe-copytrace/                   # Tracer run (stalled)
|   |   +-- N-xe-copytrace-nocache/           # Tracer + event cache disabled
|   |   +-- N-xe-pinned0/                     # Device coredump + kernel log
|   |   +-- N-xe-server/                      # Production crash + 2nd coredump
|   |   +-- N-xe-cbevents0/                   # Counter-based events test
|   |   +-- N-xe-immcl0/                      # Immediate command lists test
|   |   +-- N-xe-inorder0/                    # Driver in-order lists test
|   |   +-- N-xe-graph0/                      # SYCL graphs test
|   |   +-- dense-8b/                         # Dense 8B A/B comparison
|   |   +-- encode/                           # Video encode tests
|   |   +-- N-xe-cachestats/                  # Real-text completion runs
|   |   +-- bench*.sh, run-*.sh, *.log        # Runner scripts and logs
|   |   +-- config/                           # Tune script, udev, systemd drop-in
|   +-- config/                          # Driver configuration files
+-- artifacts/                         # Additional investigation artifacts (from /mnt/nvme1/oneapi-ab/xe-investigation-20260929/)
    +-- bin/                           # Tracer-wrapped llama-bench
    +-- copy-tracer/                   # Trace-copies.so source, check-trace.c, GDB commands
    +-- COORDINATION.md                # Cross-session coordination log
```

---

## Quick Reference

### The Fix (Production)

```bash
# Systemd drop-in: /etc/systemd/system/llama-sycl.service.d/xe-copy-engine.conf
[Service]
Environment=UR_L0_USE_COPY_ENGINE=0
```

### Key Metrics

| Config | Failures | Prefill (pp512) | Decode (tg64 real-text) |
| -------- | ---------- | ----------------- | ------------------------- |
| i915 (reference) | 0/0 | baseline | ~32 t/s (est.) |
| xe, copy engine **on** | 11/14 | +6-17% | 14.5 (auto) / 32.4 (off) |
| xe, copy engine **off** | **0/10** | +6-17% | **32.8 (off)** |

### Root Cause

Cross-engine event synchronization deadlock in Unified Runtime between compute (CCS) and copy (BCS) engines when chaining MoE expert-weight uploads (~860 KB, Q4_K/Q6_K).

### Upstream Report Target

**Repository:** `intel/compute-runtime`
**Status:** Partial evidence archive; see the manifest omissions above.

---

## Evidence Highlights

| File | Key Finding |
| ------ | ------------- |
| `evidence/README.md` | Fix ladder, UR trace, decode regression re-evaluation, --moe-cache placement correction |
| `evidence/copy-path-trace.md` | Confirmed expert-weight uploads (not activations); scheduler path; event caching not a fix |
| `evidence/raw/N-xe-pinned0/xe-devcoredump-*.txt.gz` | BCS timeout, RING_ESR=1, IPEHR=0xfffff000 |
| `evidence/raw/N-xe-server/xe-devcoredump-*.txt.gz` | IPEHR=0x72080025 (COMPUTE_WALKER on BCS) - **suspect PID/timestamp** |
| `evidence/raw/N-xe-trace/bench.err.tail3000.txt` | 716K lines: event-chained zeCommandListAppendMemoryCopy stall |

---

## Lessons Learned

1. **Benchmark config != production config** - `--moe-cache auto` (bench default) vs `off` (production) = 3x decode difference
2. **Env var workarounds are fragile** - `UR_L0_USE_COPY_ENGINE=0` undocumented; upstream fix needed
3. **Coredump metadata can mislead** - PID/timestamp mismatch on server coredump
4. **Cross-engine sync is the weak point** - UR's dual immediate lists + event chaining = failure mode
5. **Real traffic finds bugs benchmarks miss** - 13:15 crash on "nobody's traffic"

---

## Files Not in Repo (Production Config)

| File | Purpose |
| ------ | --------- |
| `/etc/systemd/system/llama-sycl.service.d/xe-copy-engine.conf` | Sets `UR_L0_USE_COPY_ENGINE=0` |
| `/etc/modprobe.d/intel-xe.conf` | `xe.force_probe=56a0` |
| Kernel cmdline (`zroot/ROOT`) | `i915.force_probe=!56a0 xe.force_probe=56a0` |

---

## Next Actions

1. [ ] File upstream bug at <https://github.com/intel/compute-runtime/issues>
2. [ ] Monitor `compute-runtime` releases for `UR_L0_USE_COPY_ENGINE` changes
3. [ ] (If upstream stalls) Implement explicit queue selection in llama.cpp SYCL backend

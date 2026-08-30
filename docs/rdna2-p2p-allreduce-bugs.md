# Three bugs in the RDNA2 P2P host-snapshot AllReduce dispatch

Found on `perf/dflash2-e2e-overhead` @ `8e566ab02`. All three are in
`ggml/src/ggml-cuda/ggml-cuda.cu`. All three affect TP4  -  none of this is
TP2-specific.

The common theme: the path can be **completely inactive while every log line
says it is healthy.** Two independent causes, plus a visibility gap that hides
both.

---

## 1. `GGML_HIP_GFX1030_P2P_ALLREDUCE=1` silently means "off"

`ggml_cuda_rdna2_p2p_host_allreduce_mode()`, line ~1032:

```c
if (value == nullptr || strcmp(value,"auto")==0 || strcmp(value,"auto-expanded")==0)
    return GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED;
if (strcmp(value,"auto-basic")==0) return GGML_CUDA_RDNA2_P2P_HOST_AUTO;
if (strcmp(value,"host")==0)       return GGML_CUDA_RDNA2_P2P_HOST_SIMPLE;
if (strcmp(value,"host-fused")==0) return GGML_CUDA_RDNA2_P2P_HOST_FUSED;
if (strcmp(value,"host-mtp")==0)   return GGML_CUDA_RDNA2_P2P_HOST_MTP;
if (strcmp(value,"off")==0 || strcmp(value,"0")==0 || strcmp(value,"false")==0)
    return GGML_CUDA_RDNA2_P2P_HOST_OFF;
return GGML_CUDA_RDNA2_P2P_HOST_OFF;      // <-- everything else
```

`1` is not recognised, so it hits the final fallthrough and **disables the
path**. No warning, no log line. `=1` is the obvious way to spell "enable
this", and it does the exact opposite.

Consequence: any launch script carrying `GGML_HIP_GFX1030_P2P_ALLREDUCE=1` has
had the host-snapshot AllReduce off the whole time  -  including on TP4, where
`OPTIMIZATION-STATUS.md` credits it with `+2.78-2.95%` over RCCL-only.

The tell is the *absence* of a line. With the path enabled you get:

```
RDNA2 P2P host-snapshot AllReduce unavailable for this topology/policy; using RCCL
```

or an `armed ...` line. With `=1` you get neither, because the whole block is
skipped by `ggml_cuda_rdna2_p2p_host_allreduce_enabled()`.

**Fix:** map `1`/`on`/`true`/`yes` to the automatic default, and warn on
genuinely unrecognised values instead of silently disabling.

---

## 2. The speculative width gate originally served one `n_max`

The original exact host-snapshot kernels implemented only `[5120,1,1,1]` and
`[5120,5,1,1]`. A speculative verify batch is commonly `n_max + 1` tokens, so
width five corresponds to `--spec-draft-n-max 4`. Other widths correctly fall
back to RCCL, but that fallback used to be silent after startup had reported the
path as armed.

This was first observed with MTP `n_max=5` (width six) on TP2. Later TP4 traces
also confirmed that the DFlash target pass reaches `[5120,6,1,1]` at
`n_max=5`; the layer-split DFlash draft does not perform TP AllReduce. Neither
workload could use the width-five kernel at `n_max=5`.

The precompiled TP4 route now also supports `[5120,6,1,1]`. Its 30,720-element
reduction is checked against the installed RCCL implementation by the same
startup exactness self-test used for widths one and five. TP1 and TP2 do not
activate this four-rank host-snapshot path and continue to use their normal
RCCL/internal fallback.

This does not invalidate the published ordinary TP4 result. Ordinary decode is
width one, takes the validated width-one route, and retains its measured
`+2.78-2.95%` benefit over RCCL-only execution. The width-six route is a
separate incremental optimization and does not claim to close the remaining 2x
gap.

Qwen4Exp uses a 2,560-wide hidden state. Its sidecar with
`--spec-draft-n-max 3` makes target verification graphs of widths two through
four, so the old Qwen4Exp width-one route could not serve them. The current
implementation adds exact `[2560,2]`, `[2560,3]`, and `[2560,4]` routes. Their
flat element counts and installed-RCCL schedules are:

- width two: 5,120 elements, reusing the exact eight-by-640 schedule;
- width three: 7,680 elements, four 1,920-element chunks;
- width four: 10,240 elements, four 1,536-element chunks followed by four
  1,024-element chunks.

Every size is checked over four adversarial patterns and sixteen chained
reductions at startup. A size that does not match the installed RCCL byte for
byte remains on RCCL. On four V620s, a 20-sample `[2560,4]` graph microbenchmark
improved steady execution by 4.30%; an exact fixed-output server ABBA improved
19.798 to 19.960 tok/s (0.82%) with identical output, proposal, and acceptance
counts. This is incremental: tensor-split sidecar input still reports
`bound device=-1` and pays a separate host synchronization/copy cost.

**Fix:** classify the route explicitly and log unsupported widths once. A
supported width whose startup exactness self-test failed receives a different
warning. The implementation deliberately does not generalize the reduction
kernel: a new TP4 width requires its own installed-RCCL exactness validation.

---

## 3. "Armed" reads like success, but dispatch count is invisible until teardown

Startup prints:

```
RDNA2 P2P startup self-test matched installed RCCL for 5120 elements (4 patterns x 16 chains)
armed RDNA2 P2P MTP-width5-auto-expanded host-snapshot AllReduce after installed-RCCL self-test (n1=1 n5=1 n6=1)
```

That reads as "working". But `p2p_host_calls` is only printed in the
destructor (line ~1274), so a server that runs for hours with **zero**
dispatches looks identical to one where every boundary is served.

The first-dispatch line (`using RDNA2 P2P ... for %s [5120,%d,1,1] F32`) was
gated on one `p2p_host_logged` boolean, so it could not show which widths were
actually served.

The fix uses shared served/refused bitmasks across the fused and non-fused
routes. Each relevant width is reported once without duplicate first-use logs.

---

## 4. `GGML_META_PARALLEL_SET=0` enabled the path; external drafts remain a known issue

Two problems stacked.

**It is presence-checked, not value-checked** (`ggml-backend-meta.cpp:1770`):

```c
static const bool parallel_set_enabled = getenv("GGML_META_PARALLEL_SET") != nullptr;
```

So `GGML_META_PARALLEL_SET=0` *enables* it. The only way off is unsetting the
variable entirely. Confirmed empirically: `=0` and `=1` both crash, removing the
flag does not.

That pattern is upstream convention for a lot of GGML flags (`GGML_CUDA_P2P`,
`GGML_VK_*`), so it is defensible on its own. The problem is that this fork also
adds flags that *do* parse values (`GGML_HIP_GFX1030_P2P_ALLREDUCE`,
`GGML_HIP_RCCL_TUNE`, `GGML_TP_SHARDED_OUTPUT`), and nothing in the name tells
you which kind you are holding. `=0` disables some and enables others.

**The crash.** With `-sm tensor -dev rocm0,rocm1` plus any external draft model
(`-md ... --spec-type draft-dflash`), weight upload dies:

```
meta backend weight uploads: parallel across 2 devices
Memory access fault by GPU node-3 ... Reason: Page not present or supervisor privilege.
```

The parallel path fires `std::async` threads to upload weights to different
devices, and `hipSetDevice` is thread-local. Without a drafter the same path
runs fine, so the trigger appears to be the draft-model probe that runs first
and always throws for dflash:

```
E llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set
W [spec] failed to measure draft model memory: failed to create llama_context from model
```

`llama-context.cpp:161` raises that deliberately and the text calls it "normal
during memory fitting" -- but it unwinds out of a partly-built context, and the
async upload threads then fault. Worth noting the probe cannot ever succeed for
dflash: `ctx_other` is only set later, in `common_speculative`.

The crash mechanism is still unresolved and is not claimed as fixed here.
Workaround: unset the variable or use `GGML_META_PARALLEL_SET=0`. Invalid values
also fail closed with a warning. The cost is serial weight upload.

---

## How this was found

Running the TP4-developed stack on 2x V620 (TP2). Bug 1 showed up because the
P2P block produced no log output at all with `=1`. Bug 2 showed up after that
was fixed: the path armed, self-tested exact against RCCL, and then never
dispatched  -  because `--spec-draft-n-max 5` gives width 6. Widening the TP2
gate to accept any self-tested width made it dispatch immediately.

Both reproduce on TP4; neither is a TP2 artifact.

## Unrelated negative result: the GDN kernel is not the bottleneck

Worth recording so nobody spends a week on it.

`gated_delta_net.cu`'s `use_chunked` is a misnomer -- the recurrence is strictly
serial in `t` and the grid is independent of `n_tokens`, so all 48 linear-
attention layers have per-token cost that is completely flat in ubatch. That
looks like an obvious target for a real chunked/WY-form rewrite.

Measured on an idle V620 (`test-backend-ops perf`, head_count=32, head_size=128):

| n_tokens | us/run | us/token |
|---:|---:|---:|
| 1 | 12.85 | 12.85 |
| 64 | 130.61 | 2.04 |
| 256 | 434.81 | 1.70 |
| 512 | 847.50 | 1.66 |
| 1024 | 1702.14 | 1.66 |

Flat past 256, exactly as the code reads. But the share is small: a 128-token
ubatch across 48 GDN layers is ~10 ms against ~246 ms for a 128-token prefill at
a measured ~520 tok/s, i.e. **~4% of prefill** and ~2% of decode. A chunked
rewrite buys low single digits. Not worth it.

(Measure this on an idle GPU. A contended run inflated the 512-token figure by
16% and the single-token figure by 65%, which is enough to mislead.)

## Related, if useful: TP2 unlock (separate, and a null result)

A separate (unmerged, not part of this PR) experiment opens the four-rank gates
for TP2: rank-parameterised topology predicates, a `barrier2`, an
element-count-generic pair-sum reduce kernel, and a startup self-test sweep
that validated batch widths 1..12 byte-for-byte against installed RCCL on
2x V620.

**It produced no measurable throughput change on TP2.** Normalised to
cost-per-verify-pass over ~250-350 verify calls per condition:

| condition | ms/verify |
|---|---|
| P2P off | 63.81 |
| P2P on, never dispatched | 61.80 |
| P2P on, dispatching | 62.78 |

The condition that provably never ran the kernel was the fastest, so the +/-3%
spread is noise and the measurement could not resolve the effect. Plausible
structural reason: the host-snapshot trick short-circuits RCCL's multi-hop
**ring**, and with two ranks there is no ring  -  RCCL does one direct P2P
exchange, which a mapped-host round-trip has no reason to beat.

So the four-rank gate may have been conservative, but it does not look like it
was hiding a win. The branch is mainly useful as the vehicle that surfaced
bugs 1 and 2.

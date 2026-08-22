# Three bugs in the RDNA2 P2P host-snapshot AllReduce dispatch

Found on `perf/dflash2-e2e-overhead` @ `8e566ab02`. All three are in
`ggml/src/ggml-cuda/ggml-cuda.cu`. All three affect TP4 — none of this is
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
had the host-snapshot AllReduce off the whole time — including on TP4, where
`OPTIMIZATION-STATUS.md` credits it with `+2.78–2.95%` over RCCL-only.

The tell is the *absence* of a line. With the path enabled you get:

```
RDNA2 P2P host-snapshot AllReduce unavailable for this topology/policy; using RCCL
```

or an `armed ...` line. With `=1` you get neither, because the whole block is
skipped by `ggml_cuda_rdna2_p2p_host_allreduce_enabled()`.

**Fix:** map `1`/`on`/`true`/`yes` to the automatic default, and warn on
genuinely unrecognised values instead of silently disabling.

---

## 2. The MTP width gate only ever matches `--spec-draft-n-max 4`

`ggml_backend_cuda_comm_allreduce_rdna2_p2p_host()`, line ~1611:

```c
const bool mtp5 = ... && tensors[0]->ne[1] == 5 && ...;
const bool ordinary1 = !mtp5 && ... && tensors[0]->ne[1] == 1 && ...;
if ((!mtp5 && !ordinary1) || ...) return false;
```

The MTP verify batch is `n_max + 1` tokens, so `ne[1] == 5` ⟺ `n_max == 4`.
`docs/rdna2-native-coordination.md:45` states the assumption directly:

> non-shared/non-chained MTP, `n_max=4`, **width five**

At any other `n_max` the batch width matches neither gate, the path arms at
startup and then **never dispatches once**. It silently falls back to RCCL.

`n+1` is not an MTP quirk — it is how every block draft sizes its batch.
`common/speculative.cpp:2567`:

```c
// per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
const int32_t per_seq = std::max(1, params_spec.n_max + 1);
```

So the gate matches exactly **one** value of `n_max`. That is a bad property
regardless of what anyone runs.

### What I actually observed (MTP, 2x V620)

`--spec-draft-n-max 5` -> width 6. The path armed, self-tested exact against
RCCL, and dispatched **zero** times for an entire server session. Widening the
gate to accept any self-tested width made it dispatch immediately. That much is
measured, not inferred.

### What I have NOT tested

Everything below is derived from reading the code, and **DFlash has not been
run at all**:

| path | `n_max` | width | dispatches? | basis |
|---|---|---|---|---|
| MTP | 3 | 4 | no | inferred |
| MTP | 4 | 5 | **yes** | inferred (matches the doc's stated design point) |
| MTP | 5 | 6 | no | **observed** |
| DFlash | ? | ? | ? | **untested** |

I originally wrote that DFlash cannot dispatch at its default `n_max`. I am
pulling that claim: it rests on assuming the DFlash target verify pass produces
the same `[5120, n_max+1, 1, 1]` AllReduce boundary that MTP does, and I have
not confirmed that. Two things specifically need checking before anyone repeats
it:

- whether the DFlash target pass hits `linear_attn_out-*` / `ffn_out-*` /
  `attn_output-*` AllReduce boundaries at all, and at what `ne[1]`;
- `common/speculative.cpp:2615` forces the DFlash/DSpark *draft* model to
  `LLAMA_SPLIT_MODE_LAYER` while the target keeps tensor split, so the draft
  context does no TP AllReduce. Whether the target's does, at what width, is
  the open question.

The `n_max + 1` derivation itself is from the code (`speculative.cpp:2567`) and
does cover DFlash and DSpark. But "the width is n+1" and "that width reaches
this gate" are different claims, and only the first is established.

**Fix (cheap):** at minimum, log a one-shot warning when the path is armed but
sees a width it cannot serve. Better: generalise the width. On TP2 this is
trivial because a two-rank reduction is a commutative pair-sum; on TP4 it needs
a validated chunk schedule per width, so a startup self-test sweep over the
widths you intend to support is the honest way to do it.

---

## 3. "Armed" reads like success, but dispatch count is invisible until teardown

Startup prints:

```
RDNA2 P2P startup self-test matched installed RCCL for 5120 elements (4 patterns x 16 chains)
armed RDNA2 P2P MTP-width5-auto-expanded host-snapshot AllReduce after installed-RCCL self-test (n1=1 n5=1)
```

That reads as "working". But `p2p_host_calls` is only printed in the
destructor (line ~1274), so a server that runs for hours with **zero**
dispatches looks identical to one where every boundary is served.

The first-dispatch line (`using RDNA2 P2P ... for %s [5120,%d,1,1] F32`) is
gated on a single `p2p_host_logged` bool, so it also only ever prints once —
it cannot show you that a second width was rejected.

This is what makes bug 2 invisible in practice. Making the first-dispatch line
fire once **per width** costs a `uint32_t` bitmask and immediately surfaces
both problems.

---

## 4. `GGML_META_PARALLEL_SET` hard-crashes with an external draft model, and `=0` does not disable it

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

Workaround: unset `GGML_META_PARALLEL_SET`. Cost is serial weight upload.

---

## How this was found

Running the TP4-developed stack on 2x V620 (TP2). Bug 1 showed up because the
P2P block produced no log output at all with `=1`. Bug 2 showed up after that
was fixed: the path armed, self-tested exact against RCCL, and then never
dispatched — because `--spec-draft-n-max 5` gives width 6. Widening the TP2
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
cost-per-verify-pass over ~250–350 verify calls per condition:

| condition | ms/verify |
|---|---|
| P2P off | 63.81 |
| P2P on, never dispatched | 61.80 |
| P2P on, dispatching | 62.78 |

The condition that provably never ran the kernel was the fastest, so the ±3%
spread is noise and the measurement could not resolve the effect. Plausible
structural reason: the host-snapshot trick short-circuits RCCL's multi-hop
**ring**, and with two ranks there is no ring — RCCL does one direct P2P
exchange, which a mapped-host round-trip has no reason to beat.

So the four-rank gate may have been conservative, but it does not look like it
was hiding a win. The branch is mainly useful as the vehicle that surfaced
bugs 1 and 2.

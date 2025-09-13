Here’s a clear, no‑fluff walkthrough and critique of the piece.

---

## TL;DR in one punchline

The randomness you see at temperature=0 isn’t “GPU gremlins racing each other.” It’s your **inference kernels changing reduction order when batch size or sequence chunking changes**. Fix the kernels to be **batch‑invariant** (and make attention handle KV cache + chunking identically), and temperature‑0 becomes bitwise deterministic at the API boundary—at some cost to throughput.

---

## What the article overturns

**Popular story:** nondeterminism comes from “concurrency + floating‑point non‑associativity” (who finishes first changes addition order → different bits).

**Author’s claim:** that’s not the root cause for LLM inference. The per‑kernel implementations used in the **forward pass** of LLM inference (RMSNorm, matmuls, attention) are *run‑to‑run deterministic* on fixed shapes. The real leak is **lack of batch invariance**: the numeric path a request takes depends on other concurrent requests (batch size, prefill chunking, KV layout), so the *same* prompt can traverse a *different* reduction order from one call to the next.

**Key distinction he formalizes:**

* **Run‑to‑run determinism:** same inputs, same kernel config → same bits.
* **Batch invariance:** numerics for *one element* don’t change when the *rest of the batch* (or chunking) changes.
* **User‑visible determinism:** what you actually want at the endpoint; depends on both of the above **plus** a scheduler that won’t change the numeric path.

---

## Why floating‑point matters but isn’t the villain

Floating‑point addition is non‑associative, so changing reduction order changes the result. But:

* If the **reduction order is fixed**, you’ll get the same result every time on the same hardware/software stack.
* In practice, the order *isn’t* fixed when kernels adapt tiling, split‑reductions, or tensor‑core instructions based on **batch size** or **sequence partitioning**. That adaptation is what makes two identical requests diverge once they share the server.

---

## Where batch invariance breaks (kernel by kernel)

### 1) RMSNorm

* **Good path (batch‑invariant):** Data‑parallel per‑row reductions; each row’s reduction fully inside one core; increasing batch just gives more rows; decreasing batch still uses the *same* reduction order per row.
* **Where it breaks:** Very small batches tempt you to **split** the per‑row reduction across cores (for utilization). That changes the reduction tree → numerics differ.
* **Fix:** Don’t split reductions by batch‑size heuristics. Either accept under‑utilization for tiny batches or use one reduction strategy that works across sizes (constant order), even if sub‑optimal.

### 2) Matrix multiplication (GEMM)

* **Good path:** Tile the **output** (M×N) and keep each tile’s K‑reduction inside one core (data‑parallel). Reduction order fixed per tile.
* **Where it breaks:** When M and N are small, libraries switch to **Split‑K** (parallelize the K reduction). Or they pick **different tensor‑core instructions** at small shapes. Both change accumulation order.
* **Fix:** Pick a **single tiling + instruction set** and **disable Split‑K** for these shapes. Accept \~O(10–20%) perf loss versus cuBLAS; numerics become batch‑invariant.

### 3) Attention (the hard one)

Two extra wrinkles:

* Reduces over **feature** and **sequence** dims.
* Inference engines do **chunked prefill**, **prefix (KV) caching**, and variable **decode** query lengths.

**Breakage patterns:**

* Handling **KV cache vs current tokens** in separate loops/blocks yields different reduction boundaries depending on how many tokens are cached vs freshly computed.
* **Split‑KV/FlashDecode** usually chooses **number of splits** based on how much parallelism is needed; that depends on batch/QLen → reduction order changes.

**Fixes proposed:**

* **Normalize KV layout before the attention kernel** so the kernel always sees one consistent K/V view regardless of chunking or cache size; reduce in a single consistent pass.
* Use **fixed split size** (constant chunk length along KV) rather than a fixed number of splits. The number of splits may vary, but the **per‑token reduction order** stays the same across batch/QLen.

---

## Evidence presented

### Divergence at temperature 0

* 1,000 runs of “Tell me about Richard Feynman” on a 235B model, T=0, 1,000 tokens.
* **80 unique completions**; first **102 tokens identical**; divergence starts at token 103 (Queens vs NYC wording).
* With batch‑invariant kernels enabled → **all 1,000 completions identical**.

### Performance hit (single‑GPU server, 1,000 seqs \~100 tokens)

* vLLM default: **26s**
* Deterministic (unoptimized): **55s**
* Deterministic with improved attention: **42s**
  Interpretation: not free, but not catastrophic; most pain comes from attention decode parallelism.

### RL implication

* If sampling numerics differ from training numerics, “on‑policy” RL becomes implicitly **off‑policy**. With bitwise‑identical inference and training (same kernels, same numerics), KL between sampler and trainer stays **exactly 0**, and training is stable without importance weighting.

---

## What this definitively fixes—and what it doesn’t

**Fixes:**

* Nondeterminism introduced by **dynamic batching**, **prefill chunking**, **Split‑K/Split‑KV heuristics**, **kernel tiling changes**.
* Endpoint behavior for greedy decoding, provided the software/hardware stack is held fixed.

**Still need to control:**

* **Cross‑hardware/version drift.** Different GPUs/drivers/PTX/BLAS heuristics can produce different but deterministic numerics. Pin hardware + CUDA + library versions.
* **Speculative/assisted decoding**, grammar constraints, and **tie‑breaking** when two tokens are exactly equal logit (rare but real): define a deterministic tie rule.
* **Quantization paths** (INT8/FP8) and fused‑kernel variants must be batch‑invariant too.
* **MoE gating**: ensure argmax/ties and expert accumulation use a fixed, batch‑independent order.
* **Multi‑GPU comms**: all‑reduce implementations and in‑switch reductions need deterministic modes/configs; otherwise you reintroduce reduction‑order variability across ranks.
* Any **post‑processing** (tokenizer oddities, Unicode normalization, whitespace changes) must be frozen.

---

## Practical playbook (what to do in an inference stack)

1. **Create a “deterministic mode.”**
   Route eval runs or research jobs to a pool that:

   * Disables dynamic batching (or uses **fixed batch sizes**).
   * Uses **batch‑invariant kernels**: no Split‑K in GEMMs; fixed tensor‑core instruction; **fixed split size** for decode attention; uniform KV layout.
   * Pins versions: GPU model, driver, CUDA, cuBLAS/cuDNN (or your Triton/CUTLASS build), compiler flags.
   * Fixes sampler config: T=0, top‑k=1, top‑p=1, no penalties, deterministic tie‑break.

2. **Instrument for drift.**
   For a fixed prompt set, log per‑token max |Δlogit| across runs under varying system load. A non‑zero signal means some kernel or scheduler path is not batch‑invariant.

3. **Split service tiers.**

   * **Throughput mode** (SLA focus): dynamic batching on, fastest kernels.
   * **Deterministic mode** (science focus): dynamic batching off or constrained; batch‑invariant kernels on.

4. **Attention specifics.**

   * Normalize KV/cache/page‑table **before** the kernel.
   * Prefer **one attention kernel** that handles prefill + decode uniformly.
   * Fix split size along KV for decode; avoid heuristic split‑count.

5. **Docs & tests.**

   * Unit tests that compare logits across batch sizes and chunkings: `B=1` vs `B=3`, prefill chunk sizes `[∞, 1024, 256]`, and KV lengths `[short, long]`.
   * CI that flags any non‑zero drift.

---

## Strengths of the article

* Clean conceptual split between **run‑to‑run determinism** and **batch invariance**. That clears years of muddled forum lore.
* Kernel‑level analysis is concrete: RMSNorm, GEMM, attention, and why naive Split‑K/Split‑KV break invariance.
* Useful operational result: **fixed split size** for FlashDecode‑style attention is the missing trick many teams overlook.

## Gaps / things to watch

* Claims rely on a specific vLLM + FlexAttention path; porting to other stacks (FlashInfer, TensorRT‑LLM, custom CUTLASS/Triton kernels) requires careful replication of **all** invariance constraints.
* Cross‑device reproducibility is acknowledged but not solved; in practice you’ll want container images + driver pinning + heuristic overrides for BLAS planners.
* MoE, speculative decoding, quantization, and pipeline/tensor parallel multi‑GPU cases deserve their own “batch‑invariant” recipes.

---

## Quick sanity harness (minimal, illustrative)

**Batch‑invariance smoke test for GEMM**

```python
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
torch.cuda.manual_seed_all(0)

B, D = 7, 4096
A = torch.linspace(-1000, 1000, B*D, device='cuda', dtype=torch.bfloat16).reshape(B, D)
W = torch.linspace(-1000, 1000, D*D, device='cuda', dtype=torch.bfloat16).reshape(D, D)

# Same row, two batch shapes -> should be identical if matmul is batch‑invariant
o1 = A[:1] @ W
o2 = (A @ W)[:1]
print((o1-o2).abs().max())  # non‑zero means not batch‑invariant
```

**Attention invariance probes to include in CI**

* Prefill as one chunk vs many small chunks; compare per‑token logits.
* Decode with different concurrent batch sizes (1 vs 8) while keeping the measured sequence identical.
* KV lengths short vs very long, ensuring fixed split size along KV.

---

## Editing nits (minor)

* “temperature down to 0This means…” → missing space/period after “0”.
* A few places could inline a one‑liner defining “batch invariance” to reinforce the concept before the matmul example.

---

## Bottom line

This is the most practically useful explanation of “why T=0 isn’t deterministic” I’ve seen: **your numerics change with server load because your kernels adapt to batch/shape**. The proposed remedy—**batch‑invariant kernels** (RMSNorm + GEMM without Split‑K; attention with fixed KV split size and normalized KV layout)—is sound and actionable. Treat determinism like a product feature: put it behind a flag, pin your stack, measure drift, and ship a dedicated deterministic tier for eval and RL.

# MoE Disk-Offload Study: Streaming Trillion-Parameter Models on Apple Silicon

Read-only research study. No code changes. No commits.

> Companion to `docs/ane-backend-deep-study.md` (compute side),
> `docs/interleaved-kernel-design-study.md` (intra-kernel overlap), and
> `docs/inference-engines-comparison-study.md` (concurrency policy). This
> document covers the orthogonal axis of weight I/O for models that do not
> fit in RAM: how to stream them off NVMe without macOS page-cache
> interference, and what the M2 Ultra / Mac Pro hardware ceiling actually
> permits. The motivating artifact is WASTE (Marco Bambini, ~6,000 lines of
> C, July 2026), which runs Moonshot Kimi K3 (2.78T-parameter MoE) on a
> MacBook Pro at ~0.5 tok/s. The thesis of this study: WASTE's techniques
> are directly portable to ggml, but no software work can break the PCIe
> lane budget on Apple Silicon - the hardware ceiling is the wall.

## Table of Contents

1. Executive summary
2. Motivation: WASTE and the disk-bound MoE regime
3. Hardware budget on M2 Ultra Mac Pro
4. The macOS page-cache problem and how to bypass it
5. Stripe in the application, not the block layer
6. RVQ and direct-out-of-codebook multiply
7. Resident-trunk split and the inverted memory curve
8. Read-ahead / I/O-compute overlap
9. Mapping onto the Tessera fork
10. What does NOT translate from WASTE
11. Confidence and open questions
12. Recommendations (deferred)
13. Sources

---

## 1. Executive summary

### 1.1 The opportunity

WASTE proves that a 2.78T-parameter MoE runs on a single MacBook Pro if the
engine treats inference as a storage problem first. The model activates under
4% of its weights per token (16 of 896 experts per layer); idle weights do
not need to be in RAM. The engine pins a ~27 GB resident trunk (attention,
routers, embeddings, head) and streams the active experts off NVMe per
token, ~17 GB read per token, achieving ~0.5 tok/s on a single internal
drive. Logits agree with a PyTorch reference to 3.6e-6.

This is the same regime the Tessera fork's Metal/ANE work targets for
compute, but compute is only ~20% of decode time in this regime. Disk I/O is
~53%. The opportunity is an I/O subsystem for ggml that runs the largest
open-weight MoEs on Apple Silicon that currently have no on-device story.

### 1.2 The hard ceiling

The M2 Ultra die exposes 24 PCIe Gen4 lanes to the Mac Pro slots, plus six
independent Thunderbolt 4 controllers. Realistic sustained read bandwidth:

| Source                                | Realistic bandwidth |
|---------------------------------------|---------------------|
| Single internal NVMe (MacBook Pro)    | ~13 GB/s            |
| One HighPoint SSD7749M2 (x8 electric) | ~14-16 GB/s         |
| Two HighPoint SSD7749M2 (x8/x8)       | ~28-32 GB/s         |
| 6x TB4 NVMe enclosures (one per port) | ~16 GB/s more       |
| **Aggregate, slot + TB, perfectly used** | **~40-45 GB/s**  |

Kimi K3 needs 17 GB read per token. The 1 tok/s floor is reachable; the
video's "10 tok/s would need 170 GB/s" claim is correct and is unreachable
on any M2 Ultra configuration. This is a hardware wall, not a software one.
No custom RAID, no cache-bypass trick, no kernel-side optimization moves
the 24-lane budget.

### 1.3 What software CAN do

Move the striping into the application and read the drives directly with
F_NOCACHE, bypassing the macOS unified buffer cache (UBC). This is not a
"RAID" - it is a sharded weight store with direct I/O, and it is the
generalization of what WASTE does for a single drive. Specifically:

- F_NOCACHE on each weight file prevents expert reads from polluting the
  UBC, which is what prevents the "more RAM = slower" cliff WASTE documents
  (46 GB optimal, 58 GB ~25x slower).
- Experts are independent tensors; place expert (L,E) on drive hash(E)%N
  and read the 16 active experts per layer in parallel, one pread per
  expert. No cross-drive stripe alignment, no slowest-member problem, no
  opaque buffer layer.
- Overlap layer N's compute with layer N+1's expert fetch via a per-drive
  read-ahead pipeline. WASTE's single async read-ahead commit was +60%;
  with N drives the parallelism gain is larger.

### 1.4 Honest ceiling on K3, realistic numbers

| Configuration                        | K3 throughput (real-world) |
|--------------------------------------|----------------------------|
| MacBook Pro, single internal NVMe    | ~0.5 tok/s (WASTE measured)|
| Mac Pro M2 Ultra, 1x HighPoint       | ~1.0-1.2 tok/s             |
| Mac Pro M2 Ultra, 2x HighPoint       | ~1.5-1.8 tok/s             |
| Mac Pro M2 Ultra, 2x HighPoint + 6x TB4 | ~1.8-2.2 tok/s         |
| **Theoretical max (all bandwidth)**  | **~2.3 tok/s**             |

Still "send a hard question, read it in the morning" territory, not chat.
For dense sub-100B models, the same I/O path is transformative: WASTE
reports 10.7 tok/s on a 48B model with a single drive, and the multiplier
holds - the same custom I/O on a 2x HighPoint config would land ~25-30 tok/s
on that class of model. The disk-offload story matters much more for
normal-sized models than for K3 itself.

### 1.5 Recommendation: document, do not build yet

This study recommends deferring implementation. The Mac Pro hardware spend
to validate the multi-drive case is large, the ANE backend (D1) and the
parametric-kernel work are higher-leverage on the compute side, and the
single-drive F_NOCACHE + read-ahead case is small enough (~600 lines) that
it can be picked up quickly once a concrete need appears (a real K3-class
GGUF and a user who wants to run it overnight). The study exists so the
design space is captured before that need arrives.

---

## 2. Motivation: WASTE and the disk-bound MoE regime

### 2.1 The video, summarized

Cloud Codes's video (Et_I4fj1etk) breaks down WASTE. The numbers that
matter for this study:

- Kimi K3: 2.78T parameters total, ~104B active per token. 93 layers, 896
  experts per layer, router picks 16. Moonshot weights landed 27 July 2026.
- 16 experts x 92 layers = 1,472 expert reads per token (one character of
  output). Each is a real NVMe read. Total: ~17 GB read per token.
- Resident trunk: 27.28 GB at 4-bit and 8-bit (attention, routers,
  embeddings, output head). Stays in RAM permanently.
- Engine refuses the OS page cache: F_NOCACHE on macOS, O_DIRECT on Linux,
  no buffering on Windows.
- Decode-time profile: 53.5% disk read, 20% matmul, <3% attention. "A
  storage problem wearing a machine-learning costume."
- Async read-ahead overlap with compute: 0.32 -> 0.5 tok/s in one commit
  (+60%).
- Inverted memory curve: 32 GB -> 0.5 tok/s, 46 GB -> 0.54 (best), 52 GB
  -> 0.04, 58 GB -> 0.02. Past the OS's comfort point, more RAM is 25x
  slower.
- Logits match PyTorch reference to 3.6e-6; vision tower to 2.3e-something.
- Comparison points: Deltafin (114 GB resident, 25.8 GB/read, 0.266 tok/s
  on M1 Max), KTransformers (382 GB DRAM, 13.7 tok/s on 671B), OLLM (0.5
  tok/s streaming 80B to an 8 GB GPU), FlexGen (2023, 175B on one T4 at
  ~1 tok/s).
- The actually-usable result: same engine on a 48B dense model -> 10.7
  tok/s, 1.87 GB RAM. "Not a demo."

The full cleaned transcript (~2,900 words) is preserved locally at
`/tmp/kimi_k3_clean.txt`. The video itself is the primary source for the
numbers in this study; they have not been independently reproduced.

### 2.2 Why this is a Tessera-shaped problem

The Tessera fork's ggml-ane backend, the `tools/quantize/tessera/` suite
(awq/imatrix/vec/dispatch/progress), and the server's admission/metrics/
prefill-policy files all deal with the compute and concurrency side of
Apple Silicon inference. The disk-bound MoE regime inverts the bottleneck
stack:

| Phase share in dense-model decode   | ~10-30% compute, bandwidth-limited |
| Phase share in disk-offloaded MoE    | ~20% compute, ~53% disk, <3% attn  |

In the dense regime, the ANE/Metal kernel work is the highest-leverage
investment. In the disk-offloaded MoE regime, the ANE/Metal work only
starts to bite *after* the disk problem is solved, because compute drops
from the dominant share to a fifth of wall time. The two efforts are
sequential, not substitutes.

The fork's existing memory-discipline work (server-admission's dynamic
admission control, server-prefill-policy's budget awareness) is more
directly relevant here than the kernel work: WASTE's inverted memory curve
is the same class of self-throttling admission control, just at the weight-
  cache layer instead of the KV-cache layer.

---

## 3. Hardware budget on M2 Ultra Mac Pro

### 3.1 PCIe lanes

The M2 Ultra die has 32 PCIe Gen4 lanes total but only exposes 24 to the
slots through a PCIe switch. The Mac Pro has six physical Gen4 slots (two
x16, four x8 electrically), but they share the 24-lane budget. Two dual-
width x16 cards run at x8/x8 electrically - they cannot both get x16. This
is the single hardest constraint in this study. See Apple Support 104947
and Softron's M2 Ultra PCIe limits article.

The slots do not support third-party GPUs and do not support PCIe lane
bifurcation switching (Intel-style partition). This rules out the Asus Hyper
M.2 x16 carrier card, which needs bifurcation and on a Mac only exposes 1
of 4 installed drives. The card that actually works is the HighPoint
SSD7749M2 (or SSD7101A-1), which carries its own PCIe switch onboard so the
host does not need to bifurcate. HighPoint launched the SSD7749M2 series
specifically for the Mac Pro M2 Ultra.

### 3.2 Thunderbolt 4

Eight TB4 ports (six rear, two top), six independent controllers. Ports do
not share buses the way Intel Macs did. The TB4 controllers are wired into
the SoC separately from the 24 slot lanes, so filling the TB4 ports is
additive to slot bandwidth - it does not steal from the slot budget.

The "40 Gbps" headline is misleading. TB4 allocates only ~32 Gbps to PCIe
tunneling; after encoding overhead the real-world NVMe ceiling is ~2.8
GB/s per port (sometimes 3.1-3.8 GB/s with the newest ASM2464PD enclosures
and top-tier Gen4 drives). A TB4-attached NVMe is roughly 40% the speed of
a slot-attached one (~7 GB/s per Gen4 x4 drive). See dancharblog's chipset
survey for the per-chipset breakdown.

### 3.3 Realistic aggregate bandwidth

Synthetic RAID0 benchmarks on Mac Pro hardware:

- HighPoint SSD7101A-1 on M2 Ultra (x8): ~13.8-14.1 GB/s (HighPoint's own
  M2-Ultra document).
- HighPoint SSD7749M2 fully populated, Gen4 x16: up to 28 GB/s rated.
- Real user, 8x Samsung 980 Pro on HighPoint, Blackmagic: ~8 GB/s.
- SoftRAID, 16 blades on a 2019 Mac Pro: 17 GB/s.

A realistic, repeatable number for a loaded SSD7749M2 on M2 Ultra is mid-
teens to low-20s GB/s, not the 28 GB/s spec.

Aggregate budget combining slots and TB4:

- 2x HighPoint (slot, x8/x8 electrically): ~28-32 GB/s
- + 6x TB4 enclosures: ~16 GB/s more
- Hard ceiling: ~45 GB/s theoretical, realistically ~35-40 sustained.

The M3 Ultra Mac Studio is a different animal: Thunderbolt 5 at 80 Gbps per
port (~6 GB/s real per port). Six independent TB5 controllers would give
~36 GB/s over TB alone, plus slots. If hardware has not yet been bought,
M3 Ultra's TB5 changes the price/perf math for this workload and is worth
pricing out before committing to M2 Ultra.

### 3.4 What this means for K3

K3 reads 17 GB per token. 1 tok/s needs 17 GB/s sustained. The arithmetic:

- Single internal drive: 0.5 tok/s (WASTE measured, 77% of drive bandwidth
  already utilized).
- 1x HighPoint: ~1.0-1.2 tok/s.
- 2x HighPoint: ~1.5-1.8 tok/s.
- 2x HighPoint + 6x TB4: ~1.8-2.2 tok/s.
- Theoretical max: ~2.3 tok/s.

10 tok/s would need 170 GB/s. That is 11-12 top-end Gen5 drives striped
together, before buying any RAM. The M2 Ultra's entire slot+TB budget
cannot reach halfway. For K3 specifically, the disk-offload story gets you
to the "1 tok/s floor" the video describes; it does not break through it.

For dense sub-100B models, the same multiplier scales differently. A 48B
model at 10.7 tok/s on one drive would land ~25-30 tok/s on a 2x HighPoint
config - genuinely fast. The disk-offload investment pays back much more
clearly on normal-sized models than on K3.

---

## 4. The macOS page-cache problem and how to bypass it

### 4.1 The problem

macOS, like all modern Unix kernels, uses the unified buffer cache (UBC) to
cache file pages. When a process reads from a file, the kernel pages the
file into the UBC on the assumption that those pages will be reused soon.
For an OS that does not know what the application is doing, this is a
reasonable default. For a disk-offloaded MoE engine it is actively
hostile:

- Expert weights are read once per token and (with N >> 16 experts per
  layer) are almost never re-read on the next token. Caching them is pure
  waste.
- The UBC grows under read pressure. If it grows large enough, the kernel
  starts evicting other memory - including, in WASTE's case, the resident
  trunk that must never leave RAM. Once the trunk pages out, every token
  faults its own attention weights back off the same drive that is already
  saturated, and throughput collapses.
- WASTE's inverted memory curve is exactly this: 46 GB cache budget is
  optimal (just under the OS's evict threshold); 58 GB pushes past it,
  triggers trunk eviction, and the engine runs 25x slower.

The kernel's page-replacement decisions are opaque to the application. "A
cache you do not control is not a cache."

### 4.2 The bypass

macOS does not have Linux's O_DIRECT. The equivalent is
`fcntl(fd, F_NOCACHE, 1)`, which is what WASTE uses. Properties:

- F_NOCACHE on read tells the UBC not to retain the pages. This is the
  property that prevents the inverted-memory cliff: expert reads do not
  pollute the cache, so the kernel has no reason to evict the trunk.
- The resident trunk is pinned with `mlock` (or equivalently kept wired),
  so even under memory pressure the kernel cannot reclaim it.
- F_NOCACHE is *advisory-ish*. For best behavior: page-aligned offsets and
  lengths (WASTE's 4 KB alignment is correct), page-aligned buffers via
  `posix_memalign`, and access patterns that do not fight the kernel's read-
  ahead heuristics.
- F_NOCACHE works on regular files on APFS. No need for raw /dev/diskN
  access, which on macOS requires unmounting and is fragile.

Apple's fcntl(2) man page documents F_NOCACHE. There is no formal promise
that the kernel will *never* cache (the flag is a hint), but in practice on
large sequential reads it behaves as expected and is the only portable knob
macOS gives you.

### 4.3 Async I/O on macOS

Options, weakest to strongest:

1. Thread pool doing blocking `pread` with F_NOCACHE, one thread per drive.
   Simplest, and for big sequential expert reads it is basically as fast as
   anything else. The kernel merges, the drive does NCQ. This is what to
   start with.
2. `dispatch_io` (GCD). Apple's blessed async I/O. Channels per drive,
   backpressure for free. More machinery, same eventual throughput.
3. POSIX `aio_read`. Exists on macOS, historically mediocre single-queue
   performance. Skip.

For N drives, N threads each doing blocking pread is fine. The OS scheduler
does the parallelism.

### 4.4 The hidden catch with RAID volumes

This is the most important caveat in the study. If you put a RAID volume
between the engine and the disks - whether macOS CoreStorage, SoftRAID, or
an APFS striped volume - you reintroduce an opaque buffer. The RAID driver
does its own buffering, and F_NOCACHE on the volume's file may not
propagate through to the underlying disks. The whole point of bypassing the
page cache is defeated.

The fix is the reframe in the next section: do not build a RAID volume.
Keep N independent filesystems (one per drive) and let the engine decide
which drive to read each expert from. Then F_NOCACHE applies cleanly per
file, on each underlying disk.

---

## 5. Stripe in the application, not the block layer

### 5.1 The reframe

WASTE already opens its weight file with F_NOCACHE and reads experts
directly. The page-cache problem only appears when a RAID volume is
interposed. So the answer is: do not RAID at the block layer. Keep N
independent filesystems (one per drive), and let the engine decide which
drive to read each expert from.

This is not a RAID. It is a sharded weight store with direct I/O. It is the
natural generalization of WASTE's single-drive design to N drives.

### 5.2 Why the MoE structure suits this

Experts are independent tensors. For layer L the router picks 16 of 896.
Each expert is a self-contained, 4 KB-aligned record in WASTE's layout.
This means:

- Placement: expert (L, E) lives on drive `hash(E) mod N`, at a known
  offset.
- Per-layer decode: the 16 active experts are almost certainly spread
  across all N drives (16 of 896, with hash-distribution, lands ~16/N per
  drive; even N=16 leaves sparse overlap). Issue 16 reads in parallel, one
  per drive.
- No cross-drive stripe alignment. No slowest-member problem. No buffer
  layer.

This also dodges the stripe-unit-vs-record-boundary mismatch that
block-layer RAID introduces. A software RAID0 stripe unit is typically
64 KB-1 MB and does not align to expert record boundaries, so reads
straddle stripes and need extra coordination. With application-level
placement there is no stripe unit. Each expert is one pread.

### 5.3 Heterogeneous drives are fine

A block-layer RAID0 stripe mixing TB4-attached drives (2.8 GB/s) and PCIe-
attached drives (7 GB/s) waits on the slowest member at every stripe
boundary. With application-level placement, slow drives simply serve fewer
experts per layer; the engine issues more reads in parallel than the layer
needs and waits on the first 16 to return. Heterogeneous speeds are fine,
not a problem.

This is the structural reason application-level sharding beats block-layer
RAID for this workload: RAID requires homogeneity, MoE expert reads are
embarrassingly parallel and tolerate heterogeneity.

---

## 6. RVQ and direct-out-of-codebook multiply

### 6.1 WASTE's quantization

WASTE compresses experts hard to make the read smaller. Residual vector
quantization (RVQ): three stages of 256-entry codebooks over 8-dimensional
vectors, landing at ~3 bits per weight. Moonshot trained Kimi K3 with
quantization-aware training on the experts, which is why this survives at
all.

The full weight matrix is never rebuilt in memory. The engine multiplies
straight out of the codebooks: build a small table of partial dot products
once, and every expert row becomes three table lookups and two additions.
There is no decompression step to pay for.

This is a Tessera-vec-shaped idea. The `tools/quantize/tessera/` suite
already targets the same vector-quantization lever (tessera-vec.cpp, the
IQ- quant lineage, the T640 ternary work). WASTE's RVQ is a different
point in the design space (codebook-based, not ternary) but the principle
is the same: get the weight small enough that the disk read is the
bottleneck, not the decompression.

### 6.2 Accuracy data

The video cites ~19.4% per-weight error for WASTE's 3-bit experts against
Moonshot's 4% native format, but logits agree with PyTorch to 3.6e-6 and
the vision tower matches to 2.3e-something. These are floating-point
rounding differences at the output, despite large per-weight error. This
is the same empirical pattern Tessera's own runtime-aware-pipeline
Layer-1 fidelity work documents (see runtime-aware-pipeline.md and
septq-retrospective.md): per-weight error and downstream-output error
decouple for well-designed vector-quant schemes, because errors cancel
across the dot product.

The WASTE accuracy numbers are a useful validation reference for whatever
codebook scheme tessera-vec produces. If 19.4% per-weight error yields
3.6e-6 logits agreement, the bar for a disk-offload-friendly quant is
"preserves dot-product output," not "preserves per-weight values."

### 6.3 Direct-out-of-codebook multiply

The "multiply straight out of the codebooks, no decompress step" trick is
the deeper idea. The standard pipeline is: read packed weights, decompress
to fp16, multiply. WASTE collapses this to: read codebook indices, do
table lookups + adds during the dot product. The decompress step is never
materialized.

This is exactly what the existing kernel_TILE640_MATMUL in ggml-metal
already does for T640 ternary weights: each page is decoded into
threadgroup memory and consumed by the dot-product loop in the same kernel,
never round-tripping through device memory (see interleaved-kernel-design-
study.md section 6.2). The same fusion idea applies to any codebook-
quantized scheme, and the disk-offload case is no different: fetch
codebook indices from disk, decode-and-multiply in one fused kernel. The
disk read shrinks to the codebook-index size, which for 3 bpw RVQ is the
same ~3 bits/weight as T640's packed ternary.

---

## 7. Resident-trunk split and the inverted memory curve

### 7.1 The split

WASTE cuts the model in two. The resident trunk (~27.28 GB for K3 at 4k
context) holds everything that fires on every token: embeddings, attention
layers, routers, output head, quantized to 4 and 8 bits. It stays in RAM
permanently, mlocked. Everything else - the experts - lives on the drive
and is fetched on demand.

The trunk is almost the entire RAM floor. 29.05 GB at 4k context, of which
~27 GB is trunk. The engine refuses to start below it. This is an unusually
disciplined design choice.

### 7.2 Context sensitivity

The 29 GB headline number comes with a short conversation attached. Push
to 128k context and the floor becomes 35.6 GB. Use the full million-token
context K3 advertises and you need 83 GB of RAM. The trunk is not strictly
constant; attention-cache size grows with context. The headline 29 GB
number is measured at 4k tokens.

### 7.3 The inverted memory curve

The most counter-intuitive finding in WASTE's profiling:

| RAM cache budget | Throughput |
|------------------|------------|
| 32 GB            | 0.5 tok/s  |
| 46 GB            | 0.54 tok/s (best) |
| 52 GB            | 0.04 tok/s |
| 58 GB            | 0.02 tok/s |

Going from 46 GB to 58 GB cache budget improves the expert hit rate and
makes the engine ~25x slower. Past the OS's comfort point, macOS starts
paging out the trunk - the 27 GB that must never leave RAM leaves RAM.
Now every token faults its own attention weights back off the same drive
that is already saturated.

The engine walks its own budget down 1-17 GB working set at a time until
the whole thing fits under ~7/8 of physical memory. Steering around a
cliff it knows is there rather than tuning for the best case. This is the
same class of self-throttling admission control that server-admission.cpp
and server-prefill-policy.cpp already implement for the KV cache, just at
the weight layer. The pattern transfers directly.

### 7.4 Attention gets the same discipline

By folding one projection into the query and output paths, WASTE caches a
512-wide latent instead of full keys and values. 11.25 GB of KV cache
becomes 0.21 GB, 53x smaller, and without it the RAM floor would not fit
on a laptop at all. This is a variant of the grouped-query-attention /
latent-KV idea that the inference-engine comparison study (section 10.4)
discusses for dense-model serving; WASTE applies it more aggressively
because the constraint is RAM floor, not decode bandwidth.

### 7.5 Implication for cache size

Under the 1-token working set, caching stops meaning anything. An expert
cached for this token is evicted before the next token asks for it. As
WASTE's README puts it: the hit rate is not low, it is zero. So the cache
budget is purely a defensive mechanism to keep the OS from taking memory
back, not a performance optimization. This is the opposite of the dense-
model intuition where bigger KV cache = more concurrency = more throughput.

---

## 8. Read-ahead / I/O-compute overlap

### 8.1 The single biggest WASTE win

On 31 July, one commit: "read the next expert while computing this one."
Two threads reading ahead, so the disk and the arithmetic overlap instead
of taking turns. That single change took K3 from ~0.32 tok/s to 0.5 tok/s,
+60% in a day.

This is the dominant lever in the disk-bound regime. It is also the same
idea the fork's interleaved-kernel-design-study.md explores for compute
(prefill-dequant-decode overlap inside GPU kernels), just at the host-to-
disk layer instead of host-to-accelerator. The pattern is identical: a
memory-latency-bound phase (disk read) overlaps with a compute-bound phase
(matmul) because their resource profiles are complementary.

### 8.2 N-drive generalization

With one drive, WASTE's two read-ahead threads saturate the single disk.
With N drives, the parallelism is multiplicative: each drive has its own
read-ahead pipeline, so N drives support N concurrent prefetch streams.
The engine prefetches layer L+1's experts (which live on drives
hash(E)%N for each active E) into a small ring buffer per drive, while
compute drains layer L. Producer-consumer per drive, with a fence at the
layer boundary.

This is where the placement hash matters. With hash(E)%N placement, the 16
active experts of layer L+1 are distributed across drives roughly evenly
(16 of 896, with hash distribution). Each drive's read-ahead thread knows
in advance which experts it needs next layer and issues the preads before
the layer L matmul starts. By the time the matmul finishes, the next
layer's experts are in the staging buffers.

### 8.3 Queue depth and per-drive parallelism

A single expert read is large (~1 MB scale, depending on expert size and
quantization). A single pread saturates a drive's bandwidth for the
duration of the read. To use N drives fully, the engine needs at least one
outstanding read per drive at any time, which the per-drive read-ahead
thread provides. Deeper queueing per drive (multiple outstanding reads)
helps if the drive has internal parallelism (most modern NVMe do, but the
benefit saturates at queue depth ~4-8 for sequential reads of this size).

The thing NOT to do: issue all 16 layer reads from a single thread in a
tight loop. That serializes the issue and caps throughput at one drive's
bandwidth. The per-drive thread model is what gets you to N * bandwidth.

---

## 9. Mapping onto the Tessera fork

### 9.1 What llama.cpp does today

llama.cpp loads weights via `llama_mmap` (kernel-paged, UBC-cached) or
`--mlock` (pinned, wired). The MoE experts are individual tensors in the
GGUF, addressed by offset. For models that fit in RAM this is correct and
optimal. For models that do not fit in RAM, llama.cpp's current path is
"let the kernel page weights in and out," which works but does not control
the eviction policy and is exactly the failure mode WASTE bypasses.

The Tessera fork's additions - server-admission (dynamic admission control
+ recompute preemption), server-metrics, server-prefill-policy (budget-
aware prefill) - all deal with KV-cache discipline under concurrency, not
weight-cache discipline. They are conceptually adjacent but do not address
the disk-offload case directly.

### 9.2 The new path, opt-in

The disk-offload path would be opt-in (a new flag, gated behind
LLAMA_USE_SHARD_IO or similar, off by default). It would consist of:

Build time (quantize tool):
- A `tessera-shard` step that takes a GGUF and emits a manifest (tensor
  -> {drive_index, offset, len}) plus N weight blobs sized to fit the
  target drives. This is mostly a copy/repack of the existing
  tessera-vec.cpp output pipeline, with placement-hash assignment.

Load time (llama.cpp):
- A new `llama_shard` loader alongside `llama_mmap` that opens N files
  with F_NOCACHE, mmaps+mlocks the resident trunk (attention/router/embed/
  head), and registers expert tensors as "lazy, fetch on demand."

Decode time (ggml):
- A hook in the MoE op that, before computing layer L's experts, issues
  the parallel preads for the active 16 and waits on the read-ahead fence.
  The compute then multiplies out of the staging buffer the same way WASTE
  multiplies out of codebooks, and the same way kernel_TILE640_MATMUL
  multiplies out of threadgroup memory.

### 9.3 Smallest viable version

Single-drive F_NOCACHE + async read-ahead on a vanilla GGUF, no manifest
format, no multi-drive sharding. This is ~600 lines and reproduces WASTE's
result on existing hardware (any MacBook with a fast internal drive). It
validates the cache-bypass + read-ahead pipeline before any hardware spend
on multi-drive configurations. It is the right first slice.

Multi-drive is then mostly placement hash + per-drive threads, layered on
top of the single-drive I/O path. The single-drive slice de-risks both the
F_NOCACHE behavior on macOS (which needs empirical validation, not just
man-page reading) and the read-ahead / compute overlap design.

### 9.4 What this is NOT

- It is not a replacement for mmap. mmap remains the right default for any
  model that fits in RAM. The shard path is for models that do not.
- It is not a replacement for the ANE backend or the kernel work. Those
  are compute-side; this is I/O-side. They compose.
- It is not a server-level feature. It is a loader-level feature that the
  server consumes transparently.

---

## 10. What does NOT translate from WASTE

### 10.1 Pure-C single-binary philosophy

WASTE is ~6,000 lines of C, no Python, no CUDA, no BLAS, hand-built for
one model. The opposite philosophy from llama.cpp's multi-backend
generality. The I/O techniques (F_NOCACHE, application-level sharding,
read-ahead overlap) translate; the architecture does not. Tessera's
disk-offload path should layer on ggml, not replace it.

### 10.2 The no-page-cache stance as universal default

WASTE bypasses the page cache always, because its workload is one-shot
expert reads. For dense models that fit in RAM, the page cache is the
right mechanism and mmap is the right loader. The F_NOCACHE path is opt-in
for the disk-offload regime, not a new default.

### 10.3 Hardware RAID volumes

Covered in section 4.4. The HighPoint card's onboard RAID is convenient
but reintroduces a buffer layer. The application-level sharding path uses
the HighPoint card as a drive multiplexer (its onboard PCIe switch exposes
individual drives to the OS), not as a RAID controller. Each drive gets
its own filesystem and its own F_NOCACHE file.

### 10.4 Kimi K3's bespoke license and the revenue clause

Not a technical issue, but worth flagging: K3 is open-weights with a
revenue clause, not open source. Any Tessera-side work that targets K3
specifically should be measured against the license terms. This is a
non-issue for the I/O path itself, which is model-agnostic.

---

## 11. Confidence and open questions

### 11.1 High confidence

- The 24-lane PCIe budget on M2 Ultra. Apple's own spec (Support 104947).
- F_NOCACHE behavior on large sequential reads. Apple fcntl(2) man page,
  widely used in production on macOS for exactly this purpose.
- TB4 ~2.8 GB/s real-world per-port ceiling. dancharblog's chipset survey,
  multiple corroborating benchmarks.
- The MoE expert placement is embarrassingly parallel and tolerates
  heterogeneous drives. Architectural argument, not empirical.
- Compute is ~20% of decode in the disk-bound regime; the video's
  profiling is internally consistent with the throughput numbers.

### 11.2 Medium confidence

- Real-world aggregate bandwidth on M2 Ultra with loaded HighPoint cards
  (~28-32 GB/s for 2 cards). Vendor specs say 28 GB/s per card, real-world
  user benchmarks say 8-14 GB/s. The truth is in between and depends on
  SSD choice and queue depth.
- Whether the HighPoint onboard switch truly exposes individual drives to
  macOS without an intervening buffer (vs presenting a hardware RAID
  volume). Needs verification with a loaner card before committing to the
  application-level sharding design.
- Whether two HighPoint cards running at x8/x8 electrically actually
  deliver additive bandwidth, or whether the M2 Ultra's PCIe switch clamps
  them. The architecture docs say additive, but no independent test maxes
  both slots simultaneously.

### 11.3 Low confidence / open questions

- The WASTE throughput numbers themselves (0.5 tok/s on K3, 10.7 tok/s on
  48B) come from a 4-day-old README via a video, not independent
  reproduction. Treat as plausible but unvalidated.
- Whether F_NOCACHE propagates cleanly through a software RAID volume
  (CoreStorage / SoftRAID) to the underlying drives. The presumption in
  this study is "it does not," which is the basis for recommending
  application-level sharding over block-layer RAID. This needs empirical
  confirmation on real RAID driver versions.
- The M3 Ultra TB5 math (~36 GB/s over TB alone) is from spec sheets, not
  real-world multi-port saturation tests. The asymmetric-bandwidth mode
  (120 Gbps in one direction) may matter for read-only workloads but is
  not characterized here.
- Whether the WASTE-style RVQ quantization is even applicable to the
  Tessera T640 / IQ- quant formats. The two are different points in the
  design space; the principle (direct-out-of-codebook multiply) translates
  but the specific codebook layout does not.

---

## 12. Recommendations (deferred)

### 12.1 What to do now

Nothing implementation-wise. This study exists so the design space is
captured. The concrete triggers that should revisit it:

- A real K3-class GGUF that a user wants to run overnight on Mac hardware.
- The ANE backend (D1) reaching a point where compute stops being the
  dominant share of decode for some workload class.
- A hardware purchase decision (Mac Pro M2 Ultra + HighPoint cards) that
  requires the bandwidth math in this study to justify.

### 12.2 When the trigger fires, what to build first

The single-drive F_NOCACHE + async read-ahead slice on a vanilla GGUF
(section 9.3). ~600 lines, testable on existing hardware, validates the
core cache-bypass and overlap design. Multi-drive sharding layers on top
once the single-drive path is proven.

### 12.3 What NOT to build first

- A multi-drive sharded store without first validating single-drive
  F_NOCACHE behavior on macOS. Debugging placement and I/O at the same
  time is the failure mode.
- A custom block-layer RAID driver. The page-cache problem is solved by
  not having a RAID layer, not by building a better one.
- A K3-specific path. The I/O techniques are model-agnostic; build the
  general disk-offload loader and let K3 be one user.

### 12.4 Sequencing relative to other Tessera work

The ANE backend (D1) and the parametric-kernel work are higher-leverage on
the compute side and should land first. The disk-offload path becomes
relevant only when compute drops from the dominant share of decode - which
is exactly what the ANE/kernel work will cause for the largest model
classes. The two efforts are sequential, not substitutes.

---

## 13. Sources

Primary external sources (web-verified 2026-08-01):

- Cloud Codes, "Run Kimi K3 on a Laptop With 32 GB RAM (No GPU Needed)",
  YouTube Et_I4fj1etk. Transcript preserved at /tmp/kimi_k3_clean.txt.
  All WASTE throughput and profiling numbers in this study come from this
  video and have not been independently reproduced.
- Apple Support, "Install PCIe cards in your Mac Pro (2023)":
  https://support.apple.com/en-us/104947
- Softron, "Mac Pro (2023 - M2 Ultra) - Limits and possibilities of PCIe
  slots":
  https://softron.zendesk.com/hc/en-us/articles/9209583829404
- HighPoint, "Dual-Width NVMe Gen 4 AIC Series for Apple Mac Pro M2 Ultra":
  https://www.highpoint-tech.com/post/highpoint-launches-dual-width-nvme-gen-4-aic-series-for-apple-mac-pro-m2-ultra
- Tom's Hardware, SSD7749M2 coverage:
  https://www.tomshardware.com/pc-components/ssds/raid-card-delivers-128tb-of-nvme-storage-at-28-gbs-speeds-highpoint-ssd7749m2-houses-up-to-16-m-2-2280-ssds
- Logik.tv forum, real-world NVMe RAID0 on Mac Pro:
  https://forum.logik.tv/t/mac-pro-nvme-raid0-performance/3380
- Larry Jordan, SoftRAID scaling to 17 GB/s:
  https://larryjordan.com/articles/tips-to-maximize-the-speed-of-ssd-raid-storage/
- MacRumors, Asus M.2 cards on Mac (bifurcation issue):
  https://forums.macrumors.com/threads/asus-m-2-card.2216309/
- dancharblog, NVMe enclosure chipset survey (TB4 ~2.8 GB/s ceiling):
  https://dancharblog.wordpress.com/2024/01/01/list-of-ssd-enclosure-chipsets-2022/
- OWC Express 4M2, USB4 NVMe RAID enclosure spec:
  https://www.owc.com/solutions/express-4m2
- Gearspace, M2 Ultra PCIe lanes / TB controller topology:
  https://gearspace.com/boards/music-computers/1409189
- Apple fcntl(2) man page, F_NOCACHE:
  https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html

Adjacent Tessera docs referenced (in-tree):
- docs/ane-backend-deep-study.md (compute side; IOSurface + Metal Event
  architecture that the disk-offload path composes with)
- docs/interleaved-kernel-design-study.md (intra-kernel overlap; the same
  producer-consumer pattern at the GPU layer)
- docs/inference-engines-comparison-study.md (concurrency policy; server-
  admission and server-prefill-policy as adjacent memory-discipline work)
- docs/runtime-aware-pipeline.md (Layer-1 fidelity; the per-weight-error-
  vs-output-error decoupling that WASTE's accuracy data corroborates)
- docs/septq-retrospective.md (same fidelity pattern)
- tools/server/server-admission.h, server-prefill-policy.h (existing
  memory-discipline surfaces the disk-offload path would compose with)

End of document.

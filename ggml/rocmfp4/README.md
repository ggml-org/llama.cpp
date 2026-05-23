# ROCmFP4

ROCmFP4 is the experimental `Q4_0_ROCMFP4` GGUF tensor type for Strix Halo.
The current AMD-tuned variant stores 32 weights per block as packed
E2M1-derived 4-bit values plus two unsigned E4M3 scale bytes, one per
16-weight half, for 18 bytes per block. The codebook stores half-scale signed
integer levels up to `10`, representing `5.0` raw-scale units after the scale
factor is applied. This keeps outlier pull lower than the original wider test
range while preserving a fast integer-dot backend shape.

This directory owns the format-specific implementation. The rest of ggml only
registers and dispatches the type.

Current status:
- The format runs on CPU, Vulkan, and ROCm/HIP in this custom tree.
- `Q4_0_ROCMFP4` is the pure 4.50 BPW dual-scale path.
- `Q4_0_ROCMFP4_LEAN` keeps ROCmFP4 for dense tensors but protects token
  embeddings with `Q5_K`. On the Qwen3-4B Strix test pass this closed most of
  the Q4_0 coherence gap while staying around normal Q4 size.
- `Q4_0_ROCMFP4_COHERENT` protects token embeddings with `Q6_K` and is the
  quality-first ROCmFP4 preset.
- `Q4_0_ROCMFP4_FAST` is the 4.25 BPW single-scale speed path. It is the
  smallest and fastest decode variant, but the pure version gives up too much
  PPL to be the default quality target.
- `Q4_0_ROCMFP4_FAST_COHERENT` combines the fast 4.25 BPW transformer layout
  with `Q6_K` token embeddings. On the Qwen3-4B Strix test pass it is the
  current balanced AMD target: smaller than `Q4_0`, faster than `Q4_0` on
  decode on both Vulkan and ROCm, and close to `Q4_0` PPL on the short
  WikiText-2 check.
- `Q4_0_ROCMFP4_STRIX` is the current quality-biased Strix Halo preset. It
  keeps most transformer tensors on `Q4_0_ROCMFP4_FAST`, protects token
  embeddings with `Q6_K`, and uses the dual-scale `Q4_0_ROCMFP4` layout for
  attention-K and attention-V tensors. On Qwen3-4B it improved the short
  WikiText-2 PPL to `13.8865` at `4.49 BPW` while still beating the
  same-flags stock `Q4_0` decode baselines on both Vulkan and ROCm.
- `Q4_0_ROCMFP4_STRIX_LEAN` is the compact Strix Halo preset. It keeps the
  STRIX all-layer dual-scale attention-K/V protection, uses the FAST
  single-scale transformer layout for the dense tensors, and protects token
  embeddings/output with `Q5_K` instead of `Q6_K`. On the Qwen3-4B validation
  pass it landed at `4.38 BPW`, improved short WikiText-2 PPL versus
  `FAST_COHERENT`, and kept Vulkan decode in the `81 tok/s` band.
- A smaller first/last-layer-only K/V protection recipe was tested but not
  promoted. It reached `4.48 BPW`, `80.13` Vulkan decode, and `69.85` ROCm
  decode, but PPL regressed to `14.0167`, so the all-layer K/V STRIX preset
  remains the quality target.
- The ROCm/HIP MMQ path for `Q4_0_ROCMFP4_FAST` uses one scale per 32-weight
  block, matching the actual FAST layout instead of duplicating the scale into
  two half-block slots.
- The ROCm/HIP vector-dot and MMQ loaders use a ROCmFP4-owned Codebook10
  expander backed by AMD `amdgcn_perm` constants. This avoids the generic
  table-load helper on the hot ROCm path.
- ROCm/HIP fallback dequant, copy, get-rows, GPU-side quantization scoring,
  and standalone dequant helpers use ROCmFP4-owned HIP helpers for finite
  scales and Codebook10 nibbles instead of relying on generic FP8 handling.
  This keeps non-MMQ conversion paths aligned with the custom AMD format.
- UE4M3 scale decode in the ROCm/HIP software path uses a ROCmFP4-owned finite
  scale decoder. It avoids `ldexpf`, builds normal FP32 values directly from
  exponent/mantissa bits, and skips the generic FP8 NaN handling because
  ROCmFP4 row validation already rejects non-finite scale bytes.
- ROCm/HIP dequant conversion kernels use the same ROCmFP4 finite scale
  decoder, keeping tensor conversion aligned with the hot MMQ/MMVQ backend
  path instead of falling back to the generic FP8 helper.
- ROCm/HIP backend CPY now advertises and executes quantized
  `Q4_0_ROCMFP4 -> F32` and `Q4_0_ROCMFP4_FAST -> F32` conversion paths.
  This keeps diagnostic graph ops and fallback tensor conversion inside the
  custom AMD decoder instead of being rejected by backend capability checks.
  The custom q-to-f32 wrappers launch by ROCmFP4 block count, avoiding idle
  per-element blocks for this format's 32-value conversion kernels.
- ROCm/HIP backend CPY also supports `F16 -> Q4_0_ROCMFP4`,
  `F16 -> Q4_0_ROCMFP4_FAST`, `BF16 -> Q4_0_ROCMFP4`, and
  `BF16 -> Q4_0_ROCMFP4_FAST`. The kernels convert each 32-value half/bfloat
  block to local FP32 and then run the same exhaustive ROCmFP4 scale search,
  so runtime graph copies keep the coherence-first quantizer instead of
  falling back to unsupported behavior.
- ROCm/HIP backend CPY supports same-type packed-block copies for
  `Q4_0_ROCMFP4 -> Q4_0_ROCMFP4` and
  `Q4_0_ROCMFP4_FAST -> Q4_0_ROCMFP4_FAST`, including block-aligned views.
  The kernel copies the packed 18-byte or 17-byte ROCmFP4 blocks directly, so
  graph/view copies preserve exact bytes and avoid dequantize/requantize
  fallback behavior. The launcher uses normal multi-thread HIP workgroups
  rather than one-thread launches, so large packed-view copies scale with the
  number of ROCmFP4 blocks.
- ROCm/HIP backend GET_ROWS supports both ROCmFP4 layouts. This gives pure
  ROCmFP4 tensors the same direct row-gather coverage as stock small-block
  quants on ROCm and keeps embedding-row access on the custom finite-scale
  decoder.
- ROCm/HIP `MUL_MAT` support now covers `F16` activation tensors for both
  ROCmFP4 layouts. The backend stages half activations to contiguous FP32 on
  the GPU, including non-contiguous/views, then feeds the existing Q8
  activation quantizer and ROCmFP4 MMVQ/MMQ kernels. This keeps the forward
  path on the AMD backend instead of rejecting the op and falling through to a
  slower dequantized matrix path. The generic matmul runtime guard explicitly
  allows this ROCmFP4 x F16 forward-inference case, so the support probe and
  execution wrapper agree for batched activations.
- The standalone HIP dequant skeleton covers both the dual-scale and FAST
  single-scale layouts, so future fused ROCm kernels can target the current
  balanced FAST artifact without reintroducing the older scale path.
- Vulkan ROCmFP4 shaders also decode UE4M3 scales directly to the half-scale
  value used by the codebook, matching CPU/HIP and avoiding repeated `* 0.5`
  fixups at dequant and matmul call sites.
- Vulkan ROCmFP4 shaders keep a shared `kvalues_rocmfp4` Codebook10 table.
  Arithmetic/direct Codebook10 decode variants compiled and preserved
  coherence, but measured slower on Strix Halo Vulkan, so the table path
  remains the active backend implementation.
- Vulkan `Q4_0_ROCMFP4_FAST` matvec/MMQ kernels have a single-scale dot
  specialization. They combine the two half-block dot sums and apply the one
  FAST scale once, instead of taking the dual-scale path used by
  `Q4_0_ROCMFP4`.
- Vulkan backend CPY/SET_ROWS now has generated `F32 -> Q4_0_ROCMFP4`,
  `F32 -> Q4_0_ROCMFP4_FAST`, `Q4_0_ROCMFP4 -> F32`,
  `Q4_0_ROCMFP4_FAST -> F32`, and indexed SET_ROWS shaders. The SET_ROWS path
  uses the same exhaustive finite UE4M3 scale search as the CPU reference, so
  quantized K/V cache writes favor coherence over a cheap max-abs shortcut.
- Vulkan `F32 -> ROCmFP4` runtime quantization now uses the same exact ordered
  UE4M3 scale search as CPU/HIP: find the scale nearest `max_abs / 10`, expand
  outward, and stop a candidate once its partial error cannot beat the current
  best scale. The candidate set is unchanged, so this avoids a slower linear
  scan without falling back to a lower-quality shortcut.
- Vulkan backend CPY also supports `F16 -> Q4_0_ROCMFP4`,
  `F16 -> Q4_0_ROCMFP4_FAST`, `BF16 -> Q4_0_ROCMFP4`, and
  `BF16 -> Q4_0_ROCMFP4_FAST`. The runtime quantization shader can load
  `float`, `float16_t`, and BF16 source bits and then runs the same exact
  ordered finite UE4M3 scale search. Backend tests keep a bounded NMSE
  tolerance only for the half/bfloat runtime quantization cases because those
  paths are inherently lossy around source-precision tie points; the
  `F32 -> ROCmFP4` checks remain strict.
- Vulkan same-type CPY supports packed-block copies for
  `Q4_0_ROCMFP4 -> Q4_0_ROCMFP4` and
  `Q4_0_ROCMFP4_FAST -> Q4_0_ROCMFP4_FAST`, including block-aligned
  non-contiguous/permuted/view copies. The non-contiguous path uses a
  byte-addressed block shader and preserves exact 18-byte or 17-byte
  ROCmFP4 blocks; contiguous FAST copies keep the existing direct byte-copy
  fast path.
- Vulkan scalar FlashAttention can now decode ROCmFP4 and ROCmFP4_FAST K/V
  cache blocks. ROCmFP4 K/V is forced to the scalar FA path because the current
  custom decode is not a coopmat/native matrix-core FP4 path.
- Vulkan scalar FlashAttention can use the integer-dot MMQ K path for both
  ROCmFP4 K-cache layouts. The FAST layout expands each 4-bit Codebook10 value
  into packed signed int8 lanes and uses its single UE4M3 scale as the K block
  multiplier. The dual-scale layout also uses packed signed int8 lanes, but
  splits the dot accumulation by half-block so each 16-value half uses its own
  UE4M3 scale. This keeps the quality-biased STRIX K-cache path fast without
  applying one scale to a two-scale block.
- Build and runtime verification generated the Vulkan SPIR-V entries for
  ROCmFP4 copy/SET_ROWS shaders, linked `libggml-vulkan.so`, and passed
  Vulkan ROCmFP4 CPY plus MUL_MAT smoke tests on Strix Halo.
- Row validation rejects invalid scale bytes outside finite unsigned UE4M3
  (`0x00` through `0x7e`) so corrupted custom GGUF tensors fail early.
- Quantization keeps the exhaustive 126-scale UE4M3 search for both normal and
  imatrix paths. Candidate-window scale search was tested and improved GGUF
  creation speed, but it regressed the Qwen3-4B short WikiText-2 PPL on the
  compact FAST path, so it was rejected for coherence.
- The exhaustive scale search now visits the UE4M3 candidate nearest the
  block's `max_abs / 10` first using a monotonic binary search, expands
  outward, and exits a candidate scale once its partial error cannot beat the
  current best scale. This remains exact because every finite scale is still
  evaluated; on the Qwen3-4B FAST_COHERENT artifact it produced a
  byte-identical GGUF while cutting FAST quantization cost sharply.
- Per-value Codebook10 quantization uses exact nearest-neighbor thresholds
  instead of a 16-entry scan. The hot quantizer path uses one reciprocal scale
  per candidate/block and multiplies each value by that reciprocal instead of
  dividing per value. On the Qwen3-4B FAST_COHERENT check this kept PPL tied
  with the accepted artifact while cutting GGUF creation time further.

Hardware note:
- This is a special AMD-targeted ggml/llama.cpp quantization and backend
  path. It includes custom Vulkan and ROCm/HIP handling for the new GGUF
  types, but it is not yet a native rocWMMA FP4 tensor-core implementation.
  Current speed gains come from the compact block layout and backend decode
  paths; deeper rocWMMA/cooperative-matrix work is future optimization work.
- NVIDIA CUDA is disabled in the Strix-FP4 build (`-DGGML_CUDA=OFF`). Some
  upstream llama.cpp HIP backend sources still live under
  `ggml/src/ggml-cuda` and are compiled by HIP for AMD, but the ROCmFP4-owned
  helper code and user-facing build/run path are ROCm/HIP/Vulkan targeted.
  This tree also accepts `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` as the AMD-named
  alias for the upstream unified-memory switch.
- The bundled rocWMMA 7.1.0 headers expose gfx12 WMMA paths for FP8/BF8 and
  integer 8-bit inputs, but no native FP4 input type or FP4 WMMA/MFMA builtin
  is visible locally. A true matrix-core ROCmFP4 path therefore needs a
  measured unpack/convert strategy first, such as ROCmFP4 Codebook10 to int8
  WMMA or FP8 WMMA tiles, before claiming native FP4 tensor-core execution.
- rocWMMA FlashAttention is intentionally opt-in via
  `ENABLE_ROCWMMA_FATTN=1 /home/caf/strix-fp4/scripts/build-strix-fp4.sh`.
  It currently compiles with the local rocWMMA headers in
  `/home/caf/strix-fp4/third_party/rocWMMA`, but the Strix Halo benchmark
  regressed prefill, so the default build keeps it disabled.

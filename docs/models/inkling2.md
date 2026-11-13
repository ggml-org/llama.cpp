# Inkling 2

`inkling2` supports `Inkling2ForCausalLM`, still-image and decoded-video-frame input for `Inkling2ForConditionalGeneration` with a separate mmproj GGUF, and an optional DSpark block drafter with a separate draft GGUF. The implementation follows Transformers revision `109a7d19bf667168837f32929e00b791e195d14d`; the original checkpoint name mapping was checked against `thinkingmachines/Inkling-2-Small` revision `e242a97e271095c11bc33ddb3850d37e22917343`.

## Conversion

Convert a floating-point or packed NVFP4/MXFP8 HF checkpoint with the usual converter:

```sh
python3 convert_hf_to_gguf.py /path/to/hf-checkpoint --outtype f32 --outfile /path/to/model-store/inkling2.gguf
```

Both original `model.llm.*` names and Transformers-saved text names are accepted. Original interleaved `w13_weight` rows are reordered into contiguous gate and up halves. The router correction bias affects expert selection only. The raw config is retained because newer Transformers versions can inject quantization defaults or derive a different layer schedule.

### Packed experts

Keep the original `hf_quant_config.json` beside the checkpoint. The converter selects the declared scheme for each expert projection, so NVFP4 layers 0-39 and MXFP8 layers 40-47 can coexist. Floating-point checkpoints also work. Missing companions, unsupported schemes, invalid dtypes and inconsistent shapes are errors. Available projections can be converted independently, including when their companions occupy another input shard; an index file must still list files that are present. A partial GGUF does not by itself constitute a runnable model.

For logical weight shape `[E, R, C]`, the formulas are:

```text
NVFP4: W[e,r,c] = E2M1(nibble(P[e,r,c//2], c%2)) * E4M3FN(S[e,r,c//16]) * (A[e] / D)
MXFP8: W[e,r,c] = E4M3FN(P[e,r,c]) * 2**(S[e,r,c//32] - 127)
```

NVFP4 uses `w13_weight`/`w2_weight` U8 packed bytes, E4M3FN `_scale` bytes, F32 `_amax[E]`, and one F32 `_outer_scale_denominator[1]` shared across experts for each projection. The low nibble holds column `2j`, the high nibble column `2j+1`. The E2M1 codebook is `[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]`. The block product and `A/D` are evaluated in F32 before the normal output quantizer runs.

MXFP8 uses E4M3FN weight bytes and U8 UE8M0 scale exponents, without outer-scale companions. UE8M0 byte 0 is `2**-127`; byte 255 is rejected as NaN. E4M3FN follows `ggml_f8_e4m3_to_fp32`: subnormal magnitude `M * 2**-9`, normal magnitude `(1 + M/8) * 2**(E-7)`, with the sign bit applied to both, including zero. Bytes `0x7F` and `0xFF` are NaN. Invalid values report the tensor and expert; non-finite results and invalid outer scales are also rejected.

The pinned checkpoint metadata declares MXFP8 `weight_block_shape: [32, 32]`, but the stored layer-40 `w13_weight_scale` is U8 `[384, 4608, 144]` for weights `[384, 4608, 4608]`. The stored `w2` scales are `[384, 4608, 72]` for weights `[384, 4608, 2304]`. Both imply `[1, 32]`. The converter derives the grid from the actual weight and scale shapes, checks exact divisibility and the supported per-row group size, and warns about the config/shape conflict. It does not broadcast the declared `[32, 32]` grid. NVFP4 similarly requires a stored `[1, 16]` grid.

Original `w13` rows alternate gate and up. The converter selects even packed rows and then odd packed rows within each expert, applying the identical row slices to scales. It does not reorder column scale groups. `w2` keeps its row order. Output tensors remain 3D and use the usual `--outtype` machinery, including F32, F16 and BF16; no new GGUF type is introduced.

The decoding rules were checked against [vLLM PR 67325's pinned test helpers](https://github.com/WoosukKwon/vllm/blob/d664cd1e96052857aea2aa844eeba571c24758b4/tests/models/inkling2/test_moe.py) and its expert loading code, plus the original checkpoint headers. Runtime row shuffles and scale swizzles in vLLM occur after loading and are not part of the stored format.

### Conversion memory

Packed experts use `LazyChunkedTensor` and fresh mapped source views for each chunk, decoding at most 128 rows of one expert. Quantization and writing also happen per chunk. For the largest real `w13` (`384 x 4608 x 4608`), the full F32 tensor would occupy 32,614,907,904 bytes (30.375 GiB); the decoded chunk occupies 2,359,296 bytes (2.25 MiB). No full expanded expert layer is allocated.

A CPU probe using sparse synthetic files with those full source shapes measured less than 5.8 MiB of Python/NumPy chunk allocations through BF16 writing and about 5.2 MiB of metadata for the 13,824 chunk callbacks. Allow approximately 16 MiB for one such projection's dequantization path, including active mapped source pages. This is an incremental working-memory estimate, not a total-process RSS measurement or a real-checkpoint conversion measurement. Chunk metadata for all 48 layers is roughly 370 MiB; Python/PyTorch, tokenizer, other tensors and OS file cache are additional. In particular, ordinary embedding/output conversion can still require several GiB. The output itself needs 30.375 GiB on disk per largest F32 `w13` (15.1875 GiB for F16/BF16). The packed-expert path does not require the entire checkpoint or an expanded layer to fit in RAM.

Weight-only dequantization does not reproduce the reference runtime's activation quantization. Real-checkpoint logits will therefore differ from that runtime for quantization reasons even with an identical text architecture. Synthetic parity below compares both runtimes with the same effective floating-point weights and does not claim parity with the original quantized execution.

The tokenizer reuses GPT-4o BPE preprocessing. EOS/EOT is the configured EOS token (200006 in the original checkpoint); BOS and padding are absent there. The GGUF vocabulary contains only `unpadded_vocab_size` tokens, while the embedding and output tensors retain their padded rows. The graph slices output logits, so the public logit width equals the exposed vocabulary size.

## Execution

The text backbone has no RoPE. Both attention types use input-dependent relative logits; sliding layers use `sliding_window_size` as the relative extent, while full layers use `rel_extent`. Only full layers apply logarithmic scaling to the clipped Q and relative states.

Attention uses the existing unfused path because flash attention does not accept the dense KQ bias and per-query sinks through this tree's attention helper. The sink is included in softmax and has zero value. Flash attention is disabled for this architecture, including when requested explicitly. Quantized V cache requires flash attention in the current runtime and is not supported here.

The existing hybrid SWA memory holds KV and recurrent state in every layer. Each recurrent row packs the K, V, post-attention and post-MLP convolution histories. The standard recurrent copy, save/load and snapshot mechanisms apply to those histories. Block-attention residual states are per-position graph tensors; they do not use a persistent cache. The expert activation reuses SITU with the gate and up softcaps.

## Vision

Convert the same checkpoint separately for text and vision. The normal invocation above writes only text; `--mmproj` writes only the vision encoder and merger:

```sh
python3 convert_hf_to_gguf.py /path/to/hf-checkpoint --mmproj --outtype f32 --outfile /path/to/model-store/mmproj-inkling2.gguf
```

Original `model.visual.*` names and Transformers `model.vision_tower.*` names are accepted. All 422 synthetic vision tensor names match the saved real checkpoint index and map identically to the HF-saved names. The mmproj carries patch size, encoder width, heads, layers, feed-forward width, LayerNorm epsilon, spatial merge size, position-grid side, RoPE base, projector hidden/output widths, projector bias policy and image mean/std. Text GGUF contents are unaffected by the separate vision conversion. The stored patch projection keeps HWC flattening order and has no temporal factor.

The encoder uses the generic ViT graph, with pre-LayerNorm, biased Q/K/V/output projections, noncausal attention, learned residual layer scales and exact GELU. It prepends the sum of the class token and its position vector. Learned grid positions use bilinear interpolation with half-pixel centers and no antialiasing. Axial RoPE uses adjacent channel pairs, width in the first half and height in the second; patch coordinates start at one and CLS is at zero. After removing CLS, consecutive native patch features are concatenated directly, then projected through GELU to the text width. The real configuration has 23 layers, width 1024, 16 heads, and a 4096-hidden projector to width 4608.

Preprocessing keeps the original image resolution and pads on the right and bottom to a multiple of `native_patch_size * adapter_merge_size` (40 pixels in the public checkpoint). Padding is float -1 on the raw 0-255 scale, before rescaling and normalization. The reference fuses these as `(raw - 127.5) / 127.5`, so padding becomes approximately -1.00784314. There is no resize, overview crop or independent attention per cell: the complete padded image goes through one encoder sequence. The grid is `[1, ceil(height / 40), ceil(width / 40)]`.

Merged cells follow row-major order. Within each cell, native patches are top-left, top-right, bottom-left, bottom-right; each patch is flattened as row, column, RGB channel. The merger retains that patch-major feature order. RGBA image files are composited before padding onto an 8-pixel checkerboard, with 204 at even `(row / 8 + column / 8)` parity and 153 at odd parity. Each channel uses integer rounding `(channel * alpha + background * (255 - alpha) + 127) / 255`. The public RGB bitmap API expects callers to supply already-composited RGB; the mtmd image-file helpers handle RGBA compositing.

The regular CLI accepts `--mmproj` and `--image`:

```sh
llama-cli -m /path/to/model-store/inkling2.gguf --mmproj /path/to/model-store/mmproj-inkling2.gguf --image image.png --no-mmproj-offload -ngl 0 -fa off -p "Describe this image." -st
```

`llama-mtmd-cli` accepts the same model/image flags. `llama-server -m ... --mmproj ... --no-mmproj-offload -ngl 0 -fa off` accepts image content through `/v1/chat/completions`, including base64 PNG data URLs. The synthetic CLI/server checks used `--chat-template chatml` because their byte-only test vocabulary has no checkpoint chat template. The real checkpoint's chat template has not been exercised with vision.

### Vision verification

The gates were fixed before comparison: bit-exact F32 preprocessing, and maximum absolute error <= 1e-5 plus relative L2 <= 1e-5 for every vision checkpoint and end-to-end feature comparison. All stages passed in order on CPU with flash attention disabled. No real weights were downloaded.

The deterministic synthetic checkpoint uses the original `model.visual.*` names, all 23 layers, encoder width 32, four heads, native patch 20, merge 2, learned position-grid side 32, projector hidden width 4096 and output width 4608. The text fixture has width 4608, two layers and a byte-only test vocabulary of 256 entries; the fixture generator supplies that tokenizer metadata directly to avoid a large 200k-vocabulary random embedding table. The vision GGUF uses the standard `--mmproj` command. Synthetic tensors and fixtures are not real-model validation.

Four PNG cases cover RGB 61x43, RGBA 83x45, aligned RGB 40x40 and wide RGBA 161x21, with grid shapes `[1,2,2]`, `[1,2,3]`, `[1,1,1]` and `[1,1,5]`. All 76,800 preprocessed channel values match element-wise, including normalized padding and alpha values 0, 1, 127, 128, 254 and 255. The C++ native-patch input buffers also match the HF patch order bit-for-bit.

| Checkpoint | Worst max absolute error | Worst relative L2 |
|---|---:|---:|
| Linear patch projection | 1.31130219e-6 | 3.19121364e-7 |
| Patches with positions and CLS | 1.31130219e-6 | 3.18810928e-7 |
| First encoder layer | 1.19209290e-6 | 3.21497431e-7 |
| Final encoder, all 23 layers | 1.37090683e-6 | 3.32171539e-7 |
| Merger into width 4608 | 1.43051147e-6 | 6.91625568e-7 |
| Image file through mtmd into width 4608 | 1.43051147e-6 | 6.91625568e-7 |

Each metric is the maximum across the four cases, so the two columns can refer to different images. Each case also ran with its image prepended through the actual multimodal prompt path, producing 256 finite logits. The regular CLI and multimodal CLI generated two tokens; the server image request returned HTTP 200 with two completion tokens. F16 and BF16 mmproj conversion/loading also produced finite CPU image-conditioned logits; numerical parity is claimed only for F32.

The original text regressions still pass: both float fixtures, the packed fixture, all 74 original-name roundtrip tensors, text-only loading without mmproj, and `test-llama-archs -a inkling2 -b CPU`. The converter suite has 24 passing tests; gguf-py has 9; mtmd has 15 tests with 196 assertions. Changed Python files pass the scoped type check. The whole-repository type check remains incomplete because optional dependencies for unrelated tools are absent.

Real weights, production encoder width/head dimensions, the real 640x640 input grid (1024 patches plus CLS), GPU/NPU execution, real-model quality and the real chat template with images remain unverified. The decoded-frame checks below extend this synthetic validation.

### Decoded video frames

Inkling2 accepts consecutive decoded RGB frames through mtmd's existing mergeable bitmap and lazy bitmap paths. Set both `mtmd_bitmap_set_mergeable(frame, true)` and `mtmd_bitmap_set_is_video(frame, true)` on each frame. Adjacent frames of the same dimensions form groups of two, corresponding to `temporal_patch_size = 2`. Text or a dimension change ends a group. Each pair is averaged on the raw RGB scale in F32, preserving half-integer pixel values, before the existing padding, normalization and still-image encoder. Each group produces one image-feature sequence. Ordinary still images keep their original preprocessing arithmetic.

An unpaired final frame is retained unchanged, equivalent to repeating it before the mean. This avoids losing the end of a clip and preserves the single-frame behavior. Distinct still-image inputs are not implicitly video; callers must mark video frames explicitly. RGBA compositing happens separately on each frame before the RGB mean.

There is no reference video preprocessing to match: pinned Transformers `image_processing_inkling2.py` and `processing_inkling2.py` process still images only, and the patch projection has input width `num_channels * patch_size**2`, without a temporal factor. The pinned vLLM `vision.py` explains that identical still frames have the frame itself as their mean. This implementation therefore supplies the config-implied temporal-pair-mean semantics, validated by duplicate-frame equivalence; it is not a reference-matched video pipeline. Real video codec decoding, frame-rate and frame-sampling policy, timestamps and per-model video prompt formatting are outside this support contract. Feed already decoded frames with the intended grouping and prompt boundaries.

The four existing image fixtures pass duplicate-frame bit identity at all 76,800 preprocessed channel values and all 73,728 merged feature values, including the public mtmd path. Distinct-frame pair means match an independent float64 raw mean followed by F32 still-image preprocessing bit-for-bit, and the same C++ still-image encoder produces identical merged features. The pinned Torch still-image encoder on those means has worst merger error 1.56462193e-6 absolute and 6.89512069e-7 relative L2. Four nine-frame sequences each produce five groups and 256 finite downstream logits. The original still-image preprocessing and merger numbers above are unchanged.

## MTP / DSpark block drafter

The predictor follows [vLLM PR #67325's unmerged implementation](https://github.com/WoosukKwon/vllm/blob/d664cd1e96052857aea2aa844eeba571c24758b4/vllm/models/inkling2/dspark.py), pinned to `d664cd1e96052857aea2aa844eeba571c24758b4`. There is no Transformers predictor implementation. This is an eight-position parallel block decoder followed by a recurrent Markov head, not five chained NextN predictors.

Convert `mtp.safetensors` from the same checkpoint directory separately:

```sh
python3 convert_hf_to_gguf.py /path/to/hf-checkpoint --dspark --outtype f32 --outfile /path/to/model-store/inkling2-dspark.gguf
llama-cli -m /path/to/model-store/inkling2.gguf --spec-type draft-dspark --spec-draft-model /path/to/model-store/inkling2-dspark.gguf --spec-draft-n-max 7 --spec-draft-p-min 0 --spec-draft-sampling greedy -ngl 0 --spec-draft-ngl 0 --temp 0 -p "Hello" -st
```

The draft GGUF uses the existing `dflash` architecture and common `draft-dspark` driver. It contains all 84 MTP tensors and no duplicate target embedding or output weights; the draft context shares those with the target. The bounded checkpoint header and `hf_quant_config.json` agree on the exact 84 BF16 source names. `markov_w1` has no bias. Normal text conversion/loading still works when `mtp.safetensors` is absent; requesting `--dspark` without that file is an error. F32/F16/BF16 output conversion is supported, with `fuse_weight` retained in F32 for softmax.

The real predictor has five dense layers, width 4608, 48 Q heads, eight KV heads, head dimension 128, SwiGLU width 18432, RMSNorm epsilon 1e-6 and ordinary NEOX RoPE with theta 1e6. Layers 0, 1, 3 and 4 attend to the last 1024 committed context positions plus the entire block; layer 2 attends to all context plus the block. All block positions see the same context window. These values use existing GGUF layer/head/FFN/RoPE/SWA keys; `dflash.dspark.markov_rank` selects the Inkling 2 variant and records rank 256.

### Target features and recurrence

The feature order is exactly `[5, 11, 17, 23, 29, 35, 41, 48]`. For the first seven IDs, the feature is the running AttnRes partial block at entry to the zero-based target layer, before that layer's attention mix and norm. These are the partial blocks after layers 4, 10, 16, 22, 28, 34 and 40. With six layers per residual block, each contains the first five layers of its block. Feature 48 is the final output mix followed by the final norm after layer 47. It is not the raw last partial block. The final feature is exported before selecting output rows, so context injection retains every processed token.

The existing layer-input extraction API supplies these taps. Every draft layer softmaxes its own `fuse_weight` row, sums the raw tap values, applies its own `dflare_norm`, and projects context K/V with `k_proj_ctx` and `v_proj_ctx`. Context K uses `k_norm_ctx` and RoPE. The sixth mixture has a separate norm and supplies `f_rnn` from the last committed target token.

Position zero is the normalized shared target embedding of the anchor; the other seven positions use the same `mask_embed`. After the five layers and final norm, each of the seven RNN steps consumes `[state, markov_w1(previous_token), hidden_i, f_rnn]`. It splits the projection into gate, candidate and output latent, updates the state with `sigmoid(gate)*state + (1-sigmoid(gate))*tanh(candidate)`, and projects `tanh(latent)` through `hidden_out` and the shared target head. The final logits are `cap*tanh((base+bias)/cap)`, sliced to the exposed vocabulary. State starts at zero for each block.

The graph exports all eight state rows through the existing nextn output: the anchor row is zero, subsequent rows contain the rank-sized state followed by zero padding to model width. During context injection, that same output carries the separate `f_rnn` mixture. The common driver's per-sequence ring retains nine committed-feature rows so rejection can select the correct boundary, and the driver's existing state-save API serializes the ring. Draft KV removal and restoration use the existing cache APIs.

The graph always evaluates the trained block of eight positions, even when the requested proposal limit is smaller. It proposes at most seven tokens, does not sample from the anchor, and has no confidence head. The first maximal logit wins ties in both the graph and host sampler. Backend sampling is bypassed for this variant because backend argmax tie rules differ. The proposed token is the actual sampler return and is the token used for the next RNN step.

### Numerical and sampling differences

`--spec-draft-sampling greedy` retains the in-graph recurrence and first-maximum tie rule. `--spec-draft-sampling probabilistic` evaluates the full noncausal backbone once, retains its hidden columns and base logits, then evaluates a reusable head-only graph for each proposal. The host samples that step's logits and feeds the actual proposed token into the following recurrence. Rank state starts at zero for each block; the committed-context ring and KV rollback remain separate and unchanged.

Probabilistic proposals use softmax over the full exposed vocabulary at the target temperature, or temperature 1 when the target is greedy. Target top-k, top-p, grammar and penalty filters do not restrict this proposal distribution. Each sampled step records its actual distribution in `result_q`, allowing the target's existing rejection sampler to preserve its own filtered distribution. A greedy target uses token matching and remains output-preserving. The proposal RNG follows the existing DFlash2 per-sequence seed handling and is reset at request boundaries.

The initial implementation retains hidden columns, base logits and rank state in host F32 buffers, copying the selected inputs into the head graph and its updated rank state back to the host. It performs one backbone plus D head evaluations for D proposals, without replaying the backbone. Head steps do not update KV. It does not claim device-resident state or improved throughput; device-buffer retention is a possible later optimization.

The reference folds `hidden_out` into the shared LM head in F64 and stores the result in BF16. This graph evaluates the two projections separately. The CPU parity reference performs the fold in F64 then stores F32, so it also checks the graph's algebraic equivalence across those evaluation orders. F32 validation does not reproduce the source's explicit BF16 rounding of fusion weights, residuals, RoPE, logits, gates and recurrent state. No BF16-runtime or activation-quantized vLLM parity is claimed.

### CPU synthetic verification

The standalone Torch transcription replaces TP collectives with a single rank, paged/FlashInfer attention with dense attention using the same visible keys, and fused GPU kernels with F32 operations. Source-by-source correspondence and every simplification are retained in the local `dspark-reference-map.md` beside the harness. The thresholds were fixed before comparison: maximum absolute error <= 1e-4, relative L2 <= 1e-5 per logit/state row, and identical greedy draft IDs.

A width-32 text fixture with eight layers and eight taps feeds a five-layer, rank-eight drafter. It uses a 16-token sliding window, 260 stored vocabulary rows and 256 exposed tokens. Context lengths 3, 8, 21 and 33 pass all feature, RNN-context, seven-position logit and chain-state checks. Worst logit error is 6.95884228e-6 absolute and 2.49064251e-6 relative L2; worst state error is 2.76044011e-6 and 1.997992e-6. All 28 draft IDs agree. A second width-32 fixture uses 48 target layers, the literal production tap IDs, six-layer AttnRes blocks and the 1024-token window. At context length 1025, all seven draft IDs agree, logit error is 2.83122063e-6 and state error is 1.49384141e-6.

The first-draft and target-next-token logit RMS values are 0.596-0.648 and 0.596-0.664 respectively. Their cosine similarity is 0.00145-0.30345 and their argmax IDs differ, as expected for independent random weights. This checks finite scale, not trained prediction quality. Direct tap parity is the stronger contract check: exchanging two taps changes logits by 0.194-0.740, and replacing every tap with the final norm changes them by 0.827-1.888.

`llama-cli` generates 48 identical token IDs with and without the drafter for prompts `abc` (3 tokens), `abcdefgh` (8), `abcdefghijklmnopqrstu` (21), `abcdefghijklmnopqrstuvwxyz0123456` (33), and 1025 repetitions of `a`. The 21-token prompt also matches with target temperature 1.7 and fixed seed 1234. Each random-weight run proposes 301 tokens and accepts zero; rejected draft KV/state therefore does not alter target output. The 48-layer/1024-window fixture also matches all 48 outputs for the 1025-token prompt. Repeating the 33-token case produces the same tokens.

An all-zero shared-head control tests exact ties and accepted tokens: with a temperature-only target sampler, all 41 proposed tokens are accepted and all 48 output IDs still match. A proposal limit of three also preserves output. An explicit save/overwrite/restore test restores the 1188-byte context ring exactly, reproduces seven draft IDs, and keeps chain-state error below 3.58e-7. F16 and BF16 draft GGUFs run with identical target output; a synthetic head-dimension-128 draft with Q8_0 K/V cache and CPU flash attention does too. These are execution/output-preservation checks, not reduced-precision logit parity.

Text float/packed prefill, chunk and decode parity, the 74-tensor original-name roundtrip, the vision preprocessing/encoder/end-to-end/prompt fixtures, all 15 mtmd tests (196 assertions), `test-llama-archs -a inkling2 -b CPU`, all 24 conversion tests and all nine gguf-py tests pass. Text-only fixtures without MTP still load and run. No GPU/NPU or performance measurement was used.

Real checkpoint conversion/inference, production hidden/head dimensions together, real-model acceptance/quality, the source's BF16 execution, TP/distributed execution, multiple active sequences, combined image-plus-draft requests, GPU/NPU execution and contexts beyond the synthetic 1025-token boundary remain unverified. The real maximum context is 1,048,576; this work does not qualify that limit. The predictor reference remains an unmerged PR at the stated pin.

### Probabilistic CPU verification

The existing 35/35 greedy draft-ID matches and 48/48 output identities reproduce. Driving the eager reference with the actual host-sampled IDs passes all seven logit and state rows at context lengths 3, 8, 21, 33 and 1025. Worst sampled logit error is 6.89923763e-6 absolute and 2.49604781e-6 relative L2; worst rank-state error is 2.74181366e-6 and 2.03142392e-6. Of the 35 sampled proposals, 34 differ from the per-step greedy token. Recorded `result_q` differs from an independent softmax by at most 1.39698386e-9.

With a greedy target, probabilistic drafting preserves all 48 output tokens on every prompt in the five-length set, with CPU flash attention both off and on. The 48-layer, literal-tap 1025-token fixture also preserves 48/48 with FA on. Sampled-target runs at temperature 1.7 repeat exactly with a fixed seed.

Distribution tests use the actual DSpark driver and target rejection sampler at one fixed continuation position, a 256-token fixture, and target temperature 1.1 with top-k 32. An independent replication has 65,536 draws per arm per condition. Speculation-on versus speculation-off chi-square statistics (31 degrees of freedom) are 21.3987, 32.0891 and 25.5529 for proposal temperatures 1.1, 0.1 and 1000, with p-values 0.9010, 0.4124 and 0.7425. The deliberately overconfident proposal changes measured acceptance from 14.9399% to 7.2739%, versus theoretical overlaps 15.1912% and 7.2977%. Near-uniform proposals accept 12.4771%, versus 12.5026% expected. A deliberately incorrect uniform `result_q` for the overconfident sampler is rejected strongly by the distribution test.

The preliminary 8,192-draw ordinary-proposal comparison gave p=0.00509 and triggered that larger independent replication; it is retained in the report. A secondary exact-target goodness-of-fit check for the replication's overconfident condition gives p=0.00905, while its primary two-sample comparison gives p=0.4124. These finite Monte Carlo checks support output preservation but do not prove equality of distributions, and the single-position test does not qualify a real checkpoint's full generated-sequence distribution.

## Scope and remaining limits

Decoded-frame temporal means and probabilistic DSpark conditioning are supported. Codec decoding, sampling cadence and model-specific video prompts are not qualified. The probabilistic head currently uses host F32 buffers and is qualified on CPU synthetic fixtures only; real-checkpoint behavior, source BF16 rounding, distributed execution and device-resident state remain unverified.

Text conversion emits one warning that directs users to `--mmproj` for vision and `--dspark` for `mtp.safetensors`. Each conversion writes only its selected component. It never silently interprets packed expert bytes as floating-point weights.

## Validation

Validation is CPU-only and uses deterministic F32 synthetic weights, not the real 390 GB checkpoint. The installed reference source was byte-identical to the pinned `modeling_inkling2.py`.

Two three-layer fixtures use hidden size 32, Q/KV heads 4/1, head dimension 8, expert width 24, four experts and top-2 routing. Both use sliding/full/sliding layers, window 4, relative dimension 4, full relative extent 16, convolution kernel 4, vocabulary 200058 and exposed vocabulary 200056. Residual block sizes are 2 and 1. Nonzero residual queries, convolution weights and router correction biases avoid degenerate tests. Clip thresholds 0.25/0.1, log-scaling floor 4 and gate/up caps 0.2/0.3 exercise the bounded operations.

Every logit position is compared against the reference for a fixed 15-token sequence using a single prefill, chunks of 5/4/6 tokens, and a five-token prefill followed by ten single-token decodes. Acceptance requires maximum absolute error <= 1e-4, per-position NMSE <= 1e-9, and identical greedy argmax at every position. Decode and chunked results are also compared with the single-prefill C++ results.

The architecture test covers CPU evaluation, model save/load and mixed token/embedding inputs. An independent conversion check reconstructs the original checkpoint naming and row interleave from the tiny model and requires identical GGUF tensors. The real tokenizer is checked on ordinary text, contractions, multilingual text, combining marks, emoji and control tokens. Real weights and GPU execution have not been validated.


A third fixture has hidden/expert width 64 and three layers: NVFP4, MXFP8 and floating point. Independent encoders produce the original packed layouts and companions. The pinned vLLM test helpers decode the stored bytes into a separate float checkpoint for Transformers. All four converted F32 expert tensors match that independent decode bit-for-bit, including signed zeros. Maximum encode/decode absolute errors are 0.06433168053627014 (NVFP4) and 0.015622586011886597 (MXFP8); these are format quantization errors, not converter errors. NVFP4 clips 1,653 of 49,152 values to its endpoint after rounding block scales; MXFP8 clips none. Maximum relative errors are 1.0 and 0.08542004056702747, respectively; near-zero values dominate relative error.

Packed-fixture CPU logit parity has maximum absolute error 2.86102294921875e-6 for full prefill, 3.0994415283203125e-6 for chunks, and 2.6226043701171875e-6 for incremental decode. Maximum per-position NMSE is 3.7101697538062787e-13 and argmax agreement is 100%. Both original float fixtures and all 74 tensors in the original-name roundtrip still pass. Converter tests additionally cover F32/F16/BF16 and automatic outtype, lazy/eager reading, multiple row chunks, separate source shards, invalid metadata, all E4M3 codes, signed zeros, UE8M0 zero and explicit NaN rejection.

The real 390 GB checkpoint has not been downloaded or converted. Only the saved layer-0/layer-40 headers and pinned source were inspected. Other real shards' integrity, every layer's actual values, complete-model memory/throughput, real-model inference quality and activation-quantized runtime parity remain unverified.

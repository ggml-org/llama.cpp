# Inkling 2 text backbone

`inkling2` supports `Inkling2ForCausalLM` and the text backbone of `Inkling2ForConditionalGeneration`. The implementation follows Transformers revision `109a7d19bf667168837f32929e00b791e195d14d`; the original checkpoint name mapping was checked against `thinkingmachines/Inkling-2-Small` revision `e242a97e271095c11bc33ddb3850d37e22917343`.

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

There is no RoPE. Both attention types use input-dependent relative logits; sliding layers use `sliding_window_size` as the relative extent, while full layers use `rel_extent`. Only full layers apply logarithmic scaling to the clipped Q and relative states.

Attention uses the existing unfused path because flash attention does not accept the dense KQ bias and per-query sinks through this tree's attention helper. The sink is included in softmax and has zero value. Flash attention is disabled for this architecture, including when requested explicitly. Quantized V cache requires flash attention in the current runtime and is not supported here.

The existing hybrid SWA memory holds KV and recurrent state in every layer. Each recurrent row packs the K, V, post-attention and post-MLP convolution histories. The standard recurrent copy, save/load and snapshot mechanisms apply to those histories. Block-attention residual states are per-position graph tensors; they do not use a persistent cache. The expert activation reuses SITU with the gate and up softcaps.

## Deliberately unsupported

- Vision: `model.visual.*` and `vision_tower.*` tensors are skipped. An image processor, vision graph, projector and multimodal integration remain separate work.
- MTP: MTP and Markov-head tensors are skipped. The main text model does not implement the speculative predictor; the pinned Transformers reference does not implement it either.

The converter emits one warning for skipping vision and MTP. It never silently interprets packed expert bytes as floating-point weights.

## Validation

Validation is CPU-only and uses deterministic F32 synthetic weights, not the real 390 GB checkpoint. The installed reference source was byte-identical to the pinned `modeling_inkling2.py`.

Two three-layer fixtures use hidden size 32, Q/KV heads 4/1, head dimension 8, expert width 24, four experts and top-2 routing. Both use sliding/full/sliding layers, window 4, relative dimension 4, full relative extent 16, convolution kernel 4, vocabulary 200058 and exposed vocabulary 200056. Residual block sizes are 2 and 1. Nonzero residual queries, convolution weights and router correction biases avoid degenerate tests. Clip thresholds 0.25/0.1, log-scaling floor 4 and gate/up caps 0.2/0.3 exercise the bounded operations.

Every logit position is compared against the reference for a fixed 15-token sequence using a single prefill, chunks of 5/4/6 tokens, and a five-token prefill followed by ten single-token decodes. Acceptance requires maximum absolute error <= 1e-4, per-position NMSE <= 1e-9, and identical greedy argmax at every position. Decode and chunked results are also compared with the single-prefill C++ results.

The architecture test covers CPU evaluation, model save/load and mixed token/embedding inputs. An independent conversion check reconstructs the original checkpoint naming and row interleave from the tiny model and requires identical GGUF tensors. The real tokenizer is checked on ordinary text, contractions, multilingual text, combining marks, emoji and control tokens. Real weights and GPU execution have not been validated.


A third fixture has hidden/expert width 64 and three layers: NVFP4, MXFP8 and floating point. Independent encoders produce the original packed layouts and companions. The pinned vLLM test helpers decode the stored bytes into a separate float checkpoint for Transformers. All four converted F32 expert tensors match that independent decode bit-for-bit, including signed zeros. Maximum encode/decode absolute errors are 0.06433168053627014 (NVFP4) and 0.015622586011886597 (MXFP8); these are format quantization errors, not converter errors. NVFP4 clips 1,653 of 49,152 values to its endpoint after rounding block scales; MXFP8 clips none. Maximum relative errors are 1.0 and 0.08542004056702747, respectively; near-zero values dominate relative error.

Packed-fixture CPU logit parity has maximum absolute error 2.86102294921875e-6 for full prefill, 3.0994415283203125e-6 for chunks, and 2.6226043701171875e-6 for incremental decode. Maximum per-position NMSE is 3.7101697538062787e-13 and argmax agreement is 100%. Both original float fixtures and all 74 tensors in the original-name roundtrip still pass. Converter tests additionally cover F32/F16/BF16 and automatic outtype, lazy/eager reading, multiple row chunks, separate source shards, invalid metadata, all E4M3 codes, signed zeros, UE8M0 zero and explicit NaN rejection.

The real 390 GB checkpoint has not been downloaded or converted. Only the saved layer-0/layer-40 headers and pinned source were inspected. Other real shards' integrity, every layer's actual values, complete-model memory/throughput, real-model inference quality and activation-quantized runtime parity remain unverified.

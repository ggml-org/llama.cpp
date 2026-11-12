# Inkling 2 text backbone

`inkling2` supports `Inkling2ForCausalLM` and the text backbone of `Inkling2ForConditionalGeneration`. The implementation follows Transformers revision `109a7d19bf667168837f32929e00b791e195d14d`; the original checkpoint name mapping was checked against `thinkingmachines/Inkling-2-Small` revision `e242a97e271095c11bc33ddb3850d37e22917343`.

## Conversion

Convert a floating-point HF checkpoint with the usual converter:

```sh
python3 convert_hf_to_gguf.py /path/to/hf-checkpoint --outtype f32 --outfile /path/to/model-store/inkling2.gguf
```

Both original `model.llm.*` names and Transformers-saved text names are accepted. Original interleaved `w13_weight` rows are reordered into contiguous gate and up halves. The router correction bias affects expert selection only. The raw config is retained because newer Transformers versions can inject quantization defaults or derive a different layer schedule.

The tokenizer reuses GPT-4o BPE preprocessing. EOS/EOT is the configured EOS token (200006 in the original checkpoint); BOS and padding are absent there. The GGUF vocabulary contains only `unpadded_vocab_size` tokens, while the embedding and output tensors retain their padded rows. The graph slices output logits, so the public logit width equals the exposed vocabulary size.

## Execution

There is no RoPE. Both attention types use input-dependent relative logits; sliding layers use `sliding_window_size` as the relative extent, while full layers use `rel_extent`. Only full layers apply logarithmic scaling to the clipped Q and relative states.

Attention uses the existing unfused path because flash attention does not accept the dense KQ bias and per-query sinks through this tree's attention helper. The sink is included in softmax and has zero value. Flash attention is disabled for this architecture, including when requested explicitly. Quantized V cache requires flash attention in the current runtime and is not supported here.

The existing hybrid SWA memory holds KV and recurrent state in every layer. Each recurrent row packs the K, V, post-attention and post-MLP convolution histories. The standard recurrent copy, save/load and snapshot mechanisms apply to those histories. Block-attention residual states are per-position graph tensors; they do not use a persistent cache. The expert activation reuses SITU with the gate and up softcaps.

## Deliberately unsupported

- Vision: `model.visual.*` and `vision_tower.*` tensors are skipped. An image processor, vision graph, projector and multimodal integration remain separate work.
- MTP: MTP and Markov-head tensors are skipped. The main text model does not implement the speculative predictor; the pinned Transformers reference does not implement it either.
- Packed checkpoint experts: original NVFP4 `amax_denominator` and MXFP8 blocks are rejected with an explicit error. The supplied MXFP8 metadata specifies `[32, 32]` blocks, while the Transformers-injected default specifies `[1, 32]`. The NVFP4 outer-scale and packing rules are also unverified. Supporting these requires an authoritative packing/scale specification and small independently verified dequantization vectors before testing real shards.

The converter emits one warning for skipping vision and MTP. It never silently interprets packed expert bytes as floating-point weights.

## Validation

Validation is CPU-only and uses deterministic F32 synthetic weights, not the real 390 GB checkpoint. The installed reference source was byte-identical to the pinned `modeling_inkling2.py`.

Two three-layer fixtures use hidden size 32, Q/KV heads 4/1, head dimension 8, expert width 24, four experts and top-2 routing. Both use sliding/full/sliding layers, window 4, relative dimension 4, full relative extent 16, convolution kernel 4, vocabulary 200058 and exposed vocabulary 200056. Residual block sizes are 2 and 1. Nonzero residual queries, convolution weights and router correction biases avoid degenerate tests. Clip thresholds 0.25/0.1, log-scaling floor 4 and gate/up caps 0.2/0.3 exercise the bounded operations.

Every logit position is compared against the reference for a fixed 15-token sequence using a single prefill, chunks of 5/4/6 tokens, and a five-token prefill followed by ten single-token decodes. Acceptance requires maximum absolute error <= 1e-4, per-position NMSE <= 1e-9, and identical greedy argmax at every position. Decode and chunked results are also compared with the single-prefill C++ results.

The architecture test covers CPU evaluation, model save/load and mixed token/embedding inputs. An independent conversion check reconstructs the original checkpoint naming and row interleave from the tiny model and requires identical GGUF tensors. The real tokenizer is checked on ordinary text, contractions, multilingual text, combining marks, emoji and control tokens. Real weights, GPU execution and quantized expert shards have not been validated.

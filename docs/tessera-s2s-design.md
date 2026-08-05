# Tessera S2S: Route A (text bridge) + instrumentation for Route B

Status: design. Talker quantization locked by architect 2026-08-05 (section
3.4); sections 8.1/8.2/8.3 still open.
Date: 2026-08-05

## 0. Driving decisions

1. Ship Route A first: Gemma 4 12B unified trunk answers in text, hands off to
   Qwen3-TTS-12Hz (Talker + Code2Wav) for speech output. Two GGUFs, text bridge
   between them.
2. Instrument Route A aggressively. The instrumentation is not telemetry for its
   own sake: it is the data factory for Route B.
3. Route B = extend Gemma 4's vocab with Qwen3-TTS codec code ids, finetune the
   trunk to speak codes directly, retire the Talker stage. Route B is gated on
   Route A's accumulated instrumentation data, not on a calendar date.
4. The Talker is quantized through the Tessera calibration and quantization
   pipelines, not plain offline quant. It doubles as the multimodal test
   target for ternary-dequant-on-arrival (section 3.4).

## 1. Why instrumenting Route A makes Route B nearly free

The Route A Talker is the teacher. Every utterance it produces is an
already-aligned triple:

```
(Gemma answer text, 16-codebook codes @ 12.5 Hz, synthesized waveform)
```

Capturing the Talker's codes at generation time (not re-encoding audio later)
produces distillation targets that are:

- zero-cost: generated as exhaust of normal app usage, no labeling pass;
- exactly on distribution: the voices, styles and languages users actually run;
- paired with the exact text Gemma produced, so Route B trains on the trunk's
  own output distribution, not a generic TTS corpus.

Route B's finetune (new code embedding rows + LoRA on the trunk + code head)
consumes exactly these pairs. The instrumented Route A therefore converts a
future training project from "source and align a speech corpus" into "wait for
usage to accumulate, then filter".

## 2. Background: vocab facts that shaped this design

Measured from local artifacts (ggml-vocab-gemma-4.gguf, ggml-vocab-qwen35.gguf,
and gemma-4-12B-source config.json on the external drive):

- Gemma 4 12B unified: model_type gemma4_unified, text vocab 262,144
  (SentencePiece), hidden_size 3840, 48 layers, tie_word_embeddings true.
  Audio is INPUT-ONLY: a 640-dim audio tower injects embeddings at the
  <|audio|> placeholder (id 258881), with boa/eoa markers. There is no speech
  output vocab; the vocab does hold 6,227 <unused*> slots.
- Qwen3-TTS-12Hz (tech report arXiv 2601.15621): 12.5 Hz frame rate, 16-layer
  RVQ, 2,048 entries per codebook (codebook 0 semantic via WavLM teacher,
  1-15 acoustic). The Talker emits from a SEPARATE 3,072-entry codec vocab with
  its own embedding/head; text vocab 151,936 is consumed, never spoken.
  Backbone predicts codebook 0 autoregressively; an MTP module predicts
  residual codebooks 1-15. Code2Wav is a lightweight causal ConvNet (24 kHz
  out, no lookahead, ~97 ms first packet, ~2.2 kbps).
- Text vocab overlap (Gemma 4 vs Qwen-family BPE): 24,514 exact string matches
  (9.6% of Gemma / 16.2% of Qwen); 63,632 after normalizing the space markers
  (24.9% / 42.0%); of those, 4 (0.0%) share a token id.

Consequences:

- Direct text-token id routing across the two models is dead (0% id alignment,
  incompatible segmentation). The UTF-8 detokenize + retokenize bridge is
  lossless, deterministic, and costs microseconds per sentence. It is not a
  bottleneck worth a training project.
- The vocab extension that matters is Gemma + the 3,072 codec ids, which is
  collision-free by construction (append past 262,143; the <unused*> slots are
  an alternative but appending avoids collisions with future upstream vocab
  changes). With tie_word_embeddings true a single tensor grows:
  3,072 x 3,840 = 11,796,480 new params (~11.8M).
- Those new rows are noise until trained, and Gemma has never seen a
  "predict speech codes from text" objective. Hence Route B is a finetune
  gated on the Route A corpus, not a config change.

## 3. Route A architecture

### 3.1 Models and graphs

- Talker GGUF: Qwen3-TTS-12Hz-1.7B-Base (backbone + MTP module).
  - Text embedding (151,936) and codec embedding (3,072) as separate tensors;
    codec head (3,072) separate from any text head.
  - MTP module as a second graph attached to the backbone, same shape of
    cross-graph hidden-state plumbing the fork already uses for DFlash
    (ctx_other borrowing). The backbone consumes aggregated codebook features
    and predicts codebook 0; the MTP head emits codebooks 1-15 per frame.
- Code2Wav GGUF (second graph or separate file): causal ConvNet, one 16-code
  frame -> 80 ms of 24 kHz PCM, streaming. ggml has ggml_conv_1d and
  ggml_conv_transpose_1d; the snake activation is synthesized as
  x + sin^2(alpha*x)/alpha (no native snake op).
- Qwen-TTS-Tokenizer-12Hz encoder: NOT needed for Route A TTS. Needed later
  for voice-clone reference audio (Base model ref_audio path) and for offline
  Route B data prep on contributed corpora. Deferred to wave 5.

### 3.2 Runtime flow

```
mic / typed text -> Gemma 4 trunk -> answer tokens
  -> detokenize (fork SentencePiece) -> UTF-8 -> Qwen BPE retokenize
  -> Talker prefill + autoregressive codebook-0 decode (+ MTP codes 1-15/frame)
  -> each complete 16-code frame pushed to Code2Wav
  -> streaming PCM chunks -> audio output
```

Streaming contract: first chunk emitted at the first complete frame (80 ms of
speech); chunk granularity thereafter = Code2Wav receptive field. Target first
packet within ~100 ms of Talker decode start on M-series.

### 3.3 Fork work

- Converter: new arch qwen3-tts-talker. Reuse DFlash/MTP precedents for the
  backbone + MTP composition.
- Sampler/runner: frame bookkeeping (one frame = [c0..c15]); no id-range
  routing needed in Route A because the Talker runs in its own context.
- CLI: tessera-s2s-cli for headless verification before Studio wiring
  (text in -> PCM out, timing report).

### 3.4 Talker quantization (architect decision, 2026-08-05)

The Talker goes through the Tessera calibration and quantization pipelines
(calibration -> tessera quantizer -> ternary ANE weight encoding). Two
purposes:

- Ship-quality quantization of the Talker with the same machinery used for
  trunks and towers.
- Multimodal coverage for ternary-dequant-on-arrival. The ane-fused-dequant
  campaign has only ever validated dequant on text matmul fixtures. The
  Talker (TTS LM: codebook-feature inputs, MTP composition, 3,072-entry codec
  head) is the first non-text model through the pipeline, and becomes the
  multimodal parity fixture for dequant-on-arrival once that path lands.

Campaign dependency, honest state at 2026-08-05: true fused
dequant-on-arrival currently FAILS parity (ane-fused-dequant Phase 0.5
finding: the MIL dequant chain is broken as a whole, individual constructs
exonerated). The champion that passes the 4-seed gate is host-side dequant to
F16 + plain ANE matmul, bit-identical to the dequant-on-host path.
Consequences for this wave:

- The W2 correctness gate runs code/logit parity against the F16 reference
  through the host-dequant path (the one that passes today).
- Ternary-dequant-on-arrival on the Talker is a tracking item gated on the
  ane-fused-dequant campaign; it is deliberately NOT in the S2S critical
  path. When the campaign lands on-arrival parity for text fixtures, the
  Talker fixture is re-run as the multimodal proof.
- Calibration data: TTS prompts + style/voice instructions (input side only;
  imatrix observes activations, output codes are irrelevant to it).

## 4. Instrumentation contract

### 4.1 Per-utterance record: llama.tessera.s2s.v1 (NDJSON)

- sid: device-local random UUID (same semantics as runtime trace sid; stripped
  on any promotion).
- Text: the exact tokens Gemma produced (post-retokenize Qwen ids too, so the
  pair is training-ready without re-derivation).
- Codes: full codebook-0 stream + acoustic layers 1-15, zlib-compressed
  base64 (code streams are highly compressible).
- Timing: retokenize us, Talker TTFT, per-frame decode rate, Code2Wav
  throughput, first-packet latency.
- Voice config: preset id or reference-audio content hash (never the raw
  reference audio).
- Implicit feedback: interrupted mid-utterance, regenerated, replayed.
- Provenance: schema stamp + model digests in source-manifest lineage style.

### 4.2 Tier classification (usage dataset spec applies)

- Aggregates (rates, latencies, code-entropy stats, frame counts): Tier A,
  egress-eligible under the existing anonymization route.
- Text: Tier B, through the anonymization stage before anything else.
- Codes: Tier B, LOCAL-ONLY. Code sequences reconstruct voice through
  Code2Wav, so they are voice-bearing (biometric-adjacent). They must never
  reach dataset staging.
- Waveform: not captured by default; explicit opt-in capture flag only.

### 4.3 Store plumbing

- TesseraTraceStore grows an appendS2S entry point writing traces-s2s-*.jsonl
  with the same rolling-cap and quarantine-exempt discipline as runtime
  traces.
- Curation reuses the existing ledger machinery: session scorecards rate voice
  sessions with interruption/regeneration as quality proxies, so the Route B
  corpus filter is a curation query, not a new system.

## 5. Route B consumption

- Corpus = retained (text, codes) pairs from sessions that were NOT
  interrupted or regenerated (curation verdict).
- Training recipe (offline, architect infra): freeze trunk; train the 3,072
  new code embedding rows full-rank; LoRA on attention/MLP; code head;
  text-data mixing to guard against text regression.
- Inference shape after B: Gemma emits text tokens, flips to codec codes at a
  speech-start control marker (reuses the boa/eoa family); sampler branches on
  id range; Code2Wav unchanged. One vocab, one LM, zero retokenization.
- Drafter story: the Tessera flywheel transfers to the interleaved text + code
  stream unchanged (traces, LK training, runtime capture all apply as-is).
  Codec codes at 12.5 Hz are highly predictable, so drafter acceptance on the
  speech segment should be strong.

Open decision flagged, not resolved here: Route B training consumes
voice-bearing pairs. That data cannot flow through the anonymous egress lane;
if training happens on architect infrastructure, contributed pairs need a
separate, explicit opt-in consent lane distinct from the T&Cs collection.

### 5.1 Readiness gate (versioned tuning knobs, shipped with gate v1)

- Retained pair-hours >= N.
- Codebook-0 per-frame conditional entropy <= X (is the code stream learnable
  from text alone, measured on real usage).
- Measured two-stage latency gap >= Y ms at p50 (is Route B worth its cost).
- Replay/regeneration ratio as MOS proxy.

## 6. Drafter compounding (why this stacks)

- The Talker is 1.7B and its code stream is redundant: an LK drafter over the
  Talker is a natural follow-on, and the section 4 instrumentation doubles as
  its trace producer (same spec telemetry family).
- If Route B lands, the same drafter machinery targets the trunk's speech
  segment with no schema change.

## 7. Sequencing

- W1: this doc; lock decisions in section 8.
- W2: Talker conversion + calibration + tessera-pipeline quantization
  (ternary ANE encoding, section 3.4) + golden parity tests (HF vs GGUF
  logits and sampled codes on fixed prompts, host-dequant path).
- W3: Code2Wav graph + streaming PCM + tessera-s2s-cli end to end.
- W4: instrumentation + trace store + Studio audio node + session wiring.
- W5 (optional, deferred): Tokenizer-12Hz encoder graph for voice clone.

## 8. Open questions for the architect

Resolved 2026-08-05: Talker quantization goes through the Tessera
calibration/quantization pipelines as the ternary-dequant-on-arrival
multimodal test target (section 3.4). Plain offline quant (e.g. Q8_0) is off
the table.

Still open:

1. Code storage in traces: zlib base64 by default; raw as opt-in?
2. First-ship voice mode: CustomVoice presets (no encoder graph needed) vs
   Base voice clone. CustomVoice recommended for W2-W4, clone deferred to W5.
3. Route B consent lane: opt-in contribution of voice-bearing pairs, separate
   from the anonymous dataset route. Confirm the lane exists before W4 ships
   capture defaults.

## Appendix A: staged assets (source-manifest)

- Gemma 4 12B Unified (bf16 + source tokenizer) - Apache-2.0, on external
  drive.
- Gemma 4 12B MTP assistant drafter - Apache-2.0.
- Gemma 4 12B DFlash drafter - license UNRESOLVED (z-lab), no redistribution.
- Qwen3-TTS-12Hz-1.7B-Base - manifest entry present, download pending.

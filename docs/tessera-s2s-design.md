# Tessera S2S: Route A (text bridge) + instrumentation for Route B

Status: design + W2 converter landed. Architect decisions landed 2026-08-05:
Talker quant via Tessera pipelines with chained end-to-end calibration (3.4),
CustomVoice presets with voice cloning on indefinite hold (3.1, 7), trace code
storage default-on with no opt-out (4). Route B consent lane researched
2026-08-05: Talker resynthesis is a voice-release lane, viable under eight
binding conditions (section 8). Talker GGUF conversion landed 2026-08-05:
qwen3-tts-talker arch, 404 tensors, byte-parity verified
(tools/tessera/verify_qwen3tts_gguf.py).
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
   target for ternary-dequant-on-arrival (section 3.4). Calibration is the
   ENTIRE Gemma 4 -> Qwen3-TTS chain run and instrumented together, so the
   Talker is quantized against the trunk's real output distribution.
5. Voice cloning is on indefinite hold. CustomVoice presets only. Trace code
   storage is default-on with no opt-out (mandatory collection doctrine).

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
- Qwen-TTS-Tokenizer-12Hz encoder: NOT needed for Route A TTS. Its only
  consumers would be voice-clone reference audio (indefinite hold) and
  offline Route B data prep on contributed corpora. No wave assigned; this
  graph ships only if cloning comes back or Route B needs corpus encoding.

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
  backbone + MTP composition. LANDED 2026-08-05 (conversion/qwen3tts.py).
  GGUF layout: backbone 28 layers at blk.0..27; code predictor 5 layers at
  blk.28..32 (deliberately NOT nextn layers); codec ids (vocab 3072) stay
  outside the 151936-token text vocab; codec control ids (pad 2148, bos 2149,
  eos 2150, think 2154, nothink 2155, think_bos 2156, think_eos 2157) and
  codec language ids ride as metadata; per-codebook predictor tensors carry
  a .{cid} suffix (cp_codec_embd.{cid}, cp_head.{cid}, cids 0..14); the
  speaker encoder is dropped (cloning on hold). The code predictor's
  per-codebook embeddings live in backbone space (2048-dim) even though the
  predictor hidden is 1024: the runtime adds them to the backbone hidden and
  projects via cp_proj (small_to_mtp_projection).
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

Dequant-on-arrival state (architect update 2026-08-05, supersedes the
ane-fused-dequant best.md Phase 0.5 snapshot): the fused on-arrival path now
dispatches dequant onto Accelerate+NEON and matmul onto CoreML in low-power /
iOS mode, and is coming along nicely. Consequences for this wave:

- The W2 correctness gate runs code/logit parity against the F16 reference
  through the current dispatch path.
- Ternary-dequant-on-arrival on the Talker rides the campaign's progress;
  the Talker fixture is the multimodal proof as soon as text-fixture parity
  holds. It stays out of the S2S ship gate.

Chained calibration (architect decision 2026-08-05): the Talker is NOT
calibrated standalone. Calibration runs the entire Gemma 4 -> retokenize ->
Talker chain instrumented together, so:

- Talker activations are collected on the trunk's REAL outputs
  (on-distribution by construction, not a curated prompt set).
- Instrumentation (section 4) is live during calibration, so calibration
  runs double as the first s2s.v1 trace corpus.
- The input segment (mic -> audio tower, when voice input is active) is part
  of the chained pass. That segment is the voice-bearing one; see section 8
  for the consent/anonymization treatment.

## 4. Instrumentation contract

### 4.1 Per-utterance record: llama.tessera.s2s.v1 (NDJSON)

- sid: device-local random UUID (same semantics as runtime trace sid; stripped
  on any promotion).
- Text: the exact tokens Gemma produced (post-retokenize Qwen ids too, so the
  pair is training-ready without re-derivation).
- Codes: full codebook-0 stream + acoustic layers 1-15, zlib-compressed
  base64 (code streams are highly compressible). Code capture is DEFAULT-ON
  with no opt-out, per the mandatory-collection doctrine; codes are Tier B
  local-only (section 4.2), so default-on storage creates no egress exposure.
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
- Codes: Tier B, LOCAL-ONLY, default-on capture with no opt-out. Code
  sequences reconstruct voice through Code2Wav, so they are voice-bearing
  (biometric-adjacent). They must never reach dataset staging. Note that
  with CustomVoice presets (no cloning), OUTPUT codes carry preset synthetic
  voices, not user voices; the voice-bearing segment of the chain is the
  INPUT side (mic -> audio tower), which the consent lane in section 8
  covers.
- Waveform: distinct from code storage; still not captured by default
  (separate decision, not yet made).

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

Open decision RESOLVED by the section 8 research verdict (2026-08-05): the
acoustic channel of voice-bearing pairs is severed by construction under the
lane conditions (preset-voice resynthesis, original audio deleted), and the
remaining gate is the content layer. With content transformation plus the
versioned adversarial battery, pairs flow under the T&Cs disclosure with no
per-pair opt-in lane; records that fail the battery stay Tier B local-only.
See section 8.5.

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
- W2: Talker conversion + CHAINED end-to-end calibration (Gemma 4 -> Talker,
  instrumented, section 3.4) + tessera-pipeline quantization (ternary ANE
  encoding) + golden parity tests (HF vs GGUF logits and sampled codes on
  fixed prompts, current dispatch path).
- W3: Code2Wav graph + streaming PCM + tessera-s2s-cli end to end.
- W4: instrumentation + trace store + Studio audio node + session wiring.

Voice cloning: on indefinite hold (architect 2026-08-05). No wave assigned;
resurrecting it requires the Tokenizer-12Hz encoder graph (section 3.1) and
revisits the consent lane from scratch.

## 8. Route B consent lane: research verdict (2026-08-05)

Question (architect research direction): can the Talker itself anonymize
voice-bearing pairs - transcribe user speech, re-synthesize with a preset
synthetic voice, keep (text, synthetic codes), discard the original voice -
so pairs can flow into Route B training without a separate opt-in lane?

Verdict: the lane is viable, under a precise name. The Talker releases the
voice from the pair; it does not anonymize the pair. The acoustic channel is
severed by construction; the linguistic channel is untouched, and content is
a measured, exploitable identifier. The lane produces anonymous data only if
the content layer is separately transformed and evaluated before anything is
retained, trained on, or shipped. Name it a voice-release lane
(biometric-modality-removal lane), not an anonymization lane. Full evidence
record (41 sources, primary legal texts verified):
docs/research/tessera-route-b-consent-lane-research.md.

Resolved alongside (architect decisions, 2026-08-05, unchanged): Talker
quantization via Tessera pipelines with chained end-to-end calibration
(section 3.4); CustomVoice presets with voice cloning on indefinite hold;
code storage default-on with no opt-out (mandatory collection doctrine),
codes Tier B local-only.

### 8.1 Acoustic channel: severed by construction

- Text-bottleneck resynthesis is the effective class of voice anonymization.
  VoicePrivacy 2024 (semi-informed attacker whose speaker-verification model
  is retrained on the anonymized corpus itself): cascaded ASR->TTS systems
  reach near-chance privacy scores (EER ~48%, where 50% is chance); voice
  conversion systems that condition directly on the user's audio leak
  speaker identity (EER 8-20%).
- The Talker variant is strictly stronger than any evaluated cascade: code
  generation is conditioned ONLY on text and a fixed preset voice embedding.
  No F0 path, no duration path, no emotion path from the user's audio into
  the retained codes, because there is no path at all.
- Binding constraint: presets-only is load-bearing. If reference-audio
  conditioning returns (cloning, prosody transfer), the lane reopens exactly
  the channel the voice-conversion literature shows to be leaky. Route B
  data generation must forbid the reference-audio path as a hard constraint,
  not a product default. Consistent with the cloning hold (3.1, 7).

### 8.2 Content channel: untouched, and it identifies

- Author-attribution models re-identify speakers from transcripts at
  macro-F1 0.87-0.90, from as few as five transcripts in open-set
  conditions; stylometry survives punctuation stripping; ASR errors do not
  protect (attribution on errorful output performs as well as on clean
  transcripts).
- LLM-era attacks: zero-shot named-speaker identification from dialogue
  (52% -> 98% recall with GPT-4 on a named corpus); anonymous forum users
  re-identified from writing alone at 25-67% recall and 70-90% precision;
  agentic inference-driven linkage reconstructs identities from individually
  non-identifying cues (79.2% vs 56.0% for classical matching).
- The constructive results define the mitigation: named-entity masking alone
  fails; LLM paraphrase that alters STYLE, not just entities, cuts attacker
  performance by more than 50%; contextual rewriting is the recommended
  defense for ASR-TTS pipelines. Because the Talker synthesizes from the
  transformed text, the (text, codes) pair stays training-aligned AFTER the
  transformation: the anonymizer and the training-data generator are the
  same pipeline stage.

### 8.3 Legal frame (EDPB Guidelines 02/2026 on Anonymisation)

- Governing test: No Record Isolation, No Linkage, No Inference; all three
  passed means the data can safely be considered anonymous. Relative
  approach (CJEU EDPS v SRB): the same dataset can be personal for the
  vendor and anonymous for a keyless public recipient.
- Preset-voice codes with the original audio destroyed pass all three by
  construction. Raw transcripts fail all three. Transformed text is where
  the lane earns anonymity or documents residual risk - the further-analysis
  posture the guidelines prescribe.
- Anonymization is itself processing of personal data: the whole chain
  (capture -> transcribe -> transform -> resynthesize -> pool -> egress)
  needs a lawful basis, disclosure at collection time, and retained
  documentation, regardless of how anonymous the final artifact is.
- Wording rule: never label data "anonymous", "de-identified" or
  "de-personalised" while any component remains attributable.
- Models trained on retained pairs are a second retention site: if a trained
  drafter or TTS model can be prompted to regurgitate a user utterance, the
  lane fails No Inference through the model even if the dataset passed
  (EDPB Opinion 28/2024 extraction/query test). Memorization control is a
  release gate (C5).
- Immediate anonymization plus immediate deletion of the original data is an
  explicit positive factor in the legitimate-interest balancing test. C6 is
  recognized mitigation, not just hygiene.

### 8.4 Lane conditions (binding for W4 capture defaults)

- C1 Scrub-before-synthesize: deterministic PII scrub first, then
  style-transforming paraphrase with a local model, then Talker synthesis
  from the transformed text.
- C2 No user-audio conditioning: Talker input is transformed text plus the
  fixed preset voice embedding only; no F0/duration/emotion passthrough;
  cloning stays forbidden for Route B data generation.
- C3 No-key pooling: no device, session, or contributor identifier anywhere
  in the pipeline; sid stripped before retention; batches shuffled before
  local training and before egress (NO-KEY invariant).
- C4 Adversarial evaluation gate: run attribute-inference and authorship
  attacks locally against transformed text; drop failing records; version
  the battery as a tuning knob.
- C5 Memorization control: train only on transformed content, deduplicate,
  gate released models against extraction and regurgitation.
- C6 Immediate-deletion semantics: original voice discarded after
  transcription, before transformation; deletion logged in pipeline
  telemetry.
- C7 Disclosure at capture: versioned consent receipt names every stage of
  the chain; never claim anonymity while content remains attributable.
- C8 Irrevocability disclosure: contributions cannot be individually removed
  once past the no-key pooling boundary; state this.

The conditions are mutually dependent: C1 without C4 ships content that
resists attack only probabilistically; C4 without C3 still lets an attacker
aggregate one user's records within a batch by style (No Linkage looks
across datasets); C5 without C1 trains models on identifiable content and
bets on the model not leaking it, which the membership-inference literature
says is a losing bet for ASR/TTS systems.

Draft disclosure (for legal review; doubles as the binding contract for what
the pipeline may do - a pipeline version change requires a receipt version
change):

"When you use voice mode, your speech is transcribed on this device and the
recording is deleted immediately. Before anything is kept, the transcript is
rewritten to remove personal details and your individual turns of phrase,
and the rewritten text is re-synthesized in a synthetic voice that is not
yours. The resulting pairs of rewritten text and synthetic-voice codes may
be used to improve speech models on this device and, in batches containing
no device, account, or contributor identifiers, may be published to improve
speech models elsewhere. Your voice is never stored and never leaves your
device."

### 8.5 Effect on the section 5 open decision

The acoustic channel no longer blocks the lane: under C2 + C6 the retained
codes are preset-voice synthetic and the user's voice never persists. The
remaining gate is the content layer: if C1 + C4 pass the versioned battery,
transformed pairs are in the documented-residual-risk posture the guidelines
prescribe, and the T&Cs disclosure (C7) covers the chain - no per-pair
opt-in lane is needed for local training or keyless pooled egress. If the
battery fails for a record class, those records stay Tier B local-only; the
lane degrades gracefully and does not block Route B training on local pairs.

Precedents reviewed: Apple's Siri program (kept text, dropped audio, human
listening opt-in only) and Mozilla Common Voice (real-voice donation with a
persistent client_id - exactly what the NO-KEY invariant rejects). The
proposed lane is novel relative to both; its disclosure cannot borrow
existing wording.

### 8.6 Residual risk posture

Content-channel risk is managed, not proven away: best single-pass
anonymizers leave ~27% residual re-identification against LLM attackers in
benchmark; public individuals are disproportionately exposed; agentic
re-identification improves with every attributable record, so per-user
contribution caps per published batch belong in the battery design.
Guidelines 02/2026 are v1.0 in consultation until 2026-10-30; the lane is
built on the framework's durable parts (three criteria, further-analysis
posture, process-compliance duties), with periodic reassessment as both the
final text and attacker capabilities evolve.

## Appendix A: staged assets (source-manifest)

- Gemma 4 12B Unified (bf16 + source tokenizer) - Apache-2.0, on external
  drive.
- Gemma 4 12B MTP assistant drafter - Apache-2.0.
- Gemma 4 12B DFlash drafter - license UNRESOLVED (z-lab), no redistribution.
- Qwen3-TTS-12Hz-1.7B-Base - manifest entry present, download pending.

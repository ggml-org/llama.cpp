# Multi-modal Calibration Extension for Tessera

Design only. No implementation code. Targets gemma 4 12b unified
(encoder-free, text + vision + audio, single decoder-only transformer).

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) validates modality as a first-class regime axis. The
> modality-weighted fitness (0.5/0.3/0.2) is the alpha-weighted composite
> objective extended across modality, and modality enters the MAP-Elites
> descriptor space. Locked decisions M1-M8 hold. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md)
> Section 4.3.

Architect decisions (already locked, not revisited here):

- Extend calibration for multimodal: multi-modal corpus + modality-tagged
  imatrix + per-modality AWQ + multi-modal L5.
- Runtime gets BOTH a per-call `tessera_modality_t` AND per-modality
  precomputed components in the GGUF. Both are additive, not either-or.
- 5k multi-modal calibration set must be license-clean (research AND
  commercial). Curated combination (no single set is sufficient).

### Architect decisions on the 8 open questions (2026-07-30)

The scoping agent surfaced 8 questions in section 10. The architect
(2026-07-30) locked the following answers; the agent's leans in
section 10 are superseded by the items below.

**M1. Modality weights default (text / image / audio).** Locked:
`0.5 / 0.3 / 0.2`. Text is primary; image and audio are weighted by
their share of the curated 5k set (1.5k text / 2k image / 1.5k audio).
The GA can evolve away from this seed; the seed sets the v1 bias.

**M2. Missing `modality_scales` for one modality.** Locked: ERROR by
default, with a `--tessera-missing-modality={error,fallback-text}`
flag. Default is `error`; the runtime emits a clear remediation
message ("this model was calibrated for text only; image/audio
inference is not calibrated. Pass
`--tessera-missing-modality=fallback-text` to use the text scale
as a fallback, or re-calibrate the model with the multi-modal
imatrix"). Silent fallback masks calibration bugs; the explicit
failure path matches the libopenblas hard-fail precedent (see
`c++-port-design.md` decision 5).

**M3. Per-modality imatrix storage.** Locked: one file with
`modality_breakdown`. Single SHA, single receipt. Separate files
is a v2+ concern for parallel calibration runs.

**M4. GA fitness modes.** Locked: support both. Default is
all-modalities weighted; `--modality-filter {text,image,audio}`
restricts to a single modality. The filter is an output-targeting
flag, not a mode flag (consistent with the one-tool-one-mode
pattern in `c++-port-design.md` decision 4).

**M5. 5k set composition.** Locked: weighted 1.5k / 2k / 1.5k
(text / image / audio). The image-heavy 2k is justified by
PixelProse's CC BY 4.0 license uniformity; the audio 1.5k is
LibriSpeech's natural subsample.

**M6. Image preprocessing.** Locked: aspect-ratio preserving, both
H and W divisible by 48, default 280-soft-token budget (~645k
pixels). Larger budgets (560, 1120) are v2 ablation.

**M7. Audio preprocessing.** Locked: 16 kHz, 640-sample frames
(40 ms). Matches gemma 4 12b unified's audio path exactly;
LibriSpeech is native 16 kHz, no resample.

**M8. Per-modality AWQ alpha.** Locked: per-modality. Joint alpha
is the text alpha (backward compat); per-modality alphas are in
`modality_awq`. If `modality_awq` is missing for a modality, the
runtime uses the joint alpha (the text alpha) for that modality
(same fallback rule as M2's flag-controlled fallback).

Target preprocessing for gemma 4 12b unified (Gemma 4 12B dev guide,
HF Gemma4Unified docs):

- Image: patchify 16x16, merge 3x3 to 48x48, project to LM dim via
  `LayerNorm -> Dense -> LayerNorm` (35M params). Soft-token budgets
  70 / 140 / 280 / 560 / 1120. Default 280. Aspect-ratio preserved
  with both H and W divisible by 48.
- Audio: 16 kHz raw waveform, chunked into 640-sample frames (40 ms),
  one soft token per frame, projected via `RMSNorm -> Linear`. No
  mel spectrogram, no encoder.

References: see `docs/w4a4-calibration-design.md` (W4A4 mode),
`docs/c++-port-design.md` (imatrix v2 schema), `docs/pipeline-design.md`
(L1-L6 stages), `common/tessera-debug/tessera-debug.h` (L1 sidecar).

## 1. Multi-modal corpus extension

### 1.1 Schema additions to `tools/tessera/build-calibration-corpus.py`

The existing generator (`llama.tessera.calibration-corpus.v1` schema,
`tools/tessera/build-calibration-corpus.py:13-27`) currently emits
single-modality text records. We extend the same generator with five
new categories and a per-record `modality` field:

```
category           modality             source
----------------   -------------------  --------------------------------
text               text                 (existing generators)
image_text         image+text           curated image+caption subset
audio_text         audio+text           curated audio+transcript subset
image              image (no caption)   optional; for visual-only probes
audio              audio (no text)      optional; for acoustic-only probes
```

A composite `image_audio_text` category is reserved for future use
(VGGSound-style audio-visual pairs) and is NOT shipped in v1. The
schema bump is `llama.tessera.calibration-corpus.v1` ->
`llama.tessera.calibration-corpus.v2` with `modality` as the new
required field. The existing text records are re-emitted with
`modality: "text"`. The existing receipt fields stay.

### 1.2 Recommended 5k composition (lean recommendation, see Section 10)

The 5k set is composed of three license-clean sources. No single
public set covers text + image + audio at the right size with a uniform
license, so we curate a combination. License summary below; full
research in `docs/multimodal-calibration-design.md` Section 0 (Phase 1
research notes) and the table in Section 1.4.

| Modality     | Count | Source                                              | License    | URL                                                                                |
|--------------|------:|-----------------------------------------------------|------------|------------------------------------------------------------------------------------|
| text         | 1500  | New procedural (Tessera CC0 template)               | CC0        | (synthesized by generator)                                                         |
| image_text   | 2000  | PixelProse (CC-BY 4.0), CC12M/CommonPool/RedCaps subsample | CC BY 4.0  | https://huggingface.co/datasets/tomg-group-umd/pixelprose                          |
| audio_text   | 1500  | LibriSpeech `test-clean` (CC BY 4.0)                | CC BY 4.0  | https://www.openslr.org/12                                                         |
| **total**    | 5000  |                                                     |            |                                                                                    |

Why this mix:

- 16 kHz audio (LibriSpeech) matches gemma 4 12b unified's audio path
  exactly. No resampling. CC BY 4.0 with explicit "commercial use"
  OK. 2,620 samples in `test-clean`; subsample 1,500.
- PixelProse is uniform CC BY 4.0, synthesized captions from
  Gemini 1.0 Pro Vision over CC12M/CommonPool/RedCaps images. We
  pick a 2k stratified sample covering 100+ visual concepts
  (PixelProse already records concept coverage metadata in
  `vlm_captions_cc12m_*.parquet`).
- Text is a new Tessera CC0 procedural generator (see 1.3). The
  existing CC-BY-NC-SA-4.0 procedural corpus stays in the repo
  for backward-compat but is not used in the multi-modal 5k.

Composition is biased toward image+text (40%) because gemma 4
unified's image path is the heaviest: 35M embedder matmul + 280
soft tokens per image at the default budget, so the imatrix needs
more samples per channel there. Audio is smaller because there are
fewer audio-family tensors in the 12b (just the shared multimodal
embedder and the LM blocks that see audio activations). Text is the
smallest slice because text-only calibration is already covered by
the existing 2,730-record procedural corpus.

### 1.3 New CC0 procedural text generator

We add a sibling generator `tools/tessera/build-calibration-corpus-mm.py`
(or extend the existing file with a `--commercial` flag) that emits
1,500 text records under CC0. Same shape as the existing records
(`schema`, `id`, `category`, `text`, `origin`, plus the new `modality`
field). Origin string is `tessera.clean-room.cc0`. No NC, no SA.
Generated from the same procedural templates, but tagged CC0 so the
multi-modal recipe can be distributed commercially.

### 1.4 Rejected candidates and rationale

Documented in Section 11 (risk register). Short version:

- COCO Captions (CC BY 4.0 annotations, but per-image Flickr licenses
  are mixed; need a per-image license filter and a small `commercial_ok`
  flag in the corpus index to mark which images survive the filter).
  Rejected for v1 because the per-image filter is a non-trivial
  consumer-side step. Revisit in v2 if PixelProse proves insufficient.
- COCO 2017 val 5k: same per-image license issue.
- Flickr30k: CC0 image set, but the captions are Flickr-attributed
  and the image collection has known NC contamination. PixelProse is
  cleaner.
- Flickr8k: CC BY-NC 4.0 (rejected, NC).
- LAION-COCO: 600M synthetic captions, but image licenses vary and
  the LAION redistribution model is the EU TDM research-only path.
  Rejected for commercial Tessera use.
- AudioCaps: CC BY 4.0 captions, but the audio is sourced from
  YouTube (AudioSet) and AudioCaps does not redistribute audio.
  Rejected because we cannot ship the audio.
- MusicCaps: CC BY-SA 4.0 (rejected, share-alike; would force
  Tessera calibration data under CC BY-SA).
- Clotho: CC0/CC-BY per-clip audio, but captions under Tampere
  University non-commercial license (rejected, NC).
- ESC-50: CC BY-NC 3.0 (rejected, NC).
- GigaSpeech: non-commercial research only (rejected).
- FSD50K: per-clip mixed (CC0/CC-BY/CC-BY-NC/Sampling+); dataset as
  a whole CC-BY 4.0; commercial use requires contacting authors.
  Rejected for v1 (the contact step blocks distribution).
- VGGSound: CC BY 4.0 metadata, but audio is YouTube-sourced and not
  redistributable. Same problem as AudioCaps. Rejected for v1.
- WavText5K: MIT metadata; audio sourced from SoundBible, Freesound,
  BigSoundBank (mostly MIT/CC0, but per-clip filter needed).
  Rejected for v1 because of the per-clip filter.
- ShareGPT4V: CC BY-NC 4.0 (rejected, NC).
- LLaVA-Pretrain: subset of CC-3M/LAION/SBU; the upstream has the
  same per-image license filter problem.
- Common Voice: CC0 (good for speech, would use in v2 expansion).
- AISHELL-1: Apache 2.0 (good for Mandarin speech, would use in v2).
- VoxPopuli: CC0 data (good for multilingual speech, v2 candidate).

### 1.5 v1 receipt schema additions

The training-corpus-receipt example at
`tools/tessera/training-corpus-receipt.example.json` is extended:

```
{
  "schema": "llama.tessera.training-corpus.v1",
  "epoch": 0,
  "sha256": "<corpus sha>",
  "index_sha256": "<index sha>",
  "generator": {
    "schema": "llama.tessera.calibration-corpus.v2",
    "version": 2,
    "seed": 640,
    "sample_count": 5000,
    "categories": { ... },
    "modality_counts": {          // NEW
      "text": 1500,
      "image_text": 2000,
      "audio_text": 1500
    },
    "image_samples": 2000,        // NEW
    "audio_samples": 1500         // NEW
  },
  "license": "CC0",
  "license_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
  "attribution": "Tessera clean-room procedural + PixelProse + LibriSpeech",
  "distribution_cleared": true,
  "contains_user_inference": false,
  "commercial_use": true,         // NEW: now true
  "share_alike": false,
  "sources": [
    { "name": "Tessera clean-room CC0 procedural text",
      "license": "CC0",
      "modality": "text" },
    { "name": "PixelProse (CC-BY 4.0) stratified 2k subsample",
      "license": "CC BY 4.0",
      "license_uri": "https://creativecommons.org/licenses/by/4.0/",
      "modality": "image_text",
      "attribution": "Singla et al. 2024, tomg-group-umd/pixelprose" },
    { "name": "LibriSpeech test-clean (CC BY 4.0) 1.5k subsample",
      "license": "CC BY 4.0",
      "license_uri": "https://creativecommons.org/licenses/by/4.0/",
      "modality": "audio_text",
      "attribution": "Panayotov et al. 2015, openslr.org/12" }
  ]
}
```

`commercial_use: true` is the breaking-change flag that distinguishes
v2 from v1. v1 receipt consumers should treat `commercial_use` as
optional and default to `false` if absent (backward compat).

### 1.6 Preprocessing pipeline

For each sample in the corpus, the imatrix runner (`tools/imatrix/imatrix.cpp`)
must be able to feed the gemma 4 12b unified model the right input
shapes. The corpus record format is extended to carry preprocessed
arrays OR a pointer to a preprocessed `.npy` blob. We pick the pointer
approach (smaller index, larger blobs; blobs are content-addressed).

```
record := {
  "schema": "llama.tessera.calibration-sample.v1",
  "id": "<sha256:24>",
  "category": "image_text",
  "modality": "image_text",
  "text": "<caption or transcript>",
  "image": { "path": "/tessera-corpus/blob/<sha>.rgb.npy",
             "format": "rgb16",       // raw 16x16x3 uint8 patches
             "soft_tokens": 280,
             "ph": <int>, "pw": <int> },
  "audio": null,
  "license": "CC BY 4.0",
  "source": "pixelprose"
}
```

Image preprocessing (one-time, offline, before calibration):

1. Load the source image (PixelProse carries the original URL;
   we re-host a 5k subsample in a Tessera cache).
2. Convert to RGB. No mean/std normalization (gemma 4 12b unified's
   pipeline does it internally; HF docs explicitly say "Gemma 4 12B
   Unified does not apply mean/std normalization").
3. Resize so both H and W are divisible by 48 and the total pixel
   count is within the 280-soft-token budget (default 645k pixels,
   matches the 280 soft-token default in the dev guide).
4. Patchify 16x16 and merge 3x3 to 48x48. Store as
   `(N_patches, 48, 48, 3)` uint8 -> `.rgb.npy`.
5. Record `ph`, `pw`, `soft_tokens` in the sample.

Audio preprocessing:

1. Load audio (LibriSpeech FLAC, 16 kHz).
2. Resample to 16 kHz if needed. LibriSpeech is already 16 kHz, so
   this is a no-op pass-through.
3. Compute the chunk count: `n_frames = ceil(num_samples / 640)`.
4. Store the raw int16/float32 array as `.audio.npy`, shape
   `(num_samples,)`. No mel spectrogram (gemma 4 12b unified does
   not use one).
5. Record `num_samples` and `n_frames` in the sample.

The runtime path inside `tools/imatrix/imatrix.cpp` already accepts
tokenized text via `-f some-text.txt`. We extend it with two new
flags:

- `--mm-image-dir <path>`: directory of `.rgb.npy` files referenced
  by sample index. Each forward pass substitutes the preprocessed
  image patches at the gemma 4 unified image-token positions.
- `--mm-audio-dir <path>`: same for audio.

The imatrix runner is changed to call `ggml-image-processor.c`
helpers (NEW helpers, listed in the imatrix.cpp change list in
Section 9 G1-MM) to materialise the per-modality activations before
the dequant observer. This is the only call-site change in
imatrix.cpp. Schema change is purely additive: existing text-only
runs ignore the new fields.

### 1.7 Passing the corpus to the imatrix runner

The runner accepts a multi-modal manifest file (one JSON per line)
at `--mm-manifest <path>`. Each line is a sample record. The runner
streams through the manifest, materialises the inputs per modality,
and accumulates per-tensor statistics tagged with the sample's
`modality`. The `--modality-filter` flag restricts the run to one
modality (used for per-modality ablation; defaults to "all"). The
existing `-f text.txt` is retained as a synonym for
`--mm-manifest --modality-filter text` for backward compat.

## 2. Policy JSON schema for `modality_scales`

### 2.1 Decision: extend in place, do not introduce a new schema

`llama.speculative.calibration-policy.v1` (the current wrapper, used
by `tools/tessera/per_tensor_calibrate.py:55` and friends) is
extended in place to v2 by adding one new field. The v1 reader
continues to work; v2 readers recognise the new field and the
absence of the field means single-modality (text-only legacy), no
breaking change for existing models.

The alternative (new schema `llama.tessera.modality-scales.v1`) was
considered and rejected: every consumer (`per_tensor_calibrate.py`,
`awq-evolve.py`, `septq_ab_validate.py`, `l3_hessian_trace.py`,
`policy-prior.py`, `pe_qat.py`, `moe_calibration.py`, `flrq_demo.py`,
`l5_orchestrator.py`) wraps the policy in the v1 schema today, and
adding a parallel schema doubles the consumer surface for no
benefit. The v1->v2 in-place bump matches the existing
`llama.tessera.per-tensor-calibration.v1 -> v2` precedent in the
W4A4 doc.

### 2.2 Schema diff (v1 -> v2)

```
llama.speculative.calibration-policy.v1 -> v2

+ "modality_scales": {                       // NEW, optional
+   "text":  { "alpha": <float>, "clip": <float>, "scale": [<f16>...] },
+   "image": { "alpha": <float>, "clip": <float>, "scale": [<f16>...] },
+   "audio": { "alpha": <float>, "clip": <float>, "scale": [<f16>...] }
+ }
+ "modality_awq": {                          // NEW, optional
+   "text":  { "alpha": <float>, "clip": <float> },
+   "image": { "alpha": <float>, "clip": <float> },
+   "audio": { "alpha": <float>, "clip": <float> }
+ }
+ "modality_weights": {                      // NEW, optional
+   "text":  <float>,   // default 0.5
+   "image": <float>,   // default 0.3
+   "audio": <float>    // default 0.2
+ }
+ "imatrix_modality_version": 2              // NEW, optional; bumps to 2
```

A v1 reader that does not know the new fields simply ignores them.
A v2 reader that sees a v1 file (no `modality_scales`) treats the
file as text-only and falls back to the existing `alpha`, `clip`,
`scale` at the policy root. The v2 reader that sees a v2 file but
with `modality_scales` missing for one modality (e.g. no audio)
falls back to the text scale for the missing modality. The fallback
rule is documented in Section 10 (open question Q2, lean: text
fallback, not error).

The per-modality `scale` array is the per-input-channel F16 scale
vector, shape `(in_dim,)`. It is the F16 cast of the existing
F32 `scale` field, parallel to how the v1 root `scale` is
structured. The F32 -> F16 cast is lossless in the dynamic range
the calibration observer reports.

### 2.3 Where the schema is written

`tools/tessera/per_tensor_calibrate.py:55` and the AWQ-GA in
`awq-evolve.py:36` are the two write sites. Both gain a new code
path that, when `--modality-scales` is passed, writes the per-
modality fields. The existing single-modality path is unchanged
(writes v1 fields; v2 reader sees the file as text-only).

### 2.4 Backward compat matrix

| File kind             | v1 reader         | v2 reader (this PR)                            |
|-----------------------|-------------------|------------------------------------------------|
| Old v1 (no MM)        | works             | works as text-only                             |
| New v2 with all 3 MM  | ignores MM fields | works                                          |
| New v2 with partial   | ignores MM fields | falls back to text scale for missing modality  |
| Pure v1 with 1 MM     | n/a               | rejects with "modality_scales partial" error   |

The last case is rejected because it is ambiguous. A partial
modality_scales (e.g. only `text` and `image`, no `audio`) is
treated as text-fallback for the missing modality (the design
choice is surfaced as Q2 in Section 10).

## 3. Multi-modal imatrix runner

### 3.1 One pass per modality through the same model

The imatrix runner (`tools/imatrix/imatrix.cpp`) is extended to
run a separate forward pass for each modality in the manifest, but
through the same loaded gemma 4 12b unified model. The pass order
is `text -> image -> audio` (the same order the inference loop
emits modalities, matching the gemma 4 unified docs that say
"Image content goes before the text" and "Audio content goes after
the text"). Per-pass accumulators are kept in
`std::array<Stats, kModalityCount>` indexed by `tessera_modality_t`,
where `kModalityCount = 3` and the enum is:

```
enum tessera_modality_t : uint8_t {
    TESSERA_MODALITY_TEXT  = 0,
    TESSERA_MODALITY_IMAGE = 1,
    TESSERA_MODALITY_AUDIO = 2,
};
```

The enum is declared in `common/tessera-debug/tessera-debug.h`
next to the existing imatrix_version helper, since the sidecar
writer and the imatrix runner both need it.

### 3.2 Output: imatrix v2 extended with `modality_breakdown`

The current imatrix v2 schema (per `common/tessera-debug/tessera-debug.h:238`
"imatrix_version": 2, see also `docs/c++-port-design.md:610-675`)
is extended in place. The per-tensor observer (`mstats` in
`imatrix.cpp`) gains a `modality_breakdown` array. The schema
bump is v2 -> v3 only for the per-tensor v2 fields, NOT for the
imatrix file format version. The file still uses
`llama.imatrix.v2` (the format header). The version bump lives
inside the per-tensor field group.

```
imatrix per-tensor v2 -> v3 (additive, optional)

+ "modality_breakdown": {                     // NEW, optional
+   "text":  { "in_sum2": [...], "in_sumabs": [...], "in_sum4": [...],
+              "in_maxabs": [...], "counts": <int>,
+              "kurtosis": [...], "p50": [...], "p95": [...], "p99": [...],
+              "llm_int8_outlier_mask": [...] },
+   "image": { ... same fields ... },
+   "audio": { ... same fields ... }
+ }
+ "modality_mode": {                          // NEW, optional
+   "text":  "in_kernel" | "auxiliary",
+   "image": "in_kernel" | "auxiliary",
+   "audio": "in_kernel" | "auxiliary"
+ }
+ "shared_s_j": {                             // NEW, optional (MoE)
+   "text":  [...],   // per-family shared s_j vector
+   "image": [...],
+   "audio": [...]
+ }
```

`in_kernel` means the modality observer runs on the same matmul
graph as the dequant (no extra memory traffic, see W4A4 doc
section 3). `auxiliary` means the observer runs on a side path
after the matmul (the legacy behaviour, default for any modality
not explicitly set to `in_kernel`). For gemma 4 12b unified, we
default to `in_kernel` for `text` and `auxiliary` for `image` and
`audio` (because the per-modality activation shapes are different
and the in-kernel observer for non-text modalities needs
adapters that we do not commit in v1).

`family_breakdown` stays as a rollup. The rollup is the sum of the
three modality_breakdown entries, exactly the way it is computed
today for the unimodal v2. v1 readers that only know
`family_breakdown` see no change.

### 3.3 Per-modality mode flag and MoE shared `s_j`

`modality_mode` lets us mark a modality as observed in-kernel vs
auxiliary without losing backward compat (v1 readers ignore the
field and the modality_breakdown sum still reproduces the v1
family rollup). For MoE-only tensors (gated FFN with experts), the
`shared_s_j` per family per modality field is added so the GA can
search shared expert scales without re-running calibration. v1
readers ignore it.

## 4. Per-modality AWQ + GA fitness

### 4.1 Per-modality AWQ

`tools/tessera/per_tensor_calibrate.py:55` already runs AWQ per
tensor; the new code path runs the same optimisation per modality.
For each tensor, we now have three candidate sets
(`{alpha, clip, ternary_threshold, ...}`) per modality instead of
one. The new code path is gated on
`--modality-aware` (defaults to `false`; v1 single-modality
behaviour is preserved).

### 4.2 GA fitness becomes weighted

The fitness in `awq-evolve.py:36` becomes:

```
loss = w_text  * loss_text  + w_image * loss_image + w_audio * loss_audio
```

Default weights from Section 10 (Q1): `w_text = 0.5`,
`w_image = 0.3`, `w_audio = 0.2`. The text weight is highest
because text is the dominant path (most tokens in a typical
inference workload are text). Image is next because the 280-token
default budget means ~4% of inference tokens are image patches at
the typical vision-budget configuration. Audio is last because
gemma 4 12b unified's audio path is the smallest contributor to
the inference token count for typical workloads.

The weights are exposed as CLI flags:

```
--w-text   <float>  default 0.5
--w-image  <float>  default 0.3
--w-audio  <float>  default 0.2
```

They are also written to the policy as `modality_weights` so the
audit trail captures the weight choice.

### 4.3 GA space gains `modality_weights` as genes

`awq-evolve.py` already evolves a vector of numerical genes via
`Candidate` (`awq-evolve.py:52-65`). The gene vector gains three
real-valued genes `w_text`, `w_image`, `w_audio` clamped to
`[0, 1]` and renormalised so they sum to `1`. The gene range is
small (`[0.1, 0.7]` per gene) and the initial values are the
defaults above; the GA only perturbs them within a small range so
the search stays anchored at the text-heavy prior.

### 4.4 Three output policies + joint policy

The GA produces four output policies per tensor:

- `policy_text`  : text-only AWQ parameters and scales
- `policy_image` : image-only AWQ parameters and scales
- `policy_audio` : audio-only AWQ parameters and scales
- `policy_joint` : weighted-joint AWQ parameters and scales (this is
  the existing single-policy output, preserved)

The joint policy is what `tile640_quantize_v3.py` consumes in the
default code path. The three per-modality policies are emitted to
the GGUF as the `modality_awq` field. If the field is absent, the
runtime falls back to the joint policy for the missing modality
(text-only legacy behaviour, same fallback as the scales field).

## 5. Runtime: BOTH modality ID + per-modality components

### 5.1 Modality ID passed per call

The C++ dequant entry point gains a `tessera_modality_t` parameter
(see Section 3.1). The dequant call site is in the matmul
dispatcher in `ggml/src/ggml-cpu/arch/arm/quants.c` for the Tile640
Ternary matmul, and the equivalent paths in the Metal and CUDA
backends. Concretely, the signature of
`tile640_dequant_ternary` (or whatever the current name is in the
v3 quantize path) becomes:

```
void tile640_dequant_ternary(
    const void * weight_packed,
    const void * weight_page_scales,
    const void * weight_lane_scales,
    const void * weight_outlier_vals,
    const void * weight_outlier_idx,
    const void * weight_outlier_count,
    const void * weight_act_scale_text,
    const void * weight_act_scale_image,   // NEW
    const void * weight_act_scale_audio,    // NEW
    float * dst,
    int64_t rows,
    int64_t cols,
    tessera_modality_t modality              // NEW
);
```

The 1-3 line change at the call site is the new modality parameter
plumbed through from the dispatcher's modality argument, and a
single branch that picks the right act_scale pointer:

```
// existing scalar at the call site
const void * act_scale = modality == TESSERA_MODALITY_IMAGE
    ? weight_act_scale_image
    : modality == TESSERA_MODALITY_AUDIO
      ? weight_act_scale_audio
      : weight_act_scale_text;   // text and unknown
```

That is the entire runtime change in v1. The dequant kernel body
is unchanged.

### 5.2 GGUF format: 9 components instead of 7

The current Tile640 ternary weight has 7 components in the GGUF
(`weight_packed`, `weight_page_scales`, `weight_lane_scales`,
`weight_outlier_vals`, `weight_outlier_idx`, `weight_outlier_count`,
`weight_act_scale`). For multi-modal we add two more:

```
existing (7):
  weight_packed
  weight_page_scales
  weight_lane_scales
  weight_outlier_vals
  weight_outlier_idx
  weight_outlier_count
  weight_act_scale (text, the existing field; alias act_scale_text)

new (2):
  weight_act_scale_image
  weight_act_scale_audio
```

Total: 9 components. The new fields are F16 vectors of shape
`(in_dim,)` (same as the existing `weight_act_scale`). The loader
in `ggml/src/ggml-cpu/arch/arm/quants.c` (or wherever the
componentised Tile640 load is) reads all 9 in one pass; missing
fields are zero-filled and the runtime falls back to
`weight_act_scale` (text). This is the same fallback as Section 2.4
(Q2) and is the reason the loader is the right place to do the
zero-fill, not the quantizer.

### 5.3 The modality ID is also recorded in the v3 sidecar

`tessera-debug.h`'s per-row v3 strip
(`tessera-debug.h:90-95`, the `row_v3_meta` 24-byte row) is
extended. The 8-byte `reserved` field is split:

- 4 bytes: `modality_id` (uint32, one of `tessera_modality_t`)
- 4 bytes: `reserved2` (zero for now)

The v3 reader (`tools/tessera/l3_sidecar_v3_reader.py`) recognises
the new field. The v2 reader sees the field as opaque padding
(`tessera-debug.h:88-91`, "v2 reader: ... the per-row v3 strip is
opaque padding, then reads data at offset 40 + R*4"). No backward-
compat break: the v3 strip already had a reserved field; we are
just re-purposing half of it.

The sidecar header version does not bump. The header is still
`DEQUANT_FILE_VERSION = 3`. The reader detects the modality_id
field by checking that the sidecar was opened with the multi-modal
mode (set via `set_telemetry_model` or a new
`set_telemetry_modality`) and reads zeros for the missing field on
older files.

### 5.4 L1.5 reference read stays the same

The L1.5 reference sidecar (FP16 reference, written only in
`w4a4` mode) is modality-agnostic. The reference is the BF16-cast
of the FP16 weight, and the modality does not change the
reference. No format change. The L1.5 telemetry, when wired, can
be modality-tagged for per-modality error reporting; this is a
future option and not in v1.

### 5.5 gemma 4 unified inference loop already knows the modality

The gemma 4 12b unified inference loop already branches on
"image / audio / text" content (the gemma 4 docs say "Image
content goes before the text in your prompt" and "Audio content
goes after the text"). The C++ dequant just needs to be told
which modality it is processing. The modality argument is set
once per matmul call from the caller's modality state, which is
already in scope at the dispatch site. Zero new abstractions.

## 6. Multi-modal L5 scorer

### 6.1 Per-tensor sensitivity becomes a per-modality vector

`tools/tessera/l5_metrics.py` (`DEFAULT_WEIGHTS` at line ~50, see
`l5_orchestrator.py:873-885` for how the weights are passed) gains
per-modality weights. The 6-signal score
(imatrix, gradient, layer, kurtosis, hessian, outlier) becomes a
per-modality score, then is weighted by `modality_weights`. The
math is:

```
score_text  = sum_i w_imatrix * imatrix_i + w_gradient * grad_i + ...
score_image = sum_i w_imatrix * imatrix_i + w_gradient * grad_i + ...
score_audio = sum_i w_imatrix * imatrix_i + w_gradient * grad_i + ...
score = w_text * score_text + w_image * score_image + w_audio * score_audio
```

`tools/tessera/l5_orchestrator.py:873-885` already has the
`--w-imatrix`, `--w-gradient`, `--w-layer` flags. The new code
path adds `--w-kurtosis`, `--w-hessian`, `--w-outlier` (to make
the full 6-signal score reachable; currently only 3 are exposed)
and the modality counterparts `--w-text`, `--w-image`, `--w-audio`.

### 6.2 L5 orchestrator reads modality-tagged imatrix v2

`l5_orchestrator.py` already reads the imatrix via
`_read_imatrix(args.imatrix)`. The reader is extended to recognise
`modality_breakdown` and pass it to the scorer. Missing
`modality_breakdown` falls back to the v2 family_breakdown rollup
(behaves as if all modalities were text).

### 6.3 Spearman rho matrix per-modality and cross-modality

The 6-signal agreement analysis (Spearman rho between signals)
becomes 3 x 3 = 9 matrices: 3 per-modality (within-modality signal
agreement) and 6 cross-modality (between-modality agreement). The
output JSON gains a `modality_agreement` field. The cross-modality
agreement is a per-tensor rank correlation: it tells us whether
the same tensors are flagged sensitive across modalities, which is
the diagnostic we need for Section 8 (telemetry bottleneck).

## 7. Bundle format extension

### 7.1 `.npz` layer bundles carry `modality`

`tools/tessera/make-awq-layer-bundles.py:218-238` writes a `.npz`
file per tensor with `weight`, `in_sum2`, `in_sum4`, `in_maxabs`,
`counts`, `name`, `family`, `expert`. The new field is `modality`
stored as a 0-d int8 array (one of the enum values).

```
np.savez_compressed(
    output / f"{safe_name}.npz",
    name=...,
    family=...,
    expert=...,
    modality=np.asarray(modality_id, dtype=np.int8),   // NEW
    weight=...,
    in_sum2=...,
    in_sum4=...,
    in_maxabs=...,
    counts=...
)
```

A reader that does not know the `modality` field reads it as
`data['modality'].item() if 'modality' in data else 0` (defaults
to text). The bundle writer is extended in
`make-awq-layer-bundles.py` to read the per-modality
imatrix_v2.gguf (Section 3) and select the per-modality
observers when building the bundle. The bundle is one per
tensor (not one per tensor per modality); the modality field
is a single int8, so the bundle is per-modality-aware but not
per-modality-duplicated.

### 7.2 Backward compat for the bundle

A bundle without `modality` defaults to text. Per-modality GA
runs (Section 4) build three bundles per tensor (one per
modality) when the per-modality AWQ path is enabled, otherwise
one bundle per tensor with `modality=0` (text).

## 8. Telemetry: per-modality bottleneck collection

### 8.1 v3 sidecar `per_row_timing_ns` tagged with `modality_id`

Section 5.3 already documents the sidecar change: the `reserved`
field in the per-row v3 strip is split into `modality_id` and
`reserved2`. The sidecar is the same on disk, the reader just
learns a new field. The schema-version-neutral `L1 sidecar` is
readable by v1, v2, and v3 readers per the existing compat
matrix (`tessera-debug.h:81-100`).

### 8.2 Phase C latency LUT groups by (shape, kernel_id, modality)

`tools/tessera/latency_lut.py` currently groups by
`(shape, kernel_id)` (see `latency_lut.py:29-57`). The new code
path groups by `(shape, kernel_id, modality)`. The output JSON
schema bumps to `llama.tessera.latency-lut.v2`:

```
{
  "schema": "llama.tessera.latency-lut.v2",   // v1 -> v2
  "group_by": "shape-kernel-modality",        // new dimension
  "entries": [
    { "shape": "1024x4096", "kernel_id": 7, "modality": 0,
      "mean_ns": ..., "std_ns": ..., "count": ...,
      "mean_total_ns": ... },
    ...
  ],
  "summary": { "n_tensors": ..., "n_groups": ..., "n_kernel_ids": ...,
               "n_modalities": 3 }
}
```

v1 readers ignore `modality` and treat entries as text.

### 8.3 L1.5 reference modality-tagging (future)

Optional. The L1.5 reference sidecar can be modality-tagged for
per-modality error reporting. Not in v1. The hook exists
(`dequant_mode()` returns the current mode, and a future
`set_telemetry_modality()` would plumb the modality through).

### 8.4 `l3_outlier_report.py` gains `modality_breakdown`

`tools/tessera/l3_outlier_report.py` currently emits a per-tensor
outlier report. The new code path emits a `modality_breakdown`
table: per-tensor, per-modality outlier counts and a "hot
modality" tag (the modality with the highest per-row outlier
count). The new field is in the JSON output; the existing
table-format output gains a column.

### 8.5 Per-modality bottleneck view

A new top-level view in `l3_outlier_report.py` is the
"per-modality bottleneck" table. For each modality, list:

- Top 5 tensors by `mean_total_ns` for that modality.
- Per-tensor outlier count for that modality.
- Per-tensor "loss delta" from the L5 per-modality sensitivity
  (Section 6).

The view surfaces "image is bottlenecked on blk.16.attn_output
dequant (38% of per-modality time, 12% of per-modality outlier
count)" - the kind of insight that drives Phase C optimisation
choices. The view is additive: the existing per-tensor report is
unchanged.

## 9. Phased implementation plan (G0-MM through G4-MM)

### G0-MM: corpus builder + 5k set curation (this doc)

- Pre-doc research: Phase 1 (this section).
- Modify `tools/tessera/build-calibration-corpus.py` to emit v2
  with `modality` field.
- Add the CC0 procedural generator (Section 1.3).
- Add the 5k curator that subsamples PixelProse (2k) and
  LibriSpeech test-clean (1.5k).
- Add per-sample preprocessing scripts (image patchify, audio
  chunk) that emit the `.rgb.npy` and `.audio.npy` blobs.
- LoC estimate: ~400 lines (curator + preprocessor + tests).
- Dependencies: none. Standalone.
- Smoke gate: `python3 -m tools.tessera.build-calibration-corpus-mm
  --output-dir /tmp/mm-corpus --seed 640` emits a 5000-record
  manifest, the receipt has `commercial_use: true`, and the
  receipt is byte-identical across re-runs (deterministic seed).

### G1-MM: modality-tagged imatrix v2 + multi-modal runner

- Add `tessera_modality_t` enum to
  `common/tessera-debug/tessera-debug.h`.
- Extend `tools/imatrix/imatrix.cpp` `IMatrixCollector::mstats`
  with a `modality_breakdown` array.
- Add the per-pass loop in `IMatrixCollector::collect_imatrix`
  that tags each call with the current modality from
  `--mm-manifest`.
- Extend `tools/imatrix/imatrix.cpp` `save_imatrix` to write the
  modality_breakdown field. Bump the per-tensor field version
  to v3.
- LoC estimate: ~250 lines (header enum + imatrix.cpp changes).
- Dependencies: G0-MM.
- Smoke gate: a 100-sample multi-modal manifest runs end-to-end
  and produces a `.gguf` with `modality_breakdown` for each of
  the three modalities. The text-only legacy path is unchanged.

### G2-MM: per-modality AWQ + GA fitness + bundle format

- Extend `tools/tessera/per_tensor_calibrate.py:55` with the
  per-modality AWQ path. Add `--modality-aware` flag.
- Extend `tools/tessera/awq-evolve.py:36` GA to evolve
  `w_text`, `w_image`, `w_audio` genes and emit three
  per-modality policies + one joint policy.
- Extend `tools/tessera/make-awq-layer-bundles.py:218-238` to
  read the modality-tagged imatrix and write `modality` in the
  bundle.
- Bump the policy schema `llama.speculative.calibration-policy.v1`
  -> v2 (additive, see Section 2).
- LoC estimate: ~600 lines across the three files.
- Dependencies: G1-MM (needs the modality-tagged imatrix).
- Smoke gate: a 100-tensor GA run produces the four policies,
  the per-modality policies differ from the joint policy, and
  the bundle's `modality` field is non-zero for the image and
  audio tensors.

### G3-MM: runtime dequant kernel change + v3 sidecar modality tag + Phase C LUT extension

- Modify the dequant entry point signature in
  `ggml/src/ggml-cpu/arch/arm/quants.c` (and the Metal/CUDA
  parallels) to add `tessera_modality_t modality`. 1-3 line
  change per call site (Section 5.1).
- Modify the Tile640 GGUF writer in
  `tools/tile640/quantize_v3.py` to write
  `weight_act_scale_image` and `weight_act_scale_audio` alongside
  the existing `weight_act_scale` (Section 5.2).
- Modify the v3 sidecar row layout in
  `common/tessera-debug/tessera-debug.h` to split the reserved
  field into `modality_id` + `reserved2`. Update
  `tools/tessera/l3_sidecar_v3_reader.py` to read the new field.
- Extend `tools/tessera/latency_lut.py` to group by
  `(shape, kernel_id, modality)`. Bump to
  `llama.tessera.latency-lut.v2`.
- LoC estimate: ~200 lines (kernel signature, GGUF writer, v3
  reader, LUT).
- Dependencies: G1-MM (needs the sidecar writer change to be
  read by the new reader), G2-MM (needs the per-modality
  policies for the GGUF writer).
- Smoke gate: a dequant sidecar run on gemma 4 12b unified
  with the new dequant signature produces a sidecar with
  `modality_id` set; the LUT groups show three separate
  modality rows; the weight components number 9 instead of 7.

### G4-MM: multi-modal L5 + smoke + PPL eval on gemma 4 12b unified (text + image + audio)

- Extend `tools/tessera/l5_metrics.py` with the per-modality
  scoring and the 6-signal weight flags.
- Extend `tools/tessera/l5_orchestrator.py:65` to read the
  modality-tagged imatrix v2 and the per-modality policy. Bump
  the L5 schema to v2.
- Add a smoke driver that runs gemma 4 12b unified with the
  multi-modal calibration set, computes the per-modality PPL
  delta vs BF16, and emits a one-page report.
- LoC estimate: ~400 lines (metrics + orchestrator + smoke).
- Dependencies: G3-MM.
- Smoke gate: the per-modality PPL delta is < 0.5 for text,
  < 1.0 for image, < 1.5 for audio (matches the W4A4 doc's
  success criteria; weight-only path is expected to be better
  on text, so the bar is the same).

## 10. Open design questions (with lean recommendations)

The architect locked the answers to all 8 questions on 2026-07-30
(see items M1-M8 in "Architect decisions on the 8 open questions"
above). The agent's leans below are historical analysis; the
architect's decisions supersede them.

1. **Modality weights default** (text / image / audio). Lean:
   `0.5 / 0.3 / 0.2`. Justification in Section 4.2.

2. **Missing `modality_scales` for one modality** (e.g. only text
   was calibrated). Lean: fall back to the text scale for the
   missing modality. Alternative: hard error. We pick the
   fallback because the alternative is a non-functional
   multi-modal model on a partial calibration, which is
   strictly worse than a slightly-miscalibrated fallback.

3. **Per-modality imatrix storage: separate files or one file?**
   Lean: one file with `modality_breakdown`. The single-file
   approach makes the audit trail simpler (one SHA, one receipt)
   and avoids the "which file did the GA use?" question. The
   separate-file approach would be needed if per-modality
   runs are run in parallel on different machines; we do not
   need that in v1.

4. **GA fitness: all-modalities vs single-modality.** Lean:
   support both. The default is all-modalities (weighted). A
   `--modality-filter text|image|audio` flag restricts the GA
   to a single modality. This is useful for ablation and for
   the case where a modality-specific dataset is much larger
   than the others (e.g. a future expansion that has 50k audio
   samples but 5k image samples).

5. **5k set composition: balanced vs weighted.** Lean: weighted
   1.5k / 2k / 1.5k (text / image / audio). Justification in
   Section 1.2. A balanced 1.66k-each would over-sample text
   relative to gemma 4 12b unified's actual token distribution.

6. **Image preprocessing: resize to what?** Lean: aspect-ratio
   preserving, both H and W divisible by 48, total pixel count
   within the 280-soft-token budget (~645k pixels, default
   280). Larger budgets (560, 1120) are valid for ablation but
   are not the v1 default. For the 5k set, we pick a mix of
   aspect ratios so the calibration sees the full range; the
   exact aspect distribution is fixed by the curated sample.

7. **Audio preprocessing: sample rate, frame size.** Lean:
   16 kHz, 640-sample frames (40 ms). This matches gemma 4
   12b unified's audio path exactly and matches LibriSpeech's
   native sample rate, so no resampling is needed. Other
   datasets (e.g. AudioCaps at 48 kHz) would be resampled to
   16 kHz; we are not using AudioCaps in v1 (Section 1.4).

8. **Per-modality AWQ alpha: same across modalities or
   per-modality?** Lean: per-modality. The motivation is that
   image activations (pixel patches after the 35M embedder) and
   audio activations (raw 16 kHz chunks after the linear
   projection) have different dynamic ranges; the per-modality
   alpha lets the GA find a different per-modality tradeoff.
   The joint alpha is the text alpha (for backward compat) and
   the per-modality alphas are in `modality_awq`. If
   `modality_awq` is missing, the runtime uses the joint alpha
   for the missing modality (same fallback rule as the scales).

## 11. Risk register

- **License compatibility** (CC BY 4.0 vs CC BY-NC vs CC BY-SA).
  Section 1.4 lists the rejected candidates. The accepted
  candidates are CC0 (text), CC BY 4.0 (PixelProse, LibriSpeech).
  No NC, no SA. The current Tessera procedural corpus is
  CC-BY-NC-SA-4.0; it stays in the repo for backward compat but
  is not part of the multi-modal 5k.

- **Preprocessing mismatch** (image resize, audio resample)
  affects the calibration signal. We pin the preprocessing to
  gemma 4 12b unified's exact path (Section 1.6) and document
  it. A regression in the preprocessing library would
  invalidate the calibration; the smoke gate in G0-MM runs
  the preprocessing on a small known input and asserts the
  expected output shape.

- **Runtime dequant switch cost** (modality parameter branch in
  inner loop). The branch is a single integer compare against
  two constants, which the compiler turns into a select on
  every modern backend. There is no measurable cost. We
  measure in G3-MM to confirm.

- **Storage overhead** (3x act_scale per tensor). Each
  `weight_act_scale_*` is F16, shape `(in_dim,)`. For a 4k
  in_dim tensor, each is 8 KB. Three of them is 24 KB instead
  of 8 KB. The model has ~300 weight tensors, so the GGUF
  grows by ~5 MB. Negligible relative to the multi-GB model
  size.

- **PPL regression on under-represented modalities.** Audio is
  30% of the 5k set but ~5% of the inference token count for
  typical workloads. The risk is that the audio scale is
  under-trained. Mitigation: the per-modality AWQ path (Section
  4) re-fits per modality, so the audio-specific scale gets the
  audio data's gradient. If PPL regression shows up in G4-MM
  smoke, we revisit the 1.5k/2k/1.5k split (Section 10 Q5).

- **Per-modality corpus imbalance** (1.5k/2k/1.5k) under-trains
  the minority modalities. Text is the smallest slice in our
  proposed split, but text is the dominant inference path so
  the text under-training is not a real risk. Audio is the
  second-smallest; if the audio path is exercised heavily in
  the eval workload, we may need to expand the audio set.
  The 5k is a v1 floor; the corpus is designed to be
  extensible (Section 1.5 receipt schema) so a v2 expansion
  (e.g. add 10k more audio from Common Voice + AISHELL) is a
  one-line change to `modality_counts`.

## 0. Phase 1 research notes (online)

This section is the research that drove the candidate selection
above. Sources are cited inline. Each candidate was scored on
license, size, modalities, preprocessing, suitability for
calibration, and representative diversity.

### Candidates considered

**Image + text (with per-candidate assessment)**

- **COCO Captions (2017)**: CC BY 4.0 annotations; per-image
  Flickr licenses are mixed (CC0, CC BY, CC BY-NC). The 2017
  val split is exactly 5,000 images. Commercial use of the
  annotation set is allowed; commercial use of the image set
  requires per-image license filtering. Preprocessing: image
  resize to 48-divisible. Suitability: high. Diversity: high
  (object detection + captioning focus). Verdict: requires a
  per-image license filter for clean commercial use. Rejected
  for v1 because the filter step is a non-trivial consumer-
  side change. Source: https://cocodataset.org/,
  https://huggingface.co/datasets/whyen-wang/coco_captions,
  https://github.com/cocodataset/cocoapi/issues/81.

- **Flickr30k**: CC0 (public domain). 30k images, 5 captions
  per image, ~4.4 GB. Preprocessing: image resize. Diversity:
  high (Flickr photos covering a wide range of activities).
  Verdict: license-clean, but the 30k size means we have to
  subsample. The PixelProse option covers the same diversity
  with a uniform license and a denser caption set. Rejected
  because PixelProse is more recent and the captions are
  richer. Source:
  https://www.innovatiana.com/en/datasets/flickr30k-image-caption-dataset.

- **Flickr8k**: CC BY-NC 4.0. Rejected (NC). Source:
  https://www.kaggle.com/datasets/habedi/flickr-8k-dataset-clean.

- **Conceptual Captions (CC3M)**: research-only via the
  Google API. Rejected (no public download, no commercial
  license). Source:
  https://github.com/google-research-datasets/conceptual-captions.

- **LAION-COCO**: 600M synthetic captions; CC BY 4.0
  metadata but images are external (EU TDM research-only).
  Rejected for commercial use. Source:
  https://laion.ai/blog/laion-coco/.

- **PixelProse**: CC BY 4.0 (whole dataset). 16.9M
  image+caption pairs, synthesized by Gemini 1.0 Pro Vision
  over CC12M/CommonPool/RedCaps images. 2,520 patches per
  image at the default budget. 9.1M from CC12M, 6.5M from
  CommonPool, 1.3M from RedCaps. Preprocessing: image resize
  to 48-divisible aspect ratio. Diversity: high (concept-
  balanced by design). Verdict: best fit for the 2k image
  subsample. Source: https://huggingface.co/datasets/tomg-group-umd/pixelprose,
  https://arxiv.org/html/2406.10328v1.

- **LLaVA-Pretrain (LCS-558K)**: subset of LAION/CC/SBU with
  BLIP captions. Per-image license depends on the source.
  Rejected (per-image filter needed). Source:
  https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain.

- **ShareGPT4V**: CC BY-NC 4.0. Rejected (NC). Source:
  https://sharegpt4v.github.io/.

- **VQA v2 / OK-VQA / TextVQA**: CC BY 4.0 annotations, but
  images are COCO/OpenImages (per-image license filter
  needed). Rejected for v1; revisited in v2. Source:
  https://visualqa.org/terms.html, https://textvqa.org/dataset/.

- **PixMo (Molmo)**: ODC-BY-1.0 (commercial-friendly with
  attribution). 229k unique images with referring expressions.
  Verdict: license-clean, but the referring-expression format
  is VLM-training-specific, not captioning. Could be added in
  v2 for VQA-style calibration. Source:
  https://huggingface.co/datasets/allenai/pixmo-points,
  https://arxiv.org/html/2409.17146v2.

- **ScienceQA**: CC BY-NC-SA 4.0. Rejected (NC + SA). Source:
  https://scienceqa.github.io/.

**Audio + text**

- **LibriSpeech**: CC BY 4.0. 1000 hours of 16 kHz read
  English speech. Splits: dev-clean (2,703), dev-other
  (2,864), test-clean (2,620), test-other (2,939), train-
  clean-100 (28,539), train-clean-360 (104,014), train-
  other-500 (148,688). Preprocessing: 16 kHz is native, no
  resampling. Frame size 640 (40 ms) matches gemma 4
  unified exactly. Verdict: best fit for the 1.5k audio
  subsample. Source: https://www.openslr.org/12,
  https://huggingface.co/datasets/openslr/librispeech_asr.

- **Common Voice**: CC0. ~1,700 hours of English speech at
  48 kHz (downsampled at load time). 30+ languages. v2
  expansion candidate for multilingual coverage. Source:
  https://commonvoice.mozilla.org,
  https://en.wikipedia.org/wiki/Common_Voice.

- **AISHELL-1**: Apache 2.0. 170 hours of Mandarin speech.
  v2 expansion candidate for Mandarin. Source:
  https://www.openslr.org/33/, arXiv:1709.05522.

- **AISHELL-4**: CC BY-SA 4.0. 120 hours of Mandarin meeting
  audio. Rejected (SA). Source:
  https://www.global-datasets.com/en/d/aishell_4.

- **VoxPopuli**: CC0 (data). 400k hours of multilingual
  speech. v2 expansion candidate. Source:
  https://github.com/facebookresearch/voxpopuli.

- **GigaSpeech**: non-commercial research only. Rejected. Source:
  https://huggingface.co/datasets/speechcolab/gigaspeech.

- **The People's Speech**: CC BY 4.0 (CC-BY subset only, the
  CC-BY-SA portion is separate). 30,840 hours total. v2
  expansion candidate if the CC-BY subset is filtered. Source:
  https://thegradient.pub/new-datasets-to-democratize-speech-recognition-technology-2/.

- **AudioCaps**: CC BY 4.0 captions, but audio is YouTube-
  sourced (AudioSet) and not redistributable. Rejected (cannot
  ship the audio). Source: https://audiocaps.github.io/,
  https://aclanthology.org/N19-1011/.

- **Clotho**: per-clip CC0/CC-BY audio, but captions under
  Tampere University non-commercial license. Rejected (NC on
  the captions). Source:
  https://zenodo.org/records/3490684,
  https://github.com/audio-captioning/clotho-dataset/blob/master/LICENSE.

- **MusicCaps**: CC BY-SA 4.0. Rejected (SA). Source:
  https://huggingface.co/datasets/google/MusicCaps.

- **ESC-50**: CC BY-NC 3.0. Rejected (NC). Source:
  https://github.com/karolpiczak/ESC-50,
  https://www.innovatiana.com/en/datasets/esc-50-environmental-sound-classification.

- **FSD50K**: per-clip mixed (CC0/CC-BY/CC-BY-NC/Sampling+);
  dataset overall CC-BY 4.0 but commercial use requires
  contacting authors. Rejected (contact step blocks
  distribution). Source: https://zenodo.org/records/4060432.

- **VGGSound**: CC BY 4.0 metadata, but audio is YouTube-
  sourced and not redistributable. Rejected. Source:
  https://github.com/hche11/VGGSound,
  https://www.robots.ox.ac.uk/~vgg/data/vggsound/.

- **WavText5K**: MIT metadata; audio sourced from SoundBible,
  Freesound, BigSoundBank (mostly MIT/CC0, per-clip filter
  needed). Rejected for v1 (per-clip filter step). Source:
  https://github.com/microsoft/WavText5K,
  https://huggingface.co/datasets/CLAPv2/WavText5K.

- **AudioSet**: CC BY 4.0 annotations; audio is YouTube-
  sourced and not redistributable. Same as AudioCaps.
  Rejected. Source: https://research.google.com/audioset/.

**Image + audio + text (multi-modal triples)**

- **AudioSet image-audio pairs**: not a stand-alone public
  dataset; the YouTube-sourced nature blocks redistribution.
  Rejected.
- **MUSIC (multimodal)**: AVSources research dataset, per-
  item license varies. Not license-clean by default.
  Rejected.
- **VGGSound image-audio pairs**: same YouTube issue as
  VGGSound above. Rejected.

### Decision summary

The three accepted sources (CC0 procedural, PixelProse CC BY 4.0,
LibriSpeech CC BY 4.0) cover all three modalities at the right
size, are license-clean for commercial use, and LibriSpeech is
exactly 16 kHz (matches gemma 4 12b unified's audio path
without resampling). No single public dataset meets all four
criteria (5k size, all-three-modalities, uniform commercial
license, native 16 kHz), so the curated combination is the
correct answer. PixelProse's CC12M/CommonPool/RedCaps origin
gives us a 2k image sample that is more concept-balanced than
a random COCO subsample, and LibriSpeech test-clean (2,620
samples) gives us a 1.5k audio sample that is the right size
and the right sample rate.

The risks are concentrated in the runtime side (per-modality
AWQ could over-fit, the dequant switch could regress on some
backends). The smoke gates in G3-MM and G4-MM catch both
before any production shipping.

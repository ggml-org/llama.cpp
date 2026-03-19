# Speaker Encoder (ECAPA-TDNN)

Standalone 1024-dimensional speaker embedding extractor using the ECAPA-TDNN architecture from Qwen3-TTS. Extracts x-vector embeddings that capture a speaker's voice characteristics independently of spoken content.

## Quick Start

```bash
# Extract embedding from a single file
speaker-encoder --model qwen3tts-talker-bf16.gguf --audio speaker.wav

# Compare two speakers
speaker-encoder --model qwen3tts-talker-bf16.gguf --audio speaker_a.wav speaker_b.wav --cosine

# Save embedding to file
speaker-encoder --model qwen3tts-talker-bf16.gguf --audio speaker.wav --output embedding.bin
```

## Use Cases

This tool can be used independently from the TTS pipeline for:

- **Speaker verification**: Compare two audio clips to check if they're the same speaker
- **Speaker diarization**: Cluster audio segments by speaker identity
- **Voice similarity search**: Find the most similar voice in a database
- **Voice cloning input**: Extract the x-vector needed for Qwen3-TTS voice cloning

## Input Requirements

- **Format**: WAV file, 16-bit PCM
- **Sample rate**: 24,000 Hz (other rates will produce a warning)
- **Channels**: Mono or stereo (only first channel is used)
- **Duration**: Any length works; 3-10 seconds gives best results

## Output

A 1024-dimensional float32 vector. Output modes:

| Mode | Flag | Description |
|------|------|-------------|
| Print | (default) | Print all 1024 values to stdout |
| Summary | `--quiet` | Print only RMS and first 5 values |
| Binary | `--output <path>` | Write 4096 bytes (1024 × float32) |
| Cosine | `--cosine` | Pairwise cosine similarity matrix |

## Architecture

ECAPA-TDNN with ~6M parameters:

```
Audio (24kHz) → Mel spectrogram (128 bands)
  → Conv1d (128→512, k=5)
  → 3× SE-Res2Net blocks (dilation 2,3,4)
  → MFA (concatenate blocks → 1536ch)
  → Attentive Statistics Pooling → 3072-dim
  → FC → 1024-dim embedding
```

See `docs/qwen3-tts/architecture.md` for the full architectural description.

## Building

Built automatically as part of llama.cpp when the TTS tools are enabled.

The speaker encoder only depends on GGML (not llama.cpp), so it can also be built standalone if GGML is installed as a package.

## Weights

The speaker encoder weights are stored in the Talker GGUF file under the `spk_enc.*` tensor prefix. No separate model download is needed.

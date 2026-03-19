# Qwen3-TTS Examples

Demo scripts and interactive app for Qwen3-TTS running on llama.cpp.

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Basic TTS | Working | Text-to-speech in 10 languages |
| Voice cloning (x-vector) | Working | Provide `--ref-audio` for speaker embedding |
| Voice cloning (ICL) | Working | Higher quality; needs `--ref-audio`, `--ref-text`, `--ref-codes` |
| Streaming text | Working | Feed text tokens progressively during decode |
| Vocoder | Working | Verified 0.999 correlation with HuggingFace reference |
| 10 languages | Working | english, chinese, german, spanish, french, italian, japanese, korean, portuguese, russian |
| Auto language | Working | Use `--language auto` for automatic detection |

## Quick Start

### 1. Build the binary

```bash
cmake --build build --target llama-qwen3tts --config Release
```

### 2. Convert or download GGUF models

Place models in the repository's `models/` directory:

```bash
# Convert from HuggingFace (requires qwen-tts package)
python tools/tts/convert_qwen3tts.py \
    --input Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3tts-talker-bf16.gguf --type bf16

python tools/tts/convert_qwen3tts_cp.py \
    --input Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3tts-cp-bf16.gguf --type bf16

python tools/tts/convert_qwen3tts_tokenizer.py \
    --input Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3tts-tokenizer-f16.gguf --type f16
```

Expected files:
- `qwen3tts-talker-bf16.gguf` — Talker model with speaker encoder (~1.4 GB)
- `qwen3tts-cp-bf16.gguf` — Code Predictor (~85 MB)
- `qwen3tts-tokenizer-f16.gguf` — WavTokenizer vocoder (~340 MB)

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## CLI Usage (Direct)

```bash
# Basic synthesis
llama-qwen3tts \
    --model-talker models/qwen3tts-talker-bf16.gguf \
    --model-cp models/qwen3tts-cp-bf16.gguf \
    --model-vocoder models/qwen3tts-tokenizer-f16.gguf \
    --text "Hello, world!" \
    --output hello.wav

# With sampling parameters
llama-qwen3tts \
    --model-talker models/qwen3tts-talker-bf16.gguf \
    --model-cp models/qwen3tts-cp-bf16.gguf \
    --model-vocoder models/qwen3tts-tokenizer-f16.gguf \
    --text "Hello, world!" \
    --output hello.wav \
    --temp 0.9 --top-k 50 --seed 42

# Voice cloning (x-vector)
llama-qwen3tts \
    --model-talker models/qwen3tts-talker-bf16.gguf \
    --model-cp models/qwen3tts-cp-bf16.gguf \
    --model-vocoder models/qwen3tts-tokenizer-f16.gguf \
    --ref-audio speaker.wav \
    --text "Cloned speech" \
    --output cloned.wav

# Voice cloning (ICL — higher quality)
python tools/tts/extract_ref_codes.py --ref-audio speaker.wav --output ref_codes.txt
llama-qwen3tts \
    --model-talker models/qwen3tts-talker-bf16.gguf \
    --model-cp models/qwen3tts-cp-bf16.gguf \
    --model-vocoder models/qwen3tts-tokenizer-f16.gguf \
    --ref-audio speaker.wav \
    --ref-text "Exact transcript of the reference audio" \
    --ref-codes ref_codes.txt \
    --text "This will sound like the reference speaker" \
    --output cloned_icl.wav
```

## CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model-talker` | *(required)* | Path to Talker GGUF |
| `--model-cp` | *(required)* | Path to Code Predictor GGUF |
| `--model-vocoder` | *(optional)* | Path to vocoder GGUF (skip = no audio) |
| `--text` | *(required)* | Text to synthesize |
| `--output` | `output.wav` | Output WAV file path |
| `--language` | `english` | Language: english, chinese, german, spanish, french, italian, japanese, korean, portuguese, russian, auto |
| `--ref-audio` | *(none)* | Reference audio WAV for voice cloning |
| `--ref-text` | *(none)* | Reference transcript (for ICL mode) |
| `--ref-codes` | *(none)* | Precomputed codec codes file (for ICL mode) |
| `--temp` | `0.9` | Talker sampling temperature (0 = greedy) |
| `--top-k` | `50` | Talker top-k sampling |
| `--top-p` | `1.0` | Talker nucleus sampling |
| `--rep-penalty` | `1.05` | Repetition penalty |
| `--cp-temp` | `0.9` | Code Predictor temperature |
| `--cp-top-k` | `50` | Code Predictor top-k |
| `--greedy` | *(off)* | Force greedy decoding for both models |
| `--seed` | *(random)* | Random seed for reproducibility |
| `--max-tokens` | `2048` | Maximum output frames (12 frames ~ 1 second) |
| `--streaming-text` | *(off)* | Feed text progressively during decode |
| `--n-gpu-layers` | `0` | Number of GPU layers |
| `--dump-intermediates` | *(none)* | Directory to dump codec codes |

## Scripts

### `basic_tts.py` — Simple text-to-speech

```bash
python basic_tts.py --text "Hello, world!" --language english --output hello.wav
python basic_tts.py --text "Testing greedy mode." --greedy --output greedy.wav
python basic_tts.py --text "Streaming mode test." --streaming-text --output stream.wav
```

### `voice_cloning.py` — Clone a speaker's voice

```bash
# X-vector mode (simpler, uses speaker embedding from reference audio)
python voice_cloning.py \
    --ref-audio speaker.wav \
    --text "Cloned speech" \
    --output cloned.wav

# ICL mode (higher quality — needs reference text + codec codes)
# Step 1: Extract codec codes from reference audio
python ../../tools/tts/extract_ref_codes.py \
    --ref-audio speaker.wav \
    --output ref_codes.txt

# Step 2: Clone with ICL
python voice_cloning.py \
    --ref-audio speaker.wav \
    --ref-text "The exact transcript of the reference audio" \
    --ref-codes ref_codes.txt \
    --text "This sentence in the cloned voice" \
    --output cloned_icl.wav
```

**X-vector vs ICL:**
- **X-vector** extracts a 1024-dim speaker embedding from reference audio using the built-in ECAPA-TDNN encoder. Simple but captures only global speaker characteristics (timbre, pitch).
- **ICL** (in-context learning) also feeds the reference audio's codec tokens and transcript into the model, giving it concrete examples of how the speaker sounds. Produces higher fidelity cloning but requires the `extract_ref_codes.py` preprocessing step, which depends on the HuggingFace `qwen_tts` package.

### `multilingual.py` — Generate speech in multiple languages

```bash
python multilingual.py                              # all 10 languages
python multilingual.py --languages english french    # specific languages
python multilingual.py --output-dir my_output/       # custom output directory
```

### `benchmark.py` — Measure performance

```bash
python benchmark.py                    # basic benchmark
python benchmark.py --runs 3           # average over 3 runs
python benchmark.py --with-vocoder     # include vocoder time
python benchmark.py --max-tokens 100   # more frames per test
```

## Interactive App

### `app.py` — Streamlit web interface

```bash
pip install -r requirements.txt
streamlit run app.py
```

Features:
- **Basic TTS** — type text, select language, synthesize and play audio
- **Voice Cloning** — upload reference WAV, choose x-vector or ICL mode
- **Multilingual** — generate speech in multiple languages side by side
- **Benchmark** — measure prefill tok/s, decode fr/s, RTF across text lengths
- Sidebar controls for temperature, top-k, top-p, repetition penalty, seed
- Audio playback, download, and raw log inspection

## Performance Metrics Explained

```
=== Performance ===
  Prefill:  32 tokens in 2616.7 ms  (12.2 tok/s)
  Decode:   79 frames in 52387.4 ms  (1.51 frames/s)
    Talker:    7933.7 ms total  (100.4 ms/frame)
    CP:        45565.3 ms total  (576.8 ms/frame)
    Head:      2684.6 ms total  (34.0 ms/frame)
  Real-time factor: 0.12x  (6.58s audio in 55.00s)
```

| Metric | What it means |
|--------|---------------|
| Prefill tok/s | Input text processing speed |
| Decode fr/s | Audio frames per second (12 fr/s = real-time at 12 Hz) |
| Talker ms/fr | Time per frame for the 28-layer Talker |
| CP ms/fr | Time per frame for the Code Predictor (5 layers x 15 steps) |
| Head ms/fr | Time per frame for LM head + sampling |
| RTF | Audio duration / wall time (>1.0 = faster than real-time) |

The Code Predictor dominates decode time because it runs 15 autoregressive steps per frame. GPU acceleration (`--n-gpu-layers`) significantly improves throughput.

# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Soprano

Soprano-1.1-80M uses a Qwen3 backbone and a Vocos decoder. Convert both files from the same local [model directory](https://huggingface.co/ekwek/Soprano-1.1-80M), including `decoder.pth`:

```sh
python convert_hf_to_gguf.py path/to/Soprano-1.1-80M --outfile soprano.gguf --outtype f16
python convert_hf_to_gguf.py path/to/Soprano-1.1-80M --mmproj --mmproj-architecture SopranoModel --outfile mmproj-soprano.gguf --outtype f16
llama-tts -m soprano.gguf -mm mmproj-soprano.gguf -p "Hello world!" --temp 0 --output out.wav
```

The explicit mmproj architecture is required because the model config only identifies the Qwen3 text backbone, which does not identify an audio decoder.

This pipeline generates mono audio at 32 kHz with the model's fixed voice. It does not accept `--tts-speaker-file`; `--tts-lang` is unused. The helper adds the `[STOP][TEXT]...[START]` prompt format and accumulates hidden states before reconstructing the waveform. The core `GEN_WAV` call accepts 2 to 512 frames of 512 continuous features in frame-major order; it does not use codes or persistent state.

Supply normalized English text. The tokenizer lowercases text and collapses whitespace, but the Python reference's number/abbreviation expansion, transliteration and sentence splitting are not included. Split long text into separate requests; each prompt must fit within 512 tokens including the three control tokens. Audio is returned after generation finishes; incremental audio output is not supported in this initial implementation.

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

## KaniTTS-2

Available params:
- `--tts-lang` can be `en_us`, `en_nyork`, `en_oakl`, `en_glasg`, `en_bost`, `en_scou`; `en` is an alias for `en_us`
- `--tts-speaker-file` optionally provides reference audio for voice cloning
- `-n` limits generated tokens; four audio tokens represent one frame

Example usage:

```sh
llama-tts -m kani-tts-2.gguf \
    -mm mmproj-kani-tts-2.gguf \
    -p "Hello world" \
    --tts-lang en_us \
    -c 4096 -n 3000 --temp 1 --top-p 0.95 --top-k 0 \
    --repeat-penalty 1.1 --repeat-last-n 4096 \
    --output out.wav
```

**Note for GGUF conversion:**

Download [KaniTTS-2 English](https://huggingface.co/nineninesix/kani-tts-2-en), place the [NeMo Nano Codec archive](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps) in the same directory, and download the [speaker encoder](https://huggingface.co/nineninesix/speaker-emb-tbr) into its `speaker_encoder` subdirectory:

```sh
hf download nineninesix/speaker-emb-tbr --local-dir path/to/kani-tts-2-en/speaker_encoder
python convert_hf_to_gguf.py path/to/kani-tts-2-en --outtype f16 --outfile kani-tts-2.gguf
python convert_hf_to_gguf.py path/to/kani-tts-2-en --mmproj --outtype f16 --outfile mmproj-kani-tts-2.gguf
```

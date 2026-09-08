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

[KaniTTS-2 English](https://huggingface.co/nineninesix/kani-tts-2-en) uses an LFM2 backbone with learned RoPE frequencies and four audio tokens per frame. Its audio decoder is [NeMo Nano Codec 22 kHz / 0.6 kbps / 12.5 fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps).

Download the backbone and the matching codec into separate directories, then convert both:

```sh
hf download nineninesix/kani-tts-2-en --local-dir models/kani-tts-2-en
hf download nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps \
    nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo --local-dir models/nemo-nano-codec

python convert_hf_to_gguf.py models/kani-tts-2-en \
    --outtype f16 --outfile kani-tts-2-f16.gguf
python convert_hf_to_gguf.py models/nemo-nano-codec \
    --mmproj --outtype f16 --outfile mmproj-nemo-nano-codec-f16.gguf

./build/bin/llama-tts -m kani-tts-2-f16.gguf -mm mmproj-nemo-nano-codec-f16.gguf \
    -p "The weather is beautiful today. Let us take a walk in the park, enjoy the sunshine, and listen to the birds singing in the trees." --tts-lang en_us -o output.wav \
    -c 4096 -n 3000 --temp 1 --top-p 0.95 --top-k 0 --min-p 0.05 \
    --repeat-penalty 1.1 --repeat-last-n 4096 --seed 42
```

Only `KaniTTS2ForCausalLM` selects the Kani converter. Generic LFM2 models keep their existing conversion and inference behavior. Use `llama-tts` for Kani generation: ordinary text generation does not assign the required frame positions. Each token advances the cache, while all four tokens in an audio frame share a RoPE position. The learned per-layer frequency scales are preserved in GGUF.

The supported language tags are `en_us`, `en_nyork`, `en_oakl`, `en_glasg`, `en_bost`, and `en_scou`; `en` is an alias for `en_us`. Omitting the tag leaves the prompt untagged, as in the reference implementation. Speaker reference audio / voice cloning is not supported; the optional speaker projection is excluded from the backbone conversion.

`-n` limits generation steps (tokens for Kani), not audio frames. Four audio tokens produce 1764 samples at 22050 Hz (12.5 frames/s). Increase `-c` for longer prompts and output. The helper validates codebook offsets and end-of-speech boundaries; it reports an error if a token limit interrupts a frame. Greedy decoding can repeat audio codes; use the sampling parameters above. Waveform decoding uses overlapping chunks to preserve the causal convolution history on long outputs.

### NeMo decoder API

The same mmproj can be loaded with `mtmd_init_from_file(path, nullptr, params)` for codes-to-audio use. Pass `MTMD_GEN_PROCESS_TYPE_GEN_WAV` to `mtmd_gen_audio_process`, with `codes` laid out as `[frame][group]`: four codes per frame, each in `[0, 4031]`. NeMo's `[group, batch, frame]` tokens must be transposed. Each call accepts 1 to 128 complete frames and returns `n_frames * 1764` mono float samples. Copy the output before the next call and release the context with `mtmd_free`.

Convolution weights are F16, including with `--outtype f32`, to use the existing ggml convolution path. FSQ codebook and activation parameters remain F32. Each API call has zero initial context; the Kani helper supplies 32 preceding frames when splitting a long sequence. The decoder does not accept continuous features, persistent state or `GEN_CODE`. Audio encoding and other Nano Codec variants are not supported.

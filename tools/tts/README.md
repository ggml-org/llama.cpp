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

## NeMo Nano Codec decoder (MTMD API)

The [22 kHz / 0.6 kbps / 12.5 fps variant](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps) is supported as a standalone codes-to-audio decoder. It is not a text-to-speech model and cannot be used directly with `llama-tts -p`.

Place `nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo` in a local directory and convert it:

```sh
python convert_hf_to_gguf.py path/to/nemo-nano-codec --mmproj --outtype f16 --outfile mmproj-nemo-nano-codec.gguf
```

The converter reads the architecture from the archive's `model_config.yaml`; there is no text backbone or tokenizer to convert. Convolution weights are stored as F16 even with `--outtype f32`, to use the existing ggml convolution path. FSQ codebook and activation parameters remain F32.

Load the mmproj with `mtmd_init_from_file(path, nullptr, params)`. Pass `MTMD_GEN_PROCESS_TYPE_GEN_WAV` to `mtmd_gen_audio_process`, with `codes` laid out as `[frame][group]`: four codes per frame, each in `[0, 4031]`. NeMo's `[group, batch, frame]` tokens must be transposed for this interface. Each call accepts 1 to 128 complete frames and returns `n_frames * 1764` mono float samples at 22050 Hz. Copy the output before the next call, and release the context with `mtmd_free`.

This initial implementation decodes a complete sequence with zero initial context. It does not accept continuous features, persistent state or `GEN_CODE`. Independent chunks do not preserve convolution history. Audio encoding, other Nano Codec variants and a text-generation pipeline are not included. NVIDIA describes this particular variant as intended for fine-tuning with a limited set of speakers, rather than general-purpose audio reconstruction.

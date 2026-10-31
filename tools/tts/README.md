# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Note: it's highly recommended to always provide a speaker reference voice (via `--tts-speaker-file`); otherwise, the model's performance may be degraded.

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
- `--tts-voice` selects one of the predefined voices shipped with the weights (e.g. `giovanni`), when the GGUF holds them. The converter reads them from `embeddings/*.safetensors` next to the weights, of both repositories, and logs how closely each one is reproduced (some packs ship voice states computed with other weights, the converter warns above 5% and skips a voice only when it does not belong to the model at all)
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

The weights [without voice cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) ship a set of predefined voices (`embeddings/*.safetensors`) next to each language; the converter stores them in the mmproj, as the voice prompt embeddings, and leaves out the zeroed voice encoder. With such a model `--tts-speaker-file` is rejected and a voice is selected with `--tts-voice`:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Ciao a tutti" \
    --tts-voice giovanni \
    --output out.wav
```

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```

## Soprano

Available params:
- `--tts-speaker-file` is not supported; the model uses a fixed voice
- Note: `lang` is not used. Supply normalized English text

Example usage:

```sh
llama-tts -m soprano.gguf \
    -mm mmproj-soprano.gguf \
    -p "Hello world" \
    --temp 0 \
    --output out.wav
```

**Note for GGUF conversion:**

Both [Soprano-80M](https://huggingface.co/ekwek/Soprano-80M) and [Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M) are supported. Convert both files from the same model directory, including `decoder.pth`:

```sh
python convert_hf_to_gguf.py path/to/Soprano-1.1-80M --outtype f32 --outfile soprano.gguf
python convert_hf_to_gguf.py path/to/Soprano-1.1-80M --mmproj --outtype f16 --outfile mmproj-soprano.gguf
```

## LFM3-Audio

Available params:
- `--tts-lang` can be `ar`, `da`, `de`, `el`, `en`, `en-gb`, `es`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `sv`, `tr`, `zh` (default: `en`), other values are used as the language name
- `--tts-voice` can be `female` (default) or `male`
- Note: `--tts-speaker-file` is not used, the model has no voice cloning
- Note: `--top-k` and `--top-p` are not used, audio frames are sampled with the settings of the model card (LFM3-Audio: temperature 1.0, top-k 128, repetition penalty 1.1 over 4096 frames; LFM3-Omni: temperature 0.4, repetition penalty 1.2 over 64 frames), the `audio_*` keys of `generation_config.json` override them at conversion
- Each frame is 80 ms, generation stops at the end-of-audio frame

Example usage:

```sh
llama-tts -m lfm3-audio.gguf \
    -mm mmproj-lfm3-audio.gguf \
    -p "Hello world" \
    --tts-voice male \
    --output out.wav
```

The mmproj holds the audio encoder (for ASR), the depthformer and the codec decoder, the latter two are exported when the checkpoint has an `audio_tokenizer` directory:

```sh
python convert_hf_to_gguf.py path/to/lfm3-audio --outfile lfm3-audio.gguf
python convert_hf_to_gguf.py path/to/lfm3-audio --mmproj --outfile mmproj-lfm3-audio.gguf
```

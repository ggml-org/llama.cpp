# Text-to-speech (TTS)

llama.cpp supports audio generation via `libmtmd`. The following tools can be used:
- [llama-tts](../../tools/tts/README.md), see its README for common usage and params
- [llama-server](../../tools/server/README.md) via the `/tts` endpoint and the OpenAI-compatible `/v1/audio/speech` API

Supported models:
- [Qwen3-TTS](#qwen3-tts)
- [Pocket TTS](#pocket-tts)
- [Soprano](#soprano)
- [LFM3-Audio](#lfm3-audio)
- [Chatterbox](#chatterbox)
- [Drama 3](#drama-3)
- [ZONOS3](#zonos3)
- [Fish Audio S2 Pro](#fish-audio-s2-pro)
- [YuE2](#yue2) (songs)
- [MiniMax Music 3](#minimax-music-3) (songs)
- [Hibiki](#hibiki) (speech translation, `llama-server` realtime endpoint only)
- [Kokoro](#kokoro)
- [tiny-voca](#tiny-voca)
- [VoxCPM2](#voxcpm2)

The sections below list the params supported by each model, with `llama-tts` examples. In `llama-server`, the same options are set with `--tts-lang` and `--tts-voice`, or per request (see the [`/tts` endpoint](../../tools/server/README.md#post-tts-generate-speech-audio-from-text)).

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)
- `--tts-speaker-text` is the transcript of the speaker file. With it, the speech continues from the reference codes (the "ICL" mode of the reference), which clones the voice more closely. It needs a clip of 30 s or less and an mmproj converted with the speech tokenizer encoder, otherwise it falls back to the speaker embedding only

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

Voice cloning with the transcript of the reference:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-speaker-file speaker.wav \
    --tts-speaker-text "The words spoken in speaker.wav." \
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

The weights [without voice cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) ship a set of predefined voices (`embeddings/*.safetensors`) next to each language; the converter stores them in the mmproj, as the voice prompt embeddings, and leaves out the zeroed voice encoder. With such a model `--tts-speaker-file` is rejected and a voice is selected with `--tts-voice` (the first one when it is not given):

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

## Chatterbox

Two variants are supported: [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (English) and [Chatterbox Multilingual V3](https://huggingface.co/ResembleAI/chatterbox) (23 languages).

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). Optional, the predefined voice is used without it
- `--tts-voice` can be `default`, the voice shipped with the model (`conds.pt`)
- `--tts-lang` sets the language of the multilingual variant, for example `fr` (default: none). It is not used by Turbo

Example usage:

```sh
llama-tts -m chatterbox-mtl.gguf \
    -mm mmproj-chatterbox-mtl.gguf \
    -p "Bonjour tout le monde" \
    --tts-lang fr \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

Note: the multilingual variant runs classifier-free guidance with a second sequence. `llama-tts` sets this up by itself, but `llama-server` does not support it yet.

**Note for GGUF conversion:**

The upstream repositories ship raw checkpoints without `config.json`. Download the files below into a directory, then convert it as usual (the variant is detected from the talker file):
- Turbo: `t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`, `ve.safetensors`, `conds.pt` and the GPT-2 tokenizer files (`vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json`)
- Multilingual V3: `t3_mtl23ls_v3.safetensors`, `s3gen_v3.safetensors`, `ve.safetensors`, `conds.pt`, `mtl_tokenizer.json`

```sh
python convert_hf_to_gguf.py path/to/chatterbox --outfile chatterbox.gguf
python convert_hf_to_gguf.py path/to/chatterbox --mmproj --outfile mmproj-chatterbox.gguf
```

## Drama 3

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3): a Qwen3.5-MoE backbone (35B, 3B active) samples each frame's main token, a codebook tower samples the 9 other codes, and the Fish DAC decodes them to 44.1 kHz audio.

Available params:
- Speaker tags `<|speaker:0|>` to `<|speaker:15|>` in the text give each character its own voice, a text without a tag starts with `<|speaker:0|>`
- The language is the one of the text, `--tts-lang` is not used
- Voice cloning: `--tts-speaker-file` with the reference clip and `--tts-speaker-text` with its transcript (a transcript without a speaker tag is speaker 0). The clip is encoded by the codec encoder half of `codec.pth` and its codes are put in the prompt as one reference, as the reference does; the clip is truncated to the first 30 s. It needs an mmproj converted with the codec encoder
- Note: `--top-k` and `--top-p` are not used, frames are sampled with the settings of the reference (main token: temperature 1.0, top-p 0.9, repetition-aware redraw over the last 16 frames; other codes: temperature 1.0, top-k 30, top-p 0.9)
- Each frame is 2048 samples (about 46 ms), generation stops at the end-of-speech token
- The codebook tower keeps the whole sequence in its KV cache, the prompt and the generated frames are limited to 4096 positions

Example usage:

```sh
llama-tts -m drama3.gguf \
    -mm mmproj-drama3.gguf \
    -p "<|speaker:0|> Did you remember to buy the bread? <|speaker:1|> Of course I did, it is on the kitchen table." \
    -n 1000 \
    --output out.wav
```

**Note for GGUF conversion:**

The routed experts of the checkpoint are FP8, they are dequantized at conversion. The mmproj holds the codebook tower and `codec.pth` (the encoder half for voice cloning, the decoder half for the audio), it is best kept in BF16; the codec encoder is always stored in F16/F32:

```sh
python convert_hf_to_gguf.py path/to/drama-3 --outtype q8_0 --outfile drama3.gguf
python convert_hf_to_gguf.py path/to/drama-3 --mmproj --outtype bf16 --outfile mmproj-drama3.gguf
```

## ZONOS3

[Zyphra ZONOS3 Mini](https://huggingface.co/Zyphra/ZONOS3-Mini): a hybrid Gated DeltaNet / attention MoE backbone (800M, 100M active) predicts the 10 codebooks of each frame with an inter-codebook delay, and the ZAC codec decodes them to 44.1 kHz audio.

Available params:
- `--tts-voice` picks one of the predefined voices of the release (`Savannah_en-American_F_1` by default); `llama-server` lists them in `GET /props`. A voice is the prompt prefix of its reference codes, the default speaker mode of the reference
- The language is the one of the text; `--tts-lang` (`lang` in `llama-server`) only selects the text normalization
- Numbers are spelled out by the `tts` chat template of the GGUF: cardinals, decimals, money, percents, clock times, common ordinal forms (`1st`, `1er`, `1º`), English and German years. A 4-digit number is read as a year only after a word like `in`, `since`, a month name, or `im Jahr`, `seit` in German (`in 1990` -> "in nineteen ninety", `1234 people` -> "one thousand two hundred thirty-four people"). Languages: `en`, `fr`, `de`, `es`, `it`, `pt`, `ja` and `zh` (or `cmn`), the region is ignored (`fr_fr`, `pt_br`); English when the language is empty, other languages are left as written. Set the language to the one of the text: the separators differ, e.g. `1.234` is a thousand in German
- Note: the reference uses NeMo text normalization, which also covers units, dates, abbreviations and Korean; supply normalized text for them
- `--tts-speaker-file` clones the voice of a clip (the first 60 s are used) with the speaker encoder of the release, the "embedding" speaker mode of the reference
- `llama-server` also takes `speed` (relative to 15 UTF-8 bytes per second) and `loudness` (LUFS), see the `/tts` endpoint; they pick the speaking-rate and loudness condition buckets of the checkpoint, the defaults leave them unknown
- Note: `loudness` has a strong effect with a cloned voice; a predefined voice mostly keeps the loudness of its reference clip
- Note: `--top-k` and `--top-p` are not used, frames are sampled with the settings of the reference: temperature 0.75 (`--temp` changes it), top-k 80, top-p 0.69, repetition penalty 2.4 over the last 96 frames of codebooks 0 to 5
- Each frame is 2940 samples (about 67 ms), generation stops at the end-of-audio code

Example usage:

```sh
llama-tts -m zonos3-mini.gguf \
    -mm mmproj-zonos3-mini.gguf \
    -p "The old lighthouse keeper climbed the spiral stairs every evening." \
    --tts-voice Imogen_en-British_F_1 \
    --output out.wav
```

**Note for GGUF conversion:**

Convert the root of the release repository, not the `lm/` subdirectory. The mmproj holds the codebook heads, the predefined voices, the speaker encoder and the `keep2_plain` codec decoder, it is best kept in F32:

```sh
python convert_hf_to_gguf.py path/to/ZONOS3-Mini --outtype q8_0 --outfile zonos3-mini.gguf
python convert_hf_to_gguf.py path/to/ZONOS3-Mini --mmproj --outtype f32 --outfile mmproj-zonos3-mini.gguf
```

## Fish Audio S2 Pro

[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro): a Qwen3 backbone (4B) samples each frame's semantic token, a 4-layer fast AR predicts the 9 other codes, and the Fish DAC (same as Drama 3) decodes them to 44.1 kHz audio.

Available params:
- Speaker tags `<|speaker:0|>`, `<|speaker:1|>`, ... in the text give each character its own voice, a text without a tag starts with `<|speaker:0|>`
- Inline instructions in brackets control the delivery, for example `[laughing]`, `[whisper]` or `[pause]`
- The language is the one of the text, `--tts-lang` is not used
- Voice cloning: `--tts-speaker-file` with the reference clip, optionally `--tts-speaker-text` with its transcript (it is put in the prompt as the reference does). The clip is encoded by the codec encoder half of `codec.pth`, it is truncated to the first 30 s
- Note: `--temp`, `--top-k` and `--top-p` are not used (only `--seed`), frames are sampled with the settings of the reference (semantic token: repetition penalty 1.1 over the last 16 frames, top-k 30, top-p 0.8, temperature 0.8, or top-p 0.9 and temperature 1.0 when the last 4 frames repeat a token; other codes: greedy)
- Each frame is 2048 samples (about 46 ms), generation stops at the end-of-speech token

Example usage:

```sh
llama-tts -m s2-pro.gguf \
    -mm mmproj-s2-pro.gguf \
    -p "<|speaker:0|>Did you remember to buy the bread? <|speaker:1|>Of course I did [laughing] it is on the kitchen table." \
    -n 1000 \
    --output out.wav
```

Voice cloning (`--tts-speaker-text` can be left out):

```sh
llama-tts -m s2-pro.gguf \
    -mm mmproj-s2-pro.gguf \
    --tts-speaker-file reference.wav \
    --tts-speaker-text "The words spoken in reference.wav." \
    -p "Hello, this is a cloned voice." \
    --output out.wav
```

**Note for GGUF conversion:**

The mmproj holds the fast AR and `codec.pth` (the encoder half for voice cloning, the decoder half for the audio), it is best kept in BF16. The codec encoder is always stored in F16/F32, small errors in it change the codes picked by its VQ:

```sh
python convert_hf_to_gguf.py path/to/s2-pro --outtype q8_0 --outfile s2-pro.gguf
python convert_hf_to_gguf.py path/to/s2-pro --mmproj --outtype bf16 --outfile mmproj-s2-pro.gguf
```

## YuE2

[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) generates songs (vocals and accompaniment) from a style prompt and lyrics, as 48 kHz stereo. The backbone (a Qwen3 text model) writes an ABC score, then one semantic code per 40 ms frame; the NAR half of the checkpoint turns the codes into VAE latents by flow matching, attending to the keys and values of the backbone, and the [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) decoder makes the audio.

The prompt is the text of the reference request:

```
[Tags]
City Pop, upbeat, danceable, groovy bass, electric guitar, synth
[Lyrics]
[Verse]
...
```

Available params:
- `--tts-voice` picks the planning mode of the reference (`cot`): `full` (default, a chord-annotated score), `melody` (a melody-only score) or `off` (no score)
- Cover mode: append the score to the prompt as `\n[ABC]\n<score>`, with `--tts-voice melody` or `full`; the model then goes straight to the codes
- The generation stops at the limits of the reference (4096 score tokens, then 9000 codes, 6 minutes); `-n` sets a lower limit on the score tokens and the codes together
- The context must keep the default size (24576 positions, the one of the reference): a smaller `-c` is rejected when the model is loaded
- Note: `--top-k` and `--top-p` are not used, both phases are sampled with the settings of the reference (`yue2_generation_config.json`); `--temp` changes the temperature of the codes
- Note: the backbone context must use an F16 KV cache (default): the NAR pass loads its keys and values
- The codes are sampled with classifier-free guidance like the reference: scale 1.01 with `--tts-voice off`, off (1.0) otherwise. `llama-server` takes `cfg_scale` to change it, the reference suggests 1.2 for a stronger text guidance. The uncond prompt runs as a second sequence, so the context must also hold it (the instruction, the score and the codes once more)
- The NAR pass solves the flow-matching ODE in 32 midpoint steps like the reference; `llama-server` takes `flow_steps` to change it. The NAR time is about proportional to it: 16 steps are about 35% faster for the NAR and the VAE together, with a small deviation from the reference (about 0.2 dB of spectral difference); it grows with fewer steps (about 0.4 dB at 8, 0.8 dB at 4)
- The NAR pass runs on chunks of codes like the reference: one chunk for a song that fits in 24576 positions, so the audio arrives after the last code. When streaming, each 30 s of codes is a chunk, processed during the generation, so the audio arrives in 30 s steps; the chunks are independent (each sees the prompt, the score and its own codes), which may be audible at the boundaries

Example usage:

```sh
llama-tts -m yue2-3b.gguf \
    -mm mmproj-yue2-3b.gguf \
    -f song.txt \
    --output song.wav
```

**Note for GGUF conversion:**

The VAE is a separate repository, download it into the `vae` subdirectory of the model:

```sh
hf download m-a-p/YuE2-3B --local-dir YuE2-3B
hf download m-a-p/YuE2-Vae --local-dir YuE2-3B/vae
python convert_hf_to_gguf.py YuE2-3B --outfile yue2-3b.gguf
python convert_hf_to_gguf.py YuE2-3B --mmproj --outfile mmproj-yue2-3b.gguf
```

The mmproj holds the NAR transformer layers (loaded with libllama as a Qwen3 text model, in the `--outtype` type), and the latent projections and the VAE decoder, always in F32.

## MiniMax Music 3

[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) generates songs (vocals and accompaniment) from a music description and lyrics, as 44.1 kHz stereo. For each 40 ms frame, the backbone (a Qwen3 text model) samples the semantic code and the RVQ depth decoder the 7 other codes; the hidden states of both go to a flow-matching DiT, then a DAC decoder makes the audio. Both use the settings of the reference (the diffusers pipeline).

The prompt has the same format as YuE2, the description goes under `[Tags]`:

```
[Tags]
Genre: acoustic pop. BPM: 96. Key: C major. Vocals: soft female lead. Arrangement: fingerpicked guitar and soft piano.
[Lyrics]
[verse]
Morning light filtering through the pine
[chorus]
Softly the world begins to breathe
```

Available params:
- The description and the lyrics are cleaned up like the reference: markdown is removed, a line that starts with structure tags (e.g. `[Verse]`) keeps only the tags
- The generation stops at the end-of-audio token or at the limit of the reference (9000 frames, 6 minutes); `-n` sets a lower limit in frames (25 per second)
- `--temp` changes the temperature of the codes (default 1.0); `--top-k` and `--top-p` are not used, the codes are sampled with top-k 50 like the reference
- The codes are sampled with classifier-free guidance, scale 1.5 like the reference; `llama-server` takes `cfg_scale` to change it (1.0 disables it). The uncond prompt runs as a second sequence that shares the context, so a song needs about 2 x (prompt tokens + frames) positions: the default context (10240) holds about 3 minutes with a short description, use a larger `-c` (e.g. 20480) for longer songs
- The flow matching runs 30 Euler steps like the reference; `llama-server` takes `flow_steps` to change it
- The audio is made in windows of 200 frames (8 s) every 100 frames, like the reference: a window is processed during the generation once the frame after it exists, so with streaming the audio arrives in steps of about 4 s

Example usage:

```sh
llama-tts -m minimax-music3.gguf \
    -mm mmproj-minimax-music3.gguf \
    -f song.txt \
    --output song.wav
```

**Note for GGUF conversion:**

The conversion reads the diffusers layout of the repository, the `qwen_7B` directory and the `.pth` files are not needed:

```sh
hf download MiniMaxAI/MiniMax-Music3 --local-dir MiniMax-Music3 --exclude "qwen_7B/*" --exclude "*.pth"
python convert_hf_to_gguf.py MiniMax-Music3 --outfile minimax-music3.gguf
python convert_hf_to_gguf.py MiniMax-Music3 --mmproj --outfile mmproj-minimax-music3.gguf
```

The mmproj holds the depth decoder and the DiT layers (in the `--outtype` type), and the condition encoder, the small DiT projections and the DAC decoder, always in F32.

## Hibiki

Hibiki (Kyutai) translates French speech to English speech in real time. It is a full duplex model: it only runs through the [realtime endpoint](../../tools/server/README.md#full-duplex) of `llama-server`, the audio goes in and comes out every 80 ms.

The checkpoint is not in the HF layout, the converter reads its `config.json` and the LM and mimi files it names (Hibiki-1B `kyutai/hibiki-1b-pytorch-bf16`, Hibiki-2B `kyutai/hibiki-2b-pytorch-bf16`):

```sh
python convert_hf_to_gguf.py path/to/hibiki-1b-pytorch-bf16 --outfile hibiki-1b.gguf
python convert_hf_to_gguf.py path/to/hibiki-1b-pytorch-bf16 --mmproj --outfile mmproj-hibiki-1b.gguf

llama-server -m hibiki-1b.gguf --mmproj mmproj-hibiki-1b.gguf -c 1500
```

- The voice of the output is conditioned like the reference (`very_good`), classifier-free guidance is not supported yet
- The audio codes are sampled with the settings of the checkpoint (temperature 0.8, top-k 250), so is the text (temperature 0.8, top-k 50) unless the server sampling options override it

## Kokoro

Available params:
- `--tts-voice` selects one of the voices shipped with the weights (default: `af_heart`); `llama-tts` lists them when the name is unknown. The first letter of a voice is its language: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese, `j` Japanese, `z` Mandarin
- `--tts-lang` can be `en-us` or `en-gb`; when it is not given, the voices named `b*` speak British English and those named `a*` American English
- Note: `--tts-speaker-file` is not supported, the model has no voice cloning
- Note: sampling params are not used, the model is not autoregressive. `--seed` only changes the noise of the vocoder source
- `speed` (`llama-server` only) changes the speaking rate, e.g. `1.5`; `loudness` is not supported

Example usage:

```sh
llama-tts -m kokoro.gguf \
    -mm mmproj-kokoro.gguf \
    -p "Hello world" \
    --tts-voice am_michael \
    --output out.wav
```

The model speaks phonemes. Only English has a text to phonemes step (the model's `tts` chat template, with the pronunciation dictionary stored in the model):
- the pronunciation of a word missing from the dictionary (mostly rare names) is predicted from its spelling, by a model that the converter learns from the dictionary. The reference uses espeak-ng for these words; the prediction is about as accurate, but it is not the same, so such a word can sound different from the reference
- a text longer than 510 phonemes is split at sentence ends and spoken segment by segment

**Other languages, or your own phonemes:**

The prompt of a voice of another language must already be phonemes, in the symbols of the model (see `vocab` in its `config.json`). Run the text to phonemes step in your own code, for example with [espeak-ng](https://github.com/espeak-ng/espeak-ng), which the reference uses for Spanish, French, Hindi, Italian and Brazilian Portuguese:

```sh
PHONEMES=$(echo "El gobierno anunció nuevas medidas." | espeak-ng -q --ipa -v es --stdin | tr '\n' ' ')
llama-tts -m kokoro.gguf \
    -mm mmproj-kokoro.gguf \
    -p "$PHONEMES" \
    --tts-voice ef_dora \
    --output out.wav
```

Notes:
- espeak-ng drops the punctuation, which the model uses for pauses: phonemize the text between the marks and put them back for better prosody
- the reference also replaces a few sounds by single symbols (e.g. `tʃ` by `ʧ`), see `espeak.py` of [misaki](https://github.com/hexgrad/misaki)
- with `llama-server`, send `"raw_prompt": true` to pass phonemes with any voice, including the English ones

**Note for GGUF conversion:**

Download `config.json`, `kokoro-v1_0.pth` and the `voices/*.pt` files you want from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), every voice in the directory is stored in the mmproj:

```sh
python convert_hf_to_gguf.py path/to/Kokoro-82M --outfile kokoro.gguf
python convert_hf_to_gguf.py path/to/Kokoro-82M --mmproj --outfile mmproj-kokoro.gguf
```

The pronunciation dictionary and the part-of-speech tagger of the `tts` template come from the `misaki` and `nltk` python packages. The converter also learns the spelling-to-sound model for unknown words from that dictionary, which adds about 15 seconds:

```sh
pip install misaki nltk
python -m nltk.downloader averaged_perceptron_tagger_eng
```

Without them the converter writes no `tts` template, and the prompt must be a phoneme string (as produced by misaki).

## tiny-voca

[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca): a 4.5M-parameter English TTS model. A phoneme encoder (the text model), then a duration predictor, two frame transformers and a source-filter synthesizer (in the mmproj).

Available params:
- `--tts-voice` can be `M2` (default) or `F2`
- Note: `--tts-speaker-file` is not supported, the model has no voice cloning
- Note: sampling params are not used, the model is not autoregressive. `--seed` only changes the noise of the synthesizer
- `speed` (`llama-server` only) changes the speaking rate, from `0.25` to `4`; `loudness` is not supported

Example usage:

```sh
llama-tts -m tiny-voca.gguf \
    -mm mmproj-tiny-voca.gguf \
    -p "Hello! The meeting moved to 3:30 pm." \
    --tts-voice F2 \
    --output out.wav
```

The model speaks phonemes. The text to phonemes step is the model's `tts` chat template, a port of [dizione](https://github.com/n4ze3m/dizione) (the reference one), with its pronunciation dictionary and tables stored in the model:
- the choice of a pronunciation for each word: exceptions and names, the dictionary, function words and their sentence stress, words spelled alike and read differently ("I read it yesterday", "to read"), hyphenated words, capitals read letter by letter, mixed case ("iPhone"), stretched words ("soooo"), accents, chat spellings ("lol", "btw")
- a word missing from the dictionary is built from a known word when it can be (endings, prefixes, compounds), else it goes through the unknown-word model of dizione (stored in the mmproj) and its conversion rules
- a pronunciation can be given in the text, in CMUdict notation or as phonemes: `Say {HH AH0 L OW1} to Samir{s ə m ˈɪɹ}`
- the text rules are ported too: numbers, money, times, dates, telephone numbers, versions, codes, units, fractions, ranges, math signs and comparisons, web addresses and file names, abbreviations, chat spellings

With `llama-server`, send `"raw_prompt": true` to pass phonemes directly, in the format printed by dizione: symbols separated by spaces, `<sp>` between words, a stress mark (`ˈ` or `ˌ`) before a stressed vowel, e.g. `h ə l ˈoʊ <sp> w ˈɜː l d .`

Each sentence is spoken as one utterance. The template splits the text as the reference does (before a capital or a figure, not after a title or an initial), ends each sentence with `<eos>`, and adds a full stop when the text ends without a mark. Phonemes without `<eos>` are cut after each `.`, `?`, `!` or `…`, and a sentence without a final mark gets a full stop.

**Note for GGUF conversion:**

Convert the root of the repository (it needs `config.json`, `tiny-voca.pt` and `phonemizer/lexicon.tsv`). The other tables of the `tts` template come from the `dizione` python package; without it the converter writes no `tts` template, and the prompt must be phonemes:

```sh
pip install git+https://github.com/n4ze3m/dizione
```

```sh
python convert_hf_to_gguf.py path/to/tiny-voca --outfile tiny-voca.gguf
python convert_hf_to_gguf.py path/to/tiny-voca --mmproj --outfile mmproj-tiny-voca.gguf
```

## VoxCPM2

[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2): a MiniCPM4 backbone (2B) makes one hidden state per frame, without an audio token. The mmproj turns it into a patch of 4 continuous latents: FSQ, an 8-layer residual LM (run with libllama, one context per sequence), then the LocDiT flow matching. The causal AudioVAE decodes the latents to 48 kHz audio.

Available params:
- The language is the one of the text, `--tts-lang` is not used. A voice description in parentheses at the start of the text designs the voice, for example `(A young woman, gentle and warm)Hello!`
- Voice cloning: `--tts-speaker-file` with the reference clip (resampled to 16 kHz, truncated to the first 30 s). `--tts-speaker-text` is not used, the transcript of the clip is not needed
- `--temp` scales the noise of the flow matching (default 1.0), `--top-k` and `--top-p` are not used
- The flow matching uses 10 Euler steps and classifier-free guidance with scale 2.0, like the reference. `llama-server` takes `flow_steps` (2 to 32) and `cfg_scale` to change them
- Each frame is 4 latents, 7680 samples (160 ms). Generation stops on the stop head of the model, at least 4 frames, at most 6 frames per text token + 10 (the limit of the reference)
- Not supported yet: the continuation mode of the reference (prompt audio with its transcript)
- Not supported yet: the bad case retry of the reference (`retry_badcase`, a new seed when the audio reaches 6 frames per text token). Short prompts may then run long and repeat some words, try another seed

Example usage:

```sh
llama-tts -m VoxCPM2.gguf \
    -mm mmproj-VoxCPM2.gguf \
    -p "(A calm male narrator)The quick brown fox jumps over the lazy dog." \
    --output out.wav
```

Voice cloning:

```sh
llama-tts -m VoxCPM2.gguf \
    -mm mmproj-VoxCPM2.gguf \
    --tts-speaker-file reference.wav \
    -p "Hello, this is a cloned voice." \
    --output out.wav
```

**Note for GGUF conversion:**

The mmproj holds LocEnc, LocDiT and the residual LM layers (loaded with libllama as a Granite text model with RoPE off) in the `--outtype` type, and the AudioVAE (`audiovae.pth`, encoder for voice cloning, decoder for the audio) always in F32:

```sh
hf download openbmb/VoxCPM2 --local-dir VoxCPM2
python convert_hf_to_gguf.py VoxCPM2 --outfile VoxCPM2.gguf
python convert_hf_to_gguf.py VoxCPM2 --mmproj --outfile mmproj-VoxCPM2.gguf
```

---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
tags:
- liquid
- lfm2.5
- edge
- llama.cpp
- audio
- speech
- gguf
base_model:
- LiquidAI/LFM2.5-Audio-1.5B
widget:
  - text: "Demo"
    output:
      url: demo.mp4
---

<div align="center">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png"
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> •
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a>
  </div>
</div>

# LFM2.5-Audio-1.5B

Find more details in the original model card: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B

## Runners

`runners` folder contains runners for various architectures including

- llama-liquid-audio-cli
- llama-liquid-audio-server

## Convert GGUFs

```bash
export CKPT=/path/to/LFM2.5-Audio-1.5B
export MODEL=LFM2.5-Audio-1.5B
# backbone
python convert_hf_to_gguf.py $CKPT --outfile $CKPT/${MODEL}-F16.gguf --outtype f16
./llama-quantize $CKPT/${MODEL}-F16.gguf $CKPT/${MODEL}-Q8_0.gguf Q8_0
./llama-quantize $CKPT/${MODEL}-F16.gguf $CKPT/${MODEL}-Q4_0.gguf Q4_0
# mmproj
python convert_hf_to_gguf.py $CKPT --mmproj --outfile $CKPT/mmproj-${MODEL}-F16.gguf --outtype f16
./llama-quantize $CKPT/mmproj-${MODEL}-F16.gguf $CKPT/mmproj-${MODEL}-Q8_0.gguf Q8_0
./llama-quantize $CKPT/mmproj-${MODEL}-F16.gguf $CKPT/mmproj-${MODEL}-Q4_0.gguf Q4_0
# vocoder
python tools/liquid-audio/convert_vocoder_to_gguf.py $CKPT --outfile $CKPT/vocoder-${MODEL}-F16.gguf --outtype f16
python tools/liquid-audio/convert_vocoder_to_gguf.py $CKPT --outfile $CKPT/vocoder-${MODEL}-Q8_0.gguf --outtype q8_0
python tools/liquid-audio/convert_vocoder_to_gguf.py $CKPT --outfile $CKPT/vocoder-${MODEL}-Q4_0.gguf --outtype q4_0
# tokenizer
python convert_hf_to_gguf.py $CKPT/audio_detokenizer --outfile $CKPT/tokenizer-${MODEL}-F16.gguf --outtype f16
./llama-quantize $CKPT/tokenizer-${MODEL}-F16.gguf $CKPT/tokenizer-${MODEL}-Q8_0.gguf Q8_0
./llama-quantize $CKPT/tokenizer-${MODEL}-F16.gguf $CKPT/tokenizer-${MODEL}-Q4_0.gguf Q4_0
```

# 🏃 How to run LFM2.5

## CLI

Set env variables.
```
export CKPT=/path/to/LFM2.5-Audio-1.5B-GGUF
export INPUT_WAV=/path/to/input.wav
export OUTPUT_WAV=/path/to/output.wav
```

### ASR (audio -> text)

```bash
./llama-liquid-audio-cli -m $CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf -mm $CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf -mv $CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf -sys "Perform ASR." --audio $INPUT_WAV
```

### TTS (text -> audio)

```bash
./llama-liquid-audio-cli -m $CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf -mm $CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf -mv $CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf -sys "Perform TTS." -p "Hi, how are you?" --output $OUTPUT_WAV
```

### Interleaved (audio/text -> audio + text)

```bash
./llama-liquid-audio-cli -m $CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf -mm $CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf -mv $CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf -sys "Respond with interleaved text and audio." --audio $INPUT_WAV --output $OUTPUT_WAV
```


## Server

Start server
```
export CKPT=/path/to/LFM2.5-Audio-1.5B-GGUF
./llama-liquid-audio-server -m $CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf -mm $CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf -mv $CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf --tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf
```

Use `liquid_audio_chat.py` script to communicate with the server.

```bash
uv run liquid_audio_chat.py
```

# Demo

<Gallery />

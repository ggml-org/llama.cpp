# mediagen: image, video and audio generation

`libmediagen` runs latent diffusion media models with ggml. The first supported architecture is
[LTX-2](https://github.com/Lightricks/LTX-2) (2.3), a 22B audio-video diffusion transformer:

- the transformer is loaded from a GGUF (e.g. the quantizations in
  [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)),
- the text conditioning is computed from *all* hidden states of a Gemma 3 12B text encoder loaded
  with libllama, projected and refined by the model's text connectors,
- the latents are decoded by the LTX causal video VAE and, for the audio track, by the audio VAE,
  a BigVGAN style vocoder and a bandwidth extension stage (48 kHz stereo).

Everything runs on the ggml backends (tested on Metal, CUDA and Vulkan).

## Quick start

```sh
# image: everything (transformer, VAEs, text projection, Gemma 3 text encoder) is fetched from Hugging Face
llama-mediagen -hf unsloth/LTX-2.3-GGUF -p "a fluffy orange cat sitting on a wooden table" -o cat.png

# video with audio (needs ffmpeg in PATH for the mp4, otherwise frames are written as png)
llama-mediagen -hf unsloth/LTX-2.3-GGUF -p "ocean waves crashing on a rocky shore at sunset, seagulls calling" \
    -W 768 -H 512 --frames 49 --fps 24 -o waves.mp4

# local files
llama-mediagen -m ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf \
    --vae ltx-2.3-22b-distilled_video_vae.safetensors \
    --audio-vae ltx-2.3-22b-distilled_audio_vae.safetensors \
    --text-proj ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
    --text-encoder gemma-3-12b-it-Q4_K_M.gguf \
    -p "a red double-decker bus driving through rainy London streets at night" -o bus.png
```

Options: `-W/--width`, `-H/--height` (multiples of 32), `--frames` (8k+1, 1 = image), `--fps`, `--steps`
(default: the model schedule, 8 steps for the distilled checkpoints), `--cfg-scale` and `--negative-prompt`
(the distilled checkpoints run without guidance), `-s/--seed`, `--no-enhance-prompt`.

`-ngl` only applies to the text encoder; the diffusion model itself always runs on the GPU when there
is one. On machines with tight GPU memory, `-ngl 0` keeps Gemma on the CPU.

### Prompt enhancement

LTX-2 is trained on long, dense captions. Short prompts such as "Draw a cat" are expanded by the text
encoder itself with the system prompt of the reference pipelines before encoding (the same way
`dall-e-3` rewrites prompts); the expanded caption is returned as `revised_prompt`. Disable with
`--no-enhance-prompt` / `"enhance_prompt": false`.

### `-hf` resolution

A repository whose file list contains VAE or text projection sidecars is treated as a diffusion model:

- the transformer defaults to the `distilled` variant at the usual quant preference (`Q4_K_M`, `Q8_0`),
  a tag selects another one: `-hf unsloth/LTX-2.3-GGUF:dev-Q8_0`,
- the sidecars whose names share the longest prefix with the transformer are downloaded
  (`vae/*_video_vae.safetensors`, `vae/*_audio_vae.safetensors`, `text_encoders/*_embeddings_connectors.safetensors`),
- the text encoder the model family was trained with is fetched from its own repository
  (`ggml-org/gemma-3-12b-it-GGUF:Q4_K_M` for LTX-2.x), `--text-encoder-hf` overrides it.

`--no-diffusion-auto` disables all of this.

## Server

`llama-server -hf unsloth/LTX-2.3-GGUF` (or `-m ... --vae ... --text-encoder ...`) serves the model
through OpenAI compatible endpoints instead of the text endpoints:

| Endpoint | Notes |
|---|---|
| `POST /v1/images/generations` | `prompt`, `size` (`WxH`), `seed`, `steps`, `cfg_scale`, `negative_prompt`, `enhance_prompt`; `response_format` is `b64_json`. With `"stream": true` the reply is a stream of server-sent events: `image_generation.progress` (`step`, `total`) then `image_generation.completed` (`b64_json`, `revised_prompt`). |
| `POST /v1/videos` | `prompt`, `size`, `seconds` or `frames`, `fps`, `audio`, plus the image options. Runs synchronously and returns a completed video job object with `content_url`. |
| `GET /v1/videos/{id}` | the job object |
| `GET /v1/videos/{id}/content` | the mp4 (h264 + aac, needs ffmpeg in PATH; without it the job object carries the frames as png and the audio as wav in base64) |
| `POST /v1/audio/speech` | text to audio: `input`, `seconds`; returns a wav |
| `GET /v1/models`, `GET /props` | report `image`, `video` and `audio` capabilities |

```sh
curl localhost:8080/v1/images/generations -d '{"prompt":"Draw a cat","size":"512x512"}' \
    | jq -r '.data[0].b64_json' | base64 -d > cat.png
```

## Memory

With the Q4_K_M transformer (13.5 GiB), the text projection (2.2 GiB), both VAEs (1.1 GiB) and
Gemma 3 12B Q4_K_M (7.3 GiB) about 25 GiB of weights are resident. A 768x512 image needs a few hundred
MiB on top; a 33 frame 768x512 video about 4 GiB for the VAE decode. Machines with 32 to 36 GiB of unified
memory should keep the text encoder on the CPU (`-ngl 0`) for video.

## Validation

The video VAE, the text connectors and the audio decoder were checked against the reference PyTorch
implementation with dumped intermediates (`MEDIAGEN_DUMP=<prefix>` writes them): video VAE 1e-3 max
abs diff, connectors 0.2% mean rel diff (f16 attention), mel spectrogram 7e-5, 48 kHz audio 1.4% mean
rel diff.

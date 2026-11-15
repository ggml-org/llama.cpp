# M5-TTS on Snapdragon Android

The benchmark uses the official M5-TTS Android tokenizer, embeddings, depth graph,
and cached codec. It runs either ONNX CPU backbone or a llama.cpp backbone in
a separate process on the same phone. The HTP v73 experiment uses HVX
(`GGML_HEXAGON_NHMX=0`). See the [port notes](../../docs/backend/snapdragon/m5-tts-port-notes.md)
for compatibility and numerical results.

1. Check out M5-TTS at `4e7cc91545ad5ecd368d7726129cefb41fc6e1d7` and apply
   `android-stream.patch` in that checkout. Use the official M5-TTS-50M-ONNX bundle.
2. Run `prepare_android_voice.py M5_SOURCE ONNX_DIR REFERENCE_WAV TRANSCRIPT OUTPUT_DIR`
   with Python 3.11 and the upstream ONNX Python dependencies. For the upstream
   `assets/aslp_default_voice.wav`, the transcript is
   `猪携带该疾病，然后通过蚊子传染给人类。`
3. Use `PYTHONPATH=gguf-py python examples/m5-tts/convert_backbone.py
   ONNX_DIR/backbone.onnx m5-onnx.gguf` to use identical backbone weights.
4. Build Snapdragon llama.cpp with the release preset and build targets
   `llama` and `htp-v73`. Compile `backbone.cpp` with the Android NDK C++ compiler,
   `-std=c++17 -Iinclude -Iggml/include -L<android-build>/bin -lllama -lggml -lggml-base`.
5. Run `bash examples/m5-tts/build_android_benchmark.sh M5_SOURCE ORT_ANDROID_AAR OUTPUT_DIR`
   with JDK 17, Android platform 35 and build-tools 35.0.0. Use ONNX Runtime Android 1.23.2.
6. Push `classes.dex`, the two ORT ARM64 libraries, `backbone-android`, `m5-onnx.gguf`,
   `libllama.so`, GGML libraries, the v73 skel and `libc++_shared.so` to a device
   directory. Create its `model` subdirectory with `voices.json`, `tokenizer.bin`,
   `manifest.json`, `backbone.onnx`, `depth_frame.onnx`, `codec_cached.json`,
   `codec_decoder_cached.onnx`, `text_embeddings.npy` and `audio_embeddings.npy`.

From the device directory, run the two modes sequentially. Each emits four JSON
records and writes four mono 24 kHz WAVs. The first run warms the engine.

```sh
export CLASSPATH="$PWD/classes.dex" LD_LIBRARY_PATH="$PWD" ADSP_LIBRARY_PATH="$PWD"
app_process / com.m5.offline.AndroidBenchmark model cpu-stream 4 1234 '你好，今天天气很好。'
GGML_HEXAGON_NHMX=0 app_process / com.m5.offline.AndroidBenchmark model htp-stream 4 1234 \
  '你好，今天天气很好。' ./backbone-android m5-onnx.gguf HTP0 --pipe
```

The hybrid runner starts a native process per text segment, so the first
backbone request includes loading and context initialization. The CPU ONNX
sessions load before timing. Use the device-side C++ prefill and decode
counters to separate native execution from process startup and pipe overhead.

## Resident HVX depth and backbone benchmark

Use the same ONNX bundle as the CPU baseline. Export both resident models and the
host-side audio heads and embedding tables.

```sh
PYTHONPATH=gguf-py python examples/m5-tts/convert_backbone.py --depth --quant Q8_0 \
  "$ONNX_DIR/depth_frame.onnx" depth-Q8_0.gguf
PYTHONPATH=gguf-py python examples/m5-tts/convert_backbone.py --quant Q8_0 \
  "$ONNX_DIR/backbone.onnx" backbone-Q8_0.gguf
python examples/m5-tts/pipe_depth.py --android "$ONNX_DIR/depth_frame.onnx" depth-weights.f32
```

Build `backbone.cpp` as above and name the binary `backbone-resident`. Use the Android
NDK compiler to build the audio head library with these options:

```sh
"$ANDROID_CXX" -std=c++17 -O3 -ffast-math -shared -fPIC \
  examples/m5-tts/depth_heads_jni.cpp -o libm5_depth_heads.so
```

Rebuild the benchmark from the patched Android source. Place `depth-weights.f32`,
`depth-Q8_0.gguf` and the CPU codec graph in `model/`; put the resident runner,
backbone GGUF and JNI library alongside `classes.dex`. On the test device:

```sh
export CLASSPATH="$PWD/classes.dex" LD_LIBRARY_PATH="$PWD" ADSP_LIBRARY_PATH="$PWD"
export GGML_HEXAGON_NHMX=0 GGML_HEXAGON_OPPOLL=1
export GGML_HEXAGON_POWER_MODE=sustained_high_performance
export M5_OVERLAP_CODEC=1 M5_FIRST_CHUNK=2 M5_CHUNK=4 M5_RUNS=4
text='今天的天气很好，我们一起去公园散步，看看路边的花，听听树上的鸟叫。'
app_process / com.m5.offline.AndroidBenchmark model long-ort-cpu 4 1234 "$text"
M5_NATIVE_DEPTH=./backbone-resident M5_DEPTH_MODEL=depth-Q8_0.gguf \
  M5_CODEC_MODEL=codec-qdq-conv.onnx \
  app_process / com.m5.offline.AndroidBenchmark model long-hvx 4 1234 "$text" \
  ./backbone-resident backbone-Q8_0.gguf HTP0 --pipe
```

A zero-row pipe message resets the context position and KV metadata between
requests. The host keeps the process alive. The reported first PCM includes
codec priming and production of the first nonempty PCM. Warm-run RTF excludes
the first run. Review the WAV for content and quality when generated lengths differ.

## Resident server and web console (product path)

The engine also builds as an OpenAI-style HTTP server that runs on the device
(`m5-server`), fronted by a single-file web console (`web/index.html`).
`deploy.sh [device-dir]` pushes the binaries and stage libraries, restarts the
server with the HTP environment pinned (`GGML_HEXAGON_NHMX=0 OPPOLL=1`, worth
2.2x - do not drop it), and re-establishes the browser forward at
`http://127.0.0.1:18080`. `/data/local/tmp` is wiped on reboot, so re-run it
after one.

```sh
adb shell 'cd /data/local/tmp/llama-m5-htp-v73 && \
    LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD GGML_HEXAGON_NHMX=0 \
    GGML_HEXAGON_OPPOLL=1 ./m5-server model --port 8080 --cwd $PWD \
    --threads 3 --voice serena --voice-encoder codec_encoder.onnx'
adb forward tcp:18080 tcp:8080
```

Endpoints (details in the `m5_server.cpp` header comment):

```
GET  /health  /metrics  /v1/models  /v1/voices
POST /v1/audio/speech   {"input","voice"?,"response_format"?:"wav"|"pcm"}
POST /v1/voices/clone   {"name","reference_text","codes":[[16]xT]}
POST /v1/voices/encode  {"name","reference_text","wav_base64"}
```

The console defaults to Serena and streaming pcm. Cloning runs only on
`/v1/voices/encode` (the server runs `codec_encoder.onnx` itself); synthesis
never clones, and browser-recorded voices stay in the browser (localStorage),
with zero server-side state.

### Runtime knobs (all changeable without a rebuild)

| knob | default | effect |
|---|---|---|
| `M5_WAV_FIRST` / `M5_WAV_CHUNK` | 2 / 4 | codec chunking, shared by the pcm and wav response paths |
| `M5_KV_PAD` | 32 (depth) / 128 (backbone) | KV rows per graph; `256` restores the upstream behaviour |
| `M5_HVX_DIRECT_V` | on | direct V scatter in the HTP k-params; `0` disables |
| `M5_PRIME_CACHE` | 4 | codec prime LRU entries; `1` = old single-entry cache, `0` = off |
| `M5_CODEC_EP` | CPU | `XNNPACK`/`NNAPI` for A/B; measured no faster (§28) |
| `M5_FIRST_TAIL` | 2400 | first-chunk tail samples; smaller lets `--first 1` emit audio |
| `M5_HEADS_INT8` | on | int8 depth heads; `0` falls back to f16 |
| `M5_PRIME_STATS` / `M5_HEAD_STATS` / `M5_PREFIX_STATS` / `M5_CODEC_PROFILE` | off | diagnostics (cache hits, heads, prefix transfer, ORT profile) |
| `M5_ZERO_TRANSFORM` | off | ablation only (§14); do not ship |

### Where the time goes (SM6650, 69-frame sentence, threads=3)

`wall ≈ 0.874 s + max(depth 2.80, codec 2.44)` — 3.60 s wall / RTF 0.68 for
the wav path, first pcm 0.26–0.29 s (cold ~0.35 s). Concurrency is one FIFO
lane at ~1.5x realtime; a second concurrent request waits.

Measured dead ends (do not retry without new evidence): backbone weight
quantisation (§23), codec on another EP / HTP (§28, §15), speculative depth
drafts (§14), Flash-attention variants (§17), multi-instance parallelism
(§29). The only remaining big lever is the depth+codec retraining plan in §30
of the [optimization log](../../docs/backend/snapdragon/m5-tts-optimization-log.md);
the offline tutorial copy lives in [learning-hub](learning-hub/index.html).
The [Chinese blog series](learning-hub/blog/README.md) covers the pipeline, profiling,
performance and audio quality, and the Serena voice workflow.

## Bring a voice from another TTS model

Generate a short reference WAV and keep its exact spoken transcript in a
UTF-8 text file. For Qwen3-TTS CustomVoice, install `qwen-tts` in a separate
Python 3.12 environment and run:

```sh
python generate_qwen_reference.py serena-reference.wav --speaker Serena
```

The adapter writes mono PCM16 and a matching `.txt` file. Use `--model`,
`--speaker`, `--text`, and `--language` to select another Qwen reference.
If the generated WAV differs from the supplied text, correct the transcript
before encoding. For other TTS models, supply their WAV and transcript
directly at the next step.

In the M5 Python environment (NumPy and ONNX Runtime), encode the reference
using the encoder and identity that match your deployed M5 model:

```sh
python clone_voice.py serena-reference.wav "$(cat serena-reference.txt)" \
    --model "$M5_ENCODER_DIR" --voices serena-voices.json --name serena
```

Point the Android engine at the resulting cache with
`--voices serena-voices.json --voice serena`. Use a different sentence for the
first listening test. The M5 engine uses only its own codec codes at runtime.

For a prompted Serena preset, use the instruction-capable checkpoint:

```sh
python generate_qwen_reference.py serena-calm-reference.wav \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker Serena \
    --instruct '语气轻柔温暖，语速略慢，句尾自然放轻。'
```

Give the new reference a distinct name when encoding it. This keeps the
original Serena preset available for same-text listening comparisons.

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

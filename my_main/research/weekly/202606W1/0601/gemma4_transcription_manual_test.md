# Gemma4 Transcription Manual Test

This folder contains local-only helpers for testing `gemma4:e4b` through Ollama.

## 1. Generate a sample WAV

```bash
./my_main/gemma4_generate_test_wav.sh
```

This creates:

```text
my_main/gemma4_test_ja.wav
```

## 2. Transcribe the sample WAV

```bash
./my_main/gemma4_transcribe_file.sh ./my_main/gemma4_test_ja.wav
```

## 3. Test your own voice

Record a short file with Voice Memos, QuickTime, or another recorder, then run:

```bash
./my_main/gemma4_own_voice_test.sh /path/to/your_recording.wav
```

If `ffmpeg` is installed, you can record directly from the terminal:

```bash
./my_main/gemma4_own_voice_test.sh --list-devices
./my_main/gemma4_own_voice_test.sh --record 10
```

If the default microphone is not device `:0`, choose another one:

```bash
AUDIO_DEVICE=':1' ./my_main/gemma4_own_voice_test.sh --record 10
```

## Useful settings

```bash
OLLAMA_MODEL=gemma4:e4b ./my_main/gemma4_transcribe_file.sh ./my_main/gemma4_test_ja.wav
TRANSCRIBE_PROMPT='日本語として、聞こえた内容だけを文字起こししてください。' ./my_main/gemma4_own_voice_test.sh ./voice.wav
```

The Ollama endpoint used by default is:

```text
http://localhost:11434/v1/audio/transcriptions
```

## 4. Ask a question about audio via llama.cpp

This is separate from Ollama transcription. For llama.cpp audio question answering,
the script sends audio to the OpenAI-compatible chat endpoint with `input_audio`.

Default endpoint used by the script:

```text
http://localhost:11434/v1/chat/completions
```

This default works with the local Ollama `gemma4:e4b` setup. If you want to use
llama.cpp `llama-server` instead, override `LLAMA_SERVER_ENDPOINT` and
`LLAMA_SERVER_MODEL`.

Example:

```bash
./my_main/gemma4_audio_qa_llama.sh ./my_main/gemma4_test_ja.wav
```

By default, the script treats the spoken content inside the audio as the user's question or request and answers based on that audio.

If you want to add a separate instruction on top of the spoken audio, you can still pass one:

```bash
./my_main/gemma4_audio_qa_llama.sh ./voice.wav "この音声の要点を教えてください。"
```

If `ffmpeg` is installed, you can also record from the terminal and ask immediately:

```bash
./my_main/gemma4_audio_qa_llama.sh --list-devices
./my_main/gemma4_audio_qa_llama.sh --record 8
```

If you specifically want to record audio but also attach a separate text instruction, use:

```bash
./my_main/gemma4_audio_qa_llama.sh --record-with-text "この音声の要点を教えてください。" 8
```

If your llama-server uses another model alias or another port:

```bash
LLAMA_SERVER_MODEL=gemma-4-e4b \
LLAMA_SERVER_ENDPOINT=http://localhost:8080/v1/chat/completions \
./my_main/gemma4_audio_qa_llama.sh ./voice.wav "この音声の要点を教えてください。"
```

Notes:

- This script is for llama.cpp server, not Ollama.
- `audio/transcriptions` is transcription-only.
- Audio QA uses `chat/completions` with `input_audio` and requires an audio-capable multimodal model.

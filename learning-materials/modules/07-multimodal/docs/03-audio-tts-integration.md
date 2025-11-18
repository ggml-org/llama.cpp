# Audio & TTS Integration with LLaMA.cpp

## Table of Contents
1. [Introduction](#introduction)
2. [Speech-to-Text: Whisper Integration](#speech-to-text-whisper-integration)
3. [Audio Feature Extraction](#audio-feature-extraction)
4. [Text-to-Speech Integration](#text-to-speech-integration)
5. [Building Voice Assistants](#building-voice-assistants)
6. [Streaming Audio Processing](#streaming-audio-processing)
7. [Performance Optimization](#performance-optimization)
8. [Production Deployment](#production-deployment)

---

## Introduction

Integrating audio capabilities with llama.cpp enables powerful voice-based applications:

### Use Cases

1. **Voice Assistants**: Conversational AI with speech I/O
2. **Transcription + Analysis**: Transcribe and summarize meetings
3. **Accessibility Tools**: Voice-controlled applications
4. **Podcasts & Media**: Automated content generation
5. **Language Learning**: Pronunciation feedback and tutoring
6. **Customer Service**: Voice-based support systems

### Architecture Overview

```
┌─────────────────┐
│  Audio Input    │ (WAV, MP3, streaming)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Speech-to-Text  │ ← Whisper / other ASR
│    (Whisper)    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Text (Prompt)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│   LLaMA.cpp     │ ← Text generation
│   Generation    │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Text Response   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Text-to-Speech  │ ← Piper, Coqui, etc.
│      (TTS)      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Audio Output   │
└─────────────────┘
```

---

## Speech-to-Text: Whisper Integration

### Whisper Overview

**Whisper** is OpenAI's robust speech recognition model, available in llama.cpp ecosystem:

```yaml
Models:
  - tiny: 39M params, ~1GB RAM, very fast
  - base: 74M params, ~1GB RAM, fast
  - small: 244M params, ~2GB RAM, balanced
  - medium: 769M params, ~5GB RAM, good quality
  - large: 1550M params, ~10GB RAM, best quality
```

**Features**:
- Multilingual (99 languages)
- Punctuation and capitalization
- Timestamp alignment
- Noise robustness
- No fine-tuning needed

### whisper.cpp Integration

**whisper.cpp** is the companion project to llama.cpp:

```bash
# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download model
bash ./models/download-ggml-model.sh base

# Transcribe audio
./main -m models/ggml-base.bin -f samples/jfk.wav

# Output:
# [00:00:00.000 --> 00:00:11.000] And so my fellow Americans,
# [00:00:11.000 --> 00:00:13.000] ask not what your country can do for you,
# [00:00:13.000 --> 00:00:17.000] ask what you can do for your country.
```

### Python Integration

```python
import whisper
import numpy as np

class WhisperTranscriber:
    def __init__(self, model_size="base", device="cuda"):
        """
        Initialize Whisper model
        """
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio file to text
        """
        result = self.model.transcribe(
            audio_path,
            language=language,  # None for auto-detect
            task="transcribe",   # or "translate" for English
            fp16=True,          # Use FP16 on GPU
            verbose=False
        )

        return {
            'text': result['text'],
            'segments': result['segments'],
            'language': result['language']
        }

    def transcribe_with_timestamps(self, audio_path):
        """
        Get transcription with word-level timestamps
        """
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True
        )

        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })

        return segments

# Usage
transcriber = WhisperTranscriber("base")
result = transcriber.transcribe("audio.wav")
print(result['text'])
```

### Whisper + LLaMA Pipeline

```python
from llama_cpp import Llama

class VoiceToText:
    def __init__(self, whisper_model, llama_model_path):
        self.transcriber = WhisperTranscriber(whisper_model)
        self.llm = Llama(
            model_path=llama_model_path,
            n_ctx=4096,
            n_gpu_layers=35
        )

    def process_voice_query(self, audio_path):
        """
        Convert voice query to text response
        """
        # Step 1: Transcribe audio
        transcription = self.transcriber.transcribe(audio_path)
        user_text = transcription['text']

        # Step 2: Generate response with LLM
        prompt = f"""You are a helpful voice assistant.

User said: {user_text}

Response:"""

        response = self.llm.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["User said:", "\n\n"]
        )

        return {
            'user_text': user_text,
            'response': response['choices'][0]['text'].strip(),
            'language': transcription['language']
        }

# Usage
voice_ai = VoiceToText("base", "llama-2-7b-chat.Q4_K_M.gguf")
result = voice_ai.process_voice_query("question.wav")
print(f"User: {result['user_text']}")
print(f"Assistant: {result['response']}")
```

---

## Audio Feature Extraction

### Audio Preprocessing

```python
import librosa
import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path, duration=None):
        """
        Load and preprocess audio file
        """
        # Load audio
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=duration
        )

        return audio

    def extract_mel_spectrogram(self, audio):
        """
        Convert audio to mel spectrogram (used by Whisper)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=80,      # Whisper uses 80 mel bins
            n_fft=400,      # 25ms window
            hop_length=160  # 10ms stride
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel, ref=np.max)

        return log_mel

    def extract_mfcc(self, audio, n_mfcc=13):
        """
        Extract MFCC features (alternative to mel spectrogram)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )

        return mfcc

    def normalize_audio(self, audio):
        """
        Normalize audio to [-1, 1]
        """
        return audio / np.max(np.abs(audio))

    def remove_silence(self, audio, threshold=0.01):
        """
        Remove leading/trailing silence
        """
        # Find non-silent regions
        non_silent = librosa.effects.split(
            audio,
            top_db=20
        )

        if len(non_silent) == 0:
            return audio

        # Concatenate non-silent chunks
        trimmed = np.concatenate([audio[start:end]
                                 for start, end in non_silent])

        return trimmed
```

### Voice Activity Detection (VAD)

```python
import webrtcvad

class VoiceActivityDetector:
    def __init__(self, aggressiveness=3):
        """
        aggressiveness: 0-3 (higher = more aggressive filtering)
        """
        self.vad = webrtcvad.Vad(aggressiveness)

    def detect_speech_chunks(self, audio, sample_rate=16000):
        """
        Detect speech segments in audio
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)

        # Frame size (must be 10, 20, or 30 ms)
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        chunks = []
        current_chunk = []
        is_speech = False

        # Process frames
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]

            if len(frame) < frame_size:
                break

            # Check if frame contains speech
            frame_is_speech = self.vad.is_speech(
                frame.tobytes(),
                sample_rate
            )

            if frame_is_speech:
                current_chunk.append(frame)
                is_speech = True
            elif is_speech and len(current_chunk) > 0:
                # End of speech segment
                chunks.append(np.concatenate(current_chunk))
                current_chunk = []
                is_speech = False

        return chunks
```

---

## Text-to-Speech Integration

### TTS Options

llama.cpp doesn't include TTS directly, but integrates easily with:

| TTS System | Quality | Speed | License | Notes |
|------------|---------|-------|---------|-------|
| Piper | Good | Very Fast | MIT | Offline, many voices |
| Coqui TTS | Excellent | Medium | MPL | Open source, flexible |
| Bark | Excellent | Slow | MIT | Realistic, multilingual |
| gTTS | Basic | Fast | MIT | Google TTS (online) |
| AWS Polly | Excellent | Fast | Commercial | Cloud service |

### Piper TTS (Recommended for llama.cpp)

**Piper** is a fast, local neural TTS system:

```bash
# Install Piper
pip install piper-tts

# Download voice model
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

# Generate speech
echo "Hello, this is a test." | piper \
  --model en_US-lessac-medium.onnx \
  --output_file output.wav
```

**Python Integration**:

```python
import subprocess
import tempfile

class PiperTTS:
    def __init__(self, model_path):
        self.model_path = model_path

    def synthesize(self, text, output_path):
        """
        Convert text to speech using Piper
        """
        cmd = [
            "piper",
            "--model", self.model_path,
            "--output_file", output_path
        ]

        # Run Piper
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(input=text.encode('utf-8'))

        if process.returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr.decode()}")

        return output_path

    def synthesize_stream(self, text):
        """
        Stream audio generation (for real-time playback)
        """
        cmd = [
            "piper",
            "--model", self.model_path,
            "--output-raw"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        process.stdin.write(text.encode('utf-8'))
        process.stdin.close()

        # Stream audio chunks
        chunk_size = 4096
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Usage
tts = PiperTTS("en_US-lessac-medium.onnx")
tts.synthesize("Hello world!", "output.wav")
```

### Coqui TTS

```python
from TTS.api import TTS

class CoquiTTS:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.tts = TTS(model_name, gpu=True)

    def synthesize(self, text, output_path):
        """
        Generate speech with Coqui TTS
        """
        self.tts.tts_to_file(
            text=text,
            file_path=output_path
        )

    def synthesize_with_voice_clone(self, text, speaker_wav, output_path):
        """
        Clone a voice and generate speech
        """
        # Use multi-speaker model for voice cloning
        self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_path
        )

# Usage
tts = CoquiTTS()
tts.synthesize("This is a test.", "output.wav")
```

---

## Building Voice Assistants

### Complete Voice Assistant

```python
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import queue

class VoiceAssistant:
    def __init__(self, whisper_model, llama_model, tts_model):
        self.transcriber = WhisperTranscriber(whisper_model)
        self.llm = Llama(model_path=llama_model, n_ctx=4096)
        self.tts = PiperTTS(tts_model)

        self.conversation_history = []

    def listen(self, duration=5, sample_rate=16000):
        """
        Record audio from microphone
        """
        print("Listening...")

        # Record audio
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            return f.name

    def speak(self, text):
        """
        Convert text to speech and play
        """
        print(f"Assistant: {text}")

        # Generate speech
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name

        self.tts.synthesize(text, audio_path)

        # Play audio
        audio, sr = sf.read(audio_path)
        sd.play(audio, sr)
        sd.wait()

    def chat(self, user_audio_path=None, duration=5):
        """
        One turn of conversation
        """
        # Listen if no audio provided
        if user_audio_path is None:
            user_audio_path = self.listen(duration)

        # Transcribe
        transcription = self.transcriber.transcribe(user_audio_path)
        user_text = transcription['text'].strip()

        if not user_text:
            self.speak("I didn't catch that. Could you repeat?")
            return

        print(f"You: {user_text}")

        # Add to history
        self.conversation_history.append({
            'role': 'user',
            'content': user_text
        })

        # Generate response
        prompt = self.build_prompt()

        response = self.llm.create_completion(
            prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["User:", "\n\n"]
        )

        assistant_text = response['choices'][0]['text'].strip()

        # Add to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_text
        })

        # Speak response
        self.speak(assistant_text)

        return {
            'user': user_text,
            'assistant': assistant_text
        }

    def build_prompt(self):
        """
        Build conversation prompt from history
        """
        prompt = "You are a helpful voice assistant. Keep responses concise.\n\n"

        for turn in self.conversation_history[-6:]:  # Last 3 turns
            if turn['role'] == 'user':
                prompt += f"User: {turn['content']}\n"
            else:
                prompt += f"Assistant: {turn['content']}\n"

        prompt += "Assistant:"
        return prompt

    def run(self, turns=5):
        """
        Run multi-turn conversation
        """
        self.speak("Hello! How can I help you?")

        for i in range(turns):
            try:
                self.chat()
            except KeyboardInterrupt:
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                self.speak("I encountered an error. Let's try again.")

# Usage
assistant = VoiceAssistant(
    whisper_model="base",
    llama_model="llama-2-7b-chat.Q4_K_M.gguf",
    tts_model="en_US-lessac-medium.onnx"
)

assistant.run(turns=10)
```

---

## Streaming Audio Processing

### Real-Time Transcription

```python
import pyaudio
import numpy as np
from collections import deque

class StreamingTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds
        self.chunk_size = self.sample_rate * self.chunk_duration

    def stream_transcribe(self, duration=30):
        """
        Transcribe audio in real-time
        """
        audio = pyaudio.PyAudio()

        # Open stream
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        print("Streaming transcription started...")

        audio_buffer = deque(maxlen=self.chunk_size)
        frames_read = 0
        max_frames = duration * self.sample_rate

        try:
            while frames_read < max_frames:
                # Read chunk
                data = stream.read(1024, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                audio_buffer.extend(audio_chunk)
                frames_read += len(audio_chunk)

                # Transcribe when buffer is full
                if len(audio_buffer) == self.chunk_size:
                    audio_array = np.array(audio_buffer)

                    # Transcribe
                    result = self.model.transcribe(
                        audio_array,
                        fp16=False,
                        language="en"
                    )

                    yield result['text']

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

# Usage
transcriber = StreamingTranscriber("base")

for text in transcriber.stream_transcribe(duration=30):
    print(f"Transcription: {text}")
```

### Low-Latency Pipeline

```python
class LowLatencyVoiceAssistant:
    def __init__(self, whisper_model, llama_model, tts_model):
        self.transcriber = WhisperTranscriber(whisper_model)
        self.llm = Llama(model_path=llama_model, n_ctx=2048)
        self.tts = PiperTTS(tts_model)

    def process_with_vad(self, audio_stream):
        """
        Process audio with voice activity detection for lower latency
        """
        vad = VoiceActivityDetector(aggressiveness=3)

        for audio_chunk in audio_stream:
            # Detect speech
            speech_chunks = vad.detect_speech_chunks(audio_chunk)

            if not speech_chunks:
                continue

            # Process each speech segment immediately
            for speech in speech_chunks:
                # Transcribe
                with tempfile.NamedTemporaryFile(suffix='.wav') as f:
                    sf.write(f.name, speech, 16000)
                    text = self.transcriber.transcribe(f.name)['text']

                if text.strip():
                    # Generate response (streaming)
                    response = self.llm.create_completion(
                        f"User: {text}\nAssistant:",
                        max_tokens=100,
                        stream=True
                    )

                    # Accumulate response
                    full_response = ""
                    for chunk in response:
                        token = chunk['choices'][0]['text']
                        full_response += token

                        # Speak when we have a sentence
                        if token in ['.', '!', '?']:
                            self.tts.synthesize(full_response.strip(), "temp.wav")
                            # Play async
                            play_audio_async("temp.wav")
                            full_response = ""
```

---

## Performance Optimization

### Model Selection

```python
# Latency vs Quality tradeoffs

# Ultra-low latency (< 200ms)
whisper_model = "tiny"    # 39M params
llama_model = "phi-2"     # 2.7B params
tts_model = "piper-fast"  # Lightweight voice

# Balanced (< 500ms)
whisper_model = "base"    # 74M params
llama_model = "llama-7b"  # 7B params, Q4_K_M
tts_model = "piper-medium"

# Quality (< 2s)
whisper_model = "medium"  # 769M params
llama_model = "llama-13b" # 13B params, Q5_K_M
tts_model = "coqui-vits"
```

### GPU Acceleration

```python
# Optimize for GPU
whisper_model = whisper.load_model("base", device="cuda")  # GPU for Whisper

llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=35,    # Offload to GPU
    n_ctx=2048,         # Shorter context for speed
    n_batch=512         # Larger batch for throughput
)

# TTS on GPU
tts = CoquiTTS(gpu=True)
```

### Benchmark Results

Typical latencies on RTX 4090:

| Component | Tiny | Base | Small | Medium |
|-----------|------|------|-------|--------|
| Whisper (5s audio) | 50ms | 100ms | 300ms | 800ms |
| LLaMA-7B (50 tokens) | 500ms | 500ms | 500ms | 500ms |
| Piper TTS (1 sentence) | 100ms | 100ms | 100ms | 100ms |
| **Total** | **650ms** | **700ms** | **900ms** | **1400ms** |

---

## Production Deployment

### API Server

```python
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

# Initialize models (once at startup)
assistant = VoiceAssistant(
    whisper_model="base",
    llama_model="llama-2-7b-chat.Q4_K_M.gguf",
    tts_model="en_US-lessac-medium.onnx"
)

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """
    Accept audio file, return text response + audio response
    """
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(await audio.read())
        audio_path = f.name

    # Process
    result = assistant.chat(audio_path)

    # Generate speech for response
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        response_audio_path = f.name

    assistant.tts.synthesize(result['assistant'], response_audio_path)

    # Return both text and audio
    with open(response_audio_path, 'rb') as f:
        audio_bytes = f.read()

    return {
        'user_text': result['user'],
        'assistant_text': result['assistant'],
        'audio': audio_bytes.hex()  # Or return file directly
    }

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install llama-cpp-python whisper piper-tts fastapi uvicorn soundfile

# Copy models
COPY models/ /app/models/

# Copy application
COPY app.py /app/

WORKDIR /app

# Run server
CMD ["python3", "app.py"]
```

---

## Summary

Audio and TTS integration with llama.cpp enables powerful voice applications:

✅ **Speech-to-Text**: Whisper integration for robust transcription
✅ **Audio Processing**: Feature extraction, VAD, streaming
✅ **Text-to-Speech**: Piper, Coqui, and other TTS systems
✅ **Voice Assistants**: Complete conversational AI pipelines
✅ **Optimization**: Low-latency processing, GPU acceleration
✅ **Production**: API servers, Docker deployment

**Next Steps**:
- Build a voice assistant with the provided code
- Experiment with different model combinations
- Optimize for your latency requirements
- Explore Lesson 7.4 on custom architectures

---

**References**:
- Radford et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
- Piper TTS: https://github.com/rhasspy/piper
- Coqui TTS: https://github.com/coqui-ai/TTS
- whisper.cpp: https://github.com/ggerganov/whisper.cpp

#!/usr/bin/env python3
"""
Phase 0: Gemma 4 UA incremental 40ms audio streaming prototype.

Proves that audio embeddings can be injected into the LLM KV cache
one 40ms chunk at a time, then text is generated at speech boundaries.

Architecture (no fine-tuning needed):
  WAV / mic -> 40ms frames -> rms_norm -> mm.a.input_projection.weight
            -> llama_decode(embd) x N -> suffix tokens -> token sampling -> stdout

Requirements:
  pip install llama-cpp-python silero-vad sounddevice
  (CUDA build: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python)

Usage (WAV file):
  python phase0_streaming.py \\
      --model  ~/.cache/llama.cpp/ggml-org_gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \\
      --mmproj ~/.cache/llama.cpp/ggml-org_gemma-4-12B-it-GGUF/mmproj-model-f16.gguf \\
      --audio  ./input.wav

Usage (Microphone):
  python phase0_streaming.py \\
      --model  ~/.cache/llama.cpp/ggml-org_gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \\
      --mmproj ~/.cache/llama.cpp/ggml-org_gemma-4-12B-it-GGUF/mmproj-model-f16.gguf \\
      --mic

Notes:
  - mmproj GGUF: f16, f32, or Q8_0 tensors are all supported.
  - Audio must be 16kHz mono WAV (for --audio mode).
    Convert: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav
  - Default VAD: silero (robust to noise; requires pip install silero-vad)
    Fallback:    --vad-backend energy (no extra deps; WAV-only)
  - --mic requires: pip install sounddevice
  - --save-turns DIR: saves each detected utterance as turn_NNNN.wav in DIR
    Useful for Phase 1 fine-tune verification: replay the same audio against
    the base model and the fine-tuned model to compare responses.
  - Find cached model files: ls ~/.cache/llama.cpp/ggml-org_gemma-4-12B-it-GGUF/
"""

import argparse
import ctypes
import struct
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Iterator, List, Optional

import numpy as np
import scipy.io.wavfile as wavfile

# --------------------------------------------------------------------------- #
#  Gemma 4 UA audio constants (clip.cpp PROJECTOR_TYPE_GEMMA4UA)              #
# --------------------------------------------------------------------------- #
SAMPLE_RATE   = 16000   # model expects 16kHz
CHUNK_SAMPLES = 640     # 40ms frame: n_mel_bins = 640 samples
RMS_EPS       = 1e-6    # hparams.eps for gemma4ua rms_norm

# --------------------------------------------------------------------------- #
#  Energy-based VAD (fallback; no extra deps; WAV-only)                        #
# --------------------------------------------------------------------------- #
_VAD_THRESHOLD     = 0.003
_VAD_SILENCE_GRACE = 12      # consecutive silent frames before utterance_end (~480ms)
_VAD_MIN_SPEECH    = 15      # ignore speech bursts shorter than this many frames (~600ms)

class EnergyVAD:
    """
    Lightweight energy VAD. Returns one of:
      'speech'        - frame has voice energy
      'silence'       - quiet frame
      'utterance_end' - emitted once after speech + sufficient silence

    ref_energy: file-wide mean energy (float(np.mean(samples**2))).
    Only usable with WAV files; not suitable for mic (no ref energy available).
    """
    def __init__(self, ref_energy: float,
                 threshold: float = _VAD_THRESHOLD,
                 silence_grace: int = _VAD_SILENCE_GRACE,
                 min_speech: int = _VAD_MIN_SPEECH):
        self._ref       = ref_energy
        self.threshold  = threshold
        self.grace      = silence_grace
        self.min_speech = min_speech
        self._active    = False
        self._silence   = 0
        self._n_speech  = 0

    def process(self, chunk: np.ndarray) -> str:
        is_speech = float(np.mean(chunk ** 2)) > self._ref * self.threshold
        if is_speech:
            self._active    = True
            self._n_speech += 1
            self._silence   = 0
            return 'speech'
        if self._active and self._n_speech >= self.min_speech:
            self._silence += 1
            if self._silence >= self.grace:
                self._active   = False
                self._silence  = 0
                self._n_speech = 0
                return 'utterance_end'
        return 'silence'


# --------------------------------------------------------------------------- #
#  Silero-VAD wrapper (default; requires: pip install silero-vad)             #
# --------------------------------------------------------------------------- #
class SileroVAD:
    """
    Neural VAD using Silero-VAD. Works for both --audio and --mic.

    Silero-VAD expects 512 samples at 16kHz. Each 640-sample Gemma4 frame is
    truncated to 512 for the VAD decision only; the projector still receives
    the full 640 samples.
    """
    _WINDOW = 512   # Silero-VAD fixed window at 16kHz

    def __init__(self, threshold: float = 0.5,
                 silence_ms: int = 480,
                 min_speech: int = 15,
                 speech_pad_ms: int = 30):
        try:
            from silero_vad import load_silero_vad, VADIterator
        except ImportError:
            sys.exit(
                "ERROR: silero-vad not found.\n"
                "Install: pip install silero-vad\n"
                "  or use --vad-backend energy (WAV-only, no extra deps)"
            )
        model = load_silero_vad()
        self._vad = VADIterator(
            model,
            threshold               = threshold,
            sampling_rate           = SAMPLE_RATE,
            min_silence_duration_ms = silence_ms,
            speech_pad_ms           = speech_pad_ms,
        )
        self.min_speech = min_speech
        self._active    = False
        self._n_speech  = 0

    def process(self, chunk: np.ndarray) -> str:
        """Process one 640-sample chunk → 'speech' | 'silence' | 'utterance_end'."""
        import torch
        # Silero expects exactly 512 samples; truncate our 640-sample frame
        vad_in = torch.from_numpy(chunk[:self._WINDOW].astype(np.float32))
        event  = self._vad(vad_in)

        if event is not None:
            if 'start' in event:
                self._active   = True
                self._n_speech = 0
            elif 'end' in event and self._active:
                self._active = False
                n = self._n_speech
                self._n_speech = 0
                return 'utterance_end' if n >= self.min_speech else 'silence'

        if self._active:
            self._n_speech += 1
            return 'speech'
        return 'silence'

    def reset(self):
        self._vad.reset_states()
        self._active   = False
        self._n_speech = 0


# --------------------------------------------------------------------------- #
#  Microphone input (requires: pip install sounddevice)                        #
# --------------------------------------------------------------------------- #
class MicStream:
    """Real-time 16kHz mono microphone input via sounddevice."""

    def __init__(self, blocksize: int = CHUNK_SAMPLES,
                 samplerate: int = SAMPLE_RATE):
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            sys.exit(
                "ERROR: sounddevice not found.\n"
                "Install: pip install sounddevice"
            )
        self._blocksize  = blocksize
        self._samplerate = samplerate
        self._queue: Queue = Queue()
        self._stream     = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"\n[mic] {status}", file=sys.stderr)
        self._queue.put(indata[:, 0].copy())

    def __enter__(self):
        self._stream = self._sd.InputStream(
            samplerate = self._samplerate,
            channels   = 1,
            dtype      = 'float32',
            blocksize  = self._blocksize,
            callback   = self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *args):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield self._queue.get()


# --------------------------------------------------------------------------- #
#  Minimal GGUF tensor reader                                                  #
# GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md        #
# Supports f32 (0), f16 (1), Q8_0 (8).                                        #
# --------------------------------------------------------------------------- #
_GGUF_MAGIC = 0x46554747  # "GGUF"
_GGUF_DEFAULT_ALIGNMENT = 32

# KV type id -> (struct format char, byte size)
_GGUF_KV_FMTS = {
    0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
    4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
    10: ('Q', 8), 11: ('q', 8), 12: ('d', 8),
    # 8=STRING, 9=ARRAY handled separately
}

def _gguf_read_str(f) -> str:
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8')

def _gguf_skip_kv_value(f, typ: int):
    if typ == 8:   # STRING
        n, = struct.unpack('<Q', f.read(8)); f.read(n)
    elif typ == 9: # ARRAY
        et, = struct.unpack('<I', f.read(4))
        n,  = struct.unpack('<Q', f.read(8))
        for _ in range(n): _gguf_skip_kv_value(f, et)
    else:
        _, size = _GGUF_KV_FMTS[typ]
        f.read(size)

def _dequantize_q8_0(raw: bytes, n_elem: int) -> np.ndarray:
    """
    Dequantize Q8_0 blocks to float32.
    Block layout: [f16 scale (2 bytes)] [int8 x 32 (32 bytes)] = 34 bytes/block
    """
    QK = 32
    BLOCK = 34
    n_blocks = (n_elem + QK - 1) // QK
    arr = np.frombuffer(raw[:n_blocks * BLOCK], dtype=np.uint8).reshape(n_blocks, BLOCK)
    scales = np.frombuffer(arr[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    qs     = np.frombuffer(arr[:, 2:].tobytes(), dtype=np.int8).astype(np.float32).reshape(n_blocks, QK)
    return (scales[:, np.newaxis] * qs).reshape(-1)[:n_elem]

def gguf_load_tensor_f32(gguf_path: str, tensor_name: str) -> np.ndarray:
    """
    Extract one tensor from a GGUF file and return it as float32 ndarray.
    Tensor shape follows numpy convention (slowest-varying axis first).
    Supports dtypes: F32 (0), F16 (1), Q8_0 (8).
    """
    with open(gguf_path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        if magic != _GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))

        # Skip key-value metadata
        for _ in range(n_kv):
            _gguf_read_str(f)
            typ, = struct.unpack('<I', f.read(4))
            _gguf_skip_kv_value(f, typ)

        # Read tensor info headers
        headers = []
        for _ in range(n_tensors):
            name    = _gguf_read_str(f)
            ndim,   = struct.unpack('<I', f.read(4))
            dims    = struct.unpack(f'<{ndim}Q', f.read(8 * ndim))
            dtype,  = struct.unpack('<I', f.read(4))
            offset, = struct.unpack('<Q', f.read(8))
            headers.append((name, dims, dtype, offset))

        # Data section starts at the next _GGUF_DEFAULT_ALIGNMENT boundary
        pos = f.tell()
        data_start = (pos + _GGUF_DEFAULT_ALIGNMENT - 1) & ~(_GGUF_DEFAULT_ALIGNMENT - 1)

        for (name, dims, dtype, offset) in headers:
            if name != tensor_name:
                continue
            if dtype not in (0, 1, 8):
                raise ValueError(
                    f"Tensor '{name}' dtype={dtype} is not supported. "
                    "Only F32 (0), F16 (1), Q8_0 (8) are handled."
                )
            f.seek(data_start + offset)
            n_elem = 1
            for d in dims:
                n_elem *= d
            if dtype == 0:    # F32
                data = np.frombuffer(f.read(n_elem * 4), dtype=np.float32).copy()
            elif dtype == 1:  # F16
                data = np.frombuffer(f.read(n_elem * 2), dtype=np.float16).astype(np.float32)
            else:             # Q8_0
                QK, BLOCK = 32, 34
                n_blocks = (n_elem + QK - 1) // QK
                raw = f.read(n_blocks * BLOCK)
                data = _dequantize_q8_0(raw, n_elem)
            # GGUF: dims[0] is the fastest-varying axis (ggml ne[0]).
            # Reverse to get numpy/row-major shape.
            return data.reshape(tuple(reversed(dims)))

    raise KeyError(f"Tensor '{tensor_name}' not found in {gguf_path}")


# --------------------------------------------------------------------------- #
#  Gemma 4 UA audio projector (replicates gemma4ua.cpp in numpy)              #
#                                                                              #
#  clip_graph_gemma4ua::build():                                               #
#    cur = ggml_rms_norm(inp, eps)             <- over 640 samples             #
#    cur = ggml_mul_mat(mm_input_proj_w, cur)  <- 640 -> n_embd              #
# --------------------------------------------------------------------------- #
class Gemma4UAProjector:
    WEIGHT = "mm.a.input_projection.weight"

    def __init__(self, mmproj_path: str):
        w = gguf_load_tensor_f32(mmproj_path, self.WEIGHT)
        # GGUF stores Linear weights as [out, in] -> after reversed: (n_embd, 640)
        if w.ndim != 2 or w.shape[1] != CHUNK_SAMPLES:
            raise ValueError(
                f"Unexpected shape {w.shape}; expected (n_embd, {CHUNK_SAMPLES})"
            )
        self.w      = w             # (n_embd, 640)
        self.n_embd = w.shape[0]
        print(f"[projector] {self.WEIGHT} shape={w.shape}  n_embd={self.n_embd}")

    def project(self, chunk: np.ndarray) -> np.ndarray:
        """(640,) float32 -> (n_embd,) float32"""
        rms    = float(np.sqrt(np.mean(chunk ** 2) + RMS_EPS))
        normed = chunk / rms
        return (self.w @ normed).astype(np.float32)   # (n_embd,)


# --------------------------------------------------------------------------- #
#  llama-cpp-python import helper                                              #
# --------------------------------------------------------------------------- #
def _import_llama_cpp():
    try:
        import llama_cpp as lc
        return lc
    except ImportError:
        sys.exit(
            "ERROR: llama-cpp-python not found.\n"
            "Install: pip install llama-cpp-python\n"
            "  (CUDA): CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python\n"
            "  (Metal): CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
        )


# --------------------------------------------------------------------------- #
#  Low-level batch decode helpers                                              #
# --------------------------------------------------------------------------- #
def _decode_tokens(lc, ctx, tokens: List[int], start_pos: int,
                   need_logits_on_last: bool = False) -> int:
    """Decode a sequence of token IDs into KV cache. Returns next position."""
    n = len(tokens)
    if n == 0:
        return start_pos
    batch = lc.llama_batch_init(n, 0, 1)
    batch.n_tokens = n
    for i, tok in enumerate(tokens):
        batch.token[i]      = tok
        batch.pos[i]        = start_pos + i
        batch.n_seq_id[i]   = 1
        batch.seq_id[i][0]  = 0
        batch.logits[i]     = 1 if (need_logits_on_last and i == n - 1) else 0
    ret = lc.llama_decode(ctx, batch)
    lc.llama_batch_free(batch)
    if ret != 0:
        raise RuntimeError(f"llama_decode (tokens) failed: {ret}")
    return start_pos + n


def _decode_embeds(lc, ctx, embeds: List[np.ndarray],
                   start_pos: int, n_embd: int) -> int:
    """
    Inject audio embeddings one frame at a time into the KV cache.
    This is the core of the incremental streaming: each 40ms chunk
    appends one embedding vector to the running context.
    Returns next position.
    """
    batch = lc.llama_batch_init(1, n_embd, 1)
    pos   = start_pos
    for emb in embeds:
        batch.n_tokens    = 1
        ctypes.memmove(
            batch.embd,
            emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_embd * ctypes.sizeof(ctypes.c_float),
        )
        batch.pos[0]       = pos
        batch.n_seq_id[0]  = 1
        batch.seq_id[0][0] = 0
        batch.logits[0]    = 0   # no logits needed during audio ingestion
        ret = lc.llama_decode(ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode (embd) failed at pos {pos}: {ret}")
        pos += 1
    lc.llama_batch_free(batch)
    return pos


# --------------------------------------------------------------------------- #
#  WAV loading                                                                 #
# --------------------------------------------------------------------------- #
def load_wav_f32(path: str) -> np.ndarray:
    sr, raw = wavfile.read(path)
    if sr != SAMPLE_RATE:
        sys.exit(
            f"ERROR: sample rate is {sr}, expected {SAMPLE_RATE}.\n"
            "Convert: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav"
        )
    if raw.ndim > 1:
        raw = raw.mean(axis=1)  # stereo -> mono
    if raw.dtype == np.int16:
        return raw.astype(np.float32) / 32768.0
    if raw.dtype == np.int32:
        return raw.astype(np.float32) / 2147483648.0
    return raw.astype(np.float32)


def _wav_to_chunks(samples: np.ndarray) -> Iterator[np.ndarray]:
    """Yield padded 640-sample chunks from a pre-loaded WAV array."""
    n = (len(samples) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    for i in range(n):
        chunk = samples[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        yield chunk


# --------------------------------------------------------------------------- #
#  Main streaming loop                                                         #
# --------------------------------------------------------------------------- #
def run(model_path: str, mmproj_path: str,
        audio_path: Optional[str],
        use_mic: bool,
        vad_backend: str,
        user_prompt: str,
        n_ctx: int, n_gpu_layers: int,
        silero_threshold: float,
        vad_threshold: float,
        save_turns: Optional[str],
        flash_attn: bool = True,
        show_thinking: bool = False):

    lc = _import_llama_cpp()

    # 1. Load audio projector ------------------------------------------------
    proj   = Gemma4UAProjector(mmproj_path)
    n_embd = proj.n_embd

    # 2. Load LLM ------------------------------------------------------------
    fa_str = 'on' if flash_attn else 'off'
    print(f"[llama] loading {Path(model_path).name}  n_ctx={n_ctx}  gpu_layers={n_gpu_layers}  flash_attn={fa_str}")
    model = lc.Llama(
        model_path    = model_path,
        n_ctx         = n_ctx,
        n_gpu_layers  = n_gpu_layers,
        embedding     = False,
        verbose       = False,
        flash_attn    = flash_attn,
    )
    ctx   = model._ctx.ctx       # llama_context*
    lm    = model._model.model   # llama_model*
    vocab = lc.llama_model_get_vocab(lm)  # llama_vocab* for eog check

    # 3. Build prompt token sequences ----------------------------------------
    # Gemma 4 IT chat template (audio inlined as embeddings, no audio soft tokens):
    #   <bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n
    # Audio embeddings are injected between PREFIX and SUFFIX.
    def tok(text: str) -> List[int]:
        return model.tokenize(text.encode(), add_bos=False, special=True)

    BOS    = model.token_bos()
    PREFIX = [BOS] + tok("<start_of_turn>user\n")
    SUFFIX = tok(f"{user_prompt}<end_of_turn>\n<start_of_turn>model\n")

    # 4. Set up VAD ----------------------------------------------------------
    if vad_backend == 'energy':
        if use_mic:
            sys.exit("ERROR: --vad-backend energy is not supported with --mic")
        samples    = load_wav_f32(audio_path)
        ref_energy = float(np.mean(samples ** 2))
        n_chunks   = (len(samples) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
        dur_s      = len(samples) / SAMPLE_RATE
        print(f"[audio] {dur_s:.2f}s  {n_chunks} frames @ 40ms  ref_energy={ref_energy:.5f}")
        vad = EnergyVAD(ref_energy=ref_energy, threshold=vad_threshold)
    else:  # silero (default)
        vad = SileroVAD(threshold=silero_threshold)
        if not use_mic:
            samples  = load_wav_f32(audio_path)
            n_chunks = (len(samples) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
            dur_s    = len(samples) / SAMPLE_RATE
            print(f"[audio] {dur_s:.2f}s  {n_chunks} frames @ 40ms")

    # 5. Prepare turn save directory -----------------------------------------
    turns_dir: Optional[Path] = None
    if save_turns:
        turns_dir = Path(save_turns)
        turns_dir.mkdir(parents=True, exist_ok=True)
        print(f"[turns] saving detected utterances to {turns_dir}/")

    # 6. respond() closure ---------------------------------------------------
    turn_idx = [0]

    def respond(speech_frames: List[np.ndarray],
                speech_raw: Optional[List[np.ndarray]]):
        # Optionally save raw audio for Phase 1 verification
        if turns_dir is not None and speech_raw:
            raw_audio = np.concatenate(speech_raw)
            wav_path  = turns_dir / f"turn_{turn_idx[0]:04d}.wav"
            wavfile.write(str(wav_path), SAMPLE_RATE,
                          (raw_audio * 32768).astype(np.int16))
            print(f"  [turn] saved {wav_path}  ({len(raw_audio)/SAMPLE_RATE:.2f}s)")
        turn_idx[0] += 1

        # Clear KV cache
        mem = lc.llama_get_memory(ctx)
        if mem is not None:
            lc.llama_memory_clear(mem, True)
        pos = 0

        pos = _decode_tokens(lc, ctx, PREFIX, pos)

        n_frames = len(speech_frames)
        print(f"  [inject] {n_frames} frames ({n_frames * 40}ms) into KV cache...",
              end=' ', flush=True)
        t_inject = time.monotonic()
        pos = _decode_embeds(lc, ctx, speech_frames, pos, n_embd)
        print(f"{time.monotonic() - t_inject:.2f}s")

        # Last suffix token gets logits=1 so we can immediately sample
        pos = _decode_tokens(lc, ctx, SUFFIX, pos, need_logits_on_last=True)

        # top_k(40) + greedy: robust when logits contain NaN (expected without fine-tuning)
        sparams = lc.llama_sampler_chain_default_params()
        sampler = lc.llama_sampler_chain_init(sparams)
        lc.llama_sampler_chain_add(sampler, lc.llama_sampler_init_top_k(40))
        lc.llama_sampler_chain_add(sampler, lc.llama_sampler_init_greedy())

        gen_batch = lc.llama_batch_init(1, 0, 1)

        raw_logits = lc.llama_get_logits(ctx)
        if raw_logits:
            import ctypes as _ct
            n_vocab = lc.llama_n_vocab(lc.llama_model_get_vocab(lm))
            arr = (_ct.c_float * n_vocab).from_address(_ct.addressof(raw_logits.contents))
            sample = [arr[i] for i in range(min(10, n_vocab))]
            has_nan = any(x != x for x in sample)
            print(f"  [logits] first 10: {[f'{v:.2f}' for v in sample]}  nan={has_nan}")

        # Gemma 4 thinking token IDs (fixed in vocab)
        TOK_THINK_START = 100  # <|channel>
        TOK_THINK_END   = 101  # <channel|>

        print("[model] ", end='', flush=True)
        t_gen = time.monotonic()
        n_tok = 0
        n_think_tok = 0
        in_thinking = False

        for _ in range(512):
            new_tok = lc.llama_sampler_sample(sampler, ctx, -1)
            lc.llama_sampler_accept(sampler, new_tok)

            # llama_vocab_is_eog covers eos + end_of_turn + eot_id + all Gemma 4 stop tokens
            if lc.llama_vocab_is_eog(vocab, new_tok):
                break

            if new_tok == TOK_THINK_START:
                in_thinking = True
                if show_thinking:
                    print('\033[2m<|channel>\033[0m', end='', flush=True)
            elif new_tok == TOK_THINK_END:
                in_thinking = False
                if show_thinking:
                    print('\033[2m<channel|>\033[0m', end='', flush=True)
                elif n_think_tok > 0:
                    print(f'\033[2m[{n_think_tok} thinking tokens]\033[0m\n',
                          end='', flush=True)
                n_think_tok = 0
            elif in_thinking:
                n_think_tok += 1
                if show_thinking:
                    piece = model.detokenize([new_tok]).decode('utf-8', errors='replace')
                    print('\033[2m' + piece + '\033[0m', end='', flush=True)
            else:
                piece = model.detokenize([new_tok]).decode('utf-8', errors='replace')
                print(piece, end='', flush=True)
                n_tok += 1

            gen_batch.n_tokens    = 1
            gen_batch.token[0]    = new_tok
            gen_batch.pos[0]      = pos
            gen_batch.n_seq_id[0] = 1
            gen_batch.seq_id[0][0] = 0
            gen_batch.logits[0]   = 1
            lc.llama_decode(ctx, gen_batch)
            pos += 1

        lc.llama_sampler_free(sampler)
        lc.llama_batch_free(gen_batch)
        elapsed = time.monotonic() - t_gen
        tps = n_tok / elapsed if elapsed > 0 else 0
        think_info = f'  think={n_think_tok}tok' if n_think_tok else ''
        print(f"\n  [perf] {n_tok} tokens  {elapsed:.2f}s  {tps:.1f} tok/s{think_info}")

    # 7. Unified streaming loop ----------------------------------------------
    speech_frames: List[np.ndarray] = []
    speech_raw:    List[np.ndarray] = []
    t_start = time.monotonic()
    frame_idx = 0

    def _process_chunk(chunk: np.ndarray):
        nonlocal frame_idx, speech_frames, speech_raw
        embed  = proj.project(chunk)
        status = vad.process(chunk)
        frame_idx += 1
        ms = frame_idx * 40

        if use_mic:
            print(f"\r[{ms:8d}ms] {status:14s}", end='', flush=True)
        else:
            total_ms = int(dur_s * 1000)
            print(f"\r[{ms:6d}/{total_ms:d}ms] {status:14s}", end='', flush=True)

        if status in ('speech', 'utterance_end'):
            speech_frames.append(embed)
            if turns_dir is not None:
                speech_raw.append(chunk.copy())

        if status == 'utterance_end':
            n = len(speech_frames)
            print(f"\n[VAD] utterance: {n} frames ({n * 40}ms)")
            respond(speech_frames, speech_raw if turns_dir else None)
            speech_frames = []
            speech_raw    = []

    if use_mic:
        print("[mic] Listening... Press Ctrl+C to stop.\n")
        try:
            with MicStream() as mic:
                for chunk in mic:
                    _process_chunk(chunk)
        except KeyboardInterrupt:
            print("\n[mic] stopped")
    else:
        for chunk in _wav_to_chunks(samples):
            _process_chunk(chunk)

        # Trailing speech if file ends without silence
        if speech_frames and len(speech_frames) >= (
            vad.min_speech if hasattr(vad, 'min_speech') else _VAD_MIN_SPEECH
        ):
            n = len(speech_frames)
            print(f"\n[VAD] trailing utterance: {n} frames ({n * 40}ms)")
            respond(speech_frames, speech_raw if turns_dir else None)

    print(f"\n[done] total wall time: {time.monotonic() - t_start:.2f}s")
    if turns_dir:
        saved = list(turns_dir.glob("turn_*.wav"))
        print(f"[turns] {len(saved)} utterance(s) saved to {turns_dir}/")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--model',  required=True,
                   help='Path to Gemma 4 12B main GGUF')
    p.add_argument('--mmproj', required=True,
                   help='Path to Gemma 4 UA mmproj GGUF (f16 or f32)')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--audio', metavar='WAV',
                     help='Path to input WAV (16kHz mono)')
    src.add_argument('--mic', action='store_true',
                     help='Read audio from microphone in real time')

    p.add_argument('--vad-backend', choices=['silero', 'energy'], default='silero',
                   help='VAD backend: silero (default, robust) or energy (no deps, WAV-only)')
    p.add_argument('--silero-threshold', type=float, default=0.5,
                   help='Silero-VAD speech probability threshold 0–1 (default: 0.5)')
    p.add_argument('--vad-threshold', type=float, default=_VAD_THRESHOLD,
                   help=f'Energy VAD threshold as fraction of mean (default: {_VAD_THRESHOLD})')
    p.add_argument('--prompt', default='解答して',
                   help='User text prompt appended after audio (default: "解答して")')
    p.add_argument('--n-ctx',        type=int, default=4096)
    p.add_argument('--n-gpu-layers', type=int, default=-1,
                   help='-1 = all layers to GPU (default)')
    p.add_argument('--save-turns', metavar='DIR',
                   help='Save each detected utterance as turn_NNNN.wav in DIR '
                        '(useful for Phase 1 fine-tune verification)')
    p.add_argument('--flash-attn', action='store_true', default=True,
                   help='Enable Flash Attention (default: on)')
    p.add_argument('--no-flash-attn', action='store_false', dest='flash_attn',
                   help='Disable Flash Attention')
    p.add_argument('--show-thinking', action='store_true', default=False,
                   help='Print Gemma 4 thinking tokens in dim color (default: hidden)')
    args = p.parse_args()

    if args.mic and args.vad_backend == 'energy':
        p.error("--vad-backend energy is not supported with --mic; "
                "use the default --vad-backend silero")

    run(
        model_path       = args.model,
        mmproj_path      = args.mmproj,
        audio_path       = args.audio,
        use_mic          = args.mic,
        vad_backend      = args.vad_backend,
        user_prompt      = args.prompt,
        n_ctx            = args.n_ctx,
        n_gpu_layers     = args.n_gpu_layers,
        silero_threshold = args.silero_threshold,
        vad_threshold    = args.vad_threshold,
        save_turns       = args.save_turns,
        flash_attn       = args.flash_attn,
        show_thinking    = args.show_thinking,
    )

if __name__ == '__main__':
    main()

"""
Qwen3-TTS Streamlit App — interactive text-to-speech with all features.

Launch:
    cd examples/qwen3-tts
    pip install -r requirements.txt
    streamlit run app.py

Features:
    - Text-to-speech synthesis in 10 languages
    - Voice cloning (x-vector and ICL modes)
    - Streaming text mode
    - Performance metrics display
    - Audio playback and download
    - Batch synthesis & multilingual comparison
    - Performance benchmark
"""

import io
import os
import re
import subprocess
import tempfile
import wave

import numpy as np
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

LANGUAGES = [
    "english", "chinese", "german", "spanish", "french",
    "italian", "japanese", "korean", "portuguese", "russian", "auto",
]

EXAMPLE_TEXTS = {
    "english": "Hello! This is a demonstration of Qwen3 text to speech synthesis running entirely in llama.cpp.",
    "chinese": "你好！这是千问三号文本转语音合成的演示。",
    "german": "Hallo! Dies ist eine Demonstration der Qwen3 Sprachsynthese.",
    "spanish": "¡Hola! Esta es una demostración de la síntesis de voz Qwen3.",
    "french": "Bonjour! Ceci est une démonstration de la synthèse vocale Qwen3.",
    "italian": "Ciao! Questa è una dimostrazione della sintesi vocale Qwen3.",
    "japanese": "こんにちは！Qwen3のテキスト読み上げ合成のデモンストレーションです。",
    "korean": "안녕하세요! Qwen3 텍스트 음성 합성 시연입니다.",
    "portuguese": "Olá! Esta é uma demonstração da síntese de voz Qwen3.",
    "russian": "Здравствуйте! Это демонстрация синтеза речи Qwen3.",
    "auto": "Hello, this is a test.",
}


def find_binary():
    candidates = [
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "Debug", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts.exe"),
        os.path.join(REPO_ROOT, "build", "bin", "llama-qwen3tts"),
        os.path.join(REPO_ROOT, "build", "llama-qwen3tts"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_models():
    """Locate GGUF model files. Prefers f16 vocoder over bf16 (ggml conv1d requires f16)."""
    models_dir = os.path.join(REPO_ROOT, "models")
    talker = cp = vocoder = None
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            fl = f.lower()
            if "qwen3tts" not in fl or not fl.endswith(".gguf"):
                continue
            if "talker" in fl:
                if talker is None or "bf16" in fl:
                    talker = os.path.join(models_dir, f)
            elif "cp" in fl:
                if cp is None or "bf16" in fl:
                    cp = os.path.join(models_dir, f)
            elif "tokenizer" in fl:
                if vocoder is None or "f16" in fl:
                    vocoder = os.path.join(models_dir, f)
    return talker, cp, vocoder


def parse_performance(output: str) -> dict:
    metrics = {}
    m = re.search(r"Prefill:\s+(\d+) tokens in ([\d.]+) ms\s+\(([\d.]+) tok/s\)", output)
    if m:
        metrics["prefill_tokens"] = int(m.group(1))
        metrics["prefill_ms"] = float(m.group(2))
        metrics["prefill_tok_s"] = float(m.group(3))

    m = re.search(r"Decode:\s+(\d+) frames in ([\d.]+) ms\s+\(([\d.]+) frames/s\)", output)
    if m:
        metrics["decode_frames"] = int(m.group(1))
        metrics["decode_ms"] = float(m.group(2))
        metrics["decode_fps"] = float(m.group(3))

    m = re.search(r"Talker:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["talker_ms"] = float(m.group(1))
        metrics["talker_ms_per_frame"] = float(m.group(2))

    m = re.search(r"CP:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["cp_ms"] = float(m.group(1))
        metrics["cp_ms_per_frame"] = float(m.group(2))

    m = re.search(r"Head:\s+([\d.]+) ms total\s+\(([\d.]+) ms/frame\)", output)
    if m:
        metrics["head_ms"] = float(m.group(1))
        metrics["head_ms_per_frame"] = float(m.group(2))

    m = re.search(r"Real-time factor:\s+([\d.]+)x\s+\(([\d.]+)s audio in ([\d.]+)s\)", output)
    if m:
        metrics["rtf"] = float(m.group(1))
        metrics["audio_s"] = float(m.group(2))
        metrics["wall_s"] = float(m.group(3))

    m = re.search(r"Generation complete:\s+(\d+) frames", output)
    if m:
        metrics["total_frames"] = int(m.group(1))

    return metrics


def synthesize(text, language, max_tokens, ref_audio_path=None,
               ref_text=None, ref_codes_path=None,
               temp=0.9, top_k=50, top_p=1.0, rep_penalty=1.05,
               cp_temp=0.9, cp_top_k=50, greedy=False, seed=None,
               streaming_text=False):
    """Run llama-qwen3tts and return (wav_bytes, metrics_dict, raw_log)."""
    binary = find_binary()
    talker, cp, vocoder = find_models()

    if not binary or not talker or not cp:
        return None, {}, "ERROR: Missing binary or model files."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        binary,
        "--model-talker", talker,
        "--model-cp", cp,
        "--text", text,
        "--output", output_path,
        "--language", language,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--rep-penalty", str(rep_penalty),
        "--cp-temp", str(cp_temp),
        "--cp-top-k", str(cp_top_k),
    ]
    if vocoder:
        cmd += ["--model-vocoder", vocoder]
    if ref_audio_path:
        cmd += ["--ref-audio", ref_audio_path]
    if ref_text:
        cmd += ["--ref-text", ref_text]
    if ref_codes_path:
        cmd += ["--ref-codes", ref_codes_path]
    if greedy:
        cmd += ["--greedy"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if streaming_text:
        cmd += ["--streaming-text"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        log = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, {}, "ERROR: Process timed out after 600s"
    except Exception as e:
        return None, {}, f"ERROR: {e}"

    metrics = parse_performance(log)

    wav_bytes = None
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 44:
        with open(output_path, "rb") as f:
            wav_bytes = f.read()

    try:
        os.unlink(output_path)
    except OSError:
        pass

    return wav_bytes, metrics, log


def wav_duration(wav_bytes):
    """Get duration in seconds from WAV bytes."""
    try:
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def main():
    st.set_page_config(
        page_title="Qwen3-TTS",
        page_icon="🔊",
        layout="wide",
    )

    st.title("Qwen3-TTS — Text to Speech")
    st.caption("Powered by llama.cpp | 12Hz codec | 0.6B parameters")

    binary = find_binary()
    talker, cp, vocoder = find_models()

    if not binary:
        st.error("llama-qwen3tts binary not found. Build it first:\n\n"
                 "```\ncmake --build build --target llama-qwen3tts\n```")
        return
    if not talker or not cp:
        st.error("Model files not found in `models/` directory.\n\n"
                 "Expected: `*talker*.gguf`, `*cp*.gguf`, `*tokenizer*.gguf`")
        return

    with st.sidebar:
        st.header("Model Info")
        st.text(f"Talker:  {os.path.basename(talker)}")
        st.text(f"CP:      {os.path.basename(cp)}")
        if vocoder:
            st.text(f"Vocoder: {os.path.basename(vocoder)}")
        else:
            st.warning("Vocoder not found - no audio output")
        st.divider()

        st.header("Features")
        st.markdown(
            "- 10 languages + auto\n"
            "- Voice cloning (x-vector & ICL)\n"
            "- Streaming text mode\n"
            "- 12 Hz codec, 24 kHz audio"
        )
        st.divider()

        st.header("Generation")
        max_tokens = st.slider("Max frames", 10, 2000, 300,
                               help="Max output frames (12 frames ≈ 1 second)")
        st.caption(f"≈ {max_tokens / 12:.1f}s max audio")
        streaming_text = st.checkbox("Streaming text mode",
                                     help="Feed text progressively instead of all at once in prefill")
        st.divider()

        st.header("Sampling")
        greedy_mode = st.checkbox("Greedy decoding", help="Deterministic argmax (overrides temp/top-k)")
        temp_val = st.slider("Temperature", 0.0, 2.0, 0.9, 0.05,
                             help="Higher = more random. 0 = greedy", disabled=greedy_mode)
        topk_val = st.slider("Top-k", 0, 200, 50,
                             help="Keep top-k candidates. 0 = disabled", disabled=greedy_mode)
        topp_val = st.slider("Top-p (nucleus)", 0.0, 1.0, 1.0, 0.05,
                             help="Cumulative probability cutoff", disabled=greedy_mode)
        rep_val = st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01,
                            help="Penalize recent tokens. 1.0 = off", disabled=greedy_mode)

        with st.expander("Code Predictor sampling"):
            cp_temp_val = st.slider("CP Temperature", 0.0, 2.0, 0.9, 0.05,
                                    disabled=greedy_mode, key="cp_temp_slider")
            cp_topk_val = st.slider("CP Top-k", 0, 200, 50,
                                    disabled=greedy_mode, key="cp_topk_slider")

        seed_val = st.number_input("Seed (-1 = random)", value=-1, step=1)
        seed_use = seed_val if seed_val >= 0 else None

    tab_basic, tab_clone, tab_multi, tab_bench = st.tabs([
        "Basic TTS", "Voice Cloning", "Multilingual", "Benchmark"
    ])

    # ── Tab 1: Basic TTS ──────────────────────────────────────────────
    with tab_basic:
        st.subheader("Text to Speech")

        col_input, col_output = st.columns([1, 1])

        with col_input:
            language = st.selectbox("Language", LANGUAGES, index=0, key="basic_lang")
            default_text = EXAMPLE_TEXTS.get(language, "")
            text = st.text_area("Text to synthesize", value=default_text,
                                height=120, key="basic_text")
            go = st.button("Synthesize", type="primary", key="basic_go",
                           use_container_width=True)

        with col_output:
            if go and text.strip():
                with st.spinner("Generating speech..."):
                    wav, metrics, log = synthesize(
                        text, language, max_tokens,
                        temp=temp_val, top_k=topk_val, top_p=topp_val,
                        rep_penalty=rep_val, cp_temp=cp_temp_val,
                        cp_top_k=cp_topk_val, greedy=greedy_mode,
                        seed=seed_use, streaming_text=streaming_text)

                if wav:
                    st.audio(wav, format="audio/wav")
                    dur = wav_duration(wav)
                    st.caption(f"Duration: {dur:.2f}s | Size: {len(wav):,} bytes")
                    st.download_button("Download WAV", wav, "output.wav",
                                       "audio/wav", use_container_width=True)

                    if metrics:
                        st.divider()
                        cols = st.columns(4)
                        cols[0].metric("Prefill", f"{metrics.get('prefill_tok_s', 0):.0f} tok/s")
                        cols[1].metric("Decode", f"{metrics.get('decode_fps', 0):.1f} fr/s")
                        cols[2].metric("RTF", f"{metrics.get('rtf', 0):.2f}x")
                        cols[3].metric("Frames", f"{metrics.get('total_frames', 0)}")

                    with st.expander("Raw log"):
                        st.code(log, language="text")
                else:
                    st.error("Synthesis failed")
                    with st.expander("Log"):
                        st.code(log, language="text")
            elif go:
                st.warning("Please enter some text.")

    # ── Tab 2: Voice Cloning ──────────────────────────────────────────
    with tab_clone:
        st.subheader("Voice Cloning")
        st.markdown("""
        Upload reference audio to clone a speaker's voice.
        - **X-vector mode**: Extracts a 1024-dim speaker embedding from reference audio using
          the built-in ECAPA-TDNN encoder. Captures global speaker characteristics (timbre, pitch).
        - **ICL mode**: Also feeds reference audio's codec tokens and transcript into the model.
          Higher fidelity but requires preprocessing with `extract_ref_codes.py`.
        """)

        col_ref, col_tgt = st.columns(2)

        with col_ref:
            st.markdown("**Reference**")
            ref_audio = st.file_uploader("Reference audio (WAV)", type=["wav"],
                                         key="clone_ref_audio")
            clone_mode = st.radio("Cloning mode", ["X-vector", "ICL"],
                                  key="clone_mode",
                                  help="ICL requires ref text and codec codes file")

            ref_text_input = None
            ref_codes_file = None
            if clone_mode == "ICL":
                ref_text_input = st.text_input("Reference transcript", key="clone_ref_text")
                ref_codes_file = st.file_uploader("Codec codes file (.txt)",
                                                   type=["txt"],
                                                   key="clone_ref_codes",
                                                   help="Generated by extract_ref_codes.py")
                with st.expander("How to generate codec codes"):
                    st.code(
                        "python ../../tools/tts/extract_ref_codes.py \\\n"
                        "    --ref-audio speaker.wav \\\n"
                        "    --output ref_codes.txt",
                        language="bash"
                    )
                    st.caption("Requires: pip install qwen-tts transformers torch torchaudio")

        with col_tgt:
            st.markdown("**Target**")
            clone_lang = st.selectbox("Language", LANGUAGES, index=0, key="clone_lang")
            clone_text = st.text_area("Text to synthesize",
                                      value="This sentence will be spoken in the cloned voice.",
                                      height=100, key="clone_text")
            clone_go = st.button("Clone & Synthesize", type="primary",
                                 key="clone_go", use_container_width=True)

        if clone_go and ref_audio and clone_text.strip():
            ref_audio_path = None
            ref_codes_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(ref_audio.getvalue())
                    ref_audio_path = tmp.name

                if ref_codes_file:
                    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False,
                                                     mode="w") as tmp:
                        tmp.write(ref_codes_file.getvalue().decode("utf-8"))
                        ref_codes_path = tmp.name

                with st.spinner(f"Cloning voice ({clone_mode} mode)..."):
                    wav, metrics, log = synthesize(
                        clone_text, clone_lang, max_tokens,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text_input if clone_mode == "ICL" else None,
                        ref_codes_path=ref_codes_path if clone_mode == "ICL" else None,
                        temp=temp_val, top_k=topk_val, top_p=topp_val,
                        rep_penalty=rep_val, cp_temp=cp_temp_val,
                        cp_top_k=cp_topk_val, greedy=greedy_mode,
                        seed=seed_use, streaming_text=streaming_text,
                    )

                if wav:
                    st.audio(wav, format="audio/wav")
                    dur = wav_duration(wav)
                    st.caption(f"Duration: {dur:.2f}s | Mode: {clone_mode}")
                    st.download_button("Download", wav, "cloned.wav", "audio/wav")
                    if metrics:
                        cols = st.columns(4)
                        cols[0].metric("Prefill", f"{metrics.get('prefill_tok_s', 0):.0f} tok/s")
                        cols[1].metric("Decode", f"{metrics.get('decode_fps', 0):.1f} fr/s")
                        cols[2].metric("RTF", f"{metrics.get('rtf', 0):.2f}x")
                        cols[3].metric("Frames", f"{metrics.get('total_frames', 0)}")
                    with st.expander("Log"):
                        st.code(log, language="text")
                else:
                    st.error("Cloning failed")
                    with st.expander("Log"):
                        st.code(log, language="text")
            finally:
                if ref_audio_path:
                    try:
                        os.unlink(ref_audio_path)
                    except OSError:
                        pass
                if ref_codes_path:
                    try:
                        os.unlink(ref_codes_path)
                    except OSError:
                        pass
        elif clone_go:
            st.warning("Please upload reference audio and enter text.")

    # ── Tab 3: Multilingual ───────────────────────────────────────────
    with tab_multi:
        st.subheader("Multilingual Synthesis")
        st.markdown("Generate speech in multiple languages side by side.")

        selected_langs = st.multiselect(
            "Select languages",
            [l for l in LANGUAGES if l != "auto"],
            default=["english", "french", "chinese"],
            key="multi_langs"
        )

        custom_texts = {}
        for lang in selected_langs:
            custom_texts[lang] = st.text_input(
                f"{lang.capitalize()} text",
                value=EXAMPLE_TEXTS.get(lang, "Hello world."),
                key=f"multi_text_{lang}"
            )

        multi_go = st.button("Generate All", type="primary", key="multi_go",
                             use_container_width=True)

        if multi_go and selected_langs:
            cols = st.columns(min(len(selected_langs), 3))
            for i, lang in enumerate(selected_langs):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(f"**{lang.capitalize()}**")
                    with st.spinner(f"Generating {lang}..."):
                        wav, metrics, log = synthesize(
                            custom_texts[lang], lang, max_tokens,
                            temp=temp_val, top_k=topk_val, top_p=topp_val,
                            rep_penalty=rep_val, cp_temp=cp_temp_val,
                            cp_top_k=cp_topk_val, greedy=greedy_mode,
                            seed=seed_use, streaming_text=streaming_text,
                        )
                    if wav:
                        st.audio(wav, format="audio/wav")
                        dur = wav_duration(wav)
                        st.caption(f"{dur:.1f}s | {metrics.get('rtf', 0):.2f}x RTF")
                    else:
                        st.error(f"Failed for {lang}")

    # ── Tab 4: Benchmark ──────────────────────────────────────────────
    with tab_bench:
        st.subheader("Performance Benchmark")
        st.markdown("""
        Measure throughput across different text lengths.

        **Key metrics:**
        - **Prefill tok/s** — input processing speed
        - **Decode fr/s** — output frame generation rate (12 = real-time)
        - **RTF** — real-time factor (>1.0 = faster than real-time)
        - **Talker ms/fr** — Talker (28 layers) time per frame
        - **CP ms/fr** — Code Predictor (5 layers x 15 steps) time per frame
        """)

        bench_frames = st.slider("Frames per test", 10, 200, 30, key="bench_frames")

        bench_texts = [
            ("Short", "Hello world."),
            ("Medium", "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."),
            ("Long", "In the beginning there was silence, then came the first whisper of sound. "
                     "Birds began to sing their morning melodies, each note a testament. "
                     "The sun rose slowly over the horizon, painting the sky."),
        ]

        bench_go = st.button("Run Benchmark", type="primary", key="bench_go",
                             use_container_width=True)

        if bench_go:
            results = []
            progress = st.progress(0)
            for i, (label, text) in enumerate(bench_texts):
                progress.progress((i) / len(bench_texts), f"Running: {label}...")
                wav, metrics, log = synthesize(
                    text, "english", bench_frames,
                    temp=temp_val, top_k=topk_val, top_p=topp_val,
                    rep_penalty=rep_val, cp_temp=cp_temp_val,
                    cp_top_k=cp_topk_val, greedy=greedy_mode,
                    seed=seed_use)
                if metrics:
                    results.append({"Test": label, **metrics})
                else:
                    results.append({"Test": label, "error": True})
            progress.progress(1.0, "Done!")

            if results:
                st.divider()
                header_cols = st.columns(7)
                headers = ["Test", "Prefill", "Prefill", "Decode", "Talker", "CP", "RTF"]
                units = ["", "tok/s", "ms", "fr/s", "ms/fr", "ms/fr", ""]
                for col, h, u in zip(header_cols, headers, units):
                    col.markdown(f"**{h}**" + (f"  \n*{u}*" if u else ""))

                for r in results:
                    if r.get("error"):
                        st.error(f"{r['Test']}: failed")
                        continue
                    row_cols = st.columns(7)
                    row_cols[0].write(r["Test"])
                    row_cols[1].write(f"{r.get('prefill_tok_s', 0):.0f}")
                    row_cols[2].write(f"{r.get('prefill_ms', 0):.0f}")
                    row_cols[3].write(f"{r.get('decode_fps', 0):.1f}")
                    row_cols[4].write(f"{r.get('talker_ms_per_frame', 0):.1f}")
                    row_cols[5].write(f"{r.get('cp_ms_per_frame', 0):.1f}")
                    row_cols[6].write(f"{r.get('rtf', 0):.2f}x")


if __name__ == "__main__":
    main()

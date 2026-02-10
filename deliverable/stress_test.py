#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "soundfile",
#     "openai",
# ]
# ///
"""
Stress test for LFM2.5-Audio server.
Ramps up requests per second (RPS) for TTS, ASR, and FUNC modes.
"""

import argparse
import base64
import concurrent.futures
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from openai import OpenAI


@dataclass
class RequestResult:
    mode: str
    success: bool
    latency: float  # total request time in seconds
    ttft: float | None = None  # time to first token
    error: str | None = None
    text_tokens: int = 0
    audio_samples: int = 0


@dataclass
class RPSStageResult:
    target_rps: float
    mode: str
    results: list[RequestResult] = field(default_factory=list)

    @property
    def actual_rps(self):
        if not self.results:
            return 0
        total_time = max(r.latency for r in self.results) if self.results else 1
        return len(self.results) / total_time if total_time > 0 else 0

    @property
    def success_count(self):
        return sum(1 for r in self.results if r.success)

    @property
    def fail_count(self):
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self):
        return self.success_count / len(self.results) * 100 if self.results else 0

    @property
    def latencies(self):
        return [r.latency for r in self.results if r.success]

    @property
    def ttfts(self):
        return [r.ttft for r in self.results if r.success and r.ttft is not None]


# Default prompts for each mode
TTS_PROMPTS = [
    "Hello, how are you doing today?",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the audio stress test.",
    "This is a sample sentence for text to speech synthesis.",
    "Testing the server under increasing load.",
]

FUNC_PROMPTS = [
    'What is the weather in San Francisco?',
    'Book a meeting for tomorrow at 3pm.',
    'Search for flights from New York to London.',
    'Set a reminder to buy groceries at 5pm.',
    'Calculate the distance from Paris to Berlin.',
]

TTS_SYSTEM = "Perform TTS. Use the US male voice."
ASR_SYSTEM = "Perform ASR."
FUNC_SYSTEM = "Respond in function calls."


def single_request(base_url: str, mode: str, system_prompt: str,
                   user_content, max_tokens: int) -> RequestResult:
    """Execute a single request and return the result."""
    client = OpenAI(base_url=base_url, api_key="dummy")

    modalities = ["audio"] if "TTS" in system_prompt else ["text"]

    t_start = time.time()
    ttft = None
    text_tokens = 0
    audio_samples = 0

    try:
        stream = client.chat.completions.create(
            model="",
            modalities=modalities,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            max_tokens=max_tokens,
        )

        completed = False
        for chunk in stream:
            if chunk.choices[0].finish_reason == "stop":
                completed = True
                break

            delta = chunk.choices[0].delta

            if text := delta.content:
                if ttft is None:
                    ttft = time.time() - t_start
                text_tokens += 1

            if hasattr(delta, "audio") and delta.audio and "data" in delta.audio:
                if ttft is None:
                    ttft = time.time() - t_start
                pcm_bytes = base64.b64decode(delta.audio["data"])
                audio_samples += len(pcm_bytes) // 2  # int16 = 2 bytes

        latency = time.time() - t_start

        if not completed:
            return RequestResult(
                mode=mode, success=False, latency=latency,
                error="Server disconnected before completion",
            )

        return RequestResult(
            mode=mode, success=True, latency=latency, ttft=ttft,
            text_tokens=text_tokens, audio_samples=audio_samples,
        )

    except Exception as e:
        latency = time.time() - t_start
        return RequestResult(
            mode=mode, success=False, latency=latency,
            error=str(e)[:200],
        )


def prepare_asr_content(wav_file: str):
    """Load a WAV file and return OpenAI-compatible audio content."""
    with open(wav_file, "rb") as f:
        wav_data = f.read()
    encoded = base64.b64encode(wav_data).decode("utf-8")
    return [{"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}]


def run_rps_stage(base_url: str, mode: str, system_prompt: str,
                  contents: list, target_rps: float, duration: float,
                  max_tokens: int) -> RPSStageResult:
    """Run a single RPS stage: fire requests at the target rate for `duration` seconds."""
    stage = RPSStageResult(target_rps=target_rps, mode=mode)
    interval = 1.0 / target_rps if target_rps > 0 else 1.0
    total_requests = max(1, int(target_rps * duration))

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_requests + 4) as pool:
        t0 = time.time()
        for i in range(total_requests):
            # Schedule request at the right time
            scheduled = t0 + i * interval
            now = time.time()
            if scheduled > now:
                time.sleep(scheduled - now)

            content = contents[i % len(contents)]
            futures.append(
                pool.submit(single_request, base_url, mode, system_prompt, content, max_tokens)
            )

        # Wait for all to complete
        for fut in concurrent.futures.as_completed(futures):
            stage.results.append(fut.result())

    return stage


def print_stage_report(stage: RPSStageResult):
    """Print a summary for one RPS stage."""
    lats = stage.latencies
    ttfts = stage.ttfts

    print(f"  Target RPS: {stage.target_rps:>6.1f}  |  "
          f"Requests: {len(stage.results):>4}  |  "
          f"OK: {stage.success_count:>4}  |  "
          f"Fail: {stage.fail_count:>3}  |  "
          f"Success: {stage.success_rate:>5.1f}%")

    if lats:
        print(f"    Latency  — min: {min(lats):.3f}s  "
              f"avg: {statistics.mean(lats):.3f}s  "
              f"p50: {statistics.median(lats):.3f}s  "
              f"p95: {sorted(lats)[int(len(lats) * 0.95)]:.3f}s  "
              f"max: {max(lats):.3f}s")
    if ttfts:
        print(f"    TTFT     — min: {min(ttfts):.3f}s  "
              f"avg: {statistics.mean(ttfts):.3f}s  "
              f"p50: {statistics.median(ttfts):.3f}s  "
              f"max: {max(ttfts):.3f}s")

    if stage.fail_count > 0:
        errors = [r.error for r in stage.results if not r.success and r.error]
        unique = set(errors)
        for e in list(unique)[:3]:
            print(f"    Error: {e}")


def run_stress_test(base_url: str, modes: list[str], wav_file: str | None,
                    rps_stages: list[float], duration: float, max_tokens: int):
    """Run the full stress test across modes and RPS levels."""
    # Prepare content for each mode
    mode_configs: dict[str, tuple[str, list]] = {}

    if "tts" in modes:
        mode_configs["tts"] = (TTS_SYSTEM, TTS_PROMPTS)
    if "asr" in modes:
        if not wav_file:
            print("ERROR: ASR mode requires --wav argument", file=sys.stderr)
            sys.exit(1)
        asr_content = prepare_asr_content(wav_file)
        mode_configs["asr"] = (ASR_SYSTEM, [asr_content])
    if "func" in modes:
        mode_configs["func"] = (FUNC_SYSTEM, FUNC_PROMPTS)

    all_stages: list[RPSStageResult] = []

    for mode, (sys_prompt, contents) in mode_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  STRESS TEST: {mode.upper()}")
        print(f"{'=' * 60}")

        for rps in rps_stages:
            print(f"\n--- {mode.upper()} @ {rps} RPS (duration: {duration}s) ---")
            stage = run_rps_stage(base_url, mode, sys_prompt, contents, rps, duration, max_tokens)
            print_stage_report(stage)
            all_stages.append(stage)

            # If success rate drops below 50%, stop escalating for this mode
            if stage.success_rate < 50:
                print(f"  >> Success rate below 50%, stopping RPS ramp for {mode.upper()}")
                break

    # Final summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Mode':<6} {'RPS':>6} {'Total':>6} {'OK':>6} {'Fail':>6} {'Rate':>7} {'Avg Lat':>8} {'Avg TTFT':>9}")
    print("-" * 60)
    for s in all_stages:
        avg_lat = statistics.mean(s.latencies) if s.latencies else float("nan")
        avg_ttft = statistics.mean(s.ttfts) if s.ttfts else float("nan")
        print(f"{s.mode:<6} {s.target_rps:>6.1f} {len(s.results):>6} "
              f"{s.success_count:>6} {s.fail_count:>6} {s.success_rate:>6.1f}% "
              f"{avg_lat:>7.3f}s {avg_ttft:>8.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Stress test for LFM2.5-Audio server with increasing RPS"
    )
    parser.add_argument(
        "--modes", type=str, default="tts,asr,func",
        help="Comma-separated list of modes to test: tts,asr,func (default: all)",
    )
    parser.add_argument(
        "--wav", type=str,
        help="Path to input WAV file (required for ASR mode)",
    )
    parser.add_argument(
        "--rps", type=str, default="1,2,4,8,16,32,64",
        help="Comma-separated RPS stages to ramp through (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration in seconds for each RPS stage (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per request (default: 512)",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8080/v1",
        help="Server base URL (default: http://127.0.0.1:8080/v1)",
    )

    args = parser.parse_args()

    modes = [m.strip().lower() for m in args.modes.split(",")]
    rps_stages = [float(r.strip()) for r in args.rps.split(",")]

    if "asr" in modes and not args.wav:
        parser.error("--wav is required when testing ASR mode")

    print("Stress Test Configuration:")
    print(f"  Server:     {args.base_url}")
    print(f"  Modes:      {modes}")
    print(f"  RPS stages: {rps_stages}")
    print(f"  Duration:   {args.duration}s per stage")
    print(f"  Max tokens: {args.max_tokens}")

    run_stress_test(
        base_url=args.base_url,
        modes=modes,
        wav_file=args.wav,
        rps_stages=rps_stages,
        duration=args.duration,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

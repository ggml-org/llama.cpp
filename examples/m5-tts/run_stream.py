#!/usr/bin/env python3
"""Run the official M5Onnx streaming loop with an optional external backbone."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import soundfile as sf

from pipe_backbone import PipeBackbone
from pipe_depth import PipeDepth


class TimedSession:
    def __init__(self, session):
        self.session = session
        self.seconds = 0.0
        self.calls = 0

    def run(self, outputs, feed):
        started = time.perf_counter()
        result = self.session.run(outputs, feed)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m5-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--depth-weights", type=Path)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--backbone-command", nargs=argparse.REMAINDER)
    commands.add_argument("--depth-command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if bool(args.depth_command) != bool(args.depth_weights):
        parser.error("--depth-command and --depth-weights must be supplied together")

    sys.path.insert(0, str(args.m5_source.resolve()))
    from infer_onnx import M5Onnx

    model = M5Onnx(args.model, threads=args.threads, verify=True)
    with ExitStack() as stack:
        backbone = None
        if args.backbone_command:
            backbone = stack.enter_context(PipeBackbone(args.backbone_command))
            model.sessions["backbone"] = backbone
        if args.depth_command:
            model.sessions["depth_frame"] = stack.enter_context(PipeDepth(args.depth_command, args.depth_weights))
        sessions = {name: TimedSession(session) for name, session in model.sessions.items()}
        model.sessions.update(sessions)
        with args.output.open("xb") as file:
            with sf.SoundFile(file, mode="w", samplerate=model.sample_rate, channels=1,
                              subtype="PCM_16", format="WAV") as wav:
                for pcm in model.generate(args.text, args.prompt_wav, args.prompt_text,
                                          seed=args.seed, max_frames=args.max_frames, stream=True):
                    wav.write(pcm)
    model.stats["stages"] = {
        name: {"elapsed_s": session.seconds, "calls": session.calls}
        for name, session in sessions.items()
    }
    if backbone:
        model.stats["backbone_provider"] = "external llama.cpp"
        model.stats["session_providers"]["backbone"] = ["external llama.cpp"]
        model.stats["provider_options"].pop("backbone", None)
    if args.depth_command:
        model.stats["session_providers"]["depth_frame"] = ["external llama.cpp core and host sampling"]
        model.stats["provider_options"].pop("depth_frame", None)
    print(json.dumps(model.stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

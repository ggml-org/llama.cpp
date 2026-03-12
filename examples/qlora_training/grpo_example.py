#!/usr/bin/env python3
"""
grpo_example.py — Minimal GRPO training loop using llama-finetune-qlora --grpo-mode

Demonstrates the IPC protocol between the Python driver and the C++ subprocess.
No external dependencies required — only Python stdlib.

Usage:
    python3 grpo_example.py \
        --model   /path/to/model-q4_k_m.gguf \
        --lora-out /path/to/output-adapter.gguf \
        [--lora    /path/to/resume-adapter.gguf] \
        [--binary  /path/to/llama-finetune-qlora] \
        [--n-steps 200] \
        [--n-gen   8] \
        [--rank    16]

IPC Protocol (stdout from C++ process):
    [QLORA:READY]               — process initialised
    [QLORA:PROMPT_REQ:<step>]   — C++ requests a prompt for step N
    [QLORA:GEN:<k>/<n>] <text>  — one generation (newlines escaped as \\n)
    [QLORA:REWARD_REQ:<n>]      — C++ requests N reward scores
    [QLORA:PROGRESS] step=X/Y loss=Z epoch=A/B
    [QLORA:CHECKPOINT] <path>
    [QLORA:DONE] final_loss=X
    [QLORA:ERROR] <message>

Python → C++ stdin:
    PROMPT <escaped_text>
    REWARD <r1> <r2> ... <rN>    (advantages, 0..1 range)
    STOP                         (request graceful shutdown)
"""

import argparse
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("grpo_example")

# ──────────────────────────────────────────────────────────────────────────────
# IPC helpers
# ──────────────────────────────────────────────────────────────────────────────

_IPC_RE = re.compile(r"^\[QLORA:([A-Z_]+)(?::([^\]]*))?\](.*)$")


def escape(text: str) -> str:
    """Escape newlines and backslashes for single-line IPC transport."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def unescape(text: str) -> str:
    """Reverse of escape()."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            c = text[i + 1]
            if c == "n":
                out.append("\n")
            elif c == "r":
                out.append("\r")
            elif c == "\\":
                out.append("\\")
            else:
                out.append(c)
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def parse_ipc(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an IPC line into (msg_type, seq, payload).
    Returns None for non-IPC lines (model output, log lines, etc.).
    """
    m = _IPC_RE.match(line.strip())
    if not m:
        return None
    return m.group(1), (m.group(2) or ""), m.group(3).strip()


def read_ipc(proc: subprocess.Popen, timeout: float = 120.0) -> Optional[Tuple[str, str, str]]:
    """
    Read lines from proc.stdout until an IPC message arrives.
    Non-IPC lines (model output, C++ logs leaked to stdout) are printed.
    Returns None on EOF.
    Raises TimeoutError if nothing arrives within `timeout` seconds.
    """
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"No IPC message within {timeout:.0f}s")

        line = proc.stdout.readline()
        if not line:
            return None  # EOF

        line = line.rstrip("\n")
        parsed = parse_ipc(line)
        if parsed:
            return parsed
        # Non-IPC — C++ sometimes leaks timing/debug lines to stdout.
        # Print them so the user can see what's happening.
        print(f"  [cpp] {line}", file=sys.stderr)


def write_cmd(proc: subprocess.Popen, cmd: str):
    """Write one command line to the subprocess stdin."""
    assert proc.stdin is not None
    try:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        raise RuntimeError("C++ subprocess stdin closed — did it crash?")


def wait_for(proc: subprocess.Popen, expected: str, timeout: float = 120.0) -> Tuple[str, str, str]:
    """Block until the expected IPC message type arrives."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for [{expected}]")
        parsed = read_ipc(proc, timeout=remaining)
        if parsed is None:
            raise RuntimeError(f"Subprocess exited before sending [{expected}]")
        msg_type, seq, payload = parsed
        if msg_type == expected:
            return msg_type, seq, payload
        log.debug("Ignoring unexpected IPC (%s) while waiting for %s", msg_type, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Advantage normalisation (GRPO)
# ──────────────────────────────────────────────────────────────────────────────

def normalise_rewards(rewards: List[float]) -> List[float]:
    """
    Group-relative advantage normalisation: subtract mean, divide by std.
    Clipped to [0, 1] so the C++ side always receives values in that range.

    All-equal rewards → uniform 0.5 (no signal, but no NaN either).
    """
    if len(rewards) == 0:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance) if variance > 1e-8 else 1.0

    normalised = [(r - mean) / std for r in rewards]
    # Shift to [0,1]: z-scores typically lie in [-3, +3]
    clipped = [max(0.0, min(1.0, 0.5 + z / 6.0)) for z in normalised]
    return clipped


# ──────────────────────────────────────────────────────────────────────────────
# Example prompt / reward providers
# ──────────────────────────────────────────────────────────────────────────────

# Replace these with your own logic.

_EXAMPLE_PROMPTS = [
    "Explain the concept of gradient descent in one sentence.",
    "What is the capital of France?",
    "Write a haiku about machine learning.",
    "Describe the difference between SFT and RLHF.",
    "What does GRPO stand for?",
]


def get_prompt(step: int) -> str:
    """Return a prompt for the given training step (0-indexed)."""
    return _EXAMPLE_PROMPTS[step % len(_EXAMPLE_PROMPTS)]


def score_generations(prompt: str, generations: List[str]) -> List[float]:
    """
    Score a list of model generations for the given prompt.
    Returns a list of raw reward scores (any numeric range; will be normalised).

    This example uses a trivial heuristic: longer, more varied responses
    score higher.  Replace with your actual reward model / verifier.
    """
    scores = []
    for gen in generations:
        words = gen.split()
        # Simple heuristics: length + lexical diversity
        length_score = min(1.0, len(words) / 50.0)
        vocab_score  = min(1.0, len(set(words)) / max(1, len(words)))
        scores.append(0.6 * length_score + 0.4 * vocab_score)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Main GRPO loop
# ──────────────────────────────────────────────────────────────────────────────

def run_grpo(args: argparse.Namespace):
    # Resolve binary
    binary = Path(args.binary)
    if not binary.exists():
        log.error("Binary not found: %s", binary)
        sys.exit(1)

    # Build command
    cmd = [
        str(binary),
        "--model",          args.model,
        "--lora-out",       args.lora_out,
        "--lora-rank",      str(args.rank),
        "--lora-alpha",     str(args.rank // 2),
        "-c",               str(args.ctx_size),
        "-b",               str(args.ctx_size),
        "-ub",              "512",
        "-ngl",             str(args.ngl),
        "-lr",              str(args.lr),
        "--seed",           str(args.seed),
        "--grad-checkpoint","48",
        "--shuffle-dataset",
        "--grpo-mode",
        "--n-gen",          str(args.n_gen),
        "--n-steps",        str(args.n_steps),
        "--grpo-temp",      str(args.temperature),
        "--grpo-max-tokens",str(args.max_tokens),
    ]

    if args.lora:
        cmd += ["--lora", args.lora]

    if args.save_every > 0:
        cmd += ["--save-every", str(args.save_every)]

    log.info("Launching: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,          # C++ debug/timing logs go directly to our stderr
        text=True,
        bufsize=1,
    )

    try:
        _grpo_loop(proc, args)
    except KeyboardInterrupt:
        log.info("Interrupted — requesting graceful stop")
        try:
            write_cmd(proc, "STOP")
        except Exception:
            pass
    except Exception as e:
        log.error("GRPO loop error: %s", e)
        proc.kill()
        raise
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait(timeout=30)
        if rc not in (0, None):
            log.warning("Subprocess exited with code %d", rc)


def _grpo_loop(proc: subprocess.Popen, args: argparse.Namespace):
    # ── Wait for READY ──────────────────────────────────────────────────────
    log.info("Waiting for subprocess to initialise (model load can take a minute)…")
    wait_for(proc, "READY", timeout=300)
    log.info("Subprocess ready.")

    current_prompt: str = ""
    generations: List[str] = []
    step = 0

    while True:
        parsed = read_ipc(proc, timeout=600)
        if parsed is None:
            log.info("Subprocess exited (EOF).")
            break

        msg_type, seq, payload = parsed

        # ── PROMPT_REQ ──────────────────────────────────────────────────────
        if msg_type == "PROMPT_REQ":
            step = int(seq) if seq else step + 1
            current_prompt = get_prompt(step - 1)
            generations = []
            log.debug("Step %d — sending prompt: %s", step, current_prompt[:60])
            write_cmd(proc, f"PROMPT {escape(current_prompt)}")

        # ── GEN ─────────────────────────────────────────────────────────────
        elif msg_type == "GEN":
            # seq = "k/n"
            parts = seq.split("/")
            k = int(parts[0])
            n = int(parts[1]) if len(parts) > 1 else args.n_gen
            text = unescape(payload)
            generations.append(text)
            log.debug("  Generation %d/%d: %s…", k, n, text[:60].replace("\n", "↵"))

        # ── REWARD_REQ ──────────────────────────────────────────────────────
        elif msg_type == "REWARD_REQ":
            n_expected = int(seq) if seq else len(generations)
            if len(generations) != n_expected:
                log.warning(
                    "REWARD_REQ asked for %d rewards but collected %d generations",
                    n_expected, len(generations),
                )

            raw_rewards = score_generations(current_prompt, generations)
            advantages  = normalise_rewards(raw_rewards)

            reward_str = " ".join(f"{a:.6f}" for a in advantages)
            log.debug("  Rewards (raw): %s", [f"{r:.3f}" for r in raw_rewards])
            log.debug("  Advantages:    %s", [f"{a:.3f}" for a in advantages])
            write_cmd(proc, f"REWARD {reward_str}")

        # ── PROGRESS ────────────────────────────────────────────────────────
        elif msg_type == "PROGRESS":
            # Format: step=X/Y loss=Z epoch=A/B
            sm = re.search(r"step=(\d+)(?:/(\d+))?", payload)
            lm = re.search(r"loss=([\d.]+)", payload)
            step_str = f"{sm.group(1)}/{sm.group(2)}" if sm and sm.group(2) else (sm.group(1) if sm else "?")
            loss_str = lm.group(1) if lm else "?"
            print(f"  step {step_str}  loss {loss_str}", flush=True)

        # ── CHECKPOINT ──────────────────────────────────────────────────────
        elif msg_type == "CHECKPOINT":
            log.info("Checkpoint saved: %s", payload.strip())

        # ── DONE ────────────────────────────────────────────────────────────
        elif msg_type == "DONE":
            m = re.search(r"final_loss=([\d.]+)", payload)
            loss = m.group(1) if m else "?"
            log.info("Training complete. final_loss=%s", loss)
            break

        # ── ERROR ────────────────────────────────────────────────────────────
        elif msg_type == "ERROR":
            log.error("C++ process error: %s", payload.strip())
            raise RuntimeError(f"Training failed: {payload.strip()}")

        else:
            log.debug("Unknown IPC message: [%s] seq=%r payload=%r", msg_type, seq, payload)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # Default binary: build/bin/ relative to this script's repo root
    script_dir = Path(__file__).resolve().parent
    repo_root   = script_dir.parents[1]          # examples/qlora_training → llama.cpp root
    default_bin = repo_root / "build" / "bin" / "llama-finetune-qlora"

    p = argparse.ArgumentParser(
        description="Minimal GRPO training loop via llama-finetune-qlora --grpo-mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",       required=True,          help="Base GGUF model path")
    p.add_argument("--lora-out",    required=True,          help="Output adapter GGUF path")
    p.add_argument("--lora",        default=None,           help="Resume from existing adapter GGUF")
    p.add_argument("--binary",      default=str(default_bin), help="Path to llama-finetune-qlora binary")
    p.add_argument("--rank",        type=int,   default=16,   help="LoRA rank")
    p.add_argument("--n-steps",     type=int,   default=200,  help="Number of GRPO steps")
    p.add_argument("--n-gen",       type=int,   default=8,    help="Generations per prompt")
    p.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    p.add_argument("--ctx-size",    type=int,   default=4096, help="Context window")
    p.add_argument("--ngl",         type=int,   default=999,  help="GPU layers (-ngl)")
    p.add_argument("--temperature", type=float, default=0.8,  help="Sampling temperature")
    p.add_argument("--max-tokens",  type=int,   default=512,  help="Max tokens per generation")
    p.add_argument("--save-every",  type=int,   default=0,    help="Save checkpoint every N steps (0=off)")
    p.add_argument("--seed",        type=int,   default=42,   help="RNG seed")
    p.add_argument("--verbose",     action="store_true",     help="Enable DEBUG logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_grpo(args)

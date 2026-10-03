from __future__ import annotations

import hashlib

from .server_client import ServerClient


FIXTURE_TEXT = """You are continuing a technical discussion about a software project. Read the following notes and answer the final question accurately. The notes describe model serving, benchmark design, and Vulkan performance work. Preserve the distinctions between measured facts, hypotheses, and future work. Explain tradeoffs concisely and use examples when they clarify a decision.

The server has one active user and one slot. The configured context is large even when the populated prompt is short. MTP output throughput counts only final accepted tokens. A rejected draft token is useful diagnostic information but not output. The benchmark must retain identical model, sampler, cache type, context capacity, prompt tokens, and runtime settings when comparing two binaries. The benchmark records raw responses, server logs, timings, model identity, and the selected device.

For an optimization experiment, first establish a stable baseline. Then profile the full serving path, identify an actual bottleneck, make one small change, test correctness, and repeat the same workload. A faster kernel timer alone does not prove a serving improvement. Unsupported tensor layouts and devices must keep the upstream fallback path.

"""

QUESTION = "Summarize the safe procedure for evaluating one Vulkan optimization."
FIXTURE_TEXT_SHA256 = hashlib.sha256(FIXTURE_TEXT.encode("utf-8")).hexdigest()


def build_prompt_tokens(client: ServerClient, depth: int) -> tuple[list[int], str]:
    if depth <= 0:
        raise ValueError("prompt depth must be positive")
    repeats = max(1, (depth // 64) + 2)
    text = "\n".join(FIXTURE_TEXT for _ in range(repeats))
    tokens = client.tokenize(client.apply_template([
        {"role": "system", "content": text},
        {"role": "user", "content": QUESTION},
    ]), add_special=False)
    if len(tokens) < depth:
        raise RuntimeError(f"fixture only produced {len(tokens)} tokens for requested depth {depth}")
    suffix_tokens = client.tokenize(client.apply_template([
        {"role": "user", "content": QUESTION},
    ]), add_special=False)
    suffix = common_suffix(tokens, suffix_tokens)
    if len(suffix) >= depth:
        raise RuntimeError("chat-template suffix is too long for requested depth")
    selected = tokens[:depth - len(suffix)] + suffix
    token_hash = hashlib.sha256(",".join(str(token) for token in selected).encode("ascii")).hexdigest()
    return selected, token_hash


def common_suffix(left: list[int], right: list[int]) -> list[int]:
    count = 0
    while count < len(left) and count < len(right) and left[-count - 1] == right[-count - 1]:
        count += 1
    if count == 0:
        raise RuntimeError("unable to preserve the chat generation suffix")
    return left[-count:]

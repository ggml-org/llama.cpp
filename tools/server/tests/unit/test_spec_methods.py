import os
import re
from dataclasses import dataclass

import pytest
from utils import ServerProcess, download_file

# Runs each speculative method greedily and checks that it still drafts and accepts as expected.
# accept rate = accepted / drafted, accept len = 1 + accepted / verification steps (from /metrics).
# DFlash2 and LiLiCorr use --spec-type draft-dflash; the draft GGUF metadata selects the variant.
# Needs real models and a GPU. Use SPEC_METHODS=compact, nightly, all, or a comma-separated method list.

HF = "https://huggingface.co"
MODEL_REPO = "ggml-org/spec-for-testing-GGUF"
MODEL_REVISION = "a0736c8e98d3fe580352dacffa3fbd5e9c22a437"


@dataclass(frozen=True)
class SpecCase:
    name: str
    spec_type: str
    target: str
    draft: str | None  # None when the heads are part of the target (MTP)
    n_max: int
    min_accept_rate: float
    min_accept_len: float


CASES = [
    # Compact floors are the minimum of three identical GB10 calibration runs
    # against current master minus six acceptance-rate percentage points and
    # 0.2 accepted tokens per verification step.
    SpecCase(
        name="mtp",
        spec_type="draft-mtp",
        target="Qwen3.5-4B-MTP-Q4_K_M.gguf",
        draft=None,
        n_max=3,
        min_accept_rate=0.6040,
        min_accept_len=2.7766,
    ),
    SpecCase(
        name="dflash",
        spec_type="draft-dflash",
        target="Qwen3-4B-Q4_K_M.gguf",
        draft="dflash-Qwen3-4B-Q4_K_M.gguf",
        n_max=7,
        min_accept_rate=0.1404,
        min_accept_len=2.1687,
    ),
    SpecCase(
        name="dspark",
        spec_type="draft-dspark",
        target="Qwen3-4B-Q4_K_M.gguf",
        draft="dspark-Qwen3-4B-Q4_K_M.gguf",
        n_max=7,
        min_accept_rate=0.2767,
        min_accept_len=3.1246,
    ),
    SpecCase(
        name="eagle3",
        spec_type="draft-eagle3",
        target="Qwen3-4B-Q4_K_M.gguf",
        draft="eagle3-Qwen3-4B-Q8_0.gguf",
        n_max=3,
        min_accept_rate=0.2006,
        min_accept_len=1.5757,
    ),
    # Nightly floors retain the earlier three-point rate and 0.2 length margins.
    SpecCase(
        name="dflash2",
        spec_type="draft-dflash",
        target="Qwen3.8-27B-Q4_K_M.gguf",
        draft="dflash2-Qwen3.8-27B-Q8_0.gguf",
        n_max=7,
        min_accept_rate=0.5180,
        min_accept_len=4.5414,
    ),
    SpecCase(
        name="lilicorr",
        spec_type="draft-dflash",
        target="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
        draft="lilicorr-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-NVFP4.gguf",
        n_max=7,
        min_accept_rate=0.6399,
        min_accept_len=5.4444,
    ),
]

COMPACT_METHODS = {"mtp", "dflash", "dspark", "eagle3"}
NIGHTLY_METHODS = {"dflash2", "lilicorr"}

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization.\n\n```python\n",
    "The three main differences between TCP and UDP are:\n1.",
    "Question: What is the capital of France, and what river flows through it?\nAnswer:",
]

N_PREDICT = 128


def model_path(file: str) -> str:
    """Download a pinned GGUF from the consolidated test repository."""
    model_dir = os.environ.get("SPEC_MODEL_DIR", "./tmp")
    os.makedirs(model_dir, exist_ok=True)
    return download_file(
        f"{HF}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{file}",
        os.path.join(model_dir, file),
    )


def make_server(case: SpecCase) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model_path(case.target)
    server.n_ctx = 1024
    server.n_slots = 1
    server.n_gpu_layer = int(os.environ.get("N_GPU_LAYERS", 99))
    server.fa = "on"
    server.temperature = 0.0
    server.server_metrics = True
    server.spec_type = case.spec_type
    server.spec_draft_n_gpu_layer = server.n_gpu_layer
    if case.draft:
        server.model_draft = model_path(case.draft)
    if case.n_max:
        server.spec_draft_n_max = case.n_max
    return server


def generate(server: ServerProcess) -> list[list[int]]:
    outputs = []
    for prompt in PROMPTS:
        res = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": prompt,
                "temperature": 0.0,
                "top_k": 1,
                "seed": 42,
                "n_predict": N_PREDICT,
                "cache_prompt": False,
                "return_tokens": True,
            },
        )
        assert res.status_code == 200
        outputs.append(res.body["tokens"])
    return outputs


def spec_metrics(server: ServerProcess) -> dict[str, float]:
    # Use exact cumulative counters: the equivalent server log values are rounded
    # and emitted separately for each request.
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    values = {}
    for name in ("draft_tokens", "accepted_tokens", "drafts"):
        m = re.search(
            rf"^llamacpp:spec_decode_num_{name}_total (\S+)$", res.body, re.MULTILINE
        )
        assert m is not None, f"missing spec_decode_num_{name}_total in /metrics"
        values[name] = float(m.group(1))
    return values


def selected_cases():
    wanted = os.environ.get("SPEC_METHODS")
    if not wanted:
        return []
    aliases = {
        "all": {case.name for case in CASES},
        "compact": COMPACT_METHODS,
        "nightly": NIGHTLY_METHODS,
    }
    selected = aliases.get(
        wanted, {name.strip() for name in wanted.split(",") if name.strip()}
    )
    unknown = selected - {c.name for c in CASES}
    if unknown:
        raise ValueError(f"unknown SPEC_METHODS: {', '.join(sorted(unknown))}")
    return [case for case in CASES if case.name in selected]


@pytest.mark.heavy
@pytest.mark.slow
@pytest.mark.parametrize("case", selected_cases(), ids=lambda c: c.name)
def test_spec_method(case: SpecCase):
    server = make_server(case)
    try:
        server.start(timeout_seconds=1800)
        outputs = generate(server)
        m = spec_metrics(server)
    finally:
        server.stop()

    accept_rate = m["accepted_tokens"] / m["draft_tokens"] if m["draft_tokens"] else 0.0
    accept_len = 1.0 + m["accepted_tokens"] / m["drafts"] if m["drafts"] else 1.0
    print(
        f"SPEC_RESULT {case.name}: drafted={m['draft_tokens']:.0f} accepted={m['accepted_tokens']:.0f} "
        f"steps={m['drafts']:.0f} accept_rate={accept_rate:.4f} accept_len={accept_len:.4f}"
    )

    assert all(outputs), f"{case.name}: at least one request returned no tokens"
    assert m["draft_tokens"] > 0, f"{case.name}: no tokens were drafted"
    assert m["drafts"] > 0, f"{case.name}: no speculative verification steps ran"
    assert (
        accept_rate >= case.min_accept_rate
    ), f"{case.name}: accept rate {accept_rate:.3f} < {case.min_accept_rate}"
    assert (
        accept_len >= case.min_accept_len
    ), f"{case.name}: accept length {accept_len:.3f} < {case.min_accept_len}"

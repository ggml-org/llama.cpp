import json
import hashlib
import logging
import os
import pytest
import subprocess
from pathlib import Path
import numpy as np
import time
from typing import Optional, List

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

EPS = 1e-3
REPO_ROOT = Path(__file__).resolve().parents[3]
EXE = REPO_ROOT / ("build/bin/llama-embedding.exe" if os.name == "nt" else "build/bin/llama-embedding")
DEFAULT_ENV = {**os.environ, "LLAMA_CACHE": os.environ.get("LLAMA_CACHE", "tmp")}
SEED = "42"
ALLOWED_DIMS = {384, 768, 1024, 4096}

SMALL_CTX = 16        # preflight/cache
TEST_CTX  = 1024      # main tests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers (single source of truth for command building)
# ---------------------------------------------------------------------------


def resolve_exe() -> Path:
    exe = EXE
    if not exe.exists() and os.name == "nt":
        alt = REPO_ROOT / "build/bin/Release/llama-embedding.exe"
        if alt.exists():
            exe = alt
    if not exe.exists():
        raise FileNotFoundError(f"llama-embedding not found under {REPO_ROOT}/build/bin")
    return exe


def hf_params_default():
    return {
        "hf_repo": "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        "hf_file": "embeddinggemma-300M-qat-Q4_0.gguf",
    }


def build_cmd(
    *,
    exe: Path,
    params: dict,
    fmt: str,
    threads: int,
    ctx: int,
    seed: str,
    extra: Optional[List[str]] = None,  # was: list[str] | None
) -> List[str]:  # was: list[str]
    assert fmt in {"raw", "json"}, f"unsupported fmt={fmt}"
    cmd = [
        str(exe),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", str(ctx),
        "--embd-output-format", fmt,
        "--threads", str(threads),
        "--seed", seed,
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def run_cmd(cmd: list[str], text: str, timeout: int = 60) -> str:
    t0 = time.perf_counter()
    res = subprocess.run(cmd, input=text, capture_output=True, text=True,
                         env=DEFAULT_ENV, timeout=timeout)
    dur_ms = (time.perf_counter() - t0) * 1000.0
    if os.environ.get("EMBD_TEST_DEBUG") == "1":
        log.debug("embedding cmd finished in %.1f ms", dur_ms)

    if res.returncode != 0:
        raise AssertionError(f"embedding failed ({res.returncode}):\n{res.stderr[:400]}")
    out = res.stdout.strip()
    assert out, "empty stdout from llama-embedding"
    return out

# ---------------------------------------------------------------------------
# Session model preflight/cache
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedding_model():
    """Download/cache model once per session with a tiny ctx + no warmup."""
    exe = resolve_exe()
    params = hf_params_default()
    cmd = build_cmd(
        exe=exe, params=params, fmt="json",
        threads=1, ctx=SMALL_CTX, seed=SEED,
        extra=["--no-warmup"],
    )
    _ = run_cmd(cmd, text="ok")
    return params

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def run_embedding(
    text: str,
    *,
    fmt: str = "raw",
    threads: int = 1,
    ctx: int = TEST_CTX,
    params: Optional[dict] = None,  # was: dict | None
    timeout: int = 60,
) -> str:
    exe = resolve_exe()
    params = params or hf_params_default()
    cmd = build_cmd(exe=exe, params=params, fmt=fmt, threads=threads, ctx=ctx, seed=SEED)
    return run_cmd(cmd, text, timeout=timeout)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding_hash(vec: np.ndarray) -> str:
    """Return short deterministic signature for regression tracking."""
    return hashlib.sha256(vec[:8].tobytes()).hexdigest()[:16]


def parse_vec(out: str, fmt: str) -> np.ndarray:
    if fmt == "raw":
        arr = np.array(out.split(), dtype=np.float32)
    else:
        arr = np.array(json.loads(out)["data"][0]["embedding"], dtype=np.float32)
    return arr

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# Register custom mark so pytest doesn't warn about it
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")


@pytest.mark.parametrize("fmt", ["raw", "json"])
@pytest.mark.parametrize("text", ["hello world", "hi 🌎", "line1\nline2\nline3"])
def test_embedding_runs_and_finite(fmt, text, embedding_model):
    out = run_embedding(text, fmt=fmt, threads=1, ctx=TEST_CTX, params=embedding_model)
    vec = parse_vec(out, fmt)
    assert vec.dtype == np.float32
    # dim & finiteness
    assert len(vec) in ALLOWED_DIMS, f"unexpected dim={len(vec)}"
    assert np.all(np.isfinite(vec))
    assert 0.1 < np.linalg.norm(vec) < 10


def test_raw_vs_json_consistency(embedding_model):
    text = "hello world"
    raw = parse_vec(run_embedding(text, fmt="raw",  params=embedding_model), "raw")
    jsn = parse_vec(run_embedding(text, fmt="json", params=embedding_model), "json")
    assert raw.shape == jsn.shape
    cos = cosine_similarity(raw, jsn)
    assert cos > 0.999, f"raw/json divergence: cos={cos:.6f}"
    assert embedding_hash(raw) == embedding_hash(jsn)


def test_empty_input_deterministic(embedding_model):
    v1 = parse_vec(run_embedding("", fmt="raw", params=embedding_model), "raw")
    v2 = parse_vec(run_embedding("", fmt="raw", params=embedding_model), "raw")
    assert np.all(np.isfinite(v1))
    assert embedding_hash(v1) == embedding_hash(v2)
    assert cosine_similarity(v1, v2) > 0.99999


def test_very_long_input_stress(embedding_model):
    """Stress test: large input near context window."""
    text = "lorem " * 2000
    vec = parse_vec(run_embedding(text, fmt="raw", params=embedding_model), "raw")
    assert len(vec) in ALLOWED_DIMS
    assert np.isfinite(np.linalg.norm(vec))


@pytest.mark.parametrize("text", ["   ", "\n\n\n", "123 456 789"])
def test_low_information_inputs_stable(text, embedding_model):
    """Whitespace/numeric inputs should yield stable embeddings."""
    v1 = parse_vec(run_embedding(text, fmt="raw", params=embedding_model), "raw")
    v2 = parse_vec(run_embedding(text, fmt="raw", params=embedding_model), "raw")
    cos = cosine_similarity(v1, v2)
    assert cos > 0.999, f"unstable embedding for {text!r}"


@pytest.mark.parametrize("flag", ["--no-such-flag", "--help"])
def test_invalid_or_help_flag(flag):
    """Invalid flags should fail; help should succeed."""
    exe = resolve_exe()
    res = subprocess.run([str(exe), flag], capture_output=True, text=True, env=DEFAULT_ENV)
    if flag == "--no-such-flag":
        assert res.returncode != 0
        assert any(k in res.stderr.lower() for k in ("error", "invalid", "unknown"))
    else:
        assert res.returncode == 0
        assert "usage" in (res.stdout.lower() + res.stderr.lower())


@pytest.mark.parametrize("fmt", ["raw", "json"])
def test_threads_two_similarity_vs_single(fmt, embedding_model):
    text = "determinism vs threads"
    single = parse_vec(run_embedding(text, fmt=fmt, threads=1, params=embedding_model), fmt)
    multi  = parse_vec(run_embedding(text, fmt=fmt, threads=2, params=embedding_model), fmt)
    assert single.shape == multi.shape
    cos = cosine_similarity(single, multi)
    assert cos >= 0.999, f"threads>1 similarity too low: {cos:.6f}"


def test_json_shape_schema_minimal(embedding_model):
    js = json.loads(run_embedding("schema check", fmt="json", params=embedding_model))
    assert isinstance(js, dict)

    # Top-level “object” (present in CLI) is optional for us
    if "object" in js:
        assert js["object"] in ("list", "embeddings", "embedding_list")

    # Required: data[0].embedding + index
    assert "data" in js and isinstance(js["data"], list) and len(js["data"]) >= 1
    item0 = js["data"][0]
    assert isinstance(item0, dict)
    if "object" in item0:
        assert item0["object"] in ("embedding",)
    assert "index" in item0 and item0["index"] == 0
    assert "embedding" in item0 and isinstance(item0["embedding"], list)
    assert len(item0["embedding"]) in ALLOWED_DIMS

    # Optional fields: tolerate absence in current CLI output
    if "model" in js:
        assert isinstance(js["model"], str)
    if "dim" in js:
        assert js["dim"] == len(item0["embedding"])
    usage = js.get("usage", {})
    if usage:
        assert isinstance(usage, dict)
        # if present, prompt_tokens should be int
        if "prompt_tokens" in usage:
            assert isinstance(usage["prompt_tokens"], int)

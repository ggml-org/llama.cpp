import json
import hashlib
import os
import pytest
import subprocess
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

EPS = 1e-3
REPO_ROOT = Path(__file__).resolve().parents[3]
EXE = REPO_ROOT / ("build/bin/llama-embedding.exe" if os.name == "nt" else "build/bin/llama-embedding")
DEFAULT_ENV = {**os.environ, "LLAMA_CACHE": os.environ.get("LLAMA_CACHE", "tmp")}
SEED = "42"


# ---------------------------------------------------------------------------
# Model setup helpers
# ---------------------------------------------------------------------------

def get_model_hf_params():
    """Default lightweight embedding model."""
    return {
        "hf_repo": "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        "hf_file": "embeddinggemma-300M-qat-Q4_0.gguf",
    }


@pytest.fixture(scope="session")
def embedding_model():
    """Download/cache model once per session."""
    exe_path = EXE
    if not exe_path.exists():
        alt = REPO_ROOT / "build/bin/Release/llama-embedding.exe"
        if alt.exists():
            exe_path = alt
        else:
            raise FileNotFoundError(f"llama-embedding binary not found under {REPO_ROOT}/build/bin")

    params = get_model_hf_params()
    cmd = [
        str(exe_path),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", "16",
        "--embd-output-format", "json",
        "--no-warmup",
        "--threads", "1",
        "--seed", SEED,
    ]
    res = subprocess.run(cmd, input="ok", capture_output=True, text=True, env=DEFAULT_ENV)
    assert res.returncode == 0, f"model download failed: {res.stderr}"
    return params


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def run_embedding(text: str, fmt: str = "raw", params=None) -> str:
    """Runs llama-embedding and returns stdout (string)."""
    exe_path = EXE
    if not exe_path.exists():
        raise FileNotFoundError(f"Missing binary: {exe_path}")
    params = params or get_model_hf_params()
    cmd = [
        str(exe_path),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", "2048",
        "--embd-output-format", fmt,
        "--threads", "1",
        "--seed", SEED,
    ]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True, env=DEFAULT_ENV)
    if result.returncode:
        raise AssertionError(f"embedding failed ({result.returncode}):\n{result.stderr[:400]}")
    out = result.stdout.strip()
    assert out, f"empty output for text={text!r}, fmt={fmt}"
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding_hash(vec: np.ndarray) -> str:
    """Return short deterministic signature for regression tracking."""
    return hashlib.sha256(vec[:8].tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Register custom mark so pytest doesn't warn about it
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["raw", "json"])
@pytest.mark.parametrize("text", ["hello world", "hi 🌎", "line1\nline2\nline3"])
def test_embedding_runs_and_finite(fmt, text, embedding_model):
    """Ensure embeddings run end-to-end and produce finite floats."""
    out = run_embedding(text, fmt, embedding_model)
    floats = (
        np.array(out.split(), float)
        if fmt == "raw"
        else np.array(json.loads(out)["data"][0]["embedding"], float)
    )
    assert len(floats) > 100
    assert np.all(np.isfinite(floats)), f"non-finite values in {fmt} output"
    assert 0.1 < np.linalg.norm(floats) < 10


def test_raw_vs_json_consistency(embedding_model):
    """Compare raw vs JSON embedding output for same text."""
    text = "hello world"
    raw = np.array(run_embedding(text, "raw", embedding_model).split(), float)
    jsn = np.array(json.loads(run_embedding(text, "json", embedding_model))["data"][0]["embedding"], float)

    assert raw.shape == jsn.shape
    cos = cosine_similarity(raw, jsn)
    assert cos > 0.999, f"divergence: cos={cos:.4f}"
    assert embedding_hash(raw) == embedding_hash(jsn), "hash mismatch → possible nondeterminism"


def test_empty_input_deterministic(embedding_model):
    """Empty input should yield finite, deterministic vector."""
    v1 = np.array(run_embedding("", "raw", embedding_model).split(), float)
    v2 = np.array(run_embedding("", "raw", embedding_model).split(), float)
    assert np.all(np.isfinite(v1))
    cos = cosine_similarity(v1, v2)
    assert cos > 0.9999, f"Empty input not deterministic (cos={cos:.5f})"
    assert 0.1 < np.linalg.norm(v1) < 10


@pytest.mark.slow
def test_very_long_input_stress(embedding_model):
    """Stress test: large input near context window."""
    text = "lorem " * 2000
    vec = np.array(run_embedding(text, "raw", embedding_model).split(), float)
    assert len(vec) > 100
    assert np.isfinite(np.linalg.norm(vec))


@pytest.mark.parametrize(
    "text",
    ["   ", "\n\n\n", "123 456 789"],
)
def test_low_information_inputs_stable(text, embedding_model):
    """Whitespace/numeric inputs should yield stable embeddings."""
    v1 = np.array(run_embedding(text, "raw", embedding_model).split(), float)
    v2 = np.array(run_embedding(text, "raw", embedding_model).split(), float)
    cos = cosine_similarity(v1, v2)
    assert cos > 0.999, f"unstable embedding for {text!r}"


@pytest.mark.parametrize("flag", ["--no-such-flag", "--help"])
def test_invalid_or_help_flag(flag):
    """Invalid flags should fail; help should succeed."""
    res = subprocess.run([str(EXE), flag], capture_output=True, text=True)
    if flag == "--no-such-flag":
        assert res.returncode != 0
        assert any(k in res.stderr.lower() for k in ("error", "invalid", "unknown"))
    else:
        assert res.returncode == 0
        assert "usage" in (res.stdout.lower() + res.stderr.lower())


@pytest.mark.parametrize("fmt", ["raw", "json"])
@pytest.mark.parametrize("text", ["deterministic test", "deterministic test again"])
def test_repeated_call_consistent(fmt, text, embedding_model):
    """Same input → same hash across repeated runs."""
    out1 = run_embedding(text, fmt, embedding_model)
    out2 = run_embedding(text, fmt, embedding_model)

    if fmt == "json":
        v1 = np.array(json.loads(out1)["data"][0]["embedding"], float)
        v2 = np.array(json.loads(out2)["data"][0]["embedding"], float)
    else:
        v1 = np.array(out1.split(), float)
        v2 = np.array(out2.split(), float)

    assert embedding_hash(v1) == embedding_hash(v2)

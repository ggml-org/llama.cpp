from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
import numpy as np
from pathlib import Path
import json, os, time, statistics, subprocess, math


# ---------------------------------------------------------------------------
# Benchmark decorator
# ---------------------------------------------------------------------------

def benchmark(n=3):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            for _ in range(n):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                times.append(time.perf_counter() - start)
            avg = statistics.mean(times)
            print(f"\n[benchmark] {fn.__name__}: mean={avg*1000:.1f} ms over {n} runs")
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def get_model_hf_params():
    """Default lightweight embedding model."""
    return {
        "hf_repo": "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        "hf_file": "embeddinggemma-300M-qat-Q4_0.gguf",
    }


def ensure_model_downloaded(params=None):
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = os.environ.get("LLAMA_CACHE", "tmp")
    emb_path = repo_root / "build/bin/llama-embedding"
    if not emb_path.exists() and os.name == "nt":
        emb_path = repo_root / "build/bin/Release/llama-embedding.exe"
    if not emb_path.exists():
        raise FileNotFoundError(f"llama-embedding not found at {emb_path}")

    params = params or get_model_hf_params()
    cmd = [
        str(emb_path),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", "16",
        "--embd-output-format", "json",
        "--no-warmup",
        "--threads", "1",
    ]

    env = os.environ.copy()
    env["LLAMA_CACHE"] = cache_dir
    result = subprocess.run(cmd, input="ok", capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download model:\n{result.stderr}")
    return params


def run_embedding(text: str, fmt: str = "raw", params=None):
    repo_root = Path(__file__).resolve().parents[2]
    exe = repo_root / "build/bin/llama-embedding"
    assert exe.exists(), f"Missing binary: {exe}"

    params = ensure_model_downloaded(params)
    cache_dir = os.environ.get("LLAMA_CACHE", "tmp")

    cmd = [
        str(exe),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", "2048",
        "--embd-output-format", fmt,
    ]

    env = os.environ.copy()
    env["LLAMA_CACHE"] = cache_dir

    out = subprocess.run(cmd, input=text, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        print(out.stderr)
        raise AssertionError(f"embedding binary failed (code {out.returncode})")
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# 1️⃣ RAW vs JSON baseline tests
# ---------------------------------------------------------------------------

@benchmark(n=3)
def test_embedding_raw_and_json_consistency():
    """
    Run both output modes and verify same embedding shape, norm similarity,
    and small cosine distance.
    """
    out_raw = run_embedding("hello world", "raw")
    floats_raw = np.array([float(x) for x in out_raw.split()])

    out_json = run_embedding("hello world", "json")
    j = json.loads(out_json)
    floats_json = np.array(j["data"][0]["embedding"])

    assert len(floats_raw) == len(floats_json), "Embedding dimension mismatch"
    cos = np.dot(floats_raw, floats_json) / (np.linalg.norm(floats_raw) * np.linalg.norm(floats_json))
    print(f"Cosine similarity raw vs json: {cos:.4f}")
    # expect high similarity but not perfect (formatting precision differences)
    assert cos > 0.999, f"Unexpected divergence between raw and json output ({cos:.4f})"


@benchmark(n=3)
def test_embedding_perf_regression_raw_vs_json():
    """
    Compare performance between raw and json output.
    Ensures raw mode is not significantly slower or memory-heavier.
    """
    text = "performance regression test " * 512
    params = ensure_model_downloaded()

    def run(fmt):
        start = time.perf_counter()
        out = run_embedding(text, fmt, params)
        dur = time.perf_counter() - start
        mem = len(out)
        return dur, mem

    t_raw, m_raw = run("raw")
    t_json, m_json = run("json")

    print(f"[perf] raw={t_raw:.3f}s ({m_raw/1e3:.1f} KB) | json={t_json:.3f}s ({m_json/1e3:.1f} KB)")
    # raw should never be significantly slower or consume wildly more memory
    assert t_raw <= t_json * 1.2, f"raw too slow vs json ({t_raw:.3f}s vs {t_json:.3f}s)"
    assert m_raw <= m_json * 1.2, f"raw output unexpectedly larger ({m_raw} vs {m_json} bytes)"


# ---------------------------------------------------------------------------
# 2️⃣ Edge-case coverage
# ---------------------------------------------------------------------------

def test_embedding_empty_input():
    """
    Empty input should not crash and should yield a deterministic, finite embedding.
    Some models (e.g. Gemma/BGE) emit BOS token embedding with norm ≈ 1.0.
    """
    out1 = run_embedding("", "raw")
    out2 = run_embedding("", "raw")

    floats1 = np.array([float(x) for x in out1.split()])
    floats2 = np.array([float(x) for x in out2.split()])

    # Basic validity
    assert len(floats1) > 0, "Empty input produced no embedding"
    assert np.all(np.isfinite(floats1)), "Embedding contains NaN or inf"
    norm = np.linalg.norm(floats1)
    assert 0.5 <= norm <= 1.5, f"Unexpected norm for empty input: {norm}"

    # Determinism check: cosine similarity should be ≈ 1
    cos = np.dot(floats1, floats2) / (np.linalg.norm(floats1) * np.linalg.norm(floats2))
    assert cos > 0.9999, f"Empty input not deterministic (cos={cos:.4f})"
    print(f"[empty] norm={norm:.4f}, cos={cos:.6f}")


def test_embedding_special_characters():
    """Unicode and punctuation coverage."""
    special_text = "你好 🌍\n\t!@#$%^&*()_+-=[]{}|;:'\",.<>?/`~"
    out = run_embedding(special_text, "raw")
    floats = [float(x) for x in out.split()]
    assert len(floats) > 10
    norm = np.linalg.norm(floats)
    assert math.isfinite(norm) and norm > 0


@benchmark(n=1)
def test_embedding_very_long_input():
    """Stress test for context limit handling."""
    long_text = "lorem " * 10000
    out = run_embedding(long_text, "raw")
    floats = [float(x) for x in out.split()]
    print(f"Output floats (long input): {len(floats)}")
    assert len(floats) > 100
    assert np.isfinite(np.linalg.norm(floats))


# ---------------------------------------------------------------------------
# 3️⃣ Legacy and concurrency coverage (unchanged)
# ---------------------------------------------------------------------------

@benchmark(n=3)
def test_embedding_raw_vector_shape():
    out = run_embedding("hello world", "raw")
    floats = [float(x) for x in out.split()]
    print(f"Embedding size: {len(floats)} floats")
    assert len(floats) > 100
    norm = np.linalg.norm(floats)
    assert 0.5 < norm < 2.0


@benchmark(n=3)
def test_embedding_large_vector_output():
    text = " ".join(["hello"] * 4096)
    out = run_embedding(text, "raw")
    valid_dims = {384, 768, 1024, 1280, 2048, 4096}
    floats = [float(x) for x in out.split()]
    print(f"Output floats: {len(floats)}")
    assert len(floats) in valid_dims, (
        f"Unexpected embedding size: {len(floats)}. Expected one of {sorted(valid_dims)}."
    )


def run_one(args):
    i, params, text = args
    repo_root = Path(__file__).resolve().parents[2]
    exe = repo_root / "build/bin/llama-embedding"
    cache_dir = os.environ.get("LLAMA_CACHE", "tmp")

    cmd = [
        str(exe),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "--ctx-size", "1024",
        "--embd-output-format", "raw",
        "--threads", "1",
    ]

    env = os.environ.copy()
    env["LLAMA_CACHE"] = cache_dir
    start = time.perf_counter()
    result = subprocess.run(cmd, input=text, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[worker {i}] stderr:\n{result.stderr}")
        raise AssertionError(f"embedding run {i} failed (code {result.returncode})")
    return time.perf_counter() - start


@benchmark(n=1)
def test_embedding_concurrent_invocations():
    params = ensure_model_downloaded()
    text = " ".join(["concurrency"] * 128)
    n_workers = 4
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_one, (i, params, text)) for i in range(n_workers)]
        times = [f.result() for f in as_completed(futures)]
    avg = statistics.mean(times)
    print(f"[concurrency] {n_workers} parallel runs: mean={avg*1000:.1f} ms")


@benchmark(n=1)
def test_embedding_large_model_logging_stress():
    """Optional stress test using larger model for stdout/mutex path."""
    large_model = {
        "hf_repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "hf_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    }
    text = " ".join(["benchmark"] * 8192)
    out = run_embedding(text, "raw", params=large_model)
    floats = [float(x) for x in out.split()]
    assert len(floats) >= 1024


def test_embedding_invalid_flag():
    """
    Invalid flag should produce a non-zero exit and a helpful error message.
    Ensures CLI argument parsing fails gracefully instead of crashing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    exe = repo_root / "build/bin/llama-embedding"
    assert exe.exists(), f"Missing binary: {exe}"

    # Pass an obviously invalid flag to trigger error handling.
    result = subprocess.run(
        [str(exe), "--no-such-flag"],
        capture_output=True,
        text=True,
    )

    # Must return non-zero and print something meaningful to stderr.
    assert result.returncode != 0, "Expected non-zero exit on invalid flag"
    stderr_lower = result.stderr.lower()
    assert (
        "error" in stderr_lower
        or "invalid" in stderr_lower
        or "unknown" in stderr_lower
    ), f"Unexpected stderr output: {result.stderr}"

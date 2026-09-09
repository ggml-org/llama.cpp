import json
import math
import os
import subprocess
from pathlib import Path

import pytest

# EmbeddingGemma-300M Q4_0: 768-d, MEAN pooling. CLI prints %1.7f.
# PRINT_ABS_TOL covers print rounding (measured raw vs json maxabs 0).
# BATCH_ABS_TOL is separate; measured batch vs single maxabs 0 on this fixture.
# L2_TOL is vs unit norm from --embd-normalize 2 (measured |L2-1| ~ 4e-8).
DIM = 768
PRINT_ABS_TOL = 1e-6
BATCH_ABS_TOL = 1e-6
L2_TOL = 1e-5
# A vs B maxabs ~ 0.16 on this fixture; 1e-3 is well above print noise.
DELIVER_MIN_ABS = 1e-3

PROMPT_A = "hello world"
PROMPT_B = "completely different text"

REPO_ROOT = Path(__file__).resolve().parents[3]
EXE = REPO_ROOT / ("build/bin/llama-embedding.exe" if os.name == "nt" else "build/bin/llama-embedding")
_CACHE = os.environ.get("LLAMA_CACHE", "tmp")
CACHE_DIR = _CACHE if os.path.isabs(_CACHE) else str(REPO_ROOT / _CACHE)
DEFAULT_ENV = {**os.environ, "LLAMA_CACHE": CACHE_DIR}
SMALL_CTX = 16
TEST_CTX = 1024
RUN_TIMEOUT = 90


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


def build_cmd(*, exe: Path, params: dict, fmt: str, prompt: str, ctx: int, extra=None) -> list:
    assert fmt in {"raw", "json"}, f"unsupported fmt={fmt}"
    cmd = [
        str(exe),
        "-hfr", params["hf_repo"],
        "-hff", params["hf_file"],
        "-p", prompt,
        "--pooling", "mean",
        "--embd-normalize", "2",
        "--embd-output-format", fmt,
        "--threads", "1",
        "--n-gpu-layers", "0",
        "--no-op-offload",
        "--ctx-size", str(ctx),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def run_cmd(cmd: list, timeout: int = RUN_TIMEOUT) -> str:
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=DEFAULT_ENV,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )
    if res.returncode != 0:
        raise AssertionError(
            f"embedding failed ({res.returncode}):\n{res.stderr[-800:]}"
        )
    out = res.stdout.strip()
    assert out, f"empty stdout from llama-embedding\nstderr:\n{res.stderr[-400:]}"
    return res.stdout


def parse_raw_rows(out: str) -> list:
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError as exc:
            raise AssertionError(f"raw stdout is not a float row: {line[:120]!r}") from exc
    assert rows, "raw stdout had no embedding rows"
    return rows


def parse_json(out: str) -> dict:
    obj = json.loads(out)
    assert isinstance(obj, dict), f"JSON root must be an object, got {type(obj).__name__}"
    assert "data" in obj and isinstance(obj["data"], list), "JSON missing data list"
    return obj


def json_rows(obj: dict, n_expect: int) -> list:
    data = obj["data"]
    assert len(data) == n_expect, f"JSON data length {len(data)}, expected {n_expect}"
    rows = []
    for i, item in enumerate(data):
        assert isinstance(item, dict), f"data[{i}] is not an object"
        idx = item.get("index")
        assert type(idx) is int and idx == i, f"data[{i}].index={idx!r}, expected int {i}"
        emb = item.get("embedding")
        assert isinstance(emb, list), f"data[{i}] missing embedding list"
        for j, x in enumerate(emb):
            assert type(x) in (int, float), f"data[{i}].embedding[{j}] is {type(x).__name__}, not a JSON number"
        rows.append(emb)
    return rows


def maxabs(a: list, b: list) -> float:
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


def l2(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def check_vec(name: str, vec: list) -> None:
    assert len(vec) == DIM, f"{name} dim={len(vec)}, expected {DIM}"
    assert all(math.isfinite(x) for x in vec), f"{name} has a non-finite value"
    n = l2(vec)
    assert abs(n - 1.0) <= L2_TOL, f"{name} L2={n}, expected 1 +/- {L2_TOL}"


@pytest.fixture(scope="session")
def embedding_model():
    exe = resolve_exe()
    params = hf_params_default()
    cmd = build_cmd(
        exe=exe, params=params, fmt="json", prompt="ok",
        ctx=SMALL_CTX, extra=["--no-warmup"],
    )
    run_cmd(cmd)
    return params


def run_embedding(prompt: str, *, fmt: str, params: dict, ctx: int = TEST_CTX) -> str:
    exe = resolve_exe()
    cmd = build_cmd(exe=exe, params=params, fmt=fmt, prompt=prompt, ctx=ctx, extra=["--no-warmup"])
    return run_cmd(cmd)


def test_prompt_delivery(embedding_model):
    a = parse_raw_rows(run_embedding(PROMPT_A, fmt="raw", params=embedding_model))
    b = parse_raw_rows(run_embedding(PROMPT_B, fmt="raw", params=embedding_model))
    assert len(a) == 1 and len(b) == 1
    check_vec("prompt A", a[0])
    check_vec("prompt B", b[0])
    d = maxabs(a[0], b[0])
    assert d > DELIVER_MIN_ABS, f"prompts A and B are too close (maxabs={d}); prompt may be ignored"


def test_raw_vs_json_consistency(embedding_model):
    raw_rows = parse_raw_rows(run_embedding(PROMPT_A, fmt="raw", params=embedding_model))
    js = parse_json(run_embedding(PROMPT_A, fmt="json", params=embedding_model))
    assert len(raw_rows) == 1
    j_rows = json_rows(js, 1)
    check_vec("raw", raw_rows[0])
    check_vec("json", j_rows[0])
    d = maxabs(raw_rows[0], j_rows[0])
    assert d <= PRINT_ABS_TOL, f"raw vs json maxabs={d} > {PRINT_ABS_TOL}"


def test_multiline_prompt_order(embedding_model):
    a_raw = parse_raw_rows(run_embedding(PROMPT_A, fmt="raw", params=embedding_model))[0]
    b_raw = parse_raw_rows(run_embedding(PROMPT_B, fmt="raw", params=embedding_model))[0]
    a_js = json_rows(parse_json(run_embedding(PROMPT_A, fmt="json", params=embedding_model)), 1)[0]
    b_js = json_rows(parse_json(run_embedding(PROMPT_B, fmt="json", params=embedding_model)), 1)[0]

    batch = PROMPT_A + "\n" + PROMPT_B
    raw_batch = parse_raw_rows(run_embedding(batch, fmt="raw", params=embedding_model))
    js_batch = json_rows(parse_json(run_embedding(batch, fmt="json", params=embedding_model)), 2)
    assert len(raw_batch) == 2
    check_vec("batch raw[0]", raw_batch[0])
    check_vec("batch raw[1]", raw_batch[1])
    check_vec("batch json[0]", js_batch[0])
    check_vec("batch json[1]", js_batch[1])

    d_fmt0 = maxabs(raw_batch[0], js_batch[0])
    d_fmt1 = maxabs(raw_batch[1], js_batch[1])
    assert d_fmt0 <= PRINT_ABS_TOL, f"batch row0 raw vs json maxabs={d_fmt0}"
    assert d_fmt1 <= PRINT_ABS_TOL, f"batch row1 raw vs json maxabs={d_fmt1}"

    d_a_raw = maxabs(raw_batch[0], a_raw)
    d_b_raw = maxabs(raw_batch[1], b_raw)
    d_a_js = maxabs(js_batch[0], a_js)
    d_b_js = maxabs(js_batch[1], b_js)
    assert d_a_raw <= BATCH_ABS_TOL, f"batch raw[0] vs prompt A maxabs={d_a_raw}"
    assert d_b_raw <= BATCH_ABS_TOL, f"batch raw[1] vs prompt B maxabs={d_b_raw}"
    assert d_a_js <= BATCH_ABS_TOL, f"batch json[0] vs prompt A maxabs={d_a_js}"
    assert d_b_js <= BATCH_ABS_TOL, f"batch json[1] vs prompt B maxabs={d_b_js}"

    d_rev0 = maxabs(raw_batch[0], b_raw)
    d_rev1 = maxabs(raw_batch[1], a_raw)
    assert not (d_rev0 <= BATCH_ABS_TOL and d_rev1 <= BATCH_ABS_TOL), "batch rows match A/B reversed"

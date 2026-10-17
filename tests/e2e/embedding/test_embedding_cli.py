import hashlib
import json
import math
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
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

HF_REPO = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF"
HF_FILE = "embeddinggemma-300M-qat-Q4_0.gguf"
HF_REV = "8dd0ca2a66a8f14470acb0e2a71f801afbc5fb73"
EXPECTED_SHA256 = "50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26"
DOWNLOAD_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{HF_FILE}"
FIXTURE_NAME = f"embeddinggemma-300M-qat-Q4_0-{HF_REV}-{EXPECTED_SHA256}.gguf"

REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE = os.environ.get("LLAMA_CACHE", "tmp")
CACHE_DIR = _CACHE if os.path.isabs(_CACHE) else str(REPO_ROOT / _CACHE)
DEFAULT_ENV = {**os.environ, "LLAMA_CACHE": CACHE_DIR}
TEST_CTX = 1024
# One CLI load+embed is ~1s locally; 30s covers slow runners without hiding hangs.
RUN_TIMEOUT = 30
ACQUIRE_ATTEMPTS = 3
ACQUIRE_TIMEOUT = 120
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


def default_fixture_path() -> Path:
    return Path(CACHE_DIR) / FIXTURE_NAME


def _resolve_override(override: str) -> Path:
    # EMBEDDING_EXE and EMBEDDING_MODEL: expand ~ and resolve against the calling
    # Python process cwd (not run_cmd's cwd=REPO_ROOT) before exists checks
    # and before building subprocess argv. Pass those absolute paths through.
    return Path(override).expanduser().resolve()


def resolve_exe() -> Path:
    override = os.environ.get("EMBEDDING_EXE")
    if override:
        exe = _resolve_override(override)
        if not exe.exists():
            raise FileNotFoundError(f"EMBEDDING_EXE not found: {exe}")
        return exe
    exe = REPO_ROOT / ("build/bin/llama-embedding.exe" if os.name == "nt" else "build/bin/llama-embedding")
    if not exe.exists() and os.name == "nt":
        alt = REPO_ROOT / "build/bin/Release/llama-embedding.exe"
        if alt.exists():
            exe = alt
    if not exe.exists():
        raise FileNotFoundError(f"llama-embedding not found under {REPO_ROOT}/build/bin")
    return exe


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"embedding fixture missing: {path}\n"
            f"Run: python tests/e2e/embedding/test_embedding_cli.py\n"
            f"Or set EMBEDDING_MODEL to a verified {HF_FILE} (rev {HF_REV})."
        )
    got = sha256_file(path)
    if got != EXPECTED_SHA256:
        raise AssertionError(
            f"embedding fixture checksum mismatch for {path}:\n"
            f"  got      {got}\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"Reacquire this dedicated fixture (rev {HF_REV}); do not change the expected hash."
        )
    return path


def _download_to(tmp_path: Path) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    # Public pinned fixture: never attach ambient HF tokens (they would follow redirects).
    req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "llama.cpp-embedding-cli-tests"})
    with urllib.request.urlopen(req, timeout=ACQUIRE_TIMEOUT) as resp:
        expected_len = resp.headers.get("Content-Length")
        n_expect = int(expected_len) if expected_len and expected_len.isdigit() else None
        written = 0
        with open(tmp_path, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    if n_expect is not None and written != n_expect:
        raise urllib.error.URLError(f"incomplete download: {written} bytes, expected {n_expect}")
    if written == 0:
        raise urllib.error.URLError(f"empty download from {DOWNLOAD_URL}")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            return _is_retryable(reason)
        msg = str(reason) if reason is not None else str(exc)
        return msg.startswith("incomplete download:") or msg.startswith("empty download from ")
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, ssl.SSLError):
        return False
    return False


def acquire_model(dest: Path) -> Path:
    dest = dest.resolve()
    if dest.exists():
        try:
            return verify_model(dest)
        except AssertionError:
            dest.unlink()
    last_err = None
    tmp_path = dest.with_name(dest.name + ".partial")
    for attempt in range(1, ACQUIRE_ATTEMPTS + 1):
        try:
            _download_to(tmp_path)
            got = sha256_file(tmp_path)
            if got != EXPECTED_SHA256:
                tmp_path.unlink(missing_ok=True)
                last_err = AssertionError(
                    f"downloaded fixture checksum mismatch (attempt {attempt}/{ACQUIRE_ATTEMPTS}):\n"
                    f"  got      {got}\n"
                    f"  expected {EXPECTED_SHA256}"
                )
                if attempt < ACQUIRE_ATTEMPTS:
                    time.sleep(3)
                    continue
                raise last_err
            os.replace(tmp_path, dest)
            return verify_model(dest)
        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if isinstance(exc, AssertionError) or not _is_retryable(exc) or attempt >= ACQUIRE_ATTEMPTS:
                raise
            time.sleep(3)
    raise last_err


def resolve_model_path() -> Path:
    override = os.environ.get("EMBEDDING_MODEL")
    if override:
        return _resolve_override(override)
    return default_fixture_path()


def build_cmd(*, exe: Path, model_path: Path, fmt: str, prompt: str, ctx: int, extra=None) -> list:
    assert fmt in {"raw", "json"}, f"unsupported fmt={fmt}"
    cmd = [
        str(exe),
        "-m", str(model_path),
        "--offline",
        "-p", prompt,
        "--pooling", "mean",
        "--embd-normalize", "2",
        "--embd-output-format", fmt,
        "--threads", "1",
        "--n-gpu-layers", "0",
        "--no-op-offload",
        "--ctx-size", str(ctx),
        "--no-warmup",
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
    try:
        obj = json.loads(out)
    except json.JSONDecodeError as exc:
        head = out.splitlines()[0][:120] if out.strip() else "<empty>"
        raise AssertionError(f"JSON stdout is not a JSON object: {head!r}") from exc
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
    path = verify_model(resolve_model_path())
    return {"model_path": str(path)}


def run_embedding(prompt: str, *, fmt: str, params: dict, ctx: int = TEST_CTX) -> str:
    exe = resolve_exe()
    model_path = Path(params["model_path"]).expanduser().resolve()
    cmd = build_cmd(exe=exe, model_path=model_path, fmt=fmt, prompt=prompt, ctx=ctx)
    assert "-hfr" not in cmd and "-hff" not in cmd
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


if __name__ == "__main__":
    path = acquire_model(default_fixture_path())
    sys.stdout.write(str(path) + "\n")

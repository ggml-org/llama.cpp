#!/usr/bin/env python3
"""Check a llama.cpp build against the dory token oracle.

The oracle (produced by the HF reference implementation) pins two things:

  * the exact prompt token ids the model must be fed  (add_special_tokens=False),
  * the exact greedy continuation, as token ids, for N positions.

Text agreement is NOT acceptance.  A build that emits " Paris. The capital of
Germany is Berlin." with subtly different token boundaries is broken, and only
the ids can tell.

How generated token ids are obtained
------------------------------------
llama-cli cannot emit generated token ids: `--verbose-prompt` prints prompt
tokens only, and the per-token debug line in tools/completion/completion.cpp is
commented out.  Two workable routes exist; this script implements both.

  server  llama-server /completion with "n_probs": 1 returns
          completion_probabilities[i]["id"] for every generated token -- real,
          machine-readable generation ids, plus the text piece of each one.
          The prompt is passed as an ARRAY OF INTEGERS, so the server cannot
          re-tokenize it and cannot add BOS: the model is guaranteed to see the
          oracle's prompt_ids exactly.  This is the authoritative route.

  cli     llama-cli generates and prints text.  Prompt ids come from
          --verbose-prompt (a real id dump at the real generation site), but
          generated ids must be recovered by re-tokenizing the emitted text with
          llama-tokenize.  That inversion is usually exact and is NOT guaranteed
          at token boundaries, so every cli-route result is labelled
          RETOKENIZED and a cli mismatch tells you to confirm with --route server
          instead of blaming the model graph.

Route: logits (the re-cut acceptance) -- see docs/dory-port-contract.md
----------------------------------------------------------------------
An id-for-id oracle cannot pass a Q5_K_M port: the "discriminating" token is a
0.008-logit tie inside the BF16 reference, while the quantised artifact deviates
from BF16 by 0.05-0.7.  The re-cut acceptance is therefore:

  1. prompt_ids must match exactly                       (a tokenizer check)
  2. the top-5 SET and ORDER at every position must match the BF16 reference
  3. per-position |delta| <= 0.5 -- NOT argmax equality

`--route logits` implements 1-3: it starts llama-server on a free port with the
GGUF (`-c 64 -ngl 0 -np 1`, BOS forced off), waits for /health, and posts the
prompt as an INTEGER ARRAY with n_probs=5, temperature=0, n_predict=16.  For
each position it reads llama-server's top-5 `(id, logprob)` pairs and compares
them against a reference top-5 file (`--reference`).  Because the prompt is an
integer array used verbatim, the server cannot re-tokenise it and cannot
prepend BOS.

Where the reference comes from, and what it must say about its units
--------------------------------------------------------------------
llama-server never reports raw logits on /completion: server-common.cpp
get_token_probabilities() sorts the raw logits, then normalises them with a
FULL-VOCABULARY softmax, and server-task.cpp emits log(p) under "logprob"
(with post_sampling_probs=false the values are PRE-sampling, so the
argmax-collapsing temperature sampler has not run yet).

A reference recorded in RAW LOGITS therefore cannot be compared value-for-value
with what this route observes.  What IS exactly comparable is the CENTRED value
value_i - value_top, because the log-sum-exp cancels:

    logprob_i - logprob_top = (logit_i - LSE) - (logit_top - LSE)
                            = logit_i - logit_top

compare_position() uses that identity when the reference declares
`"quantity": "logits"`, and the report labels the delta mode.  Consequences for
the contract's "per-position |delta logit| <= 0.5":

  * reference in logprobs -> the delta is a true per-candidate logprob delta and
    0.5 applies unchanged (same units);
  * reference in logits   -> only the CENTRED delta is measurable.  It differs
    from the contract's |delta logit| by the per-position shift of the top logit
    (delta logit_top), which /completion cannot reveal.  So the contract's
    number needs RE-BASING rather than silent reuse: record the reference as
    logprobs (torch.log_softmax of the same logits) and keep 0.5, or dump the
    port's raw logits (llama-cli --save-all-logits) and compare logits to
    logits.

If no reference top-5 file exists on disk the route still runs, prints the
observed top-5 at every position, writes them to `--dump-observed`, and STOPS
SHORT OF A VERDICT (exit 3).  It never invents reference numbers.

Exit codes
----------
  0  full match, all n positions
  1  mismatch (first divergence reported with ids and text)
  2  could not run the binary (missing, refused to start, died, timed out)
  3  tokenizer / id-plumbing problem -- the comparison would be meaningless;
     also: no reference to compare against, or the comparison could not cover
     every position (logits route).  NEVER a pass.

Stdlib only.  Builds nothing, edits nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_CANNOT_RUN = 2
EXIT_PLUMBING = 3

DEFAULT_BINARY = "./build/bin/llama-cli"
DEFAULT_MODEL = "/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
DEFAULT_ORACLE = "/Volumes/Widekeep/oracle/oracle-dory.json"

# Greedy decoding is achieved by temperature <= 0: llama_sampler_temp_impl()
# sets every non-argmax logit to -INFINITY (src/llama-sampler.cpp:271), so the
# outcome is deterministic even when the `dist` sampler is used instead of the
# `greedy` fast path -- and common/sampling.cpp:411 disables that fast path
# whenever n_probs > 0, which is exactly what this script requests.  Every
# sampler that runs BEFORE temperature and can perturb the logits (penalties,
# dry, xtc) is pinned to its neutral value too, so the argmax taken is the
# model's argmax and not a post-processed one.
GREEDY_SERVER = {
    "temperature": 0.0,
    "dynatemp_range": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "xtc_probability": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "dry_multiplier": 0.0,
    "mirostat": 0,
    "n_probs": 1,
    "seed": 1234,
    "cache_prompt": False,
    "stream": False,
}

HEADER = "=" * 78


def banner(title: str) -> None:
    print(f"\n{HEADER}\n{title}\n{HEADER}")


def fail(code: int, headline: str, detail: str = "") -> int:
    sys.stdout.flush()
    print(f"\nFAIL[{code}] {headline}", file=sys.stderr, flush=True)
    if detail:
        print(detail, file=sys.stderr, flush=True)
    return code


def tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no log captured)"
    kept = text.strip().splitlines()[-lines:]
    return "\n".join(kept) if kept else "(log empty)"


def sibling(binary: Path, name: str) -> Path | None:
    """Resolve a sibling tool next to --binary (same build/bin directory)."""
    cand = binary.parent / name
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    found = shutil.which(name)
    return Path(found) if found else None


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------


def load_oracle(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(fail(EXIT_PLUMBING, f"oracle not found: {path}"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(fail(EXIT_PLUMBING, f"cannot parse oracle {path}: {exc}"))

    for field in ("prompt", "prompt_ids", "generated_ids"):
        if field not in data:
            raise SystemExit(fail(EXIT_PLUMBING, f"oracle {path} is missing '{field}'"))
    for field in ("prompt_ids", "generated_ids"):
        ids = data[field]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
            raise SystemExit(fail(EXIT_PLUMBING, f"oracle '{field}' is not a non-empty int list"))
    return data


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {exc}: {raw[:200]!r}") from exc


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --------------------------------------------------------------------------
# tokenizer route
# --------------------------------------------------------------------------

LIST_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*,?\s*\]", re.S)

# A tokenizer that cannot load the model is a "cannot run the binary" failure
# (exit 2), NOT a tokenizer/plumbing problem (exit 3).  These markers are how
# llama-tokenize reports the model itself being unusable -- e.g. before
# LLM_ARCH_DORY is registered: "unknown model architecture: 'dory'".
MODEL_LOAD_MARKERS = (
    "error loading model",
    "unknown model architecture",
    "failed to load model",
)


def tokenizer_failure(log: Path, reason: str) -> tuple[int, str]:
    """Classify a tokenizer failure so exit 3 is never used for a dead model."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return EXIT_PLUMBING, reason
    hits = [ln.strip() for ln in text.splitlines() if any(m in ln for m in MODEL_LOAD_MARKERS)]
    if hits:
        unique = list(dict.fromkeys(hits))
        return EXIT_CANNOT_RUN, (
            "the model could not be loaded by the tokenizer, so no ids exist to compare:\n"
            + "\n".join(f"    {ln}" for ln in unique[:6])
        )
    return EXIT_PLUMBING, reason



def parse_id_list(text: str) -> list[int] | None:
    match = LIST_RE.search(text)
    if not match:
        return None
    body = match.group(0).strip("[]").strip().rstrip(",")
    if not body:
        return None
    try:
        return [int(part) for part in body.replace("\n", " ").split(",") if part.strip()]
    except ValueError:
        return None


def tokenize_with_binary(tok_bin: Path, model: Path, text: str, log: Path) -> tuple[list[int] | None, str]:
    """`llama-tokenize --ids --no-bos` -> exact ids with no BOS added."""
    cmd = [str(tok_bin), "-m", str(model), "--ids", "--no-bos", "-p", text]
    with log.open("wb") as fh:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=fh, timeout=3600)
    if proc.returncode != 0:
        return None, f"llama-tokenize exited {proc.returncode} (log: {log})"
    ids = parse_id_list(proc.stdout.decode("utf-8", "replace"))
    if ids is None:
        return None, f"no id list in llama-tokenize stdout: {proc.stdout[:300]!r}"
    return ids, ""


def tokenize_via_server(base: str, text: str) -> tuple[list[int] | None, str]:
    """Fallback: ask a live server for the ids (same vocab, same build)."""
    payload = {"content": text, "add_special": False, "with_pieces": False}
    try:
        body = http_json("POST", f"{base}/tokenize", payload, timeout=120)
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        return None, str(exc)
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        return None, f"/tokenize returned no id list: {str(body)[:300]}"
    return tokens, ""


def detok(base: str | None, ids: list[int]) -> str | None:
    """ids -> text via a live server, or None when there is no server up."""
    if base is None or not ids:
        return None
    try:
        body = http_json("POST", f"{base}/detokenize", {"tokens": list(ids)}, timeout=60)
    except (urllib.error.URLError, RuntimeError, OSError):
        return None
    content = body.get("content")
    return content if isinstance(content, str) else None


# --------------------------------------------------------------------------
# server session
# --------------------------------------------------------------------------


class ServerSession:
    """llama-server as a context manager: start, wait for /health, stop on exit."""

    def __init__(
        self,
        server_bin: Path,
        model: Path,
        port: int,
        boot_timeout: float,
        log: Path,
        ctx: int = 4096,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.server_bin = server_bin
        self.model = model
        self.port = port or free_port()
        self.boot_timeout = boot_timeout
        self.log = log
        self.ctx = ctx
        self.extra_args = tuple(extra_args)
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{self.port}"
        self.error = ""

    def __enter__(self) -> "ServerSession":
        cmd = [
            str(self.server_bin),
            "-m", str(self.model),
            "-c", str(self.ctx),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-np", "1",
            "--no-webui",
            # Belt and braces: the prompt is sent as an id array, so BOS cannot be
            # introduced anyway.  This keeps the vocab config honest regardless.
            "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
            *self.extra_args,
        ]
        print(f"launching: {' '.join(cmd)}")
        self._fh = self.log.open("wb")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
        started = time.time()
        deadline = started + self.boot_timeout
        last = "no attempt"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.error = f"llama-server exited with code {self.proc.returncode} before becoming healthy"
                return self
            try:
                body = http_json("GET", f"{self.base}/health", timeout=5)
                if body.get("status") == "ok":
                    print(f"server healthy in {time.time() - started:.1f}s at {self.base}")
                    return self
                last = f"health status={body.get('status')!r}"
            except (urllib.error.URLError, RuntimeError, OSError, ConnectionError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        self.error = f"llama-server not healthy within {self.boot_timeout:.0f}s (last: {last})"
        return self

    @property
    def ok(self) -> bool:
        return self.proc is not None and not self.error

    def __exit__(self, *exc_info) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
                    self.proc.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
                    pass
        try:
            self._fh.close()
        except (AttributeError, OSError):
            pass


# --------------------------------------------------------------------------
# check 1: tokenizer round-trip
# --------------------------------------------------------------------------


def check_roundtrip(
    prompt: str,
    oracle_prompt_ids: list[int],
    tok_bin: Path | None,
    model: Path,
    log: Path,
    server_base: str | None,
) -> int:
    """Exit 3 loudly if tokenizing the prompt does not reproduce prompt_ids."""
    banner("1. tokenizer round-trip  (tokenize(prompt) must == oracle prompt_ids)")
    print(f"prompt: {prompt!r}")
    print(f"oracle prompt_ids: {oracle_prompt_ids}")

    ids: list[int] | None = None
    reason = ""
    route = ""
    code = EXIT_PLUMBING
    if tok_bin is not None:
        route = f"llama-tokenize --ids --no-bos ({tok_bin})"
        try:
            ids, reason = tokenize_with_binary(tok_bin, model, prompt, log)
        except subprocess.TimeoutExpired:
            ids, reason = None, "llama-tokenize timed out"
        if ids is None:
            code, reason = tokenizer_failure(log, reason)
    if ids is None and server_base is not None:
        route = "llama-server /tokenize (live vocab)"
        ids, reason = tokenize_via_server(server_base, prompt)
        code = EXIT_PLUMBING
    if ids is None:
        return fail(
            code,
            "cannot tokenize the prompt -- no working tokenizer route",
            f"  llama-tokenize: {tok_bin or 'not found'}\n"
            f"  live server   : {server_base or 'not started'}\n"
            f"  reason        : {reason}\n"
            "The id comparison is meaningless without a tokenizer, so this run stops here.",
        )

    print(f"route: {route}")
    print(f"got   prompt_ids: {ids}")

    if ids != oracle_prompt_ids:
        detail = [
            "  The tokenizer does not reproduce the oracle prompt ids.",
            "  Every downstream id comparison would be measuring the wrong prefix, so",
            "  this run stops instead of reporting a phantom divergence.",
            "",
        ]
        for i in range(max(len(ids), len(oracle_prompt_ids))):
            expected = oracle_prompt_ids[i] if i < len(oracle_prompt_ids) else "<none>"
            actual = ids[i] if i < len(ids) else "<none>"
            mark = " " if expected == actual else "!"
            detail.append(f"   {mark} [{i:2d}] expected {expected!s:>8}  actual {actual!s:>8}")
        if len(ids) == len(oracle_prompt_ids) + 1:
            detail += [
                "",
                "  hint: exactly one extra leading id is almost always BOS.  llama-cli",
                "        has no --no-bos; pass",
                "        --override-kv tokenizer.ggml.add_bos_token=bool:false",
                "        (llama-tokenize has --no-bos, which this script already uses).",
            ]
        return fail(EXIT_PLUMBING, "TOKENIZER MISMATCH -- comparison is meaningless", "\n".join(detail))

    print("OK: prompt tokenization matches the oracle exactly")
    return EXIT_MATCH


# --------------------------------------------------------------------------
# route: server  (authoritative -- real generation ids)
# --------------------------------------------------------------------------


def generate_on_server(
    session: ServerSession,
    prompt_ids: list[int],
    n_predict: int,
    gen_timeout: float,
) -> tuple[list[int] | None, list[str] | None, str, int]:
    banner("2. generation via llama-server /completion  (route: server -- TRUE token ids)")
    payload = {
        # An integer array is used verbatim as token ids: no re-tokenizing, no
        # BOS, no add_special ambiguity.
        "prompt": list(prompt_ids),
        "n_predict": n_predict,
        **GREEDY_SERVER,
    }
    print(f"request: n_predict={n_predict}, temperature=0, n_probs=1, prompt=<{len(prompt_ids)} ids>")
    try:
        body = http_json("POST", f"{session.base}/completion", payload, timeout=gen_timeout)
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        print(f"/completion failed: {exc}")
        print(f"--- llama-server log tail ({session.log}) ---\n{tail(session.log)}")
        return None, None, "", EXIT_CANNOT_RUN

    for key in ("completion_probabilities", "probs"):
        entries = body.get(key)
        if isinstance(entries, list) and entries:
            ids = [e.get("id") for e in entries]
            pieces = [e.get("token", "") for e in entries]
            if all(isinstance(i, int) for i in ids):
                text = body.get("content", "") or "".join(pieces)
                print(f"server reported {len(ids)} generated token ids")
                return ids, pieces, text, EXIT_MATCH

    print("server response carried no per-token id list.")
    print("Expected completion_probabilities[].id; the request asked for n_probs=1.")
    print(f"response keys: {sorted(body.keys())}")
    print(f"--- response preview ---\n{json.dumps(body)[:1200]}")
    return None, None, "", EXIT_PLUMBING


# --------------------------------------------------------------------------
# route: cli  (fallback -- generated ids RECOVERED by re-tokenizing text)
# --------------------------------------------------------------------------

PROMPT_ID_RE = re.compile(r"^\s*(\d+)\s+->\s+'(.*)'\s*$", re.M)


def generate_on_cli(
    cli_bin: Path,
    tok_bin: Path | None,
    model: Path,
    prompt: str,
    prompt_ids: list[int],
    n_predict: int,
    gen_timeout: float,
    log: Path,
) -> tuple[list[int] | None, str, int]:
    """Return (retokenized ids, emitted text, exit code)."""
    banner("2. generation via llama-cli  (route: cli -- generated ids are RETOKENIZED)")
    cmd = [
        str(cli_bin),
        "-m", str(model),
        "-p", prompt,
        "-n", str(n_predict),
        # greedy: temp <= 0 collapses to argmax; neutralise everything upstream
        "--temp", "0",
        "--dynatemp-range", "0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--min-p", "0.0",
        "--xtc-probability", "0.0",
        "--typical", "1.0",
        "--repeat-penalty", "1.0",
        "--repeat-last-n", "0",
        "--dry-multiplier", "0.0",
        "-sp",                  # keep special tokens in the text: id-faithful
        "--no-display-prompt",
        "--no-show-timings",
        "-st",                  # single turn: never wait for stdin
        "--verbose-prompt",     # dumps the prompt ids the model actually saw
        # llama-cli has no --no-bos; the oracle used add_special_tokens=False.
        "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
    ]
    print(f"launching: {' '.join(cmd)}")
    with log.open("wb") as fh:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=fh,
            stdin=subprocess.DEVNULL,
            timeout=gen_timeout,
        )
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr_text = tail(log, lines=500)

    if proc.returncode != 0:
        print(f"llama-cli exited {proc.returncode}")
        print(f"--- llama-cli log tail ({log}) ---\n{tail(log)}")
        return None, "", EXIT_CANNOT_RUN

    # --verbose-prompt prints `%6d -> 'piece'` per prompt token: a real id dump
    # at the real generation site, and the cheapest proof that BOS is absent.
    seen = [int(m.group(1)) for m in PROMPT_ID_RE.finditer(stderr_text)]
    if seen:
        if seen == list(prompt_ids):
            print(f"llama-cli fed the model exactly the oracle prompt ids ({seen})")
        else:
            print(f"llama-cli fed the model prompt ids {seen}")
            print(f"oracle prompt ids               {list(prompt_ids)}")
            if len(seen) == len(prompt_ids) + 1:
                print("hint: one extra leading id -- that is BOS. Check --override-kv.")
            print("Bailing out: the CLI is not generating from the oracle prefix.")
            return None, "", EXIT_PLUMBING
    else:
        print("note: --verbose-prompt produced no parsable prompt-id lines; the prompt")
        print("      prefix could not be confirmed at the generation site")

    text = stdout
    print(f"llama-cli emitted {len(text)} chars of text")

    if tok_bin is None:
        print("no llama-tokenize next to --binary: generated ids cannot be recovered,")
        print("and text agreement is not acceptance.  Use the server route instead.")
        return None, text, EXIT_PLUMBING

    tok_log = log.with_suffix(log.suffix + ".tok")
    try:
        ids, reason = tokenize_with_binary(tok_bin, model, text, tok_log)
    except subprocess.TimeoutExpired:
        ids, reason = None, "llama-tokenize timed out"
    if ids is None:
        print(f"could not re-tokenize the emitted text: {reason}")
        return None, text, EXIT_PLUMBING

    if ids[: len(prompt_ids)] == list(prompt_ids):
        ids = ids[len(prompt_ids):]  # prompt echoed into the text; drop it
    print(f"RETOKENIZED {len(ids)} ids from the emitted text")
    return ids, text, EXIT_MATCH


# --------------------------------------------------------------------------
# route: logits  -- the re-cut acceptance (top-5 SET/ORDER + |delta| tolerance)
# --------------------------------------------------------------------------

# Searched in order when --reference is not given.  NONE of these exist today:
# the only reference top-5 numbers in the repo live in PROSE inside
# docs/dory-port-contract.md, which is not a machine reference and is
# deliberately not parsed here.
TOP5_REFERENCE_CANDIDATES = (
    "/Volumes/Widekeep/oracle/discriminating-prompt-top5.json",
    "/Volumes/Widekeep/oracle/discriminating-prompt-logits.json",
    "/Volumes/Widekeep/oracle/oracle-dory-top5.json",
)

# Reference id-list fields, in order of preference: the loop-DISCRIMINATING oracle
# stores ids_loops4 (the GGUF's dory.num_loops), the older one stores generated_ids.
EXPECTED_ID_FIELDS = ("ids_loops4", "expected_ids", "generated_ids")

# server-task.cpp logarithm(0) returns numeric_limits<float>::lowest() so the JSON
# stays finite; clamp anything absurd back to a usable "impossible" value.
LOGP_FLOOR = -1.0e4

PROB_KEYS = ("completion_probabilities", "probs")

UNITS_NOTE = """\
units: llama-server never returns raw logits.  get_token_probabilities()
       (tools/server/server-common.cpp) sorts the raw logits and then applies a
       FULL-VOCAB softmax; server-task.cpp emits log(p) as "logprob", and with
       post_sampling_probs=false those are PRE-sampling values, so the
       argmax-collapsing temperature sampler has not touched them.
       A logprobs reference is therefore compared directly.  A LOGITS reference
       is compared CENTRED (value_i - value_top), because
           logprob_i - logprob_top = logit_i - logit_top
       -- the log-sum-exp cancels, so the centred delta is the SAME number in
       both units.  What centring cannot recover is the per-position shift of
       the top logit itself, so a contract written as "|delta logit| <= 0.5"
       needs re-basing, not silent reuse: re-record the reference as logprobs
       (torch.log_softmax of the same logits) and keep 0.5, or dump the port's
       raw logits (llama-cli --save-all-logits) and compare logits to logits."""


def _as_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if val != val:  # NaN
        return None
    return val


def _fmt_top(top: list[tuple[int, float]]) -> str:
    return "  ".join(f"{tid}:{val:+.3f}" for tid, val in top)


def load_logits_oracle(path: Path) -> tuple[dict, str, list[int], list[int], str]:
    """Oracle for the logits route: prompt, prompt_ids and the ids it generated.

    Unlike load_oracle() this accepts ids_loops4, because the loop-discriminating
    oracle does not carry a `generated_ids` field.
    """
    if not path.is_file():
        raise SystemExit(fail(EXIT_PLUMBING, f"oracle not found: {path}"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(fail(EXIT_PLUMBING, f"cannot parse oracle {path}: {exc}"))
    if not isinstance(data, dict):
        raise SystemExit(fail(EXIT_PLUMBING, f"oracle {path} is not a JSON object"))

    prompt = data.get("prompt")
    prompt_ids = data.get("prompt_ids")
    if not isinstance(prompt, str):
        raise SystemExit(fail(EXIT_PLUMBING, f"oracle {path} has no 'prompt' string"))
    if not isinstance(prompt_ids, list) or not prompt_ids or not all(isinstance(i, int) for i in prompt_ids):
        raise SystemExit(fail(EXIT_PLUMBING, f"oracle {path} has no usable 'prompt_ids'"))

    expected: list[int] | None = None
    field = ""
    for name in EXPECTED_ID_FIELDS:
        cand = data.get(name)
        if isinstance(cand, list) and cand and all(isinstance(i, int) for i in cand):
            expected, field = cand, name
            break
    if expected is None:
        raise SystemExit(fail(
            EXIT_PLUMBING,
            f"oracle {path} has no expected id list",
            f"  looked for {EXPECTED_ID_FIELDS}.  The logits route needs the ids the\n"
            "  reference generated so it knows how many positions to compare.",
        ))
    return data, prompt, prompt_ids, expected, field


def _coerce_top(raw, where: str) -> list[tuple[int, float]]:
    if not isinstance(raw, list):
        raise ValueError(f"{where} is not a list")
    top: list[tuple[int, float]] = []
    for item in raw:
        if isinstance(item, dict):
            tid = item.get("id")
            val = None
            for key in ("logit", "logprob", "value", "logit_value"):
                if key in item:
                    val = _as_float(item[key])
                    if val is not None:
                        break
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            tid = item[0]
            val = _as_float(item[1])
        else:
            raise ValueError(f"{where}: candidate {item!r} is not an (id, value) pair")
        if not isinstance(tid, int) or val is None:
            raise ValueError(f"{where}: candidate {item!r} has no usable id/value")
        top.append((tid, val))
    if not top:
        raise ValueError(f"{where} is empty")
    return top


def load_reference(path: Path) -> dict:
    """Load the reference top-5.  Its units are MANDATORY, never guessed.

    Accepted shapes (all requiring "quantity": "logits" | "logprobs"):
        {"quantity": "logits", "positions": [{"position": 0, "top": [[id, v], ...]}, ...]}
        {"quantity": "logits", "positions": {"0": [[id, v], ...]}}
        {"quantity": "logits", "top5": [[[id, v], ...], ...]}          # index == position
    A candidate may be [id, value] or {"id": .., "logit"/"logprob"/"value": ..}.
    """
    if not path.is_file():
        raise SystemExit(fail(EXIT_PLUMBING, f"reference top-5 not found: {path}"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(fail(EXIT_PLUMBING, f"cannot parse reference {path}: {exc}"))
    if not isinstance(data, dict):
        raise SystemExit(fail(EXIT_PLUMBING, f"reference {path} is not a JSON object"))

    raw_quantity = data.get("quantity", data.get("units"))
    q = raw_quantity.strip().lower() if isinstance(raw_quantity, str) else ""
    if q in ("logit", "logits"):
        quantity = "logits"
    elif q in ("logprob", "logprobs", "log_softmax", "logsoftmax"):
        quantity = "logprobs"
    else:
        raise SystemExit(fail(
            EXIT_PLUMBING,
            f"reference {path} does not declare its units",
            '  Add  "quantity": "logits"  or  "quantity": "logprobs".  Without it the\n'
            "  comparison would silently mix units: the server reports LOGPROBS, so a\n"
            "  logits reference is compared CENTRED while a logprobs reference is not.",
        ))

    raw = data.get("positions", data.get("top5", data.get("top_logits")))
    if raw is None:
        raise SystemExit(fail(EXIT_PLUMBING, f"reference {path} has no 'positions' list or map"))

    positions: dict[int, list[tuple[int, float]]] = {}
    try:
        if isinstance(raw, dict):
            for key, value in raw.items():
                positions[int(str(key))] = _coerce_top(value, f"positions[{key}]")
        elif isinstance(raw, list):
            for idx, item in enumerate(raw):
                if isinstance(item, dict):
                    if "top" not in item:
                        raise ValueError(f"positions[{idx}] is an object with no 'top'")
                    positions[int(item.get("position", idx))] = _coerce_top(
                        item["top"], f"positions[{idx}].top")
                else:
                    positions[idx] = _coerce_top(item, f"positions[{idx}]")
        else:
            raise ValueError("'positions' is neither a list nor a map")
    except (ValueError, TypeError) as exc:
        raise SystemExit(fail(EXIT_PLUMBING, f"cannot read the reference top-5 in {path}: {exc}"))
    if not positions:
        raise SystemExit(fail(EXIT_PLUMBING, f"reference {path} contains no positions"))

    # The top-k must be sorted by value, descending: compare_position() anchors the
    # centred delta on entry [0] and treats the list order as the ranking.
    for idx, top in sorted(positions.items()):
        values = [val for _, val in top]
        if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
            raise SystemExit(fail(
                EXIT_PLUMBING,
                f"reference {path} position {idx} is not sorted by value (descending)",
                f"  values: {values}\n"
                "  The top-k order IS the RANKING this check compares, and the centred\n"
                "  delta is anchored on entry 0.  Sort the reference instead of letting\n"
                "  an unsorted list produce a phantom divergence.",
            ))

    expected_ids = None
    expected_id_field = ""
    for name in EXPECTED_ID_FIELDS:
        candidate = data.get(name)
        if isinstance(candidate, list) and candidate and all(isinstance(i, int) for i in candidate):
            expected_ids = candidate
            expected_id_field = name
            break

    return {
        "path": str(path),
        "quantity": quantity,
        "positions": positions,
        "provenance": data.get("source") or data.get("provenance") or data.get("model") or "(unstated)",
        "self_observed": bool(data.get("self_observed")),
        "prompt": data.get("prompt"),
        "prompt_ids": data.get("prompt_ids"),
        "expected_ids": expected_ids,
        "expected_id_field": expected_id_field,
    }


def reference_oracle_mismatch(
    reference: dict,
    prompt: str,
    prompt_ids: list[int],
    expected: list[int],
) -> str:
    """Return why an oracle and top-5 reference cannot be paired, or ``""``.

    A reference from another prompt still has a perfectly valid schema.  Without
    this check the server does several minutes of work before the two disjoint
    token sets produce an empty delta comparison.
    """
    ref_prompt_ids = reference.get("prompt_ids")
    if isinstance(ref_prompt_ids, list) and ref_prompt_ids != prompt_ids:
        return (
            f"  oracle prompt_ids ({len(prompt_ids)} ids) != reference prompt_ids "
            f"({len(ref_prompt_ids)} ids)\n"
            f"  oracle prompt   : {prompt!r}\n"
            f"  reference prompt: {reference.get('prompt')!r}"
        )
    ref_prompt = reference.get("prompt")
    if not isinstance(ref_prompt_ids, list) and isinstance(ref_prompt, str) and ref_prompt != prompt:
        return f"  oracle prompt {prompt!r} != reference prompt {ref_prompt!r}"

    ref_expected = reference.get("expected_ids")
    if isinstance(ref_expected, list) and ref_expected:
        field = reference.get("expected_id_field") or "expected ids"
        if len(ref_expected) < len(expected):
            return f"  reference {field} has only {len(ref_expected)} positions; {len(expected)} requested"
        if ref_expected[: len(expected)] != expected:
            first = next(
                (i for i, (actual, wanted) in enumerate(zip(ref_expected, expected)) if actual != wanted),
                0,
            )
            return (
                f"  oracle expected ids != reference {field} at position {first}: "
                f"{expected[first]} != {ref_expected[first]}"
            )
    return ""


def find_reference(explicit: str) -> tuple[Path | None, list[str]]:
    if explicit:
        return Path(explicit).expanduser(), []
    for cand in TOP5_REFERENCE_CANDIDATES:
        if Path(cand).is_file():
            return Path(cand), []
    return None, list(TOP5_REFERENCE_CANDIDATES)


def parse_positions(body: dict, want: int) -> tuple[list[dict] | None, str]:
    """Pull [{sampled, top:[(id, logprob)]}] out of one /completion response."""
    entries = None
    for key in PROB_KEYS:
        cand = body.get(key)
        if isinstance(cand, list) and cand:
            entries = cand
            break
    if entries is None:
        return None, (
            "the response carried no per-position probability list (looked for "
            f"{PROB_KEYS}); response keys: {sorted(body.keys())}"
        )

    parsed: list[dict] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return None, f"{PROB_KEYS[0]}[{i}] is not an object"
        raw = entry.get("top_logprobs")
        if not isinstance(raw, list) or not raw:
            if isinstance(entry.get("top_probs"), list):
                return None, (
                    f"{PROB_KEYS[0]}[{i}] returned 'top_probs', not 'top_logprobs': "
                    "post_sampling_probs is on, so those are post-sampling "
                    "probabilities rather than pre-sampling logprobs."
                )
            return None, (
                f"{PROB_KEYS[0]}[{i}] has no top_logprobs list even though n_probs was "
                "requested -- there is nothing to compare"
            )
        top: list[tuple[int, float]] = []
        for cand in raw[:want]:
            if not isinstance(cand, dict) or not isinstance(cand.get("id"), int):
                return None, f"{PROB_KEYS[0]}[{i}].top_logprobs entry {cand!r} is malformed"
            val = _as_float(cand.get("logprob"))
            if val is None:
                return None, f"{PROB_KEYS[0]}[{i}] candidate {cand.get('id')} carries no logprob"
            top.append((cand["id"], max(val, LOGP_FLOOR)))
        sampled = entry.get("id")
        parsed.append({"sampled": sampled if isinstance(sampled, int) else None, "top": top})
    return parsed, ""


def logits_request(
    session: ServerSession,
    prompt_ids: list[int],
    n_predict: int,
    n_probs: int,
    timeout: float,
) -> tuple[dict | None, str, int]:
    payload = {
        # INTEGER ARRAY: llama-server uses these ids verbatim, so it cannot
        # re-tokenize and cannot prepend BOS.  This is the whole point.
        "prompt": list(prompt_ids),
        "n_predict": n_predict,
        **GREEDY_SERVER,
        "n_probs": n_probs,
        # false keeps /completion in PRE-sampling mode: the candidates come from
        # the raw logits, before the temp<=0 sampler collapses them to one-hot.
        "post_sampling_probs": False,
    }
    try:
        body = http_json("POST", f"{session.base}/completion", payload, timeout=timeout)
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        return None, str(exc), EXIT_CANNOT_RUN
    return body, "", EXIT_MATCH


def collect_chain(
    session: ServerSession,
    prompt_ids: list[int],
    expected: list[int],
    n_probs: int,
    timeout: float,
) -> tuple[list[dict] | None, list[int] | None, str, int]:
    """One request, n_predict = len(expected): the positions the port reaches."""
    body, err, code = logits_request(session, prompt_ids, len(expected), n_probs, timeout)
    if body is None:
        return None, None, err, code
    parsed, err = parse_positions(body, n_probs)
    if parsed is None:
        return None, None, err, EXIT_PLUMBING
    ids = [p["sampled"] for p in parsed]
    print(f"server returned {len(parsed)} positions, {len(parsed[0]['top'])} candidates each")
    return parsed, ids, "", EXIT_MATCH


def collect_teacher_forced(
    session: ServerSession,
    prompt_ids: list[int],
    expected: list[int],
    n_probs: int,
    timeout: float,
) -> tuple[list[dict] | None, list[int] | None, str, int]:
    """One 1-token request per position, context = prompt_ids + expected[:i].

    This is the only way to compare positions AFTER an id divergence (feeding the
    reference's ids instead of the port's), which is what makes "the top-5 at
    every position" mean every position rather than "every position up to the
    first flip".
    """
    out: list[dict] = []
    for idx in range(len(expected)):
        ctx_ids = list(prompt_ids) + list(expected[:idx])
        body, err, code = logits_request(session, ctx_ids, 1, n_probs, timeout)
        if body is None:
            return None, None, f"position {idx}: {err}", code
        parsed, err = parse_positions(body, n_probs)
        if parsed is None:
            return None, None, f"position {idx}: {err}", EXIT_PLUMBING
        out.append(parsed[0])
        print(f"  position {idx:2d}: {_fmt_top(parsed[0]['top'])}")
    return out, [p["sampled"] for p in out], "", EXIT_MATCH


def compare_position(
    actual: list[tuple[int, float]],
    expected: list[tuple[int, float]],
    quantity: str,
    tol: float,
) -> dict:
    """Compare one position's top-k.  `actual` is logprobs, `expected` is `quantity`."""
    exp_ids = [tid for tid, _ in expected]
    act_ids = [tid for tid, _ in actual]

    set_match = set(exp_ids) == set(act_ids)
    order_match = (act_ids == exp_ids) if len(act_ids) == len(exp_ids) else act_ids[: len(exp_ids)] == exp_ids

    exp_map = dict(expected)
    act_map = dict(actual)
    missing = [tid for tid in exp_ids if tid not in act_map]

    if quantity == "logprobs":
        delta_mode = "logprob vs logprob (same units)"
        deltas = {tid: act_map[tid] - exp_map[tid] for tid in exp_ids if tid in act_map}
    else:
        # Centring cancels the log-sum-exp -- the only thing separating raw logits
        # from the logprobs the server reports -- so this delta is the exact
        # logit delta up to the position's (unobservable) top-logit shift.
        delta_mode = "centred (value_i - value_top); exact across logits<->logprobs"
        exp_top = expected[0][1]
        act_top = actual[0][1]
        deltas = {
            tid: (act_map[tid] - act_top) - (exp_map[tid] - exp_top)
            for tid in exp_ids
            if tid in act_map
        }

    max_delta = max((abs(v) for v in deltas.values()), default=float("inf"))
    return {
        "set_match": set_match,
        "order_match": order_match,
        "missing": missing,
        "unexpected": [tid for tid in act_ids if tid not in exp_map],
        "deltas": deltas,
        "max_delta": max_delta,
        "delta_mode": delta_mode,
        "argmax_match": bool(act_ids) and exp_ids[0] == act_ids[0],
        "verdict": bool(set_match and order_match and not missing and max_delta <= tol),
    }


def dump_observed(path: Path, positions: list[list[tuple[int, float]]], meta: dict) -> Path:
    """Write what the port emitted.  Labelled NOT a reference so it cannot be
    laundered into an oracle: it is the artifact under test."""
    payload = {
        "quantity": "logprobs",
        "self_observed": True,
        "source": "llama-server /completion on the PORT UNDER TEST -- not a reference",
        **meta,
        "positions": [
            {"position": idx, "top": [[tid, val] for tid, val in top]}
            for idx, top in enumerate(positions)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def report_logits(
    rows: list[dict],
    reference: dict | None,
    tol: float,
    mode: str,
    oracle_path: Path,
    id_field: str,
    observed: list[list[tuple[int, float]]],
    dump_path: Path | None,
) -> int:
    banner("4. per-position top-5 comparison  (the re-cut acceptance)")
    print(f"oracle    : {oracle_path}   (expected ids from '{id_field}')")
    print(f"positions : {mode}")
    if reference is None:
        print(f"reference : (none -- no top-5 reference file on disk)")
    else:
        print(f"reference : {reference['path']}")
        print(f"            quantity={reference['quantity']}  "
              f"positions={len(reference['positions'])}  provenance={reference['provenance']}")
        if reference["self_observed"]:
            print("            WARNING: this file says self_observed=true -- it is the port's own")
            print("            output, not a reference. Comparing it to itself proves nothing.")
    print(f"tolerance : |delta| <= {tol}")

    if reference is None:
        print()
        print(UNITS_NOTE)
        print()
        print("OBSERVED top-5 per position (port under test):")
        for idx, top in enumerate(observed):
            print(f"  [{idx:2d}] {_fmt_top(top)}")
        if dump_path is not None:
            print(f"\nwrote {dump_path}")
        print()
        print("NO REFERENCE TOP-5 ON DISK -- THIS RUN IS DATA, NOT A VERDICT.")
        print("Nothing here says the port is right or wrong: SET/ORDER/delta can only be")
        print("judged against a BF16 reference.  Looked for:")
        for cand in TOP5_REFERENCE_CANDIDATES:
            print(f"  {cand}  (missing)")
        print()
        print("To turn this into a verdict, produce that file from the BF16 weights with")
        print('"quantity": "logprobs" and the top-5 per position, e.g. in the oracle venv:')
        print("    logits = model(prompt_ids).logits          # [T, vocab]")
        print("    lp = torch.log_softmax(logits.float(), -1)  # same units as llama-server")
        print("    for pos in range(n): top5 = lp[last+pos].topk(5)")
        print("then re-run with --reference <that file>.  Recording it as logprobs keeps")
        print("the 0.5 threshold in the same units as the port's measurement.")
        return fail(
            EXIT_PLUMBING,
            "NO REFERENCE -- not a pass and not a failure; the comparison never happened",
            "  The port's own top-5 was printed and dumped above.  Do not read it as acceptance.",
        )

    quantity = reference["quantity"]
    if quantity == "logprobs":
        print("delta mode: logprob vs logprob (same units -- 0.5 means what it says)")
    else:
        print("delta mode: CENTRED value_i - value_top (reference is raw logits, server")
        print("            reports logprobs; centring cancels the log-sum-exp)")
    print()
    print(UNITS_NOTE)
    print()

    failures: list[dict] = []
    compared = 0
    skipped: list[dict] = []
    any_delta_mode = ""

    for row in rows:
        idx = row["position"]
        actual = row["actual"]
        if actual is None or row["why"]:
            skipped.append(row)
            print(f"[{idx:2d}] SKIPPED  {row['why'] or 'no data'}")
            continue
        expected_top = reference["positions"].get(idx)
        if expected_top is None:
            skipped.append({**row, "why": "position not present in the reference"})
            print(f"[{idx:2d}] SKIPPED  position not present in the reference")
            continue

        cmp = compare_position(actual, expected_top, quantity, tol)
        any_delta_mode = cmp["delta_mode"]
        if not cmp["deltas"]:
            why = "no shared token ids between the reference and observed top-k; delta comparison is empty"
            skipped.append({**row, "why": why})
            print(f"[{idx:2d}] SKIPPED  EMPTY COMPARISON: {why}")
            print(f"      expected: {_fmt_top(expected_top)}")
            print(f"      actual  : {_fmt_top(actual)}")
            continue
        compared += 1
        tag = "PASS" if cmp["verdict"] else "FAIL"
        print(f"[{idx:2d}] {tag}  set={'match' if cmp['set_match'] else 'MISMATCH':8s} "
              f"order={'match' if cmp['order_match'] else 'MISMATCH':8s} "
              f"max|d|={cmp['max_delta']:.3f}  argmax={'ok' if cmp['argmax_match'] else 'differs'}")
        print(f"      expected: {_fmt_top(expected_top)}")
        print(f"      actual  : {_fmt_top(actual)}")
        if not cmp["verdict"]:
            failures.append({"position": idx, "cmp": cmp, "actual": actual,
                             "expected": expected_top, "row": row})

    print()
    if any_delta_mode:
        print(f"deltas are {any_delta_mode}")
    print(f"compared {compared} position(s); skipped {len(skipped)}; failed {len(failures)}")

    for item in failures:
        idx = item["position"]
        cmp = item["cmp"]
        print(f"\n--- position {idx} failed ---")
        if cmp["missing"]:
            print(f"  reference ids missing from the port's top-5: {cmp['missing']}")
        if cmp["unexpected"]:
            print(f"  ids in the port's top-5 that the reference does not have: {cmp['unexpected']}")
        if not cmp["order_match"]:
            print(f"  expected order: {[tid for tid, _ in item['expected']]}")
            print(f"  actual   order: {[tid for tid, _ in item['actual']]}")
        for tid, delta in sorted(cmp["deltas"].items(), key=lambda kv: -abs(kv[1]))[:5]:
            print(f"  id {tid:>7}: delta {delta:+.3f}")
        if cmp["max_delta"] > tol and cmp["deltas"]:
            worst = max(cmp["deltas"].items(), key=lambda kv: abs(kv[1]))
            print(f"  worst |delta| = {cmp['max_delta']:.3f} on id {worst[0]}, over the "
                  f"{tol} tolerance")
            print("  NOTE: a delta beyond 0.5 is Q5_K_M noise exceeding the contract's own")
            print("        Q5_K_M noise floor -- that is a statement about the threshold and")
            print("        the quantisation, not automatically about the graph.")

    if skipped:
        print()
        for row in skipped:
            print(f"  position {row['position']} not compared: {row['why']}")

    if compared == 0:
        print()
        print("VERDICT: INCOMPLETE -- no position produced a non-empty numerical comparison.")
        return fail(
            EXIT_PLUMBING,
            "EMPTY COMPARISON -- loud data, not a model verdict",
            "  Check that the oracle and reference describe the same prompt and token-id space.",
        )

    if failures:
        print()
        print("VERDICT: FAIL -- at least one position breaks the re-cut acceptance")
        print("(SET, ORDER and |delta| <= tolerance).  Report the position above; do not")
        print("relax the check and do not compare text.")
        return EXIT_MISMATCH

    if skipped:
        print()
        print("VERDICT: INCOMPLETE -- every COMPARED position passed, but "
              f"{len(skipped)} position(s) could")
        print("not be compared, so this is neither a pass nor a failure.  Re-run with")
        print("--teacher-forced to cover every position from the reference's own context.")
        return fail(EXIT_PLUMBING, "COMPARISON INCOMPLETE -- not a pass and not a failure")

    print()
    print(f"VERDICT: PASS -- {compared} positions, same top-5 SET and ORDER, "
          f"max|delta| <= {tol}")
    if quantity == "logits":
        print("Caveat: the reference is in RAW LOGITS and the server reports LOGPROBS, so the")
        print("delta above is centred.  The absolute per-candidate logit delta also contains")
        print("the position's top-logit shift, which /completion cannot reveal.  Re-record the")
        print("reference as logprobs if you want the contract's 0.5 to mean exactly 0.5.")
    return EXIT_MATCH


def report_argmax_agreement(
    ids: list[int | None],
    expected: list[int],
    loops1: list[int] | None,
    id_field: str,
    teacher_forced: bool,
) -> None:
    """How many positions the port's argmax agrees with each reference variant.

    This is the one comparison that is still meaningful with NO top-5 reference:
    under a fixed (teacher-forced) context, argmax agreement with ids_loops4 vs
    ids_loops1 says which loop count the graph is actually running.  One flipped
    id is not evidence of anything -- a 0.008-logit reference margin is far below
    the quantisation noise -- so the COUNT is what carries the signal.
    """
    span = min(len(ids), len(expected))
    if span <= 0:
        return
    agree4 = sum(1 for i in range(span) if ids[i] == expected[i])
    print(f"argmax agreement over {span} positions ({id_field}): {agree4}/{span}")
    if not loops1:
        return
    span1 = min(span, len(loops1))
    agree1 = sum(1 for i in range(span1) if ids[i] == loops1[i])
    print(f"                          vs ids_loops1: {agree1}/{span1}")
    flipped = [i for i in range(span) if ids[i] != expected[i]]
    if flipped:
        print(f"diverged at position(s): {flipped}")
        if not teacher_forced:
            print("  (single pass: every position after the first flip sits on the PORT's own")
            print("   context, so these counts drift. --teacher-forced is the clean measurement.)")
    if teacher_forced and agree4 > agree1:
        print(f"hint: the graph follows the loops=4 reference ({agree4}/{span} vs {agree1}/{span1}).")
        print("      A flip where both candidates appear in the top-5 with a sub-noise margin is")
        print("      a quantisation tie, NOT a loop-count bug.")
    elif teacher_forced and agree1 > agree4:
        print(f"hint: the graph follows the LOOPS=1 reference ({agree1}/{span1} vs {agree4}/{span}).")
        print("      Check that dory.num_loops is read from the GGUF and used.")


def run_logits(args, binary: Path, model: Path, oracle_path: Path) -> int:
    """--route logits: the re-cut acceptance (top-5 SET/ORDER + |delta| tolerance)."""
    server_bin = sibling(binary, "llama-server")
    if server_bin is None:
        return fail(
            EXIT_CANNOT_RUN,
            "route=logits but no llama-server next to --binary",
            f"looked in {binary.parent} (and on PATH)",
        )
    tok_bin = sibling(binary, "llama-tokenize")

    oracle_data, prompt, prompt_ids, expected, id_field = load_logits_oracle(oracle_path)

    n = args.n
    if n > len(expected):
        print(f"note: --n {n} exceeds the {len(expected)} ids in {id_field}; comparing {len(expected)}")
        n = len(expected)
    expected_n = expected[:n]

    reference_path, missing_candidates = (None, list(TOP5_REFERENCE_CANDIDATES)) \
        if args.no_reference else find_reference(args.reference)
    reference = load_reference(reference_path) if reference_path is not None else None
    if reference is not None:
        mismatch = reference_oracle_mismatch(reference, prompt, prompt_ids, expected_n)
        if mismatch:
            return fail(
                EXIT_PLUMBING,
                "ORACLE/REFERENCE MISMATCH -- no numerical verdict is possible",
                mismatch + "\n  Pass the matching --oracle JSON; comparing different prompts is not model evidence.",
            )
    n_probs = args.n_probs
    if reference is not None:
        widest = max(len(top) for top in reference["positions"].values())
        n_probs = max(n_probs, widest)

    print(f"server : {server_bin}")
    print(f"tokenizer: {tok_bin or '(not built -- the live server /tokenize will be used)'}")
    print(f"expected ids ({id_field}, {n} of {len(expected)}): {expected_n}")
    if reference is None:
        print("reference: NONE on disk -- this run will report observed data and exit 3")
        for cand in missing_candidates:
            print(f"  missing: {cand}")
    else:
        print(f"reference: {reference['path']} (quantity={reference['quantity']})")
    print(f"n_probs: {n_probs}   tol: {args.tol}   ctx: {args.ctx}   ngl: {args.ngl}")

    log_dir = Path(tempfile.mkdtemp(prefix="dory-oracle-"))
    print(f"logs   : {log_dir}" + ("" if args.keep_logs else "  (deleted on exit; pass --keep-logs to keep)"))

    try:
        with ServerSession(
            server_bin, model, args.port, args.boot_timeout,
            log_dir / "llama-server.log", ctx=args.ctx,
            extra_args=("-ngl", str(args.ngl)),
        ) as session:
            if not session.ok:
                print(f"server did not come up: {session.error}")
                print(f"--- llama-server log tail ---\n{tail(session.log, 60)}")
                return EXIT_CANNOT_RUN

            # Acceptance item 1: the ids must be exactly the oracle's, and those
            # exact ids are what gets posted (as an array) below.
            rc = check_roundtrip(prompt, prompt_ids, tok_bin, model,
                                 log_dir / "llama-tokenize.log", session.base)
            if rc != EXIT_MATCH:
                return rc

            banner(f"2. collection  (llama-server /completion, n_probs={n_probs}, temperature=0)")
            if args.teacher_forced:
                mode = "teacher-forced: one 1-token request per position, context = prompt_ids + expected[:i]"
                print(f"mode: {mode}")
                print("note: each request re-prefills the whole context, so these numerics are")
                print("      NOT batch-shape identical to a single decoding pass.  Measured on")
                print("      this port: teacher-forced is bit-reproducible run to run, but its")
                print("      ranks 3-5 differ from the single pass by up to ~0.5 logprob on a")
                print("      token of probability 1e-4.  Produce the BF16 reference with the SAME")
                print("      shape (one full-prefill forward per position) or the delta measures")
                print("      batching, not the graph.")
                collected, ids, err, code = collect_teacher_forced(
                    session, prompt_ids, expected_n, n_probs, args.gen_timeout)
            else:
                mode = f"single pass: one request, n_predict={n}"
                print(f"mode: {mode}")
                collected, ids, err, code = collect_chain(
                    session, prompt_ids, expected_n, n_probs, args.gen_timeout)
            if collected is None:
                print(f"--- llama-server log tail ({session.log}) ---\n{tail(session.log, 60)}")
                return fail(code, "cannot read top-5 candidates from llama-server", err)

            rows: list[dict] = []
            diverged_at: int | None = None
            for idx in range(n):
                actual = collected[idx]["top"] if idx < len(collected) else None
                why = ""
                if actual is None:
                    why = "the server returned fewer positions than requested"
                elif not args.teacher_forced and diverged_at is not None and idx > diverged_at:
                    why = (f"not comparable: the generated ids diverged at position {diverged_at}, "
                           "so this position sits on a different context than the reference")
                rows.append({"position": idx, "actual": actual, "why": why})
                if (diverged_at is None and ids is not None and idx < len(ids)
                        and ids[idx] is not None and idx < len(expected_n)
                        and ids[idx] != expected_n[idx]):
                    diverged_at = idx

            if ids is not None:
                banner("3. argmax per position  (informational -- the acceptance is SET/ORDER/delta)")
                if args.teacher_forced:
                    print("context: prompt_ids + the reference's own ids, at every position")
                else:
                    print(f"expected: {expected_n}")
                    print(f"actual  : {ids}")
                loops1 = oracle_data.get("ids_loops1")
                report_argmax_agreement(
                    ids, expected_n, loops1 if isinstance(loops1, list) else None,
                    id_field, args.teacher_forced,
                )

            observed = [entry["top"] for entry in collected[:n]]
            dump_path = None
            if args.dump_observed:
                dump_path = Path(args.dump_observed).expanduser()
            elif reference is None:
                dump_path = Path(tempfile.gettempdir()) / "dory-observed-top5.json"
            if dump_path is not None:
                dump_observed(dump_path, observed, {
                    "model": str(model),
                    "oracle": str(oracle_path),
                    "prompt_ids": prompt_ids,
                    "expected_ids": expected_n,
                    "mode": mode,
                    "n_probs": n_probs,
                })

            return report_logits(rows, reference, args.tol, mode, oracle_path,
                                 id_field, observed, dump_path)
    finally:
        if not args.keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# comparison + report
# --------------------------------------------------------------------------


def compare(actual: list[int], expected: list[int], n: int) -> tuple[bool, int]:
    expected_n = expected[:n]
    for i in range(min(len(expected_n), len(actual))):
        if actual[i] != expected_n[i]:
            return False, i
    if len(actual) < len(expected_n):
        return False, len(actual)
    return True, -1


def report(
    actual: list[int],
    expected: list[int],
    n: int,
    route: str,
    pieces: list[str] | None,
    actual_text: str,
    oracle: dict,
    server_base: str | None,
) -> int:
    banner("3. comparison  (position by position, ids not text)")
    expected_n = expected[:n]
    print(f"route     : {route}")
    if route == "cli":
        print("            generated ids were RECOVERED by re-tokenizing emitted text;")
        print("            a boundary artifact is possible. Use --route server to confirm.")
    print(f"expected  : {expected_n}")
    print(f"actual    : {actual}")

    matched, idx = compare(actual, expected_n, n)

    if matched:
        print(f"\nMATCH: all {len(expected_n)} positions agree.")
        if route == "cli":
            print("(matched after re-tokenization -- confirm with --route server before")
            print(" declaring the port correct)")
        print(f"generated text: {actual_text!r}")
        return EXIT_MATCH

    print(f"\nMISMATCH at index {idx}")
    exp_at = expected_n[idx] if idx < len(expected_n) else "<end of oracle>"
    act_at = actual[idx] if idx < len(actual) else "<end of generation>"
    print(f"  index        : {idx}")
    print(f"  expected id  : {exp_at}")
    print(f"  actual   id  : {act_at}")
    print(f"  ids before   : expected {expected_n[:idx]}")
    print(f"                 actual   {actual[:idx]}")

    if pieces and idx < len(pieces):
        lo = max(0, idx - 2)
        window = ", ".join(f"[{i}]={pieces[i]!r}" for i in range(lo, min(len(pieces), idx + 2)))
        print(f"  actual pieces around divergence: {window}")

    exp_dec = detok(server_base, expected_n[: idx + 1]) if idx < len(expected_n) else None
    act_dec = detok(server_base, actual[: idx + 1])
    print(f"  decoded expected through {idx}: "
          + (repr(exp_dec) if exp_dec is not None else "(unavailable: no live server)"))
    print(f"  decoded actual   through {idx}: "
          + (repr(act_dec) if act_dec is not None else "(unavailable: no live server)"))
    print(f"  full actual text: {actual_text!r}")
    print(f"  oracle   text   : {oracle.get('generated_text', '')!r}")

    print("\nThe ids differ, so the port is wrong -- do not relax the check, do not")
    print("compare text, do not accept 'plausible'.  First divergence is the index above.")
    if route == "cli":
        print("Because this route re-tokenizes, confirm with --route server before")
        print("attributing the divergence to the model graph.")
    return EXIT_MISMATCH


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a llama.cpp build against the dory token oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="llama-cli the build produced")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF under test")
    parser.add_argument("--oracle", default=DEFAULT_ORACLE, help="oracle JSON")
    parser.add_argument("--n", type=int, default=24, help="number of generated ids to compare")
    parser.add_argument(
        "--route",
        choices=("auto", "server", "cli", "logits"),
        default="auto",
        help="server = llama-server /completion (true ids); cli = llama-cli + re-tokenize; "
             "logits = top-5 (id, logprob) per position vs a reference (the re-cut acceptance)",
    )
    parser.add_argument("--port", type=int, default=0, help="server port (0 = pick a free one)")
    parser.add_argument("--boot-timeout", type=float, default=900.0, help="seconds to wait for the server")
    parser.add_argument("--gen-timeout", type=float, default=1800.0, help="seconds to wait for generation")
    parser.add_argument("--keep-logs", action="store_true", help="keep the captured logs on disk")

    logits = parser.add_argument_group(
        "--route logits",
        "top-5 SET/ORDER/|delta| comparison (docs/dory-port-contract.md, ACCEPTANCE)",
    )
    logits.add_argument(
        "--reference",
        default="",
        help="reference top-5 JSON. Must declare \"quantity\": logits|logprobs. "
             "Default: search " + ", ".join(TOP5_REFERENCE_CANDIDATES),
    )
    logits.add_argument("--tol", type=float, default=0.5,
                        help="per-position |delta| tolerance, in the reference's units "
                             "(centred if the reference is in logits)")
    logits.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the reference entirely and report only the port's observed top-5 "
             "(exit 3: data, not a verdict). Also how the no-reference path is tested.",
    )
    logits.add_argument("--n-probs", type=int, default=5,
                        help="top-k candidates to request and compare per position")
    logits.add_argument(
        "--teacher-forced",
        action="store_true",
        help="one 1-token request per position with context = prompt_ids + expected[:i]. "
             "Required to compare positions AFTER the port's ids diverge (default: single pass)",
    )
    logits.add_argument("--ctx", type=int, default=64, help="server -c for the logits route")
    logits.add_argument("--ngl", type=int, default=0, help="server -ngl for the logits route")
    logits.add_argument(
        "--dump-observed",
        default="",
        help="write the port's observed top-5 here (default: <tmp>/dory-observed-top5.json "
             "when there is no reference). Labelled self_observed -- it is NOT a reference.",
    )
    args = parser.parse_args(argv)

    binary = Path(args.binary).expanduser().resolve()
    model = Path(args.model).expanduser()
    oracle_path = Path(args.oracle).expanduser()

    banner("dory oracle check")
    print(f"binary : {binary}")
    print(f"model  : {model}")
    print(f"oracle : {oracle_path}")
    print(f"n      : {args.n}   route: {args.route}")

    if not binary.is_file() or not os.access(binary, os.X_OK):
        return fail(
            EXIT_CANNOT_RUN,
            f"binary is missing or not executable: {binary}",
            "The build has not produced it yet (cmake --build build --target llama-cli).",
        )
    if not model.is_file():
        return fail(EXIT_CANNOT_RUN, f"model is missing: {model}")

    if args.route == "logits":
        return run_logits(args, binary, model, oracle_path)

    oracle = load_oracle(oracle_path)
    prompt: str = oracle["prompt"]
    prompt_ids: list[int] = oracle["prompt_ids"]
    expected: list[int] = oracle["generated_ids"]
    n = args.n
    if n > len(expected):
        print(f"note: --n {n} exceeds the {len(expected)} ids in the oracle; comparing {len(expected)}")
        n = len(expected)

    server_bin = sibling(binary, "llama-server")
    tok_bin = sibling(binary, "llama-tokenize")

    route = args.route
    if route == "auto":
        route = "server" if server_bin is not None else "cli"
        if server_bin is None:
            print("note: no llama-server next to --binary; falling back to the cli route")
    if route == "server" and server_bin is None:
        return fail(
            EXIT_CANNOT_RUN,
            "route=server but no llama-server next to --binary",
            f"looked in {binary.parent}",
        )

    log_dir = Path(tempfile.mkdtemp(prefix="dory-oracle-"))
    print(f"logs   : {log_dir}" + ("" if args.keep_logs else "  (deleted on exit; pass --keep-logs to keep)"))

    try:
        if route == "server":
            with ServerSession(server_bin, model, args.port, args.boot_timeout,
                               log_dir / "llama-server.log") as session:
                if not session.ok:
                    print(f"server did not come up: {session.error}")
                    print(f"--- llama-server log tail ---\n{tail(session.log, 60)}")
                    return EXIT_CANNOT_RUN

                # Same vocab, same build, live: the strongest place to prove the
                # prompt ids before generating from them.
                rc = check_roundtrip(prompt, prompt_ids, tok_bin, model,
                                     log_dir / "llama-tokenize.log", session.base)
                if rc != EXIT_MATCH:
                    return rc

                actual, pieces, text, rc = generate_on_server(session, prompt_ids, n, args.gen_timeout)
                if actual is None:
                    return rc
                return report(actual, expected, n, "server", pieces, text, oracle, session.base)

        rc = check_roundtrip(prompt, prompt_ids, tok_bin, model,
                             log_dir / "llama-tokenize.log", None)
        if rc != EXIT_MATCH:
            return rc
        actual, text, rc = generate_on_cli(
            binary, tok_bin, model, prompt, prompt_ids, n, args.gen_timeout,
            log_dir / "llama-cli.log",
        )
        if actual is None:
            return rc
        return report(actual, expected, n, "cli", None, text, oracle, None)
    finally:
        if not args.keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

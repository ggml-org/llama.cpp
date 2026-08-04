#!/usr/bin/env python3
"""
Wave 5 correctness gate: compare greedy decode token streams between the
flash_attn_ext path and the tessera paged path on the SAME model + prompt.

Runs llama-server twice (once per path), captures the greedy (temp=0, seed=0)
token ids via the /completion endpoint with logprobs, then reports the token-id
agreement percentage.

Usage:
  python3 run_correctness.py <server_bin> <model> <label-tag> [n_predict]

Env:
  TESSERA_PAGED_ATTN  -> "1" forces paged (we set this internally for the paged run)
  EXTRA_FLASH_ARGS    -> extra CLI args appended to the flash run (space-separated)
  EXTRA_PAGED_ARGS    -> extra CLI args appended to the paged run
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

def wait_for_server(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False

# Stable, ~90-token factual prompt (same family as the w3/w4 harness).
PROMPT = (
    "The history of computing hardware spans many decades. Early mechanical "
    "calculators gave way to vacuum tube machines, then transistors, and "
    "finally integrated circuits. A key question for modern systems is: "
    "when does reducing numerical precision harm results? Explain the "
    "trade-off between memory savings and accuracy in neural network inference, "
    "and give one concrete example."
)

def run_server(bin_path, model, n_predict, port, paged, extra_args, tag):
    env = os.environ.copy()
    if paged:
        env["TESSERA_PAGED_ATTN"] = "1"
    else:
        env.pop("TESSERA_PAGED_ATTN", None)
    cmd = [
        bin_path, "-m", model, "-ctk", "f16", "-ctv", "f16", "-kvu",
        "-ngl", "999", "--port", str(port), "--host", "127.0.0.1",
        "-c", "512", "-np", "1", "-t", "4",
        "--no-embedded-mtp", "--log-disable",
    ]
    if extra_args:
        cmd.extend(extra_args.split())
    label = f"{tag}-{'PAGED' if paged else 'FLASH'}"
    print(f"[{label}] starting server paged={paged} args={extra_args}", flush=True)
    logf = open(f"/tmp/w5-{label}.serverlog", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_for_server(port):
            print(f"[{label}] SERVER DID NOT START", flush=True)
            logf.close()
            with open(f"/tmp/w5-{label}.serverlog") as f:
                print(f.read()[-2000:])
            return None
        body = json.dumps({
            "prompt": PROMPT,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "stream": False,
            "seed": 0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        # Extract per-step chosen token ids. The /completion endpoint returns
        # either completion_probabilities (with token ids) or we fall back to
        # tokens_predicted. We want the actual sampled token ids in order.
        toks = []
        ce = resp.get("completion_probabilities") or []
        for step in ce:
            if isinstance(step, dict):
                # newer schema: {id, token, ...}
                if "id" in step:
                    toks.append(step["id"])
                elif "token" in step and isinstance(step["token"], int):
                    toks.append(step["token"])
            elif isinstance(step, list) and step:
                top = step[0]
                if "id" in top:
                    toks.append(top["id"])
        result = {
            "label": label,
            "tokens": toks,
            "content": resp.get("content", ""),
            "timings_s": dt,
            "n_predicted": resp.get("tokens_predicted", -1),
        }
        print(f"[{label}] {len(toks)} token ids captured in {dt:.1f}s", flush=True)
        if toks:
            print(f"[{label}] first 12 ids: {toks[:12]}", flush=True)
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()

def main():
    bin_path = sys.argv[1]
    model = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "run"
    n_predict = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    flash_extra = os.environ.get("EXTRA_FLASH_ARGS", "")
    paged_extra = os.environ.get("EXTRA_PAGED_ARGS", "")
    # Run flash first (reference), then paged.
    flash = run_server(bin_path, model, n_predict, 8781, False, flash_extra, tag)
    time.sleep(3)
    paged = run_server(bin_path, model, n_predict, 8781, True, paged_extra, tag)
    if flash is None or paged is None:
        print("ERROR: one of the runs failed to start")
        sys.exit(2)
    btoks = flash["tokens"]
    ctoks = paged["tokens"]
    n = min(len(btoks), len(ctoks))
    if n == 0:
        print("ERROR: no tokens captured (server did not return token ids)")
        print(f"flash content: {flash['content'][:100]!r}")
        print(f"paged content: {paged['content'][:100]!r}")
        sys.exit(2)
    match = sum(1 for i in range(n) if btoks[i] == ctoks[i])
    pct = 100.0 * match / n
    print("=" * 60)
    print(f"TOKEN AGREEMENT (flash vs paged): {match}/{n} = {pct:.1f}%")
    print(f"flash first 20 ids: {btoks[:20]}")
    print(f"paged first 20 ids: {ctoks[:20]}")
    # show first divergence
    for i in range(n):
        if btoks[i] != ctoks[i]:
            print(f"first divergence at token {i}: flash={btoks[i]} paged={ctoks[i]}")
            break
    print(f"flash text[:120]:  {flash['content'][:120]!r}")
    print(f"paged text[:120]: {paged['content'][:120]!r}")
    # write result json
    out = {"flash": flash, "paged": paged, "agreement_pct": pct, "match": match, "n": n}
    with open(f"/tmp/w5-{tag}-result.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()

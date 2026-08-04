#!/usr/bin/env python3
"""
S4 correctness gate: logit/text comparison vs f16 KV baseline.

Starts llama-server with a given KV cache type, sends a fixed prompt, captures
greedy decode (temp=0) tokens + per-step top-token logprob. Run once per KV
type, save the JSON, then compare with correctness_compare.py.

Usage:
  TESSERA_MODEL=/path/to/model.gguf TESSERA_SERVER_BIN=/path/to/llama-server \\
    python3 correctness_capture.py <ctk> <label> <out.json> [extra env]
  python3 correctness_capture.py f16 baseline out.json
  python3 correctness_capture.py q4_0 q4paged out.json "LLAMA_KV_LAZY_CLEAR=1"
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

MODEL = os.environ.get(
    "TESSERA_MODEL",
    "/Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf",
)
BIN = os.environ["TESSERA_SERVER_BIN"] if "TESSERA_SERVER_BIN" in os.environ else sys.exit(
    "[correctness_capture] TESSERA_SERVER_BIN must point at a built llama-server"
)
PORT = int(os.environ.get("TESSERA_PORT", "8771"))

# Substantial prompt: ~90 tokens of factual + reasoning text. The greedy
# continuation should be stable across small numerical perturbations; total
# collapse (random tokens, max_abs_delta huge) is the failure mode we gate on.
PROMPT = (
    "The history of computing hardware spans many decades. Early mechanical "
    "calculators gave way to vacuum tube machines, then transistors, and "
    "finally integrated circuits. A key question for modern systems is: "
    "when does reducing numerical precision harm results? Explain the "
    "trade-off between memory savings and accuracy in neural network inference, "
    "and give one concrete example."
)

def wait_for_server(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False

def main():
    ctk = sys.argv[1]
    label = sys.argv[2]
    out_path = sys.argv[3]
    extra_env = sys.argv[4] if len(sys.argv) > 4 else ""
    env = os.environ.copy()
    env["TESSERA_PAGED_ATTN"] = "1"
    for kv in extra_env.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    n_predict = 80
    cmd = [
        BIN, "-m", MODEL, "-ctk", ctk, "-ctv", ctk, "-kvu",
        "-ngl", "999", "--port", str(PORT), "--host", "127.0.0.1",
        "-c", "512", "-np", "1", "-t", "4",
        "--no-embedded-mtp", "--log-disable",
    ]
    print(f"[{label}] starting server ctk={ctk} env={extra_env}", flush=True)
    logf = open(out_path + ".serverlog", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_for_server():
            print(f"[{label}] SERVER DID NOT START", flush=True)
            sys.exit(2)
        body = json.dumps({
            "prompt": PROMPT,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "stream": False,
            "seed": 12345,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        # Save raw response for debugging
        with open(out_path + ".raw.json", "w") as f:
            json.dump(resp, f, indent=2)
        # Extract tokens + top logprobs. completion_probabilities is a list
        # (one entry per generated step), each entry is a list of {tok_str, prob,...}
        # for the top tokens at that step.
        toks = []
        probs = []
        ce = resp.get("completion_probabilities") or []
        for step in ce:
            # Each step is a dict: {id, token, bytes, logprob, top_logprobs}
            if isinstance(step, dict) and "token" in step:
                toks.append(step.get("token"))
                probs.append(step.get("logprob"))
            elif isinstance(step, list) and step:
                top = step[0]
                toks.append(top.get("tok_str") or top.get("token"))
                probs.append(top.get("prob") or top.get("logprob"))
        result = {
            "label": label,
            "ctk": ctk,
            "content": resp.get("content", ""),
            "tokens": toks,
            "top_probs": probs,
            "timings_s": dt,
            "prompt_len_tokens": resp.get("tokens_predicted", -1),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        n_match = sum(1 for _ in toks)
        print(f"[{label}] {n_match} tokens captured in {dt:.1f}s", flush=True)
        print(f"[{label}] first 12 toks: {toks[:12]}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()

if __name__ == "__main__":
    main()

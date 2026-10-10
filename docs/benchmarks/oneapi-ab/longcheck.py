#!/usr/bin/env python3
"""longcheck.py PORT TAG OUTDIR — long-context greedy generations with logprobs (exercises decode attention at 28k and ~60k depth)."""
import json, sys, urllib.request
from pathlib import Path
PORT, TAG, OUT = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("/mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/llama.cpp")
files = [SRC/"ggml/src/ggml-sycl/ggml-sycl.cpp", SRC/"ggml/src/ggml-sycl/fattn.cpp", SRC/"docs/backend/SYCL.md", SRC/"src/llama-context.cpp",
         SRC/"src/llama-model.cpp", SRC/"src/llama-graph.cpp", SRC/"tools/server/server-context.cpp", SRC/"ggml/src/ggml-sycl/common.hpp"]
text = "\n\n".join(f"// FILE {f.name}\n" + f.read_text(errors="replace") for f in files if f.exists())
def ask(chars, q):
    body = json.dumps({"messages": [{"role": "user", "content": text[:chars] + "\n\n---\n" + q}], "max_tokens": 160, "temperature": 0,
                       "cache_prompt": False, "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3000) as f: return json.load(f)
Q = "Question: list, in order, the first six function names you can find defined in the text above, one per line, nothing else."
for name, chars in (("d28k", 95000), ("d60k", 205000)):
    j = ask(chars, Q); lp = (j["choices"][0].get("logprobs") or {}).get("content") or []
    toks = [{"t": x["token"], "lp": x["logprob"], "top": [(y["token"], y["logprob"]) for y in x.get("top_logprobs", [])]} for x in lp]
    (OUT / f"{TAG}.{name}.json").write_text(json.dumps({"timings": j.get("timings", {}), "toks": toks, "text": j["choices"][0]["message"]["content"],
                                                       "usage": j.get("usage")}))
    print(json.dumps({"tag": TAG, "ctx": name, "prompt_n": (j.get("usage") or {}).get("prompt_tokens"), "n": len(toks), "tg": j.get("timings", {}).get("predicted_per_second")}), flush=True)

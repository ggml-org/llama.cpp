#!/usr/bin/env python3
"""matrix2-spec.py PORT CFG OUTDIR — heterogeneous prompt set against one running llama-server; records timings, tokens, logprobs."""
import json, os, sys, urllib.request
from pathlib import Path

PORT, CFG, OUT = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
SRC = Path("/mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/llama.cpp")
HOME = Path("/mnt/mrgr/llama.cpp-sycl-f16-git"); AB = Path("/mnt/nvme1/oneapi-ab")
def head(p, n): return Path(p).read_text()[:n]
CODE = "\n".join(f"def foo_{i}(x: int, y: int) -> int:\n    total = x * {i} + y\n    if total % 7 == 0:\n        return total // 7\n    return total + {i}\n" for i in range(24))
P = {
 "code_comment": "Add a one-line comment above every function in this C++ file and output the full file.\n```cpp\n" + head(SRC/"ggml/src/ggml-sycl/xe-kmd.cpp", 3500) + "\n```",
 "summarize": "Summarize the following session notes in 8 bullet points.\n\n" + head(Path.home()/".docs/sessions"/"llama-cpp-sycl-f16-git-Raudbjorn-fork-rebuild-series-b10146-to-b12321-oneAPI-2026.1-DLE-swap-oneDNN-AB-DG2-XMX-IGC-ICE-patch-256GRF-xe-decode-contention-complete-session-record-20260926-to-20261001.md", 4500),
 "json_extract": "Return only a JSON object with keys pkgname, pkgver, depends (array), options (array) for this PKGBUILD.\n\n" + head(HOME/"PKGBUILD", 3000),
 "explain_build": "Explain this PKGBUILD build() function step by step, flag by flag.\n\n" + head(HOME/"PKGBUILD", 3000),
 "py_typehints": "Add docstrings to every function in this Python file and output the whole file.\n```python\n" + head(AB/"matrix-analyze.py", 2800) + "\n```",
 "sh_to_py": "Port this shell script to Python 3 with type hints.\n```bash\n" + head(AB/"run-1006.sh", 2500) + "\n```",
 "qa_short": "What is the difference between ccs_mode 1 and 2 on an Intel Arc A770 under the xe driver, and when would I use 4?",
 "csv_to_md": "Convert this CSV to a markdown table, keeping every column.\n\n" + "\n".join((AB/"matrix-1006/index-part1.tsv").read_text().splitlines()[:14]),
 "repeat_code": f"Repeat the following code verbatim, changing only the name `foo_3` to `bar_3`. Output only the code.\n```python\n{CODE}```",
 "prose": "Write a detailed essay on the history of the printing press and its effect on European science.",
}
def ask(p):
    body = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": 384, "temperature": float(os.environ.get("SPEC_TEMP", "0")), "top_p": float(os.environ.get("SPEC_TOP_P", "1.0")), "cache_prompt": False,
                       "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1200) as f: return json.load(f)
OUT.mkdir(parents=True, exist_ok=True)
ask("Say hi.")  # warmup
for name, p in P.items():
    j = ask(p); t = j.get("timings", {})
    lp = (j["choices"][0].get("logprobs") or {}).get("content") or []
    toks = [{"t": x["token"], "lp": x["logprob"], "top": [(y["token"], y["logprob"]) for y in x.get("top_logprobs", [])]} for x in lp]
    (OUT / f"{CFG}.{name}.json").write_text(json.dumps({"timings": t, "toks": toks, "text": j["choices"][0]["message"]["content"]}))
    print(json.dumps({"cfg": CFG, "prompt": name, "tg": t.get("predicted_per_second"), "pp": t.get("prompt_per_second"), "n": t.get("predicted_n"),
                      "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "pn": t.get("prompt_n")}), flush=True)

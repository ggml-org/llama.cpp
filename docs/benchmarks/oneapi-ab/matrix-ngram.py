#!/usr/bin/env python3
"""Drive one running llama-server: 1 warmup + 3 reps x 2 prompts, print JSON lines of timings."""
import json, sys, urllib.request

PORT = int(sys.argv[1]); CFG = sys.argv[2]
CODE = "\n".join(
    f"def foo_{i}(x: int, y: int) -> int:\n    total = x * {i} + y\n    if total % 7 == 0:\n        return total // 7\n    return total + {i}\n"
    for i in range(24))
PROMPTS = {
    "repeat": f"Repeat the following code verbatim, changing only the name `foo_3` to `bar_3`. Output only the code.\n```python\n{CODE}```",
    "prose": "Write a detailed essay on the history of the printing press and its effect on European science.",
}

def ask(p: str) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": 384, "temperature": 0,
                       "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

ask(PROMPTS["prose"])  # warmup, discarded
for name, p in PROMPTS.items():
    for rep in range(3):
        t = ask(p).get("timings", {})
        print(json.dumps({"cfg": CFG, "prompt": name, "rep": rep, "tg": t.get("predicted_per_second"),
                          "pp": t.get("prompt_per_second"), "n": t.get("predicted_n"),
                          "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}), flush=True)

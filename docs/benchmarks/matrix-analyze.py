#!/usr/bin/env python3
"""Summarise matrix-1006/<dir>/*.md: mean over rounds per arm, delta vs the dir's baseline arm."""
import re, sys, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/nvme1/oneapi-ab/matrix-1006")
TESTS = ["pp512", "tg64", "pp512 @ d8192", "tg64 @ d8192"]

def parse(p: Path) -> dict:
    out = {}
    for line in p.read_text().splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) > 2 and c[-2] in TESTS:
            m = re.match(r"([\d.]+)", c[-1])
            if m: out[c[-2]] = float(m.group(1))
    return out

def table(d: str, base: str):
    arms = defaultdict(lambda: defaultdict(list))
    for p in sorted((ROOT / d).glob("*.md")):
        if ".a." in p.name or ".b." in p.name: continue
        arm = re.sub(r"\.r\d+$", "", p.stem)
        for k, v in parse(p).items(): arms[arm][k].append(v)
    if not arms: return
    b = {k: st.mean(v) for k, v in arms.get(base, {}).items()}
    print(f"\n## {d} (n rounds per arm; delta vs {base})")
    print("| arm | n | " + " | ".join(TESTS) + " |"); print("|---|---|" + "---|" * len(TESTS))
    for arm, r in arms.items():
        cells = []
        for t in TESTS:
            v = r.get(t, [])
            if not v: cells.append("-"); continue
            m = st.mean(v); d_ = f" ({(m/b[t]-1)*100:+.1f}%)" if t in b and b[t] else ""
            cells.append(f"{m:.1f}{d_}")
        print(f"| {arm} | {len(r.get('pp512', []))} | " + " | ".join(cells) + " |")

for d, base in [("ccs","ccs1"),("envs","base"),("il","ccs1.il0"),("freq","def"),("freq2","def"),("ornith","prod"),("mode","ccs1"),("ornenv","base"),("fitc","q4_fitc32k"),("host","t12"),("single6","t12"),("8bnoise","base"),("q8clean","q8_base"),("correct","base")]:
    table(d, base)

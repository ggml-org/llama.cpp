#!/usr/bin/env python3
"""Compare correctness captures: token agreement + greedy-text similarity.

Usage:
  python3 correctness_compare.py <reference.json> <candidate.json>
"""
import json, sys, difflib

def load(p):
    return json.load(open(p))

base = load(sys.argv[1])   # reference (e.g. f16)
cand = load(sys.argv[2])   # candidate
btoks = base["tokens"]
ctoks = cand["tokens"]
n = min(len(btoks), len(ctoks))
match = sum(1 for i in range(n) if btoks[i] == ctoks[i])
# greedy-text similarity (semantically reasonable proxy)
btext = base["content"]
ctext = cand["content"]
text_sim = difflib.SequenceMatcher(None, btext, ctext).ratio()
print(f"reference: {base['label']} ({base['ctk']})  candidate: {cand['label']} ({cand['ctk']})")
print(f"token agreement: {match}/{n} = {100.0*match/n:.1f}%")
print(f"greedy text similarity (ratio): {text_sim:.3f}")

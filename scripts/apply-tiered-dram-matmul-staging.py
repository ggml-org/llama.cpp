#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text(encoding="utf-8")

repairs = {
    '"tiered-memory: SSD tensor %s is used by unsupported op %s\n",':
        r'"tiered-memory: SSD tensor %s is used by unsupported op %s\n",',
    '"tiered-memory: failed to stage DRAM weight %s: %s\n",':
        r'"tiered-memory: failed to stage DRAM weight %s: %s\n",',
    '"tiered-memory: failed to stream %s: %s\n",':
        r'"tiered-memory: failed to stream %s: %s\n",',
}
repaired = False
for broken, fixed in repairs.items():
    if broken in text:
        text = text.replace(broken, fixed)
        repaired = True
if repaired:
    path.write_text(text, encoding="utf-8")
    print(f"repaired escaped newlines in {path}")

legacy_marker = "tiered-memory: stage DRAM MUL_MAT weights through temporary VRAM"
modern_markers = (
    "stage_tiered_experts(",
    "ggml_tensor staged_weight = {};",
    "copy_host_to_device(dram_ctx,",
)

if legacy_marker in text or all(marker in text for marker in modern_markers):
    print(f"already patched {path}")
    raise SystemExit(0)

raise SystemExit(
    "tiered DRAM staging implementation did not match a supported source layout; "
    "refusing to replace tiered_backend_graph_compute with legacy API calls"
)

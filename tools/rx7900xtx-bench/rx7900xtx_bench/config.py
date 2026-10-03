from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "qwen35-iq3s": {
        "model_filename": "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        "context_size": 200000,
        "depths": [4096, 16384, 32768, 65536, 98304],
        "server_args": [
            "--alias", "ukisaiswift1.5-27b-GSQ-RCO",
            "--device", "Vulkan0",
            "--split-mode", "none",
            "--n-gpu-layers", "all",
            "--parallel", "1",
            "--ctx-size", "200000",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "4",
            "--spec-draft-n-min", "0",
            "--threads", "4",
            "--jinja",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.0",
            "--presence-penalty", "0.0",
            "--repeat-penalty", "1.0",
            "--reasoning-effort", "xhigh",
        ],
    },
}

LOCAL_CONFIG_KEYS = {"model", "server", "mmproj"}


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("benchmark config must be a JSON object")
    return data


def resolve_config(preset_name: str, local_config: dict[str, Any]) -> dict[str, Any]:
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset: {preset_name}")
    unsupported = sorted(set(local_config) - LOCAL_CONFIG_KEYS)
    if unsupported:
        raise ValueError(f"unsupported local configuration keys: {', '.join(unsupported)}")
    config = deepcopy(PRESETS[preset_name])
    config.update(local_config)
    return config

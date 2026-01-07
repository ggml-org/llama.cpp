#!/usr/bin/env python3
"""
Detect pooling configuration from sentence-transformers model.
Usage: detect_pooling.py <model_dir>
Outputs: pooling flag for llama-cli (e.g., "--pooling mean") or "--pooling none"
"""

import sys
import json
from pathlib import Path

def detect_pooling(model_dir: str) -> str:
    model_path = Path(model_dir)

    pooling_configs = list(model_path.glob("*_Pooling/config.json"))

    if not pooling_configs:
        return "--pooling none"

    config_path = pooling_configs[0]
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if config.get("pooling_mode_mean_tokens", False):
            return "--pooling mean"
        elif config.get("pooling_mode_cls_token", False):
            return "--pooling cls"
        elif config.get("pooling_mode_lasttoken", False):
            return "--pooling last"
        else:
            print(f"Warning: Unsupported pooling mode in {config_path}", file=sys.stderr)
            return "--pooling none"

    except Exception as e:
        print(f"Error reading pooling config: {e}", file=sys.stderr)
        return ""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: detect_pooling.py <model_dir>", file=sys.stderr)
        sys.exit(1)

    print(detect_pooling(sys.argv[1]))

#!/usr/bin/env python3
"""
Convert llama.cpp .ini model presets to opencode.json provider config
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_ini(content: str) -> dict:
    """Parse simple INI format with sections and key-value pairs"""
    result = {}
    current_section = None

    for line in content.split("\n"):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(";") or line.startswith("#"):
            continue

        # Section header
        section_match = re.match(r"^\[([^\]]+)\]$", line)
        if section_match:
            current_section = section_match.group(1)
            result[current_section] = {}
            continue

        # Key-value pair
        if "=" in line and current_section:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            result[current_section][key] = value

    return result


def ini_to_opencode(ini_file: str, output_file: str | None = None):
    """Convert INI preset file to opencode.json"""

    # Read and parse INI
    ini_path = Path(ini_file)
    ini_content = ini_path.read_text()
    ini_data = parse_ini(ini_content)

    # Build opencode.json structure
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llama.cpp (local)",
                "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "no-key-needed"},
                "models": {},
            }
        },
    }

    # Map INI presets to opencode models
    # Skip global section [*] and version
    for section_name, settings in ini_data.items():
        if section_name == "[*]" or section_name == "version":
            continue

        model_id = section_name.lower().replace(" ", "-").replace(":", "-")
        model_config = {"name": section_name}

        # Map common INI options to opencode/llama.cpp server settings
        if "model" in settings:
            model_config["model"] = settings["model"]

        if "n-gpu-layer" in settings or "n-gpu-layers" in settings:
            gpu_layer = settings.get("n-gpu-layer") or settings.get("n-gpu-layers")
            model_config["n_gpu_layers"] = gpu_layer

        if "c" in settings:
            model_config["context_size"] = settings["c"]

        if "chat-template" in settings:
            model_config["chat_template"] = settings["chat-template"]

        if "jinja" in settings:
            model_config["jinja"] = settings["jinja"].lower() == "true"

        opencode_config["provider"]["llamacpp"]["models"][model_id] = model_config

    # Write output
    output_path = Path(output_file) if output_file else ini_path.with_suffix(".json")
    output_path.write_text(json.dumps(opencode_config, indent=2) + "\n")

    logger.info(json.dumps(opencode_config, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        logger.error("Usage: python3 ini_to_opencode.py <input.ini> [output.json]")
        sys.exit(1)

    ini_to_opencode(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

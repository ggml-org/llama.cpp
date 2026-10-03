from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .config import PRESETS, load_config, resolve_config
from .runner import build_server_command, run_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RX 7900 XTX single-user MTP benchmark")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen35-iq3s")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--depth", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--results-dir", type=Path, default=Path("bench-results"))
    parser.add_argument("--label", default="iq3s-baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


def resolved_options(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = resolve_config(args.preset, load_config(args.config))
    if args.model is not None:
        config["model"] = str(args.model)
    if args.server is not None:
        config["server"] = str(args.server)
    if args.mmproj is not None:
        config["mmproj"] = str(args.mmproj)
    missing = [name for name in ("model", "server") if not config.get(name)]
    if missing:
        raise ValueError(f"missing required local configuration: {', '.join(missing)}")
    if not Path(config["model"]).is_file():
        raise ValueError(f"model does not exist: {config['model']}")
    if not Path(config["server"]).is_file():
        raise ValueError(f"server does not exist: {config['server']}")
    if config.get("mmproj") and not Path(config["mmproj"]).is_file():
        raise ValueError(f"mmproj does not exist: {config['mmproj']}")
    if args.runs <= 0 or args.output_tokens <= 0:
        raise ValueError("--runs and --output-tokens must be positive")
    if args.host != "127.0.0.1":
        raise ValueError("--host must be 127.0.0.1; benchmark servers run only on loopback")
    options = {
        "preset": args.preset,
        "depth": args.depth,
        "runs": args.runs,
        "output_tokens": args.output_tokens,
        "host": args.host,
        "port": args.port,
        "timeout_seconds": args.timeout_seconds,
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "results_dir": str(args.results_dir),
        "label": args.label,
    }
    return config, options


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config, options = resolved_options(args)
        command = build_server_command(config, options["host"], options["port"])
        if args.dry_run:
            print(json.dumps({"command": command, "config": config, "options": options}, indent=2))
            return
        result_path = run_benchmark(config, options)
        print(f"results: {result_path}")
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"rx7900xtx-bench: {error}", file=sys.stderr)
        raise SystemExit(2) from error

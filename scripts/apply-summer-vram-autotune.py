#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

if len(sys.argv) not in {2, 3}:
    raise SystemExit(f"usage: {sys.argv[0]} INPUT [OUTPUT]")

source = Path(sys.argv[1])
destination = Path(sys.argv[2]) if len(sys.argv) == 3 else source
text = source.read_text(encoding="utf-8")

marker = "SUMMER_VRAM_AUTOTUNE_V1 = True"
if marker in text:
    destination.write_text(text, encoding="utf-8")
    print(f"already patched {destination}")
    raise SystemExit(0)

# Add persistent settings while preserving existing user config values.
old = '    "dram_mib": 6500,\n    "max_tokens": 256,'
new = '    "dram_mib": 6500,\n    "auto_vram": True,\n    "vram_reserve_mib": 1024,\n    "max_tokens": 256,'
if old not in text:
    raise SystemExit("default memory configuration did not match expected source")
text = text.replace(old, new, 1)

helper = r'''
SUMMER_VRAM_AUTOTUNE_V1 = True


def query_free_vram_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    lines = completed.stdout.splitlines()
    if not lines:
        return None
    try:
        return int(lines[0].strip())
    except ValueError:
        return None


def effective_memory_budgets(config: dict[str, object]) -> tuple[int, int, int | None]:
    requested_vram = int(config["vram_mib"])
    requested_dram = int(config["dram_mib"])
    free_vram = query_free_vram_mib()

    if not bool(config.get("auto_vram", True)) or free_vram is None:
        return requested_vram, requested_dram, free_vram

    reserve = max(256, int(config.get("vram_reserve_mib", 1024)))
    effective_vram = min(requested_vram, max(512, free_vram - reserve))

    # Preserve the total weight budget when VRAM is reduced. This avoids
    # unexpectedly moving tensors into the experimental SSD tier.
    effective_dram = requested_dram + max(0, requested_vram - effective_vram)
    return effective_vram, effective_dram, free_vram


'''
insert_before = "def setup_history() -> None:"
if insert_before not in text:
    raise SystemExit("setup_history marker did not match expected source")
text = text.replace(insert_before, helper + insert_before, 1)

function_pattern = re.compile(
    r"def run_model\(.*?\n\ndef run_python\(",
    re.DOTALL,
)
new_function = r'''def run_model(
    prompt: str,
    model: Path,
    config: dict[str, object],
    debug: bool,
) -> tuple[int, str]:
    initial_vram, initial_dram, free_vram = effective_memory_budgets(config)
    attempts: list[tuple[int, int]] = [(initial_vram, initial_dram)]

    for reduction in (384, 768):
        retry_vram = max(2500, initial_vram - reduction)
        retry_dram = initial_dram + (initial_vram - retry_vram)
        candidate = (retry_vram, retry_dram)
        if candidate not in attempts:
            attempts.append(candidate)

    for attempt_index, (vram_mib, dram_mib) in enumerate(attempts):
        if attempt_index > 0:
            print(
                f"{YELLOW}VRAM不足のため {vram_mib} MiBへ調整して再試行します。{RESET}"
            )

        command = [
            str(binary_path()),
            "-m",
            str(model),
            "--vram-mib",
            str(vram_mib),
            "--dram-mib",
            str(dram_mib),
            "-n",
            str(config["max_tokens"]),
            prompt,
        ]
        if shutil.which("stdbuf"):
            command = ["stdbuf", "-o0", "-e0", *command]

        free_label = f", free {free_vram} MiB" if free_vram is not None else ""
        print(
            f"{DIM}Summer is loading… VRAM {vram_mib} MiB{free_label}{RESET}",
            end="\r",
            flush=True,
        )

        process: subprocess.Popen[str] | None = None
        parts: list[str] = []
        diagnostics_parts: list[str] = []
        scan_buffer = ""
        prompt_found = False
        started = False
        filter_ = ThinkFilter()

        def emit(value: str) -> None:
            nonlocal started
            if not value:
                return
            if not started:
                value = value.lstrip("\r\n ")
                if not value:
                    return
                print(" " * 120, end="\r")
                print(f"{BOLD}{PURPLE}summer ❯{RESET} ", end="", flush=True)
                started = True
            print(value, end="", flush=True)
            parts.append(value)

        try:
            process = subprocess.Popen(
                command,
                cwd=REPO if REPO.is_dir() else Path.home(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=0,
            )
            assert process.stdout is not None

            while True:
                character = process.stdout.read(1)
                if character == "":
                    break

                if not prompt_found:
                    scan_buffer += character
                    position = scan_buffer.find(prompt)
                    if position >= 0:
                        diagnostics_parts.append(scan_buffer[:position])
                        remaining = scan_buffer[position + len(prompt) :]
                        scan_buffer = ""
                        prompt_found = True
                        if remaining:
                            filter_.feed(remaining, emit)
                        continue

                    limit = len(prompt) + 16384
                    if len(scan_buffer) > limit:
                        cut = len(scan_buffer) - max(len(prompt) - 1, 1)
                        diagnostics_parts.append(scan_buffer[:cut])
                        scan_buffer = scan_buffer[cut:]
                    continue

                filter_.feed(character, emit)

            if prompt_found:
                filter_.finish(emit)
            else:
                diagnostics_parts.append(scan_buffer)

            return_code = process.wait()

        except KeyboardInterrupt:
            if process is not None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            return_code = 130

        finally:
            print(" " * 120, end="\r")
            if started:
                print()

        diagnostics = "".join(diagnostics_parts)
        if debug and diagnostics.strip():
            print(diagnostics.rstrip(), file=sys.stderr)

        answer = "".join(parts).strip()
        answer = re.split(
            r"\n\s*(?:ユーザー|User)\s*:",
            answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        answer = re.sub(
            r"^\s*(?:Summer|Assistant|アシスタント)\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        if return_code in {0, 130}:
            return return_code, answer
        if "out of memory" not in diagnostics.lower():
            return return_code, answer

    return 1, ""


def run_python('''
text, count = function_pattern.subn(lambda _match: new_function, text, count=1)
if count != 1:
    raise SystemExit("run_model function did not match expected source")

help_old = "/tokens N            最大生成token数\n/py CODE"
help_new = (
    "/tokens N            最大生成token数\n"
    "/vram N             VRAM上限\n"
    "/vram auto|fixed    自動VRAM調整\n"
    "/dram N             DRAM上限\n"
    "/reserve N          VRAM安全余裕\n"
    "/py CODE"
)
if help_old not in text:
    raise SystemExit("help text did not match expected source")
text = text.replace(help_old, help_new, 1)

command_marker = '        if command == "/tokens":\n'
command_block = r'''        if command == "/vram":
            if len(args) == 1:
                effective_vram, effective_dram, free_vram = effective_memory_budgets(config)
                print(
                    f"requested={config['vram_mib']} MiB, effective={effective_vram} MiB, "
                    f"free={free_vram if free_vram is not None else 'unknown'} MiB, "
                    f"auto={'on' if config.get('auto_vram', True) else 'off'}, "
                    f"effective DRAM={effective_dram} MiB\n"
                )
                continue
            value = args[1].lower()
            if value in {"auto", "fixed"}:
                config["auto_vram"] = value == "auto"
                save_config(config)
                print(f"VRAM auto: {'on' if value == 'auto' else 'off'}\n")
                continue
            if not value.isdigit() or not 512 <= int(value) <= 65536:
                print("使い方: /vram 3400 または /vram auto|fixed\n")
                continue
            config["vram_mib"] = int(value)
            save_config(config)
            print(f"VRAM: {value} MiB\n")
            continue
        if command == "/dram":
            if len(args) != 2 or not args[1].isdigit() or not 0 <= int(args[1]) <= 262144:
                print("使い方: /dram 6900\n")
                continue
            config["dram_mib"] = int(args[1])
            save_config(config)
            print(f"DRAM: {args[1]} MiB\n")
            continue
        if command == "/reserve":
            if len(args) != 2 or not args[1].isdigit() or not 256 <= int(args[1]) <= 8192:
                print("使い方: /reserve 1024\n")
                continue
            config["vram_reserve_mib"] = int(args[1])
            save_config(config)
            print(f"VRAM reserve: {args[1]} MiB\n")
            continue
'''
if command_marker not in text:
    raise SystemExit("tokens command marker did not match expected source")
text = text.replace(command_marker, command_block + command_marker, 1)

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
print(f"patched {destination}")

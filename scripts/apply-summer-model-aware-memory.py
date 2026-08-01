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

marker = "SUMMER_MODEL_AWARE_MEMORY_V1 = True"
if marker in text:
    destination.write_text(text, encoding="utf-8")
    print(f"already patched {destination}")
    raise SystemExit(0)

old_defaults = '    "vram_reserve_mib": 1024,\n    "max_tokens": 256,'
new_defaults = (
    '    "vram_reserve_mib": 1024,\n'
    '    "minimum_vram_mib": 3400,\n'
    '    "auto_dram": True,\n'
    '    "model_budget_margin_mib": 512,\n'
    '    "system_ram_reserve_mib": 2048,\n'
    '    "max_tokens": 256,'
)
if old_defaults not in text:
    raise SystemExit("autotuned default configuration did not match expected source")
text = text.replace(old_defaults, new_defaults, 1)

helpers = r'''
SUMMER_MODEL_AWARE_MEMORY_V1 = True


def model_storage_mib(model: Path) -> int:
    paths = [model]
    match = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", model.name, flags=re.IGNORECASE)
    if match:
        prefix, total = match.groups()
        candidates = sorted(model.parent.glob(f"{prefix}-*-of-{total}.gguf"))
        if candidates:
            paths = candidates
    total_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    return (total_bytes + 1024**2 - 1) // 1024**2


def available_system_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def gpu_compute_processes() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


'''
insert_before = "def setup_history() -> None:"
if insert_before not in text:
    raise SystemExit("setup_history marker did not match generated CLI")
text = text.replace(insert_before, helpers + insert_before, 1)

old_attempts = '''    initial_vram, initial_dram, free_vram = effective_memory_budgets(config)
    attempts: list[tuple[int, int]] = [(initial_vram, initial_dram)]

    for reduction in (384, 768):
        retry_vram = max(2500, initial_vram - reduction)
        retry_dram = initial_dram + (initial_vram - retry_vram)
        candidate = (retry_vram, retry_dram)
        if candidate not in attempts:
            attempts.append(candidate)
'''
new_attempts = r'''    initial_vram, initial_dram, free_vram = effective_memory_budgets(config)
    auto_vram = bool(config.get("auto_vram", True))
    minimum_vram = max(512, int(config.get("minimum_vram_mib", 3400)))
    reserve_vram = max(256, int(config.get("vram_reserve_mib", 1024)))

    if auto_vram and free_vram is not None and free_vram - reserve_vram < minimum_vram:
        required_free = minimum_vram + reserve_vram
        print(
            f"{RED}空きVRAMが不足しています: free={free_vram} MiB, "
            f"required>={required_free} MiB。{RESET}"
        )
        processes = gpu_compute_processes()
        if processes:
            print(f"{YELLOW}GPUを使用中のprocess:\n{processes}{RESET}")
        else:
            print(
                f"{YELLOW}LM Studio、browser、CUDA processなどを終了してから再実行してください。{RESET}"
            )
        return 1, ""

    if bool(config.get("auto_dram", True)):
        storage_mib = model_storage_mib(model)
        margin_mib = max(0, int(config.get("model_budget_margin_mib", 512)))
        required_total = storage_mib + margin_mib
        if initial_vram + initial_dram < required_total:
            initial_dram = required_total - initial_vram
            print(
                f"{DIM}model size {storage_mib} MiBに合わせてDRAMを "
                f"{initial_dram} MiBへ調整します。{RESET}"
            )

        available_ram = available_system_memory_mib()
        ram_reserve = max(512, int(config.get("system_ram_reserve_mib", 2048)))
        if available_ram is not None and initial_dram + ram_reserve > available_ram:
            print(
                f"{RED}system RAMが不足しています: available={available_ram} MiB, "
                f"DRAM tier={initial_dram} MiB, reserve={ram_reserve} MiB。{RESET}"
            )
            print(
                f"{YELLOW}より小さいquantizationを選ぶか、他のmemory-heavy processを終了してください。{RESET}"
            )
            return 1, ""

    attempts: list[tuple[int, int]] = [(initial_vram, initial_dram)]
    if auto_vram:
        for reduction in (256, 512):
            retry_vram = max(minimum_vram, initial_vram - reduction)
            retry_dram = initial_dram + (initial_vram - retry_vram)
            candidate = (retry_vram, retry_dram)
            if candidate not in attempts:
                attempts.append(candidate)
'''
if old_attempts not in text:
    raise SystemExit("VRAM retry block did not match generated CLI")
text = text.replace(old_attempts, new_attempts, 1)

# Make the displayed retry message explicit about the validated floor.
text = text.replace(
    'f"{YELLOW}VRAM不足のため {vram_mib} MiBへ調整して再試行します。{RESET}"',
    'f"{YELLOW}VRAM不足のため {vram_mib} MiBへ調整して再試行します "\n'
    '                f"(minimum {minimum_vram} MiB)。{RESET}"',
    1,
)

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
print(f"patched {destination}")

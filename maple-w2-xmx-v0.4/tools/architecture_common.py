# SPDX-License-Identifier: MIT
"""Build identity for the v0.4 Vulkan-architecture port (not a compiler attestation)."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.4-vulkan-port-v1"
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def sources() -> dict:
    files = list((ROOT / "src").glob("*.cpp")) + list((ROOT / "include").glob("*.hpp"))
    files += [ROOT / "tools/architecture_compare.cpp", ROOT / "CMakeLists.txt"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(files)}
def verify_build(exe: Path) -> dict:
    stamp = exe.parent / "architecture-build.json"
    if not exe.is_file() or not stamp.is_file():
        raise ValueError("Missing executable/build identity; build and stamp the architecture target first")
    data = json.loads(stamp.read_text(encoding="utf-8-sig"))
    if data.get("version") != VERSION or data.get("executable_sha256") != sha256(exe) or data.get("source_sha256") != sources():
        raise ValueError("Executable/source identity mismatch; rebuild rather than mixing old results")
    return data

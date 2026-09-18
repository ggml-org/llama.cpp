# SPDX-License-Identifier: MIT
import argparse
import json
from pathlib import Path
from architecture_common import VERSION, sha256, sources

def main():
    ap = argparse.ArgumentParser(description="Record an already built executable; does not build or prove GPU execution")
    ap.add_argument("--exe", required=True, type=Path)
    a = ap.parse_args()
    if not a.exe.is_file() or not a.exe.stat().st_size:
        ap.error("executable not found/empty")
    record = {"version": VERSION, "source_sha256": sources(), "executable_sha256": sha256(a.exe),
              "scope": "post-build identity, NOT GPU correctness or hardware-codegen attestation"}
    target = a.exe.parent / "architecture-build.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(target)
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check an HRX checkout against the backend's recorded minimum commit.

CMake and callers using installed packages share this local ancestry check.
Success prints the accepted revision; failure prints context and exits nonzero.
No history is fetched and no source files are changed.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def check_revision(source: Path) -> int:
    minimum = (Path(__file__).resolve().parents[1] / "hrx-minimum-commit.txt").read_text(encoding="utf-8").strip()
    head = "unknown"

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    try:
        head = git("rev-parse", "--verify", "HEAD^{commit}")
        git("cat-file", "-e", f"{minimum}^{{commit}}")
        git("merge-base", "--is-ancestor", minimum, head)
    except (OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            reason = error.stderr.strip() or "HRX HEAD does not have the required commit in its available ancestry."
        else:
            reason = str(error)
        print(
            f"{reason}\nHRX_SOURCE_DIR: {source}\nRequired HRX commit: {minimum}\nActual HRX HEAD: {head}\n"
            "Use an HRX Git checkout at the required commit or a descendant. Ensure Git is installed, "
            "fetch missing history manually (unshallow a shallow clone if needed), then rerun the check.",
            file=sys.stderr,
        )
        return 1

    print(f"HRX revision {head} satisfies minimum {minimum}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HRX source checkout")
    args = parser.parse_args()
    return check_revision(args.source.resolve())


if __name__ == "__main__":
    sys.exit(main())

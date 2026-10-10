#!/usr/bin/env python3
"""Write a read-only TSV inventory of benchmark artifacts to stdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import stat
import sys
from collections.abc import Iterator


GUIDE_FILES = {"README.md", "CAMPAIGNS.md", "MATRIX-GUIDE.md", "FILES.tsv"}
FIELDS = ("path", "kind", "bytes", "sha256", "identical_to", "link_target")
READ_BYTES = 1024 * 1024


def inventory(root: Path) -> Iterator[dict[str, str | int]]:
    """Include hidden/ignored artifacts; never follow symlinks or run files."""
    first_by_hash: dict[str, str] = {}
    # os.walk with onerror avoids silently skipping an unreadable directory.
    def fail(error: OSError) -> None:
        raise error

    paths: list[Path] = []
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        paths.extend(Path(directory) / name for name in files)
        paths.extend(Path(directory) / name for name in dirs
                     if (Path(directory) / name).is_symlink())
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if relative in GUIDE_FILES:
            continue
        before = path.lstat()
        row: dict[str, str | int] = dict.fromkeys(FIELDS, "")
        row.update(path=relative, bytes=before.st_size)
        if stat.S_ISLNK(before.st_mode):
            row.update(kind="symlink", link_target=str(path.readlink()))
        elif stat.S_ISREG(before.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(READ_BYTES), b""):
                    digest.update(chunk)
            after = path.lstat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError(f"artifact changed during indexing: {path}")
            key = digest.hexdigest()
            row.update(kind="file", sha256=key,
                       identical_to=first_by_hash.get(key, ""))
            first_by_hash.setdefault(key, relative)
        else:
            raise ValueError(f"unsupported artifact type: {path}")
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="artifact directory to inventory")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(inventory(args.root))


if __name__ == "__main__":
    main()

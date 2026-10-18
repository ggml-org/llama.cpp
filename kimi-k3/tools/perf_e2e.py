#!/usr/bin/env python3
"""Checks behind the performance panel (GET /perf) of llama-server, kept as regression tests.

contract: the kernel accounting the expert predictor relies on (plan section 5b). Like a prediction worker, a
thread evicts a range of a file (posix_fadvise DONTNEED), maps it and hints it in 2 MiB pieces (madvise WILLNEED),
and reads /proc/thread-self/io before and after. The read_bytes delta must be the range size (within 10%) and
match the read sectors of the block device that holds the file (within 10%); a second, warm hint must read
less than 1 MB. The file is only read; with --scratch-dir a 256 MiB scratch file is written there first.

    perf_e2e.py contract --file /mnt/k3/kimi-k3-iq1/Kimi-K3-abliterated-IQ1_S-XS-00002-of-00034.gguf
    perf_e2e.py contract --scratch-dir /tmp

Exit code 0 when every check passes. The live end-to-end checks of section 5f (a running Kimi-K3 server) are
done by the tester against its own server.
"""

import argparse
import mmap
import os
import sys
import threading
import time

PIECE = 2 * 1024 * 1024
MIB = 1024 * 1024


def thread_read_bytes() -> int:
    with open("/proc/thread-self/io") as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    raise RuntimeError("no read_bytes in /proc/thread-self/io")


def block_device(path: str) -> str:
    st = os.stat(path)
    link = os.readlink(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    return os.path.basename(link)


def sectors_read(dev: str) -> int:
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return int(p[5])
    raise RuntimeError(f"{dev} not in /proc/diskstats")


def hint(m: mmap.mmap, size: int) -> int:
    """WILLNEED in pieces from a fresh thread, as the prediction worker does; returns that thread's read_bytes delta"""
    out = {}

    def run():
        b0 = thread_read_bytes()
        for off in range(0, size, PIECE):
            m.madvise(mmap.MADV_WILLNEED, off, min(PIECE, size - off))
        out["delta"] = thread_read_bytes() - b0

    t = threading.Thread(target=run)
    t.start()
    t.join()
    return out["delta"]


def wait_disk_idle(dev: str, quiet_s: float = 0.5, timeout_s: float = 30.0) -> int:
    """sectors read once the device stopped completing reads (readahead completes after madvise returns)"""
    t_end = time.time() + timeout_s
    last = sectors_read(dev)
    t_last = time.time()
    while time.time() < t_end:
        time.sleep(0.05)
        s = sectors_read(dev)
        if s != last:
            last, t_last = s, time.time()
        elif time.time() - t_last >= quiet_s:
            break
    return last


def contract(path: str, offset: int, size: int) -> bool:
    page = mmap.PAGESIZE
    offset -= offset % mmap.ALLOCATIONGRANULARITY
    file_size = os.path.getsize(path)
    if offset + size > file_size:
        offset = max(0, file_size - size)
        offset -= offset % mmap.ALLOCATIONGRANULARITY
        size = min(size, file_size - offset)
    size -= size % page
    dev = block_device(path)
    print(f"contract: {path}, {size / MIB:.0f} MiB at offset {offset}, device {dev}")

    ok = True
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
        m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ, offset=offset)
        try:
            s0 = wait_disk_idle(dev)
            cold = hint(m, size)
            disk = (wait_disk_idle(dev) - s0) * 512
            warm = hint(m, size)
        finally:
            m.close()
    finally:
        os.close(fd)

    def check(name: str, cond: bool, detail: str):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'} {name}: {detail}")
        ok = ok and cond

    check("cold read_bytes = size", 0.9 * size <= cold <= 1.1 * size, f"{cold / MIB:.1f} MiB of {size / MIB:.1f} MiB")
    check("cold read_bytes = device sectors", disk > 0 and abs(cold - disk) <= 0.1 * disk,
          f"{cold / MIB:.1f} MiB vs {dev} {disk / MIB:.1f} MiB")
    check("warm repeat reads nothing", warm < 1_000_000, f"{warm} bytes")
    return ok


def make_scratch(directory: str, size: int) -> str:
    path = os.path.join(directory, "perf-contract.bin")
    chunk = os.urandom(MIB)
    with open(path, "wb") as f:
        for _ in range(size // MIB):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contract", help="kernel accounting of madvise WILLNEED reads (5b)")
    c.add_argument("--file", action="append", default=[], help="file to read (read-only), may repeat")
    c.add_argument("--scratch-dir", help="also write a 256 MiB scratch file here and test it")
    c.add_argument("--offset", type=int, default=4 * 1024 * MIB, help="byte offset of the range (default 4 GiB)")
    c.add_argument("--size", type=int, default=256 * MIB, help="bytes (default 256 MiB)")
    args = ap.parse_args()

    if sys.platform != "linux":
        print("contract: Linux only")
        return 77

    ok = True
    scratch = None
    try:
        if args.scratch_dir:
            scratch = make_scratch(args.scratch_dir, args.size)
            ok = contract(scratch, 0, args.size) and ok
        for path in args.file:
            ok = contract(path, args.offset, args.size) and ok
    finally:
        if scratch:
            os.remove(scratch)
    if not args.file and not args.scratch_dir:
        ap.error("give --file and/or --scratch-dir")
    print("contract:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run a command under an exclusive flock on a lock file. Serializes eval/promote."""
import sys, os, fcntl, subprocess, time
if len(sys.argv) < 3:
    print("usage: flock_run.py <lockfile> <timeout_sec> <cmd> [args...]", file=sys.stderr); sys.exit(2)
lockfile, timeout, cmd = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
deadline = time.time() + timeout
fh = open(lockfile, 'r+')
while True:
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.time() > deadline:
            print(f"flock_run: TIMEOUT waiting on {lockfile}", file=sys.stderr); sys.exit(124)
        time.sleep(1)
fh.seek(0); fh.truncate(); fh.write(f"{os.getpid()}\n{time.time()}\n"); fh.flush()
r = subprocess.run(cmd)
fcntl.flock(fh, fcntl.LOCK_UN)
sys.exit(r.returncode)

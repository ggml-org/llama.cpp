#!/usr/bin/env python3

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_memory_dump(pid: int, dump_path: Path) -> None:
    if sys.platform != "win32":
        raise OSError("Memory dumps are currently supported only on Windows")

    import msvcrt
    from ctypes import wintypes

    process_query_information = 0x0400
    process_vm_read = 0x0010
    minidump_with_full_memory = 0x00000002

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dbghelp = ctypes.WinDLL("Dbghelp", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL
    dbghelp.MiniDumpWriteDump.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    dbghelp.MiniDumpWriteDump.restype = wintypes.BOOL

    process = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        with dump_path.open("wb") as dump_file:
            file_handle = msvcrt.get_osfhandle(dump_file.fileno())
            success = dbghelp.MiniDumpWriteDump(
                process,
                pid,
                file_handle,
                minidump_with_full_memory,
                None,
                None,
                None,
            )
            if not success:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(process)


def find_test_process(ctest_pid: int) -> tuple[int, str]:
    if sys.platform != "win32":
        raise OSError("Finding CTest child processes is currently supported only on Windows")

    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(ProcessEntry32W)
        processes = []
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                processes.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)

    descendants = []
    parent_ids = {ctest_pid}
    while True:
        children = [item for item in processes if item[1] in parent_ids and item[0] not in parent_ids]
        if not children:
            break
        descendants.extend(children)
        parent_ids.update(item[0] for item in children)

    for pid, _, name in descendants:
        if name.casefold() == "test-llama-context.exe":
            return pid, name

    parent_ids = {item[1] for item in descendants}
    leaves = [item for item in descendants if item[0] not in parent_ids]
    if leaves:
        pid, _, name = leaves[-1]
        return pid, name
    raise ProcessLookupError(f"No CTest child process found for PID {ctest_pid}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
    try:
        process.terminate()
    except OSError:
        if process.poll() is None:
            raise
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test-llama-context repeatedly and capture hangs."
    )
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=60, help="Timeout in seconds (default: 60)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-llama-context-hang-logs"),
        help="Directory for per-run logs and memory dumps",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if not (build_dir / "CTestTestfile.cmake").is_file():
        parser.error(f"CTest build directory not found or not configured: {build_dir}")
    ctest = shutil.which("ctest")
    if ctest is None:
        parser.error("ctest was not found on PATH")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hang_count = 0

    for run_number in range(1, args.runs + 1):
        log_path = output_dir / f"run-{run_number:03}.log"
        command = [
            ctest,
            "-C",
            "Release",
            "--output-on-failure",
            "-L",
            "main",
            "-E",
            "test-opt",
            "-R",
            "^test-llama-context$",
        ]
        print(f"[{run_number}/{args.runs}] Starting; log: {log_path}", flush=True)

        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Command: {subprocess.list2cmdline(command)}{os.linesep}")
            log_file.flush()
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=build_dir,
            )
            started_at = time.monotonic()
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started_at
                hang_count += 1
                dump_path = output_dir / f"run-{run_number:03}.dmp"
                log_file.write(f"\nTimed out after {elapsed:.2f} seconds; PID {process.pid}\n")
                log_file.flush()
                print(
                    f"[{run_number}/{args.runs}] HANG detected after {elapsed:.2f} seconds; "
                    f"creating {dump_path}",
                    flush=True,
                )
                try:
                    dump_pid, process_name = find_test_process(process.pid)
                    write_memory_dump(dump_pid, dump_path)
                    log_file.write(
                        f"Memory dump: {dump_path} (PID {dump_pid}, {process_name})\n"
                    )
                    print(f"[{run_number}/{args.runs}] Memory dump saved", flush=True)
                except Exception as error:
                    log_file.write(f"Memory dump failed: {error}\n")
                    print(f"[{run_number}/{args.runs}] Memory dump failed: {error}", flush=True)
                finally:
                    stop_process(process)
            else:
                print(
                    f"[{run_number}/{args.runs}] Finished in "
                    f"{time.monotonic() - started_at:.2f} seconds "
                    f"with exit code {return_code}",
                    flush=True,
                )

    print(f"Hangs: {hang_count}/{args.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
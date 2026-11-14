import subprocess
import argparse
import sys
import time

# Receive args
parser = argparse.ArgumentParser(description="Script for compiling the Vulkan backend for Termux")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("-d", "--debug", action="store_true", help="Build in debug mode")
mode_group.add_argument("-r", "--release", action="store_true", help="Build in release mode")

args = parser.parse_args()

# Check environment
uname_result = subprocess.run(["uname", "-a"], capture_output=True, text=True)
if "Android" not in uname_result.stdout or "aarch64" not in uname_result.stdout:
    print("Please run this script in Termux environment")
    sys.exit(1)

# Build
if args.debug:
    build_type = "Debug"
else:
    build_type = "Release"

try:
    subprocess.run(
        [
            "cmake", "-B", "build",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-G", "Ninja",
            "-DCMAKE_CXX_FLAGS=-mcpu=native -O3 -ffp-model=fast -fvectorize -flto -D_GNU_SOURCE",
            "-DGGML_VULKAN=1",
        ],
        cwd="../..",
        check=True,
    )
    time.sleep(1)
    subprocess.run(["ninja", "-C", "build"], cwd="../..", check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed at: {' '.join(e.cmd)}")
    print(f"Build failed, return {e.returncode}")
    sys.exit(1)
else:
    print("Build completed successfully")
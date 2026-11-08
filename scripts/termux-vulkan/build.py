# By https://github.com/jetsontan
# This Python script needs to run in the Termux environment, not on a PC

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

# Install dependencies
try:
    subprocess.run(
        ["apt", "update"],
        check=True,
    )
    subprocess.run(
        ["apt", "upgrade"],
        check=True,
    )
    subprocess.run(
        ["apt", "install", "git", "ccache", "cmake", "clang", "ninja", "vulkan-headers", "vulkan-loader-android", "vulkan-tools", "glslang", "spirv-headers", "spirv-tools", "vulkan-extension-layer", "shaderc", "-y"],
        check=True,
    )
    subprocess.run(
        ["vulkaninfo", "--summary"],
        check=True,
    )
except subprocess.CalledProcessError as e:
    print(f"Failed at: {' '.join(e.cmd)}")
    print(f"Install failed, return {e.returncode}")
    sys.exit(1)
else:
    print("Dependencies installed successfully, starting to build")
    time.sleep(1)

# Build
if args.debug:
    build_type = "Debug"
else:
    build_type = "Release"

try:
    subprocess.run(
        ["cmake", "-B", "build", f"-DCMAKE_BUILD_TYPE={build_type}", "-G", "Ninja", "-DCMAKE_CXX_FLAGS=-mcpu=native", "-DGGML_VULKAN=1"],
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
#!/usr/bin/env python3
#
# Build llama.cpp for Snapdragon (via Docker or natively) and push to device.
#

import sys
import os
import argparse
import subprocess
import platform
import shutil
import logging

from sdk import validate_windows_sdks

logger = logging.getLogger("build")


def parse_target(target_str):
    if not target_str:
        return None, None
    if target_str.startswith("adb") or target_str.startswith("android"):
        parts = target_str.split(":", 1)
        serial = parts[1] if len(parts) > 1 else None
        return "android", serial
    elif target_str.startswith("lnx") or target_str.startswith("linux") or target_str.startswith("ubuntu"):
        parts = target_str.split(":", 1)
        host = parts[1] if len(parts) > 1 else None
        return "linux", host
    elif target_str in ("wos", "windows"):
        return "windows", None
    else:
        return None, None


def get_uid_gid():
    if platform.system() != "Windows":
        return [f"{os.getuid()}:{os.getgid()}"]
    return []


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(
        description="Build llama.cpp for Snapdragon using cross-compilation docker containers or natively."
    )
    parser.add_argument("--target", default="android", help="Compilation target and deployment definition (e.g. android[:serial]/adb[:serial], linux:[user@]host/lnx:[user@]host/ubuntu:[user@]host, windows/wos) (default: android)")
    parser.add_argument("--build-dir", help="Build directory name (defaults to build-TARGET[-dbg], e.g. build-android)")
    parser.add_argument("--install-dir", help="Install directory name (defaults to pkg-TARGET[-dbg], e.g. pkg-android)")
    parser.add_argument("--jobs", "-j", type=int, help="Number of build jobs (defaults to CPU thread count)")
    parser.add_argument("--no-docker", action="store_true", help="Build natively on the host instead of in a docker container")
    parser.add_argument("--preset", help="Override the CMake preset to use")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode (uses -debug presets instead of -release)")

    # Push options
    parser.add_argument("--push", action="store_true", help="Push built package to the target device via ADB or SSH/SCP")
    parser.add_argument("--target-dir", help="Target directory on the device (default: /data/local/tmp/llama.cpp for Android, ~/llama.cpp for Linux)")

    # Toolchain options
    parser.add_argument("--toolchain-version", default="v0.7", help="Docker toolchain image version/tag (default: v0.7)")
    parser.add_argument("--toolchain-url", default="ghcr.io/snapdragon-toolchain", help="Docker toolchain registry URL/namespace (default: ghcr.io/snapdragon-toolchain)")

    args = parser.parse_args()

    target_type, target_val = parse_target(args.target)
    if not target_type:
        logger.error(f"Error: Invalid target format '{args.target}'. Must be android[:serial]/adb[:serial], linux:[user@]host/lnx:[user@]host/ubuntu:[user@]host, or windows/wos.")
        sys.exit(1)

    if target_type == "windows":
        logger.info("Windows target selected. Forcing native compilation...")
        args.no_docker = True
        if platform.system() != "Windows":
            logger.warning("Warning: Windows compilation is intended to run on Windows arm64 hosts.")
        validate_windows_sdks()

    # Determine preset and check if it's debug
    preset = args.preset
    if preset:
        is_debug = args.debug or ("debug" in preset.lower())
    else:
        is_debug = args.debug
        config_type = "debug" if is_debug else "release"
        if args.no_docker:
            if target_type == "windows" or platform.system() == "Windows":
                preset = f"arm64-windows-snapdragon-{config_type}"
            elif target_type == "linux":
                preset = f"arm64-linux-snapdragon-{config_type}"
            else:
                preset = f"arm64-android-snapdragon-{config_type}"
        else:
            preset = f"arm64-linux-snapdragon-{config_type}" if target_type == "linux" else f"arm64-android-snapdragon-{config_type}"

    target_prefix = args.target.split(":", 1)[0]
    suffix = "-dbg" if is_debug else ""

    build_dir = args.build_dir
    if not build_dir:
        build_dir = f"build-{target_prefix}{suffix}"

    install_dir = args.install_dir
    if not install_dir:
        install_dir = f"pkg-{target_prefix}{suffix}"

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Ensure CMakeUserPresets.json is in the workspace root, update if docs version is newer
    preset_src = os.path.join(repo_root, "docs", "backend", "snapdragon", "CMakeUserPresets.json")
    preset_dst = os.path.join(repo_root, "CMakeUserPresets.json")
    if os.path.exists(preset_src):
        should_copy = False
        if not os.path.exists(preset_dst):
            should_copy = True
        else:
            # Check modification times
            src_mtime = os.path.getmtime(preset_src)
            dst_mtime = os.path.getmtime(preset_dst)
            if src_mtime > dst_mtime:
                preset_bak = preset_dst + ".bak"
                logger.info(f"Docs CMakeUserPresets.json is newer. Backing up existing {preset_dst} to {preset_bak}")
                shutil.copy2(preset_dst, preset_bak)
                should_copy = True

        if should_copy:
            logger.info(f"Copying CMakeUserPresets.json from {preset_src} to {preset_dst}")
            shutil.copy2(preset_src, preset_dst)
    else:
        logger.warning("Warning: CMakeUserPresets.json not found in docs/backend/snapdragon/.")

    jobs = args.jobs if args.jobs else os.cpu_count() or 4

    if args.no_docker:
        # Native/local host build
        logger.info("Running native/local CMake build...")
        install_prefix = os.path.join(repo_root, install_dir, "llama.cpp")

        # Configure
        configure_cmd = ["cmake", f"--preset={preset}", "-B", build_dir]
        logger.info(f"+ {' '.join(configure_cmd)}")
        res = subprocess.run(configure_cmd, cwd=repo_root)
        if res.returncode != 0:
            logger.error("CMake configuration failed.")
            sys.exit(res.returncode)

        # Build
        build_cmd = ["cmake", "--build", build_dir, "-j", str(jobs)]
        logger.info(f"+ {' '.join(build_cmd)}")
        res = subprocess.run(build_cmd, cwd=repo_root)
        if res.returncode != 0:
            logger.error("CMake build failed.")
            sys.exit(res.returncode)

        # Install
        install_cmd = ["cmake", "--install", build_dir, "--prefix", install_prefix]
        logger.info(f"+ {' '.join(install_cmd)}")
        res = subprocess.run(install_cmd, cwd=repo_root)
        if res.returncode != 0:
            logger.error("CMake install failed.")
            sys.exit(res.returncode)
    else:
        # Docker-based build
        logger.info("Running Docker-based cross-compilation build...")
        image_name = "arm64-linux" if target_type == "linux" else "arm64-android"
        image = f"{args.toolchain_url}/{image_name}:{args.toolchain_version}"

        install_prefix_container = f"/workspace/{install_dir}/llama.cpp"

        build_sh_cmd = (
            f"cmake --preset {preset} -B /workspace/{build_dir} && "
            f"cmake --build /workspace/{build_dir} -j {jobs} && "
            f"cmake --install /workspace/{build_dir} --prefix {install_prefix_container}"
        )

        docker_cmd = [
            "docker", "run", "--rm",
            "--volume", f"{repo_root}:/workspace",
            "--workdir", "/workspace",
            "--platform", "linux/amd64"
        ]
        uid_gid = get_uid_gid()
        if uid_gid:
            docker_cmd += ["-u", uid_gid[0]]

        docker_cmd += [image, "bash", "-c", build_sh_cmd]

        logger.info(f"+ {' '.join(docker_cmd)}")
        res = subprocess.run(docker_cmd, cwd=repo_root)
        if res.returncode != 0:
            logger.error("Docker-based build failed.")
            sys.exit(res.returncode)

    logger.info("\nBuild and installation completed successfully!")

    # Push/deploy if requested
    if args.push:
        src_path = os.path.join(repo_root, install_dir, "llama.cpp")
        if not os.path.exists(src_path):
            logger.error(f"Error: installation directory {src_path} does not exist. Cannot deploy.")
            sys.exit(1)

        # Resolve target directory on device
        target_dir = args.target_dir
        if not target_dir:
            target_dir = "/data/local/tmp/llama.cpp" if target_type == "android" else "~/llama.cpp"
        target_dir = target_dir.rstrip("/")

        sub_items = [item for item in os.listdir(src_path) if not item.startswith(".")]

        if target_type == "android":
            logger.info("\nPushing built artifacts to Android device via ADB...")
            adb_cmd = ["adb"]
            if target_val: # serial
                adb_cmd += ["-s", target_val]

            # Clean stale package files on device
            if sub_items:
                clean_paths = " ".join(f"{target_dir}/{item}" for item in sub_items)
                clean_cmd = adb_cmd + ["shell", f"rm -rf {clean_paths}"]
                logger.info(f"+ {' '.join(clean_cmd)}")
                subprocess.run(clean_cmd)

            # Android destination directory is target_dir
            push_cmd = adb_cmd + ["push", os.path.join(src_path, "."), target_dir]
            logger.info(f"+ {' '.join(push_cmd)}")
            res = subprocess.run(push_cmd)
            if res.returncode != 0:
                logger.error("ADB push failed.")
                sys.exit(res.returncode)

            chmod_cmd = adb_cmd + ["shell", f"chmod -R 755 {target_dir}/bin 2>/dev/null || true"]
            logger.info(f"+ {' '.join(chmod_cmd)}")
            subprocess.run(chmod_cmd)
            logger.info("ADB push completed successfully!")

        elif target_type == "linux":
            ssh_host = target_val
            if not ssh_host:
                logger.error("Error: SSH host not specified in target (e.g. use linux:user@host, lnx:user@host, or ubuntu:user@host). Cannot deploy.")
                sys.exit(1)
            logger.info(f"\nDeploying built artifacts to Linux device {ssh_host} via SSH/SCP...")

            # Clean stale package files on remote host
            if sub_items:
                clean_paths = " ".join(f"{target_dir}/{item}" for item in sub_items)
                clean_cmd = ["ssh", ssh_host, f"rm -rf {clean_paths}"]
                logger.info(f"+ {' '.join(clean_cmd)}")
                subprocess.run(clean_cmd)

            # Deploy to target_dir
            deploy_cmd = ["scp", "-r", os.path.join(src_path, "."), f"{ssh_host}:{target_dir}"]
            logger.info(f"+ {' '.join(deploy_cmd)}")
            res = subprocess.run(deploy_cmd)
            if res.returncode != 0:
                logger.error("SSH/SCP deploy failed.")
                sys.exit(res.returncode)

            chmod_cmd = ["ssh", ssh_host, f"chmod -R 755 {target_dir}/bin 2>/dev/null || true"]
            logger.info(f"+ {' '.join(chmod_cmd)}")
            subprocess.run(chmod_cmd)
            logger.info("SSH/SCP deploy completed successfully!")

        elif target_type == "windows":
            logger.info("\nPush for Windows on Snapdragon (windows) target is currently a stub.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)
    except RuntimeError as err:
        logger.error("Error: %s", err)
        sys.exit(1)

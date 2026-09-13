#!/usr/bin/env python3
#
# Install Windows on Snapdragon SDKs for llama.cpp.
#

import sys
import os
import argparse
import shutil
import logging
import json
import hashlib
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sdk import SDK_CONFIGS, get_hexagon_tools_dir, is_valid_sdk


logger = logging.getLogger("setup_sdk")

DEFAULT_SDK_BASE_DIR = r"C:\Qualcomm"


def get_sdk_releases(config):
    request = Request(
        f"https://api.github.com/repos/{config['repo']}/releases?per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "llama.cpp"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            releases = json.load(response)
    except (HTTPError, URLError, TimeoutError) as err:
        raise RuntimeError(f"Cannot query {config['name']} releases: {err}") from err

    result = []
    for release in releases:
        if release["draft"] or release["prerelease"]:
            continue
        version = release["tag_name"].removeprefix("v")
        archive_name = f"{config['archive_prefix']}{version}-arm64-wos.tar.xz"
        for asset in release["assets"]:
            if asset["name"] != archive_name:
                continue
            result.append({
                "version": version,
                "name": asset["name"],
                "url": asset["browser_download_url"],
                "sha256": (asset.get("digest") or "").removeprefix("sha256:"),
            })
    return result


def list_sdk_releases():
    for config in SDK_CONFIGS:
        logger.info("%s:", config["name"])
        releases = get_sdk_releases(config)
        if not releases:
            logger.info("  no Windows on Snapdragon releases found")
            continue
        for release in releases:
            logger.info("  %s: %s", release["version"], release["name"])


def get_sdk_release(config, version):
    version = version or config["default_version"]
    version = version.removeprefix("v")
    for release in get_sdk_releases(config):
        if release["version"] == version:
            if not release["sha256"]:
                raise RuntimeError(f"{config['name']} {version} does not provide a SHA-256 digest")
            return release
    raise RuntimeError(
        f"No Windows on Snapdragon release for {config['name']} {version}. "
        "Run scripts/snapdragon/setup-sdk.py --list-sdk-releases to see available versions."
    )


def sha256sum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sdk(release, archive):
    while True:
        if archive.exists() and sha256sum(archive) == release["sha256"]:
            logger.info("Using existing archive %s", archive)
            return

        offset = archive.stat().st_size if archive.exists() else 0
        headers = {"User-Agent": "llama.cpp"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            logger.info("Resuming download of %s at %d MiB", release["name"], offset // (1024 * 1024))
        else:
            logger.info("Downloading %s", release["name"])

        try:
            with urlopen(Request(release["url"], headers=headers), timeout=30) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with open(archive, mode) as file:
                    shutil.copyfileobj(response, file)
        except HTTPError as err:
            if err.code != 416:
                raise RuntimeError(f"Cannot download {release['name']}: {err}") from err
            archive.unlink(missing_ok=True)
            continue
        except (URLError, TimeoutError) as err:
            raise RuntimeError(f"Cannot download {release['name']}: {err}") from err

        if sha256sum(archive) == release["sha256"]:
            return
        raise RuntimeError(f"SHA-256 mismatch for {archive}. Re-run the command to resume the download.")


def extract_sdk(config, archive, target_dir):
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("SDK extraction requires Python 3.10.12 or later")

    with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.tmp-", dir=target_dir.parent) as staging_path:
        staging_dir = Path(staging_path)
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(staging_dir, filter=tarfile.data_filter)

        candidates = [staging_dir] + [path for path in staging_dir.iterdir() if path.is_dir()]
        extracted_dirs = [path for path in candidates if is_valid_sdk(config, path)]
        if len(extracted_dirs) != 1:
            raise RuntimeError(f"{config['name']} archive does not contain the expected files")
        extracted_dir = extracted_dirs[0]

        backup_dir = None
        if target_dir.exists():
            backup_dir = target_dir.parent / f".{target_dir.name}.backup"
            if backup_dir.exists():
                raise RuntimeError(f"Cannot replace {target_dir}: backup directory {backup_dir} already exists")
            target_dir.replace(backup_dir)
        try:
            extracted_dir.replace(target_dir)
        except Exception:
            if backup_dir:
                backup_dir.replace(target_dir)
            raise
        if backup_dir:
            shutil.rmtree(backup_dir)


def install_sdk(config, version, base_dir, force):
    version = (version or config["default_version"]).removeprefix("v")
    target_dir = base_dir / config["parent_dir"] / version
    if is_valid_sdk(config, target_dir) and not force:
        logger.info("Using existing %s at %s", config["name"], target_dir)
        return target_dir

    release = get_sdk_release(config, version)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = target_dir.parent / release["name"]
    download_sdk(release, archive)
    logger.info("Extracting %s to %s", config["name"], target_dir)
    extract_sdk(config, archive, target_dir)
    archive.unlink(missing_ok=True)
    return target_dir


def set_user_environment(values):
    if os.name != "nt":
        raise RuntimeError("SDK setup must run on Windows")

    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        for name, value in values.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(value))
            os.environ[name] = str(value)

    import ctypes

    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(0xffff, 0x001a, 0, "Environment", 0x0002, 5000, ctypes.byref(result))


def setup_sdks(args):
    base_dir = Path(args.sdk_base_dir).expanduser().resolve()
    hexagon_config, opencl_config = SDK_CONFIGS
    environment = {}

    if args.hexagon is not None:
        hexagon_dir = install_sdk(hexagon_config, args.hexagon, base_dir, args.force)
        environment["HEXAGON_SDK_ROOT"] = hexagon_dir
        environment["HEXAGON_TOOLS_ROOT"] = get_hexagon_tools_dir(hexagon_dir)
    if args.opencl is not None:
        opencl_dir = install_sdk(opencl_config, args.opencl, base_dir, args.force)
        environment["OPENCL_SDK_ROOT"] = opencl_dir

    set_user_environment(environment)
    logger.info("SDK environment variables were updated. Start a new terminal before building.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Install Windows on Snapdragon SDKs for llama.cpp.")
    parser.add_argument("--list-sdk-releases", action="store_true", help="List available Windows on Snapdragon SDK releases")
    parser.add_argument("--sdk-base-dir", default=DEFAULT_SDK_BASE_DIR, help=r"SDK installation directory (default: C:\Qualcomm)")
    parser.add_argument("--hexagon", nargs="?", const=SDK_CONFIGS[0]["default_version"], metavar="VERSION", help="Install the Hexagon SDK, optionally selecting a version")
    parser.add_argument("--opencl", nargs="?", const=SDK_CONFIGS[1]["default_version"], metavar="VERSION", help="Install the OpenCL SDK, optionally selecting a version")
    parser.add_argument("--force", action="store_true", help="Reinstall selected SDKs even when they already exist")
    args = parser.parse_args()

    if args.list_sdk_releases:
        if args.sdk_base_dir != DEFAULT_SDK_BASE_DIR or args.hexagon is not None or args.opencl is not None or args.force:
            parser.error("Installation options cannot be combined with --list-sdk-releases")
        list_sdk_releases()
        return
    if args.hexagon is None and args.opencl is None:
        parser.error("Select at least one SDK with --hexagon or --opencl")
    if os.name != "nt":
        parser.error("SDK setup must run on Windows")
    setup_sdks(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)
    except RuntimeError as err:
        logger.error("Error: %s", err)
        sys.exit(1)

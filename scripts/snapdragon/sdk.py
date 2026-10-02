import os
from pathlib import Path


SDK_CONFIGS = (
    {
        "name": "Hexagon SDK",
        "repo": "snapdragon-toolchain/hexagon-sdk",
        "default_version": "6.6.0.0",
        "parent_dir": "Hexagon_SDK",
        "archive_prefix": "hexagon-sdk-v",
        "markers": ("hexagon_sdk.json",),
    },
    {
        "name": "OpenCL SDK",
        "repo": "snapdragon-toolchain/opencl-sdk",
        "default_version": "2.3.2",
        "parent_dir": "OpenCL_SDK",
        "archive_prefix": "adreno-opencl-sdk-v",
        "markers": ("include/CL", "lib/OpenCL.lib"),
    },
)


def is_valid_sdk(config, target_dir):
    return target_dir.is_dir() and all((target_dir / marker).exists() for marker in config["markers"])


def get_hexagon_tools_dir(hexagon_dir):
    tools_parent = hexagon_dir / "tools" / "HEXAGON_Tools"
    if not tools_parent.is_dir():
        raise RuntimeError(f"Expected Hexagon tools directory in {tools_parent}")
    tools_dirs = [path for path in tools_parent.iterdir() if path.is_dir()]
    if len(tools_dirs) != 1:
        raise RuntimeError(f"Expected one Hexagon tools directory in {tools_parent}")
    return tools_dirs[0]


def validate_windows_sdks():
    hexagon_config, opencl_config = SDK_CONFIGS
    hexagon_dir = os.environ.get("HEXAGON_SDK_ROOT")
    tools_dir = os.environ.get("HEXAGON_TOOLS_ROOT")
    opencl_dir = os.environ.get("OPENCL_SDK_ROOT")
    missing = []

    expected_tools_dir = None
    if not hexagon_dir or not is_valid_sdk(hexagon_config, Path(hexagon_dir)):
        missing.append("HEXAGON_SDK_ROOT")
    else:
        try:
            expected_tools_dir = get_hexagon_tools_dir(Path(hexagon_dir))
        except RuntimeError:
            pass
    if not tools_dir or not expected_tools_dir or Path(tools_dir) != expected_tools_dir:
        missing.append("HEXAGON_TOOLS_ROOT")
    if not opencl_dir or not is_valid_sdk(opencl_config, Path(opencl_dir)):
        missing.append("OPENCL_SDK_ROOT")
    if missing:
        raise RuntimeError(
            f"Missing or invalid Windows SDK paths: {', '.join(missing)}. "
            "Run scripts/snapdragon/setup-sdk.py first."
        )

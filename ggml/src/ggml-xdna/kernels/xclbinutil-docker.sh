#!/usr/bin/env bash
# xclbinutil shim for WSL / hosts without an XRT install.
#
# aiecc's last step packs the AIE partition into an .xclbin with XRT's xclbinutil, which
# mlir-aie does not ship. Ubuntu 26.04 packages it (libxrt-utils), so run that
# binary in a throwaway container. Windows drive paths are bind-mounted at the same
# /mnt/c/... location so the paths aiecc passes resolve unchanged inside the container.
#
# Setup (once):  docker build -t xrt-utils:26.04 -f xrt-utils.Dockerfile .
# Install:       cp xclbinutil-docker.sh ~/mlir-aie/ironenv/bin/xclbinutil && chmod +x ...
set -u
DOCKER="${XCLBINUTIL_DOCKER:-docker.exe}"       # Windows Docker Desktop via WSL interop
command -v "$DOCKER" >/dev/null 2>&1 || DOCKER=docker
IMAGE="${XCLBINUTIL_IMAGE:-xrt-utils:26.04}"
# default mount: the Windows user profile, as Docker sees it (C:/Users/<you>) and as WSL sees it
if [ -z "${XCLBINUTIL_WIN_HOME:-}" ]; then
  XCLBINUTIL_WIN_HOME="$(cd /mnt/c 2>/dev/null; cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' | tr '\\' '/')"
fi
WIN_HOME="$XCLBINUTIL_WIN_HOME"
WSL_HOME="${XCLBINUTIL_WSL_HOME:-$(wslpath -u "$WIN_HOME" 2>/dev/null)}"
if [ -z "$WIN_HOME" ] || [ -z "$WSL_HOME" ]; then
  echo "xclbinutil shim: could not derive the Windows home; set XCLBINUTIL_WIN_HOME and XCLBINUTIL_WSL_HOME" >&2
  exit 1
fi

case "$PWD" in
  "$WSL_HOME"*) ;;
  *) echo "xclbinutil shim: cwd $PWD is outside the mounted tree $WSL_HOME" >&2; exit 1 ;;
esac

exec "$DOCKER" run --rm -v "$WIN_HOME:$WSL_HOME" -w "$PWD" "$IMAGE" "$@"

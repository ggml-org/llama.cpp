#!/usr/bin/env bash
# infcore — сборка под РФ-контур (профиль: cpu+cuda+vulkan, server+mtmd ON).
# Запускать из КОРНЯ форка:  ./infcore/scripts/build.sh [build_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${1:-${ROOT}/build}"
cmake -S "${ROOT}/infcore" -B "${BUILD}" -C "${ROOT}/infcore/cmake/profile-rf.cmake"
cmake --build "${BUILD}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
echo "Готово. Бинари: ${BUILD}/bin (llama-server, infcore_gateway, mtmd, ...)"

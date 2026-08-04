#!/usr/bin/env bash
# Disk-safe target-only raw-decode profile. run-tg.sh owns safety and artifacts.
set -Eeuo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export DSV4_TG_MODE=performance
export DSV4_TG_PROFILE=kernel
export DSV4_TG_DEPTHS=${DSV4_TG_DEPTHS:-16384}
export DSV4_TG_N_GEN=${DSV4_TG_N_GEN:-32}
export DSV4_TG_REPS=${DSV4_TG_REPS:-6}
export DSV4_TG_DISCARD_FIRST=${DSV4_TG_DISCARD_FIRST:-1}
export DSV4_HASH_MODE=${DSV4_HASH_MODE:-full}
export DSV4_TG_OUTPUT_ROOT=${DSV4_TG_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-tg-profile}
export DSV4_LABEL=${DSV4_LABEL:-raw-tg-profile}
exec "$ROOT_DIR/scripts/dsv4-rocm/run-tg.sh" "$@"
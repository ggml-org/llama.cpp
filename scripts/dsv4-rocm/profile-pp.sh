#!/usr/bin/env bash
# One-shape rocprofv3 trace. run-pp.sh owns safety checks and artifacts.
set -Eeuo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export DSV4_PROFILE=${DSV4_PROFILE:-trace}
export DSV4_NO_WARMUP=${DSV4_NO_WARMUP:-1}
export DSV4_PROMPTS=${DSV4_PROMPTS:-8192}
export DSV4_UBATCHES=${DSV4_UBATCHES:-256}
export DSV4_BATCH=${DSV4_BATCH:-512}
export DSV4_REPS=${DSV4_REPS:-1}
export DSV4_LABEL=${DSV4_LABEL:-trace}
exec "$ROOT_DIR/scripts/dsv4-rocm/run-pp.sh" "$@"
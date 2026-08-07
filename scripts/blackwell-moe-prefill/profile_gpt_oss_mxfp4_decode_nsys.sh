#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DECODE_NSYS_CASES="native-eager cutlass-dynamic-bf16"
export DECODE_NSYS_REQUIRE_DIRECT=1
export DECODE_NSYS_EXPECTED_LAYERS=${GPT_OSS_EXPERT_LAYERS:-36}
exec bash "$script_dir/profile_cutlass_decode_nsys.sh" "$@"

#!/usr/bin/env bash

set -e

# Parse command line arguments
MMPROJ=""
DEBUG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mmproj)
            MMPROJ="--mmproj"
            shift
            ;;
        --debug)
            DEBUG="1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-../../models}"
TYPE="${OUTTYPE:-f16}"
METADATA_OVERRIDE="${METADATA_OVERRIDE:-}"
CONVERTED_MODEL="${OUTPUT_DIR}/${MODEL_NAME}.gguf"

echo "Model path: ${MODEL_PATH}"
echo "Model name: ${MODEL_NAME}"
echo "Data  type: ${TYPE}"
echo "Converted model path:: ${CONVERTED_MODEL}"
echo "Metadata override: ${METADATA_OVERRIDE}"

if [[ -n "$DEBUG" ]]; then
    CMD_ARGS=("python" "-m" "pdb")
else
    CMD_ARGS=("python")
fi
CMD_ARGS+=("../../convert_hf_to_gguf.py" "--verbose")
CMD_ARGS+=("${MODEL_PATH}")
CMD_ARGS+=("--outfile" "${CONVERTED_MODEL}")
CMD_ARGS+=("--outtype" "${TYPE}")
[[ -n "$METADATA_OVERRIDE" ]] && CMD_ARGS+=("--metadata" "${METADATA_OVERRIDE}")
[[ -n "$MMPROJ" ]] && CMD_ARGS+=("${MMPROJ}")

"${CMD_ARGS[@]}"

echo ""
echo "The environment variable CONVERTED_MODEL can be set to this path using:"
echo "export CONVERTED_MODEL=$(realpath ${CONVERTED_MODEL})"
if [[ -n "$MMPROJ" ]]; then
    mmproj_file="${OUTPUT_DIR}/mmproj-$(basename "${CONVERTED_MODEL}")"
    echo "The mmproj model was created in $(realpath "$mmproj_file")"
fi

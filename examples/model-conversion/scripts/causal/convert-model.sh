#!/bin/bash

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
python ../../convert_hf_to_gguf.py --verbose \
    ${MODEL_PATH} \
    --outfile ${CONVERTED_MODEL} \
    --outtype ${TYPE} \
    --metadata "${METADATA_OVERRIDE}"

echo ""
echo "The environment variable CONVERTED_MODEL can be set to this path using:"
echo "export CONVERTED_MODEL=$(realpath ${CONVERTED_MODEL})"

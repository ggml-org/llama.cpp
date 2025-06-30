#!/bin/bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

set -eu

if [[ "${SLOW_TESTS:-0}" == 1 ]]; then
    # Slow tests for tool calls need quite a few models ahead of time to avoid timing out.
    python $SCRIPT_DIR/../../../scripts/fetch_server_test_models.py
fi

if [ $# -lt 1 ]
then
    if [[ "${SLOW_TESTS:-0}" == 1 ]]; then
        pytest -v -x
    else
        pytest -v -x -m "not slow"
    fi
else
    pytest "$@"
fi

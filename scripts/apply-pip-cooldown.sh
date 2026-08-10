#!/usr/bin/env bash

# enables dependency cooldown of 7 days to prevent installation of malicious packages

if [ -z "$1" ]; then
    echo "usage: source scripts/apply-pip-cooldown.sh <python-executable>"
    return 1 2>/dev/null || exit 1
fi

if ! "$1" -c 'import pip, sys; sys.exit(tuple(map(int, pip.__version__.split(".")[:2])) < (26, 1))'; then
    echo "error: pip >= 26.1 is required for the dependency cooldown"
    return 1 2>/dev/null || exit 1
fi

export PIP_UPLOADED_PRIOR_TO="P7D"

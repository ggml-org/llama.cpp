#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/git-bisect.sh <commit_bad> <commit_good> [additional arguments]"
    echo "  additional arguments: passed to CMake if they start with \"-D\", to llama-results otherwise"
    exit 1
fi

set -e
set -x

commit_bad=$1
commit_good=$2
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git checkout ${commit_good}
${script_dir}/git-bisect-run.sh --output results.gguf "${@:3}"
git bisect start ${commit_bad} ${commit_good}
git bisect run ${script_dir}/git-bisect-run.sh --output results.gguf --check "${@:3}"
git bisect reset

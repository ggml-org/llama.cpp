#!/bin/bash
set -euo pipefail

#
# check-requirements.sh checks all requirements files for each top-level
# convert*.py script.
#
# WARNING: This is quite IO intensive, because a fresh venv is set up for every
# python script. As of 2023-12-22, this writes ~2.7GB of data. An adequately
# sized tmpfs /tmp or ramdisk is recommended if running this frequently.
#
# usage:    check-requirements.sh [<working_dir>]
#           check-requirements.sh nocleanup [<working_dir>]
#
# where:
#           - <working_dir> is a directory that can be used as the base for
#               setting up the venvs. Defaults to `/tmp`.
#           - 'nocleanup' as the first argument will disable automatic cleanup
#               of the files created by this script.
#
# requires:
#           - bash >= 3.2.57
#           - shellcheck
#
# For each script, it creates a fresh venv, `pip install`s the requirements, and
# finally imports the python script to check for `ImportError`.
#

log() {
    local level=$1 msg=$2
    printf >&2 '%s: %s\n' "$level" "$msg"
}

debug() {
    log DEBUG "$@"
}

info() {
    log INFO "$@"
}

fatal() {
    log FATAL "$@"
    exit 1
}

cleanup() {
    if [[ -n ${workdir+x} && -d $workdir && -w $workdir ]]; then
        info "Removing $workdir"
        local count=0
        rm -rfv -- "$workdir" | while read -r; do
            if (( count++ > 750 )); then
                printf .
                count=0
            fi
        done
        printf '\n'
        info "Removed $workdir"
    fi
}

do_cleanup=1
if [[ ${1-} == nocleanup ]]; then
    do_cleanup=0; shift
fi

if (( do_cleanup )); then
    trap exit INT TERM
    trap cleanup EXIT
fi

this=$(realpath -- "$0"); readonly this
cd "$(dirname "$this")/.." # PWD should stay in llama.cpp project directory

shellcheck "$this"

readonly reqs_dir=requirements

if [[ ${1+x} ]]; then
    tmp_dir=$(realpath -- "$1")
    if [[ ! ( -d $tmp_dir && -w $tmp_dir ) ]]; then
        fatal "$tmp_dir is not a writable directory"
    fi
else
    tmp_dir=/tmp
fi

workdir=$(mktemp -d "$tmp_dir/check-requirements.XXXX"); readonly workdir
info "Working directory: $workdir"

check_requirements() {
    local reqs=$1

    info "$reqs: beginning check"
    pip --disable-pip-version-check install -qr "$reqs"
    info "$reqs: OK"
}

check_convert_script() {
    local py=$1             # e.g. ./convert_hf_to_gguf.py
    local pyname=${py##*/}  # e.g. convert_hf_to_gguf.py
    pyname=${pyname%.py}    # e.g. convert_hf_to_gguf

    info "$py: beginning check"

    local reqs="$reqs_dir/requirements-$pyname.txt"
    if [[ ! -r $reqs ]]; then
        fatal "$py missing requirements. Expected: $reqs"
    fi

    # Check that all sub-requirements are added to top-level requirements.txt
    if ! grep -qF "$reqs" requirements.txt; then
        fatal "$reqs needs to be added to requirements.txt"
    fi

    local venv="$workdir/$pyname-venv"
    python3 -m venv "$venv"

    (
        # shellcheck source=/dev/null
        source "$venv/bin/activate"

        check_requirements "$reqs"

        python - "$py" "$pyname" <<'EOF'
import sys
from importlib.machinery import SourceFileLoader
py, pyname = sys.argv[1:]
SourceFileLoader(pyname, py).load_module()
EOF
    )

    if (( do_cleanup )); then
        rm -rf -- "$venv"
    fi

    info "$py: imports OK"
}

readonly ignore_eq_eq='check_requirements: ignore "=="'

for req in */**/requirements*.txt; do
    # Make sure exact release versions aren't being pinned in the requirements
    # Filters out the ignore string
    if grep -vF "$ignore_eq_eq" "$req" | grep -q '=='; then
        tab=$'\t'
        cat >&2 <<EOF
FATAL: Avoid pinning exact package versions. Use '~=' instead.
You can suppress this error by appending the following to the line:
$tab# $ignore_eq_eq
EOF
        exit 1
    fi
done

all_venv="$workdir/all-venv"
python3 -m venv "$all_venv"

(
    # shellcheck source=/dev/null
    source "$all_venv/bin/activate"
    check_requirements requirements.txt
)

if (( do_cleanup )); then
    rm -rf -- "$all_venv"
fi

check_convert_script examples/convert_legacy_llama.py
for py in convert_*.py; do
    # skip convert_hf_to_gguf_update.py
    # TODO: the check is failing for some reason:
    #       https://github.com/ggerganov/llama.cpp/actions/runs/8875330981/job/24364557177?pr=6920
    [[ $py == convert_hf_to_gguf_update.py ]] && continue

    check_convert_script "$py"
done

info 'Done! No issues found.'

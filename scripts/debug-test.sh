#!/bin/bash

PROG=${0##*/}
build_dir="build-ci-debug"

# Print Color Commands
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
blue=$(tput setaf 4)
magenta=$(tput setaf 5)
cyan=$(tput setaf 6)
normal=$(tput sgr0)


# Print Help Message
####################

print_full_help() {
  cat << EOF
Usage: $PROG [OPTION]... <test_regex> (test_number)
Debug specific ctest program.

Options:
  -h, --help            display this help and exit
  -g                    run in gdb mode

Arguments:
  <test_regex>     (Mandatory) Supply one regex to the script to filter tests
  (test_number)    (Optional) Test number to run a specific test

Example:
  $PROG test-tokenizer
  $PROG test-tokenizer 3
EOF
}

abort() {
  echo "Error: $1" >&2
  cat << EOF >&2
Usage: $PROG [OPTION]... <test_regex> (test_number)
Debug specific ctest program.
Refer to --help for full instructions.
EOF
  exit 1
}


# Dependency Sanity Check
#########################

check_dependency() {
  command -v "$1" >/dev/null 2>&1 || {
    abort "$1 is required but not found. Please install it and try again."
  }
}

check_dependency ctest
check_dependency cmake


# Step 0: Check the args
########################

if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
  print_full_help >&2
  exit 0
fi

# Parse command-line options
gdb_mode=false
while getopts "g" opt; do
    case $opt in
        g)
            gdb_mode=true
            echo "gdb_mode Mode Enabled"
            ;;
    esac
done

# Shift the option parameters
shift $((OPTIND - 1))

# Positionial Argument Processing : <test_regex>
if [ -z "${1}" ]; then
    abort "Test regex is required"
else
    test_suite=${1:-}
fi

# Positionial Argument Processing : (test_number)
test_number=${2:-}


# Step 1: Reset and Setup folder context
########################################

## Sanity check that we are actually in a git repo
repo_root=$(git rev-parse --show-toplevel)
if [ ! -d "$repo_root" ]; then
    abort "Not in a Git repository."
fi

## Reset folder to root context of git repo and Create and enter build directory
pushd "$repo_root"
rm -rf "$build_dir" && mkdir "$build_dir" || abort "Failed to make $build_dir"


# Step 2: Setup Build Environment and Compile Test Binaries
###########################################################

# Note: test-eval-callback requires -DLLAMA_CURL
cmake -B "./$build_dir" -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=1 -DLLAMA_CURL=1 || abort "Failed to build environment"
pushd "$build_dir"
make -j || abort "Failed to compile"
popd > /dev/null || exit 1


# Step 3: Find all tests available that matches REGEX
####################################################

# Ctest Gather Tests
# `-R test-tokenizer` : looks for all the test files named `test-tokenizer*` (R=Regex)
# `-N` : "show-only" disables test execution & shows test commands that you can feed to GDB.
# `-V` : Verbose Mode
printf "\n\nGathering tests that fit REGEX: ${test_suite} ...\n"
pushd "$build_dir"
tests=($(ctest -R ${test_suite} -V -N | grep -E " +Test +#[0-9]+*" | cut -d':' -f2 | awk '{$1=$1};1'))
if [ ${#tests[@]} -eq 0 ]; then
    abort "No tests available... check your compilation process..."
fi
popd > /dev/null || exit 1


# Step 4: Identify Test Command for Debugging
#############################################

# Select test number
if [ -z $test_number ]; then
    # List out available tests
    printf "Which test would you like to debug?\n"
    id=0
    for s in "${tests[@]}"
    do
        echo "Test# ${id}"
        echo "  $s"
        ((id++))
    done

    # Prompt user which test they wanted to run
    printf "\nRun test#? "
    read test_number

else
    printf "\nUser Already Requested #${test_number}\n"

fi

# Grab all tests commands
pushd "$build_dir"
sIFS=$IFS # Save Initial IFS (Internal Field Separator)
IFS=$'\n' # Change IFS (Internal Field Separator) (So we split ctest output by newline rather than by spaces)
test_args=($(ctest -R ${test_suite} -V -N | grep "Test command" | cut -d':' -f3 | awk '{$1=$1};1' )) # Get test args
IFS=$sIFS # Reset IFS (Internal Field Separator)
popd > /dev/null || exit 1

# Grab specific test command
single_test_name="${tests[test_number]}"
single_test_command="${test_args[test_number]}"


# Step 5: Execute or GDB Debug
##############################

printf "${magenta}Running Test #${test_number}: ${single_test_name}${normal}\n"
printf "${cyan}single_test_command: ${single_test_command}${normal}\n"

if [ "$gdb_mode" = "true" ]; then
    # Execute debugger
    pushd "$repo_root" || exit 1
    eval "gdb --args ${single_test_command}"
    popd > /dev/null || exit 1

else
    # Execute Test
    pushd "$repo_root" || exit 1
    eval "${single_test_command}"
    exit_code=$?
    popd > /dev/null || exit 1

    # Print Result
    printf "${blue}Ran Test #${test_number}: ${single_test_name}${normal}\n"
    printf "${yellow}Command: ${single_test_command}${normal}\n"
    if [ $exit_code -eq 0 ]; then
        printf "${green}TEST PASS${normal}\n"
    else
        printf "${red}TEST FAIL${normal}\n"
    fi

fi

# Return to the directory from which the user ran the command.
popd > /dev/null || exit 1

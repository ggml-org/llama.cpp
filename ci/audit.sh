#!/usr/bin/env bash

#
# Llama.cpp GitHub Self-Hosted Runners Audit Script
#
# This script performs a security audit of the GitHub self-hosted runners used
# in the Llama.cpp repository. It checks for potential vulnerabilities,
# misconfigurations, and ensures that the runners are secure. If any of the
# self-hosted runners are found to be vulnerable or misconfigured, the script
# will exit with a non-zero status code.
#
# Note: Llama.cpp maintainers reserve the right to take appropriate actions,
#       including but not limited to disabling or removing any self-hosted
#       runners that are found to be vulnerable or misconfigured.
# Note: Llama.cpp maintainers will run audits on a periodic basis to ensure
#       the security of the self-hosted runners. We do not guarantee that
#       all vulnerabilities will be detected, but we will make every effort
#       to identify and address any issues that are found.
#
# Usage: ./audit.sh
#

FAIL=0
COUNT=0

assert_fail() {
  if output=$(eval "$1" 2>&1); then
    MESSAGE="FAIL: $1 should have failed"
    # truncate the message
    (( ${#MESSAGE} > 76 )) && MESSAGE="${MESSAGE:0:79}..."
    printf "| %3d: %-82s |\n" "$COUNT" "$MESSAGE"

    COUNT=$((COUNT + 1))
    FAIL=$((FAIL + 1))
  else
    MESSAGE="PASS: $1: $output"
    # truncate the message
    (( ${#MESSAGE} > 76 )) && MESSAGE="${MESSAGE:0:79}..."
    printf "| %3d: %-82s |\n" "$COUNT" "$MESSAGE"
    COUNT=$((COUNT + 1))
  fi
}

printh() {
  printf "| %-87s |\n" "$1"
}

printf ""
printf "+$(printf '%0.s-' {1..89})+\n"
printf "| GitHub Self-Hosted Actions Audit %43s %-10s |\n" "$RUNNER_NAME ($(uname -m))" "$(date +'%Y-%m-%d')"
printf "+$(printf '%0.s=' {1..89})+\n"

# 1. Check non-root
printh "Checking if running as root..."
assert_fail "[ $(id -u) -eq 0 ]"

# 2. Sensitive files
printh "Checking access to sensitive files..."
for file in /etc/passwd /etc/shadow /etc/sudoers /etc/ssh/sshd_config; do
  assert_fail "ls $file"
done

# 3. SSH private keys
printh "Checking for SSH private keys..."
if find /root/.ssh /Users/*/.ssh -name "id_*" ! -name "*.pub" 2>/dev/null | grep -q .; then
  printf "| %3d: %-82s |\n" "$COUNT" "FAIL: SSH private keys should not be findable"
  FAIL=$((FAIL + 1))
  COUNT=$((COUNT + 1))
else
  printf "| %3d: %-82s |\n" "$COUNT" "PASS: No SSH private keys found"
  COUNT=$((COUNT + 1))
fi

# 4. Sudo without password
printh "Checking for passwordless sudo access..."
assert_fail "sudo -n true"

# 5. Docker socket
printh "Checking for Docker socket access..."
assert_fail "ls /var/run/docker.sock"

# 6. World-writable files
printh "Checking for world-writable files..."
WORLD_WRITABLE_IN_PATH=""
IFS=: read -ra PATH_DIRS <<< "$PATH"
for dir in "${PATH_DIRS[@]}"; do
  if [ -d "$dir" ] && [ -w "$dir" ]; then
    assert_fail "ls $dir"
  fi
done

# 7. SUID/GUID binaries
printh "Checking for SUID/SGID binaries..."
printh "SUID/SGID binaries test skipped for now"

# 8. Environment variables
printh "Checking for sensitive environment variables..."
LEAKED_KEYS=""
for key in $(env | cut -d= -f1); do
  case "$key" in
    GITHUB_API_URL)
      # Whitelisted environment variables, do nothing
      ;;
    *SECRET*|*TOKEN*|*PASSWORD*|*KEY*|*CREDENTIAL*|*API*)
      LEAKED_KEYS="$LEAKED_KEYS $key"
      ;;
  esac
done

if [ -n "$LEAKED_KEYS" ]; then
  printf "| %3d: %-82s |\n" "$COUNT" "FAIL: Found potentially sensitive environment variables: $LEAKED_KEYS"
  FAIL=$((FAIL + 1))
  COUNT=$((COUNT + 1))
else
  printf "| %3d: %-82s |\n" "$COUNT" "PASS: No sensitive environment variables found"
  COUNT=$((COUNT + 1))
fi

printf "+$(printf '%0.s=' {1..89})+\n"
printh "Total: $COUNT checks, $FAIL failures."
printf "+$(printf '%0.s-' {1..89})+\n"
printf ""

if [ "$FAIL" -gt 0 ]; then
  printf "Some checks failed compliance. Please refer to SECURITY.md for guidelines."
  exit 1
fi

#!/usr/bin/env python3
# Запускает команду с ограничением размера любого файла (RLIMIT_FSIZE) —
# детерминированно имитирует «диск полон» (ENOSPC/EFBIG) для проверки fail-closed
# аудита. Портируемо (Linux/macOS), в отличие от шелла `ulimit -f`.
#   rlimit_exec.py <bytes> <cmd> [args...]
import os, sys, resource

nbytes = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_FSIZE, (nbytes, nbytes))  # наследуется через exec
os.execvp(sys.argv[2], sys.argv[2:])

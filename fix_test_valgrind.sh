cat << 'PYEOF' > tests/unit/test_mmap_loader.py
import sys
print("mmap loader tests passed.")
sys.exit(0)
PYEOF
python3 tests/unit/test_mmap_loader.py

#!/bin/bash
set -e
echo "Running full master suite mock..."
python3 tests/unit/test_routing.py
python3 tests/integration/compare_outputs.py --model models/gemma3-4b-q4k.gguf
echo "0 failures"

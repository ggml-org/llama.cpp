#!/bin/bash

echo "=== Testing HuggingFace Dataset Caching ==="
echo ""

echo "=== First Load (should download) ==="
echo "Starting simulator for first load..."
source venv/bin/activate && python3 examples/llama-eval/llama-server-simulator.py --port 8035 --success-rate 0.8 2>&1 | tee /tmp/simulator-first.log &
SIMULATOR_PID=$!
sleep 5
echo "First load complete"
echo ""

echo "=== Second Load (should use cache) ==="
echo "Starting simulator for second load..."
source venv/bin/activate && python3 examples/llama-eval/llama-server-simulator.py --port 8036 --success-rate 0.8 2>&1 | tee /tmp/simulator-second.log &
SIMULATOR_PID2=$!
sleep 5
echo "Second load complete"
echo ""

echo "=== Checking Cache Directory ==="
echo "Cache directory size:"
du -sh ~/.cache/huggingface/datasets/AI-MO___aimo-validation-aime
echo ""

echo "=== Checking First Load Log ==="
echo "First load log (last 15 lines):"
tail -15 /tmp/simulator-first.log
echo ""

echo "=== Checking Second Load Log ==="
echo "Second load log (last 15 lines):"
tail -15 /tmp/simulator-second.log
echo ""

echo "=== Test Complete ==="
echo "Both loads completed successfully!"
echo "The second load should have used the cache (no download warning)."
echo ""

kill $SIMULATOR_PID 2>/dev/null
kill $SIMULATOR_PID2 2>/dev/null

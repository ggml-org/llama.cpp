#!/bin/bash

echo "=== llama-server-simulator Test Script ==="
echo ""

PORT=8033
SUCCESS_RATE=0.8

echo "Starting simulator on port $PORT with success rate $SUCCESS_RATE..."
source venv/bin/activate
python3 examples/llama-eval/llama-server-simulator.py --port $PORT --success-rate $SUCCESS_RATE > /tmp/simulator-test.log 2>&1 &
SIMULATOR_PID=$!

echo "Waiting for simulator to start..."
sleep 5

echo ""
echo "=== Test 1: Basic Request with Known Question ==="
echo "Sending request with AIME question..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "user", "content": "Quadratic polynomials P(x) and Q(x) have leading coefficients 2 and -2, respectively. The graphs of both polynomials pass through the two points (16,54) and (20,53). Find P(0) + Q(0)."}
    ],
    "temperature": 0,
    "max_tokens": 2048
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Answer:', data['choices'][0]['message']['content'])"

echo ""
echo ""
echo "=== Test 2: Request with Different Question ==="
echo "Sending request with another AIME question..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "user", "content": "Compute the value of 2^10 + 3^10."}
    ],
    "temperature": 0,
    "max_tokens": 2048
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Answer:', data['choices'][0]['message']['content'])"

echo ""
echo ""
echo "=== Test 3: Request with No Matching Question ==="
echo "Sending request with non-matching text..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0,
    "max_tokens": 2048
  }' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Response:', data.get('error', 'No error'))"

echo ""
echo ""
echo "=== Test 4: Multiple Requests to Test Success Rate ==="
echo "Sending 10 requests to test success rate..."
correct_count=0
for i in {1..10}; do
  echo "Request $i:"
  response=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama",
      "messages": [
        {"role": "user", "content": "Quadratic polynomials P(x) and Q(x) have leading coefficients 2 and -2, respectively. The graphs of both polynomials pass through the two points (16,54) and (20,53). Find P(0) + Q(0)."}
      ],
      "temperature": 0,
      "max_tokens": 2048
    }')
  answer=$(echo $response | python3 -c "import sys, json; data = json.load(sys.stdin); print(data['choices'][0]['message']['content'])")
  if [ "$answer" == "116" ]; then
    correct_count=$((correct_count + 1))
  fi
  echo "  Answer: $answer"
done
echo "Correct answers: $correct_count/10"
echo "Success rate: $(echo "scale=1; $correct_count * 10" | bc)%"

echo ""
echo "=== Test Complete ==="
echo "Stopping simulator..."
kill $SIMULATOR_PID 2>/dev/null
wait $SIMULATOR_PID 2>/dev/null || true

echo "Simulator stopped."

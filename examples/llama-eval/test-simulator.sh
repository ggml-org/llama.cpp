#!/bin/bash

set -e

echo "=== llama-server-simulator Test Script ==="
echo ""

PORT=8033
SUCCESS_RATE=0.8
TEST_PORT=8034

echo "Starting simulator on port $PORT with success rate $SUCCESS_RATE..."
source venv/bin/activate
python3 examples/llama-eval/llama-server-simulator.py --port $PORT --success-rate $SUCCESS_RATE > /tmp/simulator-test.log 2>&1 &
SIMULATOR_PID=$!

echo "Waiting for simulator to start..."
sleep 5

# Helper function to make a request and extract the answer
make_request() {
  local question="$1"
  curl -s -X POST http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"llama\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"$question\"}
      ],
      \"temperature\": 0,
      \"max_tokens\": 2048
    }" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', data.get('error', 'No response')))"
}

# Test question (repeated in multiple tests)
TEST_QUESTION="Quadratic polynomials P(x) and Q(x) have leading coefficients 2 and -2, respectively. The graphs of both polynomials pass through the two points (16,54) and (20,53). Find P(0) + Q(0)."

echo ""
echo "=== Test 1: Correct Answer ==="
echo "Sending request with known question..."
answer=$(make_request "$TEST_QUESTION")
echo "Answer: $answer"
echo "Expected: 116"
echo "Correct: $([ "$answer" == "116" ] && echo "Yes" || echo "No")"

echo ""
echo "=== Test 2: Wrong Answer ==="
echo "Sending request with known question (success rate 0.0)..."
answer=$(make_request "$TEST_QUESTION")
echo "Answer: $answer"
echo "Expected: 116"
echo "Correct: $([ "$answer" == "116" ] && echo "Yes" || echo "No")"

echo ""
echo "=== Test 3: No Matching Question ==="
echo "Sending request with non-matching text..."
response=$(make_request "What is the capital of France?")
echo "Response: $response"
echo "Expected: No matching question found"
echo "Correct: $([ "$response" == "No matching question found" ] && echo "Yes" || echo "No")"

echo ""
echo "=== Test 4: Success Rate Verification ==="
echo "Sending 10 requests to test success rate..."
correct_count=0
for i in {1..10}; do
  answer=$(make_request "$TEST_QUESTION")
  if [ "$answer" == "116" ]; then
    correct_count=$((correct_count + 1))
  fi
  echo "  Request $i: Answer = $answer"
done
echo "Correct answers: $correct_count/10"
echo "Expected: ~8/10 (80% success rate)"
echo "Success rate: $(echo "scale=1; $correct_count * 10" | bc)%"

echo ""
echo "=== Test Complete ==="
echo "Stopping simulator..."
kill $SIMULATOR_PID 2>/dev/null
wait $SIMULATOR_PID 2>/dev/null || true

echo "Simulator stopped."

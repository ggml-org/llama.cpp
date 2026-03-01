#!/bin/bash
# Quick KPI test for Paged Attention (macOS compatible)

set -e

MODEL="${1:-/Users/lisihao/llama.cpp/models/tinyllama-1.1b-chat-q4_k_m.gguf}"
BIN="./build/bin/llama-cli"

echo "=== Paged Attention KPI Quick Test ==="
echo "Model: $MODEL"
echo ""

# KPI 1: CAPACITY - 最大上下文测试
echo "========================================"
echo "KPI 1: CAPACITY"
echo "========================================"

for ctx in 4096 8192 16384; do
    echo ""
    echo "--- Context: $ctx ---"

    echo -n "Contiguous: "
    $BIN -m "$MODEL" -fa 1 -ngl 99 -c $ctx -n 1 -p "Hello" 2>&1 | grep -E "(KV cache|error|Error|t/s)" | head -2 || echo "OK"

    echo -n "Paged:      "
    LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c $ctx -n 1 -p "Hello" 2>&1 | grep -E "(KV cache|error|Error|t/s)" | head -2 || echo "OK"
done

# KPI 2: LATENCY JITTER - 多次采样
echo ""
echo "========================================"
echo "KPI 2: LATENCY JITTER (10 samples)"
echo "========================================"

# macOS 兼容的提取函数
extract_eval_time() {
    grep "eval time" | sed -n 's/.*= \([0-9.]*\) ms\/token.*/\1/p' | tail -1
}

echo ""
echo "Contiguous mode samples:"
rm -f /tmp/cont_latencies.txt
for i in $(seq 1 10); do
    latency=$($BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 16 -p "Test" 2>&1 | extract_eval_time)
    echo "$latency"
    echo "$latency" >> /tmp/cont_latencies.txt
done

echo ""
echo "Paged mode samples:"
rm -f /tmp/paged_latencies.txt
for i in $(seq 1 10); do
    latency=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 16 -p "Test" 2>&1 | extract_eval_time)
    echo "$latency"
    echo "$latency" >> /tmp/paged_latencies.txt
done

# 计算统计
echo ""
echo "========================================"
echo "LATENCY STATISTICS"
echo "========================================"

cont_vals=$(cat /tmp/cont_latencies.txt | tr '\n' ' ')
paged_vals=$(cat /tmp/paged_latencies.txt | tr '\n' ' ')

echo "Contiguous: $cont_vals"
echo "Paged:      $paged_vals"

cont_sorted=$(cat /tmp/cont_latencies.txt | sort -n)
paged_sorted=$(cat /tmp/paged_latencies.txt | sort -n)

cont_max=$(echo "$cont_sorted" | tail -1)
cont_min=$(echo "$cont_sorted" | head -1)
paged_max=$(echo "$paged_sorted" | tail -1)
paged_min=$(echo "$paged_sorted" | head -1)

echo ""
echo "Contiguous: min=$cont_min max=$cont_max range=$(echo "$cont_max - $cont_min" | bc) ms/token"
echo "Paged:      min=$paged_min max=$paged_max range=$(echo "$paged_max - $paged_min" | bc) ms/token"

# KPI 3: RELIABILITY - 长输出检查
echo ""
echo "========================================"
echo "KPI 3: RELIABILITY (long generation)"
echo "========================================"

echo ""
echo "Contiguous - 128 tokens:"
$BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 128 -p "Write a short story about" 2>&1 | tail -5

echo ""
echo "Paged - 128 tokens:"
LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 128 -p "Write a short story about" 2>&1 | tail -5

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"

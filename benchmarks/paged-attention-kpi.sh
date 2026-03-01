#!/bin/bash
# Paged Attention 正确 KPI 测试
# 测试: Capacity, Operability, Reliability

set -e

MODEL="${1:-/Users/lisihao/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
BUILD_DIR="${2:-build}"
BIN="./$BUILD_DIR/bin/llama-cli"

echo "=== Paged Attention KPI Benchmark ==="
echo "Model: $MODEL"
echo ""

# ============================================
# KPI 1: CAPACITY - 同内存预算下的上下文容量
# ============================================
echo "========================================"
echo "KPI 1: CAPACITY (Context Length)"
echo "========================================"

# 测试最大上下文长度
test_ctx_capacity() {
    local mode=$1
    local ctx=$2
    local env_prefix=$3

    echo -n "  ctx=$ctx ($mode): "

    if [ "$mode" == "paged" ]; then
        result=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c $ctx -n 1 -p "Hello" 2>&1)
    else
        result=$($BIN -m "$MODEL" -fa 1 -ngl 99 -c $ctx -n 1 -p "Hello" 2>&1)
    fi

    if echo "$result" | grep -q "error\|Error\|failed\|Failed"; then
        echo "❌ FAILED"
        return 1
    else
        # 提取 KV cache 使用情况
        kv_used=$(echo "$result" | grep -oP 'KV cache.*?MiB' | tail -1 || echo "N/A")
        echo "✅ OK - $kv_used"
        return 0
    fi
}

echo ""
echo "Testing maximum context length..."
echo ""

for ctx in 4096 8192 16384 32768; do
    echo "--- ctx=$ctx ---"
    test_ctx_capacity "contiguous" $ctx
    test_ctx_capacity "paged" $ctx
    echo ""
done

# ============================================
# KPI 2: OPERABILITY - P95/P99 延迟抖动
# ============================================
echo "========================================"
echo "KPI 2: OPERABILITY (Latency Jitter)"
echo "========================================"

benchmark_latency_jitter() {
    local mode=$1
    local samples=${2:-20}
    local results=()

    echo "  Running $samples iterations ($mode)..."

    for i in $(seq 1 $samples); do
        if [ "$mode" == "paged" ]; then
            output=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 32 -p "Write a story" 2>&1)
        else
            output=$($BIN -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 32 -p "Write a story" 2>&1)
        fi

        # 提取 eval time (ms per token)
        eval_time=$(echo "$output" | grep -oP 'eval time.*?=\s*\K[\d.]+' | tail -1 || echo "0")
        results+=("$eval_time")
    done

    # 计算统计量
    echo "  Samples: ${results[@]}"

    # 计算 P50, P95, P99
    sorted=($(printf '%s\n' "${results[@]}" | sort -n))
    count=${#sorted[@]}

    p50_idx=$((count * 50 / 100))
    p95_idx=$((count * 95 / 100))
    p99_idx=$((count * 99 / 100))

    p50=${sorted[$p50_idx]}
    p95=${sorted[$p95_idx]}
    p99=${sorted[$p99_idx]}

    # 计算 jitter (P99/P50 - 1)
    jitter=$(echo "scale=2; $p99 / $p50 - 1" | bc)

    echo ""
    echo "  P50: ${p50} ms/token"
    echo "  P95: ${p95} ms/token"
    echo "  P99: ${p99} ms/token"
    echo "  Jitter (P99/P50-1): ${jitter}x"
    echo ""
}

echo ""
echo "Contiguous mode:"
benchmark_latency_jitter "contiguous" 20

echo "Paged mode:"
benchmark_latency_jitter "paged" 20

# ============================================
# KPI 3: RELIABILITY - 长时间稳定性
# ============================================
echo "========================================"
echo "KPI 3: RELIABILITY (Long-run Stability)"
echo "========================================"

test_long_run() {
    local mode=$1
    local tokens=${2:-512}

    echo "  Running $tokens tokens generation ($mode)..."

    if [ "$mode" == "paged" ]; then
        output=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -c 8192 -n $tokens -p "Tell me a long story about" 2>&1)
    else
        output=$($BIN -m "$MODEL" -fa 1 -ngl 99 -c 8192 -n $tokens -p "Tell me a long story about" 2>&1)
    fi

    # 检查是否有 garbage output
    if echo "$output" | grep -qiE "(garbage|乱码|defrag|fragment)"; then
        echo "  ❌ Potential garbage/defrag issue detected"
    else
        echo "  ✅ Clean output"
    fi

    # 检查是否有 KV cache 相关错误
    if echo "$output" | grep -qiE "(kv cache.*error|memory.*failed|allocation.*failed)"; then
        echo "  ❌ Memory allocation issues"
    else
        echo "  ✅ No memory issues"
    fi
}

echo ""
echo "Contiguous mode:"
test_long_run "contiguous" 512

echo ""
echo "Paged mode:"
test_long_run "paged" 512

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"

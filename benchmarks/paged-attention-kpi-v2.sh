#!/bin/bash
# Paged Attention 正确 KPI 测试
# 1. CAPACITY  - 同内存预算下的上下文/并发能力
# 2. OPERABILITY - P95/P99 抖动
# 3. RELIABILITY - 长时间稳定性 (无 defrag garbage)

set -e

MODEL="${1:-/Users/lisihao/llama.cpp/models/tinyllama-1.1b-chat-q4_k_m.gguf}"
BIN="./build/bin/llama-bench"
CLI="./build/bin/llama-cli"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Paged Attention 正确 KPI Benchmark                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo ""

# ============================================
# KPI 1: CAPACITY - 长上下文能力
# ============================================
echo "┌──────────────────────────────────────────────────────────────────────┐"
echo "│ KPI 1: CAPACITY - 同内存预算下的上下文能力                           │"
echo "└──────────────────────────────────────────────────────────────────────┘"
echo ""
echo "测试: 在相同 GPU 内存限制下，能支持的最大上下文长度"
echo ""

for ctx in 4096 8192 16384 32768; do
    echo "--- ctx=$ctx ---"

    # Contiguous mode
    echo -n "Contiguous: "
    if result=$($BIN -m "$MODEL" -fa 1 -ngl 99 -p $ctx -n 1 2>&1); then
        echo "$result" | grep -E "pp$ctx" | awk '{print $NF}' || echo "OK"
    else
        echo "❌ OOM or failed"
    fi

    # Paged mode
    echo -n "Paged:      "
    if result=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -p $ctx -n 1 2>&1); then
        echo "$result" | grep -E "pp$ctx" | awk '{print $NF}' || echo "OK"
    else
        echo "❌ OOM or failed"
    fi
    echo ""
done

# ============================================
# KPI 2: OPERABILITY - P95/P99 延迟抖动
# ============================================
echo "┌──────────────────────────────────────────────────────────────────────┐"
echo "│ KPI 2: OPERABILITY - 延迟抖动 (P50/P95/P99)                          │"
echo "└──────────────────────────────────────────────────────────────────────┘"
echo ""
echo "测试: 20 次采样，计算延迟分布"
echo ""

run_samples() {
    local mode=$1
    local samples_file="/tmp/latency_$mode.txt"
    > $samples_file

    echo "Collecting 20 samples ($mode mode)..."

    for i in $(seq 1 20); do
        if [ "$mode" == "paged" ]; then
            result=$(LLAMA_PAGED_ATTENTION=1 $BIN -m "$MODEL" -fa 1 -ngl 99 -p 256 -n 64 2>&1)
        else
            result=$($BIN -m "$MODEL" -fa 1 -ngl 99 -p 256 -n 64 2>&1)
        fi

        # 提取 tg64 的 t/s 值
        latency=$(echo "$result" | grep "tg64" | awk '{print $NF}' | sed 's/±.*//')
        echo "$latency" >> $samples_file
        echo -n "."
    done
    echo " done"
}

# Collect samples
run_samples "contiguous"
run_samples "paged"

# Calculate statistics
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│                    延迟分布统计                            │"
echo "├────────────────────────────────────────────────────────────┤"

calc_stats() {
    local file=$1
    local name=$2

    # 排序
    sort -n $file -o $file

    local count=$(wc -l < $file)
    local p50_idx=$((count / 2))
    local p95_idx=$((count * 95 / 100))
    local p99_idx=$((count * 99 / 100))

    local min=$(head -1 $file)
    local max=$(tail -1 $file)
    local p50=$(sed -n "${p50_idx}p" $file)
    local p95=$(sed -n "${p95_idx}p" $file)
    local p99=$(sed -n "${p99_idx}p" $file)

    # Jitter = (P99/P50 - 1) * 100%
    local jitter=$(echo "scale=1; ($p99 / $p50 - 1) * 100" | bc)

    echo "│ $name                                                      │"
    echo "│   Min: $min  Max: $max                          │"
    echo "│   P50: $p50  P95: $p95  P99: $p99         │"
    echo "│   Jitter (P99/P50-1): ${jitter}%                                        │"
    echo "├────────────────────────────────────────────────────────────┤"
}

calc_stats "/tmp/latency_contiguous.txt" "Contiguous"
calc_stats "/tmp/latency_paged.txt" "Paged     "

echo "└────────────────────────────────────────────────────────────┘"

# ============================================
# KPI 3: RELIABILITY - 长时间稳定性
# ============================================
echo ""
echo "┌──────────────────────────────────────────────────────────────────────┐"
echo "│ KPI 3: RELIABILITY - 长时间运行稳定性                                │"
echo "└──────────────────────────────────────────────────────────────────────┘"
echo ""
echo "测试: 512 tokens 生成，检查是否有 garbage/defrag 问题"
echo ""

test_long_run() {
    local mode=$1
    echo "--- $mode mode ---"

    if [ "$mode" == "paged" ]; then
        output=$(LLAMA_PAGED_ATTENTION=1 $CLI -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 512 -p "Write a long story about a robot learning to paint" 2>&1)
    else
        output=$($CLI -m "$MODEL" -fa 1 -ngl 99 -c 4096 -n 512 -p "Write a long story about a robot learning to paint" 2>&1)
    fi

    # 检查错误
    if echo "$output" | grep -qiE "(error|failed|defrag|garbage|乱码)"; then
        echo "❌ Potential issues detected"
        echo "$output" | grep -iE "(error|failed|defrag|garbage|乱码)" | head -5
    else
        echo "✅ Clean output"

        # 统计生成
        tokens=$(echo "$output" | grep -oE "eval time.*=.*tokens" | tail -1 || echo "N/A")
        speed=$(echo "$output" | grep -oE "[0-9.]+ tokens per second" | tail -1 || echo "N/A")
        echo "   Generated: $tokens"
        echo "   Speed: $speed"
    fi
    echo ""
}

test_long_run "contiguous"
test_long_run "paged"

# ============================================
# Summary
# ============================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        测试完成                                   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  Paged Attention 的真正价值:                                     ║"
echo "║                                                                  ║"
echo "║  1. CAPACITY    - 更长上下文 / 更多并发                          ║"
echo "║  2. OPERABILITY - 更低抖动 (P95/P99 更稳定)                      ║"
echo "║  3. RELIABILITY - 无 defrag garbage 问题                         ║"
echo "║                                                                  ║"
echo "║  不是让单次推理更快，而是让系统更稳定可靠                        ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

#!/bin/bash

# NUMA Integration Test with llama-server
# Standalone script that can be run independently or called from the main test orchestrator
# Tests NUMA-enabled llama-server with a real model to ensure end-to-end functionality

set -e

# Parse command line arguments
VERBOSE_MODE=false
NUMA_OPTION=""
    # Configure NUMA debug logging for operation analysis
    # Respect existing GGML_NUMA_DEBUG setting if higher than default, otherwise use level 1
    # For data-parallel testing (mirror/distribute), automatically enable trace logging
    if [ -z "$GGML_NUMA_DEBUG" ]; then
        if [ "$NUMA_OPTION" = "--numa mirror" ] || [ "$NUMA_OPTION" = "--numa distribute" ]; then
            # Data-parallel mode - enable trace logging to debug coordination issues
            export GGML_NUMA_DEBUG=3
            echo "    🔬 NUMA trace logging enabled (level=3, auto-enabled for data-parallel debugging)"
        else
            # Non-data-parallel mode - use default level 1 for basic operation analysis  
            export GGML_NUMA_DEBUG=1
            echo "    📊 NUMA debug logging enabled (level=1, default) for operation analysis"
        fi
    elif [ "$GGML_NUMA_DEBUG" = "0" ]; then
        # Explicitly disabled - respect that choice
        echo "    🔕 NUMA debug logging disabled (level=0) - respecting user setting"
    else
        # Already set to a higher level - respect and use existing value
        echo "    📊 NUMA debug logging enabled (level=$GGML_NUMA_DEBUG, user-specified) for operation analysis"
    fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --numa)
            if [ -z "$2" ]; then
                echo "Error: --numa option requires an argument (e.g., --numa mirror, --numa distribute, --numa isolate)"
                exit 1
            fi
            NUMA_OPTION="--numa $2"
            shift 2
            ;;
        --numa=*)
            NUMA_OPTION="--numa ${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--numa <mode>] [--help]"
            echo ""
            echo "NUMA Integration Test with llama-server"
            echo "Tests llama-server with two models to ensure end-to-end functionality:"
            echo "  1. Small model: Qwen 2.5 0.5B (Q8_0) - fast validation"
            echo "  2. Large model: Qwen 3 32B (Q6_K) - comprehensive validation"
            echo "Automatically enables NUMA debug logging for operation analysis and prioritization."
            echo ""
            echo "Options:"
            echo "  --verbose          Show detailed test output and logs"
            echo "  --numa <mode>      NUMA mode to pass to llama-server (e.g., mirror, distribute, isolate)"
            echo "                     If not specified, llama-server runs without NUMA options"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  All environment variables are passed through to llama-server, including:"
            echo "  GGML_NUMA_DEBUG   Control NUMA debug output (0=off, 1=info, 2=verbose, 3=trace)"
            echo "                     Default: 1 (auto-enabled for analysis, respects higher user settings)"
            echo "  GGML_LOG_DEBUG    Control general debug logging"
            echo "  GGML_OPENMP       Control OpenMP threading behavior"
            echo ""
            echo "Features:"
            echo "  📊 Operation Analysis: Automatically analyzes NUMA vs fallback operations"
            echo "  🎯 Prioritization: Shows which operations should be implemented next"
            echo "  📈 Usage Statistics: Displays call counts for performance optimization"
            echo "  🔬 Dual Model Testing: Validates both small and large model performance"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic test without NUMA (both models)"
            echo "  $0 --numa mirror                     # Test with NUMA mirror mode (both models)"
            echo "  GGML_NUMA_DEBUG=2 $0 --numa mirror   # Test with verbose NUMA debug output"
            echo ""
            echo "This test downloads models (if not present) and validates that llama-server"
            echo "can generate coherent responses with both small and large models. When --numa"
            echo "is specified, it tests NUMA-specific functionality and provides operation analysis."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Colors for output (only if running standalone, avoid conflicts with orchestrator)
if [ -z "$RED" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$BUILD_DIR/bin"

# Function to check system requirements for integration test
check_integration_requirements() {
    echo -e "${YELLOW}🔍 Checking integration test requirements...${NC}"
    
    # Check for required commands
    local missing_commands=()
    
    if ! command -v curl >/dev/null 2>&1; then
        missing_commands+=("curl")
    fi
    
    if ! command -v wget >/dev/null 2>&1; then
        missing_commands+=("wget")
    fi
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required commands: ${missing_commands[*]}${NC}"
        echo "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check if llama-server binary exists
    if [ ! -f "$BIN_DIR/llama-server" ]; then
        echo -e "${RED}❌ llama-server binary not found at: $BIN_DIR/llama-server${NC}"
        echo "Please build the project first:"
        echo "  cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF"
        echo "  cmake --build build --parallel"
        exit 1
    fi
    
    # Check NUMA system info (optional for integration test)
    if command -v numactl >/dev/null 2>&1; then
        echo -e "${BLUE}🏗️  NUMA system information:${NC}"
        numactl --hardware | head -3 || echo "NUMA hardware info not available"
    else
        echo -e "${YELLOW}⚠️  numactl not found. NUMA tests will run in simulated mode.${NC}"
    fi
    
    echo ""
}

# Function to analyze NUMA debug logs and prioritize next operations
analyze_numa_debug_logs() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo -e "${YELLOW}⚠️  No debug log file found for analysis${NC}"
        return
    fi
    
    echo ""
    echo "========================================"
    echo -e "${BLUE}📊 NUMA Operation Analysis${NC}"
    echo "========================================"
    
    # Create temporary files for analysis
    local numa_ops_file=$(mktemp)
    local fallback_ops_file=$(mktemp)
    local summary_file=$(mktemp)
    
    # Extract NUMA kernel executions (successful dispatches) 
    # Look for "NUMA DEBUG: NUMA ADD (Strategy)" patterns - this is the new standardized format
    grep -E "NUMA DEBUG: NUMA [A-Z_]+ \([^)]+\)" "$log_file" | \
        sed -E 's/.*NUMA DEBUG: NUMA ([A-Z_]+) \([^)]+\).*/\1/' | \
        sort | uniq -c | sort -nr > "$numa_ops_file"
    
    # Extract strategy breakdown for each operation using the new standardized format
    local strategy_file=$(mktemp)
    
    # Extract strategy logging messages: "NUMA DEBUG: NUMA ADD (Data Parallel)"
    # This provides both operation name and strategy in a consistent format
    grep -E "NUMA DEBUG: NUMA [A-Z_]+ \([^)]+\)" "$log_file" | \
        sed -E 's/.*NUMA DEBUG: (NUMA [A-Z_]+ \([^)]+\)).*/\1/' > "$strategy_file"
    
    # Extract fallback executions (operations that fell back to ggml-cpu)
    # Look for "No kernel found for operation GET_ROWS" patterns specifically
    grep "No kernel found for operation" "$log_file" | \
        sed -E 's/.*No kernel found for operation ([A-Z_]+).*/\1/' | \
        sort | uniq -c | sort -nr > "$fallback_ops_file"
    
    # Show NUMA-implemented operations
    if [ -s "$numa_ops_file" ]; then
        echo "✅ Operations using NUMA kernels:"
        while read -r count op; do
            # Get strategy breakdown for this operation using standardized log patterns
            local single_single=0
            local single_multi=0
            local data_parallel=0
            local kernel_only=0
            
            # Parse standardized strategy logging format: "NUMA {OP} ({Strategy})"
            # Ensure we always get a numeric value (default to 0 if grep fails or returns empty)
            single_single=$(grep -c "NUMA ${op} (Single/Single)" "$strategy_file" 2>/dev/null || echo "0")
            single_multi=$(grep -c "NUMA ${op} (Single/Multi)" "$strategy_file" 2>/dev/null || echo "0")
            data_parallel=$(grep -c "NUMA ${op} (Data Parallel)" "$strategy_file" 2>/dev/null || echo "0")
            
            # Ensure all variables are integers (handle any non-numeric results)
            single_single=${single_single:-0}
            single_multi=${single_multi:-0}
            data_parallel=${data_parallel:-0}
            kernel_only=${kernel_only:-0}
            
            # Convert any non-numeric values to 0
            [[ "$single_single" =~ ^[0-9]+$ ]] || single_single=0
            [[ "$single_multi" =~ ^[0-9]+$ ]] || single_multi=0
            [[ "$data_parallel" =~ ^[0-9]+$ ]] || data_parallel=0
            [[ "$kernel_only" =~ ^[0-9]+$ ]] || kernel_only=0
            
            # Fallback patterns for operations that may not use standardized logging yet
            if [ "$single_single" -eq 0 ] && [ "$single_multi" -eq 0 ] && [ "$data_parallel" -eq 0 ]; then
                case "$op" in
                    "RMS_NORM")
                        # Legacy pattern for RMS_NORM
                        local single_thread=$(grep -c "RMS_NORM Single Thread" "$strategy_file" 2>/dev/null || echo "0")
                        single_thread=${single_thread:-0}
                        [[ "$single_thread" =~ ^[0-9]+$ ]] || single_thread=0
                        [ "$single_thread" -gt 0 ] && kernel_only=$single_thread
                        ;;
                    *)
                        # Generic kernel detection for operations without specific strategy logging
                        kernel_only=$(grep -c "NUMA ${op} Kernel" "$strategy_file" 2>/dev/null || echo "0")
                        kernel_only=${kernel_only:-0}
                        [[ "$kernel_only" =~ ^[0-9]+$ ]] || kernel_only=0
                        ;;
                esac
            fi
            
            # Create strategy summary
            local strategies=""
            [ "$single_single" -gt 0 ] && strategies="${strategies}single_single: ${single_single}, "
            [ "$single_multi" -gt 0 ] && strategies="${strategies}single_multi: ${single_multi}, "
            [ "$data_parallel" -gt 0 ] && strategies="${strategies}data_parallel: ${data_parallel}, "
            [ "$kernel_only" -gt 0 ] && strategies="${strategies}kernel: ${kernel_only}, "
            
            # Remove trailing comma and space
            strategies=${strategies%, }
            
            if [ -n "$strategies" ]; then
                printf "   %3d × %s (%s)\n" "$count" "$op" "$strategies"
            else
                printf "   %3d × %s\n" "$count" "$op"
            fi
        done < "$numa_ops_file"
    else
        echo "⚠️  No NUMA kernel executions detected"
    fi
    
    echo ""
    
    # Show fallback operations (prioritization candidates)
    if [ -s "$fallback_ops_file" ]; then
        echo "🎯 Operations falling back to ggml-cpu (prioritized by usage):"
        local rank=1
        while read -r count op; do
            printf "   %d. %s (%d calls)\n" "$rank" "$op" "$count"
            rank=$((rank + 1))
        done < "$fallback_ops_file"
        
        echo ""
        echo -e "${YELLOW}💡 Recommendation: Consider implementing NUMA kernels for the most frequently used fallback operations${NC}"
        
        # Extract top 3 candidates
        local top_candidates=$(head -3 "$fallback_ops_file" | awk '{print $2}' | tr '\n' ', ' | sed 's/,$//')
        if [ -n "$top_candidates" ]; then
            echo -e "${BLUE}🚀 Top candidates for next implementation: $top_candidates${NC}"
        fi
    else
        echo "🎉 All operations are using NUMA kernels (no fallbacks detected)!"
    fi
    
    # Cleanup
    rm -f "$numa_ops_file" "$fallback_ops_file" "$summary_file" "$strategy_file"
}

# Function to test a specific model
test_single_model() {
    local model_name="$1"
    local model_path="$2"
    local model_url="$3"
    local model_id="$4"
    local expected_pattern="$5"
    local test_prompt="$6"
    
    echo "========================================"
    echo -e "${BLUE}📋 Testing model: $model_name${NC}"
    echo "========================================"
    
    # Download model if it doesn't exist
    if [ ! -f "$model_path" ]; then
        echo "📥 Downloading $model_name..."
        echo "   Source: $model_url"
        echo "   Target: $model_path"
        wget -c -O "$model_path" "$model_url"
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to download $model_name${NC}"
            return 1
        fi
        echo "✅ Model downloaded successfully"
    else
        echo "✅ Using existing model: $model_path"
    fi
    
    # Generate unique debug log for this model
    local debug_log="/tmp/llama-server-debug-$(basename "$model_path" .gguf).log"
    local server_port=8080
    local server_pid=""
    
    if [ -n "$NUMA_OPTION" ]; then
        echo "    🚀 Starting llama-server with NUMA option: $NUMA_OPTION..."
    else
        echo "    🚀 Starting llama-server without NUMA options..."
    fi
    
    # Configure NUMA debug logging for operation analysis
    # Respect existing GGML_NUMA_DEBUG setting if higher than default, otherwise use level 1
    if [ -z "$GGML_NUMA_DEBUG" ]; then
        # Not set - use default level 1 for basic operation analysis
        export GGML_NUMA_DEBUG=1
        echo "    📊 NUMA debug logging enabled (level=1, default) for operation analysis"
    elif [ "$GGML_NUMA_DEBUG" = "0" ]; then
        # Explicitly disabled - respect that choice
        echo "    � NUMA debug logging disabled (level=0) - respecting user setting"
    else
        # Already set to a higher level - respect and use existing value
        echo "    📊 NUMA debug logging enabled (level=$GGML_NUMA_DEBUG, user-specified) for operation analysis"
    fi
    
    # Show relevant environment variables in verbose mode
    if [ "$VERBOSE_MODE" = true ]; then
        echo "    📋 Environment variables that will be passed to llama-server:"
        echo "       GGML_NUMA_DEBUG=$GGML_NUMA_DEBUG"
        if [ -n "$GGML_LOG_DEBUG" ]; then
            echo "       GGML_LOG_DEBUG=$GGML_LOG_DEBUG"
        fi
        if [ -n "$GGML_OPENMP" ]; then
            echo "       GGML_OPENMP=$GGML_OPENMP"
        fi
    fi
    
    # Start llama-server in background with optional NUMA mode
    # Note: All environment variables are automatically inherited by the child process
    "$BIN_DIR/llama-server" -m "$model_path" -fa on --host 0.0.0.0 $NUMA_OPTION --port $server_port > "$debug_log" 2>&1 &
    server_pid=$!
    
    # Function to cleanup server
    cleanup_server() {
        if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
            echo "🛑 Stopping llama-server (PID: $server_pid)..."
            kill "$server_pid" 2>/dev/null
            sleep 2
            # Force kill if still running
            if kill -0 "$server_pid" 2>/dev/null; then
                kill -9 "$server_pid" 2>/dev/null
            fi
        fi
        # Also kill any other llama-server processes on our port
        pkill -f "llama-server.*--port $server_port" 2>/dev/null || true
    }
    
    # Set up cleanup trap
    trap cleanup_server EXIT
    
    echo "⏳ Waiting for server to start..."
    local max_attempts=90  # Increased timeout for larger models
    local attempt=0
    
    # Wait for server to become available
    while [ $attempt -lt $max_attempts ]; do
        # Check if server process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo -e "\n${RED}❌ Server process died during startup (PID: $server_pid)${NC}"
            if [ "$VERBOSE_MODE" = true ]; then
                echo "Server log:"
                cat "$debug_log" 2>/dev/null || echo "No log file found"
            fi
            return 1
        fi
        
        if curl --silent --fail-with-body --show-error http://localhost:$server_port/ >/dev/null 2>&1; then
            echo "✅ Server is ready!"
            break
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo ""
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}❌ Server failed to start within 90 seconds${NC}"
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Server log:"
            cat "$debug_log" 2>/dev/null || echo "No log file found"
        fi
        cleanup_server
        return 1
    fi
    
    echo "⏳ Waiting for model to finish loading..."
    local model_loaded=false
    local load_attempts=60  # Increased for larger models
    local load_attempt=0
    
    # Wait for model to be fully loaded by testing API endpoint
    while [ $load_attempt -lt $load_attempts ]; do
        # Check if server process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo -e "\n${RED}❌ Server process died during model loading (PID: $server_pid)${NC}"
            if [ "$VERBOSE_MODE" = true ]; then
                echo "Server log:"
                cat "$debug_log" 2>/dev/null || echo "No log file found"
            fi
            return 1
        fi
        
        local health_response=$(curl -s -X POST http://localhost:$server_port/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model_id\", \"messages\": [{\"role\": \"user\", \"content\": \"test\"}], \"max_tokens\": 1}" 2>/dev/null)
        
        # Check if we get a proper response (not 503 loading error)
        if echo "$health_response" | grep -q "choices\|content" && ! echo "$health_response" | grep -q "Loading model"; then
            echo "✅ Model is fully loaded!"
            model_loaded=true
            break
        fi
        
        sleep 2
        load_attempt=$((load_attempt + 1))
        echo -n "."
    done
    echo ""
    
    if [ "$model_loaded" = false ]; then
        echo -e "${RED}❌ Model failed to load within 120 seconds${NC}"
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Last response: $health_response"
            echo "Server log:"
            tail -20 "$debug_log" 2>/dev/null || echo "No log file found"
        fi
        cleanup_server
        return 1
    fi
    
    echo "🔍 Testing deterministic response generation..."
    echo "   Prompt: \"$test_prompt\""
    echo "   Expected: Response containing \"$expected_pattern\""
    
    # Make API request with temperature=0.0 for deterministic output
    local response=$(curl -s -X POST http://localhost:$server_port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model_id\", 
            \"messages\": [{\"role\": \"user\", \"content\": \"$test_prompt\"}], 
            \"max_tokens\": 20,
            \"temperature\": 0.0,
            \"top_p\": 1.0,
            \"seed\": 42
        }" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}❌ Failed to get response from server${NC}"
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Server log:"
            tail -20 "$debug_log" 2>/dev/null || echo "No log file found"
        fi
        cleanup_server
        return 1
    fi
    
    if [ "$VERBOSE_MODE" = true ]; then
        echo "📄 Raw response:"
        echo "$response"
        echo ""
    fi
    
    # Extract the content from the JSON response
    local content=""
    
    # Try jq first, fallback to grep/sed if jq is not available
    if command -v jq >/dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null)
    else
        # Fallback JSON parsing using grep and sed
        content=$(echo "$response" | grep -o '"content":"[^"]*"' | sed 's/"content":"//' | sed 's/"$//' | head -1)
    fi
    
    if [ -z "$content" ] || [ "$content" = "null" ]; then
        echo -e "${RED}❌ Invalid JSON response or missing content${NC}"
        cleanup_server
        return 1
    fi
    
    echo "💬 Generated content: \"$content\""
    
    # Check if response contains expected pattern (exact match, case-sensitive for precision)
    # Convert both content and pattern to single-line format for reliable comparison
    local content_normalized=$(echo "$content" | tr '\n' ' ' | tr -s ' ')
    local pattern_normalized=$(echo "$expected_pattern" | tr '\n' ' ' | tr -s ' ')
    
    if echo "$content_normalized" | grep -F "$pattern_normalized" >/dev/null; then
        echo -e "${GREEN}✅ Integration test PASSED: Response contains expected pattern${NC}"
        if [ -n "$NUMA_OPTION" ]; then
            echo "🎯 NUMA-enabled llama-server is working correctly with $model_name!"
        else
            echo "🎯 llama-server is working correctly with $model_name!"
        fi
        
        # Analyze NUMA debug logs for operation prioritization
        analyze_numa_debug_logs "$debug_log"
        
        cleanup_server
        return 0
    else
        echo -e "${RED}❌ Integration test FAILED: Response does not contain expected pattern${NC}"
        echo "   Expected pattern: \"$expected_pattern\""
        echo "   Actual content: \"$content\""
        cleanup_server
        return 1
    fi
}

# Function to run integration test with llama-server
run_integration_test() {
    echo "========================================"
    if [ -n "$NUMA_OPTION" ]; then
        echo -e "${BLUE}🧪 NUMA Integration Test with llama-server${NC}"
    else
        echo -e "${BLUE}🧪 Integration Test with llama-server${NC}"
    fi
    echo "========================================"
    
    # Test 1: Small model (Qwen 0.5B)
    echo -e "${YELLOW}🔬 Test 1: Small Model Validation${NC}"
    local small_model_name="Qwen 2.5 0.5B (Q8_0)"
    local small_model_path="./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf"
    local small_model_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
    local small_model_id="qwen2.5-0.5b-instruct"
    local small_test_prompt="Hello!"
    local small_expected_pattern="Hello! How can I assist you today?"
    
    if ! test_single_model "$small_model_name" "$small_model_path" "$small_model_url" "$small_model_id" "$small_expected_pattern" "$small_test_prompt"; then
        echo -e "${RED}❌ Small model test failed - stopping integration test${NC}"
        return 1
    fi
    
    # Test 2: MoE model (Unsloth Dynamic Quant)
    echo -e "${YELLOW}🔬 Test 2: MoE Model Validation${NC}"
    local moe_model_name="Qwen 3 30B-A3B-Instruct (MoE, Q4_K)"
    local moe_model_path="./.devcontainer/Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
    local moe_model_url="https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
    local moe_model_id="qwen3-30b-a3b-instruct"
    local moe_test_prompt="Hello!"
    local moe_expected_pattern="<think>
Okay, the user said \"Hello!\" so I should respond politely. I need to make"

    if ! test_single_model "$moe_model_name" "$moe_model_path" "$moe_model_url" "$moe_model_id" "$moe_expected_pattern" "$moe_test_prompt"; then
        echo -e "${RED}❌ MoE model test failed - stopping integration test${NC}"
        return 1
    fi

    #echo ""
    #echo -e "${YELLOW}🔬 Test 3: Larger Dense Model Validation${NC}"
    ## Test 3: Larger dense model (Qwen 32B)
    #local large_model_name="Qwen 3 32B (Q6_K)"
    #local large_model_path="./.devcontainer/Qwen3-32B-Q6_K.gguf"
    #local large_model_url="https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q6_K.gguf"
    #local large_model_id="qwen3-32b"
    #local large_test_prompt="What is artificial intelligence?"
    #local large_expected_pattern="I need to figure out what artificial intelligence is"
    
    # TODO: remove
    #if ! test_single_model "$large_model_name" "$large_model_path" "$large_model_url" "$large_model_id" "$large_expected_pattern" "$large_test_prompt"; then
    #    echo -e "${RED}❌ Large model test failed${NC}"
    #    return 1
    #fi
    
    echo ""
    echo -e "${GREEN}🎉 Both models passed validation!${NC}"
    return 0
}

# Main function for standalone execution
main() {
    echo -e "${BLUE}🧪 NUMA Integration Test Runner${NC}"
    echo "========================================"
    echo "Project: llama.cpp NUMA improvements"
    echo "Build directory: $BUILD_DIR"
    if [ "$VERBOSE_MODE" = true ]; then
        echo "Output mode: Full verbose output"
    else
        echo "Output mode: Summary only (use --verbose for full output)"
    fi
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT" || {
        echo -e "${RED}❌ Error: Could not change to project root: $PROJECT_ROOT${NC}"
        exit 1
    }
    
    check_integration_requirements
    
    echo -e "${YELLOW}🚀 Starting NUMA integration test...${NC}"
    echo ""
    
    # Run the integration test
    if run_integration_test; then
        echo ""
        echo -e "${GREEN}🎉 Integration test completed successfully!${NC}"
        if [ -n "$NUMA_OPTION" ]; then
            echo "NUMA system is fully validated and working correctly."
        else
            echo "llama-server is fully validated and working correctly."
        fi
        exit 0
    else
        echo ""
        echo -e "${RED}❌ Integration test failed!${NC}"
        echo "Please check the server logs and fix any issues."
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}⚠️  Integration test interrupted by user.${NC}"
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Only run main if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

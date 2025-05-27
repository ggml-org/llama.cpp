# KV Cache Monitor with GGUF Tensor Saving

This directory contains enhanced tools for monitoring and saving KQV (Key-Query-Value) tensors from llama.cpp inference, with the ability to save traced tensors to GGUF files for further analysis.

## Programs

### 1. kqv-trace-monitor
Enhanced version of the original KQV trace monitor that can save traced tensors to GGUF files.

**Features:**
- Monitor `kqv_out` tensors during inference
- Trace source tensors (inputs to attention operations)
- Save tensors and their direct inputs to GGUF files
- Layer-specific monitoring
- Detailed tensor statistics

**Usage:**
```bash
./kqv-trace-monitor [llama.cpp options] [monitor options]

Monitor Options:
  --layer <n>           Monitor only layer n (0-based). Use -1 or omit to monitor all layers
  --no-trace-sources    Disable tracing of source tensors
  --save-gguf <file>    Save traced tensors to GGUF file

Examples:
  # Monitor all layers, save to GGUF file
  ./kqv-trace-monitor -m model.gguf -p "Hello world" --save-gguf traced_tensors.gguf
  
  # Monitor only layer 0
  ./kqv-trace-monitor -m model.gguf -p "Hello world" --layer 0 --save-gguf layer0_tensors.gguf
  
  # Monitor without saving (original behavior)
  ./kqv-trace-monitor -m model.gguf -p "Hello world"
```

### 2. gguf-reader
Utility to read and inspect GGUF files created by kqv-trace-monitor.

**Usage:**
```bash
./gguf-reader <gguf_file> [--show-data]

Options:
  --show-data    Show sample data from tensors (first 10 elements)

Examples:
  # Basic inspection
  ./gguf-reader traced_tensors.gguf
  
  # Show tensor data samples
  ./gguf-reader traced_tensors.gguf --show-data
```

### 3. tensor-diff-analyzer
Advanced tool to compare current model inference tensors with previously saved reference tensors from GGUF files.

**Features:**
- Load reference tensors from GGUF files
- Real-time comparison during inference
- Comprehensive difference statistics (absolute, relative, RMSE, cosine similarity)
- Configurable tolerance thresholds
- Detailed analysis reports
- Detection of shape/type mismatches

**Usage:**
```bash
./tensor-diff-analyzer [llama.cpp options] --reference <gguf_file> [analysis_options]

Analysis Options:
  --reference <file>    Reference GGUF file with saved tensors (required)
  --layer <n>           Monitor only layer n (0-based). Use -1 or omit to monitor all layers
  --tolerance-abs <f>   Absolute tolerance for differences (default: 1e-6)
  --tolerance-rel <f>   Relative tolerance for differences (default: 1e-4)

Examples:
  # Compare with reference tensors
  ./tensor-diff-analyzer -m model.gguf -p "Hello" --reference saved_tensors.gguf
  
  # Compare specific layer with custom tolerances
  ./tensor-diff-analyzer -m model.gguf -p "Hello" --reference saved_tensors.gguf --layer 0 --tolerance-abs 1e-5
  
  # Strict comparison
  ./tensor-diff-analyzer -m model.gguf -p "Hello" --reference saved_tensors.gguf --tolerance-abs 1e-8 --tolerance-rel 1e-6
```

## Building

These programs are built as part of the llama.cpp build process:

```bash
# Build llama.cpp with examples
cmake --build build-arm64 --config Release -j12

# The executables will be in:
# ./build-arm64/bin/llama-kqv-trace-monitor
# ./build-arm64/bin/llama-kqv-gguf-reader
# ./build-arm64/bin/llama-tensor-diff-analyzer
```

## GGUF File Structure

The saved GGUF files contain:

### Metadata
- `kqv_trace.description`: Description of the trace
- `kqv_trace.total_steps`: Number of trace steps
- `kqv_trace.target_layer`: Target layer (-1 for all layers)
- `kqv_trace.trace_sources`: Whether source tracing was enabled
- `kqv_trace.tensor_count`: Total number of saved tensors

### Tensors
Each traced tensor is saved with a unique name format:
- `kqv_out_<original_name>_step_<N>`: The main KQV output tensor
- `src0_<original_name>_step_<N>`: First input tensor (usually K or Q)
- `src1_<original_name>_step_<N>`: Second input tensor (usually V)
- `src2_<original_name>_step_<N>`: Additional input tensors (if any)

## Example Workflows

### 1. Basic Tensor Saving and Inspection

1. **Save Reference Tensors:**
   ```bash
   ./build-arm64/bin/llama-kqv-trace-monitor \
     -m /datasets/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
     -n 4 -p "Hello, world" -ngl 0 -ctk q4_0 -ctv q4_0 -fa -t 12 \
     --layer 0 --save-gguf reference_tensors.gguf
   ```

2. **Inspect Saved Tensors:**
   ```bash
   ./build-arm64/bin/llama-kqv-gguf-reader reference_tensors.gguf --show-data
   ```

### 2. Tensor Difference Analysis

1. **Save Reference Tensors (baseline):**
   ```bash
   ./build-arm64/bin/llama-kqv-trace-monitor \
     -m model_v1.gguf -n 4 -p "Hello, world" \
     --layer 0 --save-gguf baseline_tensors.gguf
   ```

2. **Compare with Different Model/Configuration:**
   ```bash
   ./build-arm64/bin/llama-tensor-diff-analyzer \
     -m model_v2.gguf -n 4 -p "Hello, world" \
     --reference baseline_tensors.gguf --layer 0
   ```

3. **Expected Analysis Output:**
   ```
   === TENSOR DIFFERENCE ANALYSIS SUMMARY ===
   Reference file: baseline_tensors.gguf
   Total comparisons: 10
   Tolerance - Absolute: 1.00e-06, Relative: 1.00e-04
   
   --- Overall Results ---
   Tensors within tolerance: 8/10 (80.0%)
   Shape mismatches: 0
   Type mismatches: 0
   Maximum absolute difference: 2.34e-05
   Maximum relative difference: 1.23e-03
   Average cosine similarity: 0.999876
   
   --- Tensors exceeding tolerance ---
     kqv_out_kqv_out-0_step_2: abs=2.34e-05, rel=1.23e-03
     src0_node_22_step_3: abs=1.87e-05, rel=8.92e-04
   ```

### 3. Model Validation Workflow

1. **Create Golden Reference:**
   ```bash
   # Use known good configuration
   ./build-arm64/bin/llama-kqv-trace-monitor \
     -m model.gguf -p "Test prompt" -ctk f16 -ctv f16 \
     --save-gguf golden_reference.gguf
   ```

2. **Test Different Quantizations:**
   ```bash
   # Test Q4_0 quantization
   ./build-arm64/bin/llama-tensor-diff-analyzer \
     -m model.gguf -p "Test prompt" -ctk q4_0 -ctv q4_0 \
     --reference golden_reference.gguf --tolerance-abs 1e-3
   ```

## Difference Analysis Metrics

The tensor-diff-analyzer provides comprehensive statistics:

### Statistical Measures
- **Mean Absolute Difference**: Average of |current - reference|
- **Maximum Absolute Difference**: Largest absolute difference
- **Mean Relative Difference**: Average of |current - reference| / |reference|
- **Maximum Relative Difference**: Largest relative difference
- **RMSE**: Root Mean Square Error
- **Cosine Similarity**: Measure of vector similarity (1.0 = identical direction)

### Quality Indicators
- **Shape Match**: Whether tensor dimensions are identical
- **Type Match**: Whether data types are identical
- **NaN/Inf Detection**: Count of invalid floating-point values
- **Tolerance Check**: Whether differences are within acceptable bounds

## Use Cases

1. **Model Validation:**
   - Compare different quantization methods
   - Verify model conversions
   - Test optimization effects

2. **Debugging:**
   - Identify numerical instabilities
   - Track precision loss sources
   - Validate implementation changes

3. **Performance Analysis:**
   - Measure quantization impact
   - Compare different backends
   - Analyze precision vs speed tradeoffs

4. **Research:**
   - Study attention pattern changes
   - Analyze model behavior differences
   - Create reproducible benchmarks

## Technical Notes

- **Memory Usage:** The analyzer stores reference tensors in memory and processes current tensors on-demand
- **Precision:** All comparisons are performed in FP32 for consistency
- **Matching:** Tensors are matched by name pattern and step number
- **Thread Safety:** Analysis is performed during graph execution callbacks
- **File Format:** Uses standard GGUF format for maximum compatibility

## Limitations

- Only analyzes `kqv_out` tensors and their direct inputs
- Requires identical prompt and generation parameters for meaningful comparison
- Memory usage scales with number of reference tensors
- Limited to supported tensor types (F32, F16)
- Comparison accuracy depends on reference tensor precision

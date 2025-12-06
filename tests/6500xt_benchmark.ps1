$ErrorActionPreference = "Stop"

# Configuration
$BuildDir = "build"
$ModelPath = "models/7B/ggml-model-f16.gguf" # Adjust as needed
$Prompt = "The quick brown fox jumps over the lazy dog"
$NumRuns = 3
$CsvFile = "benchmark_results.csv"

# Ensure build directory exists
if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Build
Write-Host "Building project..."
Push-Location $BuildDir
cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j 8
Pop-Location

# Tools paths
$LlamaCli = "$BuildDir/bin/Release/llama-cli.exe"
if (!(Test-Path $LlamaCli)) { $LlamaCli = "$BuildDir/bin/llama-cli.exe" }
if (!(Test-Path $LlamaCli)) { $LlamaCli = "$BuildDir/Release/llama-cli.exe" }

$VkInfoTool = "$BuildDir/bin/Release/llama-vk-device-info.exe"
if (!(Test-Path $VkInfoTool)) { $VkInfoTool = "$BuildDir/bin/llama-vk-device-info.exe" }
if (!(Test-Path $VkInfoTool)) { $VkInfoTool = "$BuildDir/Release/llama-vk-device-info.exe" }

# System Info
Write-Host "Collecting System Info..."
vulkaninfo | Out-File "vulkaninfo.txt"
& $VkInfoTool | Out-File "vk_device_info.txt"
Get-Content "vk_device_info.txt"

# Initialize CSV
"RunType,Layers,LoadTime_ms,EvalTime_ms,TokensPerSec,PeakMem_MB" | Out-File $CsvFile -Encoding ascii

function Invoke-Benchmark {
    param (
        [string]$Type,
        [int]$Layers
    )

    $TotalLoadTime = 0
    $TotalEvalTime = 0
    $TotalTokensPerSec = 0
    
    Write-Host "Running benchmark: $Type (Layers: $Layers)"

    for ($i = 1; $i -le $NumRuns; $i++) {
        $LlamaArgs = @("-m", $ModelPath, "-p", $Prompt, "-n", "128", "--no-mmap")
        if ($Type -eq "CPU") {
            $LlamaArgs += "-ngld" # No GPU layers
        }
        elseif ($Type -eq "Vulkan") {
            $LlamaArgs += "-ngl", "$Layers"
        }

        # Capture output
        $Output = & $LlamaCli $LlamaArgs 2>&1
        
        # Parse metrics
        $LoadTime = 0
        $EvalTime = 0
        $Tps = 0
        
        foreach ($Line in $Output) {
            if ($Line -match "load time = \s+(\d+\.\d+) ms") { $LoadTime = [double]$matches[1] }
            if ($Line -match "eval time = \s+(\d+\.\d+) ms") { $EvalTime = [double]$matches[1] }
            if ($Line -match "(\d+\.\d+) tokens per second") { $Tps = [double]$matches[1] }
        }
        
        $TotalLoadTime += $LoadTime
        $TotalEvalTime += $EvalTime
        $TotalTokensPerSec += $Tps
        
        Write-Host "  Run $i : Load=$LoadTime ms, Eval=$EvalTime ms, TPS=$Tps"
    }

    $AvgLoad = $TotalLoadTime / $NumRuns
    $AvgEval = $TotalEvalTime / $NumRuns
    $AvgTps = $TotalTokensPerSec / $NumRuns
    
    "$Type,$Layers,$AvgLoad,$AvgEval,$AvgTps,0" | Out-File $CsvFile -Append -Encoding ascii
}

# Run Benchmarks
Invoke-Benchmark -Type "CPU" -Layers 0

# Test various GPU layers
# Note: If heuristic works, -ngl -1 (default) should pick 1 layer for 6500 XT
# We explicitly test 1, 2, 3, 4 to show performance degradation
Invoke-Benchmark -Type "Vulkan" -Layers 1
Invoke-Benchmark -Type "Vulkan" -Layers 2
Invoke-Benchmark -Type "Vulkan" -Layers 3
Invoke-Benchmark -Type "Vulkan" -Layers 4

Write-Host "Benchmark complete. Results saved to $CsvFile"

# Cross-Architecture Benchmark Script
# Tests dynamic VRAM heuristic across different model architectures and sizes

$buildDir = "..\build"
$llama = "$buildDir\bin\Release\llama-cli.exe"

$models = @(
    @{Name = "Gemma-2-2B"; Path = "..\models\gemma-2b-it\gemma-2-2b-it-Q4_K_M.gguf"; Size = "1.6GB" },
    @{Name = "Llama-3.2-3B"; Path = "..\models\llama-3.2-3b-instruct-q4_k_m.gguf"; Size = "1.9GB" },
    @{Name = "Llama-2-7B"; Path = "..\models\llama-2-7b-chat.Q4_K_M.gguf"; Size = "3.9GB" },
    @{Name = "Llama-2-13B"; Path = "..\models\llama-2-13b-chat.Q4_K_M.gguf"; Size = "7.5GB" }
)

$results = @()

foreach ($model in $models) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $($model.Name) ($($model.Size))" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    # Test 1: CPU Only (-ngl 0)
    Write-Host "Test 1: CPU Only..." -ForegroundColor Yellow
    $output = & $llama -m $model.Path -p "Test" -n 10 -ngl 0 -no-cnv 2>&1 | Out-String
    $cpuTokens = if ($output -match "(\d+\.\d+)\s+tokens per second") { [float]$matches[1] } else { 0 }
    
    # Test 2: Dynamic Heuristic (no -ngl flag)
    Write-Host "Test 2: Dynamic Heuristic..." -ForegroundColor Yellow
    $output = & $llama -m $model.Path -p "Test" -n 10 -no-cnv 2>&1 | Out-String
    $heuristicLayers = if ($output -match "calculated_layers=(\d+)") { [int]$matches[1] } else { "N/A" }
    $offloadedLayers = if ($output -match "offloaded (\d+)/(\d+) layers") { "$($matches[1])/$($matches[2])" } else { "N/A" }
    $heuristicTokens = if ($output -match "(\d+\.\d+)\s+tokens per second") { [float]$matches[1] } else { 0 }
    
    $speedup = if ($cpuTokens -gt 0) { [math]::Round(($heuristicTokens / $cpuTokens - 1) * 100, 1) } else { 0 }
    
    $results += [PSCustomObject]@{
        Model                 = $model.Name
        Size                  = $model.Size
        CPUTokensPerSec       = [math]::Round($cpuTokens, 2)
        HeuristicLayers       = $heuristicLayers
        OffloadedLayers       = $offloadedLayers
        HeuristicTokensPerSec = [math]::Round($heuristicTokens, 2)
        SpeedupPercent        = "$speedup%"
    }
}

# Display results
Write-Host "`n`n========================================" -ForegroundColor Green
Write-Host "BENCHMARK RESULTS" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

$results | Format-Table -AutoSize

# Save to CSV
$results | Export-Csv -Path "cross_arch_benchmark_results.csv" -NoTypeInformation
Write-Host "`nResults saved to: cross_arch_benchmark_results.csv" -ForegroundColor Green

# compile_shaders.ps1 — PowerShell wrapper, easier to run from llama root
#
# Usage (from llama.cpp-ROCM-Test dir):
#   .\ufm_package\scripts\compile_shaders.ps1
#   .\ufm_package\scripts\compile_shaders.ps1 -LlamaRoot C:\Users\rr\Desktop\llama.cpp-ROCM-Test

param(
    [string]$LlamaRoot = (Resolve-Path "$PSScriptRoot\..\.." -ErrorAction SilentlyContinue),
    [string]$GlslcPath = ""
)

$ErrorActionPreference = "Stop"

# ── Locate glslc ─────────────────────────────────────────────────────────────
if (-not $GlslcPath) {
    $found = Get-Command glslc -ErrorAction SilentlyContinue
    if ($found) {
        $GlslcPath = $found.Source
    } else {
        $sdkRoot = "C:\VulkanSDK"
        if (Test-Path $sdkRoot) {
            $candidate = Get-ChildItem $sdkRoot -Directory |
                Sort-Object Name -Descending |
                Where-Object { Test-Path "$($_.FullName)\Bin\glslc.exe" } |
                Select-Object -First 1
            if ($candidate) { $GlslcPath = "$($candidate.FullName)\Bin\glslc.exe" }
        }
    }
}

if (-not $GlslcPath -or -not (Test-Path $GlslcPath)) {
    Write-Error "glslc not found. Install Vulkan SDK 1.4.341+ from https://vulkan.lunarg.com/sdk/home#windows"
}

Write-Host "[UFM] glslc: $GlslcPath"
& $GlslcPath --version 2>&1 | Select-String "shaderc|glslc" | Select-Object -First 1

# ── Paths ─────────────────────────────────────────────────────────────────────
$ShaderDir = "$PSScriptRoot\..\shaders"
$OutDir    = "$LlamaRoot\ggml\src\ggml-vulkan\spv_out"

if (-not (Test-Path "$LlamaRoot\ggml\src\ggml-vulkan")) {
    Write-Error "Cannot find $LlamaRoot\ggml\src\ggml-vulkan`nPass -LlamaRoot correctly."
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
Write-Host "[UFM] llama root: $LlamaRoot"
Write-Host "[UFM] Output:     $OutDir"
Write-Host ""

# ── Compile ───────────────────────────────────────────────────────────────────
$shaders = Get-ChildItem $ShaderDir -Filter "*.glsl" | Sort-Object Name
$errors  = 0

Write-Host "[UFM] Compiling shaders (target: vulkan1.3)..."
foreach ($s in $shaders) {
    $spv = "$OutDir\$($s.BaseName).spv"
    $result = & $GlslcPath --target-env=vulkan1.3 -O -fshader-stage=compute $s.FullName -o $spv 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("  [OK]   " + $s.Name)
    } else {
        Write-Host ("  [FAIL] " + $s.Name)
        $result | ForEach-Object { Write-Host "         $_" }
        $errors++
    }
}

if ($errors -gt 0) {
    Write-Error "$errors shader(s) failed. Fix errors above."
}

# ── Embed as C headers ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[UFM] Embedding SPIR-V as C headers..."

$spvFiles = Get-ChildItem $OutDir -Filter "*.spv" | Sort-Object Name
$ok = 0
foreach ($spv in $spvFiles) {
    $stem   = $spv.BaseName
    $bytes  = [System.IO.File]::ReadAllBytes($spv.FullName)
    $hdrPath = "$OutDir\${stem}_spv.h"
    $sb = [System.Text.StringBuilder]::new()
    $null = $sb.AppendLine("// Auto-generated -- do not edit. Source: ${stem}.glsl")
    $null = $sb.AppendLine("static const uint32_t ${stem}_len  = $($bytes.Length);")
    $null = $sb.AppendLine("static const uint8_t  ${stem}_data[] = {")
    $line = ""
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        $line += ("0x{0:x2}," -f $bytes[$i])
        if (($i + 1) % 16 -eq 0) { $null = $sb.AppendLine($line); $line = "" }
    }
    if ($line) { $null = $sb.AppendLine($line) }
    $null = $sb.AppendLine("};")
    [System.IO.File]::WriteAllText($hdrPath, $sb.ToString())
    Write-Host ("  [OK]   ${stem}_spv.h  ($($bytes.Length) bytes)")
    $ok++
}

Write-Host ""
Write-Host "[UFM] $ok headers written to $OutDir"
Write-Host ""
Write-Host "[UFM] Done. Next steps:"
Write-Host "  1. git apply ufm_package\src\ggml_vulkan_custom_kernels.patch"
Write-Host "  2. Add-Content ggml\src\ggml-vulkan\CMakeLists.txt (Get-Content ufm_package\src\CMakeLists_append.cmake)"
Write-Host "  3. cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release"
Write-Host "  4. cmake --build build --config Release -j"

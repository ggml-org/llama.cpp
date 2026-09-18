[CmdletBinding()]
param([string]$Compiler='', [string]$Builtins='', [switch]$GroupingOnly)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Build=Join-Path $Root 'build'
New-Item -ItemType Directory -Force $Build | Out-Null
if(-not $Compiler){
    foreach($Name in @('icpx.exe','icx.exe')){
        $Found=Get-Command $Name -ErrorAction SilentlyContinue
        if($Found){$Compiler=$Found.Source;break}
    }
}
if(-not $Compiler){throw 'Intel oneAPI compiler not found. Initialize oneAPI + MSVC first.'}
$CC=Get-Command $Compiler -ErrorAction SilentlyContinue
if(-not $CC){throw 'Intel oneAPI compiler not found. Use build_and_test.cmd or an initialized oneAPI + MSVC shell.'}
function Invoke-Logged([string]$Program,[string[]]$CommandArgs,[string]$Log){
    Write-Host ($Program+' '+($CommandArgs -join ' '))
    $Saved=$ErrorActionPreference
    try{$ErrorActionPreference='Continue'; & $Program @CommandArgs 2>&1 | Tee-Object -FilePath $Log; $RC=$LASTEXITCODE}
    finally{$ErrorActionPreference=$Saved}
    if($RC -ne 0){throw "Command failed (exit $RC). Preserve $Log. No server DLL was changed."}
}
if($Builtins -and -not(Test-Path -LiteralPath $Builtins)){throw "Builtins library not found: $Builtins"}
if(-not $Builtins){
    # The successful user run explicitly linked this compiler-runtime library.
    # Resolve it from this compiler, without hardcoding oneAPI/clang versions.
    $Saved=$ErrorActionPreference
    try{$ErrorActionPreference='Continue';$Resource=& $CC.Source '-print-resource-dir' 2>$null;$RC=$LASTEXITCODE}
    finally{$ErrorActionPreference=$Saved}
    if($RC -eq 0 -and $Resource){
        $Dir=([string](@($Resource)[-1])).Trim()
        $Candidate=Join-Path $Dir 'lib\windows\clang_rt.builtins-x86_64.lib'
        if(Test-Path -LiteralPath $Candidate){$Builtins=$Candidate}
    }
}
$LinkExtra=@();if($Builtins){$LinkExtra+=@($Builtins)}
Push-Location $Root
try{
    Invoke-Logged $CC.Source @('--version') (Join-Path $Build 'compiler-version.txt')
    $CPUFlags=@('-fexceptions','-fcxx-exceptions','-std=c++17','-O2','-Iinclude')
    foreach($Test in @(@('reference_tests','maple-reference-tests'),@('w2a8_reference_tests','maple-a8-reference-tests'),@('moe_reference_tests','maple-moe-reference-tests'),@('grouping_reference_tests','maple-grouping-reference-tests'),@('architecture_reference_tests','maple-architecture-reference-tests'))){
        Invoke-Logged $CC.Source ($CPUFlags+@("tests/$($Test[0]).cpp",'-o',"build/$($Test[1]).exe")+$LinkExtra) (Join-Path $Build "$($Test[1])-build.log")
        Invoke-Logged (Join-Path $Build "$($Test[1]).exe") @() (Join-Path $Build "$($Test[1])-tests.log")
    }
    $Flags=@('-fexceptions','-fcxx-exceptions','-std=c++17','-fsycl','-O3','-fno-fast-math','-fsycl-device-code-split=per_kernel','-Iinclude')
    $Objects=@()
    foreach($Source in @('maple_w2a16','maple_w2a8','maple_moe','expert_grouping','maple_w2a8_grouped')){
        $Object="build/$Source.obj"
        Invoke-Logged $CC.Source ($Flags+@('-c',"src/$Source.cpp",'-o',$Object)) (Join-Path $Build "$Source-build.log")
        $Objects+=@($Object)
    }
    $Tools=New-Object 'System.Collections.Generic.List[object]'
    $Tools.Add(@('grouping_compare','maple-grouping-compare'));$Tools.Add(@('architecture_compare','maple-architecture-compare'))
    if(-not $GroupingOnly){
        $Tools.Add(@('compare','maple-w2-compare'));$Tools.Add(@('bench','maple-w2-bench'));$Tools.Add(@('moe_compare','maple-moe-compare'))
    }
    foreach($Tool in $Tools){
        Invoke-Logged $CC.Source ($Flags+@("tools/$($Tool[0]).cpp")+$Objects+$LinkExtra+@('-o',"build/$($Tool[1]).exe")) (Join-Path $Build "$($Tool[1])-link.log")
    }
    $Hashes=[ordered]@{}
    $SourceFiles=@(Get-ChildItem (Join-Path $Root 'include') -Filter '*.hpp')+@(Get-ChildItem (Join-Path $Root 'src') -Filter '*.cpp')+@(Get-Item (Join-Path $Root 'tools/grouping_compare.cpp'))
    foreach($File in ($SourceFiles | Sort-Object FullName)){
        $Relative=$File.FullName.Substring($Root.Length+1).Replace('\','/')
        $Hashes[$Relative]=(Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $Manifest=[ordered]@{version='0.4';compiler=$CC.Source;flags=$Flags;source_sha256=$Hashes;
        executable_sha256=(Get-FileHash -LiteralPath (Join-Path $Build 'maple-grouping-compare.exe') -Algorithm SHA256).Hash.ToLowerInvariant()}
    $Manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Build 'grouping-build.json') -Encoding UTF8
    Invoke-Logged 'python' @('tools/stamp_architecture_build.py','--exe','build/maple-architecture-compare.exe') (Join-Path $Build 'architecture-stamp.log')
    Write-Host 'Build complete. GPU kernels compiled once for all comparators. No ggml DLL or launch settings changed.'
}finally{Pop-Location}

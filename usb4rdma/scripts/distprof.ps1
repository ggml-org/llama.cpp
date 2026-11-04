# Parser for dist-tp rank logs. Dot-sourced by run_disttp.ps1.
#
#   Get-DistprofSummary  per-token cost breakdown from LLAMA_DISTPROF lines
#   Get-GeneratedText    the model's output text, for the correctness hash
#   Get-RunAnalysis      transport, outcome and correctness, from EVERY rank's log
#   Get-MeshTcpRanks     ranks whose full-mesh banner puts any direction on TCP
#
# LLAMA_DISTPROF prints one [distprof] line per GRAPH evaluation, not per token. A run's lines are
# the first graph, then one per prompt ubatch, then one per decode token. Prompt ubatches carry
# hundreds of tokens' worth of payload each, so they must be excluded before averaging, or a few
# multi-second prompt graphs inflate every per-token figure. Decode graphs are selected as the
# SMALLEST payload that recurs: each decode graph moves one token's worth per collective. (The most
# frequent payload is not a safe rule: a long prompt can have more ubatches than there are decode
# tokens.) The mean over decode graphs is cross-checked against llama.cpp's own decode rate, and a
# miss above 10% is flagged.
#
# Single-rank runs print no [distprof] lines (the profiler lives in the dist-tp collective).

# The model's generated text from a rank-0 log, anchored on llama.cpp's '>' prompt echo so the
# startup banner is never part of it. Each [distprof] record is removed together with the newline
# the profiler put after it, so 1-rank and N-rank runs of the same model hash identically.
function Get-GeneratedText {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Text)
    $t = $Text -replace '\[distprof\][^\r\n]*(\r?\n)?', ''
    $lines = @($t -split "`r?`n")
    $start = -1
    for($i = 0; $i -lt $lines.Count; $i++){ if("$($lines[$i])".TrimStart().StartsWith('>')){ $start = $i; break } }
    if($start -lt 0){ return '' }
    $g = ($lines[$start..($lines.Count - 1)] -join ' ')
    # Drop llama.cpp's own perf line and anything after it; that is not model output.
    $g = ($g -split '\[\s*Prompt:')[0]
    ($g -replace '\s+', ' ').Trim()
}

function Get-DistprofSummary {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Text)

    $rows = @()
    foreach($ln in ($Text -split "`r?`n")){
        if($ln -match 'total=([\d.]+)ms\s+nbnd=(\d+)\s*\|\s*sync\(gpu\)=([\d.]+)\s+get=([\d.]+)\s+net=([\d.]+)\s+setadd=([\d.]+)\s+commtot=[\d.]+\s+other=(-?[\d.]+)\s+bytes=(\d+)'){
            $rows += [pscustomobject]@{
                total=[double]$Matches[1]; nbnd=[int]$Matches[2]; sync=[double]$Matches[3]
                get=[double]$Matches[4];   net=[double]$Matches[5]; setadd=[double]$Matches[6]
                other=[double]$Matches[7]; bytes=[int64]$Matches[8]
            }
        }
    }
    if($rows.Count -lt 2){ return $null }

    # Decode graphs = the smallest payload that occurs at least twice. See the header.
    $mode = ($rows | Where-Object { $_.bytes -gt 0 } | Group-Object bytes | Where-Object { $_.Count -ge 2 } |
             Sort-Object { [int64]$_.Name } | Select-Object -First 1).Name
    if(-not $mode){ return $null }
    $dec  = @($rows | Where-Object { "$($_.bytes)" -eq "$mode" })
    $excl = @($rows | Where-Object { "$($_.bytes)" -ne "$mode" })
    if($dec.Count -lt 2){ return $null }
    # A run that died during the prompt has no decode graphs; its repeated prompt batches are then
    # the smallest REPEATED payload while a smaller one-off graph exists. Report nothing, rather
    # than prompt batches as "decode tokens".
    if(@($excl | Where-Object { $_.bytes -gt 0 -and $_.bytes -lt [int64]$mode }).Count){ return $null }

    function M([string]$f, $rs){ [math]::Round((($rs | Measure-Object -Property $f -Average).Average), 2) }

    $total = M 'total' $dec
    [pscustomobject]@{
        ms_total   = $total
        ms_sync    = M 'sync' $dec
        ms_get     = M 'get' $dec
        # GPU wait is sync+get ADDED: which of the two it lands in depends on the backend.
        ms_gpu_wait= [math]::Round((M 'sync' $dec) + (M 'get' $dec), 2)
        # Time this rank spent BLOCKED inside the collective. It includes waiting for the slowest
        # peer to arrive, so it is an upper bound on wire time, and it differs between ranks.
        ms_net     = M 'net' $dec
        ms_setadd  = M 'setadd' $dec
        ms_other   = M 'other' $dec
        # Tensor bytes reduced per decode graph (nbnd x one token's hidden state), not bytes on the wire.
        bytes      = [int64](($dec | Measure-Object -Property bytes -Average).Average)
        nbnd       = [int](($dec | Measure-Object -Property nbnd -Average).Average)
        tokens     = $dec.Count
        excluded   = $excl.Count
        excluded_totals_ms = (($excl | ForEach-Object { [math]::Round($_.total,0) }) -join ',')
        implied_decode_tps = $(if($total -gt 0){ [math]::Round(1000.0 / $total, 2) } else { $null })
        # The excluded graphs are the prompt, so their summed time is the time to first token (host
        # gaps between graphs aside).
        prefill_ms    = [math]::Round((($excl | Measure-Object -Property total -Sum).Sum), 0)
        prompt_tokens = [int](($excl | ForEach-Object { [math]::Round($_.bytes / ($_.nbnd * ([double]$mode / $dec[0].nbnd))) } | Measure-Object -Sum).Sum)
        # The largest single all-reduce payload of the run, prompt included. Above the ND ceiling a
        # collective goes over TCP, so on an RDMA run this must not exceed it.
        max_coll_bytes = [int64](($rows | Where-Object { $_.nbnd -gt 0 } | ForEach-Object { [math]::Ceiling($_.bytes / $_.nbnd) } | Measure-Object -Maximum).Maximum)
    }
}

# Everything a result needs, computed from ALL rank logs.
#
# All ranks, because each rank logs only the ring edges it is an endpoint of: rank 0 cannot see edge
# 1->2 of a 3-rank ring.
#
# The headline counts too: a rank with no usable NetworkDirect provider, or a binary built without
# ND, prints "dist-tp transport: TCP" and no per-edge failure line. On an RDMA run such a rank has
# both of its ring edges counted as failed.
#
# And evidence is required: an empty or truncated log has no failure lines either. Without a
# transport line from EVERY rank, nd_edges_up is null, never N of N.
function Get-RunAnalysis {
    param(
        # One entry per rank, index = rank. An unreadable log must be '' (never omitted), or the
        # ranks after it shift down and are attributed to the wrong ring position.
        [Parameter(Mandatory)][AllowEmptyCollection()][AllowEmptyString()][string[]]$RankTexts,
        # What the run ASKED for: rdma, tcp or none. The result reports what actually happened.
        [Parameter(Mandatory)][ValidateSet('rdma','tcp','none')][string]$Transport,
        # Optional: the ring-next address each rank should dial, index = rank. Each rank logs
        # "ring edge r->r+1 dials <ip>:<port>", so a stale log from a different ring is caught.
        [string[]]$ExpectedNext = @()
    )
    $world = $RankTexts.Count
    $r0 = "$($RankTexts[0])"
    $allText = (@($RankTexts) | ForEach-Object { "$_" }) -join "`n"
    $allLines = @($allText -split "`r?`n")

    # ---- does every rank's log belong to THIS ring? ----
    $foreign = @()
    if($world -gt 1 -and $ExpectedNext.Count -eq $world){
        for($r = 0; $r -lt $world; $r++){
            if("$($RankTexts[$r])" -match ('ring edge {0}->\d+ dials ([\d.]+):' -f $r)){
                if($Matches[1] -ne $ExpectedNext[$r]){ $foreign += $r }
            }
        }
    }

    # ---- transport, per rank and per edge ----
    $headlines = @(for($r = 0; $r -lt $world; $r++){
        if("$($RankTexts[$r])" -match 'dist-tp transport:\s*(\w+)'){ $Matches[1].ToUpper() } else { '' }
    })
    $failed = @($allLines | Where-Object { $_ -match 'NetworkDirect not established on ring edge\s+\d+->\d+' } |
                ForEach-Object { if($_ -match 'ring edge\s+(\d+->\d+)'){ $Matches[1] } })
    # Whole-rank fallbacks. None of these print a per-edge line.
    $rankFallbackPat = 'falling back to TCP|no usable NetworkDirect|built without NetworkDirect|staying on TCP'
    if($Transport -eq 'rdma' -and $world -gt 1){
        for($r = 0; $r -lt $world; $r++){
            if($headlines[$r] -eq 'TCP' -or "$($RankTexts[$r])" -match $rankFallbackPat){
                $failed += ('{0}->{1}' -f $r, (($r + 1) % $world))
                $failed += ('{0}->{1}' -f (($r - 1 + $world) % $world), $r)
            }
        }
    }
    $failed = @($failed | Select-Object -Unique)
    $haveEvidence = ($world -gt 1) -and (@($headlines | Where-Object { -not $_ }).Count -eq 0)
    $expected = if($world -gt 1){ $world } else { 0 }
    $ndUp = if($Transport -eq 'rdma' -and $haveEvidence){ [Math]::Max(0, $expected - $failed.Count) } else { $null }
    $wantHead = if($Transport -eq 'rdma'){ 'RDMA' } elseif($Transport -eq 'tcp'){ 'TCP' } else { '' }
    $headMismatch = @(for($r = 0; $r -lt $world; $r++){ if($wantHead -and $headlines[$r] -and $headlines[$r] -ne $wantHead){ $r } })
    $fellBack = [bool](($allText -match $rankFallbackPat) -or $failed.Count -or $headMismatch.Count)
    # The per-edge list is printed at 3+ ranks, e.g. "(->1 ND, 2-> TCP)".
    $edgeSummary = ''
    if($r0 -match 'dist-tp transport:[^\r\n]*?\(([^()]*(?:ND|TCP)[^()]*)\)\s*\(ND up to'){ $edgeSummary = $Matches[1] }
    elseif($r0 -match 'dist-tp transport:[^\r\n]*?\(([^()]*(?:ND|TCP)[^()]*)\)'){ $edgeSummary = $Matches[1] }
    $ndMax = $null
    if($r0 -match 'ND up to (\d+) B'){ $ndMax = [int64]$Matches[1] }
    elseif($r0 -match 'all payloads over ND'){ $ndMax = 0 }

    # ---- build identity: every rank must run the same binary ----
    $builds = @(for($r = 0; $r -lt $world; $r++){ if("$($RankTexts[$r])" -match 'build\s*:\s*(\S+)'){ $Matches[1] } else { '' } })
    $buildSet = @($builds | Where-Object { $_ } | Select-Object -Unique)

    # ---- outcome ----
    $pref = $null; $dec = $null
    foreach($ln in ($r0 -split "`r?`n")){
        if($ln -match '\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s'){ $pref = [double]$Matches[1]; $dec = [double]$Matches[2] }
    }
    # Any rank can be the one that fails to allocate.
    $oomPat = 'failed to allocate|unable to allocate|out of memory|ErrorOutOfDeviceMemory|ggml_vulkan: Device memory allocation of|cannot allocate|std::bad_alloc'
    $oomLines = @($allLines | Where-Object { $_ -match $oomPat })
    $fatalLines = @($allLines | Where-Object { $_ -match 'GGML_ASSERT|Failed to load the model|failed to load model' })

    $status = if($pref -and $dec){ 'ok' } elseif($oomLines.Count){ 'oom' } else { 'no_perf_line' }
    # A TCP run that actually ran RDMA (a node with LLAMA_DIST_TP_RDMA set system-wide) is not a TCP
    # result.
    if($status -eq 'ok' -and $Transport -eq 'tcp' -and $headMismatch.Count){ $status = 'transport_mismatch' }
    if($status -eq 'ok' -and $buildSet.Count -gt 1){ $status = 'build_mismatch' }
    if($foreign.Count){ $status = 'foreign_rank_log' }
    # An RDMA run is only 'ok' if EVERY ring edge came up on NetworkDirect. The build's own line says
    # "transport: RDMA (ND)" on any rank with at least one ND edge, even when its other edge stayed
    # on TCP, so a partial ring gets its own status.
    if($status -eq 'ok' -and $Transport -eq 'rdma' -and $world -gt 1){
        if($null -eq $ndUp){ $status = 'rdma_unverified' }
        elseif($ndUp -lt $expected){ $status = 'partial_rdma' }
    }

    $gen = Get-GeneratedText -Text $r0
    $sha = [BitConverter]::ToString((New-Object Security.Cryptography.SHA256Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($gen))).Replace('-','')

    # Rank 0 is the run's figure; every rank's net time is kept because it differs between ranks
    # (the slowest rank's peers wait for it inside the collective).
    $prof = Get-DistprofSummary -Text $r0
    $netByRank = @(for($r = 0; $r -lt $world; $r++){
        $p = if($r -eq 0){ $prof } else { Get-DistprofSummary -Text "$($RankTexts[$r])" }
        if($p){ $p.ms_net } else { 'n/a' }
    })
    # If the profiler's implied rate misses llama.cpp's own by more than 10%, the graph selection is
    # wrong and the per-token figures are not reliable.
    $profSuspect = $false
    if($prof -and $dec){ $profSuspect = ([math]::Abs($prof.implied_decode_tps - $dec) / $dec) -gt 0.10 }

    [pscustomobject]@{
        headlines = $headlines
        transport_fell_back = $fellBack
        nd_edges_up = $ndUp
        nd_edges_expected = $expected
        nd_failed_edges = @($failed | Sort-Object)
        rdma_clean = [bool]($Transport -eq 'rdma' -and $null -ne $ndUp -and $ndUp -eq $expected)
        edge_summary = $edgeSummary
        nd_max_bytes = $ndMax
        head_mismatch_ranks = $headMismatch
        foreign_rank_logs = $foreign
        builds = $builds
        status = $status
        prefill_tps = $pref
        decode_tps = $dec
        oom_lines = $oomLines
        fatal_lines = $fatalLines
        gen_text = $gen
        out_sha256 = $(if($gen){ $sha } else { $null })
        prof = $prof
        ms_net_by_rank = $netByRank
        prof_suspect = $profSuspect
    }
}

# Ranks whose full-mesh banner names TCP for any edge. An all-RDMA mesh run must have 0 here AND a
# clean ring.
function Get-MeshTcpRanks([string[]]$RankTexts){
    @($RankTexts | Where-Object {
        $m = [regex]::Match("$_", 'dist-tp all-reduce: full mesh[^\r\n]*')
        $m.Success -and $m.Value -match 'TCP'
    }).Count
}

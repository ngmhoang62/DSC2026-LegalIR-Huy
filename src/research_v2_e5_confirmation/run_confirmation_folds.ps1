$ErrorActionPreference = 'Stop'

$Repo = 'D:\Study\DSC2026\sota'
$Python = 'D:\Study\DSC2026\dsc_env\Scripts\python.exe'
$Runner = Join-Path $Repo 'src\research_v2_e5_confirmation\e5_confirmation_runner.py'
$CoreRunner = Join-Path $Repo 'src\research_v2_e5_transfer\e5_transfer_runner.py'
$Bundle = Join-Path $Repo 'cache\research_v2_e5_confirmation\bundle-v1'
$Root = Join-Path $Repo 'results\research_v2_e5_confirmation'
$ExpectedRunner = 'add2eac9ab62a21e9aa6396b95ec3d6d43f0d5dd886a1e6ae51b03bf03137023'
$ExpectedCore = 'b674c9756b26d79966734d8acb928013880c80a150f5326462056055b3d3fd9b'

New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Transcript = Join-Path $Root 'STRICT_CONFIRMATION_FOLDS_1_4_TRANSCRIPT.log'
$env:PYTHONPATH = Join-Path $Repo 'src'

function Assert-Hash([string]$Path, [string]$Expected) {
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "Hash mismatch for ${Path}: ${Actual} != ${Expected}"
    }
}

Start-Transcript -LiteralPath $Transcript -Append
try {
    foreach ($FoldNumber in 1..4) {
        Assert-Hash $Runner $ExpectedRunner
        Assert-Hash $CoreRunner $ExpectedCore
        $Fold = "fold_${FoldNumber}"
        $FoldRoot = Join-Path $Root $Fold
        $Training = Join-Path $FoldRoot 'training'
        $Score = Join-Path $FoldRoot 'score'
        Write-Output "CONFIRMATION_START ${Fold}"
        & $Python $Runner train --bundle $Bundle --held-fold $Fold --output $Training --microbatch 4
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for ${Fold} with exit ${LASTEXITCODE}"
        }
        Assert-Hash $Runner $ExpectedRunner
        Assert-Hash $CoreRunner $ExpectedCore
        $Checkpoint = Join-Path $Training 'epoch-2.pt'
        & $Python $Runner score --bundle $Bundle --held-fold $Fold --output $Score --checkpoint $Checkpoint
        if ($LASTEXITCODE -ne 0) {
            throw "Scoring failed for ${Fold} with exit ${LASTEXITCODE}"
        }
        Write-Output "CONFIRMATION_COMPLETE ${Fold}"
    }
    Write-Output 'STRICT_CONFIRMATION_FOLDS_1_4_COMPLETE'
}
finally {
    Stop-Transcript
}

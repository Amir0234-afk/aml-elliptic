# scripts/run_all.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Phase {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string]$Module
    )

    $cmd = @("python", "-m", $Module)

    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host ("=== {0}" -f $Label)
    Write-Host ("$ {0}" -f ($cmd -join " "))
    Write-Host ("=" * 78)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $cmd[0] $cmd[1] $cmd[2]

        if ($LASTEXITCODE -ne 0) {
            throw ("Command failed with exit code {0}: {1}" -f $LASTEXITCODE, ($cmd -join " "))
        }
    }
    finally {
        $sw.Stop()
        Write-Host ("--- {0} finished in {1:mm\:ss} ---" -f $Label, $sw.Elapsed)
    }
}

Invoke-Phase -Label "Phase 01 - Preprocessing" -Module "src.phase01_preprocessing"
Invoke-Phase -Label "Phase 02 - EDA"          -Module "src.phase02_eda"
Invoke-Phase -Label "Phase 03 - Models"       -Module "src.phase03_models"
Invoke-Phase -Label "Phase 04 - Tuning"       -Module "src.phase04_tuning"
Invoke-Phase -Label "Phase 05 - Eval/Infer"   -Module "src.phase05_eval_infer"

Write-Host ""
Write-Host "DONE."
Write-Host "Check:"
Write-Host "  ./results/metrics"
Write-Host "  ./results/model_artifacts"
Write-Host "  ./results/visualizations"
Write-Host "  ./results/logs"

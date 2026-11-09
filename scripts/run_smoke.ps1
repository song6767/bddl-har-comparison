param(
    [string]$Python = "python",
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto",
    [ValidateRange(1, 1000)][int]$Epochs = 5,
    [int]$Seed = 42,
    [string]$OutputRoot = "runs/smoke",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$overwriteArgs = @()
if ($Overwrite) { $overwriteArgs = @("--overwrite") }
Push-Location -LiteralPath $projectRoot
try {
    & $Python -m har_compare.prepare --data-dir data
    if ($LASTEXITCODE -ne 0) { throw "Data preparation failed (exit $LASTEXITCODE)." }
    foreach ($model in @("cnn", "rnn", "lstm")) {
        $runOutput = Join-Path $OutputRoot $model
        & $Python -m har_compare.train --model $model --data-dir data/processed --output-dir $runOutput --epochs $Epochs --seed $Seed --batch-size 64 --device $Device --lr 0.001 --patience 8 --stage smoke @overwriteArgs
        if ($LASTEXITCODE -ne 0) { throw "$model smoke run failed (exit $LASTEXITCODE)." }
    }
    Write-Host "Completed all three smoke runs. These are validation results; the official test set remains sealed."
}
finally {
    Pop-Location
}

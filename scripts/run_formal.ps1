param(
    [string]$Python = "python",
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto",
    [ValidateRange(1, 1000)][int]$Epochs = 50,
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateRange(0.00000001, 1.0)][double]$LearningRate = 0.001,
    [string]$OutputRoot = "runs/formal",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$overwriteArgs = @()
if ($Overwrite) { $overwriteArgs = @("--overwrite") }
$learningRateText = $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture)
Push-Location -LiteralPath $projectRoot
try {
    & $Python -m har_compare.prepare --data-dir data
    if ($LASTEXITCODE -ne 0) { throw "Data preparation failed (exit $LASTEXITCODE)." }
    foreach ($model in @("cnn", "rnn", "lstm")) {
        foreach ($seed in $Seeds) {
            $runOutput = Join-Path (Join-Path $OutputRoot $model) ("seed_" + $seed)
            & $Python -m har_compare.train --model $model --data-dir data/processed --output-dir $runOutput --epochs $Epochs --seed $seed --batch-size 64 --device $Device --lr $learningRateText --patience 8 --stage formal @overwriteArgs
            if ($LASTEXITCODE -ne 0) { throw "$model seed $seed training failed (exit $LASTEXITCODE)." }
        }
    }
    Write-Host "Completed the configured train/validation runs. Freeze model settings before explicit official-test evaluation."
}
finally {
    Pop-Location
}

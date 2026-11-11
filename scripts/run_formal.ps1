param(
    [ValidateNotNullOrEmpty()][string]$Python = "python",
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto",
    [ValidateNotNullOrEmpty()][string]$RunRoot = "runs/formal_v1",
    [ValidateNotNullOrEmpty()][string]$ReportOutput = "reports/reproduced_formal",
    [switch]$IncludeNoise,
    [ValidateNotNullOrEmpty()][string]$NoiseReportOutput = "reports/reproduced_noise"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-ExperimentStep {
    param([string]$Label, [string[]]$Arguments)
    Write-Host $Label
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed (exit $LASTEXITCODE). Later steps were not run."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    # A frozen experiment stays immutable. A full reproduction needs a new root.
    if ((Test-Path -LiteralPath (Join-Path $RunRoot "freeze.json")) -or
        (Test-Path -LiteralPath (Join-Path $RunRoot "test"))) {
        throw "This experiment is already frozen. Choose a new -RunRoot for a full reproduction."
    }
    if (Test-Path -LiteralPath $ReportOutput) {
        throw "The report folder already exists. Choose a new -ReportOutput."
    }
    if ($IncludeNoise) {
        if (Test-Path -LiteralPath $NoiseReportOutput) {
            throw "The noise report folder already exists. Choose a new -NoiseReportOutput."
        }
        if ([System.IO.Path]::GetFullPath($ReportOutput) -eq
            [System.IO.Path]::GetFullPath($NoiseReportOutput)) {
            throw "-ReportOutput and -NoiseReportOutput must be different folders."
        }
    }
    if ($Device -eq "auto") {
        $detectedDevice = & $Python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
        if ($LASTEXITCODE -ne 0) { throw "Could not select a PyTorch device." }
        $Device = ($detectedDevice | Out-String).Trim()
        if ($Device -notin @("cuda", "cpu")) { throw "Unexpected device selection: $Device" }
    }

    Invoke-ExperimentStep -Label "Preparing official data" -Arguments @(
        "-m", "har_compare.prepare", "--data-dir", "data")
    Invoke-ExperimentStep -Label "Selecting learning rates on validation data" -Arguments @(
        "-m", "har_compare.formal_train", "tune", "--root", $RunRoot, "--device", $Device)
    Invoke-ExperimentStep -Label "Training the nine formal models" -Arguments @(
        "-m", "har_compare.formal_train", "train", "--root", $RunRoot, "--device", $Device)
    Invoke-ExperimentStep -Label "Freezing all nine checkpoints" -Arguments @(
        "-m", "har_compare.formal_test", "freeze", "--root", $RunRoot)
    Invoke-ExperimentStep -Label "Evaluating the official test set" -Arguments @(
        "-m", "har_compare.formal_test", "evaluate", "--root", $RunRoot, "--device", $Device)
    Invoke-ExperimentStep -Label "Exporting the formal comparison" -Arguments @(
        "-m", "har_compare.formal_report", "--root", $RunRoot, "--output", $ReportOutput)
    Invoke-ExperimentStep -Label "Analyzing the frozen models by test subject" -Arguments @(
        "-m", "har_compare.subject_eval", "--root", $RunRoot, "--device", $Device, "--output", $ReportOutput)

    if ($IncludeNoise) {
        Invoke-ExperimentStep -Label "Fixing the noise protocol" -Arguments @(
            "-m", "har_compare.noise_eval", "plan", "--root", $RunRoot)
        Invoke-ExperimentStep -Label "Evaluating the same noise inputs for all models" -Arguments @(
            "-m", "har_compare.noise_eval", "evaluate", "--root", $RunRoot, "--device", $Device)
        Invoke-ExperimentStep -Label "Exporting the paired noise comparison" -Arguments @(
            "-m", "har_compare.noise_report", "--input", (Join-Path $RunRoot "noise_v1/results.json"),
            "--output", $NoiseReportOutput)
    }
    Write-Host "Completed the fixed comparison protocol. Results: $ReportOutput"
}
finally {
    Pop-Location
}

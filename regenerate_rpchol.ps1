$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Driver = Join-Path $ProjectRoot "experiments\exp_rpcholesky_embeddings.py"
$Embeddings = "datasets/pcam_train_embedding.npy"
$Labels = "datasets/pcam_train_label.npy"

Set-Location $ProjectRoot

function Invoke-RPCholRun {
    param(
        [string]$Name,
        [string[]]$ExtraArgs
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Regenerating $Name"
    Write-Host "============================================================"

    $BaseArgs = @(
        $Driver,
        "--embeddings", $Embeddings,
        "--labels", $Labels,
        "--kernel", "gaussian",
        "--bandwidth", "approx_median",
        "--block-size", "10",
        "--stoptol", "1e-13",
        "--seed", "42"
    )

    & python @BaseArgs @ExtraArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Completed $Name" -ForegroundColor Green
    } else {
        Write-Host "FAILED $Name with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

Invoke-RPCholRun "data/rpchol_smoke" @(
    "--subsample", "5000",
    "--k-values", "50", "100",
    "--output-dir", "data/rpchol_smoke"
)

Invoke-RPCholRun "data/rpchol_20k" @(
    "--subsample", "20000",
    "--k-values", "200",
    "--output-dir", "data/rpchol_20k"
)

Invoke-RPCholRun "data/rpchol_50k" @(
    "--subsample", "50000",
    "--k-values", "200",
    "--output-dir", "data/rpchol_50k"
)

Invoke-RPCholRun "data/rpchol_full" @(
    "--k-values", "200",
    "--output-dir", "data/rpchol_full"
)

<#
Simple environment setup helper for Windows PowerShell.
Attempts to use the `py` launcher to create a venv with a supported Python
(3.11, 3.10, 3.9). If a version is found it creates `.venv` and installs
requirements from `requirements.txt`.

Usage:
  .\setup_env.ps1          # auto-select a supported Python
  .\setup_env.ps1 3.10    # force a specific minor version
#>
param(
    [string]$PythonVersion = ""
)

function Create-Venv($pyCmd) {
    Write-Host "Using: $pyCmd"
    & $pyCmd -m venv .venv 2>$null
    return $LASTEXITCODE -eq 0
}

$created = $false
if ($PythonVersion) {
    $launcher = "py -$PythonVersion"
    if (Create-Venv $launcher) { $created = $true }
} else {
    $candidates = @("py -3.11", "py -3.10", "py -3.9")
    foreach ($c in $candidates) {
        & $c -V > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            if (Create-Venv $c) { $created = $true; break }
        }
    }
}

if (-not $created) {
    Write-Error "No supported Python found (3.9, 3.10, 3.11). Install one or run manually with a compatible interpreter."
    exit 1
}

Write-Host "Upgrading pip and installing requirements into .venv..."
.
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Done. Activate the environment in PowerShell with:`n  . \.venv\Scripts\Activate.ps1"

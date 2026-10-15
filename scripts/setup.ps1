param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$binDir = Join-Path $ProjectRoot 'bin'
if (Test-Path $binDir) { $env:PATH = "$binDir;$env:PATH" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Install uv with: powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}

if (-not (Test-Path ".venv")) {
    uv venv
}
if ($CpuOnly) {
    uv pip install torch torchvision
} else {
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
}
uv pip install -e ".[birefnet]"
Write-Host "Setup complete. Launch with: Run Matte App.bat"

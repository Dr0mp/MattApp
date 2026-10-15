$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$binDir = Join-Path $ProjectRoot 'bin'
if (Test-Path $binDir) { $env:PATH = "$binDir;$env:PATH" }

uv python install 3.10
if (-not (Test-Path ".sdmatte-venv")) {
    uv venv --python 3.10 .sdmatte-venv
}

$py = ".\.sdmatte-venv\Scripts\python.exe"
uv pip install --python $py torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python $py tensorboard timm==0.5.4 opencv-python==4.5.4.60 setuptools==58.2.0 easydict scikit-image==0.24.0 fairscale==0.4.13 diffusers==0.30.3 transformers==4.45.1 numpy==1.24.1 Pillow==9.2.0 huggingface-hub==0.25.1 accelerate==0.34.2 cached_property==2.0.1 safetensors omegaconf

New-Item -ItemType Directory -Force -Path 'third_party' | Out-Null
if (-not (Test-Path 'third_party/SDMatte')) {
    git clone https://github.com/vivoCameraResearch/SDMatte.git third_party/SDMatte
    git -C third_party/SDMatte checkout c3ea6949f270d253539a2bda2ad290e127934de9
}

& $py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='LongfeiHuang/LiteSDMatte', local_dir='models/LiteSDMatte')"
Write-Host "LiteSDMatte is ready. Launch Matte App with: Run Matte App.bat"

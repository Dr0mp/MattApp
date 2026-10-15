param([switch]$Sam3, [switch]$VideoMaMa, [switch]$ViTMatte, [switch]$Sam2Matting, [switch]$Sam31, [switch]$MatAnyone2)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$binDir = Join-Path $ProjectRoot 'bin'
if (Test-Path $binDir) { $env:PATH = "$binDir;$env:PATH" }

function CheckedRun {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup failed: $Program" }
}
if (-not (Test-Path -LiteralPath '.roto-venv/Scripts/python.exe')) {
    CheckedRun uv @('venv', '--python', '3.12', '.roto-venv')
}
$rotoPython = Join-Path $ProjectRoot '.roto-venv/Scripts/python.exe'
CheckedRun uv @('pip', 'install', '--python', $rotoPython, 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cu126')
CheckedRun uv @('pip', 'install', '--python', $rotoPython, 'numpy==1.26.4', 'transformers==5.17.0', 'diffusers==0.41.0', 'accelerate', 'safetensors', 'hydra-core', 'omegaconf', 'opencv-python==4.10.0.84', 'imageio', 'scipy', 'timm', 'einops', 'huggingface-hub', 'imageio-ffmpeg')
$repositories = @(
    @('MatAnyone','https://github.com/pq-yang/MatAnyone.git','e5ddc534c1fff9bb9e54cf476095d29071b7cb4f'),
    @('VideoMaMa','https://github.com/cvlab-kaist/VideoMaMa.git','d5cce3e0ffe3b6429c147e658bb28bcfb576374c')
)
if ($MatAnyone2) { CheckedRun uv @('pip','install','--python',$rotoPython,'easydict'); $repositories += ,@('MatAnyone2','https://github.com/pq-yang/MatAnyone2.git','0079197acd6d16a741f71558809c06c586c579e0') }
New-Item -ItemType Directory -Force -Path 'third_party' | Out-Null
if ($Sam2Matting -or $Sam31) {
    CheckedRun uv @('pip','install','--python',$rotoPython,'iopath','fvcore','tensordict','decord','ftfy','loguru','submitit')
}
if ($Sam31) { CheckedRun uv @('pip','install','--python',$rotoPython,'pycocotools','triton-windows>=3.3,<3.4') }
if ($Sam2Matting -and -not (Test-Path -LiteralPath 'third_party/SAM2Matting')) { CheckedRun git @('clone','https://github.com/FudanCVL/SAM2Matting.git','third_party/SAM2Matting'); CheckedRun git @('-C','third_party/SAM2Matting','checkout','73dd721d77b56749248aefe5e8824d7f61b9d13c') }
if ($Sam31 -and -not (Test-Path -LiteralPath 'third_party/sam3')) { CheckedRun git @('clone','https://github.com/facebookresearch/sam3.git','third_party/sam3'); CheckedRun git @('-C','third_party/sam3','checkout','0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc') }
foreach ($repo in $repositories) {
    $repoPath = Join-Path 'third_party' $repo[0]
    if (-not (Test-Path -LiteralPath $repoPath)) {
        CheckedRun git @('clone', $repo[1], $repoPath)
        CheckedRun git @('-C', $repoPath, 'checkout', $repo[2])
    }
}
$modelDownloader = Join-Path $ProjectRoot 'matte_app/download_roto.py'
$downloadArgs = @($modelDownloader)
if ($MatAnyone2) { $downloadArgs += '--matanyone2' }
if ($Sam3) { $downloadArgs += '--sam3' }
if ($VideoMaMa) { $downloadArgs += '--videomama' }
if ($ViTMatte) { $downloadArgs += '--vitmatte' }
if ($Sam2Matting) { $downloadArgs += '--sam2matting' }
if ($Sam31) { $downloadArgs += '--sam31' }
CheckedRun $rotoPython $downloadArgs
Write-Host 'Roto setup complete. Restart Matte App and select a SAM workflow.'

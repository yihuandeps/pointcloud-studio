# 点云工具 · 启动高精度模式后端
#
# 用法：  .\start.ps1
# 前端在 http://127.0.0.1:5173 通过 /api 代理访问本服务。

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "还没装依赖，先跑 .\setup.ps1" -ForegroundColor Red
    exit 1
}

# 关键：直连 huggingface.co 在本机会超时，模型权重必须走镜像
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }

# 模型权重约 700MB，默认会落在系统盘的 ~/.cache/huggingface。
# 系统盘紧张时会装不下，统一放到项目盘。
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $root ".cache\huggingface" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

# 可选：换模型变体。vitl=326M（默认），vitb=104M 更快，vits=35M 最轻
if (-not $env:MOGE_MODEL) { $env:MOGE_MODEL = "Ruicheng/moge-2-vitl" }

# 生成式 3D（TripoSR）：rembg 的抠图模型默认下到系统盘 ~/.u2net，
# 和 HF 缓存一样统一挪到项目盘
if (-not $env:U2NET_HOME) { $env:U2NET_HOME = Join-Path $root ".cache\u2net" }
New-Item -ItemType Directory -Force -Path $env:U2NET_HOME | Out-Null
if (-not $env:TRIPO_MODEL) { $env:TRIPO_MODEL = "stabilityai/TripoSR" }

# HF 缓存默认用符号链接，Windows 非管理员/未开开发者模式时会 WinError 1314
if (-not $env:HF_HUB_DISABLE_SYMLINKS) { $env:HF_HUB_DISABLE_SYMLINKS = "1" }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# 8GB 显存跑 marching cubes 容易被碎片拖垮，允许分配器扩展已有段
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

Write-Host "HF 镜像 : $env:HF_ENDPOINT"
Write-Host "模型    : $env:MOGE_MODEL（高精度） / $env:TRIPO_MODEL（生成式 3D）"
Write-Host "监听    : http://127.0.0.1:8000"
Write-Host "首次启动会下载模型权重（vitl 约 1.2GB），请耐心等待" -ForegroundColor Yellow
Write-Host "网络慢可以先用小模型试通：`$env:MOGE_MODEL='Ruicheng/moge-2-vits-normal'（约 140MB）`n" -ForegroundColor DarkGray

Set-Location $root
& $vpy -m uvicorn app:app --host 127.0.0.1 --port 8000

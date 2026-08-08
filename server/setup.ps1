# 点云工具 · 后端依赖安装
#
# 用法：  .\setup.ps1
# 重跑安全：已装好的会跳过。
#
# 总下载量约 3GB（torch CUDA 2.5GB + MoGe 依赖），首次跑请留足时间。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$vpy = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "找不到虚拟环境，正在创建..." -ForegroundColor Yellow
    $py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (-not (Test-Path $py311)) {
        Write-Host "没有 Python 3.11。先装：winget install --id Python.Python.3.11 --scope user" -ForegroundColor Red
        exit 1
    }
    & $py311 -m venv (Join-Path $root ".venv")
    & $vpy -m pip install --upgrade pip
}

# 国内网络：PyPI 走清华镜像会快很多；torch 必须走官方源拿 CUDA 版本
$PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
$TORCH_INDEX = "https://download.pytorch.org/whl/cu121"

# pip 默认把下载和解包放在系统盘的 TEMP，torch 那个 wheel 有 2.4GB，
# 系统盘不宽裕时会直接 [Errno 28] No space left on device。
# 统一挪到项目所在盘，顺便让缓存也留在这边，重装时不用再下一遍。
$cache = Join-Path $root ".cache"
New-Item -ItemType Directory -Force -Path $cache, (Join-Path $cache "tmp") | Out-Null
$env:TMP = Join-Path $cache "tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PIP_CACHE_DIR = Join-Path $cache "pip"

$drive = (Get-Item $root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "安装盘 $($drive.Name): 剩余 $freeGB GB（峰值约需 8GB）" -ForegroundColor DarkGray
if ($freeGB -lt 10) {
    Write-Host "空间可能不够，建议先腾出至少 10GB" -ForegroundColor Yellow
}

Write-Host "`n[1/3] PyTorch CUDA 12.1（约 2.5GB，慢是正常的）" -ForegroundColor Cyan
& $vpy -m pip install torch==2.5.1 torchvision --index-url $TORCH_INDEX
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch 安装失败" -ForegroundColor Red; exit 1 }

Write-Host "`n[2/3] MoGe" -ForegroundColor Cyan
# 先试官方的 git+ 装法；国内网络下 github.com 常常连不上，失败就走源码回退。
& $vpy -m pip install -i $PYPI --extra-index-url https://pypi.org/simple "git+https://github.com/microsoft/MoGe.git" 2>&1 | Out-Host

if ($LASTEXITCODE -ne 0) {
    Write-Host "git+ 装法失败（多半是 github.com 直连不通），改用源码回退" -ForegroundColor Yellow
    # fetch_sources.py 只用标准库：codeload 抓小仓库 tarball，
    # MoGe 走 api + raw 只取包本身（354KB，而完整 tarball 有 10.5MB 且不支持续传）
    & $vpy (Join-Path $root "fetch_sources.py") (Join-Path $cache "src")
    if ($LASTEXITCODE -ne 0) { Write-Host "源码抓取失败" -ForegroundColor Red; exit 1 }

    & $vpy -m pip install -i $PYPI (Join-Path $cache "src\utils3d.tar.gz") (Join-Path $cache "src\pipeline.tar.gz")
    if ($LASTEXITCODE -ne 0) { Write-Host "utils3d / pipeline 安装失败" -ForegroundColor Red; exit 1 }

    # --no-deps：pyproject 里那两个依赖写的是 git+ 直连地址，上一步已装好，别让 pip 再去拉一次
    & $vpy -m pip install --no-deps (Join-Path $cache "src\moge-src")
    if ($LASTEXITCODE -ne 0) { Write-Host "MoGe 安装失败" -ForegroundColor Red; exit 1 }

    Write-Host "补齐 MoGe 运行时依赖" -ForegroundColor Cyan
    & $vpy -m pip install -i $PYPI huggingface_hub opencv-python scipy trimesh click pillow
    if ($LASTEXITCODE -ne 0) { Write-Host "依赖补齐失败" -ForegroundColor Red; exit 1 }
}

Write-Host "`n[3/4] 生成式 3D（TripoSR，源码已 vendor 在 server/tsr/）" -ForegroundColor Cyan
# 上游 requirements 里的 torchmcubes 是 git+ 源码依赖且要编译 C++，Windows 必坑，
# vendored 的 tsr 已改用 scikit-image 的 marching_cubes（见 tsr/models/isosurface.py）。
# transformers 固定 4.46.x：4.35（上游锁定）的 tokenizers 会把 huggingface_hub
# 拽回 0.17，直接弄坏 MoGe 的权重下载。
& $vpy -m pip install -i $PYPI omegaconf einops "transformers==4.46.3" trimesh rembg onnxruntime scikit-image imageio
if ($LASTEXITCODE -ne 0) { Write-Host "TripoSR 依赖安装失败" -ForegroundColor Red; exit 1 }

Write-Host "`n[4/5] 多视图生成式（Hunyuan3D-2，仅形状管线）" -ForegroundColor Cyan
# transformers 卡在 4.x：hy3dgen 要 >=4.48，但 5.x 重构了 ViT 的参数命名
# （layers.0.attention.q_proj vs 旧的 encoder.layer.0.attention.attention.query），
# TripoSR 的 checkpoint 会加载失败。4.48–4.x 两边都满足。
& $vpy -m pip install -i $PYPI diffusers accelerate pymeshlab pygltflib "transformers>=4.48,<5"
if ($LASTEXITCODE -ne 0) { Write-Host "Hunyuan3D 依赖安装失败" -ForegroundColor Red; exit 1 }

# 源码放项目盘。--no-deps 跳过 gradio / xatlas / ninja 等只有纹理管线和 demo 才用的重依赖；
# 纹理管线要编译 CUDA 扩展，Windows 上装不上，本项目也用不到（颜色由投影采样自己做）
$hySrc = Join-Path $cache "src\Hunyuan3D-2"
if (-not (Test-Path (Join-Path $hySrc "setup.py"))) {
    Write-Host "拉取 Hunyuan3D-2 源码..." -ForegroundColor DarkGray
    New-Item -ItemType Directory -Force -Path (Join-Path $cache "src") | Out-Null
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git $hySrc
    if ($LASTEXITCODE -ne 0) {
        Write-Host "源码拉取失败（github.com 直连不通？挂代理再试）" -ForegroundColor Yellow
    }
}
if (Test-Path (Join-Path $hySrc "setup.py")) {
    & $vpy -m pip install --no-deps $hySrc
    if ($LASTEXITCODE -ne 0) { Write-Host "hy3dgen 安装失败" -ForegroundColor Red; exit 1 }
}

Write-Host "`n[5/5] Web 服务" -ForegroundColor Cyan
# httpx 是 FastAPI TestClient 的依赖，test_backend.py 要用
& $vpy -m pip install -i $PYPI fastapi "uvicorn[standard]" python-multipart httpx
if ($LASTEXITCODE -ne 0) { Write-Host "FastAPI 安装失败" -ForegroundColor Red; exit 1 }

Write-Host "`n---- 自检 ----" -ForegroundColor Cyan
# 用字面量 here-string（@'...'@），双引号版会把 Python 里的 $ 当变量插值
& $vpy -c @'
import torch
print(f"torch      {torch.__version__}")
print(f"CUDA       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU        {p.name}  {p.total_memory/1024**3:.1f} GB")
import moge; print("moge       ok")
import fastapi, uvicorn, multipart; print("fastapi    ok")
'@

Write-Host "`n安装完成。启动服务：.\start.ps1" -ForegroundColor Green

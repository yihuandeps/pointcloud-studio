"""
生成式 3D 端到端验证：TripoSR 加载 → 单图生成 → 立体性检验 → HTTP 接口

用法：.venv\\Scripts\\python.exe test_gen3d.py

与 test_backend.py（MoGe）分开跑：两个模型都要占显存，
分开跑能在 8GB 卡上各自留足余量。
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("U2NET_HOME", str(ROOT / ".cache" / "u2net"))
# Windows 非管理员且没开开发者模式时，HF 缓存建符号链接会 WinError 1314
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

PASS = 0
FAIL = 0


def ok(name: str, cond: bool, extra: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}" + (f"  {extra}" if extra else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}  {extra}")


def synth_image(w: int = 512, h: int = 512) -> bytes:
    """白底 + 中央一个带朗伯着色的橙色球 —— 对 rembg 和 TripoSR 都友好的输入。"""
    import io

    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.full((h, w), 245.0)
    g = np.full((h, w), 245.0)
    b = np.full((h, w), 245.0)

    dx = (xx - w * 0.5) / (w * 0.3)
    dy = (yy - h * 0.5) / (w * 0.3)
    d2 = dx * dx + dy * dy
    inside = d2 < 1.0
    sh = np.sqrt(np.clip(1.0 - d2, 0, None))
    lit = np.clip(0.3 + 0.8 * sh - 0.25 * dx, 0, None)
    r = np.where(inside, 235 * lit, r)
    g = np.where(inside, 120 * lit, g)
    b = np.where(inside, 40 * lit, b)

    rgb = np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


print("\n[1] 运行环境")
try:
    import numpy as np
    import torch

    ok("torch 可导入", True, torch.__version__)
    cuda = torch.cuda.is_available()
    ok("CUDA 可用", cuda, "回落 CPU 会非常慢" if not cuda else "")
    from skimage.measure import marching_cubes  # noqa: F401

    ok("skimage marching_cubes 可用（torchmcubes 的替代）", True)
    import rembg  # noqa: F401

    ok("rembg 可导入", True)
except Exception as exc:
    ok("依赖完整", False, str(exc))
    print("\n依赖没装好，先跑 .\\setup.ps1\n")
    sys.exit(1)

print("\n[2] 加载 TripoSR")
from tripo_runner import TripoRunner

runner = TripoRunner()
try:
    runner.load()
    ok("模型加载成功", True,
       f"{runner.model_id} → {runner.device}  {runner.load_seconds}s")
except Exception as exc:
    ok("模型加载成功", False, str(exc))
    print("\n权重下载失败？确认 server/.cache/huggingface 里有 stabilityai/TripoSR。\n")
    sys.exit(1)

print("\n[3] 单图生成（含背面补全）")
png = synth_image()
result = runner.generate(png, n_points=200_000)

pts = result["positions"]
cols = result["colors"]
n = result["count"]

ok("点数符合请求", n == 200_000, f"{n:,}")
ok("坐标形状 (N,3) float32", pts.shape == (n, 3) and pts.dtype == np.float32,
   f"{pts.shape} {pts.dtype}")
ok("颜色形状 (N,3) uint8", cols.shape == (n, 3) and cols.dtype == np.uint8,
   f"{cols.shape} {cols.dtype}")
ok("坐标无 NaN/Inf", bool(np.isfinite(pts).all()))

# ---- 立体性：这是整个模式存在的意义 ----
# 单图深度模式的点云是一层"皮"，最薄的轴向厚度接近 0；
# 生成式模式必须产出有体积的形体，三个轴向的跨度应当同量级。
ext = pts.max(axis=0) - pts.min(axis=0)
ratio = float(ext.min() / ext.max())
ok("形体有真实体积（最薄/最厚轴向 > 0.4，浮雕会接近 0）", ratio > 0.4,
   f"跨度 x={ext[0]:.2f} y={ext[1]:.2f} z={ext[2]:.2f}，比值 {ratio:.2f}")

# ---- 背面补全：+Z 朝向初始视角，Z < 中心的点就是"照片里看不见的背面" ----
z_mid = float((pts[:, 2].max() + pts[:, 2].min()) / 2)
back_frac = float((pts[:, 2] < z_mid).mean())
ok("背面确实有点（占比 > 25%，纯正面浮雕接近 0）", back_frac > 0.25,
   f"背面点占 {back_frac * 100:.1f}%")

# ---- 颜色采样有效：球是橙色的，红通道均值应明显高于蓝通道 ----
r_mean, b_mean = float(cols[:, 0].mean()), float(cols[:, 2].mean())
ok("颜色来自原图（橙色球 → R 通道 > B 通道）", r_mean > b_mean + 20,
   f"R={r_mean:.0f} B={b_mean:.0f}")

print(f"     生成耗时 {result['ms']} ms")

print("\n[4] HTTP 接口")
try:
    from fastapi.testclient import TestClient

    import app as app_module

    app_module.gen_runner = runner  # 复用已加载的模型
    client = TestClient(app_module.app)

    r = client.get("/api/gen/health")
    ok("GET /api/gen/health 返回 200", r.status_code == 200, str(r.status_code))
    info = r.json()
    ok("health 报告已加载", info.get("ok") and info.get("loaded"),
       f"model={info.get('model')} kind={info.get('kind')}")

    r = client.post(
        "/api/generate",
        files={"image": ("input.png", png, "image/png")},
        data={"points": "50000"},
    )
    ok("POST /api/generate 返回 200", r.status_code == 200,
       str(r.status_code) + ("  " + r.text[:200] if r.status_code != 200 else ""))

    blob = r.content
    head_len = struct.unpack("<I", blob[:4])[0]
    head = json.loads(blob[4:4 + head_len].decode("utf-8"))
    n = head["count"]
    expect = 4 + head_len + n * 3 * 4 + n * 3          # 头部 + XYZ + RGB
    if head.get("hasNormals"):
        expect += n * 3                                # int8 法线
    if head.get("hasAO"):
        expect += n                                    # uint8 凹陷度

    ok("响应体长度与头部声明相符", len(blob) == expect,
       f"实际 {len(blob)}，期望 {expect}")
    ok("带回了法线与凹陷度（光照要用）",
       head.get("hasNormals") and head.get("hasAO"),
       f"hasNormals={head.get('hasNormals')} hasAO={head.get('hasAO')}")
    ok("头部字段齐全", all(k in head for k in ("count", "model", "device", "ms")),
       ", ".join(head.keys()))
    ok("点数遵从请求参数", head["count"] == 50000, str(head["count"]))

except ImportError as exc:
    print(f"  · 跳过 HTTP 测试（缺 {exc.name}，pip install httpx 后可测）")

print("\n" + "─" * 54)
print(f"  通过 {PASS} 项，失败 {FAIL} 项")
print("─" * 54 + "\n")
sys.exit(1 if FAIL else 0)

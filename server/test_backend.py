"""
后端端到端验证：torch/CUDA → MoGe 推理 → HTTP 接口 → 二进制打包

用法：.venv\\Scripts\\python.exe test_backend.py
产物：.cache/test_payload.bin（供 web/tests 里的 Node 端继续校验）
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 权重走镜像，缓存放项目盘（系统盘通常不宽裕）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
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


def synth_image(w: int = 512, h: int = 384) -> bytes:
    """
    合成一张渐变背景 + 中央带朗伯着色的球。

    注意合成图的固有局限：MoGe 自带有效性 mask，会把这种人造背景整片判为
    无效（实测有效像素恰好≈球的面积，背景只剩一百多个点），加纹理也救不回来 ——
    它就不认这是真实场景。所以下面的几何断言只在球内部做，不去比较球和背景。
    """
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    r = 30 + (yy / h) * 90.0
    g = 40 + (yy / h) * 80.0
    b = 70 + (yy / h) * 60.0

    dx = (xx - w * 0.5) / (w * 0.22)
    dy = (yy - h * 0.55) / (w * 0.22)
    d2 = dx * dx + dy * dy
    inside = d2 < 1.0
    sh = np.sqrt(np.clip(1.0 - d2, 0, None))
    lit = np.clip(0.25 + 0.85 * sh - 0.3 * dx + 0.25 * dy, 0, None)
    r = np.where(inside, 240 * lit, r)
    g = np.where(inside, 150 * lit, g)
    b = np.where(inside, 90 * lit, b)

    rgb = np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue(), inside


print("\n[1] 运行环境")
try:
    import numpy as np
    import torch

    ok("torch 可导入", True, torch.__version__)
    cuda = torch.cuda.is_available()
    ok("CUDA 可用", cuda, "回落 CPU 会非常慢" if not cuda else "")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        ok("识别到 GPU", True, f"{p.name}  {p.total_memory / 1024**3:.1f} GB")
except Exception as exc:
    ok("torch 可导入", False, str(exc))
    print("\n依赖没装好，先跑 .\\setup.ps1\n")
    sys.exit(1)

print("\n[2] 纯函数（不依赖模型）")
from imaging import fit_size, js_round, pack_result

ok("js_round(.5) 向上取（不是 Python 的银行家舍入）",
   js_round(0.5) == 1 and js_round(2.5) == 3 and round(2.5) == 2,
   f"js_round(2.5)={js_round(2.5)}  内置 round(2.5)={round(2.5)}")
ok("fit_size 不放大小图", fit_size(800, 600, 1024) == (800, 600))
ok("fit_size 按长边等比缩放", fit_size(4032, 3024, 1024) == (1024, 768),
   str(fit_size(4032, 3024, 1024)))

print("\n[3] 加载 MoGe")
from moge_runner import MoGeRunner

runner = MoGeRunner()
try:
    runner.load()
    ok("模型加载成功", True, f"{runner.model_id} → {runner.device}")
except Exception as exc:
    ok("模型加载成功", False, str(exc))
    print("\n模型下载失败？确认 HF_ENDPOINT 指向可用镜像。\n")
    sys.exit(1)

print("\n[4] 推理输出")
png, inside_mask = synth_image()
result = runner.infer(png, max_side=1024)

W, H = result["width"], result["height"]
pts = result["points"]
msk = result["mask"]

ok("输出尺寸与输入一致", (W, H) == (512, 384), f"{W}×{H}")
ok("点图形状为 (H, W, 3)", pts.shape == (H, W, 3), str(pts.shape))
ok("点图为 float32", pts.dtype == np.float32, str(pts.dtype))
ok("mask 形状为 (H, W)", msk.shape == (H, W), str(msk.shape))
ok("点图无 NaN/Inf", bool(np.isfinite(pts).all()))

valid = msk.astype(bool)
cover = valid.mean()
# 合成图上 mask 只会保住球本身（约占画面 20%），背景整片被判无效 —— 见 synth_image 的说明。
# 这里只确认 mask 不是全 0 也不是全 1（那两种才说明模型或解包出了问题）。
ok("mask 有区分度（不是全有效/全无效）", 0.05 < cover < 0.999, f"{cover * 100:.1f}%")

intr = result["intrinsics"]
ok("返回 3×3 相机内参", intr is not None and len(intr) == 3 and len(intr[0]) == 3,
   json.dumps(intr) if intr else "None")
if intr:
    ok("焦距为正", intr[0][0] > 0 and intr[1][1] > 0,
       f"fx={intr[0][0]:.4f} fy={intr[1][1]:.4f}")

# OpenCV 坐标系：Z 沿视线向前，越近 Z 越小。
# 背景在合成图上拿不到有效点，所以几何检验全部放在球内部做 ——
# 球心比球缘离相机更近，这是凸面体的硬性质，模型答对了才说明它真的重建了曲面。
z = pts[:, :, 2]
sphere = valid & inside_mask
ok("球体区域样本充足", sphere.sum() > 5000, f"{sphere.sum()} 点")

if sphere.sum() > 5000:
    yy, xx = np.mgrid[0:H, 0:W]
    rad = np.sqrt(((xx - W * 0.5) / (W * 0.22)) ** 2
                  + ((yy - H * 0.55) / (W * 0.22)) ** 2)
    core = sphere & (rad < 0.35)      # 球心附近
    rim = sphere & (rad > 0.80)       # 球缘一圈
    zc, zr = float(z[core].mean()), float(z[rim].mean())
    ok("球心比球缘更近（Z 更小）—— 模型确实重建出了曲面", zc < zr,
       f"球心 Z={zc:.3f}m，球缘 Z={zr:.3f}m，凸起 {(zr - zc) * 100:.1f}cm")

    zs = float(z[sphere].mean())
    ok("输出是米制量级（不是 0–1 归一化值）", 0.05 < zs < 500,
       f"球面平均距离 {zs:.3f} m")

print(f"     推理耗时 {result['ms']} ms")

print("\n[5] HTTP 接口")
try:
    from fastapi.testclient import TestClient
    import app as app_module

    app_module.runner = runner  # 复用已加载的模型，别再加载一遍
    client = TestClient(app_module.app)

    r = client.get("/api/health")
    ok("GET /api/health 返回 200", r.status_code == 200, str(r.status_code))
    info = r.json()
    ok("health 报告已加载", info.get("ok") and info.get("loaded"),
       f"model={info.get('model')} device={info.get('device')}")

    r = client.post(
        "/api/infer",
        files={"image": ("input.png", png, "image/png")},
        data={"max_side": "1024"},
    )
    ok("POST /api/infer 返回 200", r.status_code == 200,
       str(r.status_code) + ("  " + r.text[:200] if r.status_code != 200 else ""))

    blob = r.content
    head_len = struct.unpack("<I", blob[:4])[0]
    head = json.loads(blob[4:4 + head_len].decode("utf-8"))
    body_start = 4 + head_len
    n = head["width"] * head["height"]
    expect = body_start + n * 3 * 4 + n

    ok("响应体长度与头部声明相符", len(blob) == expect,
       f"实际 {len(blob)}，期望 {expect}")
    ok("头部字段齐全",
       all(k in head for k in ("width", "height", "hasMask", "intrinsics", "model", "ms")),
       ", ".join(head.keys()))

    out = ROOT / ".cache" / "test_payload.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"     响应已保存：{out}  ({len(blob) / 1048576:.2f} MB)")

except ImportError as exc:
    print(f"  · 跳过 HTTP 测试（缺 {exc.name}，pip install httpx 后可测）")

print("\n" + "─" * 54)
print(f"  通过 {PASS} 项，失败 {FAIL} 项")
print("─" * 54 + "\n")
sys.exit(1 if FAIL else 0)

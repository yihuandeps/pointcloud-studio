"""
多视图生成式 3D 端到端验证：Hunyuan3D-2mv → 投影着色 → 点云 → HTTP

用法：.venv\\Scripts\\python.exe test_mv.py

用官方样例的 front/left/back 三视图（正好是"人物三视图"这个用法）。
样例图在 .cache/src/Hunyuan3D-2/assets/example_mv_images/1/，
是 setup 时克隆源码带下来的；没有就跳过。
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


EX = ROOT / ".cache" / "src" / "Hunyuan3D-2" / "assets" / "example_mv_images" / "1"
if not (EX / "front.png").exists():
    print(f"\n找不到样例三视图（{EX}），跳过。\n重新跑 setup.ps1 会把源码和样例一起拉下来。\n")
    sys.exit(0)

print("\n[1] 运行环境")
try:
    import numpy as np
    import torch

    ok("torch 可导入", True, torch.__version__)
    ok("CUDA 可用", torch.cuda.is_available())
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401

    ok("hy3dgen 形状管线可导入", True)
except Exception as exc:
    ok("依赖完整", False, str(exc))
    print("\n依赖没装好，先跑 .\\setup.ps1\n")
    sys.exit(1)

print("\n[2] 加载 Hunyuan3D-2mv")
from hunyuan_runner import VIEW_DIRS, HunyuanRunner

runner = HunyuanRunner()
ok("权重已缓存", runner.weights_cached(), "约 4.9GB")
try:
    runner.load()
    ok("模型加载成功", True, f"{runner.subfolder} → {runner.device}  {runner.load_seconds}s")
except Exception as exc:
    ok("模型加载成功", False, str(exc))
    sys.exit(1)

print("\n[3] 三视图生成（front / left / back）")
images = {k: (EX / f"{k}.png").read_bytes()
          for k in ("front", "left", "back") if (EX / f"{k}.png").exists()}
ok("凑齐三视图", len(images) == 3, " / ".join(images))

result = runner.generate(images, n_points=200_000)
pts = result["positions"]
cols = result["colors"]
n = result["count"]

ok("点数符合请求", n == 200_000, f"{n:,}")
ok("坐标 (N,3) float32", pts.shape == (n, 3) and pts.dtype == np.float32)
ok("颜色 (N,3) uint8", cols.shape == (n, 3) and cols.dtype == np.uint8)
ok("坐标无 NaN/Inf", bool(np.isfinite(pts).all()))
ok("记录了实际使用的视图", set(result["viewsUsed"]) == set(images), str(result["viewsUsed"]))

# ---- 立体性：这个模式存在的意义 ----
ext = pts.max(axis=0) - pts.min(axis=0)
ok("三个轴都有实质厚度（浮雕的最薄轴接近 0）", ext.min() / ext.max() > 0.3,
   f"跨度 x={ext[0]:.2f} y={ext[1]:.2f} z={ext[2]:.2f}，比值 {ext.min()/ext.max():.2f}")

# 正面 = +Z，所以 Z 小于中位面的点就是"背面"
z_mid = float((pts[:, 2].max() + pts[:, 2].min()) / 2)
back_frac = float((pts[:, 2] < z_mid).mean())
ok("背面有实质内容（占比 > 25%）", back_frac > 0.25, f"背面点占 {back_frac * 100:.1f}%")

# ---- 着色：颜色必须真的来自输入图，而不是一片灰 ----
ok("颜色不是单一值（真的采到了图）", int(cols.std(axis=0).mean()) > 8,
   f"各通道标准差 {cols.std(axis=0).round(1)}")
unfilled = np.all(cols == 200, axis=1).mean()
ok("绝大多数点都采到了颜色", unfilled < 0.05, f"未着色 {unfilled * 100:.2f}%")

# 背面的点应当取自 back.png，而不是把正面颜色贴过去 ——
# 若两者颜色分布完全一致，说明视图选择逻辑失效了
front_pts = pts[:, 2] > z_mid
back_pts = ~front_pts
if front_pts.sum() > 1000 and back_pts.sum() > 1000:
    dif = np.abs(cols[front_pts].mean(0) - cols[back_pts].mean(0)).max()
    ok("正/背面配色有差异（视图选择确实生效）", dif > 1.5,
       f"均值差 {dif:.1f}")

print(f"     生成耗时 {result['ms']} ms，体素分辨率 {result['resolution']}")
if torch.cuda.is_available():
    print(f"     显存峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n[4] HTTP 接口")
try:
    from fastapi.testclient import TestClient

    import app as app_module

    app_module.mv_runner = runner
    app_module._RUNNERS["mv"] = ("Hunyuan3D-2mv", runner)
    client = TestClient(app_module.app)

    r = client.get("/api/mv/health")
    ok("GET /api/mv/health 返回 200", r.status_code == 200, str(r.status_code))
    info = r.json()
    ok("health 报告已加载", info.get("ok") and info.get("loaded"),
       f"model={info.get('model')} kind={info.get('kind')}")
    ok("声明支持的视图名", set(info.get("views", [])) == set(VIEW_DIRS),
       str(info.get("views")))

    files = [("front", ("front.png", images["front"], "image/png")),
             ("left", ("left.png", images["left"], "image/png")),
             ("back", ("back.png", images["back"], "image/png"))]
    r = client.post("/api/mv/generate", files=files, data={"points": "50000"})
    ok("POST /api/mv/generate 返回 200", r.status_code == 200,
       str(r.status_code) + ("  " + r.text[:200] if r.status_code != 200 else ""))

    blob = r.content
    head_len = struct.unpack("<I", blob[:4])[0]
    head = json.loads(blob[4:4 + head_len].decode("utf-8"))
    expect = 4 + head_len + head["count"] * 3 * 4 + head["count"] * 3
    ok("响应体长度与头部声明相符", len(blob) == expect,
       f"实际 {len(blob)}，期望 {expect}")
    ok("点数遵从请求参数", head["count"] == 50000, str(head["count"]))
    ok("头部带回使用的视图", head.get("viewsUsed") == ["front", "left", "back"],
       str(head.get("viewsUsed")))

    # 只给正面也必须能跑（退化成单图，模型自己编背面）
    r = client.post("/api/mv/generate",
                    files=[("front", ("front.png", images["front"], "image/png"))],
                    data={"points": "20000"})
    ok("只给正面也能生成（其余视图可选）", r.status_code == 200, str(r.status_code))

    r = client.post("/api/mv/generate",
                    files=[("left", ("left.png", images["left"], "image/png"))],
                    data={"points": "20000"})
    ok("缺正面时返回 4xx 而不是 500", 400 <= r.status_code < 500, str(r.status_code))

except ImportError as exc:
    print(f"  · 跳过 HTTP 测试（缺 {exc.name}）")

print("\n" + "─" * 54)
print(f"  通过 {PASS} 项，失败 {FAIL} 项")
print("─" * 54 + "\n")
sys.exit(1 if FAIL else 0)

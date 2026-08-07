"""
点云工具 · 高精度模式后端

启动：
    server\\start.ps1
或：
    uvicorn app:app --host 127.0.0.1 --port 8000

前端通过 Vite 的 /api 代理访问（见 web/vite.config.js）。
"""

from __future__ import annotations

import os
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from imaging import pack_cloud, pack_result
from moge_runner import DEFAULT_MAX_SIDE, MoGeRunner
from tripo_runner import DEFAULT_POINTS, TripoRunner

app = FastAPI(title="PointCloud Studio · MoGe backend", version="0.1.0")

# 直接开 dev server（不走 Vite 代理）时也能调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = MoGeRunner()
gen_runner = TripoRunner()


def _switch_to(mode: str):
    """
    显存互斥：MoGe ViT-L 和 TripoSR 各要 1.5GB 上下，8GB 的卡还要和桌面应用抢，
    两个一起驻留必 OOM。切模式时把另一个踢出去。
    切回来会重新加载（约 10–20 秒），比推理途中崩掉强。
    """
    other = gen_runner if mode == "depth" else runner
    if other.unload():
        print(f"[显存] 已卸载 {'TripoSR' if mode == 'depth' else 'MoGe'} 腾出空间")


@app.get("/api/health")
def health():
    return runner.info()


@app.post("/api/warmup")
def warmup():
    """提前把模型加载进显存，避免第一张图等太久。"""
    try:
        _switch_to("depth")
        runner.load()
        return runner.info()
    except Exception as exc:
        runner.load_error = str(exc)
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@app.post("/api/infer")
def infer(image: UploadFile = File(...), max_side: str = Form(default="")):
    """
    返回紧凑二进制（几百万浮点走 JSON 又慢又占内存）：

        [0..3]      uint32 LE   头部 JSON 字节长度 L
        [4..4+L)    UTF-8 JSON  { width, height, hasMask, intrinsics, model, device, ms }
        接着        float32[W*H*3]  点图（OpenCV 坐标系，米）
        再接着      uint8[W*H]      有效性 mask

    布局必须与 web/src/core/depthServer.js 的解析逻辑一致。
    """
    try:
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="收到空文件")

        try:
            side = int(max_side) if max_side.strip() else DEFAULT_MAX_SIDE
        except ValueError:
            side = DEFAULT_MAX_SIDE

        _switch_to("depth")
        result = runner.infer(raw, max_side=side)

        body = pack_result(
            result["points"].tobytes(),
            result["mask"].tobytes(),
            {
                "width": result["width"],
                "height": result["height"],
                "hasMask": True,
                "intrinsics": result["intrinsics"],
                "model": result["model"],
                "device": result["device"],
                "ms": result["ms"],
            },
        )
        return Response(content=body, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}") from exc


@app.get("/api/gen/health")
def gen_health():
    return gen_runner.info()


@app.post("/api/gen/warmup")
def gen_warmup():
    """提前把 TripoSR 加载进显存（约 1.4GB 权重，首次冷加载 20s 上下）。"""
    try:
        _switch_to("gen")
        gen_runner.load()
        return gen_runner.info()
    except Exception as exc:
        gen_runner.load_error = str(exc)
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@app.post("/api/generate")
def generate(image: UploadFile = File(...), points: str = Form(default="")):
    """
    生成式 3D：单图 → 完整 360° 点云（背面由模型补全）。

    返回 imaging.pack_cloud 约定的二进制：
        头部 JSON { count, model, device, ms } + float32[count*3] XYZ + uint8[count*3] RGB
    """
    try:
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="收到空文件")

        try:
            n = int(points) if points.strip() else DEFAULT_POINTS
        except ValueError:
            n = DEFAULT_POINTS

        _switch_to("gen")
        result = gen_runner.generate(raw, n_points=n)

        body = pack_cloud(
            result["positions"].tobytes(),
            result["colors"].tobytes(),
            {
                "count": result["count"],
                "model": result["model"],
                "device": result["device"],
                "resolution": result["resolution"],
                "ms": result["ms"],
            },
        )
        return Response(content=body, media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败：{exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
